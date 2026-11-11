"""APL + PRSN FSCIL trainer with real K-shot incremental sessions only."""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import (
    get_base_dataloader,
    get_new_dataloader,
    set_up_datasets,
)
from .apl_prsn_modules import APLPRSNViT
from .apl_prsn_losses import (
    apl_all_min_contrastive_loss,
    apl_part_diversity_loss,
    build_class_prototypes,
    build_leave_one_out_prototypes,
    cosine_prototype_logits,
    prsn_cross_image_semantic_alignment_loss,
)


class ImageLabelDataset(Dataset):
    """Normalize repository datasets to (image, integer label) samples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)):
            raise TypeError("FSCIL datasets must return image/label tuples.")
        if len(sample) == 2:
            image, label = sample
        elif len(sample) == 3:
            image, label, _ = sample
        else:
            raise ValueError("FSCIL datasets must return 2 or 3 fields.")
        return image, int(label)


class ClassBalancedBatchSampler(Sampler):
    """Sample multiple distinct base classes and multiple images per class."""

    def __init__(self, labels, classes_per_batch=16, samples_per_class=4, seed=1):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices = {
            int(class_id): np.flatnonzero(self.labels == class_id).tolist()
            for class_id in np.unique(self.labels)
        }
        if self.classes_per_batch < 2:
            raise ValueError("APL/PRSN training needs at least two classes per batch.")
        if len(self.class_to_indices) < self.classes_per_batch:
            raise ValueError(
                "Base data has {} classes, fewer than classes_per_batch={}.".format(
                    len(self.class_to_indices), self.classes_per_batch
                )
            )
        if self.samples_per_class < 2:
            raise ValueError("PRSN leave-one-out training needs >=2 images per class.")
        for class_id, indices in self.class_to_indices.items():
            if len(indices) < self.samples_per_class:
                raise ValueError(
                    "Base class {} has {} images; {} are required.".format(
                        class_id, len(indices), self.samples_per_class
                    )
                )
        batch_capacity = self.classes_per_batch * self.samples_per_class
        self.iterations = max(1, int(np.ceil(len(self.labels) / batch_capacity)))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.iterations

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        available_classes = np.asarray(sorted(self.class_to_indices))
        for _ in range(self.iterations):
            selected_classes = rng.choice(
                available_classes, self.classes_per_batch, replace=False
            )
            batch_indices = []
            for class_id in selected_classes:
                candidates = self.class_to_indices[int(class_id)]
                batch_indices.extend(
                    rng.choice(
                        candidates, self.samples_per_class, replace=False
                    ).tolist()
                )
            rng.shuffle(batch_indices)
            yield batch_indices


def validate_exact_shot_support(labels, expected_classes, shots, session):
    """Reject missing classes, duplicated support entries, or non-K-shot splits."""
    labels = np.asarray(labels, dtype=np.int64)
    expected_classes = np.asarray(expected_classes, dtype=np.int64)
    if labels.ndim != 1 or expected_classes.ndim != 1:
        raise ValueError("Support labels and expected classes must be one-dimensional.")
    if int(shots) <= 0:
        raise ValueError("The configured K-shot count must be positive.")
    observed = np.unique(labels)
    if not np.array_equal(observed, expected_classes):
        raise RuntimeError(
            "Session {} support classes {} do not match expected {}.".format(
                session, observed.tolist(), expected_classes.tolist()
            )
        )
    counts = np.asarray([(labels == class_id).sum() for class_id in expected_classes])
    if np.any(counts != int(shots)):
        raise RuntimeError(
            "Session {} is not exact {}-shot; per-class counts are {}.".format(
                session,
                int(shots),
                {int(c): int(n) for c, n in zip(expected_classes, counts)},
            )
        )


def validate_seen_class_coverage(labels, seen_classes, session):
    labels = torch.as_tensor(labels).long()
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Evaluation labels must be a non-empty vector.")
    if torch.any(labels < 0) or torch.any(labels >= int(seen_classes)):
        raise RuntimeError(
            "Session {} evaluation labels fall outside [0, {}).".format(
                session, int(seen_classes)
            )
        )
    counts = torch.bincount(labels, minlength=int(seen_classes))
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(
            "Session {} evaluation is missing seen classes {}.".format(
                session, missing
            )
        )


def compute_taskwise_accuracy(labels, predictions, base_class, way, session):
    labels = torch.as_tensor(labels).long()
    predictions = torch.as_tensor(predictions).long().to(labels.device)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("Task-wise labels and predictions must be aligned vectors.")
    task_ranges = [(0, int(base_class))]
    task_ranges.extend(
        (
            int(base_class) + (task_id - 1) * int(way),
            int(base_class) + task_id * int(way),
        )
        for task_id in range(1, int(session) + 1)
    )
    accuracies = []
    for task_id, (start, end) in enumerate(task_ranges):
        mask = (labels >= start) & (labels < end)
        if not mask.any():
            raise RuntimeError("Evaluation set has no samples for task {}.".format(task_id))
        accuracies.append(float((predictions[mask] == labels[mask]).float().mean()))
    return accuracies


class APLPRSNFSCILTrainer:
    """Train APL/PRSN on base classes, then freeze and append real K-shot prototypes."""

    def __init__(self, args):
        args.vit = True
        self.args = set_up_datasets(args)
        self.args.vit = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(getattr(self.args, "seed", 1))
        if self.seed < 0:
            self.seed = 1
        self._set_seed(self.seed)

        self.model = APLPRSNViT(self.args).to(self.device)
        self.session_accuracies = []
        self.sample_accuracies = []
        self.task_accuracies = []
        self.results = []

        output_root = getattr(self.args, "out", None) or "PCAS_APL_PRSN"
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_base = (
            output_root
            if os.path.isabs(output_root)
            else os.path.join(repo_root, "checkpoint", output_root)
        )
        self.save_dir = os.path.join(
            output_base,
            self.args.dataset,
            "APL_PRSN",
            time.strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.base_global_memory_raw = None
        self.base_global_memory_refined = None
        self.base_part_memory = None
        self.base_part_valid = None
        self.all_global_prototypes = None
        self.all_local_prototypes = None
        self.all_local_valid = None

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _make_eval_transform(self):
        if self.args.dataset == "cifar100":
            mean = [0.507, 0.487, 0.441]
            std = [0.267, 0.256, 0.276]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _loader(self, dataset, batch_size, shuffle=False, batch_sampler=None):
        return DataLoader(
            ImageLabelDataset(dataset),
            batch_size=1 if batch_sampler is not None else int(batch_size),
            shuffle=False if batch_sampler is not None else bool(shuffle),
            batch_sampler=batch_sampler,
            num_workers=int(getattr(self.args, "num_workers", 4)),
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(getattr(self.args, "num_workers", 4)) > 0,
        )

    def _audit_support_test_paths(self, support_set, test_set, session):
        if self.args.dataset not in (
            "cub200", "mini_imagenet", "FGVCAircraft", "iNF200"
        ):
            return
        if not hasattr(support_set, "data") or not hasattr(test_set, "data"):
            return
        support_paths = [
            os.path.realpath(os.path.expanduser(str(path)))
            for path in support_set.data
        ]
        test_paths = [
            os.path.realpath(os.path.expanduser(str(path)))
            for path in test_set.data
        ]
        if len(set(support_paths)) != len(support_paths):
            raise RuntimeError("Session {} contains duplicate support paths.".format(session))
        if len(set(test_paths)) != len(test_paths):
            raise RuntimeError("Session {} contains duplicate test paths.".format(session))
        overlap = set(support_paths).intersection(test_paths)
        if overlap:
            raise RuntimeError(
                "Session {} has support/test image leakage ({} shared paths).".format(
                    session, len(overlap)
                )
            )

    @torch.no_grad()
    def _extract_features(self, dataset):
        original_transform = getattr(dataset, "transform", None)
        if original_transform is None:
            raise RuntimeError("The dataset must expose a transform for deterministic extraction.")
        dataset.transform = self._make_eval_transform()
        loader = self._loader(
            dataset,
            batch_size=int(getattr(self.args, "test_batch_size", 128)),
            shuffle=False,
        )
        self.model.eval()
        global_features, part_features, part_valid, labels = [], [], [], []
        try:
            for images, batch_labels in tqdm(loader, desc="Extract frozen ViT features"):
                images = images.to(self.device, non_blocking=True)
                batch_labels = torch.as_tensor(batch_labels).long().to(self.device)
                output = self.model(images)
                global_features.append(output["global_features"])
                part_features.append(output["parts"])
                part_valid.append(output["part_valid"])
                labels.append(batch_labels)
        finally:
            dataset.transform = original_transform

        if not labels:
            raise RuntimeError("Cannot extract features from an empty dataset.")
        return (
            torch.cat(global_features, dim=0),
            torch.cat(part_features, dim=0),
            torch.cat(part_valid, dim=0),
            torch.cat(labels, dim=0),
        )

    def _train_base(self, trainset):
        labels = np.asarray(trainset.targets, dtype=np.int64)
        expected_base = np.arange(int(self.args.base_class), dtype=np.int64)
        if not np.array_equal(np.unique(labels), expected_base):
            raise RuntimeError(
                "Base training labels {} do not match [0, base_class).".format(
                    np.unique(labels).tolist()
                )
            )

        classes_per_batch = min(
            int(getattr(self.args, "apl_classes_per_batch", 16)),
            len(np.unique(labels)),
        )
        sampler = ClassBalancedBatchSampler(
            labels,
            classes_per_batch=classes_per_batch,
            samples_per_class=int(getattr(self.args, "apl_samples_per_class", 4)),
            seed=self.seed,
        )
        loader = self._loader(
            trainset,
            batch_size=0,
            batch_sampler=sampler,
        )

        trainable_parameters = list(self.model.apl.parameters()) + list(
            self.model.prsn.parameters()
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(getattr(self.args, "apl_prsn_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "apl_prsn_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

        global_weight = float(getattr(self.args, "prsn_global_loss_weight", 1.0))
        local_weight = float(getattr(self.args, "prsn_local_loss_weight", 0.1))
        cisa_weight = float(getattr(self.args, "prsn_cisa_weight", 0.1))
        all_min_weight = float(getattr(self.args, "apl_all_min_weight", 0.1))
        diversity_weight = float(getattr(self.args, "apl_diversity_weight", 0.01))
        logit_scale = float(getattr(self.args, "prsn_logit_scale", 16.0))
        all_min_temperature = float(getattr(self.args, "apl_contrastive_temperature", 0.10))

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            self.model.train()
            running = {
                "total": 0.0,
                "global": 0.0,
                "local": 0.0,
                "cisa": 0.0,
                "all_min": 0.0,
                "diversity": 0.0,
                "correct": 0,
                "count": 0,
            }

            for images, batch_labels in tqdm(
                loader, desc="APL-PRSN base epoch {}/{}".format(epoch + 1, epochs)
            ):
                images = images.to(self.device, non_blocking=True)
                batch_labels = torch.as_tensor(batch_labels).long().to(self.device)
                output = self.model(images)
                global_features = output["global_features"]
                parts = output["parts"]
                part_valid = output["part_valid"]

                cache = build_class_prototypes(
                    global_features, parts, part_valid, batch_labels
                )
                leave_one_out = build_leave_one_out_prototypes(
                    global_features, parts, part_valid, batch_labels, cache
                )
                class_count = cache["classes"].numel()
                class_indices = torch.arange(class_count, device=self.device)

                refined_global = self.model.prsn.global_refiner.refine_base(
                    cache["global"]
                )
                loo_global = self.model.prsn.global_refiner.refine_queries(
                    leave_one_out["global"],
                    cache["global"],
                    exclude_indices=leave_one_out["inverse"],
                )

                calibrated_local, calibrated_valid, _, _ = (
                    self.model.prsn.pfrsm.calibrate(
                        cache["global"],
                        cache["parts"],
                        cache["valid"],
                        cache["global"],
                        cache["parts"],
                        cache["valid"],
                        exclude_indices=class_indices,
                    )
                )
                loo_local, loo_local_valid, _, _ = self.model.prsn.pfrsm.calibrate(
                    leave_one_out["global"],
                    leave_one_out["parts"],
                    leave_one_out["valid"],
                    cache["global"],
                    cache["parts"],
                    cache["valid"],
                    exclude_indices=leave_one_out["inverse"],
                )

                global_logits = cosine_prototype_logits(
                    global_features, refined_global, logit_scale
                )
                loo_global_logits = (
                    F.normalize(global_features, dim=-1) * loo_global
                ).sum(dim=-1, keepdim=True) * logit_scale
                global_logits = global_logits.scatter(
                    1, leave_one_out["inverse"].unsqueeze(1), loo_global_logits
                )

                local_logits = self.model.prsn.pfrsm.local_logits(
                    parts,
                    part_valid,
                    calibrated_local,
                    calibrated_valid,
                    logit_scale=logit_scale,
                )
                loo_local_logits = self.model.prsn.pfrsm.pairwise_local_logits(
                    parts,
                    part_valid,
                    loo_local,
                    loo_local_valid,
                    logit_scale=logit_scale,
                ).unsqueeze(1)
                local_logits = local_logits.scatter(
                    1, leave_one_out["inverse"].unsqueeze(1), loo_local_logits
                )

                loss_global = F.cross_entropy(
                    global_logits, leave_one_out["inverse"]
                )
                loss_local = F.cross_entropy(
                    local_logits, leave_one_out["inverse"]
                )
                loss_cisa = prsn_cross_image_semantic_alignment_loss(
                    global_logits, local_logits, leave_one_out["inverse"]
                )
                loss_all_min = apl_all_min_contrastive_loss(
                    parts,
                    part_valid,
                    batch_labels,
                    temperature=all_min_temperature,
                )
                loss_diversity = apl_part_diversity_loss(parts, part_valid)

                total_loss = (
                    global_weight * loss_global
                    + local_weight * loss_local
                    + cisa_weight * loss_cisa
                    + all_min_weight * loss_all_min
                    + diversity_weight * loss_diversity
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        "Non-finite APL-PRSN loss at epoch {}.".format(epoch + 1)
                    )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                for name, parameter in list(self.model.apl.named_parameters()) + [
                    ("prsn." + name, parameter)
                    for name, parameter in self.model.prsn.named_parameters()
                ]:
                    if parameter.grad is not None and not torch.isfinite(
                        parameter.grad
                    ).all():
                        raise FloatingPointError(
                            "Non-finite trainable gradient in {}.".format(name)
                        )
                optimizer.step()

                running["total"] += float(total_loss.detach())
                running["global"] += float(loss_global.detach())
                running["local"] += float(loss_local.detach())
                running["cisa"] += float(loss_cisa.detach())
                running["all_min"] += float(loss_all_min.detach())
                running["diversity"] += float(loss_diversity.detach())
                running["correct"] += int(
                    (0.5 * (global_logits + local_logits)).argmax(dim=1)
                    .eq(leave_one_out["inverse"]).sum().item()
                )
                running["count"] += batch_labels.numel()

            scheduler.step()
            iterations = max(1, len(loader))
            record = {
                "epoch": epoch + 1,
                "total_loss": running["total"] / iterations,
                "global_loss": running["global"] / iterations,
                "local_loss": running["local"] / iterations,
                "cisa_loss": running["cisa"] / iterations,
                "apl_all_min_loss": running["all_min"] / iterations,
                "apl_diversity_loss": running["diversity"] / iterations,
                "batch_joint_accuracy": running["correct"] / max(1, running["count"]),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            self.results.append(record)
            print("[APL-PRSN base]", record)

        self.model.freeze_for_incremental()
        self._cache_base_memory(trainset)
        torch.save(
            {
                "model": self.model.state_dict(),
                "base_global_memory_raw": self.base_global_memory_raw.cpu(),
                "base_global_memory_refined": self.base_global_memory_refined.cpu(),
                "base_part_memory": self.base_part_memory.cpu(),
                "base_part_valid": self.base_part_valid.cpu(),
                "base_classes": int(self.args.base_class),
                "method": "APL-PRSN-FSCIL",
                "backbone": "frozen ImageNet-21K ViT-B/16",
                "part_prior": "final-block CLS-to-Patch attention from the same ViT",
            },
            os.path.join(self.save_dir, "apl_prsn_base.pth"),
        )

    def _cache_base_memory(self, trainset):
        global_features, parts, part_valid, labels = self._extract_features(trainset)
        cache = build_class_prototypes(global_features, parts, part_valid, labels)
        expected = torch.arange(int(self.args.base_class), device=self.device)
        if not torch.equal(cache["classes"], expected):
            raise RuntimeError("Base memory classes are not contiguous from zero.")

        # Store immutable base visual memory. Novel classes never become
        # retrieval sources, preventing recursive calibration drift.
        self.base_global_memory_raw = cache["global"].detach()
        self.base_global_memory_refined = (
            self.model.prsn.global_refiner.refine_base(cache["global"]).detach()
        )
        self.base_part_memory = cache["parts"].detach()
        self.base_part_valid = cache["valid"].detach()
        self.all_global_prototypes = self.base_global_memory_refined.clone()
        self.all_local_prototypes = self.base_part_memory.clone()
        self.all_local_valid = self.base_part_valid.clone()

    @torch.no_grad()
    def _evaluate(self, testset, seen_classes, session):
        global_features, parts, part_valid, labels = self._extract_features(testset)
        validate_seen_class_coverage(labels, seen_classes, session)
        if self.all_global_prototypes.shape[0] != int(seen_classes):
            raise RuntimeError("Global prototype count does not match seen classes.")
        if self.all_local_prototypes.shape[0] != int(seen_classes):
            raise RuntimeError("Local prototype count does not match seen classes.")

        logit_scale = float(getattr(self.args, "prsn_logit_scale", 16.0))
        global_logits = cosine_prototype_logits(
            global_features, self.all_global_prototypes, logit_scale
        )
        local_logits = self.model.prsn.pfrsm.local_logits(
            parts,
            part_valid,
            self.all_local_prototypes,
            self.all_local_valid,
            logit_scale=logit_scale,
        )
        global_weight = float(getattr(self.args, "joint_global_weight", 0.5))
        local_weight = float(getattr(self.args, "joint_local_weight", 0.5))
        weight_sum = global_weight + local_weight
        if global_weight < 0 or local_weight < 0 or weight_sum <= 0:
            raise ValueError("Joint global/local classification weights are invalid.")
        logits = (
            global_weight * global_logits + local_weight * local_logits
        ) / weight_sum

        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == labels).float().mean().item())
        class_accuracy = float(torch.stack([
            (predictions[labels == class_id] == class_id).float().mean()
            for class_id in range(int(seen_classes))
        ]).mean().item())

        base_mask = labels < int(self.args.base_class)
        novel_mask = labels >= int(self.args.base_class)
        base_accuracy = (
            float((predictions[base_mask] == labels[base_mask]).float().mean().item())
            if base_mask.any() else float("nan")
        )
        novel_accuracy = (
            float((predictions[novel_mask] == labels[novel_mask]).float().mean().item())
            if novel_mask.any() else float("nan")
        )
        hmean = (
            2.0 * base_accuracy * novel_accuracy / (base_accuracy + novel_accuracy)
            if base_mask.any() and novel_mask.any() and base_accuracy + novel_accuracy > 0
            else float("nan")
        )
        novel_to_base_error = (
            float((predictions[novel_mask] < int(self.args.base_class)).float().mean().item())
            if novel_mask.any() else float("nan")
        )
        return {
            "session": int(session),
            "accuracy": accuracy,
            "class_average_accuracy": class_accuracy,
            "base_accuracy": base_accuracy,
            "novel_accuracy": novel_accuracy,
            "base_novel_hmean": float(hmean),
            "novel_to_base_error_rate": novel_to_base_error,
            "task_wise_accuracy": compute_taskwise_accuracy(
                labels, predictions, self.args.base_class, self.args.way, session
            ),
        }

    @torch.no_grad()
    def _incremental_session(self, support_set, testset, session):
        expected = np.arange(
            int(self.args.base_class) + (session - 1) * int(self.args.way),
            int(self.args.base_class) + session * int(self.args.way),
            dtype=np.int64,
        )
        support_labels = np.asarray(support_set.targets, dtype=np.int64)
        validate_exact_shot_support(
            support_labels, expected, int(self.args.shot), session
        )
        self._audit_support_test_paths(support_set, testset, session)

        # Only the labeled K-shot support set creates new class prototypes.
        # Test images and test labels are not used in retrieval or calibration.
        global_features, parts, part_valid, labels = self._extract_features(support_set)
        support_cache = build_class_prototypes(
            global_features, parts, part_valid, labels
        )
        if not np.array_equal(
            support_cache["classes"].cpu().numpy(), expected
        ):
            raise RuntimeError("Support prototype order does not match session classes.")

        new_global = self.model.prsn.global_refiner.refine_queries(
            support_cache["global"], self.base_global_memory_raw
        )
        new_local, new_local_valid, selected_base, base_weights = (
            self.model.prsn.pfrsm.calibrate(
                support_cache["global"],
                support_cache["parts"],
                support_cache["valid"],
                self.base_global_memory_raw,
                self.base_part_memory,
                self.base_part_valid,
            )
        )

        previous_count = self.all_global_prototypes.shape[0]
        if previous_count != int(expected[0]):
            raise RuntimeError(
                "Cannot append session {} prototypes after {} classes; expected {}.".format(
                    session, previous_count, int(expected[0])
                )
            )
        # Append only. Existing base and novel prototypes are never rewritten.
        self.all_global_prototypes = torch.cat(
            [self.all_global_prototypes, new_global.detach()], dim=0
        )
        self.all_local_prototypes = torch.cat(
            [self.all_local_prototypes, new_local.detach()], dim=0
        )
        self.all_local_valid = torch.cat(
            [self.all_local_valid, new_local_valid.detach()], dim=0
        )

        metrics = self._evaluate(
            testset,
            int(self.args.base_class) + session * int(self.args.way),
            session,
        )
        metrics["pfrsm_selected_base_indices"] = selected_base.cpu().tolist()
        metrics["pfrsm_selected_base_weights"] = base_weights.cpu().tolist()
        self.results.append(metrics)
        self.session_accuracies.append(metrics["class_average_accuracy"])
        self.sample_accuracies.append(metrics["accuracy"])
        self.task_accuracies.append(metrics["task_wise_accuracy"])
        print("[APL-PRSN incremental]", metrics)

    def train(self):
        start_time = time.time()
        trainset, _, testloader = get_base_dataloader(self.args)
        self._train_base(trainset)

        base_metrics = self._evaluate(testloader.dataset, int(self.args.base_class), 0)
        self.results.append(base_metrics)
        self.session_accuracies.append(base_metrics["class_average_accuracy"])
        self.sample_accuracies.append(base_metrics["accuracy"])
        self.task_accuracies.append(base_metrics["task_wise_accuracy"])
        print("[APL-PRSN base evaluation]", base_metrics)

        for session in range(1, int(self.args.sessions)):
            support_set, _, session_testloader = get_new_dataloader(
                self.args, session
            )
            self._incremental_session(
                support_set, session_testloader.dataset, session
            )

        summary = {
            "method": "APL-PRSN-FSCIL",
            "AA": float(np.mean(self.session_accuracies)),
            "PD": float(self.session_accuracies[0] - self.session_accuracies[-1]),
            "session_class_average_accuracies": self.session_accuracies,
            "session_sample_accuracies": self.sample_accuracies,
            "task_wise_accuracies": self.task_accuracies,
            "final_session_class_accuracy": float(self.session_accuracies[-1]),
            "final_session_sample_accuracy": float(self.sample_accuracies[-1]),
        }
        self.results.append(summary)
        print("[APL-PRSN summary]", summary)

        torch.save(
            {
                "model": self.model.state_dict(),
                "base_global_memory_raw": self.base_global_memory_raw.cpu(),
                "base_global_memory_refined": self.base_global_memory_refined.cpu(),
                "base_part_memory": self.base_part_memory.cpu(),
                "base_part_valid": self.base_part_valid.cpu(),
                "all_global_prototypes": self.all_global_prototypes.cpu(),
                "all_local_prototypes": self.all_local_prototypes.cpu(),
                "all_local_valid": self.all_local_valid.cpu(),
                "summary": summary,
                "results": self.results,
            },
            os.path.join(self.save_dir, "apl_prsn_final.pth"),
        )
        with open(os.path.join(self.save_dir, "results.txt"), "w") as handle:
            for record in self.results:
                handle.write(repr(record) + "\n")
        print(
            "APL-PRSN FSCIL completed in {:.2f} minutes.".format(
                (time.time() - start_time) / 60.0
            )
        )
