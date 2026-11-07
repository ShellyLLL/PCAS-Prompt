"""End-to-end PCAS-Prompt FSCIL training and evaluation."""
import os
import time
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import set_up_datasets, get_base_dataloader, get_new_dataloader
from .pcas_modules import PCASViT
from .pcas_losses import (
    build_prototype_cache,
    leave_one_out_prototype_loss,
    sample_affinity_consistency,
    prototype_affinity_consistency,
    affinity_drift,
    teen_calibrate,
)


class ClassBalancedBatchSampler(Sampler):
    """Yield batches containing fixed samples per class, without replacement."""

    def __init__(self, labels, classes_per_batch=16, samples_per_class=4, seed=1):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices = {
            int(c): np.flatnonzero(self.labels == c).tolist()
            for c in np.unique(self.labels)
        }
        if len(self.class_to_indices) < self.classes_per_batch:
            raise ValueError(
                f"Need at least {self.classes_per_batch} base classes per batch; "
                f"dataset has {len(self.class_to_indices)}."
            )
        for cls, indices in self.class_to_indices.items():
            if len(indices) < self.samples_per_class:
                raise ValueError(
                    f"Class {cls} has {len(indices)} samples; "
                    f"need {self.samples_per_class} for LOO prototype loss."
                )
        self.iterations = max(
            1,
            int(np.ceil(len(self.labels) /
                        (self.classes_per_batch * self.samples_per_class)))
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.iterations

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        classes = np.asarray(sorted(self.class_to_indices))
        for _ in range(self.iterations):
            chosen = rng.choice(classes, self.classes_per_batch, replace=False)
            batch = []
            for cls in chosen:
                indices = self.class_to_indices[int(cls)]
                batch.extend(rng.choice(
                    indices, self.samples_per_class, replace=False
                ).tolist())
            rng.shuffle(batch)
            yield batch


class IndexedDataset(torch.utils.data.Dataset):
    """Attach the local dataset position to a (image, label) sample.

    The local position is deliberately used instead of a dataset-provided
    third field: PCAS prototype caches are built in this dataset's iteration
    order, so leave-one-out indices must address that exact order.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)):
            raise TypeError(
                "PCAS datasets must return (image, label) or "
                "(image, label, metadata); got {!r}.".format(type(sample))
            )
        if len(sample) == 2:
            image, label = sample
        elif len(sample) == 3:
            image, label, _ = sample
        else:
            raise ValueError(
                "PCAS datasets must return 2 or 3 fields; got {}."
                .format(len(sample))
            )
        return image, int(label), int(index)


def compute_taskwise_accuracy(labels, predictions, base_class, way, session):
    """Return per-task accuracies after the given FSCIL session.

    Task 0 is the base class block [0, base_class). Each incremental task
    contains one contiguous block of 'way' global class IDs. Both labels
    and predictions are evaluated in the same global label space.
    """
    labels = torch.as_tensor(labels).long()
    predictions = torch.as_tensor(predictions).long().to(labels.device)
    if labels.ndim != 1 or predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("labels and predictions must be aligned 1-D tensors.")
    if labels.numel() == 0:
        raise ValueError("Cannot compute task-wise accuracy on an empty test set.")

    base_class = int(base_class)
    way = int(way)
    session = int(session)
    if base_class <= 0 or way <= 0 or session < 0:
        raise ValueError("base_class and way must be positive; session cannot be negative.")

    seen_classes = base_class + session * way
    if torch.any(labels < 0) or torch.any(labels >= seen_classes):
        raise ValueError(
            "Test labels fall outside the classes expected at session {}."
            .format(session)
        )

    task_accuracies = []
    task_ranges = [(0, base_class)]
    task_ranges.extend(
        (base_class + (task_id - 1) * way,
         base_class + task_id * way)
        for task_id in range(1, session + 1)
    )
    for task_id, (start_class, end_class) in enumerate(task_ranges):
        mask = (labels >= start_class) & (labels < end_class)
        if not torch.any(mask):
            raise ValueError(
                "No test samples found for task {} (classes {}..{})."
                .format(task_id, start_class, end_class - 1)
            )
        task_accuracy = (predictions[mask] == labels[mask]).float().mean()
        task_accuracies.append(float(task_accuracy.item()))
    return task_accuracies


def validate_exact_shot_support(labels, expected_classes, shots, session):
    """Require exactly K unique support examples per expected class."""
    labels = np.asarray(labels, dtype=np.int64)
    expected_classes = np.asarray(expected_classes, dtype=np.int64)
    if labels.ndim != 1 or expected_classes.ndim != 1:
        raise ValueError("Support labels and expected classes must be 1-D.")
    if int(shots) <= 0:
        raise ValueError("The configured shot count must be positive.")
    observed_classes = np.unique(labels)
    if not np.array_equal(observed_classes, expected_classes):
        raise RuntimeError(
            "Session {} support classes are {}; expected {}."
            .format(session, observed_classes.tolist(), expected_classes.tolist())
        )
    counts = np.asarray([(labels == cls).sum() for cls in expected_classes])
    if np.any(counts != int(shots)):
        details = {
            int(cls): int(count)
            for cls, count in zip(expected_classes, counts)
        }
        raise RuntimeError(
            "Session {} is not a {}-shot support split; per-class counts: {}."
            .format(session, int(shots), details)
        )
    return {
        int(cls): int(count)
        for cls, count in zip(expected_classes, counts)
    }


def validate_seen_class_coverage(labels, seen_classes, session):
    """Require every seen global class to occur in the evaluation set."""
    labels = torch.as_tensor(labels).long()
    seen_classes = int(seen_classes)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Evaluation labels must be a non-empty 1-D tensor.")
    if torch.any(labels < 0) or torch.any(labels >= seen_classes):
        raise RuntimeError(
            "Session {} test labels are outside [0, {})."
            .format(session, seen_classes)
        )
    counts = torch.bincount(labels, minlength=seen_classes)
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(
            "Session {} evaluation set is missing seen classes: {}."
            .format(session, missing)
        )
    return counts


class PCASFSCILTrainer:
    def __init__(self, args):
        self.args = set_up_datasets(args)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = int(getattr(args, "seed", 1))
        if self.seed < 0:
            self.seed = 1
        self._set_seed(self.seed)
        self.model = PCASViT(self.args).to(self.device)

        self.output_root = getattr(args, "out", None) or "PCAS"
        self.save_dir = os.path.join(
            "checkpoint", self.output_root, self.args.dataset,
            time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.base_prototypes = None
        self.base_teacher_prototypes = None
        self.base_accuracy = None
        # Macro class accuracy is retained for AA/PD. These histories add the
        # standard FSCIL cumulative and task-wise session accuracy report.
        self.session_accuracies = []
        self.base_session_accuracies = []
        self.final_accuracies = []
        self.task_wise_accuracies = []
        self.results = []

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _loader(self, dataset, batch_size, shuffle=False, sampler=None):
        return DataLoader(
            dataset,
            batch_size=batch_size if sampler is None else 1,
            shuffle=shuffle if sampler is None else False,
            batch_sampler=sampler,
            num_workers=int(getattr(self.args, "num_workers", 4)),
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(getattr(self.args, "num_workers", 4)) > 0,
        )

    def _make_eval_transform(self, dataset):
        if self.args.dataset == "cifar100":
            mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def _extract_features(self, dataset, teacher=False, batch_size=None):
        original_transform = dataset.transform
        was_training = self.model.training
        dataset.transform = self._make_eval_transform(dataset)
        # All current dataset implementations return (image, label), while
        # feature extraction and the PCAS cache contract also need the stable
        # local sample position. Wrap here instead of changing each dataset's
        # public __getitem__ API.
        indexed_dataset = IndexedDataset(dataset)
        loader = self._loader(
            indexed_dataset,
            batch_size or int(getattr(self.args, "test_batch_size", 128)),
            shuffle=False,
        )
        features, labels, indices = [], [], []
        self.model.eval()
        try:
            for images, target, index in tqdm(loader, desc="Extract features"):
                images = images.to(self.device, non_blocking=True)
                output = (
                    self.model.encode_teacher(images)
                    if teacher else self.model.encode_student(images)
                )
                features.append(output.cpu())
                labels.append(torch.as_tensor(target).long().cpu())
                indices.append(torch.as_tensor(index).long().cpu())
        finally:
            dataset.transform = original_transform
            # Feature extraction is a temporary eval-only operation. Restore
            # the caller's mode so prompt dropout remains active during base
            # optimization and frozen modules stay in eval mode incrementally.
            self.model.train(was_training)
        return (
            torch.cat(features).to(self.device),
            torch.cat(labels).to(self.device),
            torch.cat(indices).to(self.device),
        )

    @torch.no_grad()
    def _cache_prototypes(self, dataset, num_classes, teacher=False):
        features, labels, _ = self._extract_features(dataset, teacher=teacher)
        return build_prototype_cache(features, labels, num_classes)

    def _base_optimizer(self):
        prompt_params = [
            p for p in self.model.pcas.parameters() if p.requires_grad
        ]
        return torch.optim.AdamW(
            [
                {"params": prompt_params,
                 "lr": float(getattr(self.args, "pcas_lr", 1e-3))},
                {"params": self.model.fc.parameters(),
                 "lr": float(getattr(self.args, "pcas_head_lr", 1e-3))},
            ],
            weight_decay=float(getattr(self.args, "pcas_weight_decay", 1e-4)),
        )

    def _train_base(self, trainset, trainloader):
        observed_base_classes = np.unique(np.asarray(trainset.targets)).astype(int)
        expected_base_classes = np.arange(int(self.args.base_class), dtype=int)
        if not np.array_equal(observed_base_classes, expected_base_classes):
            raise RuntimeError(
                "Base training split contains class IDs {} but expected {}. "
                "Check dataset class ordering and base split construction."
                .format(observed_base_classes.tolist(), expected_base_classes.tolist())
            )

        classes_per_batch = int(getattr(self.args, "classes_per_batch", 16))
        samples_per_class = int(getattr(self.args, "samples_per_class", 4))
        sampler = ClassBalancedBatchSampler(
            trainset.targets, classes_per_batch, samples_per_class, self.seed
        )
        indexed_trainset = IndexedDataset(trainset)
        trainloader = self._loader(indexed_trainset, 1, sampler=sampler)

        # Teacher cache is deterministic and uses the original frozen ViT.
        teacher_cache = self._cache_prototypes(
            trainset, self.args.base_class, teacher=True
        )
        optimizer = self._base_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.args.epochs_base)
        )

        for epoch in range(int(self.args.epochs_base)):
            sampler.set_epoch(epoch)
            self.model.train()
            totals = {"total": 0.0, "ce": 0.0, "proto": 0.0,
                      "sample_aff": 0.0, "proto_aff": 0.0,
                      "lambda_affinity": 0.0}
            num_batches = 0
            routing_usage = torch.zeros(
                self.model.pcas.num_components, device=self.device
            )
            routing_entropy_total = 0.0
            routing_batches = 0

            # Refresh deterministic student prototype cache once per epoch.
            student_cache = self._cache_prototypes(
                trainset, self.args.base_class, teacher=False
            )
            teacher_prototypes = teacher_cache["prototypes"].detach()
            student_prototypes = student_cache["prototypes"].detach()

            for images, labels, indices in tqdm(
                trainloader, desc=f"PCAS base epoch {epoch + 1}"
            ):
                # batch_sampler emits a list of indices, default collate creates
                # [1,B,...]; flatten only the sampler's singleton outer batch.
                images = images.to(self.device, non_blocking=True)
                labels = labels.long().to(self.device)
                indices = indices.long().to(self.device)

                with torch.no_grad():
                    teacher_features = self.model.encode_teacher(images)

                student_features, aux = self.model.encode_student(
                    images, return_aux=True
                )
                logits = self.model.classify_base(student_features)
                loss_ce = F.cross_entropy(logits, labels)
                loss_proto = leave_one_out_prototype_loss(
                    student_features, labels, indices, student_cache,
                    logit_scale=float(getattr(self.args, "pcas_logit_scale", 16.0))
                )
                loss_sample = sample_affinity_consistency(
                    teacher_features, student_features, labels,
                    teacher_prototypes, student_prototypes,
                    affinity_scale=float(getattr(self.args, "affinity_scale", 16.0))
                )
                loss_relation = prototype_affinity_consistency(
                    teacher_features, student_features, labels,
                    affinity_scale=float(getattr(self.args, "affinity_scale", 16.0))
                )

                lambda_proto = float(getattr(self.args, "lambda_proto", 0.5))
                lambda_aff_max = float(getattr(self.args, "lambda_affinity", 0.1))
                warmup_epochs = int(getattr(self.args, "affinity_warmup_epochs", 10))
                if warmup_epochs < 0:
                    raise ValueError("affinity_warmup_epochs cannot be negative.")
                # Linear warm-up is defined in epoch space and reaches the
                # configured coefficient exactly at the end of warm-up.
                warmup_factor = (
                    1.0 if warmup_epochs == 0
                    else min(1.0, float(epoch + 1) / warmup_epochs)
                )
                lambda_aff = lambda_aff_max * warmup_factor
                beta = float(getattr(self.args, "beta_affinity", 0.5))

                # Explicit loss switches support faithful component ablations
                # without changing data, optimizer, or incremental protocol.
                use_proto_loss = not bool(getattr(self.args, "disable_proto_loss", False))
                use_sample_affinity = not bool(
                    getattr(self.args, "disable_sample_affinity", False)
                )
                use_proto_affinity = not bool(
                    getattr(self.args, "disable_proto_affinity", False)
                )
                loss = loss_ce
                if use_proto_loss:
                    loss = loss + lambda_proto * loss_proto
                if use_sample_affinity:
                    loss = loss + lambda_aff * loss_sample
                if use_proto_affinity:
                    loss = loss + lambda_aff * beta * loss_relation

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                totals["total"] += float(loss.detach())
                totals["ce"] += float(loss_ce.detach())
                totals["proto"] += float(loss_proto.detach())
                totals["sample_aff"] += float(loss_sample.detach())
                totals["proto_aff"] += float(loss_relation.detach())
                totals["lambda_affinity"] += lambda_aff
                num_batches += 1

                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PCAS training loss.")

                if aux["routing_weights"] is not None:
                    batch_routing = aux["routing_weights"].detach()
                    if not torch.allclose(
                        batch_routing.sum(-1),
                        torch.ones(images.shape[0], device=self.device),
                        atol=1e-5,
                    ):
                        raise RuntimeError("PCAS routing weights do not sum to one.")
                    routing_usage += batch_routing.mean(0)
                    batch_entropy = -(
                        batch_routing.clamp_min(1e-12)
                        * batch_routing.clamp_min(1e-12).log()
                    ).sum(-1).mean()
                    routing_entropy_total += float(batch_entropy)
                    routing_batches += 1

            scheduler.step()
            summary = {
                key: value / max(1, num_batches)
                for key, value in totals.items()
            }
            summary["epoch"] = epoch + 1
            summary["lr"] = scheduler.get_last_lr()[0]
            summary["lambda_affinity_effective"] = (
                summary.pop("lambda_affinity")
            )
            if routing_batches:
                mean_usage = routing_usage / routing_batches
                usage_distribution = mean_usage / mean_usage.sum().clamp_min(1e-12)
                summary["routing_entropy"] = routing_entropy_total / routing_batches
                summary["routing_effective_components"] = float(
                    torch.exp(-(usage_distribution * usage_distribution.clamp_min(1e-12).log()).sum())
                )
                summary["routing_component_usage"] = usage_distribution.cpu().tolist()
            self.results.append(summary)
            print("[PCAS]", summary)

        # Freeze all learned PCAS parameters and compute final student base
        # prototypes from the complete base training split.
        self.model.freeze_for_incremental()
        final_student = self._cache_prototypes(
            trainset, self.args.base_class, teacher=False
        )
        self.base_prototypes = final_student["prototypes"].detach()
        self.base_teacher_prototypes = teacher_cache["prototypes"].detach()

        # The base prototypes are the ONLY base references used by TEEN.
        torch.save(
            {
                "model": self.model.state_dict(),
                "base_student_prototypes": self.base_prototypes.cpu(),
                "base_teacher_prototypes_for_audit": self.base_teacher_prototypes.cpu(),
                "base_classes": int(self.args.base_class),
                "method": "PCAS-Prompt",
                "feature_space": "frozen_vit_b16_with_final_pcas_prompt",
            },
            os.path.join(self.save_dir, "pcas_base.pth"),
        )

    @torch.no_grad()
    def _evaluate(self, testset, seen_classes, all_prototypes, session):
        features, labels, _ = self._extract_features(testset, teacher=False)
        validate_seen_class_coverage(labels, seen_classes, session)
        affinity_drift_value = float("nan")
        if self.base_teacher_prototypes is not None and self.base_prototypes is not None:
            teacher_features, teacher_labels, _ = self._extract_features(
                testset, teacher=True
            )
            if not torch.equal(labels, teacher_labels):
                raise RuntimeError("Teacher/student evaluation labels are misaligned.")
            affinity_drift_value = float(affinity_drift(
                teacher_features,
                features,
                labels,
                self.base_teacher_prototypes,
                self.base_prototypes,
                affinity_scale=float(getattr(self.args, "affinity_scale", 16.0)),
                base_class_count=int(self.args.base_class),
            ).item())
        logits = float(getattr(self.args, "pcas_logit_scale", 16.0)) * (
            features @ F.normalize(all_prototypes, dim=-1).t()
        )
        logits = logits[:, :seen_classes]
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        task_wise_accuracy = compute_taskwise_accuracy(
            labels, predictions,
            base_class=self.args.base_class,
            way=self.args.way,
            session=session,
        )

        per_class_accuracy = []
        for class_id in range(seen_classes):
            class_mask = labels == class_id
            if class_mask.any():
                per_class_accuracy.append(
                    (predictions[class_mask] == labels[class_mask]).float().mean()
                )
        class_average_accuracy = (
            torch.stack(per_class_accuracy).mean().item()
            if per_class_accuracy else float("nan")
        )

        base_mask = labels < self.args.base_class
        novel_mask = ~base_mask
        base_acc = (
            (predictions[base_mask] == labels[base_mask]).float().mean().item()
            if base_mask.any() else float("nan")
        )
        novel_acc = (
            (predictions[novel_mask] == labels[novel_mask]).float().mean().item()
            if novel_mask.any() else float("nan")
        )
        # TBR measures the fraction of novel examples absorbed by base classes.
        tbr = (
            ((predictions[novel_mask] < self.args.base_class).float().mean().item())
            if novel_mask.any() else float("nan")
        )
        hmean = (
            2.0 * base_acc * novel_acc / (base_acc + novel_acc + 1e-12)
            if base_mask.any() and novel_mask.any() else float("nan")
        )
        # Mean within-class cosine distance on the evaluation split. This is a
        # diagnostic only; it does not alter prototypes or classifier logits.
        class_distances = []
        for class_id in range(seen_classes):
            mask = labels == class_id
            class_center = F.normalize(features[mask].mean(0), dim=-1)
            class_distances.append(
                (1.0 - (features[mask] @ class_center)).mean()
            )
        prototype_distance = torch.stack(class_distances).mean().item()
        return {
            "accuracy": accuracy,
            "class_average_accuracy": class_average_accuracy,
            "base_accuracy": base_acc,
            "novel_accuracy": novel_acc,
            "HMean": hmean,
            "TBR": tbr,
            "prototype_distance": prototype_distance,
            "affinity_drift": affinity_drift_value,
            "task_wise_accuracy": task_wise_accuracy,
        }

    @torch.no_grad()
    def _incremental_session(self, trainset, testset, session):
        novel_classes = np.unique(np.asarray(trainset.targets)).astype(int)
        expected_classes = np.arange(
            self.args.base_class + (session - 1) * self.args.way,
            self.args.base_class + session * self.args.way,
            dtype=int,
        )
        if not np.array_equal(novel_classes, expected_classes):
            raise RuntimeError(
                "Session {} support split has class IDs {} but expected {}. "
                "Check this dataset's session_N.txt format and class ordering."
                .format(session, novel_classes.tolist(), expected_classes.tolist())
            )

        if self.args.dataset == "cub200":
            support_counts = validate_exact_shot_support(
                trainset.targets, expected_classes, self.args.shot, session
            )
            support_paths = [
                os.path.realpath(os.path.expanduser(str(path)))
                for path in trainset.data
            ]
            test_paths = [
                os.path.realpath(os.path.expanduser(str(path)))
                for path in testset.data
            ]
            if len(set(support_paths)) != len(support_paths):
                raise RuntimeError(
                    "CUB session {} support split contains duplicate image paths."
                    .format(session)
                )
            if len(set(test_paths)) != len(test_paths):
                raise RuntimeError(
                    "CUB session {} test split contains duplicate image paths."
                    .format(session)
                )
            overlap = set(support_paths).intersection(test_paths)
            if overlap:
                raise RuntimeError(
                    "CUB session {} has {} support/test image path overlaps."
                    .format(session, len(overlap))
                )
            print("[PCAS CUB support audit]", {
                "session": session,
                "shots_per_class": support_counts,
                "support_images": len(support_paths),
                "support_test_overlap": len(overlap),
            })

        support_features, support_labels, _ = self._extract_features(
            trainset, teacher=False
        )
        novel_prototypes = []
        for class_id in novel_classes:
            class_features = support_features[support_labels == int(class_id)]
            if class_features.numel() == 0:
                raise RuntimeError(f"No support features for novel class {class_id}.")
            novel_prototypes.append(F.normalize(class_features.mean(0), dim=-1))
        novel_prototypes = torch.stack(novel_prototypes, dim=0)

        alpha = float(getattr(self.args, "teen_alpha", 0.5))
        tau = float(getattr(self.args, "teen_tau", 16.0))
        calibrated, _ = teen_calibrate(
            novel_prototypes, self.base_prototypes, alpha=alpha, tau=tau
        )

        # Append new prototypes by their global class ids. Old prototypes never
        # change; PCAS and ViT are frozen for every incremental session.
        if self.all_prototypes is None:
            self.all_prototypes = torch.zeros(
                int(self.args.num_classes),
                self.base_prototypes.shape[1],
                device=self.device,
                dtype=self.base_prototypes.dtype,
            )
            self.all_prototypes[:self.args.base_class] = self.base_prototypes
        for offset, class_id in enumerate(novel_classes):
            self.all_prototypes[int(class_id)] = calibrated[offset]

        seen_classes = self.args.base_class + session * self.args.way
        metrics = self._evaluate(
            testset, seen_classes, self.all_prototypes, session=session
        )
        metrics.update({"session": session, "novel_classes": novel_classes.tolist()})
        self.results.append(metrics)
        self.session_accuracies.append(metrics["class_average_accuracy"])
        self.base_session_accuracies.append(metrics["base_accuracy"])
        self.final_accuracies.append(metrics["accuracy"])
        self.task_wise_accuracies.append(metrics["task_wise_accuracy"])
        print("[PCAS incremental]", metrics)
        print(
            "[PCAS session {}] Final acc: {:.2f}; Task-wise acc: {}".format(
                session,
                metrics["accuracy"] * 100.0,
                [round(value * 100.0, 2) for value in metrics["task_wise_accuracy"]],
            )
        )
        return metrics

    def train(self):
        start = time.time()
        trainset, trainloader, testloader = get_base_dataloader(self.args)
        if self.args.dataset == "cub200":
            train_paths = [
                os.path.realpath(os.path.expanduser(str(path)))
                for path in trainset.data
            ]
            test_paths = [
                os.path.realpath(os.path.expanduser(str(path)))
                for path in testloader.dataset.data
            ]
            if len(set(train_paths)) != len(train_paths):
                raise RuntimeError("CUB base training split contains duplicate image paths.")
            if len(set(test_paths)) != len(test_paths):
                raise RuntimeError("CUB base test split contains duplicate image paths.")
            overlap = set(train_paths).intersection(test_paths)
            if overlap:
                raise RuntimeError(
                    "CUB base train/test split has {} image path overlaps."
                    .format(len(overlap))
                )
            print("[PCAS CUB base split audit]", {
                "train_images": len(train_paths),
                "test_images": len(test_paths),
                "train_test_overlap": len(overlap),
            })
        self._train_base(trainset, trainloader)
        self.all_prototypes = torch.zeros(
            int(self.args.num_classes),
            self.base_prototypes.shape[1],
            device=self.device,
            dtype=self.base_prototypes.dtype,
        )
        self.all_prototypes[:self.args.base_class] = self.base_prototypes

        # Evaluate the base session using the final frozen student prototypes.
        base_metrics = self._evaluate(
            testloader.dataset, self.args.base_class, self.all_prototypes,
            session=0,
        )
        base_metrics.update({"session": 0})
        self.results.append(base_metrics)
        self.base_accuracy = base_metrics["accuracy"]
        self.session_accuracies.append(base_metrics["class_average_accuracy"])
        self.base_session_accuracies.append(base_metrics["base_accuracy"])
        self.final_accuracies.append(base_metrics["accuracy"])
        self.task_wise_accuracies.append(base_metrics["task_wise_accuracy"])
        print("[PCAS base]", base_metrics)
        print(
            "[PCAS session 0] Final acc: {:.2f}; Task-wise acc: {}".format(
                base_metrics["accuracy"] * 100.0,
                [round(value * 100.0, 2) for value in base_metrics["task_wise_accuracy"]],
            )
        )

        for session in range(1, int(self.args.sessions)):
            trainset, _, testloader = get_new_dataloader(self.args, session)
            self._incremental_session(trainset, testloader.dataset, session)

        # Compare base-class accuracy at session 0 and the final session.
        # Comparing macro accuracy across growing class populations conflates
        # retention with the addition of new classes.
        pd = (
            self.base_session_accuracies[0] - self.base_session_accuracies[-1]
            if len(self.base_session_accuracies) > 1 else 0.0
        )
        mean_session_accuracy = float(np.mean(self.session_accuracies))
        final_session_class_accuracy = float(self.session_accuracies[-1])
        summary = {
            # AA is the mean class-average accuracy over all evaluated sessions.
            "AA": mean_session_accuracy,
            "mean_session_class_accuracy": mean_session_accuracy,
            "final_session_class_accuracy": final_session_class_accuracy,
            "final_session_sample_accuracy": float(self.final_accuracies[-1]),
            "PD": float(pd),
            "session_accuracies": self.session_accuracies,
            "base_session_accuracies": self.base_session_accuracies,
            "final_acc_by_session": self.final_accuracies,
            "task_wise_accuracies": self.task_wise_accuracies,
        }
        print("[PCAS summary]", summary)
        print(
            "AA (mean session class accuracy): {:.2f}".format(
                mean_session_accuracy * 100.0
            )
        )
        print(
            "Final session class-average acc: {:.2f}".format(
                final_session_class_accuracy * 100.0
            )
        )
        print(
            "Final acc: {}".format(
                [round(value * 100.0, 2) for value in self.final_accuracies]
            )
        )
        print("Task-wise acc:")
        for task_id, values in enumerate(self.task_wise_accuracies):
            print(
                "task {}, acc: {}".format(
                    task_id, [round(value * 100.0, 2) for value in values]
                )
            )
        self.results.append(summary)
        torch.save(
            {
                "model": self.model.state_dict(),
                "base_student_prototypes": self.base_prototypes.cpu(),
                "all_prototypes": self.all_prototypes.cpu(),
                "session_accuracies": self.session_accuracies,
                "base_session_accuracies": self.base_session_accuracies,
                "AA": summary["AA"],
                "final_session_class_accuracy": summary["final_session_class_accuracy"],
                "final_session_sample_accuracy": summary["final_session_sample_accuracy"],
                "final_acc_by_session": self.final_accuracies,
                "task_wise_accuracies": self.task_wise_accuracies,
                "summary": summary,
                "results": self.results,
            },
            os.path.join(self.save_dir, "pcas_final.pth"),
        )
        with open(os.path.join(self.save_dir, "results.txt"), "w") as handle:
            for row in self.results:
                handle.write(repr(row) + "\n")
        print(f"PCAS completed in {(time.time() - start) / 60:.2f} minutes.")
