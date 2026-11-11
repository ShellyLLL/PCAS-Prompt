"""FSCIL trainer for Prompt-Slot decomposition and local-set OT calibration."""
import os
import time
import random

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
from .slot_ot_modules import PromptSlotViT
from .slot_ot_losses import (
    build_class_local_memory,
    calibrate_local_memory,
    joint_prototype_logits,
    pairwise_local_set_distance,
    weighted_kmeans,
)


class IndexedDataset(Dataset):
    """Attach the local dataset index without changing dataset label semantics."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) not in (2, 3):
            raise TypeError("FSCIL datasets must return (image,label[,metadata]).")
        image, label = sample[0], sample[1]
        return image, int(label), int(index)


class BalancedEpisodeSampler(Sampler):
    """Sample base-class episodes with disjoint support and query examples."""

    def __init__(
        self,
        labels,
        classes_per_episode=12,
        support_per_class=2,
        query_per_class=2,
        seed=1,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_episode = int(classes_per_episode)
        self.support_per_class = int(support_per_class)
        self.query_per_class = int(query_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices = {
            int(class_id): np.flatnonzero(self.labels == class_id).tolist()
            for class_id in np.unique(self.labels)
        }
        if self.classes_per_episode < 2:
            raise ValueError("classes_per_episode must be at least two.")
        if len(self.class_to_indices) < self.classes_per_episode:
            raise ValueError(
                "The base split has fewer classes than classes_per_episode."
            )
        required = self.support_per_class + self.query_per_class
        if self.support_per_class <= 0 or self.query_per_class <= 0:
            raise ValueError("Support and query counts must be positive.")
        for class_id, indices in self.class_to_indices.items():
            if len(indices) < required:
                raise ValueError(
                    "Base class {} has {} samples; {} are required per episode."
                    .format(class_id, len(indices), required)
                )
        self.iterations = max(
            1,
            int(np.ceil(
                len(self.labels)
                / (self.classes_per_episode * required)
            )),
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.iterations

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        classes = np.asarray(sorted(self.class_to_indices), dtype=np.int64)
        required = self.support_per_class + self.query_per_class
        for _ in range(self.iterations):
            selected = rng.choice(
                classes, self.classes_per_episode, replace=False
            )
            batch = []
            for class_id in selected:
                candidates = self.class_to_indices[int(class_id)]
                chosen = rng.choice(candidates, required, replace=False)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch


def validate_exact_shot(labels, expected_classes, shots, session):
    labels = np.asarray(labels, dtype=np.int64)
    expected_classes = np.asarray(expected_classes, dtype=np.int64)
    observed = np.unique(labels)
    if not np.array_equal(observed, expected_classes):
        raise RuntimeError(
            "Session {} support classes {} do not match expected {}."
            .format(session, observed.tolist(), expected_classes.tolist())
        )
    counts = {int(c): int((labels == c).sum()) for c in expected_classes}
    if any(value != int(shots) for value in counts.values()):
        raise RuntimeError(
            "Session {} is not an exact {}-shot split: {}."
            .format(session, shots, counts)
        )
    return counts


def validate_seen_classes(labels, seen_classes, session):
    labels = torch.as_tensor(labels).long()
    if labels.numel() == 0 or torch.any(labels < 0) or torch.any(labels >= seen_classes):
        raise RuntimeError("Invalid evaluation labels in session {}.".format(session))
    counts = torch.bincount(labels, minlength=int(seen_classes))
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(
            "Session {} evaluation split misses classes {}.".format(session, missing)
        )


class PromptSlotOTFSCILTrainer:
    """Train shared visual slots on base classes, then calibrate K-shot memories."""

    PATH_BASED_DATASETS = ("cub200", "mini_imagenet", "FGVCAircraft", "iNF200")

    def __init__(self, args):
        # This method is explicitly a ViT method; ensure the existing loaders
        # select their ViT image path and never route through a text encoder.
        args.vit = True
        args.clip = False
        self.args = set_up_datasets(args)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = int(getattr(args, "seed", 1))
        if self.seed < 0:
            self.seed = 1
        self._set_seed(self.seed)

        self.model = PromptSlotViT(self.args).to(self.device)
        self.output_root = str(getattr(args, "slot_out", "PromptSlotOT"))
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_root = (
            self.output_root if os.path.isabs(self.output_root)
            else os.path.join(repo_root, "checkpoint", self.output_root)
        )
        self.save_dir = os.path.join(
            output_root, self.args.dataset, time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.class_global = {}
        self.class_local = {}
        self.class_mass = {}
        self.base_global = None
        self.base_local = None
        self.base_mass = None
        self.session_accuracies = []
        self.sample_accuracies = []
        self.task_accuracies = []
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

    def _eval_transform(self):
        if self.args.dataset == "cifar100":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
                ),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ])

    def _loader(self, dataset, batch_size, shuffle=False, batch_sampler=None):
        kwargs = {
            "dataset": dataset,
            "num_workers": int(getattr(self.args, "num_workers", 4)),
            "pin_memory": self.device.type == "cuda",
        }
        if batch_sampler is not None:
            kwargs["batch_sampler"] = batch_sampler
        else:
            kwargs["batch_size"] = int(batch_size)
            kwargs["shuffle"] = bool(shuffle)
        return DataLoader(**kwargs)

    @torch.no_grad()
    def _extract(self, dataset, batch_size=None):
        """Extract deterministic features and return them on CPU to bound VRAM."""
        original_transform = dataset.transform
        was_training = self.model.training
        dataset.transform = self._eval_transform()
        loader = self._loader(
            IndexedDataset(dataset),
            batch_size or int(getattr(self.args, "test_batch_size", 128)),
            shuffle=False,
        )
        global_features, local_features, local_weights, labels = [], [], [], []
        self.model.eval()
        try:
            for images, targets, _ in tqdm(loader, desc="Extract Slot-OT features"):
                output = self.model(images.to(self.device, non_blocking=True))
                global_features.append(output["global_features"].cpu())
                local_features.append(output["local_features"].cpu())
                local_weights.append(output["local_weights"].cpu())
                labels.append(torch.as_tensor(targets).long().cpu())
        finally:
            dataset.transform = original_transform
            self.model.train(was_training)
        if not global_features:
            raise RuntimeError("Cannot extract features from an empty dataset.")
        return (
            torch.cat(global_features),
            torch.cat(local_features),
            torch.cat(local_weights),
            torch.cat(labels),
        )

    @staticmethod
    def _global_prototypes(features, labels, num_classes):
        prototypes = []
        for class_id in range(int(num_classes)):
            mask = labels == class_id
            if not torch.any(mask):
                raise RuntimeError("No global features for class {}.".format(class_id))
            prototypes.append(F.normalize(features[mask].mean(dim=0), dim=-1))
        return torch.stack(prototypes)

    def _split_episode(self, labels):
        support_indices, query_indices = [], []
        class_ids = torch.unique(labels, sorted=True)
        support_count = int(getattr(self.args, "slot_support_per_class", 2))
        query_count = int(getattr(self.args, "slot_query_per_class", 2))
        for class_id in class_ids:
            indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
            if indices.numel() != support_count + query_count:
                raise RuntimeError("Malformed class-balanced episode.")
            indices = indices[torch.randperm(indices.numel())]
            support_indices.extend(indices[:support_count].tolist())
            query_indices.extend(indices[support_count:].tolist())
        return (
            class_ids,
            torch.as_tensor(support_indices, dtype=torch.long),
            torch.as_tensor(query_indices, dtype=torch.long),
        )

    def _train_base(self, trainset):
        observed = np.unique(np.asarray(trainset.targets)).astype(int)
        expected = np.arange(int(self.args.base_class), dtype=int)
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                "Base classes {} do not match expected {}."
                .format(observed.tolist(), expected.tolist())
            )

        sampler = BalancedEpisodeSampler(
            trainset.targets,
            classes_per_episode=int(getattr(self.args, "slot_classes_per_episode", 12)),
            support_per_class=int(getattr(self.args, "slot_support_per_class", 2)),
            query_per_class=int(getattr(self.args, "slot_query_per_class", 2)),
            seed=self.seed,
        )
        loader = self._loader(
            IndexedDataset(trainset), batch_size=1, batch_sampler=sampler
        )
        parameters = [
            parameter for parameter in self.model.slot_attention.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(getattr(self.args, "slot_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "slot_weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.args.epochs_base))
        )
        temperature = float(getattr(self.args, "slot_temperature", 0.10))
        global_weight = float(getattr(self.args, "slot_global_weight", 0.5))
        ot_args = {
            "epsilon": float(getattr(self.args, "slot_ot_epsilon", 0.05)),
            "tau": float(getattr(self.args, "slot_ot_tau", 0.10)),
            "iterations": int(getattr(self.args, "slot_ot_iterations", 20)),
        }

        for epoch in range(int(self.args.epochs_base)):
            sampler.set_epoch(epoch)
            self.model.train()
            total_loss, total_local_acc, total_joint_acc, batches = 0.0, 0.0, 0.0, 0
            for images, labels, _ in tqdm(
                loader, desc="Prompt-Slot base epoch {}".format(epoch + 1)
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.long().to(self.device)
                class_ids, support_idx, query_idx = self._split_episode(labels)
                support_images = images[support_idx.to(images.device)]
                query_images = images[query_idx.to(images.device)]
                support_labels = labels[support_idx.to(labels.device)]
                query_labels = labels[query_idx.to(labels.device)]

                with torch.no_grad():
                    support_output = self.model(support_images)
                query_output = self.model(query_images)

                support_global = []
                support_local = []
                support_mass = []
                for class_id in class_ids:
                    mask = support_labels == class_id
                    support_global.append(
                        F.normalize(
                            support_output["global_features"][mask].mean(dim=0),
                            dim=-1,
                        )
                    )
                    support_local.append(
                        support_output["local_features"][mask].reshape(
                            -1, support_output["local_features"].shape[-1]
                        )
                    )
                    weights = support_output["local_weights"][mask].reshape(-1)
                    support_mass.append(weights / weights.sum().clamp_min(1e-8))
                support_global = torch.stack(support_global)
                support_local = torch.stack(support_local)
                support_mass = torch.stack(support_mass)
                targets = torch.searchsorted(class_ids, query_labels)

                local_distance, _ = pairwise_local_set_distance(
                    query_output["local_features"],
                    query_output["local_weights"],
                    support_local,
                    support_mass,
                    **ot_args,
                )
                global_distance = (
                    1.0
                    - query_output["global_features"] @ support_global.t()
                ).clamp_min(0.0)
                joint_distance = (
                    global_weight * global_distance
                    + (1.0 - global_weight) * local_distance
                )
                local_logits = -local_distance / temperature
                joint_logits = -joint_distance / temperature
                loss = 0.5 * (
                    F.cross_entropy(local_logits, targets)
                    + F.cross_entropy(joint_logits, targets)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Prompt-Slot OT loss.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.detach().item())
                total_local_acc += float(
                    (local_logits.argmax(dim=1) == targets).float().mean().item()
                )
                total_joint_acc += float(
                    (joint_logits.argmax(dim=1) == targets).float().mean().item()
                )
                batches += 1

            scheduler.step()
            if batches == 0:
                raise RuntimeError("No base training episodes were generated.")
            print("[Prompt-Slot base]", {
                "epoch": epoch + 1,
                "loss": total_loss / batches,
                "local_episode_accuracy": total_local_acc / batches,
                "joint_episode_accuracy": total_joint_acc / batches,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

    def _build_base_memory(self, trainset):
        global_features, local_features, local_weights, labels = self._extract(trainset)
        self.base_global = self._global_prototypes(
            global_features, labels, self.args.base_class
        )
        self.base_local, self.base_mass = build_class_local_memory(
            local_features,
            local_weights,
            labels,
            self.args.base_class,
            int(getattr(self.args, "slot_memory_size", 6)),
        )
        for class_id in range(int(self.args.base_class)):
            self.class_global[class_id] = self.base_global[class_id]
            self.class_local[class_id] = self.base_local[class_id]
            self.class_mass[class_id] = self.base_mass[class_id]

        torch.save(
            {
                "model": {
                    key: value.detach().cpu()
                    for key, value in self.model.state_dict().items()
                },
                "base_global_prototypes": self.base_global,
                "base_local_memory": self.base_local,
                "base_local_mass": self.base_mass,
            },
            os.path.join(self.save_dir, "slot_ot_base.pth"),
        )

    def _audit_support_test(self, support_set, test_set, session):
        if self.args.dataset not in self.PATH_BASED_DATASETS:
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
            raise RuntimeError("Session {} has duplicate support paths.".format(session))
        if len(set(test_paths)) != len(test_paths):
            raise RuntimeError("Session {} has duplicate test paths.".format(session))
        overlap = set(support_paths).intersection(test_paths)
        if overlap:
            raise RuntimeError(
                "Session {} has {} support/test path overlaps."
                .format(session, len(overlap))
            )

    def _incremental_session(self, support_set, test_set, session):
        start_class = int(self.args.base_class) + (session - 1) * int(self.args.way)
        expected_classes = np.arange(
            start_class, start_class + int(self.args.way), dtype=int
        )
        validate_exact_shot(
            support_set.targets, expected_classes, self.args.shot, session
        )
        self._audit_support_test(support_set, test_set, session)

        global_features, local_features, local_weights, labels = self._extract(
            support_set
        )
        memory_size = int(getattr(self.args, "slot_memory_size", 6))
        alpha = float(getattr(self.args, "slot_calibration_alpha", 0.25))
        diagnostics = {}

        for class_id in expected_classes.tolist():
            mask = labels == int(class_id)
            if not torch.any(mask):
                raise RuntimeError("No support features for novel class {}.".format(class_id))
            novel_global = F.normalize(global_features[mask].mean(dim=0), dim=-1)
            novel_local, novel_mass = weighted_kmeans(
                local_features[mask].reshape(-1, local_features.shape[-1]),
                local_weights[mask].reshape(-1),
                memory_size,
            )
            calibrated_local, calibration_info = calibrate_local_memory(
                novel_local,
                novel_mass,
                novel_global,
                self.base_local,
                self.base_mass,
                self.base_global,
                top_k=int(getattr(self.args, "slot_calibration_topk", 5)),
                max_alpha=alpha,
                global_temperature=float(
                    getattr(self.args, "slot_global_retrieval_temperature", 0.05)
                ),
                match_temperature=float(
                    getattr(self.args, "slot_match_temperature", 0.20)
                ),
                epsilon=float(getattr(self.args, "slot_ot_epsilon", 0.05)),
                tau=float(getattr(self.args, "slot_ot_tau", 0.10)),
                iterations=int(getattr(self.args, "slot_ot_iterations", 20)),
            )
            self.class_global[int(class_id)] = novel_global
            self.class_local[int(class_id)] = calibrated_local
            # Preserve the support-derived local mass; calibration changes
            # descriptor content, not the amount of visual evidence.
            self.class_mass[int(class_id)] = novel_mass
            diagnostics[int(class_id)] = {
                "alpha": calibration_info["slot_calibration_alpha"].tolist(),
                "match_confidence": calibration_info["slot_match_confidence"].tolist(),
                "retrieved_base_classes": (
                    calibration_info["retrieved_base_classes"].tolist()
                ),
                "retrieved_base_weights": (
                    calibration_info["retrieved_base_weights"].tolist()
                ),
            }

        seen_classes = int(self.args.base_class) + session * int(self.args.way)
        metrics = self._evaluate(test_set, seen_classes, session)
        metrics.update({
            "session": session,
            "novel_classes": expected_classes.tolist(),
            "local_calibration": diagnostics,
        })
        self.results.append(metrics)
        self.session_accuracies.append(metrics["class_average_accuracy"])
        self.sample_accuracies.append(metrics["accuracy"])
        self.task_accuracies.append(metrics["task_wise_accuracy"])
        print("[Prompt-Slot incremental]", metrics)
        return metrics

    @torch.no_grad()
    def _evaluate(self, dataset, seen_classes, session):
        global_features, local_features, local_weights, labels = self._extract(dataset)
        global_prototypes = torch.stack([
            self.class_global[class_id] for class_id in range(seen_classes)
        ]).to(self.device)
        local_memory = torch.stack([
            self.class_local[class_id] for class_id in range(seen_classes)
        ]).to(self.device)
        local_mass = torch.stack([
            self.class_mass[class_id] for class_id in range(seen_classes)
        ]).to(self.device)

        predictions = []
        batch_size = int(getattr(self.args, "test_batch_size", 128))
        ot_args = {
            "epsilon": float(getattr(self.args, "slot_ot_epsilon", 0.05)),
            "tau": float(getattr(self.args, "slot_ot_tau", 0.10)),
            "iterations": int(getattr(self.args, "slot_ot_iterations", 20)),
            "temperature": float(getattr(self.args, "slot_temperature", 0.10)),
            "global_weight": float(getattr(self.args, "slot_global_weight", 0.5)),
        }
        for start in range(0, labels.numel(), batch_size):
            end = min(start + batch_size, labels.numel())
            logits, _, _ = joint_prototype_logits(
                global_features[start:end].to(self.device),
                local_features[start:end].to(self.device),
                local_weights[start:end].to(self.device),
                global_prototypes,
                local_memory,
                local_mass,
                **ot_args,
            )
            predictions.append(logits.argmax(dim=1).cpu())
        predictions = torch.cat(predictions)
        labels = labels.long()
        validate_seen_classes(labels, seen_classes, session)

        class_accuracy = []
        for class_id in range(seen_classes):
            mask = labels == class_id
            class_accuracy.append(
                float((predictions[mask] == labels[mask]).float().mean().item())
            )
        class_average = float(np.mean(class_accuracy))
        accuracy = float((predictions == labels).float().mean().item())
        base_mask = labels < int(self.args.base_class)
        novel_mask = ~base_mask
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
            if base_mask.any() and novel_mask.any()
            and (base_accuracy + novel_accuracy) > 0
            else float("nan")
        )
        tbr = (
            float((predictions[novel_mask] < int(self.args.base_class)).float().mean().item())
            if novel_mask.any() else float("nan")
        )

        task_ranges = [(0, int(self.args.base_class))]
        task_ranges.extend(
            (
                int(self.args.base_class) + task_id * int(self.args.way),
                int(self.args.base_class) + (task_id + 1) * int(self.args.way),
            )
            for task_id in range(session)
        )
        task_wise = []
        for task_id, (lower, upper) in enumerate(task_ranges):
            mask = (labels >= lower) & (labels < upper)
            if not mask.any():
                raise RuntimeError(
                    "Evaluation session {} has no samples for task {}."
                    .format(session, task_id)
                )
            task_wise.append(
                float((predictions[mask] == labels[mask]).float().mean().item())
            )
        return {
            "accuracy": accuracy,
            "class_average_accuracy": class_average,
            "base_accuracy": base_accuracy,
            "novel_accuracy": novel_accuracy,
            "base_novel_hmean": hmean,
            "novel_to_base_error_rate": tbr,
            "task_wise_accuracy": task_wise,
        }

    def train(self):
        start_time = time.time()
        trainset, _, base_testloader = get_base_dataloader(self.args)
        self._train_base(trainset)
        self._build_base_memory(trainset)

        base_metrics = self._evaluate(
            base_testloader.dataset, int(self.args.base_class), session=0
        )
        base_metrics["session"] = 0
        self.results.append(base_metrics)
        self.session_accuracies.append(base_metrics["class_average_accuracy"])
        self.sample_accuracies.append(base_metrics["accuracy"])
        self.task_accuracies.append(base_metrics["task_wise_accuracy"])
        print("[Prompt-Slot base evaluation]", base_metrics)

        for session in range(1, int(self.args.sessions)):
            support_set, _, testloader = get_new_dataloader(self.args, session)
            self._incremental_session(
                support_set, testloader.dataset, session
            )

        final_class_accuracy = self.session_accuracies[-1]
        summary = {
            "AA": float(np.mean(self.session_accuracies)),
            "PD": float(self.session_accuracies[0] - final_class_accuracy),
            "base_session_class_accuracy": float(self.session_accuracies[0]),
            "final_session_class_accuracy": float(final_class_accuracy),
            "final_session_sample_accuracy": float(self.sample_accuracies[-1]),
            "session_class_accuracies": self.session_accuracies,
            "session_sample_accuracies": self.sample_accuracies,
            "task_wise_accuracies": self.task_accuracies,
        }
        self.results.append(summary)

        checkpoint = {
            "model": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "class_global_prototypes": torch.stack([
                self.class_global[class_id]
                for class_id in range(int(self.args.num_classes))
                if class_id in self.class_global
            ]),
            "class_local_memory": torch.stack([
                self.class_local[class_id]
                for class_id in range(int(self.args.num_classes))
                if class_id in self.class_local
            ]),
            "class_local_mass": torch.stack([
                self.class_mass[class_id]
                for class_id in range(int(self.args.num_classes))
                if class_id in self.class_mass
            ]),
            "summary": summary,
            "results": self.results,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "slot_ot_final.pth"))
        with open(os.path.join(self.save_dir, "results.txt"), "w") as handle:
            for row in self.results:
                handle.write(repr(row) + "\n")

        print("[Prompt-Slot FSCIL summary]", summary)
        print(
            "Prompt-Slot FSCIL completed in {:.2f} minutes.".format(
                (time.time() - start_time) / 60.0
            )
        )
        return summary
