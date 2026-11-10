"""End-to-end LGSP-LSP prototype-based FSCIL trainer."""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import set_up_datasets, get_base_dataloader, get_new_dataloader
from .lsp_modules import LSPViT
from .lsp_losses import build_prototype_cache, prototype_logits
from .pcas_trainer import (
    IndexedDataset,
    compute_taskwise_accuracy,
    validate_exact_shot_support,
)


class ClassBalancedBatchSampler(Sampler):
    """Sample distinct base classes and a fixed number of images per class."""
    def __init__(self, labels, classes_per_batch, samples_per_class, seed):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.class_to_indices = {
            int(c): np.flatnonzero(self.labels == c).tolist()
            for c in np.unique(self.labels)
        }
        if self.classes_per_batch < 2 or len(self.class_to_indices) < self.classes_per_batch:
            raise ValueError("classes_per_batch must be >=2 and available in base data.")
        if self.samples_per_class < 2:
            raise ValueError("samples_per_class must be >=2 for leave-one-out prototypes.")
        for cls, indices in self.class_to_indices.items():
            if len(indices) < self.samples_per_class:
                raise ValueError("Class {} has too few samples.".format(cls))
        self.iterations = max(
            1, int(np.ceil(len(self.labels) /
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
            selected = rng.choice(classes, self.classes_per_batch, replace=False)
            batch = []
            for cls in selected:
                candidates = self.class_to_indices[int(cls)]
                batch.extend(rng.choice(
                    candidates, self.samples_per_class, replace=False
                ).tolist())
            rng.shuffle(batch)
            yield batch


def batch_leave_one_out_logits(features, labels, scale):
    """Compare each query with batch class means excluding that query."""
    features = F.normalize(features, p=2, dim=-1)
    labels = labels.long()
    classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=classes.numel())
    if torch.any(counts < 2):
        raise ValueError("Every sampled class needs at least two examples.")
    sums = torch.zeros(
        classes.numel(), features.shape[1],
        device=features.device, dtype=features.dtype,
    )
    sums.index_add_(0, inverse, features)
    all_prototypes = F.normalize(
        sums / counts.to(features.dtype).unsqueeze(1), p=2, dim=-1
    )
    per_query = all_prototypes.unsqueeze(0).expand(
        features.shape[0], -1, -1
    ).clone()
    query_prototypes = F.normalize(
        sums[inverse] - features,
        p=2, dim=-1,
    )
    per_query[
        torch.arange(features.shape[0], device=features.device), inverse
    ] = query_prototypes
    return torch.einsum("bd,bcd->bc", features, per_query) * float(scale), inverse


class LSPPrototypeFSCILTrainer:
    """Learn LSP on base classes; freeze it and append real K-shot prototypes."""
    def __init__(self, args):
        # Reuse the repository's ViT preprocessing branch for all supported datasets.
        args.vit = True
        self.args = set_up_datasets(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(getattr(args, "seed", 1))
        if self.seed < 0:
            self.seed = 1
        self._set_seed(self.seed)
        self.model = LSPViT(self.args).to(self.device)
        self.repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_root = getattr(args, "out", None) or "LGSP_LSP_Prototype"
        output_root = (
            output_root if os.path.isabs(output_root)
            else os.path.join(self.repo_root, "checkpoint", output_root)
        )
        self.save_dir = os.path.join(
            output_root, self.args.dataset, time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.prototypes = None
        self.session_accuracies, self.sample_accuracies = [], []
        self.task_accuracies, self.results = [], []

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
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
            ])
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _loader(self, dataset, batch_size, shuffle=False, batch_sampler=None):
        workers = int(getattr(self.args, "num_workers", 4))
        return DataLoader(
            dataset, batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle if batch_sampler is None else False,
            batch_sampler=batch_sampler, num_workers=workers,
            pin_memory=self.device.type == "cuda", persistent_workers=workers > 0,
        )

    @torch.no_grad()
    def _extract_features(self, dataset, batch_size=None):
        old_transform, was_training = dataset.transform, self.model.training
        dataset.transform = self._eval_transform()
        loader = self._loader(
            IndexedDataset(dataset),
            int(batch_size or getattr(self.args, "test_batch_size", 128)),
        )
        features, labels = [], []
        self.model.eval()
        try:
            for images, targets, _ in tqdm(loader, desc="Extract LSP features"):
                images = images.to(self.device, non_blocking=True)
                features.append(self.model.encode_student(images))
                labels.append(torch.as_tensor(targets).long().to(self.device))
        finally:
            dataset.transform = old_transform
            self.model.train(was_training)
        return torch.cat(features), torch.cat(labels)

    def _train_base(self, trainset):
        observed = np.unique(np.asarray(trainset.targets)).astype(int)
        expected = np.arange(int(self.args.base_class), dtype=int)
        if not np.array_equal(observed, expected):
            raise RuntimeError("Base class IDs do not match the configured FSCIL protocol.")

        sampler = ClassBalancedBatchSampler(
            trainset.targets,
            int(getattr(self.args, "classes_per_batch", 16)),
            int(getattr(self.args, "samples_per_class", 4)),
            self.seed,
        )
        loader = self._loader(IndexedDataset(trainset), 1, batch_sampler=sampler)
        params = [p for p in self.model.lsp.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable LGSP-LSP parameters.")
        optimizer = torch.optim.AdamW(
            params,
            lr=float(getattr(self.args, "lsp_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "lsp_weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.args.epochs_base))
        )
        scale = float(getattr(self.args, "lsp_logit_scale", 16.0))

        for epoch in range(int(self.args.epochs_base)):
            sampler.set_epoch(epoch)
            self.model.train()
            total_loss, correct, count = 0.0, 0, 0
            for images, labels, _ in tqdm(loader, desc="LSP base epoch {}".format(epoch + 1)):
                images = images.to(self.device, non_blocking=True)
                labels = torch.as_tensor(labels).long().to(self.device).reshape(-1)
                features = self.model.encode_student(images)
                logits, local_targets = batch_leave_one_out_logits(features, labels, scale)
                loss = F.cross_entropy(logits, local_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite LGSP-LSP prototype loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in self.model.lsp.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite LSP gradient: {}.".format(name))
                optimizer.step()
                total_loss += float(loss.detach())
                correct += int((logits.argmax(1) == local_targets).sum().item())
                count += labels.numel()
            scheduler.step()
            record = {
                "epoch": epoch + 1,
                "loss": total_loss / max(1, len(loader)),
                "batch_leave_one_out_accuracy": correct / max(1, count),
                "lr": scheduler.get_last_lr()[0],
            }
            self.results.append(record)
            print("[LGSP-LSP base]", record)

        self.model.freeze_for_incremental()
        features, labels = self._extract_features(trainset)
        self.prototypes = build_prototype_cache(
            features, labels, self.args.base_class
        )["prototypes"].detach()
        torch.save({
            "model": self.model.state_dict(),
            "prototypes": self.prototypes.cpu(),
            "base_classes": int(self.args.base_class),
            "method": "LGSP-LSP-Prototype-FSCIL",
            "feature_space": "frozen_vit_b16_with_frozen_lgsp_lsp",
        }, os.path.join(self.save_dir, "lsp_base.pth"))

    @torch.no_grad()
    def _evaluate(self, testset, seen_classes, session):
        features, labels = self._extract_features(testset)
        if torch.any(labels < 0) or torch.any(labels >= seen_classes):
            raise RuntimeError("Evaluation labels fall outside seen classes.")
        counts = torch.bincount(labels, minlength=seen_classes)
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        if missing:
            raise RuntimeError("Session {} test set misses classes {}.".format(session, missing))
        logits = prototype_logits(
            features, self.prototypes[:seen_classes],
            float(getattr(self.args, "lsp_logit_scale", 16.0)),
        )
        predictions = logits.argmax(1)
        accuracy = float((predictions == labels).float().mean().item())
        class_accuracy = float(torch.stack([
            (predictions[labels == cls] == cls).float().mean()
            for cls in range(seen_classes)
        ]).mean().item())
        task_accuracy = compute_taskwise_accuracy(
            labels, predictions, self.args.base_class, self.args.way, session
        )
        base_mask, novel_mask = labels < self.args.base_class, labels >= self.args.base_class
        base_acc = (
            float((predictions[base_mask] == labels[base_mask]).float().mean().item())
            if base_mask.any() else float("nan")
        )
        novel_acc = (
            float((predictions[novel_mask] == labels[novel_mask]).float().mean().item())
            if novel_mask.any() else float("nan")
        )
        hmean = (
            2 * base_acc * novel_acc / (base_acc + novel_acc)
            if base_mask.any() and novel_mask.any() and base_acc + novel_acc > 0
            else float("nan")
        )
        tbr = (
            float((predictions[novel_mask] < self.args.base_class).float().mean().item())
            if novel_mask.any() else float("nan")
        )
        return {
            "session": int(session), "accuracy": accuracy,
            "class_average_accuracy": class_accuracy,
            "base_accuracy": base_acc, "novel_accuracy": novel_acc,
            "base_novel_hmean": float(hmean),
            "novel_to_base_error_rate": tbr,
            "task_wise_accuracy": task_accuracy,
        }

    @torch.no_grad()
    def _incremental_session(self, trainset, testset, session):
        expected = np.arange(
            self.args.base_class + (session - 1) * self.args.way,
            self.args.base_class + session * self.args.way,
            dtype=int,
        )
        observed = np.unique(np.asarray(trainset.targets)).astype(int)
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                "Session {} support classes {} do not match expected {}.".format(
                    session, observed.tolist(), expected.tolist()
                )
            )
        validate_exact_shot_support(trainset.targets, expected, self.args.shot, session)
        if self.args.dataset in ("cub200", "mini_imagenet", "FGVCAircraft", "iNF200"):
            support_paths = [os.path.realpath(os.path.expanduser(str(p))) for p in trainset.data]
            test_paths = [os.path.realpath(os.path.expanduser(str(p))) for p in testset.data]
            if len(set(support_paths)) != len(support_paths) or len(set(test_paths)) != len(test_paths):
                raise RuntimeError("Duplicate support or test paths in incremental session.")
            if set(support_paths).intersection(test_paths):
                raise RuntimeError("Support/test path overlap in incremental session.")

        support_features, support_labels = self._extract_features(trainset)
        local_labels = support_labels - int(expected[0])
        if torch.any(local_labels < 0) or torch.any(local_labels >= len(expected)):
            raise RuntimeError("Support labels are not contiguous in the expected global class range.")
        new_prototypes = build_prototype_cache(
            support_features, local_labels, len(expected)
        )["prototypes"]
        # Append only new class means; previously stored prototypes are immutable.
        self.prototypes[torch.as_tensor(expected, device=self.device)] = new_prototypes

        seen_classes = self.args.base_class + session * self.args.way
        metrics = self._evaluate(testset, seen_classes, session)
        self.results.append(metrics)
        self.session_accuracies.append(metrics["class_average_accuracy"])
        self.sample_accuracies.append(metrics["accuracy"])
        self.task_accuracies.append(metrics["task_wise_accuracy"])
        print("[LGSP-LSP incremental]", metrics)

    def train(self):
        start = time.time()
        trainset, _, testloader = get_base_dataloader(self.args)
        self._train_base(trainset)
        self.prototypes = torch.cat([
            self.prototypes,
            torch.zeros(
                int(self.args.num_classes) - int(self.args.base_class),
                self.prototypes.shape[1], device=self.device,
                dtype=self.prototypes.dtype,
            ),
        ], dim=0)

        base_metrics = self._evaluate(testloader.dataset, self.args.base_class, 0)
        self.results.append(base_metrics)
        self.session_accuracies.append(base_metrics["class_average_accuracy"])
        self.sample_accuracies.append(base_metrics["accuracy"])
        self.task_accuracies.append(base_metrics["task_wise_accuracy"])
        print("[LGSP-LSP base evaluation]", base_metrics)

        for session in range(1, int(self.args.sessions)):
            trainset, _, testloader = get_new_dataloader(self.args, session)
            self._incremental_session(trainset, testloader.dataset, session)

        summary = {
            "AA": float(np.mean(self.session_accuracies)),
            "PD": float(self.session_accuracies[0] - self.session_accuracies[-1]),
            "session_accuracies": self.session_accuracies,
            "sample_accuracies": self.sample_accuracies,
            "task_wise_accuracies": self.task_accuracies,
            "final_session_class_accuracy": float(self.session_accuracies[-1]),
            "final_session_sample_accuracy": float(self.sample_accuracies[-1]),
        }
        self.results.append(summary)
        print("[LGSP-LSP summary]", summary)
        torch.save({
            "model": self.model.state_dict(),
            "prototypes": self.prototypes.cpu(),
            "summary": summary, "results": self.results,
        }, os.path.join(self.save_dir, "lsp_final.pth"))
        with open(os.path.join(self.save_dir, "results.txt"), "w") as handle:
            for record in self.results:
                handle.write(repr(record) + "\n")
        print("LGSP-LSP FSCIL completed in {:.2f} minutes.".format(
            (time.time() - start) / 60.0
        ))
