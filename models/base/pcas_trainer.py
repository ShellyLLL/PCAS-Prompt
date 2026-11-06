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
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, int(label), int(index)


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
        self.session_accuracies = []
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
        dataset.transform = self._make_eval_transform(dataset)
        loader = self._loader(
            dataset,
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
                      "sample_aff": 0.0, "proto_aff": 0.0}
            num_batches = 0

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
                lambda_aff = float(getattr(self.args, "lambda_affinity", 0.1))
                beta = float(getattr(self.args, "beta_affinity", 0.5))
                loss = (
                    loss_ce
                    + lambda_proto * loss_proto
                    + lambda_aff * (loss_sample + beta * loss_relation)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                totals["total"] += float(loss.detach())
                totals["ce"] += float(loss_ce.detach())
                totals["proto"] += float(loss_proto.detach())
                totals["sample_aff"] += float(loss_sample.detach())
                totals["proto_aff"] += float(loss_relation.detach())
                num_batches += 1

                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PCAS training loss.")

                if aux["routing_weights"] is not None:
                    if not torch.allclose(
                        aux["routing_weights"].sum(-1),
                        torch.ones(images.shape[0], device=self.device),
                        atol=1e-5,
                    ):
                        raise RuntimeError("PCAS routing weights do not sum to one.")

            scheduler.step()
            summary = {
                key: value / max(1, num_batches)
                for key, value in totals.items()
            }
            summary["epoch"] = epoch + 1
            summary["lr"] = scheduler.get_last_lr()[0]
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
    def _evaluate(self, testset, seen_classes, all_prototypes):
        features, labels, _ = self._extract_features(testset, teacher=False)
        logits = float(getattr(self.args, "pcas_logit_scale", 16.0)) * (
            features @ F.normalize(all_prototypes, dim=-1).t()
        )
        logits = logits[:, :seen_classes]
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
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
        return {
            "accuracy": accuracy,
            "class_average_accuracy": class_average_accuracy,
            "base_accuracy": base_acc,
            "novel_accuracy": novel_acc,
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
        metrics = self._evaluate(testset, seen_classes, self.all_prototypes)
        metrics.update({"session": session, "novel_classes": novel_classes.tolist()})
        self.results.append(metrics)
        self.session_accuracies.append(metrics["class_average_accuracy"])
        print("[PCAS incremental]", metrics)
        return metrics

    def train(self):
        start = time.time()
        trainset, trainloader, testloader = get_base_dataloader(self.args)
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
            testloader.dataset, self.args.base_class, self.all_prototypes
        )
        base_metrics.update({"session": 0})
        self.results.append(base_metrics)
        self.base_accuracy = base_metrics["accuracy"]
        self.session_accuracies.append(base_metrics["class_average_accuracy"])
        print("[PCAS base]", base_metrics)

        for session in range(1, int(self.args.sessions)):
            trainset, _, testloader = get_new_dataloader(self.args, session)
            self._incremental_session(trainset, testloader.dataset, session)

        pd = (
            self.session_accuracies[0] - self.session_accuracies[-1]
            if len(self.session_accuracies) > 1 else 0.0
        )
        summary = {
            "AA": float(self.session_accuracies[-1]),
            "mean_session_class_accuracy": float(np.mean(self.session_accuracies)),
            "PD": float(pd),
            "session_accuracies": self.session_accuracies,
        }
        print("[PCAS summary]", summary)
        self.results.append(summary)
        torch.save(
            {
                "model": self.model.state_dict(),
                "base_student_prototypes": self.base_prototypes.cpu(),
                "all_prototypes": self.all_prototypes.cpu(),
                "session_accuracies": self.session_accuracies,
                "summary": summary,
                "results": self.results,
            },
            os.path.join(self.save_dir, "pcas_final.pth"),
        )
        with open(os.path.join(self.save_dir, "results.txt"), "w") as handle:
            for row in self.results:
                handle.write(repr(row) + "\n")
        print(f"PCAS completed in {(time.time() - start) / 60:.2f} minutes.")
