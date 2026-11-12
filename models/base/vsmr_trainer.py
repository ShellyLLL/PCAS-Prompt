"""End-to-end VSMR FSCIL trainer with append-only prototype memory."""
import json
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
    get_base_dataloader, get_new_dataloader, set_up_datasets,
)
from .vsmr_modules import VSMRViT
from .vsmr_losses import (
    attention_overlap_loss, cross_image_semantic_alignment,
    global_logits, leave_one_out_global, local_leave_one_out_logits,
    local_logits, ot_barycenter,
)
from .vsmr_resynthesis import (
    build_base_component_bank, compose_novel_prototypes,
    query_conditioned_reconstruction,
)


class ImageLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if len(sample) == 2:
            image, label = sample
        elif len(sample) == 3:
            image, label, _ = sample
        else:
            raise ValueError("Dataset samples must contain image, label[, metadata].")
        return image, int(label)


class ClassBalancedBatchSampler(Sampler):
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
        if self.classes_per_batch < 2 or self.samples_per_class < 2:
            raise ValueError("VSMR requires >=2 classes and >=2 samples per class.")
        if len(self.class_to_indices) < self.classes_per_batch:
            raise ValueError("Not enough base classes for classes_per_batch.")
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
            chosen = rng.choice(classes, self.classes_per_batch, replace=False)
            indices = []
            for cls in chosen:
                indices.extend(rng.choice(
                    self.class_to_indices[int(cls)],
                    self.samples_per_class, replace=False
                ).tolist())
            rng.shuffle(indices)
            yield indices


def validate_exact_shot(labels, expected, shots, session):
    labels = np.asarray(labels, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.int64)
    if not np.array_equal(np.unique(labels), expected):
        raise RuntimeError("Session {} support classes mismatch.".format(session))
    counts = {int(c): int((labels == c).sum()) for c in expected}
    if any(v != int(shots) for v in counts.values()):
        raise RuntimeError(
            "Session {} must be exact {}-shot; counts={}".format(
                session, shots, counts
            )
        )


def class_average_accuracy(labels, predictions, classes):
    values = []
    for cls in range(int(classes)):
        mask = labels == cls
        if not mask.any():
            raise RuntimeError("Evaluation is missing seen class {}.".format(cls))
        values.append((predictions[mask] == labels[mask]).float().mean())
    return float(torch.stack(values).mean().item())


class VSMRFSCILTrainer:
    def __init__(self, args):
        args.vit = True
        self.args = set_up_datasets(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = max(0, int(getattr(args, "seed", 1)))
        self._set_seed(self.seed)
        self.model = VSMRViT(self.args).to(self.device)

        root = getattr(args, "out", None) or "VSMR"
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        root = root if os.path.isabs(root) else os.path.join(repo_root, "checkpoint", root)
        self.save_dir = os.path.join(
            root, self.args.dataset, "VSMR",
            time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, "metrics.jsonl")
        self.results = []
        self.base_parts = None
        self.base_global = None
        self.base_bank = None
        self.all_global = None
        self.all_parts = None
        self.all_raw_parts = None

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _transform(self):
        if self.args.dataset == "cifar100":
            mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])

    def _loader(self, dataset, batch_size, batch_sampler=None):
        workers = int(getattr(self.args, "num_workers", 4))
        return DataLoader(
            ImageLabelDataset(dataset),
            batch_size=1 if batch_sampler is not None else int(batch_size),
            shuffle=False,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )

    @torch.no_grad()
    def _extract(self, dataset, batch_size=None):
        original = getattr(dataset, "transform", None)
        if original is None:
            raise RuntimeError("Dataset must expose a transform.")
        dataset.transform = self._transform()
        loader = self._loader(
            dataset, batch_size or int(getattr(self.args, "test_batch_size", 128))
        )
        globals_, parts_, labels_ = [], [], []
        self.model.eval()
        try:
            for images, labels in tqdm(loader, desc="VSMR feature extraction"):
                output = self.model(images.to(self.device, non_blocking=True))
                globals_.append(output["global_features"])
                parts_.append(output["parts"])
                labels_.append(torch.as_tensor(labels, device=self.device).long())
        finally:
            dataset.transform = original
        if not labels_:
            raise RuntimeError("Cannot extract features from an empty dataset.")
        return torch.cat(globals_), torch.cat(parts_), torch.cat(labels_)

    def _append_log(self, record):
        self.results.append(record)
        with open(self.log_path, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("[VSMR] " + json.dumps(record, ensure_ascii=False, sort_keys=True))

    def _train_base(self, trainset):
        labels = np.asarray(trainset.targets, dtype=np.int64)
        if not np.array_equal(np.unique(labels), np.arange(int(self.args.base_class))):
            raise RuntimeError("Base labels must be contiguous [0, base_class).")
        classes_per_batch = min(
            int(getattr(self.args, "vsmr_classes_per_batch", 16)),
            int(self.args.base_class),
        )
        sampler = ClassBalancedBatchSampler(
            labels, classes_per_batch,
            int(getattr(self.args, "vsmr_samples_per_class", 4)), self.seed
        )
        loader = self._loader(trainset, 1, batch_sampler=sampler)
        optimizer = torch.optim.AdamW(
            self.model.decomposer.parameters(),
            lr=float(getattr(self.args, "vsmr_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "vsmr_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        eps = float(getattr(self.args, "vsmr_ot_epsilon", 0.05))
        sinkhorn_steps = int(getattr(self.args, "vsmr_sinkhorn_iterations", 30))
        temperature = float(getattr(self.args, "vsmr_local_temperature", 0.1))
        w_local = float(getattr(self.args, "vsmr_local_weight", 1.0))
        w_align = float(getattr(self.args, "vsmr_alignment_weight", 0.2))
        w_cisa = float(getattr(self.args, "vsmr_cisa_weight", 0.1))
        w_overlap = float(getattr(self.args, "vsmr_overlap_weight", 0.01))
        scale = float(getattr(self.args, "vsmr_logit_scale", 16.0))

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            self.model.train()
            totals = {k: 0.0 for k in (
                "total", "global", "local", "alignment", "cisa", "overlap",
                "global_acc", "local_acc", "joint_acc", "ot_same"
            )}
            steps = 0
            for images, batch_labels in tqdm(
                loader, desc="VSMR base {}/{}".format(epoch + 1, epochs)
            ):
                images = images.to(self.device, non_blocking=True)
                labels_batch = torch.as_tensor(batch_labels, device=self.device).long()
                output = self.model(images)
                features, parts, attention = (
                    output["global_features"], output["parts"], output["attention"]
                )
                loo_global, inverse, classes = leave_one_out_global(features, labels_batch)
                # Other-class prototypes are batch means; own-class column is
                # replaced by its leave-one-out prototype.
                class_features = []
                for cls in classes:
                    class_features.append(F.normalize(
                        features[labels_batch == cls].mean(0), dim=-1
                    ))
                class_prototypes = torch.stack(class_features)
                g_logits = global_logits(features, class_prototypes, scale)
                g_logits = g_logits.scatter(
                    1, inverse[:, None],
                    (features * loo_global).sum(-1, keepdim=True) * scale
                )
                l_logits, local_classes = local_leave_one_out_logits(
                    parts, labels_batch, eps, sinkhorn_steps, temperature
                )
                if not torch.equal(classes, local_classes):
                    raise RuntimeError("Global/local class order mismatch.")
                # Same-class OT alignment, using all distinct image pairs.
                from .vsmr_losses import pairwise_ot_distances
                pair_dist = pairwise_ot_distances(parts, eps, sinkhorn_steps)
                same = labels_batch[:, None].eq(labels_batch[None, :])
                same.fill_diagonal_(False)
                align_loss = pair_dist[same].mean()
                global_loss = F.cross_entropy(g_logits, inverse)
                local_loss = F.cross_entropy(l_logits, inverse)
                cisa_loss = cross_image_semantic_alignment(
                    g_logits, l_logits, labels_batch
                )
                overlap_loss = attention_overlap_loss(attention)
                total = (
                    global_loss + w_local * local_loss
                    + w_align * align_loss + w_cisa * cisa_loss
                    + w_overlap * overlap_loss
                )
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite VSMR loss.")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                for name, parameter in self.model.decomposer.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite gradient: " + name)
                optimizer.step()

                with torch.no_grad():
                    totals["total"] += float(total)
                    totals["global"] += float(global_loss)
                    totals["local"] += float(local_loss)
                    totals["alignment"] += float(align_loss)
                    totals["cisa"] += float(cisa_loss)
                    totals["overlap"] += float(overlap_loss)
                    totals["global_acc"] += float((g_logits.argmax(1) == inverse).float().mean())
                    totals["local_acc"] += float((l_logits.argmax(1) == inverse).float().mean())
                    totals["joint_acc"] += float(((g_logits + w_local * l_logits).argmax(1) == inverse).float().mean())
                    totals["ot_same"] += float(pair_dist[same].mean())
                steps += 1
            scheduler.step()
            record = {"stage": "base_train", "epoch": epoch + 1}
            record.update({k: v / max(steps, 1) for k, v in totals.items()})
            record["lr"] = scheduler.get_last_lr()[0]
            self._append_log(record)

        self.model.freeze_for_incremental()
        self._build_base_memory(trainset)
        torch.save({
            "model": self.model.state_dict(),
            "base_global": self.base_global.cpu(),
            "base_parts": self.base_parts.cpu(),
            "base_component_bank": self.base_bank.cpu(),
            "method": "VSMR",
        }, os.path.join(self.save_dir, "vsmr_base.pth"))

    @torch.no_grad()
    def _build_base_memory(self, trainset):
        global_features, parts, labels = self._extract(trainset)
        prototypes = []
        global_prototypes = []
        for cls in range(int(self.args.base_class)):
            mask = labels == cls
            if not mask.any():
                raise RuntimeError("Base memory missing class {}".format(cls))
            global_prototypes.append(F.normalize(global_features[mask].mean(0), dim=-1))
            prototypes.append(ot_barycenter(
                parts[mask],
                float(getattr(self.args, "vsmr_ot_epsilon", 0.05)),
                int(getattr(self.args, "vsmr_sinkhorn_iterations", 30)),
                int(getattr(self.args, "vsmr_barycenter_iterations", 5)),
            ))
        self.base_global = torch.stack(global_prototypes).detach()
        self.base_parts = torch.stack(prototypes).detach()
        self.base_bank, self.base_owner, self.base_part_ids = build_base_component_bank(
            self.base_parts
        )
        self.all_global = self.base_global.clone()
        self.all_parts = self.base_parts.clone()
        self.all_raw_parts = self.base_parts.clone()
        self._append_log({
            "stage": "base_memory",
            "base_classes": int(self.base_global.shape[0]),
            "parts_per_class": int(self.base_parts.shape[1]),
            "component_bank_size": int(self.base_bank.shape[0]),
        })

    @torch.no_grad()
    def _evaluate(self, testset, seen_classes, session):
        original = getattr(testset, "transform", None)
        if original is None:
            raise RuntimeError("Evaluation dataset must expose a transform.")
        testset.transform = self._transform()
        loader = self._loader(
            testset, int(getattr(self.args, "vsmr_eval_batch_size", 8))
        )
        eps = float(getattr(self.args, "vsmr_ot_epsilon", 0.05))
        steps = int(getattr(self.args, "vsmr_sinkhorn_iterations", 30))
        scale = float(getattr(self.args, "vsmr_logit_scale", 16.0))
        wg = float(getattr(self.args, "vsmr_global_weight", 0.5))
        wl = float(getattr(self.args, "vsmr_local_weight_eval", 0.5))
        if wg < 0 or wl < 0 or wg + wl <= 0:
            raise ValueError("Global/local classification weights are invalid.")
        pred, target = [], []
        transfer_scores, no_transfer_scores, fixed_scores = [], [], []
        self.model.eval()
        try:
            for images, labels in tqdm(loader, desc="VSMR eval session {}".format(session)):
                images = images.to(self.device, non_blocking=True)
                labels = torch.as_tensor(labels, device=self.device).long()
                output = self.model(images)
                gf, qp = output["global_features"], output["parts"]
                gl = global_logits(gf, self.all_global, scale)
                local_score, _, transport = query_conditioned_reconstruction(
                    qp, self.all_parts, eps, steps
                )
                local_score = local_score * scale
                logits = (wg * gl + wl * local_score) / (wg + wl)
                pred.append(logits.argmax(1).cpu())
                target.append(labels.cpu())

                # Query reconstruction ablation: fixed index pairing is not
                # allowed in the main path; report it only as a diagnostic.
                fixed = (F.normalize(qp, dim=-1) *
                         F.normalize(self.all_parts, dim=-1)[None]).sum(-1).mean(-1)
                fixed_scores.append(fixed.cpu())
                transfer_scores.append(local_score.cpu())
                no_transfer = query_conditioned_reconstruction(
                    qp, self.all_raw_parts, eps, steps
                )[0] * scale
                no_transfer_scores.append(no_transfer.cpu())
        finally:
            testset.transform = original
        predictions, labels = torch.cat(pred), torch.cat(target)
        transferred_local = torch.cat(transfer_scores)
        raw_local = torch.cat(no_transfer_scores)
        fixed_local = torch.cat(fixed_scores)
        counts = torch.bincount(labels, minlength=int(seen_classes))
        if labels.numel() == 0 or torch.any(counts[:int(seen_classes)] == 0):
            raise RuntimeError("Session {} evaluation misses seen classes.".format(session))
        accuracy = float((predictions == labels).float().mean())
        macro = class_average_accuracy(labels, predictions, seen_classes)
        base_mask = labels < int(self.args.base_class)
        novel_mask = ~base_mask
        base_acc = float((predictions[base_mask] == labels[base_mask]).float().mean()) if base_mask.any() else None
        novel_acc = float((predictions[novel_mask] == labels[novel_mask]).float().mean()) if novel_mask.any() else None
        metrics = {
            "stage": "evaluation", "session": int(session),
            "sample_accuracy": accuracy, "class_average_accuracy": macro,
            "base_accuracy": base_acc, "novel_accuracy": novel_acc,
            "novel_to_base_error": float((predictions[novel_mask] < int(self.args.base_class)).float().mean()) if novel_mask.any() else None,
            "evaluation_samples": int(labels.numel()),
            "local_accuracy_after_transfer": float((transferred_local.argmax(1) == labels).float().mean()),
            "local_accuracy_before_transfer": float((raw_local.argmax(1) == labels).float().mean()),
            "local_transfer_gain": float(
                (transferred_local.argmax(1) == labels).float().mean()
                - (raw_local.argmax(1) == labels).float().mean()
            ),
            "query_ot_local_accuracy": float((transferred_local.argmax(1) == labels).float().mean()),
            "fixed_index_local_accuracy": float((fixed_local.argmax(1) == labels).float().mean()),
            "query_ot_vs_fixed_index_gain": float(
                (transferred_local.argmax(1) == labels).float().mean()
                - (fixed_local.argmax(1) == labels).float().mean()
            ),
        }
        return metrics

    @torch.no_grad()
    def _incremental(self, support_set, testset, session):
        base_class, way = int(self.args.base_class), int(self.args.way)
        expected = np.arange(base_class + (session - 1) * way,
                             base_class + session * way)
        support_labels = np.asarray(support_set.targets, dtype=np.int64)
        validate_exact_shot(support_labels, expected, int(self.args.shot), session)
        old_global, old_parts = self.all_global.clone(), self.all_parts.clone()
        global_features, parts, labels = self._extract(support_set)
        new_global, new_parts = [], []
        for cls in expected:
            mask = labels == int(cls)
            if not mask.any():
                raise RuntimeError("Support is missing class {}".format(cls))
            new_global.append(F.normalize(global_features[mask].mean(0), dim=-1))
            new_parts.append(ot_barycenter(
                parts[mask],
                float(getattr(self.args, "vsmr_ot_epsilon", 0.05)),
                int(getattr(self.args, "vsmr_sinkhorn_iterations", 30)),
                int(getattr(self.args, "vsmr_barycenter_iterations", 5)),
            ))
        new_global = torch.stack(new_global)
        raw_new_parts = torch.stack(new_parts)
        composed, alpha, reliability, selected = compose_novel_prototypes(
            raw_new_parts, self.base_bank,
            top_r=int(getattr(self.args, "vsmr_transfer_top_r", 5)),
            alpha_max=float(getattr(self.args, "vsmr_transfer_alpha_max", 0.5)),
            similarity_temperature=float(getattr(self.args, "vsmr_transfer_temperature", 0.1)),
            variance_scale=float(getattr(self.args, "vsmr_transfer_variance_scale", 0.25)),
        )
        self.all_global = torch.cat([self.all_global, new_global], dim=0)
        self.all_parts = torch.cat([self.all_parts, composed], dim=0)
        self.all_raw_parts = torch.cat([self.all_raw_parts, raw_new_parts], dim=0)
        if not torch.equal(self.all_global[:old_global.shape[0]], old_global):
            raise RuntimeError("Append-only contract violated: old global memory changed.")
        if not torch.equal(self.all_parts[:old_parts.shape[0]], old_parts):
            raise RuntimeError("Append-only contract violated: old local memory changed.")
        metrics = self._evaluate(testset, base_class + session * way, session)
        metrics.update({
            "transfer_alpha_mean": float(alpha.mean()),
            "transfer_reliability_mean": float(reliability.mean()),
            "transfer_selected_base_components": int(selected.numel()),
            "transfer_unique_base_components": int(torch.unique(selected).numel()),
        })
        self._append_log(metrics)

    def train(self):
        start = time.time()
        trainset, _, testloader = get_base_dataloader(self.args)
        self._train_base(trainset)
        base_metrics = self._evaluate(testloader.dataset, int(self.args.base_class), 0)
        self._append_log(base_metrics)
        session_macro = [base_metrics["class_average_accuracy"]]
        session_sample = [base_metrics["sample_accuracy"]]

        for session in range(1, int(self.args.sessions)):
            support, _, testloader = get_new_dataloader(self.args, session)
            self._incremental(support, testloader.dataset, session)
            record = self.results[-1]
            session_macro.append(record["class_average_accuracy"])
            session_sample.append(record["sample_accuracy"])

        summary = {
            "stage": "summary", "method": "VSMR",
            "AA": float(np.mean(session_macro)),
            "PD": float(session_macro[0] - session_macro[-1]),
            "session_class_average_accuracy": session_macro,
            "session_sample_accuracy": session_sample,
            "elapsed_minutes": (time.time() - start) / 60.0,
        }
        self._append_log(summary)
        torch.save({
            "model": self.model.state_dict(),
            "base_global": self.base_global.cpu(),
            "base_parts": self.base_parts.cpu(),
            "all_global": self.all_global.cpu(),
            "all_parts": self.all_parts.cpu(),
            "all_raw_parts": self.all_raw_parts.cpu(),
            "summary": summary,
            "results": self.results,
        }, os.path.join(self.save_dir, "vsmr_final.pth"))
        print("[VSMR] Results: {}".format(self.log_path))
