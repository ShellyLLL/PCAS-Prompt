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
from .vsmr_episodes import (
    ROLE_BASE_QUERY, ROLE_NOVEL_SUPPORT, ROLE_NOVEL_QUERY,
    VSMREpisodeBatchSampler, VSMREpisodeFeatureDataset,
    split_reference_episode_indices,
)
from .vsmr_losses import (
    attention_overlap_loss, component_diversity_loss,
    cross_image_semantic_alignment, global_logits,
    leave_one_out_global, ot_barycenter, prompt_orthogonality_loss,
)
from .vsmr_resynthesis import (
    build_base_component_bank,
    component_support_variance,
    compose_novel_prototypes,
    local_leave_one_out_query_logits,
    query_conditioned_fixed_index_reconstruction,
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
            "{}_seed{}".format(time.strftime("%Y%m%d_%H%M%S"), self.seed)
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, "metrics.jsonl")
        with open(os.path.join(self.save_dir, "config.json"), "w") as handle:
            json.dump(vars(self.args), handle, ensure_ascii=False, indent=2, default=str)
        print("[VSMR] Output directory: {}".format(self.save_dir))
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

    def _input_interpolation(self):
        names = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }
        interpolation = names.get(
            str(self.model.backbone.input_interpolation).lower()
        )
        if interpolation is None:
            raise ValueError(
                "Unsupported ViT checkpoint interpolation: {}".format(
                    self.model.backbone.input_interpolation
                )
            )
        return interpolation

    def _transform(self):
        # Deterministic checkpoint-aligned preprocessing for memory building,
        # incremental support extraction, and all evaluation sessions.
        backbone = self.model.backbone
        resize_size = int(round(224.0 / float(backbone.input_crop_pct)))
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=self._input_interpolation()),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(backbone.input_mean, backbone.input_std),
        ])

    def _training_transform(self):
        # Preserve image augmentation in the base session, but never use
        # dataset-specific normalization that conflicts with frozen ViT weights.
        backbone = self.model.backbone
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224, interpolation=self._input_interpolation()
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(backbone.input_mean, backbone.input_std),
        ])

    def _loader(self, dataset, batch_size, batch_sampler=None):
        workers = int(getattr(self.args, "num_workers", 4))
        common = {
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": workers > 0,
        }
        wrapped = ImageLabelDataset(dataset)
        if batch_sampler is not None:
            # PyTorch forbids specifying batch_size, shuffle, sampler or
            # drop_last together with batch_sampler.
            return DataLoader(wrapped, batch_sampler=batch_sampler, **common)
        return DataLoader(
            wrapped, batch_size=int(batch_size), shuffle=False, **common
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


    def _pseudo_incremental_episode(
        self, episode_batch, reference_global, reference_parts,
        epsilon, sinkhorn_steps, logit_scale,
    ):
        """Train transfer/reconstruction with full-class, disjoint FSCIL episodes.

        The reference memory is extracted from a per-class image partition
        disjoint from every episode support/query. Each episode hides a set of
        pseudo-novel classes from the full base reference memory, composes
        their exact K-shot supports, and jointly classifies pseudo-base and
        pseudo-novel queries against the complete candidate class set.
        """
        cls_features, patch_features, labels, roles = episode_batch
        cls_features = cls_features.to(self.device, non_blocking=True)
        patch_features = patch_features.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True).long()
        roles = roles.to(self.device, non_blocking=True).long()

        support_mask = roles == ROLE_NOVEL_SUPPORT
        novel_query_mask = roles == ROLE_NOVEL_QUERY
        base_query_mask = roles == ROLE_BASE_QUERY
        if not support_mask.any() or not novel_query_mask.any() or not base_query_mask.any():
            raise RuntimeError("Episode must contain novel supports and both query roles.")

        novel_classes = torch.unique(labels[support_mask], sorted=True)
        if not torch.equal(
            torch.unique(labels[novel_query_mask], sorted=True), novel_classes
        ):
            raise RuntimeError("Every pseudo-novel class must have support and query samples.")
        if torch.isin(labels[base_query_mask], novel_classes).any():
            raise RuntimeError("Pseudo-base query labels overlap pseudo-novel classes.")

        all_base_classes = torch.arange(
            int(self.args.base_class), device=self.device, dtype=torch.long
        )
        keep_base = ~torch.isin(all_base_classes, novel_classes)
        base_classes = all_base_classes[keep_base]
        if base_classes.numel() < 1:
            raise RuntimeError("Episode removed every pseudo-base class.")

        # Reference memory was built from the disjoint reference image split.
        base_global = reference_global[base_classes]
        base_parts = reference_parts[base_classes]
        base_bank, base_owner, _ = build_base_component_bank(base_parts)

        novel_global = []
        novel_raw_parts = []
        novel_variance = []
        for cls in novel_classes:
            selected = support_mask & labels.eq(cls)
            support_global = cls_features[selected]
            support_parts = episode_batch[1][selected.cpu()].to(
                self.device, non_blocking=True
            )
            if support_global.shape[0] != int(getattr(self.args, "vsmr_episode_novel_shot", 5)):
                raise RuntimeError("Pseudo-novel support count does not match exact K-shot.")
            local_prototype = ot_barycenter(
                support_parts, epsilon, sinkhorn_steps,
                int(getattr(self.args, "vsmr_barycenter_iterations", 5)),
            )
            novel_global.append(F.normalize(support_global.mean(0), dim=-1))
            novel_raw_parts.append(local_prototype)
            novel_variance.append(component_support_variance(
                support_parts, local_prototype, epsilon, sinkhorn_steps
            ))

        novel_global = torch.stack(novel_global)
        novel_raw_parts = torch.stack(novel_raw_parts)
        novel_variance = torch.stack(novel_variance)
        shots = int(getattr(self.args, "vsmr_episode_novel_shot", 5))
        novel_parts, alpha, reliability, selected, transfer_weights = compose_novel_prototypes(
            novel_raw_parts,
            base_bank,
            self.model.reliability_gate,
            base_owner=base_owner,
            top_r=int(getattr(self.args, "vsmr_transfer_top_r", 5)),
            alpha_max=float(getattr(self.args, "vsmr_transfer_alpha_max", 0.5)),
            similarity_temperature=float(getattr(self.args, "vsmr_transfer_temperature", 0.1)),
            support_variance=novel_variance,
            shots=shots,
        )

        # Candidate order is all retained base IDs in ascending order,
        # followed by the sorted pseudo-novel IDs.
        candidate_global = torch.cat([base_global, novel_global], dim=0)
        candidate_parts = torch.cat([base_parts, novel_parts], dim=0)
        base_target_map = torch.full(
            (int(self.args.base_class),), -1, dtype=torch.long, device=self.device
        )
        base_target_map[base_classes] = torch.arange(
            base_classes.numel(), device=self.device
        )
        novel_target_map = {
            int(cls): int(base_classes.numel()) + position
            for position, cls in enumerate(novel_classes.tolist())
        }

        query_mask = base_query_mask | novel_query_mask
        query_features = cls_features[query_mask]
        query_parts = episode_batch[1][query_mask.cpu()].to(
            self.device, non_blocking=True
        )
        query_patches = patch_features[query_mask]
        query_labels = labels[query_mask]
        targets = []
        for cls in query_labels.tolist():
            if cls in novel_target_map:
                targets.append(novel_target_map[cls])
            else:
                target = int(base_target_map[cls].item())
                if target < 0:
                    raise RuntimeError("Query class is absent from the candidate memory.")
                targets.append(target)
        targets = torch.tensor(targets, dtype=torch.long, device=self.device)

        global_scores = global_logits(
            query_features, candidate_global, logit_scale
        )
        local_scores, _, _, _ = query_conditioned_reconstruction(
            query_parts, query_patches, candidate_parts,
            self.model.reconstructor,
            epsilon=epsilon, iterations=sinkhorn_steps, scale=logit_scale,
        )
        joint_scores = self.model.fusion(global_scores, local_scores)

        base_query_count = int(base_query_mask.sum().item())
        novel_query_count = int(novel_query_mask.sum().item())
        if base_query_count == 0 or novel_query_count == 0:
            raise RuntimeError("Episode requires both base and novel queries.")
        return (
            F.cross_entropy(global_scores, targets),
            F.cross_entropy(local_scores, targets),
            F.cross_entropy(joint_scores, targets),
            {
                "episode_candidate_classes": int(candidate_global.shape[0]),
                "episode_base_candidate_classes": int(base_classes.numel()),
                "episode_novel_classes": int(novel_classes.numel()),
                "episode_base_queries": base_query_count,
                "episode_novel_queries": novel_query_count,
                "episode_support_shots": shots,
                "episode_transfer_alpha": float(alpha.detach().mean()),
                "episode_gate_reliability": float(reliability.detach().mean()),
            },
        )

    @torch.no_grad()
    def _build_episode_reference(
        self, trainset, reference_indices, episode_indices,
    ):
        """Recompute full base-class features and create disjoint episode cache."""
        original = getattr(trainset, "transform", None)
        if original is None:
            raise RuntimeError("Episode reference dataset must expose a transform.")
        trainset.transform = self._transform()
        loader = self._loader(
            trainset, int(getattr(self.args, "vsmr_eval_batch_size", 8))
        )
        globals_, parts_, patches_, labels_ = [], [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            for images, batch_labels in loader:
                output = self.model(images.to(self.device, non_blocking=True))
                globals_.append(output["global_features"])
                parts_.append(output["parts"])
                patches_.append(output["patch_features"])
                labels_.append(torch.as_tensor(
                    batch_labels, device=self.device
                ).long())
        finally:
            trainset.transform = original
            if was_training:
                self.model.train()

        features = torch.cat(globals_)
        parts = torch.cat(parts_)
        patches = torch.cat(patches_)
        extracted_labels = torch.cat(labels_)
        expected = torch.arange(
            int(self.args.base_class), device=self.device, dtype=torch.long
        )
        if not torch.equal(torch.unique(extracted_labels, sorted=True), expected):
            raise RuntimeError("Episode feature cache must contain every base class.")
        reference_global = []
        reference_parts = []
        reference_labels = extracted_labels[
            torch.as_tensor(reference_indices, device=self.device, dtype=torch.long)
        ]
        reference_features = features[
            torch.as_tensor(reference_indices, device=self.device, dtype=torch.long)
        ]
        reference_components = parts[
            torch.as_tensor(reference_indices, device=self.device, dtype=torch.long)
        ]
        for cls in expected:
            selected = reference_labels.eq(cls)
            if not selected.any():
                raise RuntimeError("Reference memory missing base class {}.".format(int(cls)))
            reference_global.append(F.normalize(
                reference_features[selected].mean(0), dim=-1
            ))
            reference_parts.append(ot_barycenter(
                reference_components[selected],
                float(getattr(self.args, "vsmr_ot_epsilon", 0.05)),
                int(getattr(self.args, "vsmr_sinkhorn_iterations", 30)),
                int(getattr(self.args, "vsmr_barycenter_iterations", 5)),
            ))

        episode_indices_tensor = torch.as_tensor(
            episode_indices, device=self.device, dtype=torch.long
        )
        episode_dataset = VSMREpisodeFeatureDataset(
            features[episode_indices_tensor],
            patches[episode_indices_tensor],
            extracted_labels[episode_indices_tensor],
        )
        return (
            torch.stack(reference_global).detach(),
            torch.stack(reference_parts).detach(),
            episode_dataset,
        )

    def _train_base(self, trainset):
        # Replace legacy dataset transforms before creating the DataLoader:
        # training augmentation is retained, while normalization/interpolation
        # are now tied to the frozen ViT checkpoint.
        trainset.transform = self._training_transform()
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
        reference_indices, episode_indices = split_reference_episode_indices(
            labels,
            reference_fraction=float(getattr(
                self.args, "vsmr_episode_reference_fraction", 0.7
            )),
            min_episode_samples=(
                int(getattr(self.args, "vsmr_episode_novel_shot", 5))
                + int(getattr(self.args, "vsmr_episode_query_per_class", 2))
            ),
            min_reference_samples=1,
            seed=self.seed,
        )
        episode_novel_way = int(getattr(
            self.args, "vsmr_episode_novel_way",
            min(10, int(self.args.base_class) - 1)
        ))
        episode_base_query_way = int(getattr(
            self.args, "vsmr_episode_base_query_way", 16
        ))
        episode_sampler = VSMREpisodeBatchSampler(
            labels,
            episode_indices,
            novel_way=episode_novel_way,
            base_query_way=episode_base_query_way,
            novel_shot=int(getattr(self.args, "vsmr_episode_novel_shot", 5)),
            query_per_class=int(getattr(self.args, "vsmr_episode_query_per_class", 2)),
            base_query_per_class=int(getattr(
                self.args, "vsmr_episode_base_query_per_class", 2
            )),
            iterations=len(sampler),
            seed=self.seed + 100003,
        )
        loader = self._loader(trainset, 1, batch_sampler=sampler)
        trainable_parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("VSMR has no trainable base-stage parameters.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(getattr(self.args, "vsmr_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "vsmr_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        eps = float(getattr(self.args, "vsmr_ot_epsilon", 0.05))
        sinkhorn_steps = int(getattr(self.args, "vsmr_sinkhorn_iterations", 30))
        w_local = float(getattr(self.args, "vsmr_local_weight", 1.0))
        w_joint = float(getattr(self.args, "vsmr_joint_weight", 1.0))
        w_align = float(getattr(self.args, "vsmr_alignment_weight", 0.2))
        w_cisa = float(getattr(self.args, "vsmr_cisa_weight", 0.1))
        w_overlap = float(getattr(self.args, "vsmr_overlap_weight", 0.01))
        w_prompt_orth = float(getattr(self.args, "vsmr_prompt_orth_weight", 0.2))
        w_component_div = float(getattr(
            self.args, "vsmr_component_diversity_weight", 0.1
        ))
        scale = float(getattr(self.args, "vsmr_logit_scale", 16.0))

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            episode_sampler.set_epoch(epoch)
            reference_global, reference_parts, episode_dataset = (
                self._build_episode_reference(
                    trainset, reference_indices, episode_indices
                )
            )
            episode_loader = DataLoader(
                episode_dataset,
                batch_sampler=episode_sampler,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
            episode_iterator = iter(episode_loader)
            self.model.train()
            totals = {k: 0.0 for k in (
                "total", "global", "local", "joint", "alignment", "cisa", "overlap",
                "prompt_orth", "component_diversity",
                "query_cosine_mean", "query_cosine_max",
                "part_cosine_mean", "part_cosine_max",
                "assignment_row_error", "assignment_col_error", "attention_entropy",
                "episode_global", "episode_local", "episode_joint",
                "episode_candidate_classes", "episode_base_candidate_classes",
                "episode_novel_classes", "episode_base_queries",
                "episode_novel_queries", "episode_support_shots",
                "episode_transfer_alpha", "episode_gate_reliability",
                "global_acc", "local_acc", "joint_acc", "ot_same"
            )}
            steps = 0
            for images, batch_labels in tqdm(
                loader, desc="VSMR base {}/{}".format(epoch + 1, epochs)
            ):
                images = images.to(self.device, non_blocking=True)
                labels_batch = torch.as_tensor(batch_labels, device=self.device).long()
                output = self.model(images)
                features, parts, attention, patch_features, prompt_queries, assignment = (
                    output["global_features"], output["parts"],
                    output["attention"], output["patch_features"],
                    output["prompt_queries"], output["assignment"],
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
                l_logits, local_classes = local_leave_one_out_query_logits(
                    parts,
                    patch_features,
                    labels_batch,
                    self.model.reconstructor,
                    eps,
                    sinkhorn_steps,
                    scale,
                    int(getattr(self.args, "vsmr_barycenter_iterations", 5)),
                )
                if not torch.equal(classes, local_classes):
                    raise RuntimeError("Global/local class order mismatch.")
                # Same-class OT alignment, using all distinct image pairs.
                from .vsmr_losses import pairwise_ot_distances
                pair_dist = pairwise_ot_distances(parts, eps, sinkhorn_steps)
                same = labels_batch[:, None].eq(labels_batch[None, :])
                same.fill_diagonal_(False)
                align_loss = pair_dist[same].mean()
                base_global_loss = F.cross_entropy(g_logits, inverse)
                base_local_loss = F.cross_entropy(l_logits, inverse)
                base_joint_logits = self.model.fusion(g_logits, l_logits)
                base_joint_loss = F.cross_entropy(base_joint_logits, inverse)

                try:
                    episode_batch = next(episode_iterator)
                except StopIteration:
                    episode_iterator = iter(episode_loader)
                    episode_batch = next(episode_iterator)
                (
                    episode_global_loss, episode_local_loss,
                    episode_joint_loss, episode_metrics,
                ) = self._pseudo_incremental_episode(
                    episode_batch, reference_global, reference_parts,
                    eps, sinkhorn_steps, scale,
                )
                global_loss = 0.5 * (base_global_loss + episode_global_loss)
                local_loss = 0.5 * (base_local_loss + episode_local_loss)
                joint_loss = 0.5 * (base_joint_loss + episode_joint_loss)
                cisa_loss = cross_image_semantic_alignment(
                    g_logits, l_logits, labels_batch
                )
                overlap_loss = attention_overlap_loss(attention)
                prompt_orth_loss = prompt_orthogonality_loss(prompt_queries)
                component_div_loss = component_diversity_loss(parts)
                total = (
                    global_loss + w_local * local_loss + w_joint * joint_loss
                    + w_align * align_loss + w_cisa * cisa_loss
                    + w_overlap * overlap_loss
                    + w_prompt_orth * prompt_orth_loss
                    + w_component_div * component_div_loss
                )
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite VSMR loss.")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Non-finite gradient: " + name)
                optimizer.step()

                with torch.no_grad():
                    totals["total"] += float(total)
                    totals["global"] += float(global_loss)
                    totals["local"] += float(local_loss)
                    totals["joint"] += float(joint_loss)
                    totals["episode_global"] += float(episode_global_loss)
                    totals["episode_local"] += float(episode_local_loss)
                    totals["episode_joint"] += float(episode_joint_loss)
                    for metric_name, metric_value in episode_metrics.items():
                        totals[metric_name] += float(metric_value)
                    totals["alignment"] += float(align_loss)
                    totals["cisa"] += float(cisa_loss)
                    totals["overlap"] += float(overlap_loss)
                    totals["prompt_orth"] += float(prompt_orth_loss)
                    totals["component_diversity"] += float(component_div_loss)
                    query_gram = F.normalize(prompt_queries, dim=-1) @ F.normalize(
                        prompt_queries, dim=-1
                    ).transpose(0, 1)
                    query_mask = ~torch.eye(
                        query_gram.shape[0], dtype=torch.bool, device=query_gram.device
                    )
                    part_norm = F.normalize(parts, dim=-1)
                    part_gram = torch.matmul(part_norm, part_norm.transpose(1, 2))
                    part_mask = ~torch.eye(
                        part_gram.shape[1], dtype=torch.bool, device=part_gram.device
                    )
                    totals["query_cosine_mean"] += float(query_gram[query_mask].mean())
                    totals["query_cosine_max"] += float(query_gram[query_mask].max())
                    totals["part_cosine_mean"] += float(part_gram[:, part_mask].mean())
                    totals["part_cosine_max"] += float(part_gram[:, part_mask].max())
                    expected_row = 1.0 / float(assignment.shape[1])
                    expected_col = 1.0 / float(assignment.shape[2])
                    totals["assignment_row_error"] += float(
                        (assignment.sum(-1) - expected_row).abs().max()
                    )
                    totals["assignment_col_error"] += float(
                        (assignment.sum(-2) - expected_col).abs().max()
                    )
                    totals["attention_entropy"] += float(
                        -(attention.clamp_min(1e-8)
                          * attention.clamp_min(1e-8).log()).sum(-1).mean()
                    )
                    totals["global_acc"] += float((g_logits.argmax(1) == inverse).float().mean())
                    totals["local_acc"] += float((l_logits.argmax(1) == inverse).float().mean())
                    totals["joint_acc"] += float((base_joint_logits.argmax(1) == inverse).float().mean())
                    totals["ot_same"] += float(pair_dist[same].mean())
                steps += 1
            scheduler.step()
            record = {"stage": "base_train", "epoch": epoch + 1}
            record.update({k: v / max(steps, 1) for k, v in totals.items()})
            record["fusion_alpha"] = float(self.model.fusion.alpha.detach().item())
            record["lr"] = scheduler.get_last_lr()[0]
            self._append_log(record)

        self.model.freeze_for_incremental()
        self._build_base_memory(trainset)
        torch.save({
            "model": self.model.state_dict(),
            "base_global": self.base_global.cpu(),
            "base_parts": self.base_parts.cpu(),
            "base_component_bank": self.base_bank.cpu(),
            "base_component_owner": self.base_owner.cpu(),
            "base_component_part_ids": self.base_part_ids.cpu(),
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
        normalized_parts = F.normalize(self.base_parts, dim=-1)
        component_gram = torch.matmul(
            normalized_parts, normalized_parts.transpose(1, 2)
        )
        component_mask = ~torch.eye(
            component_gram.shape[1], dtype=torch.bool,
            device=component_gram.device
        )
        within_class_similarity = component_gram[:, component_mask]
        self._append_log({
            "stage": "base_memory",
            "base_classes": int(self.base_global.shape[0]),
            "parts_per_class": int(self.base_parts.shape[1]),
            "component_bank_size": int(self.base_bank.shape[0]),
            "within_class_component_cosine_mean": float(
                within_class_similarity.mean()
            ),
            "within_class_component_cosine_max": float(
                within_class_similarity.max()
            ),
            "component_bank_finite": bool(torch.isfinite(self.base_bank).all()),
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
        pred, target = [], []
        raw_joint_predictions, fixed_joint_predictions = [], []
        transfer_scores, no_transfer_scores, fixed_scores, ot_entropies = [], [], [], []
        self.model.eval()
        try:
            for images, labels in tqdm(loader, desc="VSMR eval session {}".format(session)):
                images = images.to(self.device, non_blocking=True)
                labels = torch.as_tensor(labels, device=self.device).long()
                output = self.model(images)
                gf, qp, query_patches = (
                    output["global_features"], output["parts"],
                    output["patch_features"]
                )
                gl = global_logits(gf, self.all_global, scale)
                local_score, _, transport, _ = query_conditioned_reconstruction(
                    qp,
                    query_patches,
                    self.all_parts,
                    self.model.reconstructor,
                    eps,
                    steps,
                    scale,
                )
                transport_entropy = -(
                    transport.clamp_min(1e-8) * transport.clamp_min(1e-8).log()
                ).sum(dim=(-1, -2)).mean(dim=-1)
                ot_entropies.append(transport_entropy.cpu())
                logits = self.model.fusion(gl, local_score)
                pred.append(logits.argmax(1).cpu())
                target.append(labels.cpu())

                # Controlled ablations use the same query, global branch and
                # classifier weights; only transfer or matching is removed.
                fixed, _, _ = query_conditioned_fixed_index_reconstruction(
                    qp,
                    query_patches,
                    self.all_parts,
                    self.model.reconstructor,
                    scale,
                )
                fixed_scores.append(fixed.cpu())
                transfer_scores.append(local_score.cpu())
                no_transfer = query_conditioned_reconstruction(
                    qp,
                    query_patches,
                    self.all_raw_parts,
                    self.model.reconstructor,
                    eps,
                    steps,
                    scale,
                )[0]
                no_transfer_scores.append(no_transfer.cpu())
                raw_joint = self.model.fusion(gl, no_transfer)
                fixed_joint = self.model.fusion(gl, fixed)
                raw_joint_predictions.append(raw_joint.argmax(1).cpu())
                fixed_joint_predictions.append(fixed_joint.argmax(1).cpu())
        finally:
            testset.transform = original
        predictions, labels = torch.cat(pred), torch.cat(target)
        transferred_local = torch.cat(transfer_scores)
        raw_local = torch.cat(no_transfer_scores)
        fixed_local = torch.cat(fixed_scores)
        raw_joint_pred = torch.cat(raw_joint_predictions)
        fixed_joint_pred = torch.cat(fixed_joint_predictions)
        counts = torch.bincount(labels, minlength=int(seen_classes))
        if labels.numel() == 0 or torch.any(counts[:int(seen_classes)] == 0):
            raise RuntimeError("Session {} evaluation misses seen classes.".format(session))
        accuracy = float((predictions == labels).float().mean())
        macro = class_average_accuracy(labels, predictions, seen_classes)
        raw_joint_accuracy = float((raw_joint_pred == labels).float().mean())
        raw_joint_macro = class_average_accuracy(labels, raw_joint_pred, seen_classes)
        fixed_joint_accuracy = float((fixed_joint_pred == labels).float().mean())
        base_mask = labels < int(self.args.base_class)
        novel_mask = ~base_mask
        base_acc = float((predictions[base_mask] == labels[base_mask]).float().mean()) if base_mask.any() else None
        novel_acc = float((predictions[novel_mask] == labels[novel_mask]).float().mean()) if novel_mask.any() else None
        local_after = transferred_local.argmax(1) == labels
        local_before = raw_local.argmax(1) == labels
        local_fixed = fixed_local.argmax(1) == labels
        joint_after = predictions == labels
        joint_before = raw_joint_pred == labels
        joint_fixed = fixed_joint_pred == labels
        novel_local_after = float(local_after[novel_mask].float().mean()) if novel_mask.any() else None
        novel_local_before = float(local_before[novel_mask].float().mean()) if novel_mask.any() else None
        novel_joint_after = float(joint_after[novel_mask].float().mean()) if novel_mask.any() else None
        novel_joint_before = float(joint_before[novel_mask].float().mean()) if novel_mask.any() else None
        novel_query_ot = float(joint_after[novel_mask].float().mean()) if novel_mask.any() else None
        novel_fixed = float(joint_fixed[novel_mask].float().mean()) if novel_mask.any() else None
        metrics = {
            "stage": "evaluation", "session": int(session),
            "sample_accuracy": accuracy, "class_average_accuracy": macro,
            "base_accuracy": base_acc, "novel_accuracy": novel_acc,
            "novel_to_base_error": float((predictions[novel_mask] < int(self.args.base_class)).float().mean()) if novel_mask.any() else None,
            "evaluation_samples": int(labels.numel()),
            "joint_accuracy_before_transfer": raw_joint_accuracy,
            "class_average_accuracy_before_transfer": raw_joint_macro,
            "joint_transfer_gain": accuracy - raw_joint_accuracy,
            "novel_joint_accuracy_before_transfer": novel_joint_before,
            "novel_joint_accuracy_after_transfer": novel_joint_after,
            "novel_joint_transfer_gain": (
                novel_joint_after - novel_joint_before
                if novel_mask.any() else None
            ),
            "joint_accuracy_fixed_index": fixed_joint_accuracy,
            "query_ot_joint_gain_over_fixed_index": accuracy - fixed_joint_accuracy,
            "novel_joint_accuracy_fixed_index": novel_fixed,
            "novel_query_ot_gain_over_fixed_index": (
                novel_query_ot - novel_fixed if novel_mask.any() else None
            ),
            "fixed_index_uses_same_query_reconstruction": True,
            "local_accuracy_after_transfer": float(local_after.float().mean()),
            "local_accuracy_before_transfer": float(local_before.float().mean()),
            "local_transfer_gain": float(local_after.float().mean() - local_before.float().mean()),
            "novel_local_accuracy_before_transfer": novel_local_before,
            "novel_local_accuracy_after_transfer": novel_local_after,
            "novel_local_transfer_gain": (
                novel_local_after - novel_local_before if novel_mask.any() else None
            ),
            "query_ot_local_accuracy": float(local_after.float().mean()),
            "fixed_index_local_accuracy": float(local_fixed.float().mean()),
            "query_ot_vs_fixed_index_gain": float(
                local_after.float().mean() - local_fixed.float().mean()
            ),
            "query_ot_assignment_entropy": float(torch.cat(ot_entropies).mean()),
            "global_local_fusion_alpha": float(self.model.fusion.alpha.detach().item()),
        }
        return metrics

    def _audit_support_test_paths(self, support_set, testset, session):
        if not hasattr(support_set, "data") or not hasattr(testset, "data"):
            return
        support_items = list(support_set.data)
        test_items = list(testset.data)
        if not support_items or not test_items:
            return
        path_types = (str, bytes, os.PathLike)
        if not all(isinstance(path, path_types) for path in support_items + test_items):
            return
        support_paths = [
            os.path.realpath(os.path.expanduser(os.fsdecode(path)))
            for path in support_items
        ]
        test_paths = [
            os.path.realpath(os.path.expanduser(os.fsdecode(path)))
            for path in test_items
        ]
        if len(set(support_paths)) != len(support_paths):
            raise RuntimeError("Session {} has duplicate support image paths.".format(session))
        overlap = set(support_paths).intersection(test_paths)
        if overlap:
            raise RuntimeError(
                "Session {} support/test leakage: {} shared image paths.".format(
                    session, len(overlap)
                )
            )

    @torch.no_grad()
    def _incremental(self, support_set, testset, session):
        base_class, way = int(self.args.base_class), int(self.args.way)
        expected = np.arange(base_class + (session - 1) * way,
                             base_class + session * way)
        support_labels = np.asarray(support_set.targets, dtype=np.int64)
        validate_exact_shot(support_labels, expected, int(self.args.shot), session)
        self._audit_support_test_paths(support_set, testset, session)
        old_global = self.all_global.clone()
        old_parts = self.all_parts.clone()
        old_raw_parts = self.all_raw_parts.clone()
        old_base_bank = self.base_bank.clone()
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
        support_variance = torch.stack([
            component_support_variance(
                parts[labels == int(cls)],
                raw_new_parts[position],
                float(getattr(self.args, "vsmr_ot_epsilon", 0.05)),
                int(getattr(self.args, "vsmr_sinkhorn_iterations", 30)),
            )
            for position, cls in enumerate(expected)
        ])
        composed, alpha, reliability, selected, transfer_weights = compose_novel_prototypes(
            raw_new_parts,
            self.base_bank,
            self.model.reliability_gate,
            base_owner=self.base_owner,
            top_r=int(getattr(self.args, "vsmr_transfer_top_r", 5)),
            alpha_max=float(getattr(self.args, "vsmr_transfer_alpha_max", 0.5)),
            similarity_temperature=float(getattr(self.args, "vsmr_transfer_temperature", 0.1)),
            support_variance=support_variance,
            shots=int(self.args.shot),
        )
        self.all_global = torch.cat([self.all_global, new_global], dim=0)
        self.all_parts = torch.cat([self.all_parts, composed], dim=0)
        self.all_raw_parts = torch.cat([self.all_raw_parts, raw_new_parts], dim=0)
        if not torch.equal(self.all_global[:old_global.shape[0]], old_global):
            raise RuntimeError("Append-only contract violated: old global memory changed.")
        if not torch.equal(self.all_parts[:old_parts.shape[0]], old_parts):
            raise RuntimeError("Append-only contract violated: old local memory changed.")
        if not torch.equal(self.all_raw_parts[:old_raw_parts.shape[0]], old_raw_parts):
            raise RuntimeError("Append-only contract violated: old raw local memory changed.")
        if not torch.equal(self.base_bank, old_base_bank):
            raise RuntimeError("Append-only contract violated: base component bank changed.")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Incremental protocol violation: model parameters are trainable.")
        metrics = self._evaluate(testset, base_class + session * way, session)
        metrics.update({
            "transfer_alpha_mean": float(alpha.mean()),
            "transfer_reliability_mean": float(reliability.mean()),
            "transfer_selected_base_components": int(selected.numel()),
            "transfer_unique_base_components": int(torch.unique(selected).numel()),
            "transfer_selected_component_ids": selected.cpu().tolist(),
            "transfer_selected_weights": transfer_weights.cpu().tolist(),
            "transfer_selected_base_class_ids": self.base_owner[selected].cpu().tolist(),
            "transfer_selected_base_part_ids": self.base_part_ids[selected].cpu().tolist(),
            "transfer_alpha_per_class": alpha.mean(dim=1).cpu().tolist(),
            "transfer_reliability_per_class": reliability.mean(dim=1).cpu().tolist(),
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
            "base_component_bank": self.base_bank.cpu(),
            "base_component_owner": self.base_owner.cpu(),
            "base_component_part_ids": self.base_part_ids.cpu(),
            "all_global": self.all_global.cpu(),
            "all_parts": self.all_parts.cpu(),
            "all_raw_parts": self.all_raw_parts.cpu(),
            "summary": summary,
            "results": self.results,
        }, os.path.join(self.save_dir, "vsmr_final.pth"))
        print("[VSMR] Results: {}".format(self.log_path))
