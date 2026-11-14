"""Dataset-backed FSCIL episodes with disjoint reference memory and queries."""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


ROLE_BASE_QUERY = 0
ROLE_NOVEL_SUPPORT = 1
ROLE_NOVEL_QUERY = 2


def split_reference_episode_indices(
    labels,
    reference_fraction=0.7,
    min_episode_samples=7,
    min_reference_samples=1,
    seed=1,
):
    """Make a deterministic, per-class disjoint memory/episode split.

    Reference images are reserved exclusively for pseudo-base memory.
    Episode images are reserved for pseudo-novel support and both base/novel
    queries. The split is exhaustive and never crosses class boundaries.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if not 0.0 < float(reference_fraction) < 1.0:
        raise ValueError("reference_fraction must be in (0,1).")
    if int(min_episode_samples) < 2 or int(min_reference_samples) < 1:
        raise ValueError("Episode/reference minimum sample counts are invalid.")

    rng = np.random.default_rng(int(seed))
    reference_indices = []
    episode_indices = []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        if indices.size < int(min_episode_samples) + int(min_reference_samples):
            raise ValueError(
                "Class {} has {} images; at least {} are required for "
                "disjoint reference and episode pools.".format(
                    int(cls), int(indices.size),
                    int(min_episode_samples) + int(min_reference_samples),
                )
            )
        indices = rng.permutation(indices)
        episode_count = max(
            int(min_episode_samples),
            int(round(indices.size * (1.0 - float(reference_fraction)))),
        )
        episode_count = min(
            episode_count, indices.size - int(min_reference_samples)
        )
        reference_indices.extend(indices[:-episode_count].tolist())
        episode_indices.extend(indices[-episode_count:].tolist())

    reference_indices = np.asarray(sorted(reference_indices), dtype=np.int64)
    episode_indices = np.asarray(sorted(episode_indices), dtype=np.int64)
    if np.intersect1d(reference_indices, episode_indices).size:
        raise RuntimeError("Reference and episode partitions overlap.")
    if reference_indices.size + episode_indices.size != labels.size:
        raise RuntimeError("Reference/episode split does not cover the dataset.")
    return reference_indices, episode_indices


class VSMREpisodeFeatureDataset(Dataset):
    """Read-only frozen-backbone feature cache with role-tagged indices."""

    def __init__(self, cls_features, part_features, patch_features, labels):
        if cls_features.ndim != 2 or part_features.ndim != 3 or patch_features.ndim != 3:
            raise ValueError("Cached CLS/part/Patch features must be [N,D], [N,K,D], [N,P,D].")
        if not (cls_features.shape[0] == part_features.shape[0] == patch_features.shape[0]):
            raise ValueError("Cached CLS, part and Patch features must share N.")
        if not (cls_features.shape[-1] == part_features.shape[-1] == patch_features.shape[-1]):
            raise ValueError("Cached CLS, part and Patch dimensions must match.")
        labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
        if labels.ndim != 1 or labels.numel() != cls_features.shape[0]:
            raise ValueError("Cached labels must have one entry per image.")
        self.cls_features = cls_features.detach().cpu().contiguous()
        self.part_features = part_features.detach().cpu().contiguous()
        self.patch_features = patch_features.detach().cpu().contiguous()
        self.labels = labels.contiguous()

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Episode indices must be (dataset_index, role) pairs.")
        index, role = int(item[0]), int(item[1])
        if index < 0 or index >= len(self):
            raise IndexError("Episode feature index is out of range.")
        if role not in (ROLE_BASE_QUERY, ROLE_NOVEL_SUPPORT, ROLE_NOVEL_QUERY):
            raise ValueError("Unknown VSMR episode sample role: {}".format(role))
        return (
            self.cls_features[index],
            self.part_features[index],
            self.patch_features[index],
            self.labels[index],
            torch.tensor(role, dtype=torch.long),
        )


class VSMREpisodeBatchSampler(Sampler):
    """Sample exact K-shot novel supports and disjoint base/novel queries.

    Every episode hides novel_way classes from the full reference memory.
    The candidate classifier retains every other base class and appends the
    pseudo-novel classes. Base queries are drawn from disjoint episode images
    belonging to pseudo-base classes.
    """

    def __init__(
        self,
        labels,
        episode_indices,
        novel_way,
        base_query_way=16,
        novel_shot=5,
        query_per_class=2,
        base_query_per_class=2,
        iterations=1,
        seed=1,
    ):
        labels = np.asarray(labels, dtype=np.int64)
        episode_indices = np.asarray(episode_indices, dtype=np.int64)
        if labels.ndim != 1 or episode_indices.ndim != 1:
            raise ValueError("labels and episode_indices must be one-dimensional.")
        if episode_indices.size == 0:
            raise ValueError("Episode pool cannot be empty.")
        if np.any(episode_indices < 0) or np.any(episode_indices >= labels.size):
            raise ValueError("Episode indices contain out-of-range entries.")
        if len(np.unique(episode_indices)) != episode_indices.size:
            raise ValueError("Episode indices must be unique.")

        self.class_to_indices = {
            int(cls): episode_indices[labels[episode_indices] == cls].tolist()
            for cls in np.unique(labels)
        }
        self.classes = np.asarray(sorted(self.class_to_indices), dtype=np.int64)
        self.novel_way = int(novel_way)
        self.base_query_way = int(base_query_way)
        self.novel_shot = int(novel_shot)
        self.query_per_class = int(query_per_class)
        self.base_query_per_class = int(base_query_per_class)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.epoch = 0

        if self.novel_way < 1 or self.base_query_way < 1:
            raise ValueError("An episode needs novel classes and base-query classes.")
        if self.novel_way + self.base_query_way > self.classes.size:
            raise ValueError("Not enough classes for the requested episode composition.")
        if self.novel_shot < 1 or self.query_per_class < 1:
            raise ValueError("Novel support/query counts must be positive.")
        if self.base_query_per_class < 1 or self.iterations < 1:
            raise ValueError("Base query count and episode iterations must be positive.")
        for cls, indices in self.class_to_indices.items():
            required = max(
                self.novel_shot + self.query_per_class,
                self.base_query_per_class,
            )
            if len(indices) < required:
                raise ValueError(
                    "Class {} has {} episode images; {} are required.".format(
                        cls, len(indices), required
                    )
                )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.iterations

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.iterations):
            novel_classes = rng.choice(
                self.classes, self.novel_way, replace=False
            )
            novel_set = set(int(x) for x in novel_classes.tolist())
            base_candidates = np.asarray(
                [int(x) for x in self.classes if int(x) not in novel_set],
                dtype=np.int64,
            )
            base_query_classes = rng.choice(
                base_candidates, self.base_query_way, replace=False
            )

            tagged_indices = []
            for cls in novel_classes:
                selected = rng.choice(
                    self.class_to_indices[int(cls)],
                    self.novel_shot + self.query_per_class,
                    replace=False,
                )
                support = selected[:self.novel_shot]
                query = selected[self.novel_shot:]
                tagged_indices.extend(
                    (int(index), ROLE_NOVEL_SUPPORT) for index in support
                )
                tagged_indices.extend(
                    (int(index), ROLE_NOVEL_QUERY) for index in query
                )

            for cls in base_query_classes:
                selected = rng.choice(
                    self.class_to_indices[int(cls)],
                    self.base_query_per_class,
                    replace=False,
                )
                tagged_indices.extend(
                    (int(index), ROLE_BASE_QUERY) for index in selected
                )

            rng.shuffle(tagged_indices)
            yield tagged_indices
