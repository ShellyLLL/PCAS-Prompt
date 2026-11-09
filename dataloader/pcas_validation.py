"""Leakage-free pseudo-FSCIL validation splits built only from official train data."""
import numpy as np


class IndexedClassSubset:
    """Dataset view with remapped labels and stable source sample IDs."""

    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_map = {int(k): int(v) for k, v in label_map.items()}
        source_targets = np.asarray(base_dataset.targets, dtype=np.int64)
        self.targets = np.asarray(
            [self.label_map[int(source_targets[i])] for i in self.indices],
            dtype=np.int64,
        )
        self.sample_ids = self.indices.copy()

    @property
    def transform(self):
        return self.base_dataset.transform

    @transform.setter
    def transform(self, value):
        self.base_dataset.transform = value

    @property
    def data(self):
        raw = getattr(self.base_dataset, "data", None)
        if isinstance(raw, (list, tuple)):
            return [raw[int(i)] for i in self.indices]
        return None

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        image, source_label = self.base_dataset[int(self.indices[index])]
        source_label = int(source_label)
        if source_label not in self.label_map:
            raise RuntimeError("Validation subset contains an unmapped source class.")
        return image, self.label_map[source_label]


def build_pseudo_fscil_validation(args, validation_seed=2026,
                                  base_query_fraction=0.2):
    """Create class-disjoint pseudo-base/novel tasks from official TRAIN split only.

    The official test partition and official FSCIL session index files are never
    opened. Base-class images are split into train/query; pseudo-novel images
    into K-shot support/query. All partitions are sample-disjoint.
    """
    from dataloader import data_utils

    if not 0.0 < float(base_query_fraction) < 1.0:
        raise ValueError("base_query_fraction must be in (0, 1).")

    all_classes = np.arange(int(args.num_classes), dtype=np.int64)
    if args.dataset == "cifar100":
        source = args.Dataset.CIFAR100(
            root=args.dataroot, train=True, download=False,
            index=all_classes, base_sess=True, is_vit=args.vit, is_clip=args.clip,
        )
    elif args.dataset == "cub200":
        source = args.Dataset.CUB200(
            root=args.dataroot, train=True, index=all_classes,
            base_sess=True, is_clip=args.clip,
        )
    elif args.dataset == "mini_imagenet":
        source = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True, index=all_classes,
            base_sess=True, is_clip=args.clip,
        )
    else:
        raise ValueError("Pseudo-FSCIL validation supports mini_imagenet, cub200 and cifar100.")

    targets = np.asarray(source.targets, dtype=np.int64)
    if not np.array_equal(np.unique(targets), all_classes):
        raise RuntimeError("Official training split does not contain every expected class.")

    base_count = int(args.base_class)
    novel_count = int(args.num_classes) - base_count
    way, shots = int(args.way), int(args.shot)
    if novel_count <= 0 or way <= 0 or novel_count % way:
        raise ValueError("Pseudo-novel classes must be positive and divisible by way.")
    expected_sessions = novel_count // way
    if int(args.sessions) != expected_sessions + 1:
        raise RuntimeError("Pseudo-FSCIL session count conflicts with dataset configuration.")

    rng = np.random.default_rng(int(validation_seed))
    class_order = rng.permutation(all_classes).astype(int).tolist()
    label_map = {int(src): int(dst) for dst, src in enumerate(class_order)}
    train_ids, base_query_ids = [], []
    novel_support_by_session = [[] for _ in range(expected_sessions)]
    novel_query_by_class = {}
    class_manifest = {}

    for pseudo_id, source_class in enumerate(class_order):
        ids = np.flatnonzero(targets == int(source_class))
        ids = rng.permutation(ids)
        if pseudo_id < base_count:
            n_query = max(1, int(round(len(ids) * float(base_query_fraction))))
            n_train = len(ids) - n_query
            if n_train < max(4, int(getattr(args, "samples_per_class", 4))):
                raise RuntimeError("Insufficient pseudo-base samples for disjoint train/query split.")
            train_ids.extend(ids[:n_train].tolist())
            base_query_ids.extend(ids[n_train:].tolist())
            class_manifest[str(pseudo_id)] = {
                "source_class": int(source_class), "role": "pseudo_base",
                "train": int(n_train), "query": int(n_query),
            }
        else:
            if len(ids) <= shots:
                raise RuntimeError("Pseudo-novel class has no query samples after K-shot support.")
            session_id = (pseudo_id - base_count) // way
            novel_support_by_session[session_id].extend(ids[:shots].tolist())
            novel_query_by_class[pseudo_id] = ids[shots:].tolist()
            class_manifest[str(pseudo_id)] = {
                "source_class": int(source_class), "role": "pseudo_novel",
                "support": shots, "query": int(len(ids) - shots),
                "session": session_id + 1,
            }

    trainset = IndexedClassSubset(source, train_ids, label_map)
    base_query = IndexedClassSubset(source, base_query_ids, label_map)
    sessions, cumulative_query_ids = [], list(base_query_ids)
    for session_id, support_ids in enumerate(novel_support_by_session, start=1):
        start = base_count + (session_id - 1) * way
        for pseudo_id in range(start, start + way):
            cumulative_query_ids.extend(novel_query_by_class[pseudo_id])
        sessions.append((
            IndexedClassSubset(source, support_ids, label_map),
            IndexedClassSubset(source, cumulative_query_ids, label_map),
        ))

    train_set = set(trainset.sample_ids.tolist())
    base_query_set = set(base_query.sample_ids.tolist())
    support_set = set(i for s, _ in sessions for i in s.sample_ids.tolist())
    novel_query_set = set(i for values in novel_query_by_class.values() for i in values)
    if train_set & base_query_set or train_set & support_set or train_set & novel_query_set:
        raise RuntimeError("Leakage detected: pseudo-base train overlaps a validation partition.")
    if support_set & novel_query_set or base_query_set & support_set or base_query_set & novel_query_set:
        raise RuntimeError("Leakage detected between pseudo-FSCIL support/query partitions.")
    if len(train_set) != len(train_ids) or len(base_query_set) != len(base_query_ids):
        raise RuntimeError("Duplicate sample indices detected in pseudo-base partitions.")
    if len(support_set) != sum(map(len, novel_support_by_session)):
        raise RuntimeError("Duplicate sample indices detected in pseudo-novel support partitions.")
    if len(novel_query_set) != sum(map(len, novel_query_by_class.values())):
        raise RuntimeError("Duplicate sample indices detected in pseudo-novel query partitions.")

    manifest = {
        "protocol": "pseudo_fscil_validation_from_official_train_only_v1",
        "dataset": str(args.dataset),
        "validation_seed": int(validation_seed),
        "base_query_fraction": float(base_query_fraction),
        "shots_per_class": shots,
        "base_class_count": base_count,
        "novel_class_count": novel_count,
        "way": way,
        "sessions": expected_sessions + 1,
        "source_class_order": [int(x) for x in class_order],
        "class_assignments": class_manifest,
        "sample_indices": {
            "pseudo_base_train": [int(x) for x in train_ids],
            "pseudo_base_query": [int(x) for x in base_query_ids],
            "pseudo_novel_support_by_session": [
                [int(x) for x in ids] for ids in novel_support_by_session
            ],
            "pseudo_novel_query_by_class": {
                str(k): [int(x) for x in v] for k, v in novel_query_by_class.items()
            },
        },
        "audit": {
            "train_query_overlap": 0,
            "train_support_overlap": 0,
            "train_novel_query_overlap": 0,
            "support_novel_query_overlap": 0,
            "base_query_novel_query_overlap": 0,
        },
    }
    return trainset, base_query, sessions, manifest
