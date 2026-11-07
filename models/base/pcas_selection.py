"""Leakage-free base-validation selection for the PCAS component capacity."""
import json
import os

import numpy as np
import torch


class DatasetIndexView(torch.utils.data.Dataset):
    """Stable indexed view that delegates transforms to its source dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("Dataset view indices must be a non-empty 1-D array.")
        if np.any(self.indices < 0) or np.any(self.indices >= len(dataset)):
            raise IndexError("Dataset view contains an out-of-range source index.")
        targets = np.asarray(dataset.targets, dtype=np.int64)
        if targets.shape[0] != len(dataset):
            raise ValueError("Dataset targets are not aligned with samples.")
        self.targets = targets[self.indices].tolist()
        source_data = getattr(dataset, "data", None)
        self.data = (
            [source_data[int(i)] for i in self.indices]
            if isinstance(source_data, (list, tuple)) else None
        )

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, value):
        self.dataset.transform = value

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[int(index)])]


def stratified_base_split_indices(
    labels, validation_fraction=0.1, seed=1, min_train_per_class=4
):
    """Split only base-training samples; preserve every class in both subsets."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Base labels must be a non-empty 1-D array.")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")
    if int(min_train_per_class) < 1:
        raise ValueError("min_train_per_class must be positive.")

    rng = np.random.default_rng(int(seed))
    train_indices, validation_indices = [], []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id).copy()
        if class_indices.size <= int(min_train_per_class):
            raise ValueError(
                "Class {} has {} samples; need more than {} to create a "
                "validation holdout and preserve LOO training support."
                .format(class_id, class_indices.size, int(min_train_per_class))
            )
        rng.shuffle(class_indices)
        validation_count = max(
            1, int(round(class_indices.size * float(validation_fraction)))
        )
        validation_count = min(
            validation_count, class_indices.size - int(min_train_per_class)
        )
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Base train/validation split is not disjoint.")
    return train_indices, validation_indices


def run_component_capacity_selection(
    trainer, base_trainset, candidates=(1, 2, 4, 8, 16),
    validation_fraction=0.1, split_seed=None
):
    """Select M by base-training validation macro accuracy, never test accuracy."""
    candidates = [int(x) for x in candidates]
    if not candidates or any(x < 1 for x in candidates):
        raise ValueError("Component candidates must be positive integers.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Component candidates must not contain duplicates.")
    if 1 not in candidates:
        raise ValueError("M=1 single-generator control must be included.")

    training_seed = int(trainer.seed)
    split_seed = training_seed if split_seed is None else int(split_seed)
    train_idx, val_idx = stratified_base_split_indices(
        base_trainset.targets,
        validation_fraction=validation_fraction,
        seed=split_seed,
        min_train_per_class=int(getattr(trainer.args, "samples_per_class", 4)),
    )
    train_view = DatasetIndexView(base_trainset, train_idx)
    val_view = DatasetIndexView(base_trainset, val_idx)
    selection_root = trainer.save_dir
    rows = []

    for candidate_index, num_components in enumerate(candidates):
        trainer.args.num_components = num_components
        if candidate_index > 0:
            trainer._set_seed(training_seed)
            trainer.model = trainer.model.__class__(trainer.args).to(trainer.device)
        # Reset RNG after initialization so candidates share batch and augmentation
        # randomness as far as the deterministic data pipeline permits.
        trainer._set_seed(training_seed)
        trainer.save_dir = os.path.join(
            selection_root, "components_M{}".format(num_components)
        )
        os.makedirs(trainer.save_dir, exist_ok=True)
        trainer.results = []
        trainer.base_prototypes = None
        trainer.base_teacher_prototypes = None

        trainer._train_base(train_view, trainloader=None)
        metrics = trainer._evaluate(
            val_view, int(trainer.args.base_class),
            trainer.base_prototypes, session=0
        )
        prompt_parameters = sum(
            p.numel() for p in trainer.model.pcas.parameters()
        )
        rows.append({
            "num_components": num_components,
            "validation_class_average_accuracy": float(
                metrics["class_average_accuracy"]
            ),
            "validation_sample_accuracy": float(metrics["accuracy"]),
            "validation_mean_prototype_distance": float(
                metrics["mean_prototype_distance"]
            ),
            "prompt_parameter_count": int(prompt_parameters),
            "validation_samples": int(len(val_view)),
            "training_samples": int(len(train_view)),
            "checkpoint_dir": trainer.save_dir,
        })
        print("[PCAS component selection]", rows[-1])

    best = sorted(
        rows,
        key=lambda row: (
            -row["validation_class_average_accuracy"],
            row["prompt_parameter_count"],
            row["num_components"],
        ),
    )[0]
    summary = {
        "method": "PCAS-Prompt",
        "dataset": str(trainer.args.dataset),
        "training_seed": training_seed,
        "validation_split_seed": split_seed,
        "validation_source": "stratified holdout from base training split",
        "validation_fraction": float(validation_fraction),
        "selection_metric": "validation_class_average_accuracy",
        "tie_break": "smaller prompt parameter count, then smaller M",
        "candidate_components": candidates,
        "selected_num_components": int(best["num_components"]),
        "candidate_results": rows,
    }
    trainer.save_dir = selection_root
    os.makedirs(selection_root, exist_ok=True)
    output_path = os.path.join(selection_root, "pcas_component_selection.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[PCAS component selection summary]", summary)
    return summary, output_path
