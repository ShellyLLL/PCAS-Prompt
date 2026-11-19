#!/usr/bin/env python3
"""Preflight the configured FSCIL dataset and CSPC class descriptions.

This check does not load CLIP or start training. It validates class/session
coverage, exact K-shot sampling, image-path integrity, split disjointness,
and class-description coverage before an expensive CSPC run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.data_manager import DatasetManager
from engine.cspc_engine import CSPCRunner
from main import setup_cfg


def validate_dataset(data_cfg: str, train_cfg: str) -> dict:
    cfg = setup_cfg(data_cfg, train_cfg)
    if str(cfg.METHOD).upper() != "CSPC":
        raise ValueError(
            f"Expected METHOD=CSPC in trainer config, got {cfg.METHOD!r}."
        )

    manager = DatasetManager(cfg)
    runner = CSPCRunner.__new__(CSPCRunner)
    runner.class_names = list(manager.class_names)
    descriptions = runner._load_class_texts(cfg.DATASET.GPT_PATH)

    if len(descriptions) != manager.num_total_classes:
        raise ValueError(
            "Class-description count does not match the dataset class count."
        )

    train_paths = set(manager.train_data) if isinstance(manager.train_data, list) else None
    test_paths = set(manager.test_data) if isinstance(manager.test_data, list) else None
    overlap_count = 0
    if train_paths is not None and test_paths is not None:
        overlap_count = len(train_paths.intersection(test_paths))
        if overlap_count:
            raise ValueError(
                f"Train/test image path overlap detected: {overlap_count} paths."
            )

    missing_files = []
    if train_paths is not None:
        for split_name, paths in (
            ("train", manager.train_data),
            ("test", manager.test_data),
        ):
            for image_path in paths:
                if not os.path.isfile(image_path):
                    missing_files.append(image_path)
                    if len(missing_files) >= 20:
                        break
            if missing_files:
                break
    if missing_files:
        raise FileNotFoundError(
            "Missing image files (first 20): " + ", ".join(missing_files)
        )

    sessions = []
    all_train_class_ids = []
    for task_id, class_ids_array in enumerate(manager.class_index_in_task):
        class_ids = [int(class_id) for class_id in class_ids_array]
        train_dataset = manager.get_dataset(
            task_id, source="train", mode="test", accumulated_past=False
        )
        test_dataset = manager.get_dataset(
            task_id, source="test", mode="test", accumulated_past=False
        )
        train_counts = Counter(int(label) for label in train_dataset.labels)
        test_counts = Counter(int(label) for label in test_dataset.labels)

        expected_train_counts = {}
        for class_id in class_ids:
            available = int(np.count_nonzero(manager.train_targets == class_id))
            if task_id == 0 and int(cfg.DATASET.NUM_BASE_SHOT) == -1:
                expected = available
            elif task_id == 0:
                expected = int(cfg.DATASET.NUM_BASE_SHOT)
            else:
                expected = int(cfg.DATASET.NUM_INC_SHOT)
            expected_train_counts[str(class_id)] = expected
            actual = train_counts.get(class_id, 0)
            if actual != expected:
                raise ValueError(
                    f"Session {task_id}, class {class_id}: expected {expected} "
                    f"training samples, selected {actual}."
                )
            if test_counts.get(class_id, 0) <= 0:
                raise ValueError(
                    f"Session {task_id}, class {class_id}: no test samples."
                )

        all_train_class_ids.extend(class_ids)
        sessions.append(
            {
                "session": task_id,
                "class_ids": class_ids,
                "train_sample_counts": {
                    str(class_id): int(train_counts[class_id])
                    for class_id in class_ids
                },
                "expected_train_sample_counts": expected_train_counts,
                "cumulative_test_class_counts": {
                    str(class_id): int(test_counts[class_id])
                    for class_id in sorted(test_counts)
                },
            }
        )

    expected_ids = list(range(manager.num_total_classes))
    if all_train_class_ids != expected_ids:
        raise ValueError(
            "FSCIL sessions do not partition every class exactly once in order."
        )

    return {
        "status": "passed",
        "method": str(cfg.METHOD),
        "dataset": str(cfg.DATASET.NAME),
        "dataset_root": str(cfg.DATASET.ROOT),
        "num_classes": manager.num_total_classes,
        "num_sessions": manager.num_tasks,
        "train_samples": int(len(manager.train_targets)),
        "test_samples": int(len(manager.test_targets)),
        "train_test_path_overlap": overlap_count,
        "description_classes": len(descriptions),
        "missing_image_files": missing_files,
        "sessions": sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        report = validate_dataset(args.data_cfg, args.train_cfg)
    except Exception as exc:
        print(f"[CSPC dataset preflight] FAILED: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved preflight report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
