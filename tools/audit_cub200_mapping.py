#!/usr/bin/env python3
"""Audit CUB-200 image IDs, class IDs, canonical class names, and descriptions.

The --root argument is the directory that directly contains classes.txt,
images.txt, image_class_labels.txt, train_test_split.txt, and images/.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple



def load_canonical_classes(repo_root: Path) -> List[str]:
    """Read CLASSES from datasets/cub200.py without importing training dependencies."""
    source_path = repo_root / "datasets" / "cub200.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Cannot locate canonical CUB class file: {source_path}")

    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CLASSES"
            for target in node.targets
        ):
            classes = ast.literal_eval(node.value)
            if not isinstance(classes, list) or not classes or not all(
                isinstance(name, str) and name.strip() for name in classes
            ):
                raise ValueError(f"CLASSES in {source_path} must be a non-empty list of strings")
            return classes

    raise ValueError(f"Could not find a literal CLASSES list in {source_path}")


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = load_canonical_classes(REPO_ROOT)


def normalize_name(value: str) -> str:
    value = re.sub(r"^\s*\d+\.", "", str(value))
    value = value.replace("'", "").replace("_", " ").replace("-", " ")
    value = re.sub(r"[^A-Za-z0-9\s]", " ", value)
    return " ".join(value.lower().split())


def read_id_value_file(path: Path) -> Dict[int, str]:
    records: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed row at {path}:{line_number}: {line.rstrip()!r}"
                )
            try:
                record_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer ID at {path}:{line_number}: {parts[0]!r}"
                ) from exc
            if record_id in records:
                raise ValueError(f"Duplicate ID {record_id} in {path}")
            records[record_id] = parts[1].strip()
    return records


def audit_class_names(
    metadata_classes: Dict[int, str], code_classes: List[str]
) -> Dict:
    expected_ids = set(range(1, len(code_classes) + 1))
    observed_ids = set(metadata_classes)
    mismatches = []
    for class_id in sorted(observed_ids & expected_ids):
        dataset_name = normalize_name(metadata_classes[class_id])
        code_name = normalize_name(code_classes[class_id - 1])
        if dataset_name != code_name:
            mismatches.append(
                {
                    "class_id": class_id,
                    "dataset_class_name": metadata_classes[class_id],
                    "code_class_name": code_classes[class_id - 1],
                }
            )
    return {
        "dataset_class_count": len(metadata_classes),
        "code_class_count": len(code_classes),
        "missing_class_ids": sorted(expected_ids - observed_ids),
        "extra_class_ids": sorted(observed_ids - expected_ids),
        "name_mismatches": mismatches,
        "status": (
            "verified"
            if len(metadata_classes) == len(code_classes)
            and observed_ids == expected_ids
            and not mismatches
            else "mismatch"
        ),
    }


def audit_description_names(
    description_path: Path, metadata_classes: Dict[int, str]
) -> Dict:
    with description_path.open("r", encoding="utf-8") as handle:
        descriptions = json.load(handle)
    if not isinstance(descriptions, dict):
        raise ValueError("CUB description JSON must be an object keyed by class name.")

    normalized_keys: Dict[str, str] = {}
    duplicate_keys = []
    for key, value in descriptions.items():
        normalized = normalize_name(key)
        if normalized in normalized_keys:
            duplicate_keys.append(key)
        normalized_keys[normalized] = key

    expected = {
        normalize_name(value): value
        for _, value in sorted(metadata_classes.items())
    }
    expected_keys = set(expected)
    observed_keys = set(normalized_keys)
    missing = [expected[key] for key in sorted(expected_keys - observed_keys)]
    extra = [normalized_keys[key] for key in sorted(observed_keys - expected_keys)]
    empty = []
    for normalized, original in normalized_keys.items():
        value = descriptions[original]
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not any(
            isinstance(item, str) and item.strip() for item in values
        ):
            empty.append(original)

    return {
        "description_key_count": len(descriptions),
        "missing_keys": missing,
        "extra_keys": extra,
        "duplicate_normalized_keys": duplicate_keys,
        "classes_without_nonempty_descriptions": empty,
        "status": (
            "verified"
            if not (missing or extra or duplicate_keys or empty)
            else "mismatch"
        ),
    }


def audit_image_records(
    images: Dict[int, str],
    labels: Dict[int, str],
    splits: Dict[int, str],
    class_count: int,
) -> Dict:
    image_ids = set(images)
    label_ids = set(labels)
    split_ids = set(splits)
    errors = []
    if image_ids != label_ids:
        errors.append(
            "images.txt and image_class_labels.txt ID sets differ: "
            f"images_only={sorted(image_ids - label_ids)[:8]}, "
            f"labels_only={sorted(label_ids - image_ids)[:8]}."
        )
    if image_ids != split_ids:
        errors.append(
            "images.txt and train_test_split.txt ID sets differ: "
            f"images_only={sorted(image_ids - split_ids)[:8]}, "
            f"splits_only={sorted(split_ids - image_ids)[:8]}."
        )

    invalid_labels = []
    for image_id, raw_class_id in labels.items():
        try:
            class_id = int(raw_class_id)
        except ValueError:
            invalid_labels.append({"image_id": image_id, "class_id": raw_class_id})
            continue
        if not 1 <= class_id <= class_count:
            invalid_labels.append({"image_id": image_id, "class_id": class_id})

    invalid_splits = [
        {"image_id": image_id, "split": split}
        for image_id, split in splits.items()
        if split not in {"0", "1"}
    ]
    counts = Counter()
    class_split_counts = {
        class_id: {"train": 0, "test": 0}
        for class_id in range(1, class_count + 1)
    }
    for image_id in image_ids & label_ids & split_ids:
        try:
            class_id = int(labels[image_id])
        except ValueError:
            continue
        split = splits[image_id]
        if 1 <= class_id <= class_count and split in {"0", "1"}:
            split_name = "train" if split == "1" else "test"
            counts[class_id] += 1
            class_split_counts[class_id][split_name] += 1

    missing_classes = [
        class_id for class_id in range(1, class_count + 1)
        if counts[class_id] == 0
    ]
    if invalid_labels:
        errors.append(f"Found {len(invalid_labels)} invalid image class labels.")
    if invalid_splits:
        errors.append(f"Found {len(invalid_splits)} invalid split flags.")
    if missing_classes:
        errors.append(f"Classes without images: {missing_classes}.")

    return {
        "image_count": len(images),
        "label_record_count": len(labels),
        "split_record_count": len(splits),
        "train_image_count": sum(1 for value in splits.values() if value == "1"),
        "test_image_count": sum(1 for value in splits.values() if value == "0"),
        "invalid_labels": invalid_labels[:20],
        "invalid_split_flags": invalid_splits[:20],
        "classes_without_images": missing_classes,
        "per_class_image_counts": {
            str(class_id): class_split_counts[class_id]
            for class_id in range(1, class_count + 1)
        },
        "errors": errors,
        "status": "verified" if not errors else "mismatch",
    }


def audit_dataset(
    root: Path,
    description_path: Path,
    check_image_files: bool = False,
) -> Tuple[Dict, int]:
    required = [
        root / "classes.txt",
        root / "images.txt",
        root / "image_class_labels.txt",
        root / "train_test_split.txt",
        description_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required audit file does not exist: {path}")

    metadata_classes = read_id_value_file(root / "classes.txt")
    images = read_id_value_file(root / "images.txt")
    labels = read_id_value_file(root / "image_class_labels.txt")
    splits = read_id_value_file(root / "train_test_split.txt")

    class_audit = audit_class_names(metadata_classes, CLASSES)
    description_audit = audit_description_names(description_path, metadata_classes)
    record_audit = audit_image_records(images, labels, splits, len(CLASSES))

    errors = []
    if class_audit["status"] != "verified":
        errors.append("classes.txt class IDs/names do not match the CLASSES list in datasets/cub200.py.")
    if description_audit["status"] != "verified":
        errors.append("CUB description JSON keys/content do not cover classes.txt exactly.")
    errors.extend(record_audit["errors"])

    missing_image_files = []
    if check_image_files:
        for image_id, relative_path in images.items():
            image_path = root / "images" / relative_path
            if not image_path.is_file():
                missing_image_files.append(
                    {"image_id": image_id, "path": relative_path}
                )
        if missing_image_files:
            errors.append(
                f"Found {len(missing_image_files)} image paths missing under images/."
            )

    status = "passed" if not errors else "failed"
    report = {
        "status": status,
        "dataset_root": str(root),
        "canonical_class_count": len(CLASSES),
        "class_name_audit": class_audit,
        "description_audit": description_audit,
        "image_record_audit": record_audit,
        "missing_image_files": missing_image_files[:20],
        "errors": errors,
    }
    return report, 0 if status == "passed" else 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="CUB_200_2011 directory directly containing classes.txt and images/.",
    )
    parser.add_argument(
        "--description",
        type=Path,
        default=repo_root / "description" / "modified_cub200_output.json",
        help="CUB class-description JSON file.",
    )
    parser.add_argument(
        "--check-image-files",
        action="store_true",
        help="Also check that every path in images.txt exists under images/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args()

    try:
        report, exit_code = audit_dataset(
            args.root, args.description, args.check_image_files
        )
    except Exception as exc:
        print(f"[CUB-200 audit] ERROR: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved audit report: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
