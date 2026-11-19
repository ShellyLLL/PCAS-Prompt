#!/usr/bin/env python3
"""Audit MiniImageNet CSV label IDs against the repository's canonical class order.

The current loader assigns integer IDs in the order labels first appear in
train.csv. Opaque labels (for example WordNet IDs) require an explicit JSON
mapping from raw label to human-readable class name.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets.miniimagenet import CLASSES


def normalize_name(value: str) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def read_split_labels(csv_path: Path) -> Tuple[List[str], Counter, int]:
    first_seen: List[str] = []
    counts: Counter = Counter()
    seen = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) != 2:
            raise ValueError(
                f"{csv_path} must have a two-column CSV header and rows "
                f"(image_path,label); observed header={header!r}"
            )
        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Malformed row at {csv_path}:{line_number}: {row!r}")
            label = row[1].strip()
            if not label:
                raise ValueError(f"Empty class label at {csv_path}:{line_number}")
            if label not in seen:
                seen.add(label)
                first_seen.append(label)
            counts[label] += 1
            row_count += 1
    if row_count == 0:
        raise ValueError(f"No image rows found in {csv_path}")
    return first_seen, counts, row_count


def load_label_name_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Label-name map must be a JSON object: raw_label -> class_name")
    mapping = {}
    for raw_label, class_name in raw.items():
        if not isinstance(raw_label, str) or not isinstance(class_name, str):
            raise ValueError("Every label-name map entry must contain string keys and values.")
        if not raw_label.strip() or not class_name.strip():
            raise ValueError("Label-name map cannot contain empty labels or class names.")
        mapping[raw_label] = class_name
    return mapping


def infer_expected_ids(
    labels: List[str],
    class_names: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, int]], str]:
    class_index = {}
    for index, class_name in enumerate(class_names):
        key = normalize_name(class_name)
        if key in class_index:
            return None, f"Canonical class names are not unique after normalization: {class_name!r}"
        class_index[key] = index

    if label_name_map is not None:
        label_set = set(labels)
        missing = [label for label in labels if label not in label_name_map]
        extra = [label for label in label_name_map if label not in label_set]
        if missing or extra:
            return None, f"Label-name map coverage mismatch; missing={missing[:8]}, extra={extra[:8]}"
        expected = {}
        for label in labels:
            class_key = normalize_name(label_name_map[label])
            if class_key not in class_index:
                return None, f"Mapped class name {label_name_map[label]!r} is not in CLASSES."
            expected[label] = class_index[class_key]
        if len(set(expected.values())) != len(class_names):
            return None, "Label-name map must map one-to-one onto all canonical classes."
        return expected, "verified using the explicit raw-label-to-class-name map"

    normalized_class_index = {normalize_name(name): i for i, name in enumerate(class_names)}
    if all(normalize_name(label) in normalized_class_index for label in labels):
        return (
            {label: normalized_class_index[normalize_name(label)] for label in labels},
            "verified because CSV labels are canonical class names",
        )

    if all(label.isdigit() for label in labels):
        numeric = {label: int(label) for label in labels}
        values = set(numeric.values())
        if values == set(range(len(class_names))):
            return numeric, "verified using zero-based numeric class IDs"
        if values == set(range(1, len(class_names) + 1)):
            return {label: value - 1 for label, value in numeric.items()}, (
                "verified using one-based numeric class IDs"
            )
        return None, "Numeric CSV labels are not exactly the canonical 0..C-1 or 1..C range."

    return None, (
        "CSV labels are opaque identifiers. Their semantic mapping to CLASSES "
        "cannot be proven from train.csv/test.csv alone; provide --label-name-map."
    )


def audit_label_order(
    first_seen_labels: List[str],
    class_names: List[str],
    label_name_map: Optional[Dict[str, str]] = None,
) -> Dict:
    expected_ids, evidence = infer_expected_ids(first_seen_labels, class_names, label_name_map)
    actual_ids = {label: index for index, label in enumerate(first_seen_labels)}
    if expected_ids is None:
        status = "unverified" if evidence.startswith("CSV labels are opaque") else "mismatch"
        return {
            "status": status,
            "evidence": evidence,
            "mismatches": [],
            "rows": [
                {
                    "actual_class_id": index,
                    "raw_label": label,
                    "class_name_assigned_by_current_loader": (
                        class_names[index] if index < len(class_names) else None
                    ),
                    "expected_class_id": None,
                }
                for index, label in enumerate(first_seen_labels)
            ],
        }

    mismatches = []
    rows = []
    for label, actual_id in actual_ids.items():
        expected_id = expected_ids[label]
        row = {
            "actual_class_id": actual_id,
            "raw_label": label,
            "class_name_assigned_by_current_loader": (
                class_names[actual_id] if actual_id < len(class_names) else None
            ),
            "expected_class_id": expected_id,
            "expected_class_name": class_names[expected_id],
        }
        rows.append(row)
        if actual_id != expected_id:
            mismatches.append(row)
    return {
        "status": "mismatch" if mismatches else "verified",
        "evidence": evidence,
        "mismatches": mismatches,
        "rows": rows,
    }


def audit_description_keys(description_path: Path, class_names: List[str]) -> Dict:
    with description_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Description JSON must be an object keyed by class name.")

    normalized_keys = {}
    duplicate_keys = []
    for key, value in raw.items():
        normalized = normalize_name(key)
        if normalized in normalized_keys:
            duplicate_keys.append(key)
        normalized_keys[normalized] = value

    expected = {normalize_name(name) for name in class_names}
    observed = set(normalized_keys)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    empty = []
    for class_name in class_names:
        value = normalized_keys.get(normalize_name(class_name))
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not any(
            isinstance(item, str) and item.strip() for item in values
        ):
            empty.append(class_name)

    return {
        "class_count": len(class_names),
        "description_key_count": len(raw),
        "missing_keys": missing,
        "extra_keys": extra,
        "duplicate_normalized_keys": duplicate_keys,
        "classes_without_nonempty_descriptions": empty,
        "status": "verified" if not (missing or extra or duplicate_keys or empty) else "mismatch",
    }


def audit_dataset(
    root: Path,
    description_path: Path,
    label_name_map_path: Optional[Path] = None,
) -> Tuple[Dict, int]:
    split_dir = root / "miniimagenet" / "split"
    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"
    for path in (train_csv, test_csv, description_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit file does not exist: {path}")

    train_order, train_counts, train_rows = read_split_labels(train_csv)
    test_order, test_counts, test_rows = read_split_labels(test_csv)
    label_name_map = load_label_name_map(label_name_map_path)

    errors = []
    if len(train_order) != len(CLASSES):
        errors.append(
            f"train.csv has {len(train_order)} unique labels; CLASSES has {len(CLASSES)}."
        )
    if set(train_order) != set(test_order):
        errors.append(
            "train.csv and test.csv do not contain the same class-label set; "
            f"train-only={sorted(set(train_order) - set(test_order))[:8]}, "
            f"test-only={sorted(set(test_order) - set(train_order))[:8]}."
        )

    order_audit = audit_label_order(train_order, CLASSES, label_name_map)
    if order_audit["status"] == "mismatch":
        if order_audit["mismatches"]:
            errors.append(
                f"CSV first-seen label order disagrees with CLASSES at "
                f"{len(order_audit['mismatches'])} class IDs."
            )
        else:
            errors.append("CSV label mapping is invalid: " + order_audit["evidence"])

    description_audit = audit_description_keys(description_path, CLASSES)
    if description_audit["status"] != "verified":
        errors.append("Description JSON keys/content do not cover CLASSES exactly.")

    if errors:
        overall, exit_code = "failed", 1
    elif order_audit["status"] == "unverified":
        overall, exit_code = "unverified", 2
    else:
        overall, exit_code = "passed", 0

    mapping_rows = order_audit["rows"]
    for row in mapping_rows:
        label = row["raw_label"]
        row["train_image_count"] = int(train_counts.get(label, 0))
        row["test_image_count"] = int(test_counts.get(label, 0))

    report = {
        "status": overall,
        "dataset_root": str(root),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "canonical_class_count": len(CLASSES),
        "train_unique_label_count": len(train_order),
        "test_unique_label_count": len(test_order),
        "class_order_audit": order_audit,
        "description_audit": description_audit,
        "errors": errors,
    }
    return report, exit_code


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing miniimagenet/split/train.csv and test.csv.",
    )
    parser.add_argument(
        "--description",
        type=Path,
        default=repo_root / "description" / "modified_miniimagent_output_final.json",
        help="CSPC class-description JSON path.",
    )
    parser.add_argument(
        "--label-name-map",
        type=Path,
        default=None,
        help="Optional JSON object mapping every raw CSV label to its class name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the complete audit report as JSON.",
    )
    args = parser.parse_args()

    try:
        report, exit_code = audit_dataset(args.root, args.description, args.label_name_map)
    except Exception as exc:
        print(f"[MiniImageNet audit] ERROR: {exc}", file=sys.stderr)
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
