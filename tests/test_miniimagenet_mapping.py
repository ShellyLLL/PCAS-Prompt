import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_miniimagenet_mapping import (
    audit_description_keys,
    audit_label_order,
    infer_expected_ids,
    read_split_labels,
)


class MiniImageNetLabelOrderTests(unittest.TestCase):
    def test_canonical_name_labels_are_verified(self):
        names = ["house finch", "robin", "triceratops"]
        result = audit_label_order(names, names)
        self.assertEqual(result["status"], "verified")
        self.assertEqual([row["expected_class_id"] for row in result["rows"]], [0, 1, 2])

    def test_shuffled_name_labels_are_detected(self):
        names = ["house finch", "robin", "triceratops"]
        result = audit_label_order(["robin", "house finch", "triceratops"], names)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(len(result["mismatches"]), 2)

    def test_zero_based_numeric_labels_are_verified(self):
        mapping, evidence = infer_expected_ids(["0", "1", "2"], ["a", "b", "c"])
        self.assertEqual(mapping, {"0": 0, "1": 1, "2": 2})
        self.assertIn("zero-based", evidence)

    def test_one_based_numeric_labels_are_verified(self):
        mapping, evidence = infer_expected_ids(["1", "2", "3"], ["a", "b", "c"])
        self.assertEqual(mapping, {"1": 0, "2": 1, "3": 2})
        self.assertIn("one-based", evidence)

    def test_opaque_labels_are_not_claimed_as_verified(self):
        result = audit_label_order(["n0001", "n0002", "n0003"], ["cat", "dog", "bird"])
        self.assertEqual(result["status"], "unverified")
        self.assertIn("opaque", result["evidence"])

    def test_explicit_label_name_map_detects_order_mismatch(self):
        result = audit_label_order(
            ["n0002", "n0001"],
            ["cat", "dog"],
            {"n0002": "dog", "n0001": "cat"},
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(len(result["mismatches"]), 2)

    def test_explicit_label_name_map_requires_complete_one_to_one_coverage(self):
        mapping, evidence = infer_expected_ids(
            ["n0001", "n0002"], ["cat", "dog"], {"n0001": "cat"}
        )
        self.assertIsNone(mapping)
        self.assertIn("coverage mismatch", evidence)


class MiniImageNetFileAuditTests(unittest.TestCase):
    def test_read_split_labels_preserves_first_seen_order_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "label"])
                writer.writerow(["a.jpg", "cat"])
                writer.writerow(["b.jpg", "dog"])
                writer.writerow(["c.jpg", "cat"])
            order, counts, rows = read_split_labels(path)
            self.assertEqual(order, ["cat", "dog"])
            self.assertEqual(counts, {"cat": 2, "dog": 1})
            self.assertEqual(rows, 3)

    def test_description_key_audit_detects_missing_and_extra_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.json"
            path.write_text(
                json.dumps({"cat": ["a cat"], "extra": ["extra"]}),
                encoding="utf-8",
            )
            result = audit_description_keys(path, ["cat", "dog"])
            self.assertEqual(result["status"], "mismatch")
            self.assertEqual(result["missing_keys"], ["dog"])
            self.assertEqual(result["extra_keys"], ["extra"])


if __name__ == "__main__":
    unittest.main()
