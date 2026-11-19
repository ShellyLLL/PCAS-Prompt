import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_cub200_mapping import (
    audit_class_names,
    audit_description_names,
    audit_image_records,
    normalize_name,
    read_id_value_file,
)


class CUB200NameAuditTests(unittest.TestCase):
    def test_normalize_cub_class_metadata_name(self):
        self.assertEqual(
            normalize_name("001.Black_footed_Albatross"),
            normalize_name("Black footed Albatross"),
        )

    def test_class_ids_map_to_canonical_code_order(self):
        result = audit_class_names(
            {1: "001.Black_footed_Albatross", 2: "002.Laysan_Albatross"},
            ["Black footed Albatross", "Laysan Albatross"],
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["name_mismatches"], [])

    def test_class_name_typo_is_detected(self):
        result = audit_class_names(
            {1: "144.Arctic_Tern"},
            ["Artic Tern"],
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["name_mismatches"][0]["class_id"], 1)

    def test_missing_class_id_is_detected(self):
        result = audit_class_names(
            {1: "001.Black_footed_Albatross"},
            ["Black footed Albatross", "Laysan Albatross"],
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["missing_class_ids"], [2])


class CUB200DescriptionAuditTests(unittest.TestCase):
    def test_description_keys_match_dataset_class_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.json"
            path.write_text(
                json.dumps({"Black footed Albatross": ["A bird."]}),
                encoding="utf-8",
            )
            result = audit_description_names(
                path, {1: "001.Black_footed_Albatross"}
            )
            self.assertEqual(result["status"], "verified")

    def test_description_typo_is_detected_against_metadata_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.json"
            path.write_text(
                json.dumps({"Artic Tern": ["A tern."]}),
                encoding="utf-8",
            )
            result = audit_description_names(path, {1: "144.Arctic_Tern"})
            self.assertEqual(result["status"], "mismatch")
            self.assertEqual(result["missing_keys"], ["144.Arctic_Tern"])
            self.assertEqual(result["extra_keys"], ["Artic Tern"])


class CUB200RecordAuditTests(unittest.TestCase):
    def test_image_label_and_split_ids_are_consistent(self):
        result = audit_image_records(
            {1: "a.jpg", 2: "b.jpg"},
            {1: "1", 2: "2"},
            {1: "1", 2: "0"},
            2,
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["train_image_count"], 1)
        self.assertEqual(result["test_image_count"], 1)

    def test_missing_label_record_is_detected(self):
        result = audit_image_records(
            {1: "a.jpg", 2: "b.jpg"},
            {1: "1"},
            {1: "1", 2: "0"},
            2,
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertTrue(any("ID sets differ" in item for item in result["errors"]))

    def test_out_of_range_class_id_is_detected(self):
        result = audit_image_records(
            {1: "a.jpg"},
            {1: "3"},
            {1: "1"},
            2,
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["invalid_labels"][0]["class_id"], 3)

    def test_pair_file_reader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.txt"
            path.write_text("1 bird\n1 dog\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_id_value_file(path)


if __name__ == "__main__":
    unittest.main()
