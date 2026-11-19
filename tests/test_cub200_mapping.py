import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from datasets.cub200 import CUB200, CLASSES as CUB_CLASSES

from tools.audit_cub200_mapping import (
    CLASSES,
    audit_class_names,
    audit_description_names,
    audit_image_records,
    normalize_name,
    read_id_value_file,
)



class CUB200CLILaunchTests(unittest.TestCase):
    def test_cli_starts_from_outside_repository_without_pythonpath(self):
        script_path = Path(__file__).resolve().parents[1] / "tools" / "audit_cub200_mapping.py"
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--check-image-files", result.stdout)

    def test_canonical_classes_are_loaded_without_importing_dataset_package(self):
        self.assertEqual(len(CLASSES), 200)
        self.assertEqual(CLASSES[0], "Black footed Albatross")


class CUB200NameAuditTests(unittest.TestCase):
    def test_normalize_cub_class_metadata_name(self):
        self.assertEqual(
            normalize_name("001.Black_footed_Albatross"),
            normalize_name("Black footed Albatross"),
        )

    def test_apostrophe_in_official_cub_class_name_is_normalized(self):
        self.assertEqual(
            normalize_name("146.Forster's_Tern"),
            normalize_name("Forsters Tern"),
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
            {1: "141.Common_Tern"},
            ["Arctic Tern"],
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
            result = audit_description_names(path, {1: "141.Common_Tern"})
            self.assertEqual(result["status"], "mismatch")
            self.assertEqual(result["missing_keys"], ["141.Common_Tern"])
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




class CUB200RuntimeLoaderTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(root, class_141_name=None):
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)

        class_rows = []
        for class_id, class_name in enumerate(CUB_CLASSES, start=1):
            metadata_name = class_name.replace(" ", "_")
            if class_id == 141 and class_141_name is not None:
                metadata_name = class_141_name
            class_rows.append(f"{class_id} {class_id:03d}.{metadata_name}")
        (root / "classes.txt").write_text("\n".join(class_rows) + "\n", encoding="utf-8")

        (root / "images.txt").write_text(
            "1 001.Black_footed_Albatross/sample_train.jpg\n"
            "2 141.Arctic_Tern/sample_test.jpg\n",
            encoding="utf-8",
        )
        (root / "image_class_labels.txt").write_text("1 1\n2 141\n", encoding="utf-8")
        (root / "train_test_split.txt").write_text("1 1\n2 0\n", encoding="utf-8")

    def test_direct_cub_200_2011_root_is_loaded_without_duplicate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CUB_200_2011"
            self._write_dataset(root)
            dataset = CUB200(str(root))

            self.assertEqual(dataset.dataset_root, str(root))
            self.assertEqual(dataset.train_targets, [0])
            self.assertEqual(dataset.test_targets, [140])
            self.assertEqual(
                dataset.train_data[0],
                str(root / "images" / "001.Black_footed_Albatross" / "sample_train.jpg"),
            )

    def test_parent_directory_layout_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            nested = parent / "CUB_200_2011"
            self._write_dataset(nested)
            dataset = CUB200(str(parent))
            self.assertEqual(dataset.dataset_root, str(nested))

    def test_class_141_uses_canonical_arctic_tern_name(self):
        metadata = {
            class_id: f"{class_id:03d}." + class_name.replace(" ", "_")
            for class_id, class_name in enumerate(CUB_CLASSES, start=1)
        }
        self.assertEqual(metadata[141], "141.Arctic_Tern")
        result = audit_class_names(metadata, CUB_CLASSES)
        self.assertEqual(result["status"], "verified")

    def test_released_artic_tern_metadata_alias_is_accepted(self):
        metadata = {
            class_id: f"{class_id:03d}." + class_name.replace(" ", "_")
            for class_id, class_name in enumerate(CUB_CLASSES, start=1)
        }
        metadata[141] = "141.Artic_Tern"
        result = audit_class_names(metadata, CUB_CLASSES)
        self.assertEqual(result["status"], "verified")

    def test_runtime_rejects_class_metadata_name_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CUB_200_2011"
            self._write_dataset(root, class_141_name="Common_Tern")
            with self.assertRaisesRegex(ValueError, "class ID 141"):
                CUB200(str(root))

    def test_description_keys_cover_canonical_cub_classes(self):
        metadata = {
            class_id: f"{class_id:03d}." + class_name.replace(" ", "_")
            for class_id, class_name in enumerate(CUB_CLASSES, start=1)
        }
        description_path = (
            Path(__file__).resolve().parents[1]
            / "description"
            / "modified_cub200_output.json"
        )
        result = audit_description_names(description_path, metadata)
        self.assertEqual(result["status"], "verified", msg=str(result))


if __name__ == "__main__":
    unittest.main()
