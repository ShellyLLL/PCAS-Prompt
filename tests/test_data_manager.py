import unittest
from types import SimpleNamespace

import numpy as np

from datasets.data_manager import DatasetManager


class DatasetManagerSamplingTests(unittest.TestCase):
    @staticmethod
    def manager():
        manager = DatasetManager.__new__(DatasetManager)
        manager.num_base_shot = -1
        manager.num_inc_shot = 2
        return manager

    def test_incremental_sampling_returns_exact_k_per_class(self):
        manager = self.manager()
        images = np.asarray([f"image_{i}" for i in range(8)])
        labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2])
        selected_x, selected_y = manager._select_data_from_class_index(
            images, labels, np.asarray([1, 2]), 2, "train"
        )
        self.assertEqual(len(selected_x), 4)
        self.assertEqual(np.bincount(selected_y, minlength=3)[1:3].tolist(), [2, 2])

    def test_incremental_sampling_fails_instead_of_silently_using_fewer_shots(self):
        manager = self.manager()
        images = np.asarray(["a", "b", "c"])
        labels = np.asarray([0, 1, 1])
        with self.assertRaisesRegex(ValueError, "only 2 training samples"):
            manager._select_data_from_class_index(
                images, labels, np.asarray([1]), 3, "train"
            )

    def test_base_all_samples_setting_keeps_every_base_sample(self):
        manager = self.manager()
        images = np.asarray(["a", "b", "c"])
        labels = np.asarray([0, 0, 1])
        selected_x, selected_y = manager._select_data_from_class_index(
            images, labels, np.asarray([0, 1]), -1, "train"
        )
        self.assertEqual(len(selected_x), 3)
        self.assertEqual(selected_y.tolist(), [0, 0, 1])

    def test_test_split_is_never_shot_subsampled(self):
        manager = self.manager()
        images = np.asarray(["a", "b", "c", "d"])
        labels = np.asarray([0, 0, 0, 1])
        selected_x, selected_y = manager._select_data_from_class_index(
            images, labels, np.asarray([0]), 1, "test"
        )
        self.assertEqual(len(selected_x), 3)
        self.assertEqual(selected_y.tolist(), [0, 0, 0])

    def test_empty_class_selection_is_rejected(self):
        manager = self.manager()
        with self.assertRaisesRegex(ValueError, "no samples for class ID 2"):
            manager._select_data_from_class_index(
                np.asarray(["a"]), np.asarray([0]), np.asarray([2]), 1, "train"
            )

    def test_dataset_target_validation_rejects_missing_test_class(self):
        manager = self.manager()
        manager.num_total_classes = 2
        manager.num_init_cls = 1
        manager.train_targets = np.asarray([0, 1, 1], dtype=np.int64)
        manager.test_targets = np.asarray([0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "Test split has no images"):
            manager._validate_dataset_targets()

    def test_dataset_target_validation_rejects_out_of_range_labels(self):
        manager = self.manager()
        manager.num_total_classes = 2
        manager.num_init_cls = 1
        manager.train_targets = np.asarray([0, 1, 2], dtype=np.int64)
        manager.test_targets = np.asarray([0, 1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "out-of-range class IDs"):
            manager._validate_dataset_targets()

    def test_config_validation_rejects_non_positive_incremental_shot(self):
        manager = self.manager()
        manager.num_init_cls = 1
        manager.num_inc_cls = 1
        manager.num_base_shot = -1
        manager.num_inc_shot = 0
        manager.train_batchsize_base = 4
        manager.train_batchsize_inc = 4
        manager.test_batchsize = 4
        manager.num_workers = 0
        with self.assertRaisesRegex(ValueError, "NUM_INC_SHOT must be positive"):
            manager._validate_config()


if __name__ == "__main__":
    unittest.main()
