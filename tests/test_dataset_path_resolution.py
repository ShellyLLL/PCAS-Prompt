"""Regression tests for the shared three-dataset directory layout."""
import os
import tempfile
import unittest
from types import SimpleNamespace

from dataloader.data_utils import _resolve_data_roots, _session_index_path


class DatasetPathResolutionTests(unittest.TestCase):
    def test_cub_nested_layout_and_default_index_root(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = os.path.join(root, "CUB_200_2011", "CUB_200_2011")
            os.makedirs(os.path.join(metadata, "images"))
            open(os.path.join(metadata, "images.txt"), "w").close()
            args = SimpleNamespace(dataset="cub200", dataroot=root, index_list_root=None)
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, os.path.join(root, "CUB_200_2011"))
            split_dir = os.path.join(root, "index_list", "cub200")
            os.makedirs(split_dir)
            split = os.path.join(split_dir, "session_2.txt")
            open(split, "w").close()
            self.assertEqual(_session_index_path(args, 1), split)

    def test_cifar100_nested_layout(self):
        with tempfile.TemporaryDirectory() as root:
            cifar = os.path.join(root, "CIFAR_100", "cifar-100-python")
            os.makedirs(cifar)
            open(os.path.join(cifar, "train"), "w").close()
            args = SimpleNamespace(dataset="cifar100", dataroot=root, index_list_root=None)
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, os.path.join(root, "CIFAR_100"))

    def test_miniimagenet_csv_at_dataset_root(self):
        with tempfile.TemporaryDirectory() as root:
            mini = os.path.join(root, "miniimagenet")
            os.makedirs(os.path.join(mini, "images"))
            open(os.path.join(mini, "train.csv"), "w").close()
            open(os.path.join(mini, "test.csv"), "w").close()
            args = SimpleNamespace(dataset="mini_imagenet", dataroot=root, index_list_root=None)
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)

    def test_miniimagenet_session_directory_alias(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "miniimagenet"))
            args = SimpleNamespace(
                dataset="mini_imagenet",
                dataroot=root,
                _configured_data_root=root,
                index_list_root=None,
            )
            split_dir = os.path.join(root, "index_list", "miniimagenet")
            os.makedirs(split_dir)
            split = os.path.join(split_dir, "session_3.txt")
            open(split, "w").close()
            self.assertEqual(_session_index_path(args, 2), split)


if __name__ == "__main__":
    unittest.main()
