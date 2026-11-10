"""Regression tests for dataset roots and FSCIL session-index paths."""
import os
import tempfile
import unittest
from types import SimpleNamespace

from dataloader.data_utils import _resolve_data_roots, _session_index_path


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("fixture\n")


def make_cub(root, nested=False):
    dataset_root = os.path.join(root, "CUB_200_2011")
    if nested:
        dataset_root = os.path.join(dataset_root, "CUB_200_2011")
    os.makedirs(os.path.join(dataset_root, "images"), exist_ok=True)
    for name in (
        "images.txt", "train_test_split.txt",
        "image_class_labels.txt", "classes.txt",
    ):
        touch(os.path.join(dataset_root, name))


class DatasetPathResolutionTests(unittest.TestCase):
    def test_cub_shared_root_nested_layout_and_incremental_split(self):
        with tempfile.TemporaryDirectory() as root:
            make_cub(root, nested=True)
            args = SimpleNamespace(
                dataset="cub200", dataroot=root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, os.path.join(root, "CUB_200_2011"))
            self.assertEqual(args._configured_data_root, root)

            split = os.path.join(root, "index_list", "cub200", "session_2.txt")
            touch(split)
            self.assertEqual(_session_index_path(args, 1), split)

    def test_cub_direct_dataset_root_is_normalized(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_root = os.path.join(root, "CUB_200_2011")
            make_cub(root, nested=False)
            args = SimpleNamespace(
                dataset="cub200", dataroot=dataset_root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)
            self.assertEqual(args._configured_data_root, root)

    def test_cifar100_shared_root_and_direct_root(self):
        with tempfile.TemporaryDirectory() as root:
            cifar_root = os.path.join(root, "CIFAR_100", "cifar-100-python")
            for name in ("train", "test", "meta"):
                touch(os.path.join(cifar_root, name))
            args = SimpleNamespace(
                dataset="cifar100", dataroot=root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, os.path.join(root, "CIFAR_100"))
            self.assertEqual(args._configured_data_root, root)

            args = SimpleNamespace(
                dataset="cifar100",
                dataroot=os.path.join(root, "CIFAR_100"),
                index_list_root=None,
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, os.path.join(root, "CIFAR_100"))
            self.assertEqual(args._configured_data_root, root)

    def test_miniimagenet_shared_and_direct_roots(self):
        with tempfile.TemporaryDirectory() as root:
            mini_root = os.path.join(root, "miniimagenet")
            os.makedirs(os.path.join(mini_root, "images"), exist_ok=True)
            touch(os.path.join(mini_root, "train.csv"))
            touch(os.path.join(mini_root, "test.csv"))

            args = SimpleNamespace(
                dataset="mini_imagenet", dataroot=root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)
            self.assertEqual(args._configured_data_root, root)

            args = SimpleNamespace(
                dataset="mini_imagenet", dataroot=mini_root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)
            self.assertEqual(args._configured_data_root, root)

    def test_aircraft_shared_and_direct_roots(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = os.path.join(root, "fgvc-aircraft-2013b", "data")
            os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
            for name in (
                "variants.txt", "images_variant_train.txt",
                "images_variant_test.txt",
            ):
                touch(os.path.join(data_dir, name))

            args = SimpleNamespace(
                dataset="FGVCAircraft", dataroot=root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)

            args = SimpleNamespace(
                dataset="FGVCAircraft",
                dataroot=os.path.join(root, "fgvc-aircraft-2013b"),
                index_list_root=None,
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)

    def test_inf200_shared_and_direct_roots(self):
        with tempfile.TemporaryDirectory() as root:
            inf_root = os.path.join(root, "iNF200")
            os.makedirs(os.path.join(inf_root, "train_mini", "class_a"))
            os.makedirs(os.path.join(inf_root, "val", "class_a"))

            args = SimpleNamespace(
                dataset="iNF200", dataroot=root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)

            args = SimpleNamespace(
                dataset="iNF200", dataroot=inf_root, index_list_root=None
            )
            _resolve_data_roots(args)
            self.assertEqual(args.dataroot, root)
            self.assertEqual(args._configured_data_root, root)

    def test_index_root_override_and_original_session_numbering(self):
        with tempfile.TemporaryDirectory() as root:
            override = os.path.join(root, "my_splits")
            split = os.path.join(override, "iNF200", "session_1.txt")
            touch(split)
            args = SimpleNamespace(
                dataset="iNF200",
                dataroot=root,
                _configured_data_root=root,
                index_list_root=override,
            )
            self.assertEqual(_session_index_path(args, 0), split)

    def test_empty_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            split = os.path.join(root, "index_list", "cub200", "session_2.txt")
            os.makedirs(os.path.dirname(split), exist_ok=True)
            open(split, "w", encoding="utf-8").close()
            args = SimpleNamespace(
                dataset="cub200",
                dataroot=root,
                _configured_data_root=root,
                index_list_root=None,
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                _session_index_path(args, 1)


if __name__ == "__main__":
    unittest.main()
