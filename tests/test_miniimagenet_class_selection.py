import unittest

import numpy as np

from dataloader.miniimagenet.miniimagenet import MiniImageNet


class MiniImageNetClassSelectionTests(unittest.TestCase):
    def test_class_selection_uses_global_integer_labels(self):
        data = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        targets = [0, 1, 0, 2]
        selected_data, selected_targets = MiniImageNet.SelectfromClasses(
            object.__new__(MiniImageNet), data, targets, np.asarray([0, 2])
        )
        self.assertEqual(selected_data, ["a.jpg", "c.jpg", "d.jpg"])
        self.assertEqual(selected_targets, [0, 0, 2])

    def test_empty_class_selection_fails_explicitly(self):
        dataset = object.__new__(MiniImageNet)
        with self.assertRaisesRegex(RuntimeError, "selection is empty"):
            dataset.SelectfromClasses(["a.jpg"], [0], np.asarray([5]))


if __name__ == "__main__":
    unittest.main()
