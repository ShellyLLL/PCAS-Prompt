import numpy as np
import torch

from models.base.pcas_selection import DatasetIndexView, stratified_base_split_indices


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.targets = [0] * 10 + [1] * 12 + [2] * 14
        self.data = list(range(len(self.targets)))
        self.transform = lambda value: value

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([index]), self.targets[index]


def test_stratified_split_is_reproducible_disjoint_and_class_complete():
    dataset = ToyDataset()
    train_a, val_a = stratified_base_split_indices(
        dataset.targets, 0.2, 13, min_train_per_class=4
    )
    train_b, val_b = stratified_base_split_indices(
        dataset.targets, 0.2, 13, min_train_per_class=4
    )
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert not np.intersect1d(train_a, val_a).size
    labels = np.asarray(dataset.targets)
    assert set(labels[train_a]) == {0, 1, 2}
    assert set(labels[val_a]) == {0, 1, 2}


def test_dataset_index_view_delegates_transform_and_preserves_local_order():
    dataset = ToyDataset()
    view = DatasetIndexView(dataset, [3, 11, 25])
    assert view.targets == [0, 1, 2]
    assert view.data == [3, 11, 25]
    image, label = view[2]
    assert image.item() == 25
    assert label == 2
    marker = lambda value: value + 1
    view.transform = marker
    assert dataset.transform is marker


def test_stratified_split_rejects_class_without_validation_capacity():
    try:
        stratified_base_split_indices([0] * 4 + [1] * 8, 0.1, 1, 4)
    except ValueError as error:
        assert "need more than" in str(error)
    else:
        raise AssertionError("A class without validation capacity must be rejected")
