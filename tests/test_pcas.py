import unittest
import torch
import torch.nn.functional as F

from models.base.pcas_losses import (
    build_prototype_cache,
    leave_one_out_prototype_loss,
    sample_affinity_consistency,
    prototype_affinity_consistency,
    teen_calibrate,
)
from models.base.pcas_modules import PCASPrompt
from models.base.pcas_trainer import (
    IndexedDataset,
    compute_taskwise_accuracy,
    validate_exact_shot_support,
    validate_seen_class_coverage,
)


class TestPCASLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.features = F.normalize(torch.randn(6, 16), dim=-1)
        self.cache = build_prototype_cache(self.features, self.labels, 3)

    def test_prototype_cache_and_loo(self):
        self.assertEqual(tuple(self.cache["prototypes"].shape), (3, 16))
        loss = leave_one_out_prototype_loss(
            self.features, self.labels, torch.arange(6), self.cache
        )
        self.assertTrue(torch.isfinite(loss))

    def test_affinity_losses_are_finite_and_differentiable(self):
        student = F.normalize(
            (self.features + 0.02 * torch.randn_like(self.features)).requires_grad_(),
            dim=-1,
        )
        sample_loss = sample_affinity_consistency(
            self.features, student, self.labels,
            self.cache["prototypes"], F.normalize(student.detach().reshape(3, 2, 16).mean(1), dim=-1)
        )
        relation_loss = prototype_affinity_consistency(
            self.features, student, self.labels
        )
        total = sample_loss + relation_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(student.grad_fn)

    def test_teen_uses_common_feature_space(self):
        base = F.normalize(torch.randn(4, 16), dim=-1)
        novel = F.normalize(torch.randn(2, 16), dim=-1)
        calibrated, weights = teen_calibrate(novel, base, alpha=0.5, tau=16)
        self.assertEqual(tuple(calibrated.shape), (2, 16))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(calibrated.norm(dim=-1), torch.ones(2), atol=1e-5))


class TestPCASPrompt(unittest.TestCase):
    def test_spatial_prompt_preserves_patch_shape_and_router(self):
        module = PCASPrompt(
            embed_dim=32, bottleneck_dim=8, num_components=4,
            dropout=0.0, route_temperature=0.2,
        )
        patches = torch.randn(3, 16, 32, requires_grad=True)
        cls = torch.randn(3, 32)
        enhanced, weights = module(patches, cls, (4, 4))
        self.assertEqual(tuple(enhanced.shape), (3, 16, 32))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6))
        enhanced.square().mean().backward()
        self.assertIsNotNone(module.K.grad)
        self.assertIsNotNone(module.A.grad)



class TestIndexedDataset(unittest.TestCase):
    class TwoFieldDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.tensor([index]), index + 10

    class ThreeFieldDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.tensor([index]), index + 10, 1000 + index

    def test_two_field_dataset_gets_local_index(self):
        wrapped = IndexedDataset(self.TwoFieldDataset())
        image, label, index = wrapped[2]
        self.assertEqual(label, 12)
        self.assertEqual(index, 2)

    def test_three_field_dataset_metadata_is_not_used_as_cache_index(self):
        wrapped = IndexedDataset(self.ThreeFieldDataset())
        image, label, index = wrapped[1]
        self.assertEqual(label, 11)
        self.assertEqual(index, 1)


class TestTaskwiseAccuracy(unittest.TestCase):
    def test_triangular_task_accuracy_by_global_class_ranges(self):
        # Base task has 4 classes; each incremental task adds 2 classes.
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predictions = torch.tensor([0, 1, 0, 3, 4, 4, 6, 7, 8, 0])
        values = compute_taskwise_accuracy(
            labels, predictions, base_class=4, way=2, session=2
        )
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 0.75)
        self.assertAlmostEqual(values[1], 0.50)
        self.assertAlmostEqual(values[2], 0.50)

    def test_session_zero_contains_only_base_task(self):
        labels = torch.tensor([0, 1, 2, 3])
        predictions = torch.tensor([0, 1, 1, 3])
        values = compute_taskwise_accuracy(
            labels, predictions, base_class=4, way=2, session=0
        )
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0], 0.75)

    def test_rejects_missing_task_samples(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5])
        predictions = labels.clone()
        with self.assertRaisesRegex(ValueError, "No test samples found for task 2"):
            compute_taskwise_accuracy(
                labels, predictions, base_class=4, way=2, session=2
            )


class TestFSCILProtocolAudits(unittest.TestCase):
    def test_accepts_exact_five_shot_support(self):
        labels = [100] * 5 + [101] * 5 + [102] * 5
        counts = validate_exact_shot_support(
            labels, expected_classes=[100, 101, 102], shots=5, session=1
        )
        self.assertEqual(counts, {100: 5, 101: 5, 102: 5})

    def test_rejects_support_with_more_than_five_images_per_class(self):
        labels = [100] * 6 + [101] * 5
        with self.assertRaisesRegex(RuntimeError, "not a 5-shot support split"):
            validate_exact_shot_support(
                labels, expected_classes=[100, 101], shots=5, session=1
            )

    def test_rejects_missing_test_class(self):
        labels = torch.tensor([0, 0, 1, 1, 3, 3])
        with self.assertRaisesRegex(RuntimeError, "missing seen classes"):
            validate_seen_class_coverage(labels, seen_classes=4, session=1)

    def test_accepts_complete_test_class_coverage(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        counts = validate_seen_class_coverage(labels, seen_classes=4, session=1)
        self.assertEqual(counts.tolist(), [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
