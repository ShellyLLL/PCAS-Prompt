import unittest

import torch
import torch.nn.functional as F

from models.base.pcas_losses import (
    build_component_prototype_cache,
    component_prototype_logits,
    componentwise_prototype_calibration,
    leave_one_out_component_prototype_loss,
    leave_one_out_component_prototype_losses,
    component_sample_affinity_consistency,
    component_prototype_affinity_consistency,
)
from models.base.pcas_modules import PCASPrompt
from models.base.pcas_trainer import (
    IndexedDataset,
    compute_taskwise_accuracy,
    validate_exact_shot_support,
    validate_seen_class_coverage,
)


class TestComponentPrototypeCalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.features = F.normalize(torch.randn(6, 4, 16), dim=-1)
        self.cache = build_component_prototype_cache(self.features, self.labels, 3)
        self.weights = torch.softmax(torch.randn(6, 4), dim=-1)

    def test_component_cache_shapes_and_normalization(self):
        prototypes = self.cache["prototypes"]
        self.assertEqual(tuple(prototypes.shape), (4, 3, 16))
        self.assertTrue(torch.allclose(
            prototypes.norm(dim=-1), torch.ones(4, 3), atol=1e-5
        ))

    def test_component_leave_one_out_loss_is_differentiable(self):
        raw_student = (self.features + 0.01 * torch.randn_like(self.features)).requires_grad_()
        student = F.normalize(raw_student, dim=-1)
        loss = leave_one_out_component_prototype_loss(
            student, self.weights, self.labels, torch.arange(6), self.cache
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student.grad_fn)

    def test_component_logits_use_weighted_component_similarity(self):
        query = F.normalize(torch.randn(5, 4, 16), dim=-1)
        weights = torch.softmax(torch.randn(5, 4), dim=-1)
        prototypes = F.normalize(torch.randn(4, 7, 16), dim=-1)
        logits = component_prototype_logits(query, weights, prototypes)
        expected = (
            torch.einsum("bmd,mcd->bmc", query, prototypes)
            * weights.unsqueeze(-1)
        ).sum(1) * 16.0
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_calibration_is_componentwise_and_normalized(self):
        novel = F.normalize(torch.randn(3, 4, 16), dim=-1)
        base = F.normalize(torch.randn(4, 6, 16), dim=-1)
        calibrated, weights, indices = componentwise_prototype_calibration(
            novel, base, alpha=0.4, tau=12.0, top_r=3
        )
        self.assertEqual(tuple(calibrated.shape), (3, 4, 16))
        self.assertEqual(tuple(weights.shape), (3, 4, 3))
        self.assertEqual(tuple(indices.shape), (3, 4, 3))
        self.assertTrue(torch.allclose(
            calibrated.norm(dim=-1), torch.ones(3, 4), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            weights.sum(-1), torch.ones(3, 4), atol=1e-6
        ))

    def test_unified_component_prototype_objective_supervises_each_branch(self):
        raw_student = (self.features + 0.02 * torch.randn_like(self.features)).requires_grad_()
        student = F.normalize(raw_student, dim=-1)
        losses = leave_one_out_component_prototype_losses(
            student, self.weights, self.labels, torch.arange(6), self.cache
        )
        self.assertEqual(set(losses), {"mixture", "component", "total"})
        for value in losses.values():
            self.assertTrue(torch.isfinite(value))
        losses["total"].backward()
        self.assertIsNotNone(raw_student.grad)
        self.assertGreater(float(raw_student.grad.abs().sum()), 0.0)

    def test_component_affinity_losses_match_component_feature_spaces(self):
        teacher = F.normalize(torch.randn(6, 16), dim=-1)
        teacher_prototypes = F.normalize(torch.randn(3, 16), dim=-1)
        raw_student = (self.features + 0.01 * torch.randn_like(self.features)).requires_grad_()
        student = F.normalize(raw_student, dim=-1)
        sample_loss = component_sample_affinity_consistency(
            teacher, student, self.labels, teacher_prototypes,
            self.cache["prototypes"],
        )
        relation_loss = component_prototype_affinity_consistency(
            teacher, student, self.labels
        )
        total = sample_loss + relation_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(raw_student.grad)
        self.assertGreater(float(raw_student.grad.abs().sum()), 0.0)
    def test_rejects_mismatched_component_count(self):
        with self.assertRaisesRegex(ValueError, "component counts"):
            componentwise_prototype_calibration(
                torch.randn(2, 3, 8), torch.randn(4, 5, 8)
            )


class TestSpatialPrompt(unittest.TestCase):
    def test_prompt_returns_fused_and_isolated_component_maps(self):
        module = PCASPrompt(
            embed_dim=32, bottleneck_dim=8, num_components=4,
            dropout=0.0, attention_temperature=0.2,
        )
        patches = torch.randn(3, 16, 32, requires_grad=True)
        cls = torch.randn(3, 32)
        fused, isolated, weights = module(patches, cls, (4, 4))
        self.assertEqual(tuple(fused.shape), (3, 16, 32))
        self.assertEqual(tuple(isolated.shape), (3, 4, 16, 32))
        self.assertEqual(tuple(weights.shape), (3, 4))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6))
        (fused.square().mean() + isolated.square().mean()).backward()
        self.assertIsNotNone(module.K.grad)
        self.assertIsNotNone(module.A.grad)
        self.assertTrue(0.0 < float(module.residual_scale) < 0.5)


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
        image, label, index = IndexedDataset(self.TwoFieldDataset())[2]
        self.assertEqual(label, 12)
        self.assertEqual(index, 2)

    def test_dataset_metadata_is_not_used_as_cache_index(self):
        image, label, index = IndexedDataset(self.ThreeFieldDataset())[1]
        self.assertEqual(label, 11)
        self.assertEqual(index, 1)


class TestFSCILProtocolAudits(unittest.TestCase):
    def test_triangular_task_accuracy(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predictions = torch.tensor([0, 1, 0, 3, 4, 4, 6, 7, 8, 0])
        values = compute_taskwise_accuracy(labels, predictions, 4, 2, 2)
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 0.75)
        self.assertAlmostEqual(values[1], 0.50)
        self.assertAlmostEqual(values[2], 0.50)

    def test_exact_shot_support(self):
        labels = [100] * 5 + [101] * 5 + [102] * 5
        counts = validate_exact_shot_support(labels, [100, 101, 102], 5, 1)
        self.assertEqual(counts, {100: 5, 101: 5, 102: 5})

    def test_rejects_wrong_shot_count(self):
        with self.assertRaisesRegex(RuntimeError, "not a 5-shot support split"):
            validate_exact_shot_support([100] * 6 + [101] * 5, [100, 101], 5, 1)

    def test_seen_class_coverage(self):
        counts = validate_seen_class_coverage(
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 4, 1
        )
        self.assertEqual(counts.tolist(), [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
