import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.pcas_losses import (
    build_component_conditioned_prototype_cache,
    build_component_conditioned_prototypes,
    build_prototype_cache,
    component_conditioned_prototype_logits,
    component_prototype_affinity_consistency,
    component_sample_affinity_consistency,
    componentwise_prototype_calibration,
    leave_one_out_component_conditioned_losses,
)
from models.base.pcas_metrics import component_weight_statistics
from models.base.pcas_modules import PCASPrompt, PCASViT
from models.base.pcas_trainer import (
    IndexedDataset,
    compute_taskwise_accuracy,
    validate_exact_shot_support,
    validate_seen_class_coverage,
)


class TestComponentConditionedPrototypes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.features = F.normalize(torch.randn(6, 16), dim=-1)
        self.weights = torch.softmax(torch.randn(6, 4), dim=-1)
        self.cache = build_component_conditioned_prototype_cache(
            self.features, self.weights, self.labels, 3
        )

    def test_cache_uses_one_unified_feature_per_sample(self):
        self.assertEqual(tuple(self.cache["features"].shape), (6, 16))
        self.assertEqual(tuple(self.cache["prototypes"].shape), (4, 3, 16))
        self.assertTrue(torch.allclose(
            self.cache["prototypes"].norm(dim=-1), torch.ones(4, 3), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            self.cache["class_mean_weights"].sum(-1), torch.ones(3), atol=1e-5
        ))
        mask = self.labels == 1
        expected = F.normalize(
            (self.features[mask] * self.weights[mask, 2, None]).sum(0)
            / self.weights[mask, 2].sum(), dim=-1
        )
        self.assertTrue(torch.allclose(expected, self.cache["prototypes"][2, 1], atol=1e-6))

    def test_real_support_prototype_uses_component_weights(self):
        prototype = build_component_conditioned_prototypes(
            self.features[:2], self.weights[:2]
        )
        self.assertEqual(tuple(prototype.shape), (4, 16))
        self.assertTrue(torch.allclose(prototype.norm(dim=-1), torch.ones(4), atol=1e-5))

    def test_classifier_uses_normalized_query_conditioned_recomposition(self):
        query = F.normalize(torch.randn(5, 16), dim=-1)
        weights = torch.softmax(torch.randn(5, 4), dim=-1)
        logits = component_conditioned_prototype_logits(query, weights, self.cache["prototypes"])
        recomposed = F.normalize(
            torch.einsum("bm,mcd->bcd", weights, self.cache["prototypes"]), dim=-1
        )
        expected = torch.einsum("bd,bcd->bc", query, recomposed) * 16.0
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_leave_one_out_objectives_are_differentiable(self):
        raw = (self.features + 0.02 * torch.randn_like(self.features)).requires_grad_()
        losses = leave_one_out_component_conditioned_losses(
            F.normalize(raw, dim=-1), self.weights, self.labels, torch.arange(6), self.cache
        )
        self.assertEqual(set(losses), {"mixture", "component", "total"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(float(raw.grad.abs().sum().item()), 0.0)

    def test_componentwise_calibration_is_normalized(self):
        novel = torch.stack([
            build_component_conditioned_prototypes(self.features[:2], self.weights[:2]),
            build_component_conditioned_prototypes(self.features[2:4], self.weights[2:4]),
        ])
        calibrated, weights, indices = componentwise_prototype_calibration(
            novel, self.cache["prototypes"], alpha=0.4, tau=12.0, top_r=2
        )
        self.assertEqual(tuple(calibrated.shape), (2, 4, 16))
        self.assertEqual(tuple(weights.shape), (2, 4, 2))
        self.assertEqual(tuple(indices.shape), (2, 4, 2))
        self.assertTrue(torch.allclose(calibrated.norm(dim=-1), torch.ones(2, 4), atol=1e-5))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 4), atol=1e-6))

    def test_pcac_losses_are_finite_and_differentiable(self):
        teacher = F.normalize(torch.randn(6, 16), dim=-1)
        teacher_cache = build_prototype_cache(teacher, self.labels, 3)
        raw = (self.features + 0.01 * torch.randn_like(self.features)).requires_grad_()
        student = F.normalize(raw, dim=-1)
        sample = component_sample_affinity_consistency(
            teacher, student, self.weights, self.labels,
            teacher_cache["prototypes"], self.cache["prototypes"]
        )
        relation = component_prototype_affinity_consistency(
            teacher, student, self.weights, self.labels
        )
        self.assertTrue(torch.isfinite(sample))
        self.assertTrue(torch.isfinite(relation))
        (sample + relation).backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(float(raw.grad.abs().sum().item()), 0.0)

    def test_leave_one_out_rejects_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "duplicate dataset indices"):
            leave_one_out_component_conditioned_losses(
                self.features[:2], self.weights[:2], self.labels[:2],
                torch.tensor([0, 0]), self.cache
            )


class TestSingleStreamSpatialPrompt(unittest.TestCase):
    def test_prompt_returns_only_patches_and_weights(self):
        module = PCASPrompt(32, 8, 4, dropout=0.0, attention_temperature=0.2)
        patches = torch.randn(3, 16, 32, requires_grad=True)
        cls = torch.randn(3, 32)
        enhanced, weights = module(patches, cls, (4, 4))
        self.assertEqual(tuple(enhanced.shape), (3, 16, 32))
        self.assertEqual(tuple(weights.shape), (3, 4))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6))
        (enhanced.square().mean() + weights.square().mean()).backward()
        self.assertIsNotNone(module.K.grad)
        self.assertIsNotNone(module.A.grad)
        self.assertTrue(0.0 < float(module.residual_scale.detach().item()) < 0.5)

    def test_vit_tail_is_called_once(self):
        class FakePatchEmbed(nn.Module):
            grid_size = (14, 14)
            def forward(self, images):
                return images.new_ones((images.shape[0], 196, 32))

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = FakePatchEmbed()
                self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
            def _pos_embed(self, patches):
                cls = patches.new_zeros((patches.shape[0], 1, patches.shape[-1]))
                return torch.cat((cls, patches), dim=1)

        class CountingPCASViT(PCASViT):
            def __init__(self):
                nn.Module.__init__(self)
                self.embed_dim = 32
                self.insert_after = 1
                self._incremental_frozen = False
                self.grid_size = (14, 14)
                self.encoder = FakeEncoder()
                self.pcas = PCASPrompt(32, 8, 4, dropout=0.0)
                self.tail_calls = 0
            def _run_frozen_tail(self, tokens):
                self.tail_calls += 1
                return F.normalize(tokens[:, 0], dim=-1)

        model = CountingPCASViT()
        features, auxiliary = model.encode_student(
            torch.randn(2, 3, 224, 224), return_aux=True
        )
        self.assertEqual(tuple(features.shape), (2, 32))
        self.assertEqual(model.tail_calls, 1)
        self.assertEqual(set(auxiliary), {"component_weights", "residual_scale"})


class TestFSCILProtocolAudits(unittest.TestCase):
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return torch.tensor([index]), index + 10, 1000 + index

    def test_indexed_dataset_uses_local_index(self):
        _, label, index = IndexedDataset(self.TinyDataset())[1]
        self.assertEqual(label, 11)
        self.assertEqual(index, 1)

    def test_taskwise_accuracy(self):
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predictions = torch.tensor([0, 1, 0, 3, 4, 4, 6, 7, 8, 0])
        values = compute_taskwise_accuracy(labels, predictions, 4, 2, 2)
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 0.75)
        self.assertAlmostEqual(values[1], 0.50)
        self.assertAlmostEqual(values[2], 0.50)

    def test_exact_shot_and_seen_class_coverage(self):
        counts = validate_exact_shot_support([100] * 5 + [101] * 5, [100, 101], 5, 1)
        self.assertEqual(counts, {100: 5, 101: 5})
        observed = validate_seen_class_coverage(torch.tensor([0, 0, 1, 1]), 2, 1)
        self.assertEqual(observed.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
