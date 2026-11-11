import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.slot_ot_modules import PromptSlotAttention, PromptSlotViT
from models.base.slot_ot_losses import (
    build_class_local_memory,
    calibrate_local_memory,
    joint_prototype_logits,
    pairwise_local_set_distance,
    unbalanced_sinkhorn,
    weighted_kmeans,
)


class TestPromptSlotAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_shapes_mass_and_gradients(self):
        module = PromptSlotAttention(input_dim=32, slot_dim=16, num_slots=4, iterations=2)
        patches = torch.randn(3, 25, 32)
        slots, mass, assignment = module(patches)
        self.assertEqual(tuple(slots.shape), (3, 4, 16))
        self.assertEqual(tuple(mass.shape), (3, 4))
        self.assertEqual(tuple(assignment.shape), (3, 4, 25))
        self.assertTrue(torch.allclose(mass.sum(-1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.isfinite(slots).all())
        slots.square().mean().backward()
        self.assertIsNotNone(module.slot_queries.grad)
        self.assertGreater(module.slot_queries.grad.abs().sum().item(), 0.0)

    def test_slot_permutation_does_not_change_set_distance(self):
        q = F.normalize(torch.randn(2, 4, 12), dim=-1)
        r = F.normalize(torch.randn(3, 5, 12), dim=-1)
        qm = torch.softmax(torch.randn(2, 4), dim=-1)
        rm = torch.softmax(torch.randn(3, 5), dim=-1)
        distance, _ = pairwise_local_set_distance(q, qm, r, rm)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_distance, _ = pairwise_local_set_distance(
            q[:, permutation], qm[:, permutation], r, rm
        )
        self.assertTrue(torch.allclose(distance, permuted_distance, atol=1e-5))


class TestSlotOTAndCalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_unbalanced_sinkhorn_is_finite_and_differentiable(self):
        cost = torch.rand(2, 4, 5, requires_grad=True)
        source = torch.softmax(torch.randn(2, 4), dim=-1)
        target = torch.softmax(torch.randn(2, 5), dim=-1)
        distance, plan = unbalanced_sinkhorn(cost, source, target)
        self.assertEqual(tuple(plan.shape), (2, 4, 5))
        self.assertTrue(torch.isfinite(distance).all())
        distance.sum().backward()
        self.assertIsNotNone(cost.grad)

    def test_memory_and_calibration_preserve_shapes(self):
        features = F.normalize(torch.randn(12, 4, 16), dim=-1)
        weights = torch.softmax(torch.randn(12, 4), dim=-1)
        labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        memory, mass = build_class_local_memory(features, weights, labels, 3, 5)
        self.assertEqual(tuple(memory.shape), (3, 5, 16))
        self.assertEqual(tuple(mass.shape), (3, 5))
        self.assertTrue(torch.allclose(mass.sum(-1), torch.ones(3), atol=1e-5))

        novel, novel_mass = weighted_kmeans(
            F.normalize(torch.randn(9, 16), dim=-1),
            torch.ones(9),
            5,
        )
        calibrated, info = calibrate_local_memory(
            novel,
            novel_mass,
            F.normalize(torch.randn(16), dim=-1),
            memory,
            mass,
            F.normalize(torch.randn(3, 16), dim=-1),
            top_k=2,
        )
        self.assertEqual(tuple(calibrated.shape), (5, 16))
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertTrue(torch.allclose(calibrated.norm(dim=-1), torch.ones(5), atol=1e-5))
        self.assertEqual(tuple(info["slot_calibration_alpha"].shape), (5,))

    def test_joint_classifier_uses_global_and_local(self):
        global_query = F.normalize(torch.randn(3, 8), dim=-1)
        local_query = F.normalize(torch.randn(3, 4, 8), dim=-1)
        local_mass = torch.softmax(torch.randn(3, 4), dim=-1)
        global_proto = F.normalize(torch.randn(5, 8), dim=-1)
        local_memory = F.normalize(torch.randn(5, 4, 8), dim=-1)
        memory_mass = torch.softmax(torch.randn(5, 4), dim=-1)
        logits, global_dist, local_dist = joint_prototype_logits(
            global_query, local_query, local_mass,
            global_proto, local_memory, memory_mass,
        )
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertEqual(tuple(global_dist.shape), (3, 5))
        self.assertEqual(tuple(local_dist.shape), (3, 5))
        self.assertTrue(torch.isfinite(logits).all())


class TestFrozenEncoderContract(unittest.TestCase):
    class FakeEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 8)
        def forward_features(self, images):
            pooled = images.mean(dim=(-1, -2))
            cls = self.proj(pooled)
            patches = cls[:, None, :].expand(-1, 4, -1)
            return torch.cat([cls[:, None, :], patches], dim=1)
        def forward_head(self, tokens):
            return tokens[:, 0]

    def test_backbone_frozen_slot_trainable(self):
        class Args:
            slot_dim = 4
            slot_num_slots = 2
            slot_iterations = 1
        model = PromptSlotViT(Args(), encoder=self.FakeEncoder())
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.slot_attention.parameters()))
        output = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output["global_features"].shape), (2, 8))
        self.assertEqual(tuple(output["local_features"].shape), (2, 2, 4))
        model.freeze_for_incremental()
        self.assertTrue(all(not p.requires_grad for p in model.slot_attention.parameters()))


if __name__ == "__main__":
    unittest.main()
