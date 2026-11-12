"""Unit tests for the unified visual-semantic FSCIL components."""
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.vsmr_losses import (
    attention_overlap_loss,
    cross_image_semantic_alignment,
    leave_one_out_global,
    local_leave_one_out_logits,
    local_logits,
    ot_barycenter,
    ot_distance,
    sinkhorn_transport,
)
from models.base.vsmr_resynthesis import (
    build_base_component_bank,
    compose_novel_prototypes,
    query_conditioned_reconstruction,
)
from models.base.vsmr_modules import VisualSemanticDecomposer, VSMRViT


class TestOptimalTransport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_sinkhorn_has_uniform_marginals_and_finite_gradient(self):
        cost = torch.rand(3, 5, 5, requires_grad=True)
        transport = sinkhorn_transport(cost, epsilon=0.08, iterations=60)
        expected = torch.full((3, 5), 0.2)
        self.assertTrue(torch.allclose(transport.sum(-1), expected, atol=2e-3))
        self.assertTrue(torch.allclose(transport.sum(-2), expected, atol=2e-3))
        (transport * cost).sum().backward()
        self.assertTrue(torch.isfinite(cost.grad).all())

    def test_ot_is_permutation_invariant(self):
        a = F.normalize(torch.randn(2, 6, 16), dim=-1)
        b = F.normalize(torch.randn(2, 6, 16), dim=-1)
        permutation_a = torch.tensor([3, 0, 5, 2, 1, 4])
        permutation_b = torch.tensor([2, 5, 1, 4, 0, 3])
        distance = ot_distance(a, b)
        permuted = ot_distance(a[:, permutation_a], b[:, permutation_b])
        self.assertTrue(torch.allclose(distance, permuted, atol=2e-4))

    def test_barycenter_handles_singleton_and_multiple_sets(self):
        one = F.normalize(torch.randn(1, 4, 8), dim=-1)
        self.assertTrue(torch.allclose(ot_barycenter(one), one[0]))
        many = F.normalize(torch.randn(3, 4, 8), dim=-1)
        center = ot_barycenter(many, outer_iterations=2)
        self.assertEqual(center.shape, (4, 8))
        self.assertTrue(torch.isfinite(center).all())


class TestVisualSemanticDecomposition(unittest.TestCase):
    def test_prompt_attention_is_soft_and_differentiable(self):
        torch.manual_seed(11)
        module = VisualSemanticDecomposer(embed_dim=24, num_parts=4)
        cls = F.normalize(torch.randn(3, 24), dim=-1)
        patches = F.normalize(torch.randn(3, 16, 24), dim=-1)
        global_features, parts, attention = module(cls, patches)
        self.assertEqual(global_features.shape, (3, 24))
        self.assertEqual(parts.shape, (3, 4, 24))
        self.assertEqual(attention.shape, (3, 4, 16))
        self.assertTrue(torch.allclose(
            attention.sum(-1), torch.ones(3, 4), atol=1e-6
        ))
        self.assertTrue(torch.all((attention > 0).sum(-1) > 1))
        loss = parts.square().mean() + attention_overlap_loss(attention)
        loss.backward()
        self.assertIsNotNone(module.shared_prompts.grad)


class TestPrototypeAndTransfer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_leave_one_out_excludes_self(self):
        features = F.normalize(torch.randn(6, 12), dim=-1)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        prototypes, inverse, _ = leave_one_out_global(features, labels)
        expected = F.normalize(features[1:3].mean(0), dim=-1)
        self.assertTrue(torch.allclose(prototypes[0], expected, atol=1e-6))
        self.assertEqual(inverse.shape, (6,))

    def test_local_loo_logits_and_full_logits(self):
        parts = F.normalize(torch.randn(8, 4, 12), dim=-1)
        labels = torch.tensor([0] * 4 + [1] * 4)
        logits, classes = local_leave_one_out_logits(parts, labels)
        self.assertEqual(logits.shape, (8, 2))
        self.assertTrue(torch.isfinite(logits).all())
        prototypes = torch.stack([
            ot_barycenter(parts[:4]), ot_barycenter(parts[4:])
        ])
        scores = local_logits(parts[:2], prototypes)
        self.assertEqual(scores.shape, (2, 2))

    def test_cross_class_retrieval_is_componentwise_and_reliability_gated(self):
        base = F.normalize(torch.randn(3, 4, 8), dim=-1)
        bank, owner, part_id = build_base_component_bank(base)
        novel = F.normalize(torch.randn(2, 4, 8), dim=-1)
        composed, alpha, reliability, indices = compose_novel_prototypes(
            novel, bank, top_r=3
        )
        self.assertEqual(composed.shape, novel.shape)
        self.assertEqual(alpha.shape, novel.shape[:2])
        self.assertTrue(torch.all(alpha <= 0.5 + 1e-6))
        self.assertEqual(indices.shape, (2, 4, 3))
        self.assertEqual(owner.shape, (12,))
        self.assertEqual(part_id.shape, (12,))
        self.assertTrue(torch.isfinite(reliability).all())

    def test_query_reconstruction_values_come_from_class_prototypes(self):
        query = F.normalize(torch.randn(2, 4, 8), dim=-1)
        prototypes = F.normalize(torch.randn(3, 4, 8), dim=-1)
        scores, reconstructed, transport = query_conditioned_reconstruction(
            query, prototypes
        )
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(reconstructed.shape, (2, 3, 4, 8))
        self.assertEqual(transport.shape, (2, 3, 4, 4))
        # Reconstructed values equal normalized query-conditioned weighted
        # sums of stored class prototype parts; query values are not injected.
        flat_proto = F.normalize(prototypes, dim=-1)[None].expand(2, -1, -1, -1)
        expected = torch.matmul(
            transport.reshape(-1, 4, 4) * 4.0,
            flat_proto.reshape(-1, 4, 8),
        )
        expected = F.normalize(expected, dim=-1).reshape(2, 3, 4, 8)
        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-5))
        self.assertTrue(torch.isfinite(scores).all())


class TestAlignment(unittest.TestCase):
    def test_cisa_is_finite_and_differentiable(self):
        global_logits = torch.randn(8, 3, requires_grad=True)
        local_logits = torch.randn(8, 3, requires_grad=True)
        labels = torch.tensor([0] * 4 + [1] * 4)
        loss = cross_image_semantic_alignment(
            global_logits, local_logits, labels
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(global_logits.grad)
        self.assertIsNotNone(local_logits.grad)


if __name__ == "__main__":
    unittest.main()
