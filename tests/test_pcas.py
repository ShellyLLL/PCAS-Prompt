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


if __name__ == "__main__":
    unittest.main()
