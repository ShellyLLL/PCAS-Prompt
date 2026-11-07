import unittest

import torch
import torch.nn.functional as F

from models.base.pcas_losses import (
    affinity_drift,
    build_prototype_cache,
    leave_one_out_prototype_loss,
    prototype_affinity_consistency,
    sample_affinity_consistency,
    teen_calibrate,
)


class PCASLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.teacher = F.normalize(torch.randn(8, 6), dim=-1)
        self.student = F.normalize(
            (self.teacher + 0.05 * torch.randn_like(self.teacher)).requires_grad_(),
            dim=-1,
        )
        self.student.retain_grad()
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    def test_prototype_cache_and_leave_one_out_loss(self):
        cache = build_prototype_cache(self.student.detach(), self.labels, 4)
        indices = torch.tensor([0, 2, 4, 6])
        loss = leave_one_out_prototype_loss(
            self.student[indices],
            self.labels[indices],
            indices,
            cache,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.student.grad)

    def test_sample_affinity_consistency_is_finite_and_differentiable(self):
        teacher_cache = build_prototype_cache(self.teacher, self.labels, 4)
        student_cache = build_prototype_cache(self.student.detach(), self.labels, 4)
        loss = sample_affinity_consistency(
            self.teacher,
            self.student,
            self.labels,
            teacher_cache["prototypes"],
            student_cache["prototypes"],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.student.grad)

    def test_prototype_affinity_consistency_is_finite_and_differentiable(self):
        loss = prototype_affinity_consistency(
            self.teacher, self.student, self.labels
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.student.grad)

    def test_teen_calibration_preserves_unit_norm(self):
        novel = F.normalize(torch.randn(3, 6), dim=-1)
        base = F.normalize(torch.randn(4, 6), dim=-1)
        calibrated, weights = teen_calibrate(novel, base, alpha=0.5, tau=16.0)
        self.assertEqual(tuple(calibrated.shape), (3, 6))
        self.assertEqual(tuple(weights.shape), (3, 4))
        self.assertTrue(torch.allclose(
            calibrated.norm(dim=-1), torch.ones(3), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            weights.sum(dim=-1), torch.ones(3), atol=1e-5
        ))

    def test_affinity_drift_is_zero_for_identical_spaces(self):
        cache = build_prototype_cache(self.teacher, self.labels, 4)
        drift = affinity_drift(
            self.teacher,
            self.teacher,
            self.labels,
            cache["prototypes"],
            cache["prototypes"],
            base_class_count=4,
        )
        self.assertTrue(torch.isfinite(drift))
        self.assertAlmostEqual(float(drift), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
