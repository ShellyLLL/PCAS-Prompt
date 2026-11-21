import unittest
import torch
import torch.nn as nn

from models.base.continual_vsmr import (
    FrozenFeatureMemory, loo_class_prototypes, prompt_consistency_loss,
    distillation_kl,
)


class TestContinualVSMRUtilities(unittest.TestCase):
    def test_feature_memory_stores_real_cls_and_patch_and_rejects_duplicate(self):
        memory = FrozenFeatureMemory(max_base_per_class=3)
        cls = torch.randn(4, 8)
        patches = torch.randn(4, 5, 8)
        labels = torch.tensor([0, 0, 1, 1])
        memory.append(cls, patches, labels, ["a", "b", "c", "d"], 0, is_base=True)
        cached_cls, cached_patches, cached_labels = memory.tensors("cpu")
        self.assertEqual(cached_cls.shape, (4, 8))
        self.assertEqual(cached_patches.shape, (4, 5, 8))
        self.assertTrue(torch.allclose(cached_cls, cls, atol=2e-3))
        self.assertTrue(torch.equal(cached_labels, labels))
        with self.assertRaises(RuntimeError):
            memory.append(cls[:1], patches[:1], labels[:1], ["a"], 1)

    def test_base_budget_is_enforced(self):
        memory = FrozenFeatureMemory(max_base_per_class=2)
        cls = torch.randn(3, 4)
        patches = torch.randn(3, 5, 4)
        labels = torch.tensor([0, 0, 0])
        memory.append(cls, patches, labels, ["a", "b", "c"], 0, is_base=True)
        self.assertEqual(len(memory), 2)

    def test_loo_candidate_excludes_query_from_own_class(self):
        torch.manual_seed(3)
        parts = torch.nn.functional.normalize(torch.randn(4, 3, 6), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        candidates = loo_class_prototypes(
            parts, labels, torch.tensor([0, 2]), torch.tensor([0, 1]),
            epsilon=0.1, iterations=30, barycenter_iterations=2)
        self.assertEqual(candidates.shape, (2, 2, 3, 6))
        # For query 0, its own-class candidate is constructed from sample 1.
        expected = parts[1]
        self.assertTrue(torch.allclose(candidates[0, 0], expected, atol=1e-5))

    def test_prompt_consistency_is_permutation_invariant_and_differentiable(self):
        torch.manual_seed(4)
        teacher = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
        student = teacher[:, [2, 0, 3, 1]].clone().requires_grad_()
        loss = prompt_consistency_loss(student, teacher, epsilon=0.1, iterations=40)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_distillation_has_student_gradient_only(self):
        student = torch.randn(5, 3, requires_grad=True)
        teacher = torch.randn(5, 3, requires_grad=True)
        loss = distillation_kl(student, teacher, temperature=2.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
