"""Unit tests for the unified visual-semantic FSCIL components."""
import unittest

import numpy as np
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.base.vsmr_losses import (
    attention_overlap_loss,
    component_diversity_loss,
    fixed_index_component_logits,
    cross_image_semantic_alignment,
    leave_one_out_global,
    local_leave_one_out_logits,
    local_logits,
    ot_barycenter,
    ot_distance,
    prompt_orthogonality_loss,
    sinkhorn_rectangular_transport,
    sinkhorn_transport,
)
from models.base.vsmr_resynthesis import (
    build_base_component_bank,
    component_support_variance,
    compose_novel_prototypes,
    local_leave_one_out_query_logits,
    query_conditioned_fixed_index_reconstruction,
    query_conditioned_reconstruction,
)
from models.base.vsmr_modules import (
    GlobalLocalFusion,
    QueryConditionedReconstructor,
    ReliabilityGate,
)
from models import model_My
from models.base.vsmr_modules import VisualSemanticDecomposer, VSMRViT
from models.base.vsmr_trainer import ClassBalancedBatchSampler, VSMRFSCILTrainer


class TestFixedIndexMatching(unittest.TestCase):
    def test_fixed_index_logits_support_batch_and_class_axes(self):
        torch.manual_seed(11)
        query = torch.randn(8, 6, 32)
        prototypes = torch.randn(100, 6, 32)
        logits = fixed_index_component_logits(query, prototypes, 16.0)
        self.assertEqual(logits.shape, (8, 100))
        q, p = F.normalize(query, dim=-1), F.normalize(prototypes, dim=-1)
        expected = torch.einsum("bkd,ckd->bc", q, p) * (16.0 / 6)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_fixed_index_matching_rejects_incompatible_parts(self):
        with self.assertRaises(ValueError):
            fixed_index_component_logits(torch.randn(2, 6, 8), torch.randn(3, 5, 8))

    def test_fixed_index_matching_is_differentiable(self):
        query = torch.randn(4, 6, 8, requires_grad=True)
        prototypes = torch.randn(7, 6, 8, requires_grad=True)
        fixed_index_component_logits(query, prototypes).square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(prototypes.grad).all())


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


class TestBalancedPromptAssignment(unittest.TestCase):
    def test_rectangular_sinkhorn_enforces_both_marginals_and_gradients(self):
        torch.manual_seed(23)
        logits = torch.randn(3, 4, 19, requires_grad=True)
        transport = sinkhorn_rectangular_transport(
            logits, epsilon=0.7, iterations=80
        )
        self.assertEqual(transport.shape, (3, 4, 19))
        self.assertTrue(torch.allclose(
            transport.sum(-1), torch.full((3, 4), 0.25), atol=2e-3
        ))
        self.assertTrue(torch.allclose(
            transport.sum(-2), torch.full((3, 19), 1.0 / 19), atol=2e-3
        ))
        (transport * logits).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prompt_orthogonality_penalizes_collapsed_queries(self):
        collapsed = torch.ones(4, 12, requires_grad=True)
        orthogonal = torch.eye(4, 12)
        collapsed_loss = prompt_orthogonality_loss(collapsed)
        orthogonal_loss = prompt_orthogonality_loss(orthogonal)
        self.assertGreater(float(collapsed_loss), float(orthogonal_loss))
        collapsed_loss.backward()
        self.assertIsNotNone(collapsed.grad)
        self.assertTrue(torch.isfinite(collapsed.grad).all())

    def test_component_diversity_penalizes_duplicate_components(self):
        collapsed = torch.ones(2, 4, 12, requires_grad=True)
        diverse = torch.stack([torch.eye(4, 12), -torch.eye(4, 12)])
        collapsed_loss = component_diversity_loss(collapsed, similarity_margin=0.5)
        diverse_loss = component_diversity_loss(diverse, similarity_margin=0.5)
        self.assertGreater(float(collapsed_loss), float(diverse_loss))
        collapsed_loss.backward()
        self.assertIsNotNone(collapsed.grad)
        self.assertTrue(torch.isfinite(collapsed.grad).all())


class TestGlobalLocalFusion(unittest.TestCase):
    def test_fusion_is_learned_convex_and_differentiable(self):
        fusion = GlobalLocalFusion(global_init=0.5, local_init=0.5)
        global_scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        local_scores = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
        fused = fusion(global_scores, local_scores)
        self.assertTrue(torch.allclose(fused, 0.5 * global_scores + 0.5 * local_scores))
        self.assertGreater(float(fusion.alpha.item()), 0.0)
        self.assertLess(float(fusion.alpha.item()), 1.0)
        targets = torch.tensor([0, 1])
        F.cross_entropy(fusion(global_scores, local_scores), targets).backward()
        self.assertIsNotNone(fusion.logit.grad)
        self.assertGreater(float(fusion.logit.grad.abs().item()), 0.0)


class TestVisualSemanticDecomposition(unittest.TestCase):
    def test_vsmr_integrates_with_repository_vit_and_freeze_contract(self):
        # Use the repository's actual ViT implementation with a small shape;
        # pretrained weights are intentionally disabled in this unit test.
        encoder = model_My.VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=0,
            global_pool="token",
            embed_dim=24,
            depth=1,
            num_heads=3,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        encoder.default_cfg = {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic",
            "crop_pct": 0.9,
        }
        args = SimpleNamespace(
            scratch=False,
            vsmr_num_parts=4,
            vsmr_prompt_temperature=1.0,
        )
        model = VSMRViT(args, encoder=encoder)
        self.assertEqual(model.backbone.input_mean, (0.5, 0.5, 0.5))
        self.assertEqual(model.backbone.input_std, (0.5, 0.5, 0.5))
        self.assertEqual(model.backbone.input_interpolation, "bicubic")
        self.assertEqual(model.backbone.input_crop_pct, 0.9)
        output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output["global_features"].shape, (2, 24))
        self.assertEqual(output["parts"].shape, (2, 4, 24))
        self.assertEqual(output["attention"].shape, (2, 4, 196))
        self.assertEqual(output["patch_features"].shape, (2, 196, 24))
        self.assertEqual(output["prompt_queries"].shape, (4, 24))
        self.assertEqual(output["assignment"].shape, (2, 4, 196))
        self.assertTrue(torch.allclose(
            output["attention"].sum(-1), torch.ones(2, 4), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            output["assignment"].sum(-1), torch.full((2, 4), 0.25), atol=2e-3
        ))
        self.assertTrue(torch.allclose(
            output["assignment"].sum(-2), torch.full((2, 196), 1.0 / 196), atol=2e-3
        ))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.decomposer.parameters()))

        model.train()
        self.assertFalse(model.backbone.encoder.training)
        model.freeze_for_incremental()
        model.train()
        self.assertFalse(model.decomposer.training)
        self.assertFalse(model.reliability_gate.training)
        self.assertFalse(model.reconstructor.training)
        self.assertFalse(model.fusion.training)
        self.assertTrue(all(not p.requires_grad for p in model.decomposer.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.reliability_gate.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.reconstructor.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.fusion.parameters()))

    def test_prompt_attention_is_soft_and_differentiable(self):
        torch.manual_seed(11)
        module = VisualSemanticDecomposer(embed_dim=24, num_parts=4)
        cls = F.normalize(torch.randn(3, 24), dim=-1)
        patches = F.normalize(torch.randn(3, 16, 24), dim=-1)
        global_features, parts, attention, queries, assignment = module(
            cls, patches, return_aux=True
        )
        self.assertEqual(global_features.shape, (3, 24))
        self.assertEqual(parts.shape, (3, 4, 24))
        self.assertEqual(attention.shape, (3, 4, 16))
        self.assertEqual(queries.shape, (4, 24))
        self.assertEqual(assignment.shape, (3, 4, 16))
        self.assertTrue(torch.allclose(
            attention.sum(-1), torch.ones(3, 4), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            assignment.sum(-1), torch.full((3, 4), 0.25), atol=2e-3
        ))
        self.assertTrue(torch.allclose(
            assignment.sum(-2), torch.full((3, 16), 1.0 / 16), atol=2e-3
        ))
        self.assertTrue(torch.all((attention > 0).sum(-1) > 1))
        loss = (
            parts.square().mean()
            + attention_overlap_loss(attention)
            + prompt_orthogonality_loss(queries)
            + component_diversity_loss(parts)
        )
        loss.backward()
        self.assertIsNotNone(module.shared_prompts.grad)
        self.assertTrue(torch.isfinite(module.shared_prompts.grad).all())


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
        parts = F.normalize(torch.randn(8, 4, 12), dim=-1).requires_grad_()
        labels = torch.tensor([0] * 4 + [1] * 4)
        logits, classes = local_leave_one_out_logits(parts, labels)
        self.assertEqual(logits.shape, (8, 2))
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits, torch.tensor([0] * 4 + [1] * 4)).backward()
        self.assertIsNotNone(parts.grad)
        self.assertTrue(torch.isfinite(parts.grad).all())
        prototypes = torch.stack([
            ot_barycenter(parts[:4]), ot_barycenter(parts[4:])
        ])
        scores = local_logits(parts[:2], prototypes)
        self.assertEqual(scores.shape, (2, 2))

    def test_cross_class_retrieval_is_componentwise_and_reliability_gated(self):
        base = F.normalize(torch.randn(3, 4, 8), dim=-1)
        bank, owner, part_id = build_base_component_bank(base)
        novel = F.normalize(torch.randn(2, 4, 8), dim=-1)
        gate = ReliabilityGate(embed_dim=8, hidden_dim=16)
        support_variance = torch.rand(2, 4)
        composed, alpha, reliability, indices, weights = compose_novel_prototypes(
            novel,
            bank,
            gate,
            base_owner=owner,
            top_r=3,
            support_variance=support_variance,
            shots=torch.tensor([1, 3]),
        )
        self.assertEqual(composed.shape, novel.shape)
        self.assertEqual(alpha.shape, novel.shape[:2])
        self.assertTrue(torch.all(alpha <= 0.5 + 1e-6))
        self.assertEqual(indices.shape, (2, 4, 3))
        self.assertEqual(weights.shape, (2, 4, 3))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 4), atol=1e-6))
        self.assertEqual(owner.shape, (12,))
        self.assertEqual(part_id.shape, (12,))
        self.assertTrue(torch.isfinite(reliability).all())
        selected_owners = owner[indices]
        for row in selected_owners.reshape(-1, selected_owners.shape[-1]):
            self.assertEqual(torch.unique(row).numel(), row.numel())

        target = torch.randn_like(composed)
        (composed * target).sum().add(alpha.mean()).backward()
        gate_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in gate.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gate_grad, 0.0)

    def test_support_variance_is_ot_aligned_and_finite(self):
        prototype = F.normalize(torch.randn(4, 8), dim=-1)
        supports = F.normalize(torch.randn(3, 4, 8), dim=-1)
        variance = component_support_variance(supports, prototype, 0.1, 40)
        self.assertEqual(variance.shape, (4,))
        self.assertTrue(torch.isfinite(variance).all())
        singleton = component_support_variance(supports[:1], prototype)
        self.assertTrue(torch.equal(singleton, torch.zeros_like(singleton)))

    def test_query_patch_values_condition_reconstructed_class_parts(self):
        query = F.normalize(torch.randn(2, 4, 8), dim=-1)
        patches = F.normalize(torch.randn(2, 12, 8), dim=-1).requires_grad_()
        prototypes = F.normalize(torch.randn(3, 4, 8), dim=-1)
        memory_before = prototypes.clone()
        reconstructor = QueryConditionedReconstructor(embed_dim=8)
        scores, reconstructed, transport, attention = query_conditioned_reconstruction(
            query, patches, prototypes, reconstructor, 0.08, 50, 16.0
        )
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(reconstructed.shape, (2, 3, 4, 8))
        self.assertEqual(transport.shape, (2, 3, 4, 4))
        self.assertEqual(attention.shape, (2, 3, 4, 12))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(prototypes, memory_before))

        changed_patches = patches.detach().clone()
        changed_patches[:, :6] = -changed_patches[:, :6]
        changed, _, _, _ = query_conditioned_reconstruction(
            query, changed_patches, prototypes, reconstructor, 0.08, 50, 16.0
        )
        self.assertFalse(torch.allclose(scores, changed, atol=1e-6))
        (-scores.mean()).backward()
        self.assertIsNotNone(patches.grad)
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertIsNotNone(reconstructor.value_projection.weight.grad)
        self.assertGreater(
            float(reconstructor.value_projection.weight.grad.abs().sum()), 0.0
        )

    def test_fixed_index_ablation_uses_same_query_reconstruction(self):
        query = F.normalize(torch.randn(2, 4, 8), dim=-1).requires_grad_()
        patches = F.normalize(torch.randn(2, 9, 8), dim=-1).requires_grad_()
        prototypes = F.normalize(torch.randn(3, 4, 8), dim=-1)
        reconstructor = QueryConditionedReconstructor(embed_dim=8)
        scores, reconstructed, attention = query_conditioned_fixed_index_reconstruction(
            query, patches, prototypes, reconstructor, scale=16.0
        )
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(reconstructed.shape, (2, 3, 4, 8))
        self.assertEqual(attention.shape, (2, 3, 4, 9))
        self.assertTrue(torch.isfinite(scores).all())
        (-scores.mean()).backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(patches.grad)
        self.assertIsNotNone(reconstructor.value_projection.weight.grad)

    def test_local_loo_uses_same_query_reconstruction_path(self):
        parts = F.normalize(torch.randn(6, 4, 8), dim=-1).requires_grad_()
        patches = F.normalize(torch.randn(6, 10, 8), dim=-1).requires_grad_()
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        reconstructor = QueryConditionedReconstructor(embed_dim=8)
        logits, classes = local_leave_one_out_query_logits(
            parts, patches, labels, reconstructor, 0.08, 40, 16.0, 2
        )
        self.assertEqual(logits.shape, (6, 2))
        self.assertEqual(classes.tolist(), [0, 1])
        F.cross_entropy(logits, torch.tensor([0, 0, 0, 1, 1, 1])).backward()
        self.assertIsNotNone(parts.grad)
        self.assertIsNotNone(patches.grad)
        self.assertIsNotNone(reconstructor.query_projection.weight.grad)

    def test_pseudo_incremental_episode_trains_gate_and_reconstructor(self):
        trainer = VSMRFSCILTrainer.__new__(VSMRFSCILTrainer)
        trainer.args = SimpleNamespace(
            vsmr_episode_novel_fraction=0.34,
            vsmr_episode_max_shot=2,
            vsmr_barycenter_iterations=2,
            vsmr_transfer_top_r=2,
            vsmr_transfer_alpha_max=0.5,
            vsmr_transfer_temperature=0.1,
            vsmr_global_weight=0.5,
            vsmr_local_weight_eval=0.5,
        )
        gate = ReliabilityGate(embed_dim=8, hidden_dim=16)
        reconstructor = QueryConditionedReconstructor(embed_dim=8)
        trainer.model = SimpleNamespace(
            reliability_gate=gate,
            reconstructor=reconstructor,
            fusion=GlobalLocalFusion(),
        )
        torch.manual_seed(31)
        features = F.normalize(torch.randn(12, 8), dim=-1).requires_grad_()
        parts = F.normalize(torch.randn(12, 4, 8), dim=-1).requires_grad_()
        patches = F.normalize(torch.randn(12, 10, 8), dim=-1).requires_grad_()
        labels = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
        global_loss, local_loss, joint_loss = trainer._pseudo_incremental_episode(
            features, parts, patches, labels, 0.08, 30, 16.0
        )
        total = global_loss + local_loss + joint_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in gate.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in reconstructor.parameters()
        ))


class TestPreprocessingContract(unittest.TestCase):
    def test_transform_uses_frozen_backbone_config_for_every_dataset(self):
        trainer = VSMRFSCILTrainer.__new__(VSMRFSCILTrainer)
        trainer.args = SimpleNamespace(dataset="cifar100")
        trainer.model = SimpleNamespace(backbone=SimpleNamespace(
            input_mean=(0.5, 0.5, 0.5),
            input_std=(0.5, 0.5, 0.5),
            input_interpolation="bicubic",
            input_crop_pct=0.9,
        ))
        transform = trainer._transform()
        resize, crop, to_tensor, normalize = transform.transforms
        self.assertEqual(resize.size, 249)
        self.assertEqual(resize.interpolation, transforms.InterpolationMode.BICUBIC)
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(normalize.mean, (0.5, 0.5, 0.5))
        self.assertEqual(normalize.std, (0.5, 0.5, 0.5))

        training_transform = trainer._training_transform()
        train_normalize = training_transform.transforms[-1]
        self.assertEqual(
            training_transform.transforms[0].interpolation,
            transforms.InterpolationMode.BICUBIC,
        )
        self.assertEqual(train_normalize.mean, (0.5, 0.5, 0.5))
        self.assertEqual(train_normalize.std, (0.5, 0.5, 0.5))


class TestTrainingDataPipeline(unittest.TestCase):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            self.transform = None

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.tensor([float(index)]), self.targets[index]

    def test_class_balanced_sampler_and_dataloader_contract(self):
        dataset = self.TinyDataset()
        sampler = ClassBalancedBatchSampler(
            dataset.targets, classes_per_batch=3, samples_per_class=2, seed=5
        )
        trainer = VSMRFSCILTrainer.__new__(VSMRFSCILTrainer)
        trainer.args = SimpleNamespace(num_workers=0)
        trainer.device = torch.device("cpu")
        loader = trainer._loader(dataset, batch_size=1, batch_sampler=sampler)
        images, labels = next(iter(loader))
        self.assertEqual(images.shape[0], 6)
        self.assertEqual(torch.bincount(labels, minlength=3).tolist(), [2, 2, 2])

    def test_exact_shot_validation_rejects_wrong_counts(self):
        from models.base.vsmr_trainer import validate_exact_shot
        validate_exact_shot(np.array([3, 3, 4, 4]), np.array([3, 4]), 2, 1)
        with self.assertRaises(RuntimeError):
            validate_exact_shot(np.array([3, 4, 4]), np.array([3, 4]), 2, 1)


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
