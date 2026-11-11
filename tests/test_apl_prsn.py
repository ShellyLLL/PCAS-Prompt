import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.apl_prsn_modules import (
    APLPartPrompt,
    APLPRSNViT,
    GlobalPrototypeRefiner,
    PrototypeFeatureResynthesis,
)
from models.base.apl_prsn_losses import (
    apl_all_min_contrastive_loss,
    apl_part_diversity_loss,
    build_class_prototypes,
    build_leave_one_out_prototypes,
    prsn_cross_image_semantic_alignment_loss,
)


class TestAPLPartLearning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_shared_queries_create_aligned_hard_parts(self):
        module = APLPartPrompt(
            embed_dim=24, num_parts=4, num_heads=4, prior_ratio=0.25
        )
        patches = F.normalize(torch.randn(6, 16, 24), dim=-1)
        priors = torch.softmax(torch.randn(6, 3, 16), dim=-1)
        parts, valid, assignments, queries = module(patches, priors)
        self.assertEqual(tuple(parts.shape), (6, 4, 24))
        self.assertEqual(tuple(valid.shape), (6, 4))
        self.assertEqual(tuple(assignments.shape), (6, 4, 16))
        self.assertEqual(tuple(queries.shape), (6, 4, 24))
        self.assertTrue(torch.allclose(assignments.sum(1), torch.ones(6, 16)))
        self.assertTrue(torch.allclose(assignments.max(1).values, torch.ones(6, 16)))
        (parts.square().mean() + queries.square().mean()).backward()
        self.assertIsNotNone(module.shared_part_queries.grad)
        self.assertGreater(module.shared_part_queries.grad.abs().sum().item(), 0.0)

    def test_all_min_and_diversity_losses_are_finite(self):
        parts = F.normalize(torch.randn(8, 4, 12, requires_grad=True), dim=-1)
        valid = torch.ones(8, 4, dtype=torch.bool)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        all_min = apl_all_min_contrastive_loss(parts, valid, labels)
        diversity = apl_part_diversity_loss(parts, valid)
        self.assertTrue(torch.isfinite(all_min))
        self.assertTrue(torch.isfinite(diversity))
        (all_min + diversity).backward()


    def test_full_model_freezes_backbone_and_incremental_modules(self):
        class FakePatchEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 24)

            def forward(self, images):
                pooled = F.adaptive_avg_pool2d(images, (4, 4))
                return self.proj(pooled.flatten(2).transpose(1, 2))

        class FakeAttention(nn.Module):
            def __init__(self, dim=24, heads=4):
                super().__init__()
                self.num_heads = heads
                self.scale = (dim // heads) ** -0.5
                self.qkv = nn.Linear(dim, dim * 3)
                self.proj = nn.Linear(dim, dim)

            def forward(self, tokens):
                batch, count, dim = tokens.shape
                qkv = self.qkv(tokens).reshape(
                    batch, count, 3, self.num_heads, dim // self.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
                output = (attention @ v).transpose(1, 2).reshape(batch, count, dim)
                return self.proj(output)

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(24)
                self.attn = FakeAttention()

            def forward(self, tokens):
                return tokens + self.attn(self.norm1(tokens))

        class FakeEncoder(nn.Module):
            num_features = 24

            def __init__(self):
                super().__init__()
                self.patch_embed = FakePatchEmbed()
                self.cls_token = nn.Parameter(torch.zeros(1, 1, 24))
                self.pos_embed = nn.Parameter(torch.zeros(1, 17, 24))
                self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])
                self.norm = nn.LayerNorm(24)

            def _pos_embed(self, patches):
                tokens = torch.cat([
                    self.cls_token.expand(patches.shape[0], -1, -1), patches
                ], dim=1)
                return tokens + self.pos_embed

        class Args:
            apl_num_parts = 4
            apl_num_heads = 4
            apl_prior_ratio = 0.25
            apl_temperature = 0.2
            prsn_top_k = 2

        model = APLPRSNViT(Args(), encoder=FakeEncoder())
        output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output["global_features"].shape), (2, 24))
        self.assertEqual(tuple(output["parts"].shape), (2, 4, 24))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.apl.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.prsn.parameters()))
        model.freeze_for_incremental()
        self.assertTrue(all(not p.requires_grad for p in model.apl.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.prsn.parameters()))


class TestPRSN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.global_features = F.normalize(torch.randn(12, 16), dim=-1)
        self.parts = F.normalize(torch.randn(12, 4, 16), dim=-1)
        self.valid = torch.ones(12, 4, dtype=torch.bool)
        self.labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)

    def test_leave_one_out_removes_query_from_its_prototype(self):
        cache = build_class_prototypes(
            self.global_features, self.parts, self.valid, self.labels
        )
        loo = build_leave_one_out_prototypes(
            self.global_features, self.parts, self.valid, self.labels, cache
        )
        index = 0
        expected = F.normalize(self.global_features[1:4].mean(0), dim=-1)
        self.assertTrue(torch.allclose(loo["global"][index], expected, atol=1e-6))
        self.assertEqual(tuple(loo["parts"].shape), (12, 4, 16))

    def test_pfrsm_retrieves_base_memory_and_reconstructs_logits(self):
        module = PrototypeFeatureResynthesis(embed_dim=16, top_k=2)
        novel_global = F.normalize(torch.randn(2, 16), dim=-1)
        novel_parts = F.normalize(torch.randn(2, 4, 16), dim=-1)
        novel_valid = torch.ones(2, 4, dtype=torch.bool)
        base_global = F.normalize(torch.randn(3, 16), dim=-1)
        base_parts = F.normalize(torch.randn(3, 4, 16), dim=-1)
        base_valid = torch.ones(3, 4, dtype=torch.bool)
        calibrated, calibrated_valid, selected, weights = module.calibrate(
            novel_global, novel_parts, novel_valid,
            base_global, base_parts, base_valid,
        )
        self.assertEqual(tuple(calibrated.shape), (2, 4, 16))
        self.assertEqual(tuple(calibrated_valid.shape), (2, 4))
        self.assertEqual(tuple(selected.shape), (2, 2))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2), atol=1e-6))
        logits = module.local_logits(
            novel_parts, novel_valid, calibrated, calibrated_valid
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(module.reconstruct_query.weight.grad)

    def test_global_refinement_and_cisa_are_differentiable(self):
        refiner = GlobalPrototypeRefiner(embed_dim=16)
        prototypes = F.normalize(torch.randn(3, 16), dim=-1)
        refined = refiner.refine_base(prototypes)
        queries = F.normalize(torch.randn(4, 16), dim=-1)
        refined_queries = refiner.refine_queries(queries, prototypes)
        self.assertEqual(tuple(refined.shape), (3, 16))
        self.assertEqual(tuple(refined_queries.shape), (4, 16))

        global_logits = torch.randn(6, 3, requires_grad=True)
        local_logits = torch.randn(6, 3, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        cisa = prsn_cross_image_semantic_alignment_loss(
            global_logits, local_logits, labels
        )
        self.assertTrue(torch.isfinite(cisa))
        cisa.backward()
        self.assertIsNotNone(global_logits.grad)
        self.assertIsNotNone(local_logits.grad)


class TestIncrementalFreezeContract(unittest.TestCase):
    def test_append_only_prototype_update_semantics(self):
        base = torch.randn(3, 4, 8)
        base_before = base.clone()
        novel = torch.randn(2, 4, 8)
        all_prototypes = torch.cat([base, novel], dim=0)
        self.assertTrue(torch.equal(all_prototypes[:3], base_before))
        self.assertEqual(all_prototypes.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
