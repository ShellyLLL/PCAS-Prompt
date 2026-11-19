import math
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ftfy  # noqa: F401
except ImportError:
    stub = types.ModuleType("ftfy")
    stub.fix_text = lambda text: text
    sys.modules["ftfy"] = stub

from models.cspc import CSPCModel, PrototypeTransfer
from engine.cspc_engine import CSPCRunner


class PrototypeTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 16
        self.base_v = F.normalize(torch.randn(12, self.dim), dim=-1)
        self.base_t = F.normalize(torch.randn(12, self.dim), dim=-1)
        self.query = F.normalize(torch.randn(self.dim), dim=-1)
        self.current = F.normalize(torch.randn(self.dim), dim=-1)
        self.support = F.normalize(torch.randn(4, self.dim), dim=-1)
        self.module = PrototypeTransfer(0.1)

    def call_module(self, **kw):
        args = dict(query_text=self.query, current_proto=self.current,
                    support_features=self.support, base_text_embeddings=self.base_t,
                    base_prototypes=self.base_v)
        args.update(kw)
        return self.module(**args)

    def test_v21_initialization(self):
        self.assertTrue(torch.allclose(self.module.scorer_weight, torch.full((4,), .25)))
        self.assertAlmostEqual(float(torch.sigmoid(self.module.gate_bias)), .1, places=6)
        self.assertTrue(torch.equal(self.module.gate_weight, torch.zeros(8)))

    def test_calibration_output_and_attention_distribution(self):
        out, info = self.call_module()
        self.assertEqual(tuple(out.shape), (self.dim,))
        self.assertAlmostEqual(float(out.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(info["weights"].sum()), 1.0, places=6)
        self.assertEqual(tuple(info["gate_features"].shape), (8,))
        self.assertTrue(0.0 <= float(info["transfer_ratio"]) <= 1.0)

    def test_gradient_reaches_scorer_gate_and_inputs(self):
        query = self.query.clone().requires_grad_(True)
        current = self.current.clone().requires_grad_(True)
        out, _ = self.call_module(query_text=query, current_proto=current)
        (out * F.normalize(torch.randn_like(out), dim=-1)).sum().backward()
        for p in (self.module.scorer_weight, self.module.gate_weight, self.module.gate_bias):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(current.grad)

    def test_invalid_shapes_devices_and_empty_support_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            self.call_module(base_text_embeddings=self.base_t[:-1])
        with self.assertRaisesRegex(ValueError, "same device"):
            self.call_module(base_prototypes=self.base_v.to("meta"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.call_module(support_features=self.support[:0])

    def test_single_base_class_is_supported(self):
        out, info = self.call_module(base_text_embeddings=self.base_t[:1],
                                     base_prototypes=self.base_v[:1])
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(info["weights"].shape), (1,))
        self.assertAlmostEqual(float(info["margin"]), 0.0, places=6)


class EngineHelperTests(unittest.TestCase):
    def test_local_target_mapping_and_unknown_rejection(self):
        labels = torch.tensor([7, 5, 6, 5])
        self.assertEqual(CSPCRunner._local_targets(labels, [5, 6, 7]).tolist(), [2, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, "does not belong"):
            CSPCRunner._local_targets(torch.tensor([5, 9]), [5, 6])

    def test_prototype_builder(self):
        features = F.normalize(torch.randn(8, 12), dim=-1)
        labels = torch.tensor([3, 3, 3, 3, 4, 4, 4, 4])
        prototypes, supports = CSPCRunner._build_prototypes(features, labels, [3, 4])
        self.assertEqual(tuple(prototypes.shape), (2, 12))
        self.assertEqual([len(x) for x in supports], [4, 4])
        self.assertTrue(torch.allclose(prototypes.norm(dim=-1), torch.ones(2), atol=1e-5))
        with self.assertRaisesRegex(ValueError, "no training samples"):
            CSPCRunner._build_prototypes(features[:2], torch.tensor([3, 3]), [3, 4])


class IdentityTransformer(nn.Module):
    def forward(self, x):
        return x


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(32, 4)
        self.positional_embedding = nn.Parameter(torch.zeros(8, 4))
        self.transformer = IdentityTransformer()
        self.ln_final = nn.LayerNorm(4)
        self.text_projection = nn.Parameter(torch.eye(4))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.visual = nn.Module()
        self.visual.conv1 = nn.Conv2d(3, 4, 1)
        self.dtype = torch.float32


class PromptEncodingTests(unittest.TestCase):
    def cfg(self):
        return types.SimpleNamespace(TRAINER=types.SimpleNamespace(CSPC=types.SimpleNamespace(
            N_CTX=2, TEXT_CHUNK_SIZE=8, PREC="fp32",
            PROMPT_INIT_TEXT="one two", SCORER_TAU=.1
        )))

    def test_prompt_insertion_eot_offset_and_gradient(self):
        fake = FakeCLIP()

        def tokenize(texts, context_length=77, truncate=False):
            return torch.tensor([[1, 4, 5, 31, 0, 0]], dtype=torch.long).repeat(len(texts), 1)

        with mock.patch("models.cspc.load_clip_to_cpu", return_value=fake), \
             mock.patch("models.cspc.clip.tokenize", side_effect=tokenize):
            model = CSPCModel(self.cfg(), torch.device("cpu"))
            output = model._encode_text_chunk(["class description"])
            self.assertEqual(tuple(output.shape), (1, 4))
            output.sum().backward()
            self.assertIsNotNone(model.prompt.grad)
            self.assertTrue(torch.isfinite(model.prompt.grad).all())

    def test_frozen_encoders_and_transfer_device(self):
        with mock.patch("models.cspc.load_clip_to_cpu", return_value=FakeCLIP()), \
             mock.patch("models.cspc.clip.tokenize", return_value=torch.tensor([[1, 4, 5, 31, 0, 0]])):
            model = CSPCModel(self.cfg(), torch.device("cpu"))
        self.assertTrue(all(not p.requires_grad for p in model.clip_model.parameters()))
        self.assertTrue(model.prompt.requires_grad)
        self.assertEqual(model.transfer.scorer_weight.device, model.prompt.device)


if __name__ == "__main__":
    unittest.main()
