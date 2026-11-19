import unittest

import torch
import torch.nn.functional as F

from models.cspc import PrototypeTransfer


class PrototypeTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dimension = 16
        self.base_prototypes = F.normalize(torch.randn(12, self.dimension), dim=-1)
        self.base_text = F.normalize(torch.randn(12, self.dimension), dim=-1)
        self.query_text = F.normalize(torch.randn(self.dimension), dim=-1)
        self.current_proto = F.normalize(torch.randn(self.dimension), dim=-1)
        self.support = F.normalize(torch.randn(4, self.dimension), dim=-1)
        self.transfer = PrototypeTransfer(temperature=0.1)

    def test_initialization_matches_v21(self):
        self.assertTrue(
            torch.allclose(
                self.transfer.scorer_weight,
                torch.full((4,), 0.25),
            )
        )
        initial_gate = torch.sigmoid(self.transfer.gate_bias)
        self.assertAlmostEqual(float(initial_gate.item()), 0.1, places=6)
        self.assertTrue(torch.equal(self.transfer.gate_weight, torch.zeros(8)))

    def test_calibration_is_normalized_and_probabilities_sum_to_one(self):
        calibrated, diagnostics = self.transfer(
            self.query_text,
            self.current_proto,
            self.support,
            self.base_text,
            self.base_prototypes,
        )
        self.assertEqual(tuple(calibrated.shape), (self.dimension,))
        self.assertAlmostEqual(float(calibrated.norm().item()), 1.0, places=5)
        self.assertAlmostEqual(
            float(diagnostics["weights"].sum().item()), 1.0, places=6
        )
        self.assertEqual(tuple(diagnostics["gate_features"].shape), (8,))
        self.assertGreaterEqual(float(diagnostics["transfer_ratio"].item()), 0.0)
        self.assertLessEqual(float(diagnostics["transfer_ratio"].item()), 1.0)

    def test_gradients_reach_scorer_and_gate(self):
        query_text = self.query_text.clone().requires_grad_(True)
        current_proto = self.current_proto.clone().requires_grad_(True)
        calibrated, _ = self.transfer(
            query_text,
            current_proto,
            self.support,
            self.base_text,
            self.base_prototypes,
        )
        probe = F.normalize(torch.randn_like(calibrated), dim=-1)
        loss = torch.dot(calibrated, probe)
        loss.backward()

        self.assertIsNotNone(self.transfer.scorer_weight.grad)
        self.assertIsNotNone(self.transfer.gate_weight.grad)
        self.assertIsNotNone(self.transfer.gate_bias.grad)
        self.assertIsNotNone(query_text.grad)
        self.assertIsNotNone(current_proto.grad)
        self.assertTrue(torch.isfinite(self.transfer.scorer_weight.grad).all())
        self.assertTrue(torch.isfinite(self.transfer.gate_weight.grad).all())


if __name__ == "__main__":
    unittest.main()
