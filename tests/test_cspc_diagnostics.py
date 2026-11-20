import unittest

import torch
import torch.nn.functional as F

from utils.cspc_diagnostics import summarize_classifier, summarize_transfer


class CSPCDiagnosticsTests(unittest.TestCase):
    def test_classifier_summary_reports_sample_macro_and_confusions(self):
        features = F.normalize(
            torch.tensor([
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]),
            dim=-1,
        )
        prototypes = F.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            dim=-1,
        )
        labels = torch.tensor([0, 1, 1, 1])
        result = summarize_classifier(
            features, labels, prototypes, [0, 1], ["zero", "one"]
        )

        self.assertAlmostEqual(result["sample_accuracy"], 75.0, places=5)
        self.assertAlmostEqual(result["class_macro_accuracy"], 75.0, places=5)
        self.assertEqual(result["per_class"]["1"]["top_confusions"][0]["class_id"], 0)
        self.assertIn("mean_cosine_margin", result["per_class"]["0"])

    def test_classifier_summary_rejects_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "unseen class IDs"):
            summarize_classifier(
                torch.eye(2),
                torch.tensor([0, 2]),
                torch.eye(2),
                [0, 1],
                ["zero", "one"],
            )

    def test_transfer_summary_reports_top_base_classes_and_shift(self):
        raw = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
        calibrated = F.normalize(torch.tensor([0.8, 0.6]), dim=0)
        diagnostics = {
            "weights": torch.tensor([0.7, 0.2, 0.1]),
            "transfer_ratio": torch.tensor(0.25),
            "entropy": torch.tensor(0.5),
            "margin": torch.tensor(0.5),
            "intra_uncertainty": torch.tensor(0.1),
            "weighted_base_norm": torch.tensor(0.8),
        }
        result = summarize_transfer(
            3, "new class", diagnostics, raw, calibrated,
            [0, 1, 2], ["base zero", "base one", "base two"], top_k=2,
        )
        self.assertEqual(
            [item["class_id"] for item in result["top_base_classes"]],
            [0, 1],
        )
        self.assertAlmostEqual(result["transfer_ratio"], 0.25)
        self.assertGreater(result["prototype_angular_shift_degrees"], 0.0)
        self.assertLess(result["prototype_angular_shift_degrees"], 90.0)


if __name__ == "__main__":
    unittest.main()
