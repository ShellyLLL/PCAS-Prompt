import json
import math
import os
import tempfile
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


class CSPCConfigurationValidationTests(unittest.TestCase):
    @staticmethod
    def make_runner(inc_shot=5):
        runner = CSPCRunner.__new__(CSPCRunner)
        runner.device = torch.device("cpu")
        runner.class_names = ["class zero", "class one", "class two"]
        runner.base_class_ids = [0, 1]
        runner.data_manager = types.SimpleNamespace(
            class_index_in_task=[[0, 1], [2]],
            num_total_classes=3,
        )
        runner.cfg = types.SimpleNamespace(
            DATASET=types.SimpleNamespace(
                NUM_CLASSES=3,
                NUM_INIT_CLS=2,
                NUM_INC_CLS=1,
                NUM_BASE_SHOT=-1,
                NUM_INC_SHOT=inc_shot,
            ),
            TRAINER=types.SimpleNamespace(
                CSPC=types.SimpleNamespace(
                    BASE_EPOCHS=2,
                    INC_EPOCHS=2,
                    TEXT_CHUNK_SIZE=8,
                    N_CTX=2,
                    GRAD_CLIP=1.0,
                    SCORER_TAU=0.1,
                    BASE_LR=0.001,
                    INC_PROMPT_LR=0.001,
                    INC_SCORER_LR=0.001,
                    INC_GATE_LR=0.001,
                    WEIGHT_DECAY=0.0,
                )
            ),
            DATALOADER=types.SimpleNamespace(
                TRAIN=types.SimpleNamespace(BATCH_SIZE_BASE=4, BATCH_SIZE_INC=4),
                TEST=types.SimpleNamespace(BATCH_SIZE=4),
                NUM_WORKERS=0,
            ),
        )
        return runner

    def test_valid_few_shot_configuration_passes(self):
        self.make_runner()._validate_session_partition()

    def test_zero_incremental_shot_is_rejected_before_training(self):
        runner = self.make_runner(inc_shot=0)
        with self.assertRaisesRegex(ValueError, "NUM_INC_SHOT must be positive"):
            runner._validate_session_partition()


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


class RecordingTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.supports = []

    def forward(self, query_text, current_proto, support_features,
                base_text_embeddings, base_prototypes):
        self.supports.append(support_features.detach().clone())
        return F.normalize(current_proto + 0.01 * query_text, dim=-1), {}


class LOOTests(unittest.TestCase):
    def test_loo_excludes_the_held_out_sample_from_its_own_class_prototype(self):
        runner = CSPCRunner.__new__(CSPCRunner)
        transfer = RecordingTransfer()
        runner.model = types.SimpleNamespace(
            transfer=transfer, logit_scale=torch.tensor(10.0)
        )
        runner.base_text_embeddings = F.normalize(torch.randn(2, 8), dim=-1)
        runner.base_prototypes = F.normalize(torch.randn(2, 8), dim=-1)

        # Each support image is an orthogonal basis vector, making leakage
        # directly observable in the support tensors received by the scorer.
        features = torch.eye(8)[:4]
        labels = torch.tensor([0, 0, 1, 1])
        class_ids = [0, 1]
        current_text = F.normalize(torch.randn(2, 8), dim=-1)
        old_prototypes = F.normalize(torch.randn(3, 8), dim=-1)
        loss = runner._loo_loss(
            features, labels, class_ids, current_text, old_prototypes
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(transfer.supports), 8)
        # For query e0, call 0 builds class-0's prototype and must contain
        # only e1, never the held-out e0 query.
        self.assertEqual(tuple(transfer.supports[0].shape), (1, 8))
        self.assertTrue(torch.equal(transfer.supports[0][0], features[1]))
        self.assertFalse(bool((transfer.supports[0] == features[0]).all(dim=1).any()))
        # For query e2, call 5 builds class-1's prototype and must contain e3.
        self.assertEqual(tuple(transfer.supports[5].shape), (1, 8))
        self.assertTrue(torch.equal(transfer.supports[5][0], features[3]))


class DescriptionAggregationTests(unittest.TestCase):
    def test_all_class_descriptions_contribute_to_one_normalized_embedding(self):
        runner = CSPCRunner.__new__(CSPCRunner)
        runner.class_descriptions = [["description A", "description B"], ["description C"]]

        class TextModel:
            def encode_text(self, texts):
                vectors = {
                    "description A": torch.tensor([1.0, 0.0, 0.0]),
                    "description B": torch.tensor([0.0, 1.0, 0.0]),
                    "description C": torch.tensor([0.0, 0.0, 1.0]),
                }
                return torch.stack([vectors[text] for text in texts])

        runner.model = TextModel()
        embeddings = runner._encode_class_descriptions([0, 1])
        expected_first = F.normalize(torch.tensor([0.5, 0.5, 0.0]), dim=0)
        self.assertTrue(torch.allclose(embeddings[0], expected_first))
        self.assertTrue(torch.allclose(embeddings[1], torch.tensor([0.0, 0.0, 1.0])))


    def test_description_key_normalization_handles_possessive_apostrophe(self):
        self.assertEqual(
            CSPCRunner._normalize_name("Forster's_Tern"),
            CSPCRunner._normalize_name("Forsters Tern"),
        )

    def test_duplicate_normalized_description_keys_are_rejected(self):
        runner = CSPCRunner.__new__(CSPCRunner)
        runner.class_names = ["Forsters Tern"]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "descriptions.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "Forster's Tern": ["description A"],
                        "Forsters Tern": ["description B"],
                    },
                    handle,
                )
            with self.assertRaisesRegex(ValueError, "collide after normalization"):
                runner._load_class_texts(path)

    def test_non_object_description_json_is_rejected(self):
        runner = CSPCRunner.__new__(CSPCRunner)
        runner.class_names = ["class zero"]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "descriptions.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(["not", "a", "mapping"], handle)
            with self.assertRaisesRegex(ValueError, "must be an object"):
                runner._load_class_texts(path)


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

    def encode_image(self, images):
        features = self.visual.conv1(images)
        return features.mean(dim=(2, 3))


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

        model.train()
        self.assertFalse(model.clip_model.training)
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        image_features = model.encode_images(images)
        self.assertEqual(tuple(image_features.shape), (2, 4))
        self.assertFalse(image_features.requires_grad)
        self.assertIsNone(images.grad)



class TinyFeatureDatasetManager:
    def __init__(self, cfg):
        self.dataset_name = "tiny"
        self.class_names = ["class 0", "class 1", "class 2"]
        self.class_index_in_task = [
            torch.tensor([0, 1]).numpy(),
            torch.tensor([2]).numpy(),
        ]
        self.num_tasks = 2
        self.num_total_classes = 3
        self.train_features = torch.tensor(
            [
                [1.00, 0.04, 0.00],
                [0.98, 0.08, 0.00],
                [0.04, 1.00, 0.00],
                [0.08, 0.98, 0.00],
                [0.02, 0.00, 1.00],
                [0.04, 0.02, 0.98],
            ],
            dtype=torch.float32,
        )
        self.train_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.test_features = self.train_features.clone()
        self.test_labels = self.train_labels.clone()

    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        features = self.train_features if source == "train" else self.test_features
        labels = self.train_labels if source == "train" else self.test_labels
        if source == "train":
            class_ids = self.class_index_in_task[task_id].tolist()
        else:
            class_ids = torch.cat(self.class_index_in_task[: task_id + 1]).tolist()
        selected = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in class_ids:
            selected |= labels == int(class_id)
        features = features[selected]
        labels = labels[selected]
        return [
            {"image": features[start : start + 2], "label": labels[start : start + 2]}
            for start in range(0, len(labels), 2)
        ]


class TinyPromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(3))
        self.register_buffer(
            "class_vectors",
            torch.eye(3, dtype=torch.float32),
        )
        self.transfer = PrototypeTransfer(temperature=0.1)

    @property
    def logit_scale(self):
        return self.prompt.new_tensor(10.0)

    def encode_images(self, images):
        with torch.no_grad():
            return F.normalize(images.to(self.prompt.device).float(), dim=-1)

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        class_ids = [
            int(text.rsplit(" ", 1)[-1].rstrip("."))
            for text in texts
        ]
        vectors = self.class_vectors[class_ids]
        return F.normalize(vectors + 0.05 * self.prompt.unsqueeze(0), dim=-1)


class CSPCEndToEndTests(unittest.TestCase):
    @staticmethod
    def make_cfg(output_dir):
        return types.SimpleNamespace(
            DEVICE=types.SimpleNamespace(DEVICE_NAME="cpu"),
            SEED=7,
            DATASET=types.SimpleNamespace(
                NAME="tiny",
                NUM_CLASSES=3,
                GPT_PATH="",
                NUM_INIT_CLS=2,
                NUM_INC_CLS=1,
                NUM_BASE_SHOT=-1,
                NUM_INC_SHOT=2,
            ),
            DATALOADER=types.SimpleNamespace(
                TRAIN=types.SimpleNamespace(BATCH_SIZE_BASE=4, BATCH_SIZE_INC=2),
                TEST=types.SimpleNamespace(BATCH_SIZE=2),
                NUM_WORKERS=0,
            ),
            TRAINER=types.SimpleNamespace(
                CSPC=types.SimpleNamespace(
                    N_CTX=2,
                    TEXT_CHUNK_SIZE=8,
                    PREC="fp32",
                    PROMPT_INIT_TEXT="a photo of a",
                    BASE_EPOCHS=2,
                    INC_EPOCHS=2,
                    BASE_LR=0.01,
                    INC_PROMPT_LR=0.005,
                    INC_SCORER_LR=0.01,
                    INC_GATE_LR=0.01,
                    WEIGHT_DECAY=0.0,
                    SCORER_TAU=0.1,
                    ALPHA_IMG_TXT=1.0,
                    BETA_SEM=0.1,
                    GAMMA_PROX=0.01,
                    GRAD_CLIP=1.0,
                    OUTPUT_DIR=output_dir,
                )
            ),
        )

    def test_two_session_training_preserves_history_and_writes_complete_metrics(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = self.make_cfg(output_dir)
            manager = TinyFeatureDatasetManager(cfg)
            model = TinyPromptModel()
            with mock.patch(
                "engine.cspc_engine.DatasetManager", return_value=manager
            ), mock.patch(
                "engine.cspc_engine.CSPCModel", return_value=model
            ):
                runner = CSPCRunner(cfg)
                runner.run()

            self.assertEqual(runner.seen_class_ids, [0, 1, 2])
            self.assertEqual(len(runner.session_metrics), 2)
            self.assertEqual(tuple(runner.global_prototypes.shape), (3, 3))
            self.assertTrue(torch.isfinite(runner.global_prototypes).all())

            session0_path = os.path.join(
                runner.output_dir, "session_00.pth"
            )
            session1_path = os.path.join(
                runner.output_dir, "session_01.pth"
            )
            metrics_path = os.path.join(runner.output_dir, "metrics.json")
            self.assertTrue(os.path.isfile(session0_path))
            self.assertTrue(os.path.isfile(session1_path))
            self.assertTrue(os.path.isfile(metrics_path))

            session0 = torch.load(session0_path, map_location="cpu")
            session1 = torch.load(session1_path, map_location="cpu")
            self.assertTrue(
                torch.equal(
                    session0["global_prototypes"],
                    session1["global_prototypes"][:2],
                )
            )
            self.assertTrue(
                torch.equal(
                    session0["base_prototypes"],
                    session1["base_prototypes"],
                )
            )
            self.assertTrue(
                torch.equal(
                    session0["base_text_embeddings"],
                    session1["base_text_embeddings"],
                )
            )

            with open(metrics_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            for key in ("AA", "AA_class_macro", "PD"):
                self.assertTrue(math.isfinite(float(summary[key])))
            self.assertEqual(len(summary["sessions"]), 2)
            self.assertEqual(
                summary["sessions"][-1]["per_class_accuracy"].keys(),
                {"0", "1", "2"},
            )

if __name__ == "__main__":
    unittest.main()
