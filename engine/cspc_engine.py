import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from datasets.data_manager import DatasetManager
from models.cspc import CSPCModel
from utils.evaluator import AccuracyEvaluator


class CSPCRunner:
    """CSPC runner: base prompt learning followed by real-session PE-CFT."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE.DEVICE_NAME)
        self.data_manager = DatasetManager(cfg)
        self.model = CSPCModel(cfg, self.device)
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)

        self.class_names = list(self.data_manager.class_names)
        if len(self.class_names) != int(cfg.DATASET.NUM_CLASSES):
            raise ValueError("NUM_CLASSES does not match the dataset class-name list.")
        self.class_descriptions = self._load_class_texts(cfg.DATASET.GPT_PATH)
        self.base_class_ids = [int(x) for x in self.data_manager.class_index_in_task[0]]
        if self.base_class_ids != list(range(len(self.base_class_ids))):
            raise ValueError("Base classes must occupy the initial contiguous class IDs.")
        self.base_texts = [self.class_descriptions[c] for c in self.base_class_ids]
        self._validate_session_partition()

        output_root = os.path.abspath(cfg.TRAINER.CSPC.OUTPUT_DIR)
        self.output_dir = os.path.join(
            output_root, str(cfg.DATASET.NAME).lower(), f"seed_{int(cfg.SEED)}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_prototypes = None
        self.base_text_embeddings = None
        self.global_prototypes = None
        self.seen_class_ids: List[int] = []
        self.session_metrics: List[Dict] = []
        self.base_accuracy_at_session0 = None
        self.previous_parameters = None

        self.output_dir = os.path.abspath(cfg.TRAINER.CSPC.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.lower().replace("_", " ").replace("-", " ").split())

    def _load_class_texts(self, description_path: str) -> List[str]:
        if not description_path:
            return [f"a photo of a {name}." for name in self.class_names]

        resolved = description_path
        if not os.path.isabs(resolved) and not os.path.exists(resolved):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved = os.path.join(repo_root, description_path)

        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"CSPC class-description file does not exist: {description_path} "
                f"(resolved to {resolved})"
            )

        with open(resolved, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        normalized = {self._normalize_name(key): value for key, value in raw.items()}

        descriptions = []
        missing = []
        for class_name in self.class_names:
            key = self._normalize_name(class_name)
            value = normalized.get(key)
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                value = []
            clean = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if not clean:
                missing.append(class_name)
            descriptions.append(clean)
        if missing:
            raise KeyError("Missing CSPC descriptions for classes: " + ", ".join(missing[:10]))
        return descriptions

    def _validate_session_partition(self):
        flattened = [
            int(class_id)
            for task_classes in self.data_manager.class_index_in_task
            for class_id in task_classes
        ]
        expected = list(range(len(self.class_names)))
        if flattened != expected or len(set(flattened)) != len(flattened):
            raise ValueError("FSCIL sessions must partition all class IDs exactly once in order.")
        if int(self.cfg.DATASET.NUM_INIT_CLS) != len(self.base_class_ids):
            raise ValueError("NUM_INIT_CLS does not match the base session.")
        if int(self.cfg.DATASET.NUM_INC_CLS) <= 0:
            raise ValueError("NUM_INC_CLS must be positive.")
        for name in ("BASE_EPOCHS", "INC_EPOCHS", "TEXT_CHUNK_SIZE", "N_CTX", "GRAD_CLIP", "SCORER_TAU"):
            if float(getattr(self.cfg.TRAINER.CSPC, name)) <= 0:
                raise ValueError(f"TRAINER.CSPC.{name} must be positive.")
        for name in ("BASE_LR", "INC_PROMPT_LR", "INC_SCORER_LR", "INC_GATE_LR", "WEIGHT_DECAY"):
            if float(getattr(self.cfg.TRAINER.CSPC, name)) < 0:
                raise ValueError(f"TRAINER.CSPC.{name} cannot be negative.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is not available.")

    def _encode_class_descriptions(self, class_ids: Sequence[int]) -> torch.Tensor:
        features = []
        for class_id in class_ids:
            descriptions = self.class_descriptions[int(class_id)]
            embeddings = self.model.encode_text(descriptions)
            features.append(F.normalize(embeddings.mean(dim=0), dim=-1))
        return torch.stack(features, dim=0)

    def _collect_features(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = self.data_manager.get_dataloader(
            task_id, source="train", mode="test", accumulate_past=False
        )
        feature_batches = []
        label_batches = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extract train features / session {task_id}"):
                features = self.model.encode_images(batch["image"].to(self.device))
                labels = torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)
                feature_batches.append(features)
                label_batches.append(labels)

        if not feature_batches:
            raise RuntimeError(f"No training features were collected for session {task_id}.")
        features = torch.cat(feature_batches, dim=0).float()
        labels = torch.cat(label_batches, dim=0).long()
        expected = [int(x) for x in self.data_manager.class_index_in_task[task_id]]
        observed = sorted(int(x) for x in torch.unique(labels).cpu().tolist())
        if observed != expected:
            raise ValueError(f"Session {task_id} labels {observed} do not match expected {expected}.")
        return features, labels

    @staticmethod
    def _local_targets(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
        class_to_local = {int(class_id): index for index, class_id in enumerate(class_ids)}
        mapped = [class_to_local.get(int(label), -1) for label in labels.detach().cpu().tolist()]
        if any(index < 0 for index in mapped):
            raise ValueError("A training label does not belong to the current session.")
        return torch.tensor(mapped, device=labels.device, dtype=torch.long)

    @staticmethod
    def _build_prototypes(
        features: torch.Tensor, labels: torch.Tensor, class_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        prototypes = []
        support_sets = []
        for class_id in class_ids:
            class_features = features[labels == int(class_id)]
            if class_features.shape[0] == 0:
                raise ValueError(f"Session class {class_id} has no training samples.")
            support_sets.append(class_features)
            prototypes.append(F.normalize(class_features.mean(dim=0), dim=-1))
        return torch.stack(prototypes, dim=0), support_sets

    def _snapshot_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "prompt": self.model.prompt.detach().clone(),
            "scorer_weight": self.model.transfer.scorer_weight.detach().clone(),
            "gate_weight": self.model.transfer.gate_weight.detach().clone(),
            "gate_bias": self.model.transfer.gate_bias.detach().clone(),
        }

    def _parameter_proximity_loss(self) -> torch.Tensor:
        current = {
            "prompt": self.model.prompt,
            "scorer_weight": self.model.transfer.scorer_weight,
            "gate_weight": self.model.transfer.gate_weight,
            "gate_bias": self.model.transfer.gate_bias,
        }
        terms = [
            (current[name] - self.previous_parameters[name]).pow(2).mean()
            for name in current
        ]
        return torch.stack(terms).mean()

    def _run_base_session(self):
        print("\n========== CSPC Base Session ==========")
        features, labels = self._collect_features(0)
        class_ids = self.base_class_ids
        local_labels = self._local_targets(labels, class_ids)

        optimizer = AdamW(
            [self.model.prompt],
            lr=float(self.cfg.TRAINER.CSPC.BASE_LR),
            weight_decay=float(self.cfg.TRAINER.CSPC.WEIGHT_DECAY),
        )
        self.model.train()
        for epoch in range(int(self.cfg.TRAINER.CSPC.BASE_EPOCHS)):
            optimizer.zero_grad(set_to_none=True)
            text_features = self.model.encode_text(self.base_texts)
            logits = self.model.logit_scale * torch.matmul(features, text_features.t())
            loss = F.cross_entropy(logits, local_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite base loss at epoch {epoch + 1}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.model.prompt], float(self.cfg.TRAINER.CSPC.GRAD_CLIP)
            )
            optimizer.step()
            print(
                f"[Base] epoch {epoch + 1:03d}/{self.cfg.TRAINER.CSPC.BASE_EPOCHS} "
                f"loss={loss.item():.6f}"
            )

        self.model.eval()
        with torch.no_grad():
            self.base_prototypes, _ = self._build_prototypes(features, labels, class_ids)
            self.base_text_embeddings = self._encode_class_descriptions(self.base_class_ids).detach()
            self.global_prototypes = self.base_prototypes.detach().clone()
        self.seen_class_ids = list(class_ids)
        self.previous_parameters = self._snapshot_trainable_parameters()

    def _loo_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_ids: Sequence[int],
        current_text_features: torch.Tensor,
        old_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        _, support_sets = self._build_prototypes(features, labels, class_ids)
        if min(int(support.shape[0]) for support in support_sets) < 2:
            raise ValueError("LOO loss requires at least two real samples per current class.")

        losses = []
        old_count = int(old_prototypes.shape[0])
        for target_local, target_support in enumerate(support_sets):
            for held_out_index in range(target_support.shape[0]):
                query = target_support[held_out_index]
                calibrated_current = []

                for class_local, full_support in enumerate(support_sets):
                    if class_local == target_local:
                        keep = torch.ones(
                            full_support.shape[0], device=full_support.device, dtype=torch.bool
                        )
                        keep[held_out_index] = False
                        support = full_support[keep]
                    else:
                        support = full_support

                    raw_proto = F.normalize(support.mean(dim=0), dim=-1)
                    calibrated_proto, _ = self.model.transfer(
                        query_text=current_text_features[class_local],
                        current_proto=raw_proto,
                        support_features=support,
                        base_text_embeddings=self.base_text_embeddings,
                        base_prototypes=self.base_prototypes,
                    )
                    calibrated_current.append(calibrated_proto)

                current_matrix = torch.stack(calibrated_current, dim=0)
                classifier = torch.cat((old_prototypes.detach(), current_matrix), dim=0)
                logits = self.model.logit_scale * torch.matmul(query.unsqueeze(0), classifier.t())
                target = torch.tensor(
                    [old_count + target_local], device=features.device, dtype=torch.long
                )
                losses.append(F.cross_entropy(logits, target))

        return torch.stack(losses).mean()

    def _run_incremental_session(self, task_id: int):
        print(f"\n========== CSPC Incremental Session {task_id} ==========")
        class_ids = [int(x) for x in self.data_manager.class_index_in_task[task_id]]
        expected_start = len(self.seen_class_ids)
        if class_ids != list(range(expected_start, expected_start + len(class_ids))):
            raise ValueError(
                "CSPC expects the repository's FSCIL class order to be contiguous and append-only."
            )

        features, labels = self._collect_features(task_id)
        local_labels = self._local_targets(labels, class_ids)
        _, support_sets = self._build_prototypes(features, labels, class_ids)
        loo_enabled = min(int(support.shape[0]) for support in support_sets) >= 2

        self.model.transfer.scorer_weight.requires_grad_(loo_enabled)
        self.model.transfer.gate_weight.requires_grad_(loo_enabled)
        self.model.transfer.gate_bias.requires_grad_(loo_enabled)

        parameter_groups = [
            {
                "params": [self.model.prompt],
                "lr": float(self.cfg.TRAINER.CSPC.INC_PROMPT_LR),
            }
        ]
        if loo_enabled:
            parameter_groups.extend(
                [
                    {
                        "params": [self.model.transfer.scorer_weight],
                        "lr": float(self.cfg.TRAINER.CSPC.INC_SCORER_LR),
                    },
                    {
                        "params": [self.model.transfer.gate_weight, self.model.transfer.gate_bias],
                        "lr": float(self.cfg.TRAINER.CSPC.INC_GATE_LR),
                    },
                ]
            )
        optimizer = AdamW(
            parameter_groups,
            weight_decay=float(self.cfg.TRAINER.CSPC.WEIGHT_DECAY),
        )

        old_prototypes = self.global_prototypes.detach().clone()
        self.model.train()

        for epoch in range(int(self.cfg.TRAINER.CSPC.INC_EPOCHS)):
            optimizer.zero_grad(set_to_none=True)
            current_text_features = self._encode_class_descriptions(class_ids)
            current_base_text = self._encode_class_descriptions(self.base_class_ids)

            image_text_logits = self.model.logit_scale * torch.matmul(
                features, current_text_features.t()
            )
            loss_img_txt = F.cross_entropy(image_text_logits, local_labels)
            loss_sem = (
                1.0
                - F.cosine_similarity(
                    current_base_text, self.base_text_embeddings, dim=-1
                )
            ).mean()
            loss_prox = self._parameter_proximity_loss()

            if loo_enabled:
                loss_loo = self._loo_loss(
                    features,
                    labels,
                    class_ids,
                    current_text_features,
                    old_prototypes,
                )
            else:
                loss_loo = loss_img_txt.new_zeros(())

            loss = (
                loss_loo
                + float(self.cfg.TRAINER.CSPC.ALPHA_IMG_TXT) * loss_img_txt
                + float(self.cfg.TRAINER.CSPC.BETA_SEM) * loss_sem
                + float(self.cfg.TRAINER.CSPC.GAMMA_PROX) * loss_prox
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite incremental loss at session {task_id}, epoch {epoch + 1}."
                )
            loss.backward()
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            torch.nn.utils.clip_grad_norm_(
                trainable, float(self.cfg.TRAINER.CSPC.GRAD_CLIP)
            )
            optimizer.step()

            print(
                f"[Session {task_id}] epoch {epoch + 1:03d}/"
                f"{self.cfg.TRAINER.CSPC.INC_EPOCHS} "
                f"total={loss.item():.6f} loo={loss_loo.item():.6f} "
                f"img_txt={loss_img_txt.item():.6f} sem={loss_sem.item():.6f} "
                f"prox={loss_prox.item():.6f} loo_enabled={loo_enabled}"
            )

        self.model.eval()
        with torch.no_grad():
            current_text_features = self._encode_class_descriptions(class_ids)
            current_prototypes, support_sets = self._build_prototypes(features, labels, class_ids)
            calibrated = []
            gate_values = []
            for local_index, class_id in enumerate(class_ids):
                calibrated_proto, diagnostics = self.model.transfer(
                    query_text=current_text_features[local_index],
                    current_proto=current_prototypes[local_index],
                    support_features=support_sets[local_index],
                    base_text_embeddings=self.base_text_embeddings,
                    base_prototypes=self.base_prototypes,
                )
                calibrated.append(calibrated_proto)
                gate_values.append(float(diagnostics["transfer_ratio"].item()))

            calibrated_matrix = torch.stack(calibrated, dim=0)
            # Historical prototypes are never recalibrated or overwritten.
            self.global_prototypes = torch.cat(
                (self.global_prototypes.detach(), calibrated_matrix.detach()), dim=0
            )
        self.seen_class_ids.extend(class_ids)
        self.previous_parameters = self._snapshot_trainable_parameters()
        print(f"[Session {task_id}] transfer ratios: {gate_values}")

    def _evaluate_session(self, task_id: int) -> Dict:
        loader = self.data_manager.get_dataloader(task_id, source="test", mode="test")
        logits_batches = []
        target_batches = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluate session {task_id}"):
                features = self.model.encode_images(batch["image"].to(self.device))
                logits = self.model.logit_scale * torch.matmul(
                    features, self.global_prototypes.t()
                )
                logits_batches.append(logits)
                target_batches.append(
                    torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)
                )

        logits = torch.cat(logits_batches, dim=0)
        targets = torch.cat(target_batches, dim=0)
        result = self.evaluator.calc_accuracy(logits, targets, task_id)

        predictions = logits.argmax(dim=1)
        class_accuracy = {}
        for class_id in self.seen_class_ids:
            mask = targets == int(class_id)
            if not bool(mask.any()):
                continue
            class_accuracy[str(class_id)] = float(
                (predictions[mask] == int(class_id)).float().mean().item() * 100.0
            )

        missing = [int(c) for c in self.seen_class_ids if str(int(c)) not in class_accuracy]
        if missing:
            raise ValueError(f"Test split has no samples for seen classes: {missing[:10]}")
        class_avg = float(np.mean(list(class_accuracy.values())))
        base_values = [
            value for class_id, value in class_accuracy.items()
            if int(class_id) < len(self.base_class_ids)
        ]
        novel_values = [
            value for class_id, value in class_accuracy.items()
            if int(class_id) >= len(self.base_class_ids)
        ]
        base_avg = float(np.mean(base_values)) if base_values else 0.0
        novel_avg = float(np.mean(novel_values)) if novel_values else 0.0

        metrics = {
            "session": int(task_id),
            "sample_accuracy": float(result["mean_acc"]),
            "class_average_accuracy": class_avg,
            "base_class_accuracy": base_avg,
            "novel_class_accuracy": novel_avg,
            "task_accuracy": result["task_acc"],
            "per_class_accuracy": class_accuracy,
        }
        self.session_metrics.append(metrics)
        if task_id == 0:
            self.base_accuracy_at_session0 = base_avg

        print(
            f"[Eval {task_id}] sample_acc={metrics['sample_accuracy']:.2f} "
            f"class_avg={class_avg:.2f} base_avg={base_avg:.2f} "
            f"novel_avg={novel_avg:.2f}"
        )
        return metrics

    def _save_session(self, task_id: int):
        checkpoint = {
            "method": "CSPC-V2.1",
            "session": int(task_id),
            "prompt": self.model.prompt.detach().cpu(),
            "scorer_weight": self.model.transfer.scorer_weight.detach().cpu(),
            "gate_weight": self.model.transfer.gate_weight.detach().cpu(),
            "gate_bias": self.model.transfer.gate_bias.detach().cpu(),
            "base_prototypes": self.base_prototypes.detach().cpu(),
            "base_text_embeddings": self.base_text_embeddings.detach().cpu(),
            "global_prototypes": self.global_prototypes.detach().cpu(),
            "seen_class_ids": list(self.seen_class_ids),
            "metrics": self.session_metrics[-1],
        }
        path = os.path.join(self.output_dir, f"session_{task_id:02d}.pth")
        torch.save(checkpoint, path)
        print(f"Saved prototype-only session state: {path}")

    def _finalize(self):
        class_avg_values = [
            item["class_average_accuracy"] for item in self.session_metrics
        ]
        average_accuracy = float(np.mean([m["sample_accuracy"] for m in self.session_metrics]))
        average_class_accuracy = float(np.mean(class_avg_values))
        final_base_accuracy = self.session_metrics[-1]["base_class_accuracy"]
        performance_drop = float(self.base_accuracy_at_session0 - final_base_accuracy)

        summary = {
            "method": "CSPC-V2.1",
            "AA": average_accuracy,
            "AA_class_macro": average_class_accuracy,
            "PD": performance_drop,
            "final_sample_accuracy": self.session_metrics[-1]["sample_accuracy"],
            "final_class_average_accuracy": self.session_metrics[-1]["class_average_accuracy"],
            "initial_base_class_accuracy": float(self.base_accuracy_at_session0),
            "final_base_class_accuracy": float(final_base_accuracy),
            "sessions": self.session_metrics,
        }
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(
            f"\n[CSPC Final] AA={average_accuracy:.2f} "
            f"PD={performance_drop:.2f} "
            f"Final class-average={summary['final_class_average_accuracy']:.2f}"
        )

    def run(self):
        print(
            f"Starting CSPC on {self.data_manager.dataset_name}: "
            f"{self.data_manager.num_tasks} sessions"
        )
        self._run_base_session()
        self._evaluate_session(0)
        self._save_session(0)

        for task_id in range(1, self.data_manager.num_tasks):
            self._run_incremental_session(task_id)
            self._evaluate_session(task_id)
            self._save_session(task_id)

        self._finalize()
