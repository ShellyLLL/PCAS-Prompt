from collections import Counter
from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def summarize_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    class_ids: Sequence[int],
    class_names: Sequence[str],
    logit_scale: float = 1.0,
    top_confusions: int = 3,
) -> Dict:
    """Return label-aware accuracy and cosine-margin diagnostics.

    This function is evaluation-only: it does not modify model parameters,
    prototypes, features, or labels.
    """
    class_ids = [int(class_id) for class_id in class_ids]
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be non-empty and unique.")
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must be rank-2 tensors.")
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("Feature and prototype dimensions must match.")
    if prototypes.shape[0] != len(class_ids):
        raise ValueError("Prototype count must match class_ids.")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be rank-1 and aligned with features.")
    if features.shape[0] == 0:
        raise ValueError("Cannot summarize an empty evaluation set.")
    if features.device != prototypes.device or labels.device != features.device:
        raise ValueError("features, labels, and prototypes must share a device.")
    if len(class_names) <= max(class_ids):
        raise ValueError("class_names does not cover all requested class IDs.")

    id_to_local = {class_id: index for index, class_id in enumerate(class_ids)}
    label_values = [int(value) for value in labels.detach().cpu().tolist()]
    unknown = sorted(set(label_values) - set(class_ids))
    if unknown:
        raise ValueError(f"Evaluation labels contain unseen class IDs: {unknown[:10]}.")
    target_local = torch.tensor(
        [id_to_local[value] for value in label_values],
        device=features.device,
        dtype=torch.long,
    )

    normalized_features = F.normalize(features.float(), dim=-1)
    normalized_prototypes = F.normalize(prototypes.float(), dim=-1)
    similarities = normalized_features @ normalized_prototypes.t()
    predictions_local = similarities.argmax(dim=1)
    predictions = torch.tensor(
        class_ids, device=features.device, dtype=torch.long
    )[predictions_local]

    per_class_accuracy = {}
    per_class = {}
    for class_id in class_ids:
        mask = labels == class_id
        count = int(mask.sum().item())
        if count == 0:
            raise ValueError(f"Evaluation set has no samples for class ID {class_id}.")

        local_index = id_to_local[class_id]
        class_similarities = similarities[mask]
        class_predictions = predictions[mask]
        correct = class_predictions == class_id
        accuracy = float(correct.float().mean().item() * 100.0)
        per_class_accuracy[str(class_id)] = accuracy

        true_similarity = class_similarities[:, local_index]
        if len(class_ids) > 1:
            competitors = class_similarities.clone()
            competitors[:, local_index] = -torch.inf
            best_other = competitors.max(dim=1).values
            margin = true_similarity - best_other
            mean_best_other = float(best_other.mean().item())
            mean_margin = float(margin.mean().item())
        else:
            mean_best_other = None
            mean_margin = None

        wrong_predictions = class_predictions[~correct].detach().cpu().tolist()
        confusion_counts = Counter(int(value) for value in wrong_predictions)
        confusion = [
            {
                "class_id": predicted_id,
                "class_name": str(class_names[predicted_id]),
                "count": int(count_value),
            }
            for predicted_id, count_value in confusion_counts.most_common(
                max(0, int(top_confusions))
            )
        ]
        per_class[str(class_id)] = {
            "class_id": class_id,
            "class_name": str(class_names[class_id]),
            "sample_count": count,
            "accuracy": accuracy,
            "mean_true_cosine": float(true_similarity.mean().item()),
            "mean_best_other_cosine": mean_best_other,
            "mean_cosine_margin": mean_margin,
            "top_confusions": confusion,
        }

    sample_accuracy = float(
        (predictions == labels).float().mean().item() * 100.0
    )
    class_macro_accuracy = float(
        sum(per_class_accuracy.values()) / len(per_class_accuracy)
    )
    return {
        "sample_accuracy": sample_accuracy,
        "class_macro_accuracy": class_macro_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class": per_class,
    }


def summarize_transfer(
    class_id: int,
    class_name: str,
    diagnostics: Dict[str, torch.Tensor],
    raw_prototype: torch.Tensor,
    calibrated_prototype: torch.Tensor,
    base_class_ids: Sequence[int],
    class_names: Sequence[str],
    top_k: int = 5,
) -> Dict:
    """Convert one class's transfer tensors into JSON-safe diagnostics."""
    weights = diagnostics.get("weights")
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise ValueError("Transfer diagnostics must contain rank-1 weights.")
    base_class_ids = [int(value) for value in base_class_ids]
    if weights.numel() != len(base_class_ids):
        raise ValueError("Transfer weights must align with base_class_ids.")
    if len(class_names) <= max(base_class_ids):
        raise ValueError("class_names does not cover all base class IDs.")

    weights_cpu = weights.detach().float().cpu()
    count = min(max(1, int(top_k)), int(weights_cpu.numel()))
    top_values, top_indices = torch.topk(weights_cpu, k=count)
    top_base_classes = [
        {
            "class_id": base_class_ids[int(index)],
            "class_name": str(class_names[base_class_ids[int(index)]]),
            "weight": float(value),
        }
        for value, index in zip(top_values.tolist(), top_indices.tolist())
    ]

    raw = F.normalize(raw_prototype.detach().float(), dim=-1)
    calibrated = F.normalize(calibrated_prototype.detach().float(), dim=-1)
    cosine = torch.dot(raw, calibrated).clamp(-1.0, 1.0)
    angular_shift = torch.rad2deg(torch.acos(cosine))

    def scalar(name):
        value = diagnostics.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu().item())
        return float(value)

    return {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "transfer_ratio": scalar("transfer_ratio"),
        "normalized_transfer_entropy": scalar("entropy"),
        "top1_transfer_weight": float(top_values[0].item()),
        "top2_transfer_weight": (
            float(top_values[1].item()) if top_values.numel() > 1 else 0.0
        ),
        "topk_transfer_mass": float(top_values.sum().item()),
        "top1_top2_margin": scalar("margin"),
        "intra_uncertainty": scalar("intra_uncertainty"),
        "weighted_base_norm": scalar("weighted_base_norm"),
        "raw_calibrated_cosine": float(cosine.item()),
        "prototype_angular_shift_degrees": float(angular_shift.item()),
        "top_base_classes": top_base_classes,
    }
