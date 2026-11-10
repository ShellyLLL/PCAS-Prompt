"""Global class-prototype objectives for LGSP-LSP FSCIL."""
import torch
import torch.nn.functional as F


def _validate_features_labels(features, labels):
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features [N,D] and labels [N].")
    if features.shape[0] != labels.numel() or labels.numel() == 0:
        raise ValueError("Features and labels must be non-empty and aligned.")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain non-finite values.")


def build_prototype_cache(features, labels, num_classes):
    """Build normalized global class means."""
    labels = labels.long()
    _validate_features_labels(features, labels)
    num_classes = int(num_classes)
    if num_classes <= 0 or torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Prototype labels are outside [0, num_classes).")
    features = F.normalize(features, p=2, dim=-1)
    counts = torch.bincount(labels, minlength=num_classes)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError("Prototype cache is missing classes: {}.".format(missing))
    sums = torch.zeros(
        num_classes, features.shape[1],
        device=features.device, dtype=features.dtype,
    )
    sums.index_add_(0, labels, features)
    prototypes = F.normalize(sums / counts.to(features.dtype).unsqueeze(1), p=2, dim=-1)
    return {
        "features": features.detach(),
        "labels": labels.detach(),
        "class_sums": sums.detach(),
        "class_counts": counts.detach(),
        "prototypes": prototypes.detach(),
    }


def prototype_logits(features, prototypes, logit_scale=16.0):
    """Cosine-similarity logits over a [C,D] global prototype bank."""
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Expected features [B,D] and prototypes [C,D].")
    if features.shape[1] != prototypes.shape[1] or prototypes.shape[0] == 0:
        raise ValueError("Feature/prototype dimensions or prototype count are invalid.")
    return torch.matmul(
        F.normalize(features, p=2, dim=-1),
        F.normalize(prototypes, p=2, dim=-1).transpose(0, 1),
    ) * float(logit_scale)
