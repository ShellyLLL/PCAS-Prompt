"""PCAS prototype, affinity-consistency, and TEEN calibration objectives."""
import torch
import torch.nn.functional as F


def build_prototype_cache(features, labels, num_classes):
    """Build normalized class prototypes and leave-one-out source features.

    features must be L2-normalized and aligned with the original dataset index.
    """
    labels = labels.long()
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("features and labels have incompatible shapes.")

    counts = torch.bincount(labels, minlength=num_classes)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Base prototype cache is missing classes: {missing}")

    class_sums = torch.zeros(
        num_classes, features.shape[1], device=features.device, dtype=features.dtype
    )
    class_sums.index_add_(0, labels, features)
    class_means = class_sums / counts.to(features.dtype).unsqueeze(1)
    prototypes = F.normalize(class_means, p=2, dim=-1)

    return {
        "features": features,
        "labels": labels,
        "class_sums": class_sums,
        "class_counts": counts,
        "prototypes": prototypes,
    }


def leave_one_out_prototype_loss(
    student_features,
    labels,
    dataset_indices,
    prototype_cache,
    logit_scale=16.0,
):
    """Prototype CE with the current training sample removed from its class mean."""
    labels = labels.long()
    dataset_indices = dataset_indices.long()
    cache_features = prototype_cache["features"]
    cache_labels = prototype_cache["labels"]
    class_sums = prototype_cache["class_sums"]
    class_counts = prototype_cache["class_counts"]
    prototypes = prototype_cache["prototypes"]

    if not torch.equal(cache_labels[dataset_indices], labels):
        raise ValueError("Batch labels do not match the indexed prototype cache.")
    if torch.any(class_counts[labels] <= 1):
        raise ValueError("Leave-one-out prototypes require at least two samples per class.")

    loo_sums = class_sums[labels] - cache_features[dataset_indices]
    loo_means = loo_sums / (class_counts[labels] - 1).to(
        student_features.dtype
    ).unsqueeze(1)
    loo_prototypes = F.normalize(loo_means, p=2, dim=-1)

    logits = student_features @ prototypes.t()
    loo_target_logits = (student_features * loo_prototypes).sum(
        dim=-1, keepdim=True
    )
    target_mask = F.one_hot(
        labels, num_classes=prototypes.shape[0]
    ).to(torch.bool)
    logits = torch.where(target_mask, loo_target_logits, logits)
    return F.cross_entropy(logits * float(logit_scale), labels)


def sample_affinity_consistency(
    teacher_features,
    student_features,
    labels,
    teacher_prototypes,
    student_prototypes,
    affinity_scale=16.0,
):
    """KL(teacher || student) over non-target base-class prototype affinities."""
    labels = labels.long()
    teacher_logits = teacher_features @ teacher_prototypes.t()
    student_logits = student_features @ student_prototypes.t()

    valid_mask = torch.ones_like(teacher_logits, dtype=torch.bool)
    valid_mask.scatter_(1, labels.unsqueeze(1), False)
    teacher_logits = teacher_logits.masked_fill(~valid_mask, -1e9)
    student_logits = student_logits.masked_fill(~valid_mask, -1e9)

    teacher_distribution = F.softmax(
        teacher_logits * float(affinity_scale), dim=-1
    ).detach()
    student_log_distribution = F.log_softmax(
        student_logits * float(affinity_scale), dim=-1
    )
    return F.kl_div(
        student_log_distribution,
        teacher_distribution,
        reduction="batchmean",
    )


def prototype_affinity_consistency(
    teacher_features,
    student_features,
    labels,
    affinity_scale=16.0,
):
    """KL consistency between class-relation distributions in the current batch."""
    labels = labels.long()
    classes, inverse = torch.unique(
        labels, sorted=True, return_inverse=True
    )
    if classes.numel() < 2:
        return student_features.sum() * 0.0

    num_classes = classes.numel()
    teacher_means = torch.zeros(
        num_classes,
        teacher_features.shape[1],
        device=teacher_features.device,
        dtype=teacher_features.dtype,
    )
    student_means = torch.zeros_like(teacher_means)
    counts = torch.zeros(
        num_classes, device=labels.device, dtype=teacher_features.dtype
    )
    teacher_means.index_add_(0, inverse, teacher_features)
    student_means.index_add_(0, inverse, student_features)
    counts.index_add_(
        0, inverse, torch.ones_like(inverse, dtype=teacher_features.dtype)
    )

    teacher_means = F.normalize(teacher_means / counts.unsqueeze(1), dim=-1)
    student_means = F.normalize(student_means / counts.unsqueeze(1), dim=-1)

    teacher_relation = teacher_means @ teacher_means.t()
    student_relation = student_means @ student_means.t()
    diagonal = torch.eye(
        num_classes, device=labels.device, dtype=torch.bool
    )
    teacher_relation = teacher_relation.masked_fill(diagonal, -1e9)
    student_relation = student_relation.masked_fill(diagonal, -1e9)

    teacher_distribution = F.softmax(
        teacher_relation * float(affinity_scale), dim=-1
    ).detach()
    student_log_distribution = F.log_softmax(
        student_relation * float(affinity_scale), dim=-1
    )
    return F.kl_div(
        student_log_distribution,
        teacher_distribution,
        reduction="batchmean",
    )


def teen_calibrate(
    novel_prototypes,
    base_prototypes,
    alpha=0.5,
    tau=16.0,
):
    """TEEN calibration using only final PCAS student base prototypes."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("TEEN alpha must be in [0, 1].")
    if novel_prototypes.ndim != 2 or base_prototypes.ndim != 2:
        raise ValueError("Prototype tensors must be rank-2.")
    if novel_prototypes.shape[1] != base_prototypes.shape[1]:
        raise ValueError("Novel and base prototypes must share one feature space.")

    novel_prototypes = F.normalize(novel_prototypes, p=2, dim=-1)
    base_prototypes = F.normalize(base_prototypes, p=2, dim=-1)
    similarities = novel_prototypes @ base_prototypes.t()
    weights = F.softmax(similarities * float(tau), dim=-1)
    calibrated = (1.0 - float(alpha)) * novel_prototypes + float(alpha) * (
        weights @ base_prototypes
    )
    return F.normalize(calibrated, p=2, dim=-1), weights
