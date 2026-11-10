"""Diagnostics for component-conditioned prototype classification."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def affinity_drift(teacher_prototypes, student_prototypes, scale=16.0):
    """Mean row-wise KL drift of off-diagonal prototype affinity graphs."""
    if teacher_prototypes.ndim != 2 or student_prototypes.ndim != 2:
        raise ValueError("Prototype tensors must be rank-2.")
    if teacher_prototypes.shape != student_prototypes.shape:
        raise ValueError("Teacher and student prototype shapes must match.")
    if teacher_prototypes.shape[0] < 2:
        return 0.0
    teacher = F.normalize(teacher_prototypes, dim=-1)
    student = F.normalize(student_prototypes, dim=-1)
    teacher_logits = (teacher @ teacher.t()) * float(scale)
    student_logits = (student @ student.t()) * float(scale)
    diagonal = torch.eye(teacher.shape[0], dtype=torch.bool, device=teacher.device)
    teacher_logits = teacher_logits.masked_fill(diagonal, -1e9)
    student_logits = student_logits.masked_fill(diagonal, -1e9)
    return float(F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ).item())


@torch.no_grad()
def mean_prototype_distance(features, labels, prototypes):
    """Mean cosine distance from each unified feature to its class prototype."""
    if features.ndim != 2 or labels.ndim != 1 or prototypes.ndim != 2:
        raise ValueError("Expected features/prototypes [N,D]/[C,D] and labels [N].")
    if features.shape[0] != labels.numel() or features.shape[1] != prototypes.shape[1]:
        raise ValueError("Feature, label, and prototype shapes are incompatible.")
    labels = labels.long()
    if labels.numel() == 0 or torch.any(labels < 0) or torch.any(labels >= prototypes.shape[0]):
        raise ValueError("Labels are empty or outside the prototype range.")
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return float((1.0 - (features * prototypes[labels]).sum(dim=-1)).mean().item())


@torch.no_grad()
def mean_component_conditioned_prototype_distance(
    features, component_weights, labels, component_conditioned_prototypes
):
    """Distance to the query-conditioned recomposition of the target prototype."""
    if features.ndim != 2 or component_conditioned_prototypes.ndim != 3:
        raise ValueError("Expected unified features [N,D] and prototypes [M,C,D].")
    batch_size, feature_dim = features.shape
    num_components, num_classes, prototype_dim = component_conditioned_prototypes.shape
    if feature_dim != prototype_dim:
        raise ValueError("Feature and prototype dimensions must match.")
    if component_weights.shape != (batch_size, num_components):
        raise ValueError("Component weights must have shape [N,M].")
    if labels.ndim != 1 or labels.numel() != batch_size or labels.numel() == 0:
        raise ValueError("Labels must contain one entry per feature.")
    if not torch.isfinite(component_weights).all() or torch.any(component_weights < 0):
        raise ValueError("Component weights must be finite and non-negative.")
    row_sums = component_weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise ValueError("Each component-weight row must sum to one.")
    labels = labels.long()
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Labels are outside the component prototype range.")
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(component_conditioned_prototypes, dim=-1)
    recomposed = F.normalize(
        torch.einsum("nm,mcd->ncd", component_weights, prototypes), dim=-1
    )
    target = recomposed[torch.arange(batch_size, device=labels.device), labels]
    return float((1.0 - (features * target).sum(dim=-1)).mean().item())


@torch.no_grad()
def component_weight_statistics(component_weights):
    """Summarize learned component usage and per-sample attention entropy."""
    if component_weights.ndim != 2 or component_weights.shape[0] == 0:
        raise ValueError("Component weights must be a non-empty [N,M] tensor.")
    if not torch.isfinite(component_weights).all() or torch.any(component_weights < 0):
        raise ValueError("Component weights must be finite and non-negative.")
    row_sums = component_weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise ValueError("Each component-weight row must sum to one.")
    eps = torch.finfo(component_weights.dtype).eps
    usage = component_weights.mean(dim=0)
    usage_entropy = -(usage * usage.clamp_min(eps).log()).sum()
    sample_entropy = -(
        component_weights * component_weights.clamp_min(eps).log()
    ).sum(dim=-1).mean()
    num_components = component_weights.shape[1]
    normalized_entropy = (
        float((usage_entropy / math.log(num_components)).item())
        if num_components > 1 else 0.0
    )
    return {
        "component_usage": usage.detach().cpu().tolist(),
        "component_usage_entropy": float(usage_entropy.item()),
        "component_effective_count": float(usage_entropy.exp().item()),
        "component_normalized_entropy": normalized_entropy,
        "component_mean_attention_entropy": float(sample_entropy.item()),
    }
