"""Diagnostics for PCAS-Prompt feature geometry and component routing."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def affinity_drift(teacher_prototypes, student_prototypes, scale=16.0):
    """Mean row-wise KL drift of off-diagonal base-prototype affinity graphs."""
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
    diagonal = torch.eye(
        teacher.shape[0], dtype=torch.bool, device=teacher.device
    )
    teacher_logits = teacher_logits.masked_fill(diagonal, -torch.inf)
    student_logits = student_logits.masked_fill(diagonal, -torch.inf)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    return float(F.kl_div(
        student_log_prob, teacher_prob, reduction="batchmean"
    ).item())


@torch.no_grad()
def mean_prototype_distance(features, labels, prototypes):
    """Mean cosine distance from each feature to its own class prototype."""
    if features.ndim != 2 or labels.ndim != 1 or prototypes.ndim != 2:
        raise ValueError("Expected features/prototypes [N,D]/[C,D] and labels [N].")
    if features.shape[0] != labels.numel():
        raise ValueError("Features and labels must have the same sample count.")
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("Feature and prototype dimensions must match.")
    labels = labels.long()
    if labels.numel() == 0 or torch.any(labels < 0) or torch.any(labels >= prototypes.shape[0]):
        raise ValueError("Labels are empty or outside the prototype range.")
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return float((1.0 - (features * prototypes[labels]).sum(dim=-1)).mean().item())


@torch.no_grad()
def routing_diversity(routing_weights):
    """Summarize marginal component usage and per-example routing entropy."""
    if routing_weights.ndim != 2 or routing_weights.shape[0] == 0:
        raise ValueError("Routing weights must be a non-empty [N,M] tensor.")
    if not torch.isfinite(routing_weights).all():
        raise ValueError("Routing weights contain non-finite values.")
    if torch.any(routing_weights < 0):
        raise ValueError("Routing weights must be non-negative.")
    row_sums = routing_weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise ValueError("Each routing-weight row must sum to one.")

    eps = torch.finfo(routing_weights.dtype).eps
    usage = routing_weights.mean(dim=0)
    usage_entropy = -(usage * usage.clamp_min(eps).log()).sum()
    sample_entropy = -(
        routing_weights * routing_weights.clamp_min(eps).log()
    ).sum(dim=-1).mean()
    num_components = routing_weights.shape[1]
    normalized_entropy = (
        float((usage_entropy / math.log(num_components)).item())
        if num_components > 1 else 0.0
    )
    return {
        "routing_usage": usage.detach().cpu().tolist(),
        "routing_usage_entropy": float(usage_entropy.item()),
        "routing_effective_components": float(usage_entropy.exp().item()),
        "routing_normalized_entropy": normalized_entropy,
        "routing_mean_sample_entropy": float(sample_entropy.item()),
    }
