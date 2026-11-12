"""Unified OT losses for visual-semantic FSCIL.

All local matching is permutation invariant. The same entropic OT cost is used
for cross-image alignment, local classification, prototype construction and
query-conditioned inference.
"""
import math

import torch
import torch.nn.functional as F


def pairwise_component_cost(query_parts, prototype_parts):
    """Cosine cost matrix [B,K,K] for two unordered component sets."""
    if query_parts.ndim != 3 or prototype_parts.ndim != 3:
        raise ValueError("Component sets must be [B,K,D] and [B,K,D].")
    if query_parts.shape != prototype_parts.shape:
        raise ValueError("Pairwise component sets must have identical shapes.")
    q = F.normalize(query_parts, dim=-1)
    p = F.normalize(prototype_parts, dim=-1)
    return (1.0 - torch.matmul(q, p.transpose(1, 2))).clamp_min(0.0)


def sinkhorn_transport(cost, epsilon=0.05, iterations=30):
    """Entropic OT with uniform marginals; stable log-domain Sinkhorn."""
    if cost.ndim < 2 or cost.shape[-1] != cost.shape[-2]:
        raise ValueError("Sinkhorn cost must end in a square [K,K] matrix.")
    if not torch.isfinite(cost).all():
        raise ValueError("Sinkhorn cost contains non-finite values.")
    if float(epsilon) <= 0 or int(iterations) < 1:
        raise ValueError("Sinkhorn epsilon and iteration count must be positive.")
    k = cost.shape[-1]
    log_a = cost.new_full(cost.shape[:-1], -math.log(k))
    log_b = cost.new_full(cost.shape[:-2] + (k,), -math.log(k))
    log_kernel = -cost / float(epsilon)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(int(iterations)):
        u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(-2), dim=-1)
        v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
    transport = torch.exp(log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2))
    return transport


def ot_distance(query_parts, prototype_parts, epsilon=0.05, iterations=30):
    """Permutation-invariant entropic transport cost, reduced to [B]."""
    cost = pairwise_component_cost(query_parts, prototype_parts)
    transport = sinkhorn_transport(cost, epsilon, iterations)
    return (transport * cost).sum(dim=(-1, -2))


def pairwise_ot_distances(parts, epsilon=0.05, iterations=30):
    """All-pairs OT distances for a batch of component sets [B,K,D]."""
    if parts.ndim != 3:
        raise ValueError("parts must have shape [B,K,D].")
    normalized = F.normalize(parts, dim=-1)
    similarity = torch.einsum("ikd,jld->ijkl", normalized, normalized)
    cost = (1.0 - similarity).clamp_min(0.0)
    transport = sinkhorn_transport(cost, epsilon, iterations)
    return (transport * cost).sum(dim=(-1, -2))


def ot_barycenter(component_sets, epsilon=0.05, iterations=30, outer_iterations=5):
    """Approximate an OT barycenter by medoid initialization and soft alignment."""
    if component_sets.ndim != 3 or component_sets.shape[0] < 1:
        raise ValueError("component_sets must be non-empty [N,K,D].")
    sets = F.normalize(component_sets, dim=-1)
    if sets.shape[0] == 1:
        return sets[0]
    distances = pairwise_ot_distances(sets, epsilon, iterations)
    medoid = distances.sum(dim=1).argmin()
    center = sets[medoid]
    for _ in range(int(outer_iterations)):
        aligned = []
        for sample in sets:
            cost = pairwise_component_cost(sample.unsqueeze(0), center.unsqueeze(0))
            transport = sinkhorn_transport(cost, epsilon, iterations)[0]
            # Each sample component receives a normalized barycentric value.
            aligned_sample = (transport.t() @ sample) * sets.shape[1]
            aligned.append(F.normalize(aligned_sample, dim=-1))
        center = F.normalize(torch.stack(aligned, dim=0).mean(dim=0), dim=-1)
    return center


def leave_one_out_global(features, labels):
    """Leave-one-out global class prototypes; no sample contributes to itself."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("Expected features [B,D] and labels [B].")
    classes, inverse = torch.unique(labels.long(), sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=classes.numel()).to(features.dtype)
    if torch.any(counts < 2):
        raise ValueError("Each training class needs at least two samples per batch.")
    normalized = F.normalize(features, dim=-1)
    sums = features.new_zeros(classes.numel(), features.shape[-1])
    sums.index_add_(0, inverse, normalized)
    loo = (sums[inverse] - normalized) / (counts[inverse] - 1).unsqueeze(-1)
    return F.normalize(loo, dim=-1), inverse, classes


def local_leave_one_out_logits(parts, labels, epsilon=0.05, iterations=30,
                               logit_scale=16.0):
    """Query-conditioned local logits against OT class barycenters.

    The target-class barycenter excludes the query. Other class barycenters
    use all batch samples of that class. Query-conditioned OT assignments
    reconstruct values only from the candidate class prototype set, exactly
    as in incremental inference.
    """
    if parts.ndim != 3 or labels.ndim != 1 or parts.shape[0] != labels.numel():
        raise ValueError("Expected parts [B,K,D] and labels [B].")
    labels = labels.long()
    classes = torch.unique(labels, sorted=True)
    class_indices = {
        int(cls): torch.nonzero(labels == cls, as_tuple=False).flatten()
        for cls in classes
    }
    if any(indices.numel() < 2 for indices in class_indices.values()):
        raise ValueError("Each class requires at least two samples for local LOO.")
    full_prototypes = {
        int(cls): ot_barycenter(
            parts[indices], epsilon, iterations, outer_iterations=3
        )
        for cls, indices in class_indices.items()
    }
    rows = []
    k = parts.shape[1]
    for sample_index in range(parts.shape[0]):
        row_scores = []
        own_class = int(labels[sample_index])
        query = F.normalize(parts[sample_index:sample_index + 1], dim=-1)
        for cls in classes:
            cls_id = int(cls)
            indices = class_indices[cls_id]
            if cls_id == own_class:
                retained = indices[indices != sample_index]
                prototype = ot_barycenter(
                    parts[retained], epsilon, iterations, outer_iterations=3
                )
            else:
                prototype = full_prototypes[cls_id]
            prototype = F.normalize(prototype.unsqueeze(0), dim=-1)
            cost = pairwise_component_cost(query, prototype)
            transport = sinkhorn_transport(cost, epsilon, iterations)
            reconstructed = torch.matmul(transport * float(k), prototype)
            reconstructed = F.normalize(reconstructed, dim=-1)
            similarity = F.cosine_similarity(query, reconstructed, dim=-1).mean()
            row_scores.append(similarity * float(logit_scale))
        rows.append(torch.stack(row_scores))
    return torch.stack(rows, dim=0), classes

def js_divergence(p_logits, q_logits):
    """Symmetric Jensen-Shannon divergence between two categorical logits."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-8).log()
    return 0.5 * (
        (p * (p.clamp_min(1e-8).log() - log_m)).sum(-1)
        + (q * (q.clamp_min(1e-8).log() - log_m)).sum(-1)
    )


def cross_image_semantic_alignment(global_logits, local_logits, labels):
    """Align global/local predictive distributions across distinct same-class images."""
    if global_logits.shape != local_logits.shape or global_logits.ndim != 2:
        raise ValueError("Global and local logits must share [B,C] shape.")
    labels = labels.long()
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    pairs = torch.nonzero(same, as_tuple=False)
    if pairs.numel() == 0:
        return (global_logits.sum() + local_logits.sum()) * 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    return 0.5 * (
        js_divergence(global_logits[i], local_logits[j]).mean()
        + js_divergence(local_logits[i], global_logits[j]).mean()
    )


def attention_overlap_loss(attention):
    """Penalize overlap between different prompt-to-Patch distributions."""
    if attention.ndim != 3:
        raise ValueError("attention must have shape [B,K,N].")
    normalized = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    overlap = torch.matmul(normalized, normalized.transpose(1, 2))
    k = attention.shape[1]
    offdiag = ~torch.eye(k, dtype=torch.bool, device=attention.device)
    return overlap[:, offdiag].mean()


def global_logits(features, prototypes, scale=16.0):
    return F.normalize(features, dim=-1) @ F.normalize(prototypes, dim=-1).t() * float(scale)


def local_logits(query_parts, class_prototypes, epsilon=0.05,
                 iterations=30, scale=16.0):
    """OT local logits against [C,K,D] class prototype sets."""
    if query_parts.ndim != 3 or class_prototypes.ndim != 3:
        raise ValueError("Expected query [B,K,D] and prototypes [C,K,D].")
    if query_parts.shape[1:] != class_prototypes.shape[1:]:
        raise ValueError("Query and class component dimensions must match.")
    batch, k, dim = query_parts.shape
    classes = class_prototypes.shape[0]
    q = F.normalize(query_parts, dim=-1)
    p = F.normalize(class_prototypes, dim=-1)
    q = q[:, None].expand(batch, classes, k, dim).reshape(-1, k, dim)
    p = p[None].expand(batch, classes, k, dim).reshape(-1, k, dim)
    distances = ot_distance(q, p, epsilon, iterations).reshape(batch, classes)
    return -distances * float(scale)
