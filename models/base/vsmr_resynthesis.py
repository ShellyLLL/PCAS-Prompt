"""Reliability-aware visual-part transfer and query-conditioned reconstruction."""
import math

import torch
import torch.nn.functional as F

from .vsmr_losses import sinkhorn_transport, pairwise_component_cost


@torch.no_grad()
def build_base_component_bank(class_prototypes):
    """Flatten frozen [C,K,D] class memories into an immutable [C*K,D] bank."""
    if class_prototypes.ndim != 3:
        raise ValueError("Base component prototypes must be [C,K,D].")
    classes, parts, dim = class_prototypes.shape
    if classes < 1 or parts < 1:
        raise ValueError("Base component memory cannot be empty.")
    normalized = F.normalize(class_prototypes, dim=-1)
    owner = torch.arange(classes, device=normalized.device).repeat_interleave(parts)
    part_id = torch.arange(parts, device=normalized.device).repeat(classes)
    return normalized.reshape(classes * parts, dim).detach(), owner, part_id


@torch.no_grad()
def compose_novel_prototypes(novel_parts, base_bank, top_r=5,
                             alpha_max=0.5, similarity_temperature=0.1,
                             variance_scale=0.25):
    """Transfer related base parts independently with a reliability gate.

    Reliability combines mean cosine similarity and the dispersion of the
    retrieved sources. No whole-class base prototype is transferred.
    """
    if novel_parts.ndim != 3 or base_bank.ndim != 2:
        raise ValueError("Expected novel [C,K,D] and base bank [M,D].")
    if novel_parts.shape[-1] != base_bank.shape[-1]:
        raise ValueError("Novel and base component dimensions must match.")
    if int(top_r) < 1 or float(similarity_temperature) <= 0 or float(variance_scale) <= 0:
        raise ValueError("top_r and transfer scales must be positive.")
    if not 0.0 <= float(alpha_max) <= 1.0:
        raise ValueError("alpha_max must be in [0,1].")

    novel = F.normalize(novel_parts, dim=-1)
    bank = F.normalize(base_bank, dim=-1)
    if bank.shape[0] == 0:
        empty = novel.new_zeros(novel.shape[:2])
        return novel, empty, empty, novel.new_zeros(novel.shape[:2]), novel.new_zeros(novel.shape[0], novel.shape[1], 0)

    similarities = torch.einsum("ckd,md->ckm", novel, bank)
    count = min(int(top_r), bank.shape[0])
    scores, indices = similarities.topk(count, dim=-1)
    weights = F.softmax(scores / float(similarity_temperature), dim=-1)
    retrieved = bank[indices]
    transferred = (weights.unsqueeze(-1) * retrieved).sum(dim=-2)
    transferred = F.normalize(transferred, dim=-1)

    mean_similarity = (weights * scores).sum(dim=-1)
    dispersion = (
        weights * (1.0 - F.cosine_similarity(
            retrieved, transferred.unsqueeze(-2), dim=-1
        ))
    ).sum(dim=-1)
    similarity_reliability = mean_similarity.clamp(0.0, 1.0)
    consistency_reliability = torch.exp(
        -dispersion / float(variance_scale)
    )
    reliability = similarity_reliability * consistency_reliability
    alpha = float(alpha_max) * reliability

    composed = F.normalize(
        (1.0 - alpha.unsqueeze(-1)) * novel
        + alpha.unsqueeze(-1) * transferred,
        dim=-1,
    )
    return composed, alpha, reliability, indices, weights


def query_conditioned_reconstruction(query_parts, class_parts,
                                     epsilon=0.05, iterations=30):
    """Reconstruct each class prototype using query-conditioned OT weights.

    The query supplies only the assignment matrix. Reconstructed values are
    weighted sums of that class's stored prototype components; query values
    never enter the prototype content.
    """
    if query_parts.ndim != 3 or class_parts.ndim != 3:
        raise ValueError("Expected query [B,K,D] and class parts [C,K,D].")
    if query_parts.shape[1:] != class_parts.shape[1:]:
        raise ValueError("Query and class component dimensions must match.")
    batch, k, dim = query_parts.shape
    classes = class_parts.shape[0]
    query = F.normalize(query_parts, dim=-1)
    prototypes = F.normalize(class_parts, dim=-1)

    q = query[:, None].expand(batch, classes, k, dim)
    p = prototypes[None].expand(batch, classes, k, dim)
    cost = pairwise_component_cost(
        q.reshape(-1, k, dim), p.reshape(-1, k, dim)
    )
    transport = sinkhorn_transport(cost, epsilon, iterations)
    # Row-normalized transport maps each query component to stored prototype
    # components. Only prototype values contribute to the reconstruction.
    assignment = transport * float(k)
    reconstructed = torch.matmul(
        assignment, p.reshape(-1, k, dim)
    )
    reconstructed = F.normalize(reconstructed, dim=-1)
    # Compare each query part to its reconstructed class part via the
    # same transport plan, preserving permutation invariance.
    similarity = torch.einsum(
        "bkd,bkd->bk",
        q.reshape(-1, k, dim),
        reconstructed,
    )
    local_score = (similarity.mean(dim=-1)).reshape(batch, classes)
    return local_score, reconstructed.reshape(batch, classes, k, dim), transport.reshape(batch, classes, k, k)


@torch.no_grad()
def component_retrieval_diagnostics(novel_parts, base_bank, selected_indices,
                                    alpha, reliability):
    """Compact, auditable diagnostics for cross-class component transfer."""
    if selected_indices.numel() == 0:
        return {
            "retrieval_count": 0,
            "mean_transfer_alpha": 0.0,
            "mean_transfer_reliability": 0.0,
            "unique_base_components": 0,
        }
    return {
        "retrieval_count": int(selected_indices.numel()),
        "mean_transfer_alpha": float(alpha.mean().item()),
        "mean_transfer_reliability": float(reliability.mean().item()),
        "unique_base_components": int(torch.unique(selected_indices).numel()),
    }
