"""Visual component transfer and query-conditioned prototype reconstruction."""
import torch
import torch.nn.functional as F

from .vsmr_losses import (
    pairwise_component_cost,
    sinkhorn_transport,
    ot_barycenter,
)


@torch.no_grad()
def build_base_component_bank(class_prototypes):
    """Flatten immutable [C,K,D] class memories and retain class ownership."""
    if class_prototypes.ndim != 3:
        raise ValueError("Base component prototypes must be [C,K,D].")
    classes, parts, dim = class_prototypes.shape
    if classes < 1 or parts < 1 or dim < 1:
        raise ValueError("Base component memory cannot be empty.")
    normalized = F.normalize(class_prototypes, dim=-1)
    owner = torch.arange(classes, device=normalized.device).repeat_interleave(parts)
    part_id = torch.arange(parts, device=normalized.device).repeat(classes)
    return normalized.reshape(classes * parts, dim).detach(), owner, part_id


def component_support_variance(support_parts, prototype, epsilon=0.05,
                               iterations=30):
    """OT-align K-shot components and measure per-component support dispersion.

    Args:
        support_parts: [S,K,D] components from support images of one class.
        prototype: [K,D] OT barycenter for that class.
    Returns:
        [K] mean cosine dispersion after permutation-invariant alignment.
    """
    if support_parts.ndim != 3 or prototype.ndim != 2:
        raise ValueError("Expected support [S,K,D] and prototype [K,D].")
    if support_parts.shape[1:] != prototype.shape:
        raise ValueError("Support and prototype component shapes must match.")
    if support_parts.shape[0] < 1:
        raise ValueError("At least one support image is required.")
    if support_parts.shape[0] == 1:
        return prototype.new_zeros(prototype.shape[0])

    target = F.normalize(prototype, dim=-1)
    aligned = []
    count = target.shape[0]
    for sample in support_parts:
        cost = pairwise_component_cost(
            F.normalize(sample, dim=-1).unsqueeze(0),
            target.unsqueeze(0),
        )
        transport = sinkhorn_transport(cost, epsilon, iterations)[0]
        # Rows are support components; columns are prototype components.
        mapped = (transport.transpose(0, 1) @ F.normalize(sample, dim=-1)) * count
        aligned.append(F.normalize(mapped, dim=-1))
    aligned = torch.stack(aligned, dim=0)
    cosine = F.cosine_similarity(aligned, target.unsqueeze(0), dim=-1)
    return (1.0 - cosine).clamp_min(0.0).mean(dim=0)


def compose_novel_prototypes(
    novel_parts,
    base_bank,
    gate,
    base_owner=None,
    top_r=5,
    alpha_max=0.5,
    similarity_temperature=0.1,
    support_variance=None,
    shots=1,
):
    """Class-balanced component retrieval followed by a learned reliability gate.

    At most one component is retrieved from each source base class for each
    novel component. The gate is trained end-to-end by pseudo-incremental
    classification episodes; it is not a hand-written similarity heuristic.
    """
    if novel_parts.ndim != 3 or base_bank.ndim != 2:
        raise ValueError("Expected novel [C,K,D] and base bank [M,D].")
    if novel_parts.shape[-1] != base_bank.shape[-1]:
        raise ValueError("Novel and base component dimensions must match.")
    if gate is None:
        raise ValueError("A learned ReliabilityGate is required.")
    if int(top_r) < 1 or float(similarity_temperature) <= 0:
        raise ValueError("top_r and similarity_temperature must be positive.")
    if not 0.0 <= float(alpha_max) <= 1.0:
        raise ValueError("alpha_max must be in [0,1].")

    novel = F.normalize(novel_parts, dim=-1)
    bank = F.normalize(base_bank, dim=-1)
    if bank.shape[0] == 0:
        zeros = novel.new_zeros(novel.shape[:2])
        return (
            novel, zeros, zeros,
            torch.empty(*novel.shape[:2], 0, dtype=torch.long, device=novel.device),
            novel.new_zeros(*novel.shape[:2], 0),
        )

    if base_owner is None:
        # Compatibility fallback: each bank vector is treated as its own source.
        owner = torch.arange(bank.shape[0], device=bank.device)
    else:
        owner = torch.as_tensor(base_owner, device=bank.device, dtype=torch.long)
        if owner.ndim != 1 or owner.numel() != bank.shape[0]:
            raise ValueError("base_owner must have one source-class id per bank item.")

    similarities = torch.einsum("ckd,md->ckm", novel, bank)
    source_classes = torch.unique(owner, sorted=True)
    if source_classes.numel() == 0:
        raise ValueError("Base component bank has no source classes.")

    # First reduce each source class to its best matching component. This
    # prevents a base class with many similar parts from dominating Top-r.
    source_scores = []
    source_indices = []
    bank_indices = torch.arange(bank.shape[0], device=bank.device)
    for source_class in source_classes:
        positions = bank_indices[owner == source_class]
        scores_for_class, local_indices = similarities[..., positions].max(dim=-1)
        source_scores.append(scores_for_class)
        source_indices.append(positions[local_indices])
    source_scores = torch.stack(source_scores, dim=-1)
    source_indices = torch.stack(source_indices, dim=-1)

    count = min(int(top_r), int(source_classes.numel()))
    top_scores, top_source_positions = source_scores.topk(count, dim=-1)
    selected = source_indices.gather(-1, top_source_positions)
    weights = F.softmax(top_scores / float(similarity_temperature), dim=-1)
    retrieved = bank[selected]
    transferred = F.normalize(
        (weights.unsqueeze(-1) * retrieved).sum(dim=-2), dim=-1
    )

    mean_similarity = (weights * top_scores).sum(dim=-1)
    if source_classes.numel() > 1:
        two_best = source_scores.topk(2, dim=-1).values
        margin = (two_best[..., 0] - two_best[..., 1]).clamp_min(0.0)
    else:
        margin = torch.zeros_like(mean_similarity)
    if count > 1:
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=-1) / float(torch.log(weights.new_tensor(float(count))))
    else:
        entropy = torch.zeros_like(mean_similarity)

    if support_variance is None:
        support_variance = torch.zeros_like(mean_similarity)
    else:
        support_variance = torch.as_tensor(
            support_variance, device=novel.device, dtype=novel.dtype
        )
        if support_variance.shape != novel.shape[:2]:
            raise ValueError("support_variance must have shape [C,K].")

    reliability = gate(
        novel,
        transferred,
        mean_similarity,
        margin,
        entropy,
        support_variance,
        shots,
    )
    alpha = float(alpha_max) * reliability
    composed = F.normalize(
        (1.0 - alpha.unsqueeze(-1)) * novel
        + alpha.unsqueeze(-1) * transferred,
        dim=-1,
    )
    return composed, alpha, reliability, selected, weights


def query_conditioned_reconstruction(
    query_parts,
    query_patches,
    class_parts,
    reconstructor,
    epsilon=0.05,
    iterations=30,
    scale=16.0,
):
    """Cross-attend class prototypes to query Patch tokens, then OT-match.

    Query Patch tokens provide Q/K/V-conditioned context for a temporary class
    representation. The resulting representation is never written to memory.
    The final local score is the negative entropic OT cost between the query's
    decomposed components and the reconstructed class components.
    """
    if query_parts.ndim != 3 or query_patches.ndim != 3:
        raise ValueError("Expected query parts [B,K,D] and patches [B,N,D].")
    if query_parts.shape[0] != query_patches.shape[0]:
        raise ValueError("Query parts and Patch tokens must share batch size.")
    if class_parts.ndim not in (3, 4):
        raise ValueError("class_parts must be [C,K,D] or [B,C,K,D].")
    if query_parts.shape[-1] != query_patches.shape[-1]:
        raise ValueError("Query parts and Patch tokens must share embedding dimension.")
    if class_parts.shape[-2:] != query_parts.shape[-2:]:
        raise ValueError("Query and class component shapes must match.")
    if float(epsilon) <= 0 or int(iterations) < 1:
        raise ValueError("OT epsilon and iterations must be positive.")

    reconstructed, attention = reconstructor(query_patches, class_parts)
    batch, classes, count, dim = reconstructed.shape
    query = F.normalize(query_parts, dim=-1)
    query = query[:, None].expand(batch, classes, count, dim)
    flat_query = query.reshape(batch * classes, count, dim)
    flat_reconstructed = reconstructed.reshape(batch * classes, count, dim)
    cost = pairwise_component_cost(flat_query, flat_reconstructed)
    transport = sinkhorn_transport(cost, epsilon, iterations)
    distance = (transport * cost).sum(dim=(-1, -2))
    scores = -distance.reshape(batch, classes) * float(scale)
    return (
        scores,
        reconstructed,
        transport.reshape(batch, classes, count, count),
        attention,
    )


def local_leave_one_out_query_logits(
    parts,
    query_patches,
    labels,
    reconstructor,
    epsilon=0.05,
    iterations=30,
    logit_scale=16.0,
    barycenter_iterations=5,
):
    """LOO local logits using the same Q/K/V reconstruction and OT as inference."""
    if parts.ndim != 3 or query_patches.ndim != 3 or labels.ndim != 1:
        raise ValueError("Expected parts [B,K,D], patches [B,N,D], labels [B].")
    if parts.shape[0] != labels.numel() or query_patches.shape[0] != labels.numel():
        raise ValueError("Parts, Patch tokens and labels must share batch size.")
    labels = labels.long()
    classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    class_indices = {
        int(cls): torch.nonzero(labels == cls, as_tuple=False).flatten()
        for cls in classes
    }
    if any(indices.numel() < 2 for indices in class_indices.values()):
        raise ValueError("Each class requires at least two samples for local LOO.")

    full_prototypes = torch.stack([
        ot_barycenter(
            parts[class_indices[int(cls)]],
            epsilon,
            iterations,
            outer_iterations=barycenter_iterations,
        )
        for cls in classes
    ])
    loo_prototypes = []
    for sample_index in range(parts.shape[0]):
        cls_id = int(labels[sample_index])
        retained = class_indices[cls_id]
        retained = retained[retained != sample_index]
        loo_prototypes.append(ot_barycenter(
            parts[retained],
            epsilon,
            iterations,
            outer_iterations=barycenter_iterations,
        ))
    loo_prototypes = torch.stack(loo_prototypes)

    batch, count, dim = parts.shape
    candidate = full_prototypes.unsqueeze(0).expand(
        batch, classes.numel(), count, dim
    ).clone()
    batch_indices = torch.arange(batch, device=parts.device)
    candidate[batch_indices, inverse] = loo_prototypes
    scores, _, _, _ = query_conditioned_reconstruction(
        parts,
        query_patches,
        candidate,
        reconstructor,
        epsilon=epsilon,
        iterations=iterations,
        scale=logit_scale,
    )
    return scores, classes


@torch.no_grad()
def component_retrieval_diagnostics(selected_indices, alpha, reliability,
                                    base_owner=None):
    """Auditable retrieval and learned-gate diagnostics."""
    if selected_indices.numel() == 0:
        return {
            "retrieval_count": 0,
            "mean_transfer_alpha": 0.0,
            "mean_transfer_reliability": 0.0,
            "unique_base_components": 0,
        }
    result = {
        "retrieval_count": int(selected_indices.numel()),
        "mean_transfer_alpha": float(alpha.mean().item()),
        "mean_transfer_reliability": float(reliability.mean().item()),
        "unique_base_components": int(torch.unique(selected_indices).numel()),
    }
    if base_owner is not None:
        selected_owners = base_owner[selected_indices]
        rows = selected_owners.reshape(-1, selected_owners.shape[-1])
        unique_counts = torch.tensor(
            [torch.unique(row).numel() for row in rows],
            dtype=torch.float32,
            device=selected_owners.device,
        )
        result["mean_unique_source_classes_per_component"] = float(
            unique_counts.mean().item()
        )
    return result
