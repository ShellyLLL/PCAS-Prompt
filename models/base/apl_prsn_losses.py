"""Training objectives for APL part learning and PRSN prototype resynthesis."""
import torch
import torch.nn.functional as F


def build_class_prototypes(global_features, parts, part_valid, labels):
    """Build global and aligned local prototypes from one feature stream."""
    if global_features.ndim != 2 or parts.ndim != 3:
        raise ValueError("Expected global features [B,D] and parts [B,T,D].")
    batch_size, num_parts, feature_dim = parts.shape
    if global_features.shape != (batch_size, feature_dim):
        raise ValueError("Global and local feature dimensions must match.")
    if part_valid.shape != (batch_size, num_parts):
        raise ValueError("part_valid must have shape [B,T].")
    labels = labels.long()
    if labels.shape != (batch_size,) or batch_size == 0:
        raise ValueError("labels must contain one class ID per sample.")

    classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    num_classes = classes.numel()
    counts = torch.bincount(inverse, minlength=num_classes).to(global_features.dtype)

    global_sums = global_features.new_zeros(num_classes, feature_dim)
    global_sums.index_add_(0, inverse, F.normalize(global_features, dim=-1))
    global_prototypes = F.normalize(
        global_sums / counts.unsqueeze(-1), dim=-1
    )

    normalized_parts = F.normalize(parts, dim=-1)
    valid_float = part_valid.to(dtype=parts.dtype)
    part_sums = parts.new_zeros(num_classes, num_parts, feature_dim)
    part_counts = parts.new_zeros(num_classes, num_parts)
    for part_index in range(num_parts):
        weights = valid_float[:, part_index]
        part_sums[:, part_index].index_add_(
            0, inverse, normalized_parts[:, part_index] * weights.unsqueeze(-1)
        )
        part_counts[:, part_index].index_add_(0, inverse, weights)

    prototype_valid = part_counts > 0
    prototype_parts = part_sums / part_counts.clamp_min(1.0).unsqueeze(-1)
    prototype_parts = F.normalize(prototype_parts, dim=-1)
    prototype_parts = prototype_parts * prototype_valid.unsqueeze(-1)
    return {
        "classes": classes,
        "inverse": inverse,
        "counts": counts,
        "global_sums": global_sums,
        "global": global_prototypes,
        "part_sums": part_sums,
        "part_counts": part_counts,
        "parts": prototype_parts,
        "valid": prototype_valid,
    }


def build_leave_one_out_prototypes(
    global_features, parts, part_valid, labels, prototype_cache
):
    """Construct each sample's class prototype without using that sample."""
    labels = labels.long()
    inverse = prototype_cache["inverse"]
    counts = prototype_cache["counts"]
    if labels.shape != inverse.shape or not torch.equal(
        torch.unique(labels, sorted=True), prototype_cache["classes"]
    ):
        raise ValueError("Labels do not match the class prototype cache.")
    if torch.any(counts[inverse] <= 1):
        raise ValueError("Leave-one-out prototypes require at least two samples per class.")

    normalized_global = F.normalize(global_features, dim=-1)
    loo_global = (
        prototype_cache["global_sums"][inverse] - normalized_global
    ) / (counts[inverse] - 1.0).unsqueeze(-1)
    loo_global = F.normalize(loo_global, dim=-1)

    normalized_parts = F.normalize(parts, dim=-1)
    valid_float = part_valid.to(dtype=parts.dtype)
    loo_part_sums = (
        prototype_cache["part_sums"][inverse]
        - normalized_parts * valid_float.unsqueeze(-1)
    )
    loo_part_counts = (
        prototype_cache["part_counts"][inverse] - valid_float
    )
    loo_valid = loo_part_counts > 0
    loo_parts = loo_part_sums / loo_part_counts.clamp_min(1.0).unsqueeze(-1)
    loo_parts = F.normalize(loo_parts, dim=-1) * loo_valid.unsqueeze(-1)
    return {
        "global": loo_global,
        "parts": loo_parts,
        "valid": loo_valid,
        "inverse": inverse,
    }


def apl_all_min_contrastive_loss(parts, part_valid, labels, temperature=0.10):
    """APL all-min loss: all corresponding positives, least-similar negatives."""
    if parts.ndim != 3 or part_valid.shape != parts.shape[:2]:
        raise ValueError("Expected parts [B,T,D] and part_valid [B,T].")
    if float(temperature) <= 0:
        raise ValueError("The all-min temperature must be positive.")
    labels = labels.long()
    batch_size = parts.shape[0]
    if labels.shape != (batch_size,):
        raise ValueError("labels must contain one class ID per part set.")
    if batch_size < 3:
        return parts.sum() * 0.0

    normalized = F.normalize(parts, dim=-1)
    similarities = torch.einsum("itd,jtd->ijt", normalized, normalized)
    common_valid = part_valid.bool()[:, None, :] & part_valid.bool()[None, :, :]
    same_class = labels[:, None] == labels[None, :]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=parts.device)
    positive_pairs = same_class & ~diagonal & common_valid.any(dim=-1)
    negative_pairs = (~same_class) & common_valid.any(dim=-1)

    losses = []
    for anchor in range(batch_size):
        positive_mask = positive_pairs[anchor]
        negative_mask = negative_pairs[anchor]
        if not positive_mask.any() or not negative_mask.any():
            continue

        positive_similarities = similarities[anchor, positive_mask]
        positive_valid = common_valid[anchor, positive_mask]
        positive_logits = (
            positive_similarities / float(temperature)
        ).masked_fill(~positive_valid, -1e4).reshape(-1)
        log_positive = torch.logsumexp(positive_logits, dim=0)

        negative_similarities = similarities[anchor, negative_mask]
        negative_valid = common_valid[anchor, negative_mask]
        least_similar = negative_similarities.masked_fill(
            ~negative_valid, 1e4
        ).min(dim=-1).values
        log_negative = torch.logsumexp(
            least_similar / float(temperature), dim=0
        )
        losses.append(F.softplus(log_negative - log_positive))

    if not losses:
        return parts.sum() * 0.0
    return torch.stack(losses).mean()


def apl_part_diversity_loss(parts, part_valid):
    """APL diversity objective discouraging collapsed, redundant part queries."""
    if parts.ndim != 3 or part_valid.shape != parts.shape[:2]:
        raise ValueError("Expected parts [B,T,D] and part_valid [B,T].")
    normalized = F.normalize(parts, dim=-1)
    similarities = torch.matmul(normalized, normalized.transpose(1, 2))
    num_parts = parts.shape[1]
    off_diagonal = ~torch.eye(
        num_parts, dtype=torch.bool, device=parts.device
    ).unsqueeze(0)
    valid_pairs = (
        part_valid.bool().unsqueeze(2)
        & part_valid.bool().unsqueeze(1)
        & off_diagonal
    )
    penalties = F.relu(similarities) * valid_pairs
    return penalties.sum() / valid_pairs.sum().clamp_min(1)


def prsn_cross_image_semantic_alignment_loss(
    global_logits, local_logits, labels
):
    """PRSN CISA: align global/local predictions across distinct same-class images."""
    if global_logits.ndim != 2 or local_logits.shape != global_logits.shape:
        raise ValueError("Global and local logits must have the same [B,C] shape.")
    labels = labels.long()
    batch_size = global_logits.shape[0]
    if labels.shape != (batch_size,):
        raise ValueError("labels must contain one class ID per sample.")
    if batch_size < 2:
        return global_logits.sum() * 0.0

    same_class = labels[:, None] == labels[None, :]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    pairs = same_class & ~diagonal
    if not pairs.any():
        return (global_logits.sum() + local_logits.sum()) * 0.0

    global_targets = F.softmax(global_logits.detach(), dim=-1)
    local_targets = F.softmax(local_logits.detach(), dim=-1)
    global_log_prob = F.log_softmax(global_logits, dim=-1)
    local_log_prob = F.log_softmax(local_logits, dim=-1)

    global_to_local = -(
        global_targets[:, None, :] * local_log_prob[None, :, :]
    ).sum(dim=-1)
    local_to_global = -(
        local_targets[:, None, :] * global_log_prob[None, :, :]
    ).sum(dim=-1)
    return 0.5 * (
        global_to_local[pairs].mean() + local_to_global[pairs].mean()
    )


def cosine_prototype_logits(features, prototypes, logit_scale=16.0):
    """Cosine classifier shared by the global and local PRSN branches."""
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Features and prototypes must be rank-2 tensors.")
    if features.shape[-1] != prototypes.shape[-1]:
        raise ValueError("Feature and prototype dimensions must match.")
    return (
        F.normalize(features, dim=-1)
        @ F.normalize(prototypes, dim=-1).t()
    ) * float(logit_scale)
