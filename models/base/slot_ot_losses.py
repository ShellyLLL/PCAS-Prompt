"""Permutation-invariant local-set matching and prototype calibration."""
import torch
import torch.nn.functional as F


def _check_mass(mass, expected_shape, name):
    if tuple(mass.shape) != tuple(expected_shape):
        raise ValueError("{} has shape {}, expected {}.".format(
            name, tuple(mass.shape), tuple(expected_shape)
        ))
    if not torch.isfinite(mass).all() or torch.any(mass < 0):
        raise ValueError("{} must be finite and non-negative.".format(name))
    if not torch.allclose(
        mass.sum(dim=-1),
        torch.ones_like(mass.sum(dim=-1)),
        atol=1e-4,
    ):
        raise ValueError("{} must sum to one along its set dimension.".format(name))


def unbalanced_sinkhorn(
    cost,
    source_mass,
    target_mass,
    epsilon=0.05,
    tau=0.10,
    iterations=20,
    unmatched_penalty=0.25,
):
    """Unbalanced entropic OT; unmatched local evidence is allowed.

    cost: [..., P, Q]
    source_mass: [..., P]
    target_mass: [..., Q]
    Returns a normalized transport cost and the transport plan.
    """
    if cost.ndim < 2:
        raise ValueError("OT cost must have at least two dimensions.")
    if epsilon <= 0 or tau <= 0 or iterations <= 0:
        raise ValueError("epsilon, tau and iterations must be positive.")
    if source_mass.shape != cost.shape[:-1]:
        raise ValueError("Source masses do not match the OT cost matrix.")
    if target_mass.shape != cost.shape[:-2] + (cost.shape[-1],):
        raise ValueError("Target masses do not match the OT cost matrix.")
    if not torch.isfinite(cost).all():
        raise ValueError("OT cost contains non-finite values.")

    tiny = torch.finfo(cost.dtype).tiny
    log_source = source_mass.clamp_min(tiny).log()
    log_target = target_mass.clamp_min(tiny).log()
    log_kernel = -cost / float(epsilon)
    rho = float(tau) / (float(tau) + float(epsilon))

    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        log_u = rho * (
            log_source
            - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        )
        log_v = rho * (
            log_target
            - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        )

    log_plan = log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
    plan = torch.exp(log_plan.clamp(min=-80.0, max=30.0))
    transported_mass = plan.sum(dim=(-2, -1)).clamp_min(1e-8)
    transport_cost = (plan * cost).sum(dim=(-2, -1)) / transported_mass
    missing_mass = (1.0 - transported_mass).clamp(min=0.0, max=1.0)
    distance = transport_cost + float(unmatched_penalty) * missing_mass
    return distance, plan


def pairwise_local_set_distance(
    query_features,
    query_mass,
    reference_features,
    reference_mass,
    epsilon=0.05,
    tau=0.10,
    iterations=20,
    unmatched_penalty=0.25,
):
    """Compare query sets [B,P,D] against reference sets [C,Q,D]."""
    if query_features.ndim != 3 or reference_features.ndim != 3:
        raise ValueError("Local feature sets must have shapes [B,P,D] and [C,Q,D].")
    batch, query_count, dim = query_features.shape
    classes, ref_count, ref_dim = reference_features.shape
    if dim != ref_dim:
        raise ValueError("Query and reference local feature dimensions differ.")
    _check_mass(query_mass, (batch, query_count), "query_mass")
    _check_mass(reference_mass, (classes, ref_count), "reference_mass")

    query = F.normalize(query_features, p=2, dim=-1)
    reference = F.normalize(reference_features, p=2, dim=-1)
    similarity = torch.einsum("bpd,cqd->bcpq", query, reference)
    cost = (1.0 - similarity).clamp_min(0.0)
    source = query_mass[:, None, :].expand(batch, classes, query_count)
    target = reference_mass[None, :, :].expand(batch, classes, ref_count)
    distance, plan = unbalanced_sinkhorn(
        cost, source, target, epsilon, tau, iterations, unmatched_penalty
    )
    return distance, plan


@torch.no_grad()
def weighted_kmeans(features, weights, num_centers, iterations=12):
    """Compress an unordered local-feature set into weighted set descriptors."""
    if features.ndim != 2 or weights.ndim != 1:
        raise ValueError("Expected features [N,D] and weights [N].")
    if features.shape[0] != weights.numel() or features.shape[0] == 0:
        raise ValueError("Features and weights must have the same non-zero length.")
    if num_centers <= 0 or iterations <= 0:
        raise ValueError("num_centers and iterations must be positive.")

    features = F.normalize(features.float(), p=2, dim=-1)
    weights = weights.float().clamp_min(0.0)
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("K-means input contains non-finite values.")
    if weights.sum() <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    count = min(int(num_centers), features.shape[0])
    first = int(weights.argmax().item())
    selected = [first]
    min_distance = (1.0 - features @ features[first]).clamp_min(0.0)
    for _ in range(1, count):
        score = min_distance * weights
        next_index = int(score.argmax().item())
        selected.append(next_index)
        distance = (1.0 - features @ features[next_index]).clamp_min(0.0)
        min_distance = torch.minimum(min_distance, distance)

    centers = features[selected].clone()
    for _ in range(int(iterations)):
        assignment = (features @ centers.t()).argmax(dim=-1)
        cluster_mass = torch.zeros(count, device=features.device, dtype=features.dtype)
        cluster_sum = torch.zeros_like(centers)
        cluster_mass.index_add_(0, assignment, weights)
        cluster_sum.index_add_(
            0, assignment, features * weights.unsqueeze(-1)
        )
        updated = F.normalize(
            cluster_sum / cluster_mass.clamp_min(1e-8).unsqueeze(-1),
            p=2,
            dim=-1,
        )
        centers = torch.where(
            (cluster_mass > 1e-8).unsqueeze(-1), updated, centers
        )

    assignment = (features @ centers.t()).argmax(dim=-1)
    cluster_mass = torch.zeros(count, device=features.device, dtype=features.dtype)
    cluster_mass.index_add_(0, assignment, weights)
    cluster_mass = cluster_mass / cluster_mass.sum().clamp_min(1e-8)

    if count < int(num_centers):
        padding = int(num_centers) - count
        centers = torch.cat([centers, centers[-1:].expand(padding, -1)], dim=0)
        cluster_mass = torch.cat([
            cluster_mass,
            torch.zeros(padding, device=cluster_mass.device, dtype=cluster_mass.dtype),
        ])
    return F.normalize(centers, p=2, dim=-1), cluster_mass


@torch.no_grad()
def build_class_local_memory(
    local_features, local_weights, labels, num_classes, memory_size
):
    """Build per-class unordered local memories; no slot index is preserved."""
    if local_features.ndim != 3 or local_weights.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected local features [N,M,D], weights [N,M], labels [N].")
    if local_features.shape[:2] != local_weights.shape:
        raise ValueError("Local features and slot weights are not aligned.")
    if local_features.shape[0] != labels.numel():
        raise ValueError("Local features and labels are not aligned.")

    memories, masses = [], []
    labels = labels.long()
    for class_id in range(int(num_classes)):
        mask = labels == class_id
        if not torch.any(mask):
            raise ValueError("No local features for class {}.".format(class_id))
        features = local_features[mask].reshape(-1, local_features.shape[-1])
        weights = local_weights[mask].reshape(-1)
        memory, mass = weighted_kmeans(features, weights, memory_size)
        memories.append(memory)
        masses.append(mass)
    return torch.stack(memories), torch.stack(masses)


@torch.no_grad()
def calibrate_local_memory(
    novel_features,
    novel_mass,
    novel_global,
    base_memory,
    base_mass,
    base_global,
    top_k=5,
    max_alpha=0.25,
    global_temperature=0.05,
    match_temperature=0.20,
    epsilon=0.05,
    tau=0.10,
    iterations=30,
):
    """Recompose only reliably matched novel descriptors from fixed base memory.

    Base memories are sources only. The caller must not feed calibrated novel
    memories back into this function as transfer sources.
    """
    if novel_features.ndim != 2 or base_memory.ndim != 3:
        raise ValueError("Expected novel [R,D] and base memory [C,R,D].")
    if novel_mass.shape != novel_features.shape[:1]:
        raise ValueError("Novel local masses do not match novel descriptors.")
    if base_memory.shape[1:] != (novel_features.shape[0], novel_features.shape[1]):
        raise ValueError("Base and novel local memories must share set size and dimension.")
    if base_mass.shape != base_memory.shape[:2]:
        raise ValueError("Base local masses do not match base memory.")
    if base_global.ndim != 2 or novel_global.ndim != 1:
        raise ValueError("Invalid global prototype shapes.")
    if base_global.shape[0] != base_memory.shape[0]:
        raise ValueError("Base global and local memories have different class counts.")
    if base_global.shape[1] != novel_global.numel():
        raise ValueError("Global prototype dimensions differ.")
    _check_mass(novel_mass.unsqueeze(0), (1, novel_mass.numel()), "novel_mass")
    _check_mass(base_mass, tuple(base_memory.shape[:2]), "base_mass")

    novel = F.normalize(novel_features, p=2, dim=-1)
    base = F.normalize(base_memory, p=2, dim=-1)
    global_similarity = F.normalize(novel_global, dim=-1) @ F.normalize(
        base_global, dim=-1
    ).t()
    candidate_count = min(max(1, int(top_k)), base_memory.shape[0])
    candidate_scores, candidate_ids = global_similarity.topk(candidate_count)

    candidate_weights = torch.softmax(
        candidate_scores / max(float(global_temperature), 1e-6), dim=0
    )
    barycenter_sum = torch.zeros_like(novel)
    confidence_sum = torch.zeros_like(novel_mass)

    for rank, class_id_tensor in enumerate(candidate_ids):
        class_id = int(class_id_tensor.item())
        candidate = base[class_id]
        cost = (1.0 - novel @ candidate.t()).clamp_min(0.0)
        _, plan = unbalanced_sinkhorn(
            cost.unsqueeze(0),
            novel_mass.unsqueeze(0),
            base_mass[class_id].unsqueeze(0),
            epsilon=epsilon,
            tau=tau,
            iterations=iterations,
            unmatched_penalty=0.0,
        )
        plan = plan[0]
        row_mass = plan.sum(dim=-1)
        row_cost = (plan * cost).sum(dim=-1) / row_mass.clamp_min(1e-8)
        coverage = (row_mass / novel_mass.clamp_min(1e-8)).clamp(0.0, 1.0)
        confidence = torch.exp(-row_cost / max(float(match_temperature), 1e-6))
        confidence = confidence * coverage
        matched = (plan @ candidate) / row_mass.clamp_min(1e-8).unsqueeze(-1)
        matched = F.normalize(matched, p=2, dim=-1)

        weighted_confidence = candidate_weights[rank] * confidence
        barycenter_sum += weighted_confidence.unsqueeze(-1) * matched
        confidence_sum += weighted_confidence

    barycenter = barycenter_sum / confidence_sum.clamp_min(1e-8).unsqueeze(-1)
    alpha = float(max_alpha) * confidence_sum.clamp(0.0, 1.0)
    calibrated = F.normalize(
        (1.0 - alpha.unsqueeze(-1)) * novel
        + alpha.unsqueeze(-1) * barycenter,
        p=2,
        dim=-1,
    )
    return calibrated, {
        "slot_calibration_alpha": alpha,
        "slot_match_confidence": confidence_sum,
        "retrieved_base_classes": candidate_ids,
        "retrieved_base_weights": candidate_weights,
    }


def joint_prototype_logits(
    global_features,
    local_features,
    local_weights,
    global_prototypes,
    local_memory,
    local_mass,
    global_weight=0.5,
    epsilon=0.05,
    tau=0.10,
    iterations=20,
    temperature=0.10,
):
    """Unified global cosine + permutation-invariant local-set classifier."""
    global_features = F.normalize(global_features, p=2, dim=-1)
    global_prototypes = F.normalize(global_prototypes, p=2, dim=-1)
    global_distance = (1.0 - global_features @ global_prototypes.t()).clamp_min(0.0)
    local_distance, _ = pairwise_local_set_distance(
        local_features, local_weights, local_memory, local_mass,
        epsilon=epsilon, tau=tau, iterations=iterations,
    )
    weight = float(global_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("global_weight must be in [0,1].")
    combined_distance = weight * global_distance + (1.0 - weight) * local_distance
    return -combined_distance / float(temperature), global_distance, local_distance
