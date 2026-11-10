"""Prototype objectives for single-stream component-conditioned PCAS."""
import torch
import torch.nn.functional as F


def _validate_features_labels(features, labels):
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features [N,D] and labels [N].")
    if features.shape[0] != labels.numel():
        raise ValueError("Features and labels have incompatible sample counts.")
    if labels.numel() == 0:
        raise ValueError("Cannot build prototypes from an empty dataset.")


def _validate_component_weights(weights, batch_size, num_components):
    if weights.ndim != 2 or weights.shape != (batch_size, num_components):
        raise ValueError("Component weights must have shape [N,M].")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("Component weights must be finite and non-negative.")
    if not torch.allclose(
        weights.sum(dim=-1),
        torch.ones(batch_size, device=weights.device, dtype=weights.dtype),
        atol=1e-5,
    ):
        raise ValueError("Each component-weight row must sum to one.")


def build_prototype_cache(features, labels, num_classes):
    """Build normalized global class prototypes for the frozen ViT teacher."""
    labels = labels.long()
    _validate_features_labels(features, labels)
    num_classes = int(num_classes)
    if num_classes <= 0 or torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Prototype labels are outside [0, num_classes).")
    features = F.normalize(features, p=2, dim=-1)
    counts = torch.bincount(labels, minlength=num_classes)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Prototype cache is missing classes: {missing}")
    class_sums = torch.zeros(
        num_classes, features.shape[1], device=features.device, dtype=features.dtype
    )
    class_sums.index_add_(0, labels, features)
    prototypes = F.normalize(class_sums / counts.to(features.dtype).unsqueeze(1), dim=-1)
    return {
        "features": features.detach(),
        "labels": labels.detach(),
        "class_sums": class_sums.detach(),
        "class_counts": counts.detach(),
        "prototypes": prototypes.detach(),
    }


def build_component_conditioned_prototypes(features, component_weights):
    """Build [M,D] prototypes by weighting one shared feature per sample."""
    if features.ndim != 2:
        raise ValueError("Unified features must have shape [N,D].")
    if features.shape[0] == 0:
        raise ValueError("Cannot build component-conditioned prototypes from no samples.")
    if component_weights.ndim != 2 or component_weights.shape[0] != features.shape[0]:
        raise ValueError("Component weights must have shape [N,M].")
    _validate_component_weights(component_weights, features.shape[0], component_weights.shape[1])
    features = F.normalize(features, p=2, dim=-1)
    weighted_sums = torch.einsum("nm,nd->md", component_weights, features)
    weight_sums = component_weights.sum(dim=0)
    eps = torch.finfo(features.dtype).eps
    if torch.any(weight_sums <= eps):
        raise ValueError("A component has insufficient support weight to form a prototype.")
    return F.normalize(weighted_sums / weight_sums.unsqueeze(-1), p=2, dim=-1)


def build_component_conditioned_prototype_cache(
    features, component_weights, labels, num_classes
):
    """Build class/component prototypes from a single unified feature stream.

    For class c and component m, p[c,m] is the weighted mean of the same
    final student features z_i, with each sample weighted by its image-level
    component contribution w[i,m]. No isolated component feature is encoded.
    """
    labels = labels.long()
    _validate_features_labels(features, labels)
    num_classes = int(num_classes)
    if num_classes <= 0 or torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Prototype labels are outside [0, num_classes).")
    if component_weights.ndim != 2 or component_weights.shape[0] != features.shape[0]:
        raise ValueError("Component weights must have shape [N,M].")
    num_components = component_weights.shape[1]
    _validate_component_weights(component_weights, features.shape[0], num_components)
    features = F.normalize(features, p=2, dim=-1)
    component_weights = component_weights.to(device=features.device, dtype=features.dtype)
    counts = torch.bincount(labels, minlength=num_classes)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Component-conditioned prototype cache is missing classes: {missing}")
    class_sums = torch.zeros(
        num_components, num_classes, features.shape[1],
        device=features.device, dtype=features.dtype,
    )
    class_weight_sums = torch.zeros(
        num_components, num_classes, device=features.device, dtype=features.dtype
    )
    for component_index in range(num_components):
        sample_weights = component_weights[:, component_index]
        class_sums[component_index].index_add_(
            0, labels, features * sample_weights.unsqueeze(-1)
        )
        class_weight_sums[component_index].index_add_(0, labels, sample_weights)
    eps = torch.finfo(features.dtype).eps
    if torch.any(class_weight_sums <= eps):
        raise ValueError("A class/component has insufficient weight to form a prototype.")
    prototypes = F.normalize(
        class_sums / class_weight_sums.unsqueeze(-1), p=2, dim=-1
    )
    class_mean_weights = (class_weight_sums / counts.to(features.dtype).unsqueeze(0)).t()
    return {
        "features": features.detach(),
        "component_weights": component_weights.detach(),
        "labels": labels.detach(),
        "class_sums": class_sums.detach(),
        "class_weight_sums": class_weight_sums.detach(),
        "class_counts": counts.detach(),
        "class_mean_weights": class_mean_weights.detach(),
        "prototypes": prototypes.detach(),
    }


def component_conditioned_prototype_logits(
    features, component_weights, component_conditioned_prototypes, logit_scale=16.0
):
    """Score classes by normalized, query-conditioned prototype recomposition.

    features: [B,D]; component_weights: [B,M]; prototypes: [M,C,D].
    The query weights recompose one class prototype per query/class, then the
    unified query feature is compared with that recomposed prototype.
    """
    if features.ndim != 2 or component_conditioned_prototypes.ndim != 3:
        raise ValueError("Expected unified features [B,D] and prototypes [M,C,D].")
    batch_size, feature_dim = features.shape
    num_components, _, prototype_dim = component_conditioned_prototypes.shape
    if feature_dim != prototype_dim:
        raise ValueError("Unified feature and prototype dimensions must match.")
    _validate_component_weights(component_weights, batch_size, num_components)
    if component_weights.device != features.device:
        raise ValueError("Features and component weights must be on the same device.")
    query = F.normalize(features, p=2, dim=-1)
    prototypes = F.normalize(component_conditioned_prototypes, p=2, dim=-1)
    recomposed = torch.einsum("bm,mcd->bcd", component_weights, prototypes)
    recomposed = F.normalize(recomposed, p=2, dim=-1)
    return torch.einsum("bd,bcd->bc", query, recomposed) * float(logit_scale)


def leave_one_out_component_conditioned_losses(
    student_features,
    component_weights,
    labels,
    dataset_indices,
    component_cache,
    logit_scale=16.0,
):
    """Train final classification and component prototypes from one feature.

    The epoch cache is built from deterministic views. For every query, its
    cached weighted contribution is removed from each class/component sum.
    """
    labels = labels.long()
    dataset_indices = dataset_indices.long()
    if student_features.ndim != 2:
        raise ValueError("Student features must have shape [B,D].")
    batch_size, feature_dim = student_features.shape
    num_components = component_cache["prototypes"].shape[0]
    _validate_component_weights(component_weights, batch_size, num_components)
    if component_weights.device != student_features.device:
        raise ValueError("Student features and component weights must share a device.")
    if labels.ndim != 1 or labels.numel() != batch_size:
        raise ValueError("Labels must contain one class ID per student feature.")
    if dataset_indices.ndim != 1 or dataset_indices.numel() != batch_size:
        raise ValueError("Dataset indices must contain one index per student feature.")
    cache_features = component_cache["features"]
    cache_weights = component_cache["component_weights"]
    cache_labels = component_cache["labels"]
    class_sums = component_cache["class_sums"]
    class_weight_sums = component_cache["class_weight_sums"]
    prototypes = component_cache["prototypes"]
    num_classes = prototypes.shape[1]
    if cache_features.ndim != 2 or cache_features.shape[1] != feature_dim:
        raise ValueError("Cached unified features do not match student features.")
    if cache_weights.shape != (cache_features.shape[0], num_components):
        raise ValueError("Cached component weights have an invalid shape.")
    if class_sums.shape != (num_components, num_classes, feature_dim):
        raise ValueError("Component-conditioned class sums have an invalid shape.")
    if class_weight_sums.shape != (num_components, num_classes):
        raise ValueError("Component-conditioned weight sums have an invalid shape.")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("Labels are outside the component prototype range.")
    if torch.any(dataset_indices < 0) or torch.any(dataset_indices >= cache_labels.numel()):
        raise IndexError("A dataset index falls outside the component cache.")
    if dataset_indices.unique().numel() != dataset_indices.numel():
        raise ValueError("A leave-one-out batch cannot contain duplicate dataset indices.")
    if not torch.equal(cache_labels[dataset_indices], labels):
        raise ValueError("Batch labels do not match the indexed component cache.")

    query = F.normalize(student_features, p=2, dim=-1)
    cached_query = cache_features[dataset_indices]
    cached_weights = cache_weights[dataset_indices]
    selected_sums = class_sums[:, labels, :].permute(1, 0, 2)
    selected_weight_sums = class_weight_sums[:, labels].transpose(0, 1)
    removed_sums = cached_weights.unsqueeze(-1) * cached_query.unsqueeze(1)
    loo_weight_sums = selected_weight_sums - cached_weights
    eps = torch.finfo(student_features.dtype).eps
    if torch.any(loo_weight_sums <= eps):
        raise ValueError("Leave-one-out component prototypes have insufficient remaining weight.")
    loo_prototypes = F.normalize(
        (selected_sums - removed_sums) / loo_weight_sums.unsqueeze(-1), p=2, dim=-1
    )
    normalized_prototypes = F.normalize(prototypes, p=2, dim=-1)
    component_similarities = torch.einsum("bd,mcd->bmc", query, normalized_prototypes)
    loo_component_similarities = torch.einsum("bd,bmd->bm", query, loo_prototypes)

    # Final query-conditioned recomposition, exactly matching FSCIL inference.
    recomposed_prototypes = torch.einsum(
        "bm,mcd->bcd", component_weights, normalized_prototypes
    )
    recomposed_prototypes = F.normalize(recomposed_prototypes, p=2, dim=-1)
    mixture_logits = torch.einsum("bd,bcd->bc", query, recomposed_prototypes)
    loo_recomposed = F.normalize(
        torch.einsum("bm,bmd->bd", component_weights, loo_prototypes), p=2, dim=-1
    )
    loo_target_logits = (query * loo_recomposed).sum(dim=-1)
    mixture_logits = mixture_logits.scatter(
        1, labels.unsqueeze(1), loo_target_logits.unsqueeze(1)
    )
    mixture_loss = F.cross_entropy(mixture_logits * float(logit_scale), labels)

    # Component-wise supervision reuses the same unified query feature. It
    # does not create or encode component-specific visual features.
    component_logits = component_similarities.scatter(
        2,
        labels[:, None, None].expand(-1, num_components, 1),
        loo_component_similarities.unsqueeze(-1),
    )
    component_loss = F.cross_entropy(
        component_logits.reshape(batch_size * num_components, num_classes)
        * float(logit_scale),
        labels[:, None].expand(-1, num_components).reshape(-1),
    )
    return {
        "mixture": mixture_loss,
        "component": component_loss,
        "total": 0.5 * (mixture_loss + component_loss),
    }


def component_sample_affinity_consistency(
    teacher_features,
    student_features,
    component_weights,
    labels,
    teacher_prototypes,
    student_component_conditioned_prototypes,
    affinity_scale=16.0,
):
    """Preserve teacher sample-to-class affinity under the final PCAS classifier."""
    labels = labels.long()
    if teacher_features.ndim != 2 or student_features.ndim != 2:
        raise ValueError("Teacher and student features must have shape [B,D].")
    if teacher_features.shape != student_features.shape:
        raise ValueError("Teacher and student feature shapes must match.")
    if teacher_prototypes.ndim != 2:
        raise ValueError("Teacher prototypes must have shape [C,D].")
    if teacher_prototypes.shape[1] != student_features.shape[1]:
        raise ValueError("Teacher and student feature dimensions must match.")
    if labels.shape != (student_features.shape[0],) or labels.numel() == 0:
        raise ValueError("Labels must contain one class ID per sample.")
    if torch.any(labels < 0) or torch.any(labels >= teacher_prototypes.shape[0]):
        raise ValueError("Labels are outside the teacher prototype range.")
    if teacher_prototypes.shape[0] < 2:
        return student_features.sum() * 0.0
    teacher_logits = F.normalize(teacher_features, dim=-1) @ F.normalize(
        teacher_prototypes, dim=-1
    ).t()
    student_logits = component_conditioned_prototype_logits(
        student_features, component_weights,
        student_component_conditioned_prototypes,
        logit_scale=affinity_scale,
    )
    valid_mask = torch.ones_like(teacher_logits, dtype=torch.bool)
    valid_mask.scatter_(1, labels.unsqueeze(1), False)
    # Finite masking avoids 0 * (-inf) in KL divergence.
    teacher_logits = teacher_logits.masked_fill(~valid_mask, -1e9)
    student_logits = student_logits.masked_fill(~valid_mask, -1e9)
    teacher_distribution = F.softmax(
        teacher_logits * float(affinity_scale), dim=-1
    ).detach()
    student_log_distribution = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(
        student_log_distribution, teacher_distribution, reduction="batchmean"
    )


def component_prototype_affinity_consistency(
    teacher_features, student_features, component_weights, labels, affinity_scale=16.0
):
    """Preserve teacher class-prototype affinities after component recomposition."""
    labels = labels.long()
    if teacher_features.ndim != 2 or student_features.ndim != 2:
        raise ValueError("Teacher and student features must have shape [B,D].")
    if teacher_features.shape != student_features.shape:
        raise ValueError("Teacher and student feature shapes must match.")
    batch_size, feature_dim = student_features.shape
    if labels.shape != (batch_size,) or labels.numel() == 0:
        raise ValueError("Labels must contain one class ID per sample.")
    if torch.any(labels < 0):
        raise ValueError("Class labels must be non-negative.")
    if component_weights.ndim != 2 or component_weights.shape[0] != batch_size:
        raise ValueError("Component weights must have shape [B,M].")
    num_components = component_weights.shape[1]
    _validate_component_weights(component_weights, batch_size, num_components)
    if torch.unique(labels).numel() < 2:
        return student_features.sum() * 0.0

    query = F.normalize(student_features, p=2, dim=-1)
    teacher = F.normalize(teacher_features, p=2, dim=-1)
    _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    num_classes = int(inverse.max().item()) + 1
    counts = torch.bincount(inverse, minlength=num_classes).to(query.dtype)
    teacher_sums = torch.zeros(
        num_classes, feature_dim, device=query.device, dtype=query.dtype
    )
    teacher_sums.index_add_(0, inverse, teacher)
    teacher_class_prototypes = F.normalize(
        teacher_sums / counts.unsqueeze(-1), p=2, dim=-1
    )

    component_sums = torch.zeros(
        num_components, num_classes, feature_dim,
        device=query.device, dtype=query.dtype
    )
    component_mass = torch.zeros(
        num_components, num_classes, device=query.device, dtype=query.dtype
    )
    for component_index in range(num_components):
        sample_weights = component_weights[:, component_index]
        component_sums[component_index].index_add_(
            0, inverse, query * sample_weights.unsqueeze(-1)
        )
        component_mass[component_index].index_add_(0, inverse, sample_weights)
    eps = torch.finfo(query.dtype).eps
    if torch.any(component_mass <= eps):
        raise ValueError("A batch class/component has insufficient weight for affinity consistency.")
    component_prototypes = F.normalize(
        component_sums / component_mass.unsqueeze(-1), p=2, dim=-1
    )

    class_weight_sums = torch.zeros(
        num_classes, num_components, device=query.device, dtype=query.dtype
    )
    class_weight_sums.index_add_(0, inverse, component_weights)
    mean_component_weights = class_weight_sums / counts.unsqueeze(-1)
    student_class_prototypes = F.normalize(
        torch.einsum("cm,mcd->cd", mean_component_weights, component_prototypes),
        p=2,
        dim=-1,
    )

    teacher_relation = teacher_class_prototypes @ teacher_class_prototypes.t()
    student_relation = student_class_prototypes @ student_class_prototypes.t()
    diagonal = torch.eye(num_classes, device=query.device, dtype=torch.bool)
    teacher_relation = teacher_relation.masked_fill(diagonal, -1e9)
    student_relation = student_relation.masked_fill(diagonal, -1e9)
    teacher_distribution = F.softmax(
        teacher_relation * float(affinity_scale), dim=-1
    ).detach()
    student_log_distribution = F.log_softmax(
        student_relation * float(affinity_scale), dim=-1
    )
    return F.kl_div(
        student_log_distribution, teacher_distribution, reduction="batchmean"
    )


@torch.no_grad()
def componentwise_prototype_calibration(
    novel_component_conditioned_prototypes,
    base_component_conditioned_prototypes,
    alpha=0.5,
    tau=16.0,
    top_r=5,
):
    """Calibrate each novel component using only the matching base component."""
    if novel_component_conditioned_prototypes.ndim != 3:
        raise ValueError("Novel prototypes must have shape [Nnovel,M,D].")
    if base_component_conditioned_prototypes.ndim != 3:
        raise ValueError("Base prototypes must have shape [M,Nbase,D].")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Component calibration alpha must be in [0,1].")
    if float(tau) <= 0:
        raise ValueError("Component calibration temperature must be positive.")
    if int(top_r) < 1:
        raise ValueError("top_r must be at least one.")
    _, num_components, feature_dim = novel_component_conditioned_prototypes.shape
    if base_component_conditioned_prototypes.shape[0] != num_components:
        raise ValueError("Base and novel component counts must match.")
    if base_component_conditioned_prototypes.shape[2] != feature_dim:
        raise ValueError("Base and novel prototype dimensions must match.")
    num_base = base_component_conditioned_prototypes.shape[1]
    if num_base < 1:
        raise ValueError("At least one base-class prototype is required.")

    novel = F.normalize(novel_component_conditioned_prototypes, p=2, dim=-1)
    base = F.normalize(base_component_conditioned_prototypes, p=2, dim=-1)
    selected_count = min(int(top_r), num_base)
    calibrated_components, selected_weights, selected_indices = [], [], []
    for component_index in range(num_components):
        similarities = novel[:, component_index] @ base[component_index].t()
        top_scores, top_indices = torch.topk(similarities, k=selected_count, dim=-1)
        weights = F.softmax(top_scores * float(tau), dim=-1)
        transferred = (
            base[component_index][top_indices] * weights.unsqueeze(-1)
        ).sum(dim=1)
        calibrated = F.normalize(
            (1.0 - float(alpha)) * novel[:, component_index]
            + float(alpha) * transferred,
            p=2,
            dim=-1,
        )
        calibrated_components.append(calibrated)
        selected_weights.append(weights)
        selected_indices.append(top_indices)
    return (
        torch.stack(calibrated_components, dim=1),
        torch.stack(selected_weights, dim=1),
        torch.stack(selected_indices, dim=1),
    )
