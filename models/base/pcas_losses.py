"""Prototype and affinity objectives for decomposed spatial prompting."""
import torch
import torch.nn.functional as F


def _validate_features_labels(features, labels):
    if features.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected features [N,D] and labels [N].")
    if features.shape[0] != labels.numel():
        raise ValueError("Features and labels have incompatible sample counts.")
    if labels.numel() == 0:
        raise ValueError("Cannot build prototypes from an empty dataset.")


def build_prototype_cache(features, labels, num_classes):
    """Build normalized global class prototypes and indexed sample features."""
    labels = labels.long()
    _validate_features_labels(features, labels)
    if torch.any(labels < 0) or torch.any(labels >= int(num_classes)):
        raise ValueError("Prototype labels are outside [0, num_classes).")

    counts = torch.bincount(labels, minlength=int(num_classes))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Prototype cache is missing classes: {missing}")

    class_sums = torch.zeros(
        int(num_classes), features.shape[1],
        device=features.device, dtype=features.dtype,
    )
    class_sums.index_add_(0, labels, features)
    class_means = class_sums / counts.to(features.dtype).unsqueeze(1)
    prototypes = F.normalize(class_means, p=2, dim=-1)

    return {
        "features": features.detach(),
        "labels": labels.detach(),
        "class_sums": class_sums.detach(),
        "class_counts": counts.detach(),
        "prototypes": prototypes.detach(),
    }


def build_component_prototype_cache(component_features, labels, num_classes):
    """Build one prototype per base class and spatial component.

    Args:
        component_features: [N,M,D] isolated-component ViT features.
        labels: [N] global base-class labels in the dataset's local order.
    Returns:
        A cache whose prototypes and class sums have shape [M,C,D].
    """
    labels = labels.long()
    if component_features.ndim != 3:
        raise ValueError("component_features must have shape [N,M,D].")
    if component_features.shape[0] != labels.numel() or labels.ndim != 1:
        raise ValueError("Component features and labels have incompatible shapes.")
    if labels.numel() == 0:
        raise ValueError("Cannot build component prototypes from an empty dataset.")
    if torch.any(labels < 0) or torch.any(labels >= int(num_classes)):
        raise ValueError("Component prototype labels are outside [0, num_classes).")

    features = F.normalize(component_features, p=2, dim=-1)
    counts = torch.bincount(labels, minlength=int(num_classes))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Component prototype cache is missing classes: {missing}")

    num_components, feature_dim = features.shape[1], features.shape[2]
    class_sums = torch.zeros(
        num_components, int(num_classes), feature_dim,
        device=features.device, dtype=features.dtype,
    )
    for component_index in range(num_components):
        class_sums[component_index].index_add_(
            0, labels, features[:, component_index, :]
        )
    class_means = class_sums / counts.to(features.dtype)[None, :, None]
    prototypes = F.normalize(class_means, p=2, dim=-1)

    return {
        "features": features.detach(),
        "labels": labels.detach(),
        "class_sums": class_sums.detach(),
        "class_counts": counts.detach(),
        "prototypes": prototypes.detach(),
    }


def component_prototype_logits(
    component_features,
    component_weights,
    component_prototypes,
    logit_scale=16.0,
):
    """Class logits from query-weighted similarities to component prototypes.

    Shapes:
        component_features: [B,M,D]
        component_weights: [B,M]
        component_prototypes: [M,C,D]
    """
    if component_features.ndim != 3 or component_prototypes.ndim != 3:
        raise ValueError("Component features/prototypes must be rank-3 tensors.")
    if component_weights.ndim != 2:
        raise ValueError("component_weights must have shape [B,M].")
    batch_size, num_components, feature_dim = component_features.shape
    if component_weights.shape != (batch_size, num_components):
        raise ValueError("Component weights do not align with component features.")
    if component_prototypes.shape[0] != num_components:
        raise ValueError("Prototype component count does not match feature components.")
    if component_prototypes.shape[2] != feature_dim:
        raise ValueError("Component feature dimensions do not match prototypes.")
    if not torch.isfinite(component_weights).all() or torch.any(component_weights < 0):
        raise ValueError("Component weights must be finite and non-negative.")
    if not torch.allclose(
        component_weights.sum(dim=-1),
        torch.ones(batch_size, device=component_weights.device,
                   dtype=component_weights.dtype),
        atol=1e-5,
    ):
        raise ValueError("Each component-weight row must sum to one.")

    features = F.normalize(component_features, p=2, dim=-1)
    prototypes = F.normalize(component_prototypes, p=2, dim=-1)
    component_similarities = torch.einsum(
        "bmd,mcd->bmc", features, prototypes
    )
    logits = (
        component_similarities * component_weights.unsqueeze(-1)
    ).sum(dim=1)
    return logits * float(logit_scale)


def leave_one_out_component_prototype_loss(
    student_component_features,
    component_weights,
    labels,
    dataset_indices,
    component_cache,
    logit_scale=16.0,
):
    """Component-weighted prototype CE with each query removed from its class mean."""
    labels = labels.long()
    dataset_indices = dataset_indices.long()
    if student_component_features.ndim != 3:
        raise ValueError("Student component features must have shape [B,M,D].")

    cache_features = component_cache["features"]
    cache_labels = component_cache["labels"]
    class_sums = component_cache["class_sums"]
    class_counts = component_cache["class_counts"]
    prototypes = component_cache["prototypes"]

    batch_size, num_components, feature_dim = student_component_features.shape
    if component_weights.shape != (batch_size, num_components):
        raise ValueError("Component weights do not align with student features.")
    if cache_features.ndim != 3 or cache_features.shape[1:] != (
        num_components, feature_dim
    ):
        raise ValueError("Component cache shape does not match student features.")
    if dataset_indices.ndim != 1 or dataset_indices.numel() != batch_size:
        raise ValueError("Dataset indices must contain one index per sample.")
    if torch.any(dataset_indices < 0) or torch.any(dataset_indices >= cache_labels.numel()):
        raise IndexError("A dataset index falls outside the component cache.")
    if not torch.equal(cache_labels[dataset_indices], labels):
        raise ValueError("Batch labels do not match the indexed component cache.")
    if torch.any(class_counts[labels] <= 1):
        raise ValueError("Leave-one-out prototypes require at least two samples per class.")

    # [M,B,D] -> [B,M,D]
    selected_sums = class_sums[:, labels, :].permute(1, 0, 2)
    cached_samples = cache_features[dataset_indices]
    denominator = (class_counts[labels] - 1).to(
        student_component_features.dtype
    ).view(batch_size, 1, 1)
    loo_means = (selected_sums - cached_samples) / denominator
    loo_prototypes = F.normalize(loo_means, p=2, dim=-1)

    normalized_features = F.normalize(
        student_component_features, p=2, dim=-1
    )
    normalized_prototypes = F.normalize(prototypes, p=2, dim=-1)
    all_component_similarities = torch.einsum(
        "bmd,mcd->bmc", normalized_features, normalized_prototypes
    )
    loo_target_similarities = (
        normalized_features * loo_prototypes
    ).sum(dim=-1)

    logits = (
        all_component_similarities * component_weights.unsqueeze(-1)
    ).sum(dim=1)
    loo_target_logits = (
        loo_target_similarities * component_weights
    ).sum(dim=1)
    logits = logits.scatter(
        1, labels.unsqueeze(1), loo_target_logits.unsqueeze(1)
    )
    return F.cross_entropy(logits * float(logit_scale), labels)


def sample_affinity_consistency(
    teacher_features,
    student_features,
    labels,
    teacher_prototypes,
    student_prototypes,
    affinity_scale=16.0,
):
    """Preserve each sample's affinity distribution to non-target base classes."""
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
    """Preserve the pairwise class-prototype affinity structure in each batch."""
    labels = labels.long()
    classes, inverse = torch.unique(
        labels, sorted=True, return_inverse=True
    )
    if classes.numel() < 2:
        return student_features.sum() * 0.0

    num_classes = classes.numel()
    teacher_means = torch.zeros(
        num_classes, teacher_features.shape[1],
        device=teacher_features.device, dtype=teacher_features.dtype,
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


@torch.no_grad()
def componentwise_prototype_calibration(
    novel_component_prototypes,
    base_component_prototypes,
    alpha=0.5,
    tau=16.0,
    top_r=5,
):
    """Calibrate each novel component from the matching base component only.

    Args:
        novel_component_prototypes: [Nnovel,M,D].
        base_component_prototypes: [M,Nbase,D].
    Returns:
        calibrated prototypes [Nnovel,M,D], selected weights [Nnovel,M,R],
        and selected base-class indices [Nnovel,M,R].
    """
    if novel_component_prototypes.ndim != 3:
        raise ValueError("Novel component prototypes must have shape [N,M,D].")
    if base_component_prototypes.ndim != 3:
        raise ValueError("Base component prototypes must have shape [M,C,D].")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Component calibration alpha must be in [0,1].")
    if float(tau) <= 0:
        raise ValueError("Component calibration temperature must be positive.")

    num_novel, num_components, feature_dim = novel_component_prototypes.shape
    if base_component_prototypes.shape[0] != num_components:
        raise ValueError("Base and novel component counts must match.")
    if base_component_prototypes.shape[2] != feature_dim:
        raise ValueError("Base and novel component feature dimensions must match.")
    num_base = base_component_prototypes.shape[1]
    if num_base < 1:
        raise ValueError("At least one base class prototype is required.")

    selected_count = min(int(top_r), num_base)
    if selected_count < 1:
        raise ValueError("top_r must be at least one.")

    novel = F.normalize(novel_component_prototypes, p=2, dim=-1)
    base = F.normalize(base_component_prototypes, p=2, dim=-1)
    calibrated_components = []
    selected_weights = []
    selected_indices = []

    for component_index in range(num_components):
        similarities = novel[:, component_index] @ base[component_index].t()
        top_scores, top_indices = torch.topk(
            similarities, k=selected_count, dim=-1
        )
        weights = F.softmax(top_scores * float(tau), dim=-1)
        selected_base = base[component_index][top_indices]
        transferred = (
            selected_base * weights.unsqueeze(-1)
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

    calibrated = torch.stack(calibrated_components, dim=1)
    weights = torch.stack(selected_weights, dim=1)
    indices = torch.stack(selected_indices, dim=1)
    return calibrated, weights, indices
