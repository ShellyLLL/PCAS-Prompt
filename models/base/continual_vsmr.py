"""Core utilities for Continual VSMR.

The module deliberately contains no data-loader policy. It implements the
feature-memory contract, current-Prompt decomposition, class OT prototypes,
and differentiable LOO prototype construction used by the trainer.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from .vsmr_losses import ot_barycenter
from .vsmr_resynthesis import query_conditioned_reconstruction


@dataclass
class FeatureRecord:
    cls: torch.Tensor       # [D], frozen ViT CLS
    patches: torch.Tensor   # [N,D], frozen ViT Patch tokens
    label: int
    sample_id: str
    session: int


class FrozenFeatureMemory:
    """CPU FP16 feature replay; no raw images and no evaluation samples."""
    def __init__(self, max_base_per_class: int):
        if max_base_per_class < 2:
            raise ValueError("max_base_per_class must be >=2 for exact LOO.")
        self.max_base_per_class = int(max_base_per_class)
        self.records = []
        self._ids = set()

    def append(self, cls, patches, labels, sample_ids, session, is_base=False):
        if patches.ndim != 3 or labels.ndim != 1:
            raise ValueError("Expected patches [B,N,D] and labels [B].")
        if len(sample_ids) != labels.numel() or patches.shape[0] != labels.numel():
            raise ValueError("Feature, label and sample-id counts differ.")
        counts: Dict[int, int] = {}
        for label in labels.detach().cpu().tolist():
            counts[int(label)] = counts.get(int(label), 0) + 1
        existing = {}
        for record in self.records:
            existing[record.label] = existing.get(record.label, 0) + 1
        for i, label in enumerate(labels.detach().cpu().tolist()):
            sid = str(sample_ids[i])
            if sid in self._ids:
                raise RuntimeError("Duplicate sample id in feature memory: " + sid)
            label = int(label)
            if is_base and existing.get(label, 0) >= self.max_base_per_class:
                continue
            if is_base and existing.get(label, 0) + counts.get(label, 0) > self.max_base_per_class:
                # Deterministic first-m samples are selected by the caller.
                if existing.get(label, 0) >= self.max_base_per_class:
                    continue
            self.records.append(FeatureRecord(
                cls=torch.zeros(patches.shape[-1], dtype=torch.float16),
                patches=patches[i].detach().to(device="cpu", dtype=torch.float16).contiguous(),
                label=label, sample_id=sid, session=int(session)))
            self._ids.add(sid)
            existing[label] = existing.get(label, 0) + 1

    def tensors(self, device):
        if not self.records:
            raise RuntimeError("Feature memory is empty.")
        cls = torch.stack([r.cls for r in self.records]).to(device=device, dtype=torch.float32)
        patches = torch.stack([r.patches for r in self.records]).to(device=device, dtype=torch.float32)
        labels = torch.tensor([r.label for r in self.records], device=device, dtype=torch.long)
        return cls, patches, labels

    def metadata(self):
        return [{"label": r.label, "sample_id": r.sample_id, "session": r.session}
                for r in self.records]

    def __len__(self):
        return len(self.records)


def decompose_cached(model, cls_features, patch_features):
    """Run only the trainable shared-query decomposer on frozen ViT features."""
    return model.decomposer(cls_features, patch_features)


def class_ot_prototypes(parts, labels, epsilon=0.05, iterations=30,
                        barycenter_iterations=5):
    classes = torch.unique(labels, sorted=True)
    prototypes = []
    for cls in classes:
        values = parts[labels == cls]
        if values.shape[0] < 2:
            raise ValueError(
                "Every seen class needs >=2 memory samples for exact LOO; "
                "increase the Base memory budget or use >=2 support shots."
            )
        prototypes.append(ot_barycenter(
            values, epsilon, iterations, barycenter_iterations))
    return classes, torch.stack(prototypes)


def loo_class_prototypes(parts, labels, query_indices, classes,
                         epsilon=0.05, iterations=30,
                         barycenter_iterations=5):
    """Build [B,C,M,D] candidates; a query is excluded only from its own class."""
    full = {}
    for cls in classes.tolist():
        values = parts[labels == int(cls)]
        if values.shape[0] < 2:
            raise ValueError("Exact LOO requires at least two exemplars per class.")
        full[int(cls)] = ot_barycenter(
            values, epsilon, iterations, barycenter_iterations)
    rows = []
    for index in query_indices.tolist():
        label = int(labels[index].item())
        row = []
        for cls in classes.tolist():
            cls = int(cls)
            if cls != label:
                row.append(full[cls])
            else:
                mask = (labels == cls)
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                indices = indices[indices != index]
                if indices.numel() < 1:
                    raise RuntimeError("LOO removed the only class exemplar.")
                row.append(ot_barycenter(
                    parts[indices], epsilon, iterations, barycenter_iterations))
        rows.append(torch.stack(row))
    return torch.stack(rows)


def prompt_consistency_loss(student_parts, teacher_parts, labels, mask=None,
                            epsilon=0.05, iterations=30):
    """Mean permutation-invariant OT consistency over selected old samples."""
    if student_parts.shape != teacher_parts.shape:
        raise ValueError("Student/teacher component tensor shapes differ.")
    if mask is not None:
        student_parts = student_parts[mask]
        teacher_parts = teacher_parts[mask]
    if student_parts.shape[0] == 0:
        return student_parts.sum() * 0.0
    distances = []
    from .vsmr_losses import ot_distance
    for i in range(student_parts.shape[0]):
        distances.append(ot_distance(
            student_parts[i:i+1], teacher_parts[i:i+1],
            epsilon, iterations).mean())
    return torch.stack(distances).mean()


def distillation_kl(student_logits, teacher_logits, temperature=2.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student distillation logits must align.")
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits.detach() / t, dim=-1),
        reduction="batchmean") * (t * t)


def build_query_logits(query_parts, query_patches, candidate_parts,
                       reconstructor, epsilon=0.05, iterations=30, scale=16.0):
    return query_conditioned_reconstruction(
        query_parts, query_patches, candidate_parts, reconstructor,
        epsilon=epsilon, iterations=iterations, scale=scale)[0]
