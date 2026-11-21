"""Core utilities for Continual VSMR feature replay and OT prototypes."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .vsmr_losses import ot_barycenter, ot_distance
from .vsmr_resynthesis import query_conditioned_reconstruction


@dataclass
class FeatureRecord:
    cls: torch.Tensor
    patches: torch.Tensor
    label: int
    sample_id: str
    session: int


class FrozenFeatureMemory:
    """CPU FP16 cache of frozen ViT CLS/Patch features; never stores test data."""
    def __init__(self, max_base_per_class):
        if int(max_base_per_class) < 2:
            raise ValueError("max_base_per_class must be >=2 for exact LOO.")
        self.max_base_per_class = int(max_base_per_class)
        self.records, self._ids = [], set()

    def append(self, cls_features, patches, labels, sample_ids, session,
               is_base=False):
        if cls_features.ndim != 2 or patches.ndim != 3 or labels.ndim != 1:
            raise ValueError("Expected CLS [B,D], Patch [B,N,D], labels [B].")
        n = labels.numel()
        if cls_features.shape[0] != n or patches.shape[0] != n or len(sample_ids) != n:
            raise ValueError("Feature, label and sample-id counts differ.")
        existing = {}
        for record in self.records:
            existing[record.label] = existing.get(record.label, 0) + 1
        for i, label in enumerate(labels.detach().cpu().tolist()):
            sid, label = str(sample_ids[i]), int(label)
            if sid in self._ids:
                raise RuntimeError("Duplicate sample id: " + sid)
            if is_base and existing.get(label, 0) >= self.max_base_per_class:
                continue
            self.records.append(FeatureRecord(
                cls=cls_features[i].detach().cpu().half().contiguous(),
                patches=patches[i].detach().cpu().half().contiguous(),
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
    return model.decomposer(cls_features, patch_features)


def class_ot_prototypes(parts, labels, epsilon=0.05, iterations=30,
                        barycenter_iterations=5):
    classes = torch.unique(labels, sorted=True)
    prototypes = []
    for cls in classes:
        values = parts[labels == cls]
        if values.shape[0] < 2:
            raise ValueError("Every seen class needs >=2 exemplars for exact LOO.")
        prototypes.append(ot_barycenter(values, epsilon, iterations, barycenter_iterations))
    return classes, torch.stack(prototypes)


def loo_class_prototypes(parts, labels, query_indices, classes,
                         epsilon=0.05, iterations=30, barycenter_iterations=5):
    """Return [B,C,M,D], excluding each query from its own class prototype."""
    full = {}
    for cls in classes.tolist():
        values = parts[labels == int(cls)]
        if values.shape[0] < 2:
            raise ValueError("Exact LOO requires >=2 memory samples per class.")
        full[int(cls)] = ot_barycenter(values, epsilon, iterations, barycenter_iterations)
    rows = []
    for index in query_indices.tolist():
        label = int(labels[index].item())
        row = []
        for cls in classes.tolist():
            cls = int(cls)
            if cls == label:
                retained = torch.nonzero(labels == cls, as_tuple=False).flatten()
                retained = retained[retained != index]
                row.append(ot_barycenter(parts[retained], epsilon, iterations, barycenter_iterations))
            else:
                row.append(full[cls])
        rows.append(torch.stack(row))
    return torch.stack(rows)


def prompt_consistency_loss(student_parts, teacher_parts, mask=None,
                            epsilon=0.05, iterations=30):
    if student_parts.shape != teacher_parts.shape:
        raise ValueError("Student/teacher component shapes differ.")
    if mask is not None:
        student_parts, teacher_parts = student_parts[mask], teacher_parts[mask]
    if student_parts.shape[0] == 0:
        return student_parts.sum() * 0.0
    return torch.stack([
        ot_distance(student_parts[i:i+1], teacher_parts[i:i+1], epsilon, iterations).mean()
        for i in range(student_parts.shape[0])
    ]).mean()


def distillation_kl(student_logits, teacher_logits, temperature=2.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student logits must align.")
    t = float(temperature)
    return F.kl_div(F.log_softmax(student_logits / t, dim=-1),
                    F.softmax(teacher_logits.detach() / t, dim=-1),
                    reduction="batchmean") * (t * t)


def build_query_logits(query_parts, query_patches, candidate_parts,
                       reconstructor, epsilon=0.05, iterations=30, scale=16.0):
    return query_conditioned_reconstruction(
        query_parts, query_patches, candidate_parts, reconstructor,
        epsilon=epsilon, iterations=iterations, scale=scale)[0]
