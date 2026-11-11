"""Prompt-guided Slot Attention over frozen ViT Patch features for FSCIL."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My


class PromptSlotAttention(nn.Module):
    """Deterministic Prompt-Slot Attention with competitive Patch assignment.

    Learned slot queries are shared across images. Each iteration performs
    competition over slots for every Patch, normalizes assignments over Patches
    to aggregate values, and updates slots with a GRU and residual MLP. Slot
    indices are not treated as semantic labels; downstream set matching is
    permutation invariant.
    """

    def __init__(
        self,
        input_dim: int = 768,
        slot_dim: int = 256,
        num_slots: int = 6,
        iterations: int = 2,
        eps: float = 1e-8,
    ):
        super().__init__()
        if input_dim <= 0 or slot_dim <= 0 or num_slots <= 0:
            raise ValueError("input_dim, slot_dim and num_slots must be positive.")
        if iterations <= 0:
            raise ValueError("Slot Attention iterations must be positive.")

        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.iterations = int(iterations)
        self.eps = float(eps)
        self.scale = self.slot_dim ** -0.5

        self.slot_queries = nn.Parameter(
            torch.empty(1, self.num_slots, self.slot_dim)
        )
        nn.init.normal_(self.slot_queries, std=self.slot_dim ** -0.5)

        self.norm_inputs = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)
        self.to_key = nn.Linear(self.input_dim, self.slot_dim, bias=False)
        self.to_value = nn.Linear(self.input_dim, self.slot_dim, bias=False)
        self.to_query = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
        )
        self.output_norm = nn.LayerNorm(self.slot_dim)

    def forward(self, patch_features: torch.Tensor):
        if patch_features.ndim != 3:
            raise ValueError("Patch features must have shape [B, P, C].")
        batch, num_patches, channels = patch_features.shape
        if channels != self.input_dim or num_patches <= 0:
            raise ValueError(
                "Patch feature shape is incompatible with PromptSlotAttention."
            )

        inputs = self.norm_inputs(patch_features)
        keys = self.to_key(inputs)
        values = self.to_value(inputs)
        slots = self.slot_queries.expand(batch, -1, -1)

        assignments = None
        for _ in range(self.iterations):
            previous_slots = slots
            queries = self.to_query(self.norm_slots(slots))
            logits = torch.einsum("bmd,bpd->bmp", queries, keys) * self.scale

            # Competition: each Patch distributes its mass across the slots.
            assignments = torch.softmax(logits, dim=1)
            # Aggregation: each slot pools from the Patch set independently.
            normalized_assignments = assignments + self.eps
            normalized_assignments = normalized_assignments / (
                normalized_assignments.sum(dim=-1, keepdim=True) + self.eps
            )
            updates = torch.einsum(
                "bmp,bpd->bmd", normalized_assignments, values
            )

            slots = self.gru(
                updates.reshape(batch * self.num_slots, self.slot_dim),
                previous_slots.reshape(batch * self.num_slots, self.slot_dim),
            ).reshape(batch, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Assignment mass is image-conditioned and sums to one per image.
        slot_mass = assignments.mean(dim=-1)
        slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        slots = F.normalize(self.output_norm(slots), p=2, dim=-1)
        return slots, slot_mass, assignments


class PromptSlotViT(nn.Module):
    """Frozen ViT-B/16 plus a trainable Prompt-Slot visual decomposition."""

    def __init__(self, args, encoder=None):
        super().__init__()
        self.args = args
        self._incremental_frozen = False

        if encoder is None:
            encoder = model_My.vit_base_patch16_224_in21k(
                pretrained=not bool(getattr(args, "scratch", False)),
                num_classes=0,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
            )
        self.encoder = encoder
        self.embed_dim = int(
            getattr(self.encoder, "num_features",
                    getattr(self.encoder, "embed_dim", 768))
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        self.slot_attention = PromptSlotAttention(
            input_dim=self.embed_dim,
            slot_dim=int(getattr(args, "slot_dim", 256)),
            num_slots=int(getattr(args, "slot_num_slots", 6)),
            iterations=int(getattr(args, "slot_iterations", 2)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self._incremental_frozen:
            self.slot_attention.eval()
        return self

    def forward(self, images: torch.Tensor):
        tokens = self.encoder.forward_features(images)
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise RuntimeError(
                "The ViT must return token features [B, 1+P, C] from forward_features."
            )
        if tokens.shape[-1] != self.embed_dim:
            raise RuntimeError("ViT token dimension changed unexpectedly.")

        global_features = self.encoder.forward_head(tokens)
        patch_features = tokens[:, 1:, :]
        local_features, local_weights, assignments = self.slot_attention(
            patch_features
        )
        return {
            "global_features": F.normalize(global_features, p=2, dim=-1),
            "local_features": local_features,
            "local_weights": local_weights,
            "assignments": assignments,
        }

    def encode(self, images: torch.Tensor):
        return self.forward(images)

    def freeze_for_incremental(self):
        self._incremental_frozen = True
        self.slot_attention.requires_grad_(False)
        self.eval()
