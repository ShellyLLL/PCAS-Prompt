"""Visual semantic decomposition modules for the unified FSCIL method.

The backbone is a frozen ImageNet-21K ViT-B/16. Shared learnable prompts
attend softly to its final Patch features and produce an unordered set of
visual semantic components. No hard assignment, text branch, expert routing,
or incremental parameter expansion is used.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My
from .vsmr_losses import sinkhorn_rectangular_transport


class FrozenViTBackbone(nn.Module):
    """Frozen ViT-B/16 returning normalized CLS and Patch representations."""

    def __init__(self, pretrained=True, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = model_My.vit_base_patch16_224_in21k(
                pretrained=bool(pretrained),
                num_classes=0,
                global_pool="token",
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
            )
        else:
            self.encoder = encoder

        self.embed_dim = int(
            getattr(self.encoder, "num_features",
                    getattr(self.encoder, "embed_dim", 768))
        )
        if encoder is None:
            patch_size = getattr(
                getattr(self.encoder, "patch_embed", None), "patch_size", None
            )
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            elif patch_size is not None:
                patch_size = tuple(patch_size)
            if (
                self.embed_dim != 768
                or len(self.encoder.blocks) != 12
                or patch_size != (16, 16)
            ):
                raise RuntimeError(
                    "VSMR requires the repository's pretrained ViT-B/16 "
                    "(patch 16, 768 dimensions and 12 Transformer blocks)."
                )

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        # timm 0.6.12 exposes the resolved checkpoint metadata as default_cfg.
        # Read it from the instantiated model so preprocessing follows the
        # actual ViT-B/16 weights rather than the downstream dataset defaults.
        pretrained_cfg = getattr(self.encoder, "pretrained_cfg", None)
        if not isinstance(pretrained_cfg, dict) or "mean" not in pretrained_cfg:
            pretrained_cfg = getattr(self.encoder, "default_cfg", {})
        if not isinstance(pretrained_cfg, dict):
            pretrained_cfg = {}
        self.input_mean = tuple(float(x) for x in pretrained_cfg.get(
            "mean", (0.5, 0.5, 0.5)
        ))
        self.input_std = tuple(float(x) for x in pretrained_cfg.get(
            "std", (0.5, 0.5, 0.5)
        ))
        self.input_interpolation = str(
            pretrained_cfg.get("interpolation", "bicubic")
        ).lower()
        self.input_crop_pct = float(pretrained_cfg.get("crop_pct", 0.9))
        if len(self.input_mean) != 3 or len(self.input_std) != 3:
            raise RuntimeError("ViT preprocessing config must define RGB mean/std.")
        if any(value <= 0 for value in self.input_std):
            raise RuntimeError("ViT preprocessing standard deviations must be positive.")
        if not 0.0 < self.input_crop_pct <= 1.0:
            raise RuntimeError("ViT preprocessing crop_pct must be in (0,1].")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images with shape [B,3,H,W].")
        if tuple(images.shape[-2:]) != (224, 224):
            raise ValueError("The frozen ViT-B/16 requires 224x224 images.")

        tokens = self.encoder.patch_embed(images)
        tokens = self.encoder._pos_embed(tokens)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)

        global_features = F.normalize(tokens[:, 0], p=2, dim=-1)
        patch_features = F.normalize(tokens[:, 1:], p=2, dim=-1)
        if patch_features.shape[1] == 0:
            raise RuntimeError("The ViT backbone returned no Patch features.")
        return global_features, patch_features


class VisualSemanticDecomposer(nn.Module):
    """Shared Prompt queries with balanced, permutation-free spatial pooling.

    Prompt indices remain latent and have no fixed semantic labels. Each
    query competes for Patch-token capacity through a balanced transport plan.
    """

    def __init__(self, embed_dim=768, num_parts=6, temperature=1.0,
                 logit_scale=10.0, assignment_epsilon=1.0,
                 sinkhorn_iterations=50):
        super().__init__()
        if int(embed_dim) <= 0 or int(num_parts) < 2:
            raise ValueError("embed_dim must be positive and num_parts >= 2.")
        if float(temperature) <= 0 or float(logit_scale) <= 0:
            raise ValueError("Prompt temperature and logit scale must be positive.")
        if float(assignment_epsilon) <= 0 or int(sinkhorn_iterations) < 1:
            raise ValueError("Assignment epsilon and iterations must be positive.")
        self.embed_dim = int(embed_dim)
        self.num_parts = int(num_parts)
        self.temperature = float(temperature)
        self.logit_scale = float(logit_scale)
        self.assignment_epsilon = float(assignment_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

        self.shared_prompts = nn.Parameter(
            torch.empty(self.num_parts, self.embed_dim)
        )
        nn.init.trunc_normal_(self.shared_prompts, std=0.02)

        self.query_projection = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        self.global_projection = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )

    def forward(self, cls_features, patch_features, return_aux=False):
        if cls_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("Expected CLS [B,D] and Patch [B,N,D] features.")
        batch_size, num_patches, feature_dim = patch_features.shape
        if num_patches < self.num_parts:
            raise ValueError("The number of Patch tokens must be >= num_parts.")
        if feature_dim != self.embed_dim:
            raise ValueError("Patch feature dimension does not match VSMR.")
        if cls_features.shape != (batch_size, self.embed_dim):
            raise ValueError("CLS and Patch feature dimensions do not match.")

        # Normalize query/key directions so attention scale is controlled by
        # explicit temperature and logit-scale parameters, not vector norms.
        queries = F.normalize(
            self.query_projection(self.shared_prompts), p=2, dim=-1
        )
        keys = F.normalize(self.key_projection(patch_features), p=2, dim=-1)
        values = self.value_projection(patch_features)

        attention_logits = torch.einsum(
            "kd,bnd->bkn", queries, keys
        ) * (self.logit_scale / self.temperature)
        assignment = sinkhorn_rectangular_transport(
            attention_logits,
            epsilon=self.assignment_epsilon,
            iterations=self.sinkhorn_iterations,
        )
        # The transport has row mass 1/K. Rescale each row to sum to one for
        # weighted feature pooling; its column sums remain K/N.
        attention = assignment * float(self.num_parts)
        local_parts = torch.bmm(attention, values)
        local_parts = F.normalize(local_parts, p=2, dim=-1)
        global_features = F.normalize(
            self.global_projection(cls_features), p=2, dim=-1
        )
        if return_aux:
            return global_features, local_parts, attention, queries, assignment
        return global_features, local_parts, attention

class ReliabilityGate(nn.Module):
    """Learned component-wise transfer reliability gate.

    Inputs are entirely visual: novel component, retrieved base component,
    their interaction/difference, retrieval confidence/ambiguity, support
    dispersion, and the number of support shots.
    """

    def __init__(self, embed_dim=768, hidden_dim=None):
        super().__init__()
        embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim or max(128, embed_dim // 2))
        input_dim = 4 * embed_dim + 5
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Conservative initial transfer; the gate is subsequently learned
        # from pseudo-incremental classification episodes.
        nn.init.constant_(self.network[-1].bias, -1.5)

    @staticmethod
    def _scalar_feature(value, reference, name):
        if value is None:
            return torch.zeros_like(reference)
        if not torch.is_tensor(value):
            value = reference.new_tensor(float(value))
        else:
            value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            return value.expand_as(reference)
        if value.ndim == 1 and value.shape[0] == reference.shape[0]:
            return value[:, None].expand_as(reference)
        if value.shape != reference.shape:
            raise ValueError("{} must be scalar, [C], or [C,K].".format(name))
        return value

    def forward(self, novel, retrieved, mean_similarity, similarity_margin,
                retrieval_entropy, support_variance=None, shots=1):
        if novel.ndim != 3 or retrieved.shape != novel.shape:
            raise ValueError("Gate inputs must share [C,K,D] shapes.")
        reference = mean_similarity
        margin = self._scalar_feature(
            similarity_margin, reference, "similarity_margin"
        )
        entropy = self._scalar_feature(
            retrieval_entropy, reference, "retrieval_entropy"
        )
        variance = self._scalar_feature(
            support_variance, reference, "support_variance"
        )
        shot_values = self._scalar_feature(shots, reference, "shots")
        log_shots = shot_values.clamp_min(1.0).log()

        features = torch.cat([
            novel,
            retrieved,
            novel * retrieved,
            (novel - retrieved).abs(),
            reference.unsqueeze(-1),
            margin.unsqueeze(-1),
            entropy.unsqueeze(-1),
            variance.unsqueeze(-1),
            log_shots.unsqueeze(-1),
        ], dim=-1)
        if features.shape[-1] != self.input_dim:
            raise RuntimeError("Reliability gate feature dimension mismatch.")
        return torch.sigmoid(self.network(features).squeeze(-1))


class QueryConditionedReconstructor(nn.Module):
    """Cross-attention reconstruction of class parts from query Patch tokens.

    Class prototypes provide the queries/residual content. Query Patch tokens
    provide keys and values. The output is ephemeral and never written back to
    persistent class memory.
    """

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query_patches, class_parts):
        if query_patches.ndim != 3:
            raise ValueError("query_patches must have shape [B,N,D].")
        if class_parts.ndim not in (3, 4):
            raise ValueError("class_parts must have shape [C,K,D] or [B,C,K,D].")
        batch, num_patches, dim = query_patches.shape
        if dim != self.embed_dim or class_parts.shape[-1] != self.embed_dim:
            raise ValueError("Query and prototype embedding dimensions must match.")
        if class_parts.ndim == 3:
            class_parts = class_parts.unsqueeze(0).expand(
                batch, -1, -1, -1
            )
        elif class_parts.shape[0] != batch:
            raise ValueError("Per-query class prototypes must match query batch size.")

        queries = self.query_projection(class_parts)
        keys = self.key_projection(query_patches)
        values = self.value_projection(query_patches)
        logits = torch.einsum("bckd,bnd->bckn", queries, keys)
        logits = logits / math.sqrt(self.embed_dim)
        attention = torch.softmax(logits, dim=-1)
        context = torch.einsum("bckn,bnd->bckd", attention, values)
        reconstructed = self.norm(queries + context)
        reconstructed = F.normalize(reconstructed, p=2, dim=-1)
        return reconstructed, attention


class GlobalLocalFusion(nn.Module):
    """Learn a single global/local mixture coefficient during base training."""

    def __init__(self, global_init=0.5, local_init=0.5):
        super().__init__()
        global_init = float(global_init)
        local_init = float(local_init)
        if global_init < 0 or local_init < 0 or global_init + local_init <= 0:
            raise ValueError("Initial global/local fusion weights are invalid.")
        alpha = global_init / (global_init + local_init)
        alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
        self.logit = nn.Parameter(torch.tensor(
            math.log(alpha / (1.0 - alpha)), dtype=torch.float32
        ))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit)

    def forward(self, global_scores, local_scores):
        if global_scores.shape != local_scores.shape:
            raise ValueError("Global and local scores must have identical shapes.")
        alpha = self.alpha.to(dtype=global_scores.dtype)
        return alpha * global_scores + (1.0 - alpha) * local_scores


class VSMRViT(nn.Module):
    """Frozen ViT plus shared decomposition, learned gate and query reconstructor."""

    def __init__(self, args, encoder=None):
        super().__init__()
        if bool(getattr(args, "scratch", False)):
            raise ValueError(
                "VSMR requires the pretrained ImageNet-21K ViT-B/16; "
                "scratch backbone initialization is not supported."
            )
        self.backbone = FrozenViTBackbone(pretrained=True, encoder=encoder)
        dim = self.backbone.embed_dim
        self.decomposer = VisualSemanticDecomposer(
            embed_dim=dim,
            num_parts=int(getattr(args, "vsmr_num_parts", 6)),
            temperature=float(getattr(args, "vsmr_prompt_temperature", 1.0)),
            logit_scale=float(getattr(args, "vsmr_prompt_logit_scale", 10.0)),
            assignment_epsilon=float(getattr(
                args, "vsmr_prompt_assignment_epsilon", 1.0
            )),
            sinkhorn_iterations=int(getattr(
                args, "vsmr_prompt_sinkhorn_iterations", 50
            )),
        )
        self.reliability_gate = ReliabilityGate(
            embed_dim=dim,
            hidden_dim=int(getattr(args, "vsmr_gate_hidden_dim", max(128, dim // 2))),
        )
        self.reconstructor = QueryConditionedReconstructor(embed_dim=dim)
        self.fusion = GlobalLocalFusion(
            global_init=float(getattr(args, "vsmr_global_weight", 0.5)),
            local_init=float(getattr(args, "vsmr_local_weight_eval", 0.5)),
        )
        self._incremental_frozen = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self._incremental_frozen:
            self.decomposer.eval()
            self.reliability_gate.eval()
            self.reconstructor.eval()
            self.fusion.eval()
        return self

    def forward(self, images):
        cls_features, patch_features = self.backbone(images)
        global_features, parts, attention, prompt_queries, assignment = (
            self.decomposer(cls_features, patch_features, return_aux=True)
        )
        return {
            "global_features": global_features,
            "parts": parts,
            "attention": attention,
            "patch_features": patch_features,
            "prompt_queries": prompt_queries,
            "assignment": assignment,
        }

    def freeze_for_incremental(self):
        """Freeze every learned component before the first incremental session."""
        self._incremental_frozen = True
        self.decomposer.requires_grad_(False)
        self.reliability_gate.requires_grad_(False)
        self.reconstructor.requires_grad_(False)
        self.fusion.requires_grad_(False)
        self.eval()
