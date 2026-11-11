"""Frozen ViT-B/16, APL-inspired shared part prompts, and PRSN modules.

The APL adaptation uses the existing ImageNet-21K ViT's own final-block
CLS-to-Patch attention as a prior. No DINO model or extra prompt tokens are
introduced: shared learnable part queries attend to Patch features after the
frozen backbone and induce cross-image part correspondence.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My


class FrozenViTBackbone(nn.Module):
    """Frozen ImageNet-21K ViT-B/16 returning CLS, Patch and attention priors."""

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
            getattr(self.encoder, "num_features", getattr(self.encoder, "embed_dim", 768))
        )
        self.num_blocks = len(self.encoder.blocks)
        if encoder is None and (self.embed_dim != 768 or self.num_blocks != 12):
            raise RuntimeError(
                "The active method requires the existing ViT-B/16 "
                "(768 dimensions, 12 Transformer blocks)."
            )

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB images with shape [B,3,H,W].")
        if tuple(images.shape[-2:]) != (224, 224):
            raise ValueError("The frozen ViT-B/16 requires 224x224 input images.")

        tokens = self.encoder.patch_embed(images)
        tokens = self.encoder._pos_embed(tokens)
        patch_attention = None

        for block_index, block in enumerate(self.encoder.blocks):
            if block_index == self.num_blocks - 1:
                # Reproduce the final ViT block's CLS-to-Patch attention from
                # its frozen Q/K projections; do not instantiate another model.
                normalized = block.norm1(tokens)
                batch_size, token_count, channels = normalized.shape
                attention = block.attn
                qkv = attention.qkv(normalized).reshape(
                    batch_size,
                    token_count,
                    3,
                    attention.num_heads,
                    channels // attention.num_heads,
                ).permute(2, 0, 3, 1, 4)
                query, key, _ = qkv.unbind(0)
                cls_to_all = (
                    query[:, :, :1, :] @ key.transpose(-2, -1)
                ) * attention.scale
                patch_attention = cls_to_all.softmax(dim=-1).squeeze(2)[:, :, 1:]

            tokens = block(tokens)

        if patch_attention is None:
            raise RuntimeError("The ViT backbone did not expose final-block Patch priors.")

        tokens = self.encoder.norm(tokens)
        global_features = F.normalize(tokens[:, 0], p=2, dim=-1)
        patch_features = F.normalize(tokens[:, 1:], p=2, dim=-1)
        return global_features, patch_features, patch_attention


class APLPartPrompt(nn.Module):
    """Shared learnable part-prompt queries with APL-style hard Patch assignment."""

    def __init__(
        self,
        embed_dim=768,
        num_parts=6,
        num_heads=12,
        prior_ratio=0.20,
        temperature=0.20,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by apl_num_heads.")
        if int(num_parts) < 2:
            raise ValueError("APL requires at least two shared part queries.")
        if not 0.0 < float(prior_ratio) <= 1.0:
            raise ValueError("apl_prior_ratio must be in (0,1].")
        if float(temperature) <= 0:
            raise ValueError("apl_temperature must be positive.")

        self.embed_dim = int(embed_dim)
        self.num_parts = int(num_parts)
        self.prior_ratio = float(prior_ratio)
        self.temperature = float(temperature)

        self.shared_part_queries = nn.Parameter(
            torch.empty(self.num_parts, self.embed_dim)
        )
        nn.init.trunc_normal_(self.shared_part_queries, std=0.02)

        self.prior_norm = nn.LayerNorm(self.embed_dim)
        self.query_attention = nn.MultiheadAttention(
            self.embed_dim, int(num_heads), batch_first=True
        )
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.patch_query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.patch_key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.part_norm = nn.LayerNorm(self.embed_dim)

    def _part_priors(self, patches, attention_priors):
        if patches.ndim != 3 or attention_priors.ndim != 3:
            raise ValueError("Expected Patch features [B,N,D] and priors [B,H,N].")
        batch_size, num_patches, channels = patches.shape
        if channels != self.embed_dim or attention_priors.shape[0] != batch_size:
            raise ValueError("APL Patch/prior dimensions are incompatible.")
        if attention_priors.shape[-1] != num_patches:
            raise ValueError("APL attention priors do not match the Patch count.")

        selected = max(1, int(math.ceil(num_patches * self.prior_ratio)))
        selected_indices = attention_priors.topk(selected, dim=-1).indices
        masks = torch.zeros_like(attention_priors).scatter_(
            -1, selected_indices, 1.0
        )
        weights = attention_priors * masks
        mass = weights.sum(dim=-1, keepdim=True)
        priors = torch.einsum("bhn,bnd->bhd", weights, patches)
        priors = priors / mass.clamp_min(torch.finfo(patches.dtype).eps)
        return self.prior_norm(priors)

    def forward(self, patches, attention_priors):
        batch_size, num_patches, _ = patches.shape
        part_priors = self._part_priors(patches, attention_priors)

        shared_queries = self.shared_part_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        attended_queries, _ = self.query_attention(
            shared_queries, part_priors, part_priors, need_weights=False
        )
        image_queries = self.query_norm(shared_queries + attended_queries)

        assignment_logits = torch.bmm(
            self.patch_query(image_queries),
            self.patch_key(patches).transpose(1, 2),
        ) / math.sqrt(self.embed_dim)

        if self.training:
            # Each Patch is assigned to one shared query; the straight-through
            # estimator preserves gradients through the hard correspondence.
            assignments = F.gumbel_softmax(
                assignment_logits,
                tau=self.temperature,
                hard=True,
                dim=1,
            )
        else:
            hard_indices = assignment_logits.argmax(dim=1)
            assignments = F.one_hot(
                hard_indices, num_classes=self.num_parts
            ).permute(0, 2, 1).to(dtype=patches.dtype)

        part_mass = assignments.sum(dim=-1)
        parts = torch.bmm(assignments, patches)
        parts = parts / part_mass.clamp_min(1.0).unsqueeze(-1)
        part_valid = part_mass > 0
        parts = self.part_norm(parts) * part_valid.unsqueeze(-1)
        return parts, part_valid, assignments, image_queries


class GlobalPrototypeRefiner(nn.Module):
    """PRSN global-branch prototype self-attention with immutable base memory."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def _refine(self, queries, context):
        # queries [B,D], context [B,K,D]
        q = self.query(queries).unsqueeze(1)
        k = self.key(context)
        v = self.value(context)
        attention = torch.softmax(
            torch.matmul(q, k.transpose(1, 2)) / math.sqrt(queries.shape[-1]),
            dim=-1,
        )
        update = torch.matmul(attention, v).squeeze(1)
        return F.normalize(self.norm(queries + self.output(update)), dim=-1)

    def refine_base(self, prototypes):
        if prototypes.ndim != 2 or prototypes.shape[0] == 0:
            raise ValueError("Base global prototypes must have shape [C,D].")
        normalized = F.normalize(prototypes, dim=-1)
        q = self.query(normalized)
        k = self.key(normalized)
        v = self.value(normalized)
        attention = torch.softmax(
            q @ k.t() / math.sqrt(normalized.shape[-1]), dim=-1
        )
        return F.normalize(
            self.norm(normalized + self.output(attention @ v)), dim=-1
        )

    def refine_queries(self, queries, base_memory, exclude_indices=None):
        if queries.ndim != 2 or base_memory.ndim != 2:
            raise ValueError("Global queries and memory must be rank-2 tensors.")
        if queries.shape[-1] != base_memory.shape[-1]:
            raise ValueError("Global query and memory dimensions must match.")
        if base_memory.shape[0] == 0:
            return F.normalize(queries, dim=-1)

        batch_size = queries.shape[0]
        context = base_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if exclude_indices is None:
            context = torch.cat([context, queries.unsqueeze(1)], dim=1)
        else:
            exclude_indices = exclude_indices.long().to(queries.device)
            if exclude_indices.shape != (batch_size,):
                raise ValueError("exclude_indices must contain one index per query.")
            if torch.any(exclude_indices < 0) or torch.any(
                exclude_indices >= base_memory.shape[0]
            ):
                raise IndexError("A global-memory exclusion index is out of range.")
            context[
                torch.arange(batch_size, device=queries.device), exclude_indices
            ] = queries
        return self._refine(queries, context)


class PrototypeFeatureResynthesis(nn.Module):
    """PRSN PFRSM: retrieve base semantic parts and reconstruct with query parts."""

    def __init__(self, embed_dim=768, top_k=5):
        super().__init__()
        if int(top_k) < 1:
            raise ValueError("prsn_top_k must be at least one.")
        self.embed_dim = int(embed_dim)
        self.top_k = int(top_k)

        self.base_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.base_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.base_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.base_output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.base_norm = nn.LayerNorm(embed_dim)

        self.reconstruct_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reconstruct_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reconstruct_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.prototype_value = nn.Linear(embed_dim, embed_dim, bias=False)

    def calibrate(
        self,
        prototype_global,
        prototype_parts,
        prototype_valid,
        base_global,
        base_parts,
        base_valid,
        exclude_indices=None,
    ):
        if prototype_global.ndim != 2 or prototype_parts.ndim != 3:
            raise ValueError("PFRSM prototypes must be [B,D] and [B,T,D].")
        if base_global.ndim != 2 or base_parts.ndim != 3:
            raise ValueError("PFRSM base memory must be [C,D] and [C,T,D].")
        batch_size, num_parts, feature_dim = prototype_parts.shape
        if prototype_global.shape != (batch_size, feature_dim):
            raise ValueError("Global and local prototype dimensions do not match.")
        if base_parts.shape[1:] != (num_parts, feature_dim):
            raise ValueError("Base part memory dimensions do not match novel parts.")
        if base_global.shape != (base_parts.shape[0], feature_dim):
            raise ValueError("Base global and local memory sizes do not match.")
        if prototype_valid.shape != (batch_size, num_parts):
            raise ValueError("prototype_valid must have shape [B,T].")
        if base_valid.shape != base_parts.shape[:2]:
            raise ValueError("base_valid must have shape [C,T].")

        if base_global.shape[0] == 0:
            return (
                F.normalize(prototype_parts, dim=-1) * prototype_valid.unsqueeze(-1),
                prototype_valid.bool(),
                torch.empty(batch_size, 0, dtype=torch.long, device=prototype_parts.device),
                prototype_parts.new_zeros(batch_size, 0, num_parts),
            )

        similarity = F.normalize(prototype_global, dim=-1) @ F.normalize(
            base_global, dim=-1
        ).t()
        if exclude_indices is not None:
            exclude_indices = exclude_indices.long().to(similarity.device)
            if exclude_indices.shape != (batch_size,):
                raise ValueError("exclude_indices must contain one index per prototype.")
            if torch.any(exclude_indices < 0) or torch.any(
                exclude_indices >= base_global.shape[0]
            ):
                raise IndexError("A PFRSM memory exclusion index is out of range.")
            similarity[
                torch.arange(batch_size, device=similarity.device), exclude_indices
            ] = float("-inf")
            available = base_global.shape[0] - 1
        else:
            available = base_global.shape[0]

        selected_count = min(self.top_k, available)
        if selected_count <= 0:
            return (
                F.normalize(prototype_parts, dim=-1) * prototype_valid.unsqueeze(-1),
                prototype_valid.bool(),
                torch.empty(batch_size, 0, dtype=torch.long, device=prototype_parts.device),
                prototype_parts.new_zeros(batch_size, 0, num_parts),
            )

        selected_indices = similarity.topk(selected_count, dim=-1).indices
        selected_parts = base_parts[selected_indices]
        selected_valid = base_valid[selected_indices].bool()

        query = self.base_query(prototype_parts).unsqueeze(1)
        keys = self.base_key(selected_parts)
        values = self.base_value(selected_parts)
        scores = (query * keys).sum(dim=-1) / math.sqrt(feature_dim)
        scores = scores.masked_fill(~selected_valid, -1e4)
        weights = torch.softmax(scores, dim=1)

        has_memory = selected_valid.any(dim=1)
        transferred = (weights.unsqueeze(-1) * values).sum(dim=1)
        transferred = self.base_output(transferred)
        transferred = transferred * has_memory.unsqueeze(-1)
        calibrated = self.base_norm(prototype_parts + transferred)
        calibrated_valid = prototype_valid.bool() | has_memory
        calibrated = F.normalize(calibrated, dim=-1) * calibrated_valid.unsqueeze(-1)
        return calibrated, calibrated_valid, selected_indices, weights

    def local_logits(
        self,
        query_parts,
        query_valid,
        prototype_parts,
        prototype_valid,
        logit_scale=16.0,
    ):
        """Query-conditioned local prototype reconstruction for all classes."""
        if query_parts.ndim != 3 or prototype_parts.ndim != 3:
            raise ValueError("Expected query [B,T,D] and prototype [C,T,D] parts.")
        batch_size, num_parts, feature_dim = query_parts.shape
        if prototype_parts.shape[1:] != (num_parts, feature_dim):
            raise ValueError("Query and prototype part dimensions must match.")
        if query_valid.shape != (batch_size, num_parts):
            raise ValueError("query_valid must have shape [B,T].")
        if prototype_valid.shape != prototype_parts.shape[:2]:
            raise ValueError("prototype_valid must have shape [C,T].")

        safe_query_valid = query_valid.bool().clone()
        empty_queries = ~safe_query_valid.any(dim=-1)
        if empty_queries.any():
            safe_query_valid[empty_queries, 0] = True

        proto_query = self.reconstruct_query(prototype_parts)
        query_key = self.reconstruct_key(query_parts)
        query_value = self.reconstruct_value(query_parts)
        scores = torch.einsum(
            "ctd,bud->bctu", proto_query, query_key
        ) / math.sqrt(feature_dim)
        scores = scores.masked_fill(
            ~safe_query_valid[:, None, None, :], -1e4
        )
        attention = torch.softmax(scores, dim=-1)
        reconstructed = torch.einsum(
            "bctu,bud->bctd", attention, query_value
        ) + proto_query.unsqueeze(0)
        prototype_value = self.prototype_value(prototype_parts).unsqueeze(0)
        similarity = F.cosine_similarity(
            prototype_value, reconstructed, dim=-1
        )
        valid = prototype_valid.bool().unsqueeze(0)
        logits = (similarity * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        return logits * float(logit_scale)

    def pairwise_local_logits(
        self,
        query_parts,
        query_valid,
        prototype_parts,
        prototype_valid,
        logit_scale=16.0,
    ):
        """Local scores for aligned query/prototype pairs, used by leave-one-out."""
        if query_parts.shape != prototype_parts.shape:
            raise ValueError("Pairwise query and prototype parts must have equal shapes.")
        batch_size, num_parts, feature_dim = query_parts.shape
        safe_query_valid = query_valid.bool().clone()
        empty_queries = ~safe_query_valid.any(dim=-1)
        if empty_queries.any():
            safe_query_valid[empty_queries, 0] = True

        proto_query = self.reconstruct_query(prototype_parts)
        query_key = self.reconstruct_key(query_parts)
        query_value = self.reconstruct_value(query_parts)
        scores = torch.einsum(
            "btd,bud->btu", proto_query, query_key
        ) / math.sqrt(feature_dim)
        scores = scores.masked_fill(~safe_query_valid[:, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        reconstructed = torch.einsum(
            "btu,bud->btd", attention, query_value
        ) + proto_query
        prototype_value = self.prototype_value(prototype_parts)
        similarity = F.cosine_similarity(
            prototype_value, reconstructed, dim=-1
        )
        valid = prototype_valid.bool()
        logits = (similarity * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        return logits * float(logit_scale)


class PRSN(nn.Module):
    """Global prototype refinement and local PFRSM reconstruction."""

    def __init__(self, embed_dim=768, top_k=5):
        super().__init__()
        self.global_refiner = GlobalPrototypeRefiner(embed_dim)
        self.pfrsm = PrototypeFeatureResynthesis(embed_dim, top_k=top_k)


class APLPRSNViT(nn.Module):
    """Single frozen ViT stream with shared APL part prompts and PRSN."""

    def __init__(self, args, encoder=None):
        super().__init__()
        self.args = args
        if bool(getattr(args, "scratch", False)):
            raise ValueError(
                "The APL-PRSN method requires the existing pretrained ViT-B/16; "
                "random/scratch backbone initialization is not supported."
            )
        self.backbone = FrozenViTBackbone(pretrained=True, encoder=encoder)
        embed_dim = self.backbone.embed_dim
        attention_heads = int(
            getattr(
                args,
                "apl_num_heads",
                getattr(self.backbone.encoder.blocks[-1].attn, "num_heads", 12),
            )
        )
        self.apl = APLPartPrompt(
            embed_dim=embed_dim,
            num_parts=int(getattr(args, "apl_num_parts", 6)),
            num_heads=attention_heads,
            prior_ratio=float(getattr(args, "apl_prior_ratio", 0.20)),
            temperature=float(getattr(args, "apl_temperature", 0.20)),
        )
        self.prsn = PRSN(
            embed_dim=embed_dim,
            top_k=int(getattr(args, "prsn_top_k", 5)),
        )
        self._incremental_frozen = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self._incremental_frozen:
            self.apl.eval()
            self.prsn.eval()
        return self

    def forward(self, images):
        global_features, patches, attention_priors = self.backbone(images)
        parts, part_valid, assignments, image_queries = self.apl(
            patches, attention_priors
        )
        return {
            "global_features": global_features,
            "parts": parts,
            "part_valid": part_valid,
            "assignments": assignments,
            "image_queries": image_queries,
        }

    def freeze_for_incremental(self):
        """Freeze all learned components after base training."""
        self._incremental_frozen = True
        self.apl.requires_grad_(False)
        self.prsn.requires_grad_(False)
        self.eval()
