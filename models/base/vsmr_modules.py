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
            if self.embed_dim != 768 or len(self.encoder.blocks) != 12:
                raise RuntimeError(
                    "VSMR requires the repository's pretrained ViT-B/16 "
                    "(768 dimensions and 12 Transformer blocks)."
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
    """Shared soft prompts that pool Patch features into unordered parts."""

    def __init__(self, embed_dim=768, num_parts=6):
        super().__init__()
        if int(embed_dim) <= 0 or int(num_parts) < 2:
            raise ValueError("embed_dim must be positive and num_parts >= 2.")
        self.embed_dim = int(embed_dim)
        self.num_parts = int(num_parts)

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

    def forward(self, cls_features, patch_features):
        if cls_features.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("Expected CLS [B,D] and Patch [B,N,D] features.")
        batch_size, num_patches, feature_dim = patch_features.shape
        if feature_dim != self.embed_dim:
            raise ValueError("Patch feature dimension does not match VSMR.")
        if cls_features.shape != (batch_size, self.embed_dim):
            raise ValueError("CLS and Patch feature dimensions do not match.")

        queries = self.query_projection(self.shared_prompts)
        keys = self.key_projection(patch_features)
        values = self.value_projection(patch_features)

        attention_logits = torch.einsum(
            "kd,bnd->bkn", queries, keys
        ) / math.sqrt(self.embed_dim)
        attention = torch.softmax(attention_logits, dim=-1)

        local_parts = torch.bmm(attention, values)
        local_parts = F.normalize(local_parts, p=2, dim=-1)
        global_features = F.normalize(
            self.global_projection(cls_features), p=2, dim=-1
        )
        return global_features, local_parts, attention


class VSMRViT(nn.Module):
    """Frozen ViT plus shared visual-semantic decomposition."""

    def __init__(self, args, encoder=None):
        super().__init__()
        if bool(getattr(args, "scratch", False)):
            raise ValueError(
                "VSMR requires the pretrained ImageNet-21K ViT-B/16; "
                "scratch backbone initialization is not supported."
            )
        self.backbone = FrozenViTBackbone(pretrained=True, encoder=encoder)
        self.decomposer = VisualSemanticDecomposer(
            embed_dim=self.backbone.embed_dim,
            num_parts=int(getattr(args, "vsmr_num_parts", 6)),
        )
        self._incremental_frozen = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self._incremental_frozen:
            self.decomposer.eval()
        return self

    def forward(self, images):
        cls_features, patch_features = self.backbone(images)
        global_features, parts, attention = self.decomposer(
            cls_features, patch_features
        )
        return {
            "global_features": global_features,
            "parts": parts,
            "attention": attention,
        }

    def freeze_for_incremental(self):
        """Freeze every learned component before the first incremental session."""
        self._incremental_frozen = True
        self.decomposer.requires_grad_(False)
        self.eval()
