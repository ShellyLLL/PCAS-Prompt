"""Single-stream PCAS spatial prompting and unified visual feature extraction."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My


class SpatialPromptComponent(nn.Module):
    """One local spatial transformation over the shared bottleneck map."""

    def __init__(self, bottleneck_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=bottleneck_dim,
            bias=False,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.up = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.depthwise(x))))


class PCASPrompt(nn.Module):
    """Fixed-capacity spatial prompt components composed by CLS attention.

    Components operate on one shared Patch map. CLS Query-Key attention
    continuously weights their spatial prompt maps; the weighted sum is
    injected once into the Patch stream. This module never creates component
    feature streams.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        bottleneck_dim: int = 96,
        num_components: int = 4,
        dropout: float = 0.1,
        attention_temperature: float = 0.2,
        residual_init: float = 0.1,
        residual_max: float = 0.5,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be at least one.")
        if embed_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        if not 0.0 < residual_init < residual_max:
            raise ValueError(
                "residual_init must be positive and strictly below residual_max."
            )

        self.embed_dim = int(embed_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_components = int(num_components)
        self.attention_temperature = float(attention_temperature)
        self.residual_max = float(residual_max)

        self.down = nn.Conv2d(
            self.embed_dim, self.bottleneck_dim, kernel_size=1, bias=False
        )
        self.query = nn.Linear(self.embed_dim, self.bottleneck_dim, bias=False)
        self.query_norm = nn.LayerNorm(self.bottleneck_dim, eps=1e-6)

        self.A = nn.Parameter(torch.empty(self.num_components, self.bottleneck_dim))
        self.K = nn.Parameter(torch.empty(self.num_components, self.bottleneck_dim))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.K)

        self.components = nn.ModuleList(
            [
                SpatialPromptComponent(
                    self.bottleneck_dim, self.embed_dim, dropout
                )
                for _ in range(self.num_components)
            ]
        )
        self.output_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        initial_ratio = float(residual_init) / self.residual_max
        residual_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.residual_logit = nn.Parameter(
            torch.tensor(residual_logit, dtype=torch.float32)
        )

        nn.init.kaiming_normal_(
            self.down.weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.xavier_uniform_(self.query.weight)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_max * torch.sigmoid(self.residual_logit)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_feature: torch.Tensor,
        grid_size,
    ):
        if patch_tokens.ndim != 3 or cls_feature.ndim != 2:
            raise ValueError(
                "Expected patch tokens [B,N,C] and CLS features [B,C]."
            )

        batch_size, num_patches, channels = patch_tokens.shape
        height, width = int(grid_size[0]), int(grid_size[1])
        if channels != self.embed_dim or num_patches != height * width:
            raise ValueError(
                "Patch-token shape does not match the configured ViT grid: "
                f"got {tuple(patch_tokens.shape)}, grid={grid_size}."
            )
        if cls_feature.shape != (batch_size, self.embed_dim):
            raise ValueError("CLS feature shape does not match Patch features.")

        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        shared_map = self.down(patch_map)

        component_maps = []
        for component in self.components:
            component_map = component(shared_map)
            component_map = self.output_norm(
                component_map.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            component_maps.append(component_map)
        component_maps = torch.stack(component_maps, dim=1)
        # [B, M, C, H, W]

        query = self.query_norm(self.query(cls_feature))
        query = F.normalize(query, p=2, dim=-1)
        gated_queries = F.normalize(
            query.unsqueeze(1) * torch.sigmoid(self.A).unsqueeze(0),
            p=2,
            dim=-1,
        )
        normalized_keys = F.normalize(self.K, p=2, dim=-1)
        attention_scores = torch.einsum(
            "bmd,md->bm", gated_queries, normalized_keys
        )
        component_weights = F.softmax(
            attention_scores / self.attention_temperature, dim=-1
        )

        mixed_prompt = (
            component_maps
            * component_weights[:, :, None, None, None]
        ).sum(dim=1)
        gamma = self.residual_scale
        enhanced_map = patch_map + gamma * mixed_prompt
        enhanced_patch_tokens = enhanced_map.flatten(2).transpose(1, 2)

        return enhanced_patch_tokens, component_weights


class PCASViT(nn.Module):
    """Frozen ViT-B/16 with one PCAS-enhanced visual feature stream."""

    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.embed_dim = 768
        self.insert_after = int(getattr(args, "pcas_layer", 6))
        self._incremental_frozen = False

        self.encoder = model_My.vit_base_patch16_224_in21k(
            pretrained=not bool(getattr(args, "scratch", False)),
            num_classes=0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        if len(self.encoder.blocks) != 12:
            raise RuntimeError(
                f"PCAS expects a 12-block ViT-B/16, got {len(self.encoder.blocks)} blocks."
            )
        if not 1 <= self.insert_after < len(self.encoder.blocks):
            raise ValueError("pcas_layer must identify a block between 1 and 11.")

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        self.grid_size = tuple(self.encoder.patch_embed.grid_size)
        if self.grid_size != (14, 14):
            raise RuntimeError(
                f"PCAS expects a 14x14 ViT-B/16 patch grid, got {self.grid_size}."
            )

        self.pcas = PCASPrompt(
            embed_dim=self.embed_dim,
            bottleneck_dim=int(getattr(args, "bottleneck_dim", 96)),
            num_components=int(getattr(args, "num_components", 4)),
            dropout=float(getattr(args, "prompt_dropout", 0.1)),
            attention_temperature=float(
                getattr(args, "attention_temperature", 0.2)
            ),
            residual_init=float(getattr(args, "residual_init", 0.1)),
            residual_max=float(getattr(args, "residual_max", 0.5)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self._incremental_frozen:
            self.pcas.eval()
        return self

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Original frozen ViT feature without spatial prompt enhancement."""
        if images.shape[-2:] != (224, 224):
            raise ValueError("PCAS ViT-B/16 expects 224x224 input images.")
        tokens = self.encoder.forward_features(images)
        features = self.encoder.forward_head(tokens)
        return F.normalize(features, p=2, dim=-1)

    def _run_frozen_tail(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.encoder.blocks[self.insert_after:]:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        features = self.encoder.forward_head(tokens)
        return F.normalize(features, p=2, dim=-1)

    def encode_student(self, images: torch.Tensor, return_aux: bool = False):
        """Encode one image batch once and optionally expose its component weights."""
        if images.shape[-2:] != (224, 224):
            raise ValueError("PCAS ViT-B/16 expects 224x224 input images.")
        tokens = self.encoder.patch_embed(images)
        tokens = self.encoder._pos_embed(tokens)
        for block in self.encoder.blocks[:self.insert_after]:
            tokens = block(tokens)
        enhanced_patches, component_weights = self.pcas(
            tokens[:, 1:, :], tokens[:, 0, :], self.grid_size
        )
        enhanced_tokens = torch.cat([tokens[:, :1, :], enhanced_patches], dim=1)
        # Exactly one frozen-tail pass and one final student feature per image.
        features = self._run_frozen_tail(enhanced_tokens)
        if not return_aux:
            return features
        return features, {
            "component_weights": component_weights,
            "residual_scale": self.pcas.residual_scale,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_student(images, return_aux=False)

    def forward(self, images: torch.Tensor):
        features, auxiliary = self.encode_student(images, return_aux=True)
        return {
            "features": features,
            "component_weights": auxiliary["component_weights"],
            "residual_scale": auxiliary["residual_scale"],
        }

    def freeze_for_incremental(self):
        """Freeze all learned prompt and classifier parameters for real sessions."""
        self._incremental_frozen = True
        self.pcas.requires_grad_(False)
        self.eval()


# Backwards-compatible name used by the original LGSP repository imports.
ViT_MYNET = PCASViT
