"""PCAS-Prompt modules for frozen-ViT FSCIL.

The backbone is a pretrained ViT-B/16. A decomposed spatial prompt bank is
inserted after the sixth transformer block; no extra tokens are introduced.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My


class SpatialPromptComponent(nn.Module):
    """One local spatial component operating on a shared bottleneck map."""

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
    """Prototype-calibrated affinity-consistent spatial prompt bank.

    Query-Key attention follows the decomposed attention formulation in
    CODA-Prompt: query is modulated by a component-specific A vector and
    matched against a component-specific normalized key. Unlike CODA's
    task-wise prompt allocation, PCAS uses a fixed-capacity component bank
    and a softmax over all components. The M=1 control removes the router.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        bottleneck_dim: int = 96,
        num_components: int = 4,
        dropout: float = 0.1,
        route_temperature: float = 0.2,
        residual_init: float = 0.1,
        residual_max: float = 0.5,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be at least one.")
        if bottleneck_dim <= 0 or embed_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive.")
        residual_max = float(residual_max)
        residual_init = float(residual_init)
        if not 0.0 < residual_max <= 1.0:
            raise ValueError("residual_max must be in (0, 1].")
        if not 0.0 < residual_init < residual_max:
            raise ValueError("residual_init must be in (0, residual_max).")

        self.embed_dim = int(embed_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_components = int(num_components)
        self.route_temperature = float(route_temperature)
        self.residual_max = residual_max

        self.down = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1, bias=False)
        if self.num_components > 1:
            self.query = nn.Linear(embed_dim, bottleneck_dim, bias=False)
            self.query_norm = nn.LayerNorm(bottleneck_dim, eps=1e-6)
            self.A = nn.Parameter(torch.empty(num_components, bottleneck_dim))
            self.K = nn.Parameter(torch.empty(num_components, bottleneck_dim))
            nn.init.orthogonal_(self.A)
            nn.init.orthogonal_(self.K)
        else:
            # M=1 is a true single-generator control with no inert router.
            self.query = None
            self.query_norm = None
            self.register_parameter("A", None)
            self.register_parameter("K", None)

        self.components = nn.ModuleList(
            [
                SpatialPromptComponent(bottleneck_dim, embed_dim, dropout)
                for _ in range(num_components)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        initial_fraction = float(residual_init) / self.residual_max
        residual_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.gamma_logit = nn.Parameter(torch.tensor(residual_logit, dtype=torch.float32))

        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="linear")
        if self.query is not None:
            nn.init.xavier_uniform_(self.query.weight)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_max * torch.sigmoid(self.gamma_logit)

    def route(self, cls_feature: torch.Tensor) -> torch.Tensor:
        """Return per-example soft assignments over the fixed component bank."""
        if cls_feature.ndim != 2 or cls_feature.shape[-1] != self.embed_dim:
            raise ValueError(
                "Expected CLS features [B, embed_dim], got {}.".format(
                    tuple(cls_feature.shape)
                )
            )
        if self.num_components == 1:
            return cls_feature.new_ones((cls_feature.shape[0], 1))

        query = self.query_norm(self.query(cls_feature))
        gates = torch.sigmoid(self.A)
        modulated_query = F.normalize(
            query.unsqueeze(1) * gates.unsqueeze(0), p=2, dim=-1
        )
        normalized_keys = F.normalize(self.K, p=2, dim=-1)
        scores = torch.einsum("bmd,md->bm", modulated_query, normalized_keys)
        return F.softmax(scores / self.route_temperature, dim=-1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_feature: torch.Tensor,
        grid_size,
    ):
        if patch_tokens.ndim != 3 or cls_feature.ndim != 2:
            raise ValueError("Expected patch tokens [B,N,C] and CLS features [B,C].")

        batch_size, num_patches, channels = patch_tokens.shape
        if cls_feature.shape[0] != batch_size:
            raise ValueError("Patch-token and CLS batch sizes must match.")
        height, width = int(grid_size[0]), int(grid_size[1])
        if channels != self.embed_dim or num_patches != height * width:
            raise ValueError(
                "Patch-token shape does not match the configured ViT grid: "
                f"got {tuple(patch_tokens.shape)}, grid={grid_size}."
            )

        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        shared_map = self.down(patch_map)

        weights = self.route(cls_feature)

        mixed_prompt = torch.zeros_like(patch_map)
        for component_index, component in enumerate(self.components):
            component_map = component(shared_map)
            component_map = self.output_norm(
                component_map.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            component_weight = weights[:, component_index].view(
                batch_size, 1, 1, 1
            )
            mixed_prompt = mixed_prompt + component_weight * component_map

        enhanced_map = patch_map + self.residual_scale * mixed_prompt
        enhanced_tokens = enhanced_map.flatten(2).transpose(1, 2)
        return enhanced_tokens, weights


class PCASViT(nn.Module):
    """Frozen ViT-B/16 with a trainable PCAS spatial prompt and base head."""

    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.embed_dim = 768
        self.insert_after = int(getattr(args, "pcas_layer", 6))
        self.temperature = float(getattr(args, "temperature", 16.0))
        self._incremental_frozen = False

        # Use the repository's registered timm-compatible ViT-B/16 model and
        # its ImageNet-21K pretrained configuration.
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

        grid_size = self.encoder.patch_embed.grid_size
        if tuple(grid_size) != (14, 14):
            raise RuntimeError(
                f"PCAS expects a 14x14 ViT-B/16 patch grid, got {grid_size}."
            )
        self.grid_size = tuple(grid_size)

        self.pcas = PCASPrompt(
            embed_dim=self.embed_dim,
            bottleneck_dim=int(getattr(args, "bottleneck_dim", 96)),
            num_components=int(getattr(args, "num_components", 4)),
            dropout=float(getattr(args, "prompt_dropout", 0.1)),
            route_temperature=float(getattr(args, "route_temperature", 0.2)),
            residual_init=float(getattr(args, "residual_init", 0.1)),
            residual_max=float(getattr(args, "residual_max", 0.5)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained transformer is always frozen and always in eval mode.
        self.encoder.eval()
        if self._incremental_frozen:
            self.pcas.eval()
        return self

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Original frozen ViT feature, without any PCAS modification."""
        if images.shape[-2:] != (224, 224):
            raise ValueError("PCAS ViT-B/16 expects 224x224 input images.")
        tokens = self.encoder.forward_features(images)
        features = self.encoder.forward_head(tokens)
        return F.normalize(features, p=2, dim=-1)

    def encode_student(self, images: torch.Tensor, return_aux: bool = False):
        """PCAS-enhanced feature from the same frozen ViT feature space."""
        if images.shape[-2:] != (224, 224):
            raise ValueError("PCAS ViT-B/16 expects 224x224 input images.")

        tokens = self.encoder.patch_embed(images)
        tokens = self.encoder._pos_embed(tokens)
        route_weights = None

        for block_index, block in enumerate(self.encoder.blocks):
            tokens = block(tokens)
            if block_index + 1 == self.insert_after:
                enhanced_patches, route_weights = self.pcas(
                    tokens[:, 1:, :],
                    tokens[:, 0, :],
                    self.grid_size,
                )
                # CLS remains unchanged at the injection point; only patch
                # tokens receive the spatial residual. Sequence length is fixed.
                tokens = torch.cat(
                    [tokens[:, :1, :], enhanced_patches], dim=1
                )

        tokens = self.encoder.norm(tokens)
        features = self.encoder.forward_head(tokens)
        features = F.normalize(features, p=2, dim=-1)

        if return_aux:
            return features, {
                "routing_weights": route_weights,
                "residual_scale": self.pcas.residual_scale,
            }
        return features

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_student(images, return_aux=False)

    def classify_base(
        self, features: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Classify against the same feature-space prototypes used at test time."""
        if prototypes.ndim != 2 or prototypes.shape[-1] != features.shape[-1]:
            raise ValueError("Base prototypes must have shape [C, embed_dim].")
        normalized_prototypes = F.normalize(prototypes, p=2, dim=-1)
        return self.temperature * (features @ normalized_prototypes.t())

    def forward(self, images: torch.Tensor, prototypes: torch.Tensor):
        features, auxiliary = self.encode_student(images, return_aux=True)
        return {
            "features": features,
            "logit": self.classify_base(features, prototypes),
            "routing": auxiliary["routing_weights"],
            "residual_scale": auxiliary["residual_scale"],
        }

    def freeze_for_incremental(self):
        """Freeze PCAS and the base classifier before real FSCIL sessions."""
        self._incremental_frozen = True
        self.pcas.requires_grad_(False)
        self.eval()


# Backwards-compatible name used by the original LGSP repository imports.
ViT_MYNET = PCASViT
