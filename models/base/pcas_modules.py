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
    task-wise prompt allocation, PCAS uses a fixed four-component bank and
    a softmax over all components.
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
        dynamic_routing: bool = True,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("PCAS requires at least one spatial component.")
        if bottleneck_dim <= 0 or embed_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive.")
        if not 0.0 < residual_max <= 1.0:
            raise ValueError("residual_max must be in (0, 1].")
        if not 0.0 < residual_init < residual_max:
            raise ValueError(
                "residual_init must be strictly between 0 and residual_max."
            )

        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_components = num_components
        self.route_temperature = float(route_temperature)
        self.residual_max = float(residual_max)
        # A one-component bank has no meaningful routing decision. Likewise,
        # the routing ablation uses a fixed uniform mixture and freezes its
        # otherwise-unused query/key parameters.
        self.dynamic_routing = bool(dynamic_routing) and num_components > 1

        self.down = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1, bias=False)
        # The mathematical definition is q = LN(W_q h), so normalization is
        # applied after projection, in the bottleneck/query space.
        self.query = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.query_norm = nn.LayerNorm(bottleneck_dim, eps=1e-6)

        # CODA-style decomposed attention parameters:
        # q_m = normalize(q * A_m), score_m = q_m dot normalize(K_m).
        self.A = nn.Parameter(torch.empty(num_components, bottleneck_dim))
        self.K = nn.Parameter(torch.empty(num_components, bottleneck_dim))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.K)
        if not self.dynamic_routing:
            self.query.requires_grad_(False)
            self.query_norm.requires_grad_(False)
            self.A.requires_grad_(False)
            self.K.requires_grad_(False)

        self.components = nn.ModuleList(
            [
                SpatialPromptComponent(bottleneck_dim, embed_dim, dropout)
                for _ in range(num_components)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Parameterize gamma in (0, residual_max], while preserving the
        # requested initial residual magnitude exactly.
        residual_ratio = residual_init / residual_max
        residual_logit = math.log(residual_ratio / (1.0 - residual_ratio))
        self.gamma_logit = nn.Parameter(torch.tensor(residual_logit, dtype=torch.float32))

        nn.init.kaiming_normal_(self.down.weight, mode="fan_out", nonlinearity="linear")
        nn.init.xavier_uniform_(self.query.weight)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_max * torch.sigmoid(self.gamma_logit)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_feature: torch.Tensor,
        grid_size,
    ):
        if patch_tokens.ndim != 3 or cls_feature.ndim != 2:
            raise ValueError("Expected patch tokens [B,N,C] and CLS features [B,C].")

        batch_size, num_patches, channels = patch_tokens.shape
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

        query = self.query_norm(self.query(cls_feature))
        query = F.normalize(query, p=2, dim=-1)
        if self.dynamic_routing:
            # PCAS definition: q_m = q * sigmoid(A_m), followed by cosine
            # matching against the component-specific key K_m.
            modulated_query = F.normalize(
                query.unsqueeze(1) * torch.sigmoid(self.A).unsqueeze(0),
                p=2,
                dim=-1,
            )
            normalized_keys = F.normalize(self.K, p=2, dim=-1)
            scores = torch.einsum("bmd,md->bm", modulated_query, normalized_keys)
            weights = F.softmax(scores / self.route_temperature, dim=-1)
        else:
            weights = patch_tokens.new_full(
                (batch_size, self.num_components),
                1.0 / self.num_components,
            )

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
            dynamic_routing=not bool(
                getattr(args, "disable_dynamic_routing", False)
            ),
        )
        self.fc = nn.Linear(self.embed_dim, int(args.base_class), bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained transformer is always frozen and always in eval mode.
        self.encoder.eval()
        if self._incremental_frozen:
            self.pcas.eval()
            self.fc.eval()
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

    def classify_base(self, features: torch.Tensor) -> torch.Tensor:
        normalized_weight = F.normalize(self.fc.weight, p=2, dim=-1)
        return self.temperature * F.linear(features, normalized_weight)

    def forward(self, images: torch.Tensor):
        features, auxiliary = self.encode_student(images, return_aux=True)
        return {
            "features": features,
            "logit": self.classify_base(features),
            "routing": auxiliary["routing_weights"],
            "residual_scale": auxiliary["residual_scale"],
        }

    def freeze_for_incremental(self):
        """Freeze PCAS and the base classifier before real FSCIL sessions."""
        self._incremental_frozen = True
        self.pcas.requires_grad_(False)
        self.fc.requires_grad_(False)
        self.eval()


# Backwards-compatible name used by the original LGSP repository imports.
ViT_MYNET = PCASViT
