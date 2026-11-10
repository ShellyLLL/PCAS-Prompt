"""LGSP Local Spatial Prompting with a frozen ViT-B/16 backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_My


class LocalPromptGenerator(nn.Module):
    """One image-conditioned local spatial prompt generator.

    This follows the released LGSP construction: two convolutions map an
    RGB image to an RGB residual prompt while preserving spatial resolution.
    """

    def __init__(
        self,
        hidden_dim: int = 3,
        kernel_size1: int = 3,
        kernel_size2: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        for name, kernel_size in (
            ("kernel_size1", kernel_size1),
            ("kernel_size2", kernel_size2),
        ):
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("{} must be a positive odd integer.".format(name))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.network = nn.Sequential(
            nn.Conv2d(
                3, hidden_dim, kernel_size1,
                stride=1, padding=kernel_size1 // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim, 3, kernel_size2,
                stride=1, padding=kernel_size2 // 2,
            ),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.network(images))


class LocalSpatialPromptPool(nn.Module):
    """Query-selected pool of independent local spatial prompt generators.

    The CLS token of the original image is projected to a query. Cosine
    similarity selects top-k learned prompt keys. A softmax over only those
    selected scores weights their image-conditioned spatial prompt maps.
    Unselected generators are not evaluated, which is equivalent to
    selecting first and avoids materializing all M full-resolution maps.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        pool_size: int = 30,
        top_k: int = 10,
        hidden_dim: int = 3,
        kernel_size1: int = 3,
        kernel_size2: int = 5,
        dropout: float = 0.1,
        temperature: float = 0.2,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if top_k <= 0 or top_k > pool_size:
            raise ValueError("top_k must be in [1, pool_size].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.embed_dim = int(embed_dim)
        self.pool_size = int(pool_size)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

        self.generators = nn.ModuleList(
            [
                LocalPromptGenerator(
                    hidden_dim=hidden_dim,
                    kernel_size1=kernel_size1,
                    kernel_size2=kernel_size2,
                    dropout=dropout,
                )
                for _ in range(self.pool_size)
            ]
        )

        self.query_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.key_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.LayerNorm(self.embed_dim),
        )
        self.prompt_keys = nn.Parameter(torch.empty(self.pool_size, self.embed_dim))
        nn.init.uniform_(self.prompt_keys, -1.0, 1.0)

    def select_prompts(self, cls_features: torch.Tensor):
        if cls_features.ndim != 2 or cls_features.shape[-1] != self.embed_dim:
            raise ValueError(
                "CLS features must have shape [B, {}].".format(self.embed_dim)
            )

        query = F.normalize(self.query_projection(cls_features), p=2, dim=-1)
        keys = F.normalize(self.key_projection(self.prompt_keys), p=2, dim=-1)
        similarities = torch.matmul(query, keys.transpose(0, 1))
        selected_scores, selected_indices = torch.topk(
            similarities, k=self.top_k, dim=1, largest=True, sorted=True
        )
        weights = F.softmax(selected_scores / self.temperature, dim=1)
        return weights, selected_indices, similarities

    def forward(self, images: torch.Tensor, cls_features: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected input images with shape [B, 3, H, W].")
        if images.shape[0] != cls_features.shape[0]:
            raise ValueError("Image and CLS feature batch sizes must match.")

        weights, selected_indices, similarities = self.select_prompts(cls_features)
        local_prompt = torch.zeros_like(images)

        # Generate only the prompts selected for at least one image in this
        # batch. Each image still receives its own query-conditioned weights.
        for generator_index, generator in enumerate(self.generators):
            sample_indices, prompt_slots = torch.where(
                selected_indices == generator_index
            )
            if sample_indices.numel() == 0:
                continue

            selected_images = images.index_select(0, sample_indices)
            generated_prompt = generator(selected_images)
            selected_weights = weights[sample_indices, prompt_slots].view(
                -1, 1, 1, 1
            )
            contribution = generated_prompt * selected_weights
            local_prompt = local_prompt.index_add(
                0, sample_indices, contribution
            )

        return local_prompt, {
            "weights": weights,
            "selected_indices": selected_indices,
            "similarities": similarities,
        }


class LSPViT(nn.Module):
    """Frozen ImageNet-21K ViT-B/16 enhanced by LGSP local spatial prompts."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = 768
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
                "LSP expects a 12-block ViT-B/16, got {} blocks.".format(
                    len(self.encoder.blocks)
                )
            )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

        self.lsp = LocalSpatialPromptPool(
            embed_dim=self.embed_dim,
            pool_size=int(getattr(args, "lsp_pool_size", 30)),
            top_k=int(getattr(args, "lsp_top_k", 10)),
            hidden_dim=int(getattr(args, "lsp_hidden_dim", 3)),
            kernel_size1=int(getattr(args, "lsp_kernel_size1", 3)),
            kernel_size2=int(getattr(args, "lsp_kernel_size2", 5)),
            dropout=float(getattr(args, "lsp_dropout", 0.1)),
            temperature=float(getattr(args, "lsp_temperature", 0.2)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The visual backbone is frozen in both base and incremental phases.
        self.encoder.eval()
        if self._incremental_frozen:
            self.lsp.eval()
        return self

    def _validate_images(self, images: torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W].")
        if images.shape[-2:] != (224, 224):
            raise ValueError("LGSP ViT-B/16 expects 224x224 input images.")

    @torch.no_grad()
    def encode_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Return the original frozen ViT representation without LSP."""
        self._validate_images(images)
        tokens = self.encoder.forward_features(images)
        features = self.encoder.forward_head(tokens)
        return F.normalize(features, p=2, dim=-1)

    def encode_student(self, images: torch.Tensor, return_aux: bool = False):
        """Enhance each image with its dynamically selected local prompts."""
        self._validate_images(images)

        # Query extraction follows LGSP: use the original image's frozen ViT
        # CLS feature, then project it in the trainable prompt-pool module.
        with torch.no_grad():
            original_tokens = self.encoder.forward_features(images)
            cls_features = original_tokens[:, 0, :]

        local_prompt, auxiliary = self.lsp(images, cls_features)
        enhanced_images = images + local_prompt

        # The frozen encoder remains differentiable with respect to its input,
        # allowing gradients to train the selected prompt generators.
        enhanced_tokens = self.encoder.forward_features(enhanced_images)
        features = self.encoder.forward_head(enhanced_tokens)
        features = F.normalize(features, p=2, dim=-1)

        if not return_aux:
            return features
        return features, auxiliary

    def forward(self, images: torch.Tensor):
        features, auxiliary = self.encode_student(images, return_aux=True)
        return {"features": features, **auxiliary}

    def freeze_for_incremental(self):
        """Freeze the complete LSP pool after base training."""
        self._incremental_frozen = True
        self.lsp.requires_grad_(False)
        self.eval()
