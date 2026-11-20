import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import models.clip.clip as clip
from models.bimc import load_clip_to_cpu


class PrototypeTransfer(nn.Module):
    """Pairwise base-prototype retrieval followed by an adaptive transfer gate."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("SCORER_TAU must be positive.")
        self.temperature = float(temperature)
        self.scorer_weight = nn.Parameter(torch.full((4,), 0.25))
        self.gate_weight = nn.Parameter(torch.zeros(8))
        self.gate_bias = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(
        self,
        query_text: torch.Tensor,
        current_proto: torch.Tensor,
        support_features: torch.Tensor,
        base_text_embeddings: torch.Tensor,
        base_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if query_text.ndim != 1 or current_proto.ndim != 1:
            raise ValueError("query_text and current_proto must be rank-1 feature vectors.")
        if query_text.shape != current_proto.shape:
            raise ValueError("query_text and current_proto must have the same feature dimension.")
        if base_text_embeddings.ndim != 2 or base_prototypes.ndim != 2:
            raise ValueError("Base text embeddings and prototypes must be rank-2 tensors.")
        if base_text_embeddings.shape != base_prototypes.shape:
            raise ValueError("Base text embeddings and visual prototypes must have matching shapes.")
        if base_prototypes.shape[0] == 0:
            raise ValueError("CSPC requires at least one base class prototype.")
        if base_prototypes.shape[1] != query_text.shape[0]:
            raise ValueError("Base and current features must share the same embedding dimension.")
        if support_features.ndim != 2 or support_features.shape[0] == 0:
            raise ValueError("support_features must be a non-empty [K, D] tensor.")
        if support_features.shape[1] != query_text.shape[0]:
            raise ValueError("support_features has an incompatible embedding dimension.")

        tensors = (
            query_text, current_proto, support_features,
            base_text_embeddings, base_prototypes,
        )
        if any(tensor.device != query_text.device for tensor in tensors):
            raise ValueError("All prototype-transfer inputs must be on the same device.")
        if self.scorer_weight.device != query_text.device:
            raise RuntimeError(
                "PrototypeTransfer parameters and feature tensors are on different devices. "
                "Move the complete CSPC model to the selected device."
            )

        query_text = F.normalize(query_text.float(), dim=-1)
        current_proto = F.normalize(current_proto.float(), dim=-1)
        base_text_embeddings = F.normalize(base_text_embeddings.float(), dim=-1)
        base_prototypes = F.normalize(base_prototypes.float(), dim=-1)
        support_features = F.normalize(support_features.float(), dim=-1)

        s_tt = torch.matmul(base_text_embeddings, query_text)
        s_vv = torch.matmul(base_prototypes, current_proto)
        s_tv = torch.matmul(base_prototypes, query_text)
        pair_features = torch.stack((s_tt, s_vv, s_tv, s_tt * s_vv), dim=-1)
        scores = torch.matmul(pair_features, self.scorer_weight.float())
        transfer_weights = torch.softmax(scores / self.temperature, dim=0)

        weighted_base = torch.matmul(transfer_weights, base_prototypes)
        transfer_proto = F.normalize(weighted_base, dim=-1)

        intra_uncertainty = 1.0 - torch.matmul(support_features, current_proto).mean()

        num_base = int(transfer_weights.numel())
        entropy = -(transfer_weights * torch.log(transfer_weights.clamp_min(1e-8))).sum()
        entropy = entropy / math.log(max(num_base, 2))
        top_values = torch.topk(transfer_weights, k=min(2, num_base)).values
        max_weight = top_values[0]
        margin = top_values[0] - top_values[1] if num_base > 1 else top_values[0].new_zeros(())

        gate_features = torch.stack(
            (
                torch.dot(query_text, current_proto),
                torch.dot(query_text, transfer_proto),
                torch.dot(current_proto, transfer_proto),
                intra_uncertainty,
                entropy,
                max_weight,
                margin,
                torch.linalg.vector_norm(weighted_base),
            )
        )
        transfer_ratio = torch.sigmoid(
            torch.dot(self.gate_weight.float(), gate_features) + self.gate_bias.float()
        )
        calibrated_proto = F.normalize(
            (1.0 - transfer_ratio) * current_proto + transfer_ratio * transfer_proto,
            dim=-1,
        )

        diagnostics = {
            "weights": transfer_weights,
            "transfer_proto": transfer_proto,
            "gate_features": gate_features,
            "transfer_ratio": transfer_ratio,
            "intra_uncertainty": intra_uncertainty,
            "entropy": entropy,
            "max_weight": max_weight,
            "margin": margin,
            "weighted_base_norm": torch.linalg.vector_norm(weighted_base),
        }
        return calibrated_proto, diagnostics


class CSPCModel(nn.Module):
    """Frozen CLIP encoders with one shared, continually updated text prompt."""

    def __init__(self, cfg, device: torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.n_ctx = int(cfg.TRAINER.CSPC.N_CTX)
        self.text_chunk_size = int(cfg.TRAINER.CSPC.TEXT_CHUNK_SIZE)
        if self.n_ctx <= 0:
            raise ValueError("N_CTX must be positive.")
        if self.text_chunk_size <= 0:
            raise ValueError("TEXT_CHUNK_SIZE must be positive.")
        if cfg.TRAINER.CSPC.PREC not in {"fp16", "fp32"}:
            raise ValueError("CSPC.PREC must be either 'fp16' or 'fp32'.")

        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.CSPC.PREC == "fp32":
            clip_model.float()
        clip_model.eval()
        clip_model.requires_grad_(False)
        self.clip_model = clip_model.to(self.device)
        self.context_length = int(self.clip_model.positional_embedding.shape[0])
        self.text_width = int(self.clip_model.ln_final.weight.shape[0])
        if self.n_ctx >= self.context_length - 1:
            raise ValueError(
                f"N_CTX={self.n_ctx} leaves fewer than two CLIP text tokens "
                "for SOS and EOT."
            )

        init_tokens = clip.tokenize(cfg.TRAINER.CSPC.PROMPT_INIT_TEXT)
        init_eot = int(init_tokens[0].argmax().item())
        init_ids = init_tokens[0, 1:init_eot][: self.n_ctx].to(self.device)
        with torch.no_grad():
            init_context = self.clip_model.token_embedding(init_ids).float()
        if init_context.shape[0] < self.n_ctx:
            missing = self.n_ctx - init_context.shape[0]
            padding = torch.empty(missing, self.text_width, device=self.device)
            nn.init.normal_(padding, std=0.02)
            init_context = torch.cat((init_context, padding), dim=0)
        self.prompt = nn.Parameter(init_context.detach().clone())

        # The prompt and frozen CLIP are already on the requested device. The
        # transfer module must be moved explicitly because CSPCModel itself is
        # not subsequently wrapped in a model.to(device) call by the runner.
        self.transfer = PrototypeTransfer(cfg.TRAINER.CSPC.SCORER_TAU).to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip_model.logit_scale.detach().float().exp().clamp(max=100.0)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_dtype = self.clip_model.dtype
            features = self.clip_model.encode_image(images.to(self.device, dtype=image_dtype))
            return F.normalize(features.float(), dim=-1)

    def _encode_text_chunk(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("Cannot encode an empty text batch.")
        token_context = self.context_length - self.n_ctx
        token_ids = clip.tokenize(
            list(texts), context_length=token_context, truncate=True
        ).to(self.device)

        dtype = self.clip_model.token_embedding.weight.dtype
        token_embeddings = self.clip_model.token_embedding(token_ids).to(dtype)
        prompt = self.prompt.to(dtype).unsqueeze(0).expand(token_ids.shape[0], -1, -1)
        x = torch.cat((token_embeddings[:, :1, :], prompt, token_embeddings[:, 1:, :]), dim=1)
        x = x + self.clip_model.positional_embedding[: self.context_length].to(dtype)
        x = x.permute(1, 0, 2)
        # CLIP weights are frozen, but the shared prompt still requires
        # gradients through every text-transformer block. Checkpoint the
        # frozen transformer during prompt optimization so activations from
        # all description chunks do not accumulate in GPU memory.
        if torch.is_grad_enabled() and self.prompt.requires_grad:
            x = checkpoint(
                self.clip_model.transformer,
                x,
                use_reentrant=False,
            )
        else:
            x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).to(dtype)

        eot_positions = token_ids.argmax(dim=-1) + self.n_ctx
        rows = torch.arange(x.shape[0], device=x.device)
        text_features = x[rows, eot_positions] @ self.clip_model.text_projection
        return F.normalize(text_features.float(), dim=-1)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)
        if not texts:
            raise ValueError("At least one class description is required.")
        chunks = []
        for start in range(0, len(texts), self.text_chunk_size):
            chunks.append(self._encode_text_chunk(texts[start : start + self.text_chunk_size]))
        return torch.cat(chunks, dim=0)

    def trainable_parameters(self, include_transfer: bool = True):
        params = [self.prompt]
        if include_transfer:
            params.extend(
                [
                    self.transfer.scorer_weight,
                    self.transfer.gate_weight,
                    self.transfer.gate_bias,
                ]
            )
        return params
