import torch

from models.base.lsp_modules import LocalSpatialPromptPool
from models.base.lsp_losses import build_prototype_cache, prototype_logits
from models.base.lsp_fscil_trainer import batch_leave_one_out_logits


def test_local_spatial_prompt_pool_selects_and_composes_topk():
    torch.manual_seed(7)
    module = LocalSpatialPromptPool(
        embed_dim=16, pool_size=5, top_k=2, hidden_dim=4,
        kernel_size1=3, kernel_size2=5, dropout=0.0, temperature=0.2,
    )
    images = torch.randn(3, 3, 16, 16)
    cls = torch.randn(3, 16)
    prompt, aux = module(images, cls)
    assert prompt.shape == images.shape
    assert aux["selected_indices"].shape == (3, 2)
    assert aux["weights"].shape == (3, 2)
    torch.testing.assert_close(
        aux["weights"].sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6
    )
    (prompt.square().mean() + aux["weights"].square().mean()).backward()
    assert module.prompt_keys.grad is not None
    assert torch.isfinite(module.prompt_keys.grad).all()


def test_prototype_cache_and_cosine_classifier():
    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    cache = build_prototype_cache(features, labels, num_classes=2)
    logits = prototype_logits(features, cache["prototypes"], logit_scale=10.0)
    assert logits.shape == (4, 2)
    assert torch.equal(logits.argmax(dim=1), labels)


def test_batch_leave_one_out_prototype_logits():
    features = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
    ])
    labels = torch.tensor([3, 3, 8, 8])
    logits, targets = batch_leave_one_out_logits(features, labels, scale=8.0)
    assert logits.shape == (4, 2)
    assert torch.equal(targets, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(logits.argmax(dim=1), targets)
