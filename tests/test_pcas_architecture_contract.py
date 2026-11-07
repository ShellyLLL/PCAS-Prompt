import torch

from models.base.pcas_modules import PCASPrompt


def test_component_bank_supports_capacity_candidates_and_preserves_shapes():
    for m in (1, 2, 4, 8, 16):
        module = PCASPrompt(
            embed_dim=32, bottleneck_dim=8, num_components=m, dropout=0.0
        )
        patches = torch.randn(3, 16, 32, requires_grad=True)
        cls = torch.randn(3, 32)
        enhanced, weights = module(patches, cls, (4, 4))
        assert enhanced.shape == patches.shape
        assert weights.shape == (3, m)
        assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6)
        (enhanced.square().mean() + weights.square().mean()).backward()
        assert module.components[0].depthwise.weight.grad is not None


def test_single_component_is_valid_and_residual_is_initialized_correctly():
    module = PCASPrompt(
        embed_dim=32, bottleneck_dim=8, num_components=1,
        dropout=0.0, residual_init=0.1, residual_max=0.5
    )
    output, weights = module(torch.randn(2, 16, 32), torch.randn(2, 32), (4, 4))
    assert output.shape == (2, 16, 32)
    assert torch.equal(weights, torch.ones(2, 1))
    assert abs(module.residual_scale.item() - 0.1) < 1e-6


def test_residual_scale_cannot_exceed_configured_maximum():
    module = PCASPrompt(
        embed_dim=32, bottleneck_dim=8, num_components=2,
        residual_init=0.1, residual_max=0.5
    )
    with torch.no_grad():
        module.gamma_logit.fill_(100.0)
    assert module.residual_scale.item() <= 0.5


def test_query_layer_norm_operates_on_projected_dimension_and_sigmoid_gate():
    module = PCASPrompt(
        embed_dim=32, bottleneck_dim=8, num_components=2, dropout=0.0
    )
    assert module.query.out_features == 8
    assert tuple(module.query_norm.normalized_shape) == (8,)
    weights = module.route(torch.randn(4, 32))
    assert weights.shape == (4, 2)
    assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-6)


def test_invalid_component_and_residual_configurations_are_rejected():
    try:
        PCASPrompt(embed_dim=32, bottleneck_dim=8, num_components=0)
    except ValueError:
        pass
    else:
        raise AssertionError("M=0 must be rejected")
    try:
        PCASPrompt(
            embed_dim=32, bottleneck_dim=8, num_components=1,
            residual_init=0.6, residual_max=0.5
        )
    except ValueError:
        pass
    else:
        raise AssertionError("residual_init above residual_max must be rejected")
