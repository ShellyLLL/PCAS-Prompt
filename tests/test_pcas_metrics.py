import torch
import torch.nn.functional as F

from models.base.pcas_metrics import (
    component_weight_statistics,
    mean_component_prototype_distance,
)


def test_component_weight_statistics_are_normalized():
    weights = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    metrics = component_weight_statistics(weights)
    assert len(metrics["component_usage"]) == 3
    assert abs(sum(metrics["component_usage"]) - 1.0) < 1e-6
    assert 0.0 <= metrics["component_normalized_entropy"] <= 1.0


def test_component_weight_statistics_reject_invalid_rows():
    try:
        component_weight_statistics(torch.tensor([[0.2, 0.2], [0.5, 0.5]]))
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid component weights should be rejected.")


def test_component_prototype_distance_is_zero_for_matching_features():
    prototypes = F.normalize(torch.randn(3, 5, 8), dim=-1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    features = prototypes[:, labels, :].permute(1, 0, 2)
    weights = torch.softmax(torch.randn(6, 3), dim=-1)
    distance = mean_component_prototype_distance(
        features, weights, labels, prototypes
    )
    assert distance < 1e-6
