import torch

from models.base.pcas_metrics import (
    affinity_drift,
    mean_prototype_distance,
    routing_diversity,
)


def test_affinity_drift_is_zero_for_identical_prototypes():
    prototypes = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    assert affinity_drift(prototypes, prototypes) < 1e-7


def test_mean_prototype_distance_is_zero_for_matching_prototypes():
    prototypes = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    features = prototypes[labels]
    assert mean_prototype_distance(features, labels, prototypes) < 1e-6


def test_routing_diversity_reports_normalized_usage():
    routing = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    metrics = routing_diversity(routing)
    assert len(metrics["routing_usage"]) == 3
    assert abs(sum(metrics["routing_usage"]) - 1.0) < 1e-6
    assert 0.0 <= metrics["routing_normalized_entropy"] <= 1.0


def test_routing_diversity_rejects_invalid_rows():
    routing = torch.tensor([[0.2, 0.2], [0.5, 0.5]])
    try:
        routing_diversity(routing)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid routing weights should be rejected.")
