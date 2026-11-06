"""Legacy helper import surface retained for existing repository modules.

PCAS training, prototype construction, affinity losses, and TEEN calibration
are implemented in pcas_trainer.py and pcas_losses.py.
"""
from .pcas_losses import (
    build_prototype_cache,
    leave_one_out_prototype_loss,
    sample_affinity_consistency,
    prototype_affinity_consistency,
    teen_calibrate,
)

__all__ = [
    "build_prototype_cache",
    "leave_one_out_prototype_loss",
    "sample_affinity_consistency",
    "prototype_affinity_consistency",
    "teen_calibrate",
]
