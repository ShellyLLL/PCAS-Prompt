"""Compatibility entry point for the unified APL-PRSN FSCIL trainer."""
from .apl_prsn_trainer import APLPRSNFSCILTrainer


class ViT_FSCILTrainer(APLPRSNFSCILTrainer):
    """The default ViT FSCIL path is the unified APL-PRSN architecture."""

    pass
