"""Compatibility entry point for PCAS-Prompt FSCIL training."""
from .pcas_trainer import PCASFSCILTrainer


class ViT_FSCILTrainer(PCASFSCILTrainer):
    """The main ViT FSCIL trainer now follows the PCAS mathematical protocol."""
    pass
