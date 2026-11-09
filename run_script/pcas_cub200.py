"""Run one CUB200 pseudo-FSCIL validation configuration.

Change NUM_COMPONENTS manually between runs. Keep VALIDATION_SEED fixed so
all M configurations use exactly the same pseudo-base/pseudo-novel classes
and the same train/support/query sample partition.
"""
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Experiment controls: change NUM_COMPONENTS only for the M ablation.
BOTTLENECK_DIM = 192
NUM_COMPONENTS = 1
TRAIN_SEED = 1
VALIDATION_SEED = 2026
BASE_QUERY_FRACTION = 0.2

ARGS = [
    "-project", "base",
    "-dataset", "cub200",
    "-vit",
    "-dataroot", "/data/lx/fscil/data",

    "-gpu", "0",
    "-seed", str(TRAIN_SEED),
    "-epochs_base", "50",
    "-num_workers", "4",
    "-test_batch_size", "64",
    "--classes_per_batch", "16",
    "--samples_per_class", "4",

    # Validation mode: build the pseudo-FSCIL task from official TRAIN data.
    # Do not pass --index_list_root here; official test/session split files
    # must not be read during hyperparameter selection.
    "--validation_mode",
    "--validation_seed", str(VALIDATION_SEED),
    "--validation_base_query_fraction", str(BASE_QUERY_FRACTION),

    # PCAS configuration.
    "--pcas_layer", "6",
    "--bottleneck_dim", str(BOTTLENECK_DIM),
    "--num_components", str(NUM_COMPONENTS),
    "--affinity_warmup_epochs", "10",

    # PCASFSCILTrainer appends "_validation" to this output root automatically.
    "-out", "cub200_d{}_m{}".format(BOTTLENECK_DIM, NUM_COMPONENTS),
]


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd=str(PROJECT_ROOT),
        check=True,
    )
