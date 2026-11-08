"""PCAS-Prompt MiniImageNet runner.

Run from any directory:
    python /data/lx/fscil/code/PCAS-Prompt/run_script/pcas_mini_imagenet.py

Optional environment overrides:
    PCAS_GPU=2
    PCAS_DATA_ROOT=/data/lx/fscil/data
    PCAS_NUM_COMPONENTS=4
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("PCAS_DATA_ROOT", "/data/lx/fscil/data")
INDEX_LIST_ROOT = os.environ.get(
    "PCAS_INDEX_LIST_ROOT", str(Path(DATA_ROOT) / "index_list")
)
GPU = os.environ.get("PCAS_GPU", "0")
NUM_COMPONENTS = os.environ.get("PCAS_NUM_COMPONENTS", "4")

ARGS = [
    "-project", "base",
    "-dataset", "mini_imagenet",
    "-vit",
    "-dataroot", DATA_ROOT,
    "--index_list_root", INDEX_LIST_ROOT,
    "-gpu", GPU,
    "-seed", "1",
    "-epochs_base", "50",
    "-num_workers", "4",
    "-test_batch_size", "64",
    "--classes_per_batch", "16",
    "--samples_per_class", "4",
    "--num_components", NUM_COMPONENTS,
    "-out", "PCAS",
]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd=str(REPO_ROOT),
        check=True,
    )
