"""Reproducible CUB-200 PCAS launcher.

Override FSCIL_DATA_ROOT and FSCIL_INDEX_LIST_ROOT when the experiment
server stores datasets or session splits outside the documented defaults.
"""
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_DATA_ROOT", "/data/lx/fscil/data")
))
INDEX_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_INDEX_LIST_ROOT", os.path.join(DATA_ROOT, "index_list"))
))

ARGS = [
    "-project", "base",
    "-dataset", "cub200",
    "-vit",
    "-dataroot", DATA_ROOT,
    "--index_list_root", INDEX_ROOT,
    "-gpu", "0",
    "-seed", "1",
    "-epochs_base", "50",
    "-num_workers", "4",
    "-test_batch_size", "64",
    "--classes_per_batch", "16",
    "--samples_per_class", "4",
    "-out", "PCAS",
]


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd=str(REPO_ROOT),
        check=True,
    )
