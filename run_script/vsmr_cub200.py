"""Reproducible CUB-200 launcher for the unified VSMR method."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_DATA_ROOT", "/data/lx/fscil/data")
))
INDEX_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("FSCIL_INDEX_LIST_ROOT", os.path.join(DATA_ROOT, "index_list"))
))

ARGS = [
    "--method", "vsmr",
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
    "--vsmr_classes_per_batch", "16",
    "--vsmr_samples_per_class", "4",
    "--vsmr_episode_reference_fraction", "0.7",
    "--vsmr_episode_novel_way", "10",
    "--vsmr_episode_base_query_way", "16",
    "--vsmr_episode_novel_shot", "5",
    "--vsmr_episode_query_per_class", "2",
    "--vsmr_episode_base_query_per_class", "2",
    "-out", "VSMR",
]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd=str(ROOT),
        check=True,
    )
