import subprocess
import sys

ARGS = [
    "-project", "base",
    "-dataset", "cub200",
    "-vit",
    "-dataroot", "/data/lx/fscil/data",
    "--index_list_root", "/data/lx/fscil/data/index_list",
    "-gpu", "0",
    "-seed", "1",
    "-epochs_base", "100",
    "-num_workers", "4",
    "-test_batch_size", "64",
    "--classes_per_batch", "16",
    "--samples_per_class", "4",
    "-out", "PCAS",
]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "train.py"] + ARGS,
        cwd="/data/lx/fscil/code/PCAS-Prompt",
        check=True,
    )
