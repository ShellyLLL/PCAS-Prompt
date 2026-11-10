This repository implements PCAS-Prompt on top of the existing FSCIL data protocol. The ViT-B/16 backbone is frozen; a fixed-capacity decomposed spatial prompt is injected after transformer block 6 without adding tokens. All spatial components are composed before injection and the model emits exactly one final student feature per image.

## Method contract

The implementation follows one unified PCAS architecture:

- Frozen ImageNet-21K ViT-B/16, 224x224 input, 14x14 Patch grid.
- Patch features from block 6 pass through one shared bottleneck and a fixed bank of spatial transformation components.
- The CLS feature generates a Query. Component-specific gates and Keys produce continuous Query-Key-Attention weights over the fixed spatial components.
- The weighted sum of spatial prompt maps is injected once into the Patch tokens. CLS is unchanged at injection; token count remains 197.
- The frozen ViT tail runs exactly once. Each image produces one normalized student feature \(z_i\) and its component weights \(\omega_i\). No isolated component features or independent component encoders are created.
- Class/component prototypes are conditioned by the image-level component weights:
  \[
  p_{c,m}=\operatorname{Norm}\left(
  \frac{\sum_{i:y_i=c}\omega_{i,m}z_i}
  {\sum_{i:y_i=c}\omega_{i,m}}
  \right).
  \]
- Query classification recomposes one class prototype per query:
  \[
  s_c(x)=\kappa\,z(x)^\top
  \operatorname{Norm}\left(\sum_m\omega_m(x)p_{c,m}\right).
  \]
  Base training and real FSCIL inference use this same score definition.
- Base training uses weighted leave-one-out prototype classification, component-specific prototype supervision over the same unified feature, and the two PCAC affinity objectives. The original frozen ViT supplies teacher targets only for PCAC.
- At each real incremental session, each novel class's component-conditioned prototypes are computed from its actual K-shot support features and weights. Each component is calibrated only against the matching component prototypes of base classes. The calibrated prototypes are appended without changing old prototypes.
- The ViT and learned spatial prompt are frozen throughout incremental sessions. There is no replay, pseudo-class construction, pseudo-incremental training, or incremental Prompt fine-tuning.

## Repository layout

- `models/base/pcas_modules.py`: frozen ViT-B/16 and PCAS spatial prompt bank.
- `models/base/pcas_losses.py`: component-conditioned prototype construction, weighted leave-one-out classification, same-component calibration, and PCAC affinity losses.
- `models/base/pcas_trainer.py`: single-stream base training, real K-shot prototype construction, incremental calibration, metrics and checkpoints.
- `models/base/ViT_Network.py`: compatibility import for the PCAS model.
- `models/base/ViT_fscil_trainer.py`: compatibility import for the PCAS trainer.

## Data preparation and path contract

The default shared data root is `/data/lx/fscil/data`, matching the repository's CUB launcher and original experiment configuration. Override it with `FSCIL_DATA_ROOT` or `-dataroot`. The loaders validate the actual files they consume before model construction; they do not silently fall back to another machine-specific directory.

Expected layouts under `$FSCIL_DATA_ROOT`:

```text
data/
  CUB_200_2011/
    CUB_200_2011/
      images/
      images.txt
      train_test_split.txt
      image_class_labels.txt
      classes.txt
  CIFAR_100/
    cifar-100-python/
      train
      test
      meta
  miniimagenet/
    images/
    train.csv
    test.csv
    # train.csv/test.csv may instead be under miniimagenet/split/
  fgvc-aircraft-2013b/
    data/
      images/
      variants.txt
      images_variant_train.txt
      images_variant_test.txt
  iNF200/
    train_mini/
    val/
  index_list/
    cub200/
      session_2.txt ... session_11.txt
    cifar100/
      session_2.txt ... session_9.txt
    mini_imagenet/
      session_2.txt ... session_9.txt
    FGVCAircraft/
      session_1.txt ... session_11.txt
    iNF200/
      session_1.txt ... session_11.txt
```

The root resolver also accepts a `-dataroot` that points directly at a dataset directory (for example, `.../CUB_200_2011`, `.../CIFAR_100`, `.../miniimagenet`, `.../fgvc-aircraft-2013b`, or `.../iNF200`) and normalizes it to the root expected by that dataset's existing loader. If the session lists live elsewhere, set `--index_list_root` or `FSCIL_INDEX_LIST_ROOT`.

**Session numbering follows the original loader protocol:** trainer session 1 (the first incremental task) reads `session_2.txt`. `session_1.txt` is the base-session split for FGVCAircraft and iNF200; CUB, CIFAR-100 and MiniImageNet select base classes directly and do not consume that file in base training. The split file contents must retain each loader's original format:

- CUB-200: image paths resolvable to entries in `CUB_200_2011/images/`.
- CIFAR-100: 25 training-array indices for each 5-way, 5-shot incremental task.
- MiniImageNet: image filenames or relative paths resolvable through `miniimagenet/train.csv`.
- FGVC-Aircraft: image IDs or paths resolvable through `fgvc-aircraft-2013b/data/images/`.
- iNF200: paths in the form `train_mini/<class_folder>/<image_name>`.

The setup checks all required session files before constructing the ViT. Incremental sessions additionally validate the expected global class IDs and exact 5-shot support count. The path-based datasets audit duplicate support/test paths and train/test overlap.

## Installation

Use a PyTorch/torchvision build compatible with the selected CUDA runtime. Install the versions in `requirements.txt` and ensure the repository's `timm` dependency is available. Pretrained ViT-B/16 ImageNet-21K weights are loaded through the repository's timm model registration. PyTorch Hub caches downloaded weights under its cache root (controlled by `TORCH_HOME`); set `TORCH_HOME` to a persistent cache directory on the experiment server. Use `--scratch` only when intentionally training without pretrained weights.

## Run

Use the server's shared data directory. The root can be overridden without editing source files:

```bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

# CUB-200-2011
python train.py -project base -dataset cub200 -vit \
  -dataroot "$FSCIL_DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4

# CIFAR-100
python train.py -project base -dataset cifar100 -vit \
  -dataroot "$FSCIL_DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4

# MiniImageNet
python train.py -project base -dataset mini_imagenet -vit \
  -dataroot "$FSCIL_DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4
```

The default `--index_list_root` is `$FSCIL_DATA_ROOT/index_list`; pass it explicitly only when your session splits are stored elsewhere. `train.py` also reads `FSCIL_DATA_ROOT` and `FSCIL_INDEX_LIST_ROOT` directly.


Important PCAS defaults:

| Parameter | Default |
|---|---:|
| Prompt insertion | after block 6 |
| Components | 4 |
| Bottleneck | 96 |
| Component attention temperature | 0.2 |
| Prompt dropout | 0.1 |
| Residual scale initial value | 0.1 |
| Classes per batch | 16 |
| Samples per class | 4 |
| Prompt learning rate | 1e-3 |
| Affinity loss weight | 0.1 |
| Class-affinity coefficient | 0.5 |
| Affinity temperature | 16 |
| Component calibration alpha / tau | 0.5 / 16 |
| Component calibration top-r | 5 |

These are reproducible starting settings, not claimed dataset-wise optima.

## Outputs

Runs are saved under `<repository_root>/checkpoint/PCAS/<dataset>/<timestamp>/`, independent of the shell's current working directory.

- `pcas_base.pth`: final frozen PCAS model, component-conditioned base prototypes, and global teacher/student prototypes for diagnostics.
- `pcas_final.pth`: model, all component-conditioned class prototypes, session accuracies and summary.
- `results.txt`: per-epoch and per-session records.

The summary reports mean session class-average accuracy (AA) and PD = base-session class-average accuracy minus final-session class-average accuracy. Each session also reports base/novel accuracy, their harmonic mean (HMean), novel-to-base error rate (TBR), distance to the query-conditioned recomposed target prototype, and component usage/attention-entropy diagnostics. TBR is the fraction of novel test examples predicted as base classes; it is not limited to semantically nearest base classes. Component usage and attention entropy are descriptive diagnostics, not proof by themselves that components are functionally distinct.

## Important limitations

- Session split files and image datasets are not bundled in this repository; provide them locally.
- The current implementation uses the original dataset loaders and their class/session definitions.
- Full-scale CUDA experiments are required to validate performance claims. A successful code run does not imply an AA/PD improvement.
