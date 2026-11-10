# PCAS-Prompt

Prototype-Calibrated Affinity-Consistent Spatial Prompt Learning for Few-Shot Class-Incremental Learning.

This repository implements PCAS-Prompt on top of the existing LGSP-Prompt FSCIL data protocol. The ViT-B/16 backbone is frozen; a four-component decomposed spatial prompt is injected after transformer block 6 without adding tokens. The fixed spatial component bank is fused by image-conditioned attention; all calibration and classification use visual features from the same frozen ViT.

## Method contract

The implementation follows the revised PCAS definition:

- Frozen ImageNet-21K ViT-B/16, 224x224 input, 14x14 patch grid.
- Spatial prompts are generated from block-6 patch features through a shared 96-channel bottleneck and four depthwise 3x3 component branches.
- The CLS feature generates a query in the shared bottleneck space. Component-specific sigmoid gates modulate the normalized query, which is cosine-matched with normalized component keys; a temperature-scaled softmax produces image-conditioned component weights. These weights fuse the fixed spatial components.
- The residual scale is parameterized as `gamma_max * sigmoid(rho)`, initialized to 0.1 and bounded above by 0.5.
- Only patch tokens receive the residual spatial prompt. CLS is unchanged at the injection point and the token count remains 197.
- Base training optimizes the spatial prompt with leave-one-out component-prototype classification, sample-prototype affinity KL, and batch class-prototype affinity KL. The combined affinity coefficient is linearly warmed up over the first 10 epochs by default.
- Teacher features/prototypes come from the original frozen ViT. Student features/prototypes come from the final PCAS-enhanced ViT.
- After base training, PCAS and the backbone are frozen. Base-class prototypes are stored separately for every isolated spatial component in the final PCAS feature space. Teacher global features are used only for affinity-consistency training.
- Each novel class has one prototype per spatial component, computed from its real K-shot support set. Each component is calibrated only against the corresponding component prototypes of base classes; the calibrated component prototypes are retained separately for final query-weighted component-similarity classification. Prompt parameters remain frozen after base training.

## Repository layout

- `models/base/pcas_modules.py`: frozen ViT-B/16 and PCAS spatial prompt bank.
- `models/base/pcas_losses.py`: component prototype caches, component-weighted leave-one-out classification, component-wise calibration, and global-feature affinity-consistency losses.
- `models/base/pcas_trainer.py`: base training, cache refresh, real incremental sessions, metrics and checkpoints.
- `models/base/ViT_Network.py`: compatibility import for the PCAS model.
- `models/base/ViT_fscil_trainer.py`: compatibility import for the PCAS trainer.

## Data preparation

The loaders accept the shared dataset root used in the experiment server. For the layout shown in the project setup, use `/data/lx/fscil/data` as `-dataroot`. The resolver maps each dataset to its actual nested directory and reports a clear error if required files are missing. Session split files default to `<dataroot>/index_list`; `--index_list_root` can override this.

```text
data/
  index_list/
    cub200/session_1.txt ... session_11.txt
    FGVCAircraft/session_1.txt ... session_11.txt
    iNF200/session_1.txt ... session_11.txt
    cifar100/session_1.txt ... session_9.txt
    mini_imagenet/session_1.txt ... session_9.txt
```

The split files must follow the format expected by each original dataset loader. CUB base training uses all base-class train images; CUB and other novel sessions use the corresponding support split.

## Installation

Use a PyTorch/torchvision build compatible with the selected CUDA runtime. Install the versions in `requirements.txt` and ensure the repository's `timm` dependency is available. Pretrained ViT-B/16 ImageNet-21K weights are loaded through the existing timm model registration; the first run may need network access or a locally cached pretrained checkpoint.

## Run

Use the server's shared data directory for all three datasets:

```bash
DATA_ROOT=/data/lx/fscil/data

# CUB-200-2011
python train.py -project base -dataset cub200 -vit \
  -dataroot "$DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4

# CIFAR-100
python train.py -project base -dataset cifar100 -vit \
  -dataroot "$DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4

# MiniImageNet
python train.py -project base -dataset mini_imagenet -vit \
  -dataroot "$DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4
```

The default `--index_list_root` is `$DATA_ROOT/index_list`; pass it explicitly only when your session splits are stored elsewhere. Before constructing the model, setup checks that every incremental session file exists.


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

Runs are saved under `checkpoint/PCAS/<dataset>/<timestamp>/`:

- `pcas_base.pth`: final frozen PCAS model, final component-wise base prototypes, and global teacher/student prototypes for diagnostics.
- `pcas_final.pth`: model, all component-wise class prototypes, session accuracies and summary.
- `results.txt`: per-epoch and per-session records.

The summary reports mean session class-average accuracy (AA) and PD = base-session class-average accuracy minus final-session class-average accuracy. Each session also reports base/novel accuracy, their harmonic mean (HMean), novel-to-base error rate (TBR), attention-weighted component-prototype distance and component usage/attention-entropy diagnostics. TBR is the fraction of novel test examples predicted as base classes; it is not limited to semantically nearest base classes. Component usage and attention entropy are descriptive diagnostics, not proof by themselves that components are functionally distinct.

## Important limitations

- Session split files and image datasets are not bundled in this repository; provide them locally.
- The current implementation uses the original dataset loaders and their class/session definitions.
- Full-scale CUDA experiments are required to validate performance claims. A successful code run does not imply an AA/PD improvement.
