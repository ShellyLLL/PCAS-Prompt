# PCAS-Prompt

Prototype-Calibrated Affinity-Consistent Spatial Prompt Learning for Few-Shot Class-Incremental Learning.

This repository implements PCAS-Prompt on top of the existing LGSP-Prompt FSCIL data protocol. The ViT-B/16 backbone is frozen; a four-component decomposed spatial prompt is injected after transformer block 6 without adding tokens. CODA-Prompt is used only as a reference for the decomposed Query-Key-Attention routing.

## Method contract

The implementation follows the revised PCAS definition:

- Frozen ImageNet-21K ViT-B/16, 224x224 input, 14x14 patch grid.
- Spatial prompts are generated from block-6 patch features through a shared 96-channel bottleneck and four depthwise 3x3 component branches.
- The CLS feature generates the query. Component-specific A and K vectors produce cosine-like routing scores, followed by a softmax over all four components.
- Only patch tokens receive the residual spatial prompt. CLS is unchanged at the injection point and the token count remains 197.
- Base training optimizes the spatial prompt and base classifier with CE, leave-one-out prototype CE, sample-prototype affinity KL, and batch class-prototype affinity KL.
- Teacher features/prototypes come from the original frozen ViT. Student features/prototypes come from the final PCAS-enhanced ViT.
- After base training, PCAS and the backbone are frozen. The base prototypes used by TEEN are recomputed from the final PCAS student feature space. Teacher prototypes are retained only for auditing and PCAC.
- Each novel class prototype is computed from its K-shot support set and calibrated by TEEN against the frozen PCAS student base prototypes. No prompt fine-tuning, replay, or pseudo-incremental episodes are used.

## Repository layout

- `models/base/pcas_modules.py`: frozen ViT-B/16 and PCAS spatial prompt bank.
- `models/base/pcas_losses.py`: prototype cache, LOO prototype loss, PCAC affinity losses, TEEN.
- `models/base/pcas_trainer.py`: base training, cache refresh, real incremental sessions, metrics and checkpoints.
- `models/base/ViT_Network.py`: compatibility import for the PCAS model.
- `models/base/ViT_fscil_trainer.py`: compatibility import for the PCAS trainer.

## Data preparation

Keep the dataset layout expected by the original LGSP loaders. Session split files are configured independently from the image root:

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

Example CUB-200 run:

```bash
python train.py \
  -project base \
  -dataset cub200 \
  -vit \
  -dataroot /path/to/dataset-root \
  --index_list_root /path/to/data/index_list \
  -gpu 0 \
  -seed 1 \
  -epochs_base 100 \
  -batch_size_base 64 \
  --classes_per_batch 16 \
  --samples_per_class 4
```

For CUB, `dataroot` should contain `CUB_200_2011/images.txt` and the other CUB metadata files. The index-list root should contain `cub200/session_N.txt` files. Adjust the dataset root and split root for other datasets.

Important PCAS defaults:

| Parameter | Default |
|---|---:|
| Prompt insertion | after block 6 |
| Components | 4 |
| Bottleneck | 96 |
| Router temperature | 0.2 |
| Prompt dropout | 0.1 |
| Residual scale initial value | 0.1 |
| Classes per batch | 16 |
| Samples per class | 4 |
| Prompt learning rate | 1e-3 |
| Prototype loss weight | 0.5 |
| Affinity loss weight | 0.1 |
| Class-affinity coefficient | 0.5 |
| Affinity temperature | 16 |
| TEEN alpha / tau | 0.5 / 16 |

These are reproducible starting settings, not claimed dataset-wise optima.

## Outputs

Runs are saved under `checkpoint/PCAS/<dataset>/<timestamp>/`:

- `pcas_base.pth`: final frozen PCAS model, final student base prototypes, and teacher prototypes for audit.
- `pcas_final.pth`: model, all class prototypes, session accuracies and summary.
- `results.txt`: per-epoch and per-session records.

The summary reports mean session accuracy (AA_session_mean) and PD = base-session accuracy minus final-session accuracy. For paper reporting, also report per-class averaged accuracy and the exact metric convention used by the comparison methods.

## Important limitations

- Session split files and image datasets are not bundled in this repository; provide them locally.
- The current implementation uses the original dataset loaders and their class/session definitions.
- Full-scale CUDA experiments are required to validate performance claims. A successful code run does not imply an AA/PD improvement.
