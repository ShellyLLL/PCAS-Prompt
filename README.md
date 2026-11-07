# PCAS-Prompt

**Prototype-Calibrated Affinity-Consistent Spatial Prompt Learning for Few-Shot Class-Incremental Learning.**

This repository implements PCAS-Prompt on the existing LGSP-Prompt FSCIL data protocol. The method uses a frozen ImageNet-21K ViT-B/16 and a fixed-capacity decomposed spatial-prompt bank inserted after transformer block 6. The bank is learned only during base training and frozen for all incremental sessions. CODA-Prompt is a reference for decomposed Query-Key attention; PCAS does not use task-wise component expansion.

## Unified architecture

### Frozen backbone and spatial insertion

- Frozen pretrained ViT-B/16, 224x224 input, 14x14 patch grid, 12 transformer blocks.
- Prompt insertion after block 6.
- Only patch tokens receive the spatial residual. CLS is unchanged at the insertion point; later frozen blocks can attend to enhanced patches.
- Sequence length remains 197; no prompt tokens are appended.

Let block-6 patch features be F_l in R^(B x D x H x W), CLS be h_l in R^(B x D), D=768 and H=W=14.

### Decomposed spatial prompt components

A shared 1x1 projection produces the bottleneck map:

U = W_d F_l, where bottleneck width r defaults to 96.

Each component m has its own depthwise 3x3 spatial convolution and pointwise up-projection:

P_m = LN(W_m_up(Dropout2d(GELU(DWConv_m(U))))).

Each P_m is a B x D x H x W feature map. The shared projection controls input width; independent spatial branches provide distinct learnable local transformations. Component semantics are not manually assigned to texture, shape, edge, or a particular class.

### Query-Key-Attention composition

The block-6 CLS feature generates the query:

q = LN(W_q h_l).

Each component has a learned gate vector A_m and key K_m:

q_m = normalize(q * sigmoid(A_m))
s_m = cosine(q_m, K_m)
w_m = softmax(s_m / T_g)
P = sum_m w_m P_m.

All components participate through soft weighting. There is no Top-K truncation, no additional router network, and no load-balancing objective.

### Spatial residual

F'_l = F_l + gamma P
gamma = gamma_max * sigmoid(rho).

The default gamma_max is 0.5 and initialization gives gamma=0.1. The upper bound is enforced by the parameterization.

## Component-count definition and selection

M is a **fixed model-capacity hyperparameter**, not the number of FSCIL sessions and not a dynamically growing expert count. M is selected before the final experiment and remains fixed from base training through the last incremental session.

Candidate set: {1, 2, 4, 8, 16}. M=1 is mandatory as the single-generator control. For D=768 and bottleneck r, each additional component contributes D*r + 11*r + D parameters. The prompt bank parameter count is:

P_prompt = 2*D*r + 2*r + 2*D + 1 + M*(D*r + 11*r + D).

Choose M by macro class accuracy on a stratified holdout taken only from the base training split. Exact ties prefer the smaller prompt parameter count, then smaller M. Do not choose M using novel-session or final test accuracy. Repeat selection across training seeds while keeping the validation split seed fixed; aggregate validation scores before fixing M. No dataset-specific optimal M is claimed before these experiments.

Run selection first:

```bash
DATA_ROOT=/data/lx/fscil/data
python run_script/select_pcas_components.py -project base -dataset cub200 -vit \
  -dataroot "$DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4 \
  --component_candidates 1 2 4 8 16 \
  --base_val_fraction 0.1 --base_val_seed 1
```

The script trains every candidate on the same stratified subset of base training data and evaluates only on its held-out base-training validation subset. Candidate checkpoints and pcas_component_selection.json are saved under one timestamped directory. Repeat with -seed 2, -seed 3, etc. while keeping --base_val_seed fixed.

After selecting M, run the final FSCIL experiment using all base training data and explicitly pass the selected --num_components value:

```bash
python train.py -project base -dataset cub200 -vit \
  -dataroot "$DATA_ROOT" -gpu 0 -seed 1 \
  -epochs_base 50 -num_workers 4 -test_batch_size 64 \
  --classes_per_batch 16 --samples_per_class 4 --num_components 4
```

Replace 4 with the selected value. The parser default of 4 is retained only as a reproducible baseline, not as an established optimum.

## Base training and PCAC

The original frozen ViT provides teacher features and teacher base prototypes. The PCAS-enhanced frozen-backbone path provides student features and student base prototypes. Student prototype caches are refreshed once per epoch.

The base objective is:

L = L_proto-CE + lambda_p * L_LOO + lambda_a(t) * (L_sample-aff + beta * L_batch-proto-aff).

- L_proto-CE: cosine classification against the current student base-prototype cache. This uses the same prototype classifier as final inference; there is no separate trainable linear head.
- L_LOO: leave-one-out prototype cross-entropy. The current training sample is removed from its own class prototype.
- L_sample-aff: KL consistency between teacher and student sample-to-base-prototype affinity distributions, excluding the sample's ground-truth class.
- L_batch-proto-aff: KL consistency between pairwise class-prototype affinity distributions computed from the class-balanced batch; diagonal self-affinity is excluded.
- lambda_a(t): linear warm-up of the combined affinity coefficient.

The class-balanced sampler is required by the batch prototype-affinity term and leave-one-out objective. Configure it with --classes_per_batch and --samples_per_class.

## Incremental protocol

After base training:

1. Freeze ViT, spatial components, Query-Key parameters, residual scale, and all base-trained prompt parameters.
2. Recompute base prototypes from the final PCAS student feature space.
3. Compute each new class prototype from its real K-shot support set.
4. Calibrate new prototypes with TEEN using only frozen PCAS student base prototypes.
5. Append calibrated prototypes at global class IDs and classify all seen classes in one cosine feature space.

There is no prompt fine-tuning, rehearsal, replay, or pseudo-incremental episode training. Existing base prototypes are not rewritten in later sessions. Every configured dataset is checked for exact K-shot support counts and expected global class IDs.

## Repository layout

- models/base/pcas_modules.py: frozen ViT-B/16 and PCAS spatial-prompt bank.
- models/base/pcas_losses.py: prototype cache, LOO loss, PCAC affinity losses, and TEEN.
- models/base/pcas_metrics.py: feature-geometry and routing diagnostics.
- models/base/pcas_trainer.py: base training, cache refresh, real incremental sessions, metrics, checkpoints.
- models/base/pcas_selection.py: base-only validation split and component-capacity selection.
- run_script/select_pcas_components.py: executable component selection workflow.
- models/base/ViT_Network.py and ViT_fscil_trainer.py: compatibility imports.

## Data and installation

The loaders accept the shared dataset root used in the experiment server. For the described setup use /data/lx/fscil/data as -dataroot. Session splits default to <dataroot>/index_list; --index_list_root overrides this. Split files must follow the format expected by each original dataset loader. Dataset images and split files are not bundled.

Use a PyTorch/torchvision build compatible with the CUDA runtime, install requirements.txt, and ensure timm is available. The first pretrained ViT-B/16 ImageNet-21K load may require network access or a locally cached checkpoint.

## Outputs and metrics

Runs are saved under checkpoint/PCAS/<dataset>/<timestamp>/:

- pcas_base.pth: final frozen PCAS model, final student base prototypes, teacher prototypes for audit.
- pcas_final.pth: model, all class prototypes, session accuracies, and summary.
- results.txt: per-epoch and per-session records.

AA is mean session class-average accuracy. PD is base-session class-average accuracy minus final-session class-average accuracy. Per-session diagnostics include base/novel accuracy, HMean, novel-to-base error rate (TBR), mean cosine distance to the ground-truth prototype, base-prototype affinity drift, and routing usage/entropy. Routing entropy is diagnostic, not proof that components are functionally distinct.

## Limitations

Full-scale CUDA experiments and multi-seed comparisons are required to establish any AA/PD improvement or preferred component count. A successful code run alone does not demonstrate a performance improvement.
