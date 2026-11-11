# APL-PRSN for Few-Shot Class-Incremental Learning

This branch's default method is a unified **APL + PRSN** architecture. It uses the repository's existing ImageNet-21K pretrained ViT-B/16 and preserves the original FSCIL dataset/session loaders.

## Unified architecture

1. **Frozen visual backbone.** The ImageNet-21K ViT-B/16 remains frozen in every session. Its final-block CLS-to-Patch attention is reused as an APL prior; no DINO checkpoint or second backbone is introduced.
2. **APL shared part learning.** A fixed set of learnable part queries attends to per-image Patch priors. The queries induce hard Patch-to-part assignments and produce an unordered set of aligned local part descriptors. No additional Transformer tokens or independent Slot Attention module is used.
3. **PRSN global branch.** Global class prototypes are refined with a learnable prototype self-attention module.
4. **PRSN local branch (PFRSM).** Novel/local prototypes retrieve semantically similar base-class part memories and reconstruct local prototypes with query-conditioned attention. Cross-image semantic alignment (CISA) aligns global and local branch predictions for distinct images of the same class.
5. **Joint classifier.** Global cosine-prototype logits and query-conditioned local reconstruction logits are combined with fixed, explicit weights.
6. **Strict FSCIL update.** Base training learns APL and PRSN only. At each incremental session, the backbone, APL, and PRSN are frozen. Exactly the real K-shot support samples form each novel prototype. Calibration retrieves from the immutable base memory only. Old class prototypes are append-only and are never rewritten; test images/labels are used only for evaluation.

The APL prior is deliberately adapted to the available ViT: final-block CLS-to-Patch attention is extracted from the same frozen ImageNet-21K encoder. This is not a claim to reproduce a DINO-based prior.

## Base training objectives

- Leave-one-out global prototype classification.
- Leave-one-out PRSN local reconstruction classification.
- PRSN cross-image semantic alignment (CISA).
- APL all-min contrastive part learning.
- APL part-diversity regularization.

The leave-one-out construction excludes each training sample from its own class prototype to prevent self-match leakage. Class-balanced batches contain multiple classes and multiple images per class.

## Run CUB-200

```bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

python train.py --method apl_prsn -project base -dataset cub200 -vit \
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \
  -gpu 0 -seed 1 -epochs_base 50 -num_workers 4 -test_batch_size 64
```

The same method entry supports CIFAR-100, MiniImageNet, FGVC-Aircraft, and iNF200 subject to the repository's original dataset/session split files and directory layouts.

## Main settings

| Setting | Default |
|---|---:|
| Shared APL part queries | 6 |
| APL prior top fraction | 0.20 |
| APL hard-assignment temperature | 0.20 |
| PRSN base-memory top-k | 5 |
| Base batch classes | 16 |
| Samples per base class per batch | 4 |
| APL/PRSN learning rate | 2e-4 |
| Global/local inference weights | 0.5 / 0.5 |
| PRSN logit scale | 16 |

These are initial experimental settings, not claims of optimality. Tune using base-class validation only.

## Implementation

- `models/base/apl_prsn_modules.py`: frozen ViT, APL shared part prompts, PRSN global refiner and PFRSM.
- `models/base/apl_prsn_losses.py`: class/leave-one-out prototypes, APL all-min and diversity losses, PRSN CISA, cosine classifier.
- `models/base/apl_prsn_trainer.py`: base training, immutable base memory, real K-shot incremental updates, evaluation and AA/PD reporting.
- `models/base/ViT_fscil_trainer.py`: compatibility entry point routed to APL-PRSN.
- `tests/test_apl_prsn.py`: tensor shape, gradients, leave-one-out, calibration, frozen-backbone and incremental immutability tests.

The previous PCAS, LSP and Prompt-Slot-OT files are retained as historical/explicit baselines only. They are not part of the default APL-PRSN computation graph.

## Evaluation

The trainer reports session sample accuracy, class-average accuracy, base/novel accuracy, harmonic mean, novel-to-base error, task-wise accuracy, AA (mean session class-average accuracy) and PD (base-session class-average accuracy minus final-session class-average accuracy).

A successful unit-test run does not establish an accuracy improvement. Full FSCIL experiments with identical splits and multiple seeds are required before making performance claims.
