# Prompt-Slot-OT for Few-Shot Class-Incremental Learning

This branch contains a unified FSCIL method built around **frozen ViT Patch features, Prompt-guided Slot Attention, permutation-invariant local-set matching, and reliability-controlled local prototype calibration**.

The method is intentionally kept compact. It does not use the legacy PCAS component bank, LSP image-space prompt pool, CODA prompt expansion, MoE, text encoders, replay, or incremental prompt tuning as part of the new method. Legacy implementations remain available as separate baselines; run the new method explicitly with \`--method prompt_slot_ot\`.

## Method

1. **Frozen visual backbone.** A pretrained ViT-B/16 provides a global CLS representation and a set of Patch features. The backbone remains frozen.
2. **Prompt-Slot decomposition.** Shared learned slot queries attend to the Patch set. Competitive Patch assignment and GRU slot updates produce an image-conditioned, unordered set of local visual descriptors. Slot indices are not treated as semantic labels.
3. **Cross-image set matching.** Local descriptor sets are compared with unbalanced entropic optimal transport. Unmatched mass is allowed, so occluded, absent, or unreliable regions are not forced into a correspondence.
4. **Cross-class recomposition.** Base classes retain a compact set of local visual descriptors. A novel class retrieves candidate base memories using global similarity, then receives only confidence-weighted local corrections supported by OT matching.
5. **Joint classification.** All seen classes are evaluated with a common combination of global cosine distance and local-set OT distance.

## Base and incremental protocol

- Base training uses class-balanced episodes with disjoint support and query examples. Only the Slot Attention module is optimized; the ViT is frozen.
- Episode classification uses the same global-plus-local distance used at inference.
- After base training, deterministic base features are compressed into per-class local descriptor memories and global class prototypes.
- In each real FSCIL session, exactly the configured K-shot support examples are used to construct each novel class prototype.
- Novel local memories may be calibrated from the **fixed base-class memory only**. Calibrated novel memories are not reused as transfer sources, preventing recursive error propagation.
- ViT and Slot Attention are frozen during all incremental sessions. Existing class prototypes are append-only and never modified.

## Run

The existing dataset/session protocol is retained. Example for CUB-200:

\`\`\`bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

python train.py --method prompt_slot_ot -project base -dataset cub200 -vit \\
  -dataroot "$FSCIL_DATA_ROOT" -gpu 0 -seed 1 \\
  -epochs_base 50 -num_workers 4 -test_batch_size 64
\`\`\`

The same method entry supports the repository's existing CIFAR-100, MiniImageNet, FGVC-Aircraft, and iNF200 loaders, subject to their original split files and data layouts.

### Main settings

| Setting | Default |
|---|---:|
| Slot count | 6 |
| Slot dimension | 256 |
| Slot update iterations | 2 |
| Classes per base episode | 12 |
| Support / query per episode class | 2 / 2 |
| Local memory descriptors per class | 6 |
| Slot learning rate | 2e-4 |
| Maximum novel-memory calibration | 0.25 |
| OT entropy regularization | 0.05 |
| OT marginal relaxation | 0.10 |
| Global/local classification weight | 0.5 / 0.5 |

These are starting values, not claims of optimality. Select settings using base-class validation only.

## Implementation

- \`models/base/slot_ot_modules.py\`: frozen ViT and Prompt-Slot Attention.
- \`models/base/slot_ot_losses.py\`: unbalanced Sinkhorn matching, local-memory compression, calibration, and joint classifier.
- \`models/base/slot_ot_trainer.py\`: base episodic training, memory construction, real incremental sessions, evaluation, and checkpoints.
- \`tests/test_slot_ot.py\`: shape, gradient, permutation-invariance, OT, calibration, and frozen-backbone contract tests.

## Evaluation and limitations

The trainer reports session sample accuracy, session class-average accuracy, base/novel accuracy, harmonic mean, novel-to-base error, task-wise accuracy, AA, and PD. AA is the mean class-average accuracy over evaluated sessions. PD is base-session class-average accuracy minus final-session class-average accuracy.

The implementation is not evidence of an accuracy improvement until it has been run on the full FSCIL protocols with multiple seeds. In particular, verify that local-set matching improves novel-class accuracy without increasing base-class degradation. Report compute cost and compare against the repository's unchanged baselines under identical data splits and seeds.
