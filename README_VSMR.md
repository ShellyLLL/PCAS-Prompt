# VSMR for Few-Shot Class-Incremental Learning

The default method on \`dev\` is **VSMR (Visual Semantic Decomposition and Prototype Reconstruction)**. It implements the agreed visual-only FSCIL architecture with a frozen ImageNet-21K ViT-B/16.

## Architecture

1. **Frozen ViT-B/16:** one frozen visual backbone returns CLS and final Patch features. No text encoder, second backbone, expert routing, or incremental parameter expansion is used.
2. **Shared soft-prompt decomposition:** a fixed set of learnable visual prompts attends continuously to Patch features using softmax. Each prompt pools one latent local component. Prompt indices are not treated as semantic labels.
3. **Cross-image component matching:** entropic optimal transport (OT) with uniform marginals establishes permutation-invariant correspondences between unordered component sets.
4. **Unified base training:** leave-one-out global prototype classification, leave-one-out local OT classification, same-class cross-image OT alignment, symmetric global/local cross-image semantic alignment (JS), and attention-overlap regularization are optimized jointly.
5. **Base knowledge memory:** each base class stores a global mean prototype and an OT component-set barycenter. The flattened base component bank is immutable.
6. **Novel prototype composition:** each novel component independently retrieves top-r base components. Softmax retrieval weights, mean similarity, and source dispersion determine a bounded reliability gate. No whole base-class prototype is transferred.
7. **Query-conditioned reconstruction:** a query-to-class component OT plan supplies only the assignment weights. Reconstructed component values are weighted combinations of the candidate class's stored components; query values never enter prototype content.
8. **Unified classifier:** global cosine similarity and query-conditioned local OT reconstruction scores classify all seen classes. Incremental sessions append prototypes only; the backbone, prompts, and old memories remain frozen.

## Objective

\`L = L_global + w_local L_local + w_align L_OT + w_CISA L_JS + w_overlap L_overlap\`

- \`L_global\`: cross-entropy over leave-one-out global class prototypes.
- \`L_local\`: cross-entropy over leave-one-out local OT distances.
- \`L_OT\`: mean OT distance between distinct images of the same class.
- \`L_JS\`: symmetric Jensen-Shannon alignment of global/local predictive distributions for distinct same-class images.
- \`L_overlap\`: overlap penalty between different prompt-to-Patch attention distributions.

The ViT backbone is frozen. Only the shared decomposition prompts and their feature projections are optimized.

## Run CUB-200

\`\`\`bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

python train.py --method vsmr -project base -dataset cub200 -vit \\
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \\
  -gpu 0 -seed 1 -epochs_base 50 -num_workers 4 -test_batch_size 64
\`\`\`

## Key settings

| Setting | Default |
|---|---:|
| Shared visual components | 6 |
| Base batch classes / images per class | 16 / 4 |
| Prompt/projection learning rate | 2e-4 |
| OT entropy epsilon / Sinkhorn iterations | 0.05 / 30 |
| OT barycenter iterations | 5 |
| Base component retrieval top-r | 5 |
| Maximum transfer strength | 0.5 |
| Global/local inference weights | 0.5 / 0.5 |
| Evaluation batch size | 8 |

These are reproducible starting settings, not optimality claims. Tune only on a base-class validation split.

## Diagnostics and acceptance

Each run writes \`metrics.jsonl\`, \`vsmr_base.pth\`, and \`vsmr_final.pth\` under \`checkpoint/<out>/<dataset>/VSMR/<timestamp>/\`.

The log includes per-epoch total/global/local/alignment/CISA/overlap losses, global/local/joint batch accuracy, mean same-class OT distance, learning rate, and per-session sample/macro accuracy, base/novel accuracy, novel-to-base error, transfer reliability/strength, and AA/PD.

Two critical test-set diagnostics are reported without using test labels for model updates:
- \`local_transfer_gain\`: local classification accuracy with transferred novel component prototypes minus accuracy with raw novel prototypes.
- \`query_ot_vs_fixed_index_gain\`: query-conditioned OT local accuracy minus fixed component-index local accuracy.

These are evaluation diagnostics, not training signals or hyperparameter-selection criteria. The test labels are used only to calculate metrics. A positive gain is not guaranteed; results must be reported across the prescribed datasets and multiple seeds.

## Tests

\`\`\`bash
python -m py_compile train.py models/base/vsmr_modules.py models/base/vsmr_losses.py models/base/vsmr_resynthesis.py models/base/vsmr_trainer.py
python -m unittest discover -s tests -p 'test_vsmr.py' -v
\`\`\`
