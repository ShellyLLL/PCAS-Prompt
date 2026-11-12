# VSMR: Unified Visual-Semantic Decomposition and Prototype Recomposition for FSCIL

The default method on \`dev\` is VSMR. The unified implementation is developed from \`dev\` and keeps the pretrained ImageNet-21K ViT-B/16 frozen in every session.

## 0. Frozen-backbone preprocessing contract

VSMR reads the mean, standard deviation, interpolation mode, and crop percentage from the instantiated pretrained ViT checkpoint configuration. Base-session training retains random resized-crop and horizontal-flip augmentation, but uses the checkpoint's interpolation and normalization. Base-memory extraction, incremental support extraction, and evaluation use the corresponding deterministic resize/center-crop transform with the same checkpoint statistics. Dataset-specific CIFAR/ImageNet normalization is never substituted for the checkpoint's input statistics.

For the repository's ImageNet-21K ViT-B/16 configuration, the input mean/std and resize/crop policy follow that checkpoint's timm metadata. If a checkpoint does not expose valid RGB statistics, VSMR uses the explicit fallback mean/std of 0.5/0.5 and validates the resulting transform.

## 1. Architecture and information flow

### 1.1 Frozen visual backbone

A frozen ViT-B/16 produces normalized CLS and final Patch tokens:

- Global feature: \`g(x) in R^768\` from CLS and a trainable base-stage projection.
- Patch features: \`X(x) in R^(196 x 768)\` from the final 196 spatial tokens.

No text encoder, second backbone, expert routing, or incremental parameter expansion is used.

### 1.2 Shared Prompt visual decomposition

A shared set of \`M\` learnable visual queries attends softly to Patch tokens. Each query pools one latent visual component. The components are unordered; Prompt indices are not assigned human semantic labels.

The decomposer is trained only during the base session. The number of components is configurable through \`--vsmr_num_parts\`.

### 1.3 Cross-image component alignment

Entropic balanced OT with uniform marginals and log-domain Sinkhorn establishes permutation-invariant correspondences between component sets. OT is used for:

- same-class cross-image alignment;
- class prototype barycenters;
- leave-one-out local classification;
- query-to-reconstructed-class matching.

All uses share the same cosine cost and Sinkhorn implementation.

### 1.4 Base knowledge memory

After base training, every base class stores:

- one normalized global mean prototype;
- one local OT barycenter containing \`M\` visual components.

The flattened base component bank stores source-class ownership and component indices. The bank is immutable after the base session.

### 1.5 Pseudo-incremental training and learned transfer gate

The base-stage trainer constructs pseudo-incremental episodes from each class-balanced batch:

1. Randomly reserve a subset of batch classes as pseudo-novel classes.
2. Build a temporary source bank only from the remaining pseudo-base classes.
3. Sample K-shot support examples and disjoint query examples for each pseudo-novel class.
4. Build raw novel prototypes and OT-aligned support-dispersion statistics.
5. Retrieve at most one component from each source base class for every novel component.
6. Use a trainable ReliabilityGate MLP to estimate component-wise transfer reliability.
7. Classify pseudo-novel queries against both pseudo-base and recomposed pseudo-novel prototypes.

The gate receives the novel component, retrieved component, their product and absolute difference, retrieval similarity, top-source margin, normalized retrieval entropy, support dispersion, and log shot count. Its output is learned from pseudo-incremental global/local/joint classification losses. It is not a fixed similarity heuristic.

Top-r retrieval is source-class balanced: multiple highly similar components from the same base class cannot occupy several Top-r positions for one novel component.

### 1.6 Query-conditioned Q/K/V reconstruction

For each query and candidate class:

- stored class components are projected to reconstruction queries;
- the query's Patch tokens are projected to keys and values;
- cross-attention produces query-conditioned context;
- a residual connection and LayerNorm produce temporary reconstructed class components;
- balanced OT matches the query's decomposed components to the reconstructed components.

The query changes only the temporary representation used for its own score. It never changes or writes to persistent class memory.

### 1.7 Unified classifier

The final score uses a learned convex fusion:

`score = alpha * global_score + (1 - alpha) * local_score`

The scalar `alpha = sigmoid(a)` is trained by the explicit joint classification objective during the base session. Its initial value is set by `--vsmr_global_weight` and `--vsmr_local_weight_eval`; it is frozen before the first incremental session.

- Global score: cosine similarity between the query CLS feature and class global prototype.
- Local score: negative OT distance between query components and query-conditioned reconstructed class components.

All seen classes use the same scoring rule. No task ID is required.

## 2. Base-stage objective

The base objective is:

\[
L = L_g + w_l L_l + w_j L_{joint}
  + w_{OT} L_{OT} + w_{JS} L_{JS} + w_{div} L_{div}.
\]

- \`L_g\`: global prototype cross-entropy. It combines base-batch leave-one-out classification with pseudo-incremental query classification.
- \`L_l\`: local Q/K/V reconstruction + OT classification. It combines leave-one-out base-batch classification with pseudo-incremental query classification.
- \`L_joint\`: explicit cross-entropy on the fused global/local scores, including pseudo-incremental queries.
- \`L_OT\`: mean balanced-OT distance between distinct same-class images.
- \`L_JS\`: symmetric Jensen-Shannon alignment between global and local predictive distributions for distinct same-class images.
- \`L_div\`: overlap penalty between different Prompt-to-Patch attention distributions.

The frozen ViT receives no gradients. The shared decomposer, ReliabilityGate, QueryConditionedReconstructor, and global/local fusion coefficient are optimized in the base session. All learned modules are frozen before incremental learning.

## 3. Incremental update protocol

For each novel class:

1. Extract its K-shot global features, Patch tokens, and visual components using the frozen model.
2. Build the raw global mean prototype and local OT barycenter.
3. OT-align support components to the novel barycenter to estimate per-component support dispersion.
4. Retrieve source-class-balanced Top-r base components from the immutable base bank.
5. Apply the learned frozen ReliabilityGate and compose each novel component with its retrieved base knowledge.
6. Append the novel global and recomposed local prototypes.

The update is append-only:

- ViT, decomposer, gate, reconstructor, and global/local fusion coefficient remain frozen.
- Base global/local memories never change.
- Historical novel prototypes never change.
- The base component bank remains the only transfer source.
- Query inference never updates persistent memory.

## 4. Run CUB-200

\`\`\`bash
export FSCIL_DATA_ROOT=/data/lx/fscil/data
export FSCIL_INDEX_LIST_ROOT="$FSCIL_DATA_ROOT/index_list"

python train.py --method vsmr -project base -dataset cub200 -vit \\
  -dataroot "$FSCIL_DATA_ROOT" --index_list_root "$FSCIL_INDEX_LIST_ROOT" \\
  -gpu 0 -seed 1 -epochs_base 50 -num_workers 4 -test_batch_size 64 \\
  --vsmr_num_parts 6 --vsmr_classes_per_batch 16 \\
  --vsmr_samples_per_class 4 --vsmr_episode_novel_fraction 0.25 \\
  --vsmr_episode_max_shot 3 --vsmr_joint_weight 1.0
\`\`\`

## 5. Main configuration

| Setting | Default |
|---|---:|
| Shared visual components | 6 |
| Base batch classes / images per class | 16 / 4 |
| Pseudo-novel class fraction | 0.25 |
| Pseudo-incremental maximum shots | 3 |
| Gate hidden dimension | 384 |
| Prompt/projection learning rate | 2e-4 |
| OT entropy epsilon / Sinkhorn iterations | 0.05 / 30 |
| OT barycenter iterations | 5 |
| Base component retrieval Top-r | 5 source classes |
| Maximum transfer strength | 0.5 |
| Initial global/local fusion weights | 0.5 / 0.5 (learned in base session) |
| Evaluation batch size | 8 |

These are reproducible starting settings, not optimality claims. Hyperparameters must be selected using a base-class validation protocol, not the final incremental test labels.

## 6. Diagnostics and acceptance

Each run writes \`metrics.jsonl\`, \`vsmr_base.pth\`, and \`vsmr_final.pth\` under \`checkpoint/<out>/<dataset>/VSMR/<timestamp>/\`.

Logs include:

- base global/local/joint losses and pseudo-incremental losses;
- OT, JS, and Prompt-overlap losses;
- per-session sample and class-average accuracy;
- base and novel accuracy;
- novel-to-base error;
- learned transfer alpha/reliability and source-class/component IDs;
- transfer gain over raw novel prototypes;
- Q/K/V + OT gain over fixed-index matching. The fixed-index ablation uses the exact same query-conditioned Q/K/V reconstruction and changes only the final matching rule, so the comparison isolates OT alignment rather than removing reconstruction;
- Average Accuracy (AA) and Performance Drop (PD).

The controlled diagnostics use test labels only to calculate metrics. They are not used for gradient updates or hyperparameter selection. A positive transfer or reconstruction gain is not guaranteed and must be reported across datasets and random seeds.

## 7. Tests

\`\`\`bash
python -m py_compile train.py models/base/vsmr_modules.py models/base/vsmr_losses.py models/base/vsmr_resynthesis.py models/base/vsmr_trainer.py
python -m unittest discover -s tests -p 'test_vsmr.py' -v
\`\`\`

The VSMR tests cover frozen-backbone and incremental-freeze contracts, soft Prompt gradients, balanced OT marginals and permutation invariance, OT barycenters, leave-one-out classification, source-class-balanced retrieval, learned gate gradients, support dispersion, query Patch-conditioned Q/K/V reconstruction, append-only memory, and pseudo-incremental gate/reconstructor training.
