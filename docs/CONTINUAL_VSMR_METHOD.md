# Continual VSMR: Method Definition

## 1. Problem formulation

Let the FSCIL stream contain a base session \(t=0\) and incremental sessions
\(t=1,\ldots,T\). The base label set is \(\mathcal C_0\); session \(t\) introduces
a disjoint set \(\mathcal C_t\) with \(K\) labelled support images per class.
After session \(t\), the seen classes are
\(\bar{\mathcal C}_t=\bigcup_{s=0}^{t}\mathcal C_s\).

The method freezes a pretrained ViT throughout the stream and uses a fixed-size
shared visual Prompt as the only parameter block updated in incremental
sessions. It does not use pseudo-incremental episodes, a Prompt pool, experts,
or a learned cross-class transfer gate.

## 2. Frozen feature extractor and feature memory

For an image \(x_i\), the frozen ViT \(F_\phi\) returns a CLS token and final
Patch tokens:

\[
(c_i,X_i)=F_\phi(x_i),\quad
c_i\in\mathbb R^d,\quad X_i\in\mathbb R^{N\times d},
\]

where \(\phi\) is never optimized, \(d=768\), and \(N=196\) for ViT-B/16 at
224x224 resolution.

The feature memory after session \(t\) is

\[
\mathcal M_t=
\{(c_i,X_i,g_i,y_i,\mathrm{id}_i,s_i)\}_{i=1}^{n_t}.
\]

Here \(g_i=\operatorname{norm}(W_g c_i)\) is the global feature produced by
the Base-trained global projection, \(y_i\) is the global class ID,
\(\mathrm{id}_i\) is a sample identifier, and \(s_i\) is its source session.
For each base class, the memory stores a deterministic class-balanced budget
of \(m\ge2\) training exemplars. For every incremental class it stores all
\(K\) labelled support examples. No raw image pixels or test examples are
stored. CLS and Patch features are cached in FP16 on CPU; the global features
are cached as a derived representation. The global projection \(W_g\) is
frozen after Base, so cached global features remain valid.

This is feature replay, not raw-image replay and not a no-replay protocol.

## 3. Prompt-conditioned visual component decomposition

The Base-trained decomposer contains projections
\(W_Q,W_K,W_V,W_g\) and a fixed number \(M\) of shared Prompt vectors
\(P_t\in\mathbb R^{M\times d}\). For a cached image:

\[
Q_t=W_QP_t,\quad K_i=W_KX_i,\quad V_i=W_VX_i,
\]

\[
A_i^t=
\operatorname{softmax}_{N}
\left(\frac{Q_tK_i^\top}{\sqrt d\,\tau}\right)
\in\mathbb R^{M\times N},
\]

\[
Z_i^t=
\operatorname{norm}(A_i^tV_i)
\in\mathbb R^{M\times d}.
\]

The \(M\) components form an unordered set. Component indices are not assigned
human-readable part labels. In incremental sessions only \(P_t\) is optimized;
\(W_Q,W_K,W_V,W_g\), the frozen ViT, the query reconstructor, and the fusion
coefficient are frozen.

## 4. Class prototypes and permutation-invariant matching

For each seen class \(c\), the global prototype is

\[
\mu_c^g=
\operatorname{norm}\left(
\frac{1}{|\mathcal M_c|}
\sum_{i:y_i=c}g_i
\right).
\]

The local prototype is an entropic-OT barycenter of the current Prompt
components:

\[
\Pi_c^t=
\operatorname{Barycenter}_{\epsilon}
\{Z_i^t:y_i=c\}
\in\mathbb R^{M\times d}.
\]

For two component sets \(U,V\in\mathbb R^{M\times d}\), define
\(C_{ab}=1-\langle\operatorname{norm}(u_a),
\operatorname{norm}(v_b)\rangle\). Let \(T_\epsilon(U,V)\) be the balanced
entropic OT coupling with uniform marginals \(1/M\). The matching distance used
for scoring and consistency is

\[
D_\epsilon(U,V)=\langle T_\epsilon(U,V),C(U,V)\rangle.
\]

The entropic term regularizes the coupling; the reported distance is the
transported cosine cost.

For query \(x\), the query-conditioned reconstructor uses class components as
queries and the query Patch tokens as keys and values:

\[
\widetilde\Pi_c^t(x)=
\operatorname{norm}\left[
\operatorname{LN}\left(
W_q^r\Pi_c^t+
\operatorname{Attn}(W_q^r\Pi_c^t,W_k^rX_x,W_v^rX_x)
\right)\right].
\]

The global, local and fused scores are

\[
s_g(x,c)=\gamma\langle g_x,\mu_c^g\rangle,
\]

\[
s_l^t(x,c)=-\gamma
D_\epsilon(Z_x^t,\widetilde\Pi_c^t(x)),
\]

\[
s^t(x,c)=\alpha s_g(x,c)+(1-\alpha)s_l^t(x,c),
\]

where \(\alpha\) is learned during Base and frozen afterwards.

## 5. Leave-one-out (LOO) training prototypes

A memory sample must not contribute to its own class prototype. For a query
sample \(i\) with class \(c=y_i\), define

\[
\Pi_{c,-i}^t=
\operatorname{Barycenter}_{\epsilon}
\{Z_j^t:y_j=c,\ j\ne i\},
\]

\[
\mu_{c,-i}^g=
\operatorname{norm}\left(
\frac{1}{|\mathcal M_c|-1}
\sum_{j:y_j=c,j\ne i}g_j
\right).
\]

The candidate prototype bank for query \(i\) uses these LOO prototypes for
class \(c\), and full-memory prototypes for every other class. Each memory
class therefore requires at least two examples. The implementation checks
this condition rather than silently including the query in its own prototype.

## 6. Base-session objective

Base training uses real base images and class-balanced batches. It does not
simulate pseudo-novel classes. Let \(z_i^g,z_i^l,z_i^j\) denote the global,
local and fused LOO logits for a batch. The objective is

\[
\mathcal L_{\mathrm{base}}=
\mathcal L_g+
\lambda_l\mathcal L_l+
\lambda_j\mathcal L_j+
\lambda_{\mathrm{ot}}\mathcal L_{\mathrm{ot}}+
\lambda_{\mathrm{js}}\mathcal L_{\mathrm{js}}+
\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}.
\]

- \(\mathcal L_g=\operatorname{CE}(z^g,y)\): global LOO classification.
- \(\mathcal L_l=\operatorname{CE}(z^l,y)\): query-conditioned local LOO
  classification using the same Q/K/V reconstruction and OT scorer as
  inference.
- \(\mathcal L_j=\operatorname{CE}(\alpha z^g+(1-\alpha)z^l,y)\): fused LOO
  classification.
- \(\mathcal L_{\mathrm{ot}}\): mean balanced-OT distance between distinct
  same-class component sets in the batch.
- \(\mathcal L_{\mathrm{js}}\): symmetric Jensen-Shannon alignment between
  global and local predictive distributions for distinct same-class images.
- \(\mathcal L_{\mathrm{div}}\): mean pairwise overlap of different
  Prompt-to-Patch attention distributions.

After Base training, \(P_0\) is saved. The backbone, decomposer projections,
global projection, reconstructor and fusion coefficient are frozen. Only the
shared Prompt matrix can subsequently receive gradients.

## 7. Incremental-session objective

At the start of session \(t\), append the real \(K\)-shot support features to
\(\mathcal M_{t-1}\). Copy \(P_{t-1}\) as the previous-session teacher Prompt.
Keep \(P_0\) as a permanent Base reference. The teacher components for old
memory are computed once using \(P_{t-1}\); Base-reference components are
computed using \(P_0\).

For every optimization step, the student Prompt \(P_t\) re-decomposes the
cached Patch features of all seen classes. Current class prototypes are
recomputed from these current components. A class-balanced query minibatch is
then classified against the full set of seen classes using LOO candidate
prototypes.

The incremental objective is

\[
\begin{aligned}
\mathcal L_t={}&
\mathcal L_{\mathrm{joint}}
+\lambda_{\mathrm{local}}\mathcal L_{\mathrm{local}}
+\lambda_{\mathrm{prev}}\mathcal L_{\mathrm{prev}}
+\lambda_{\mathrm{base}}\mathcal L_{\mathrm{base-cons}}\\
&+\lambda_{\mathrm{kd}}\mathcal L_{\mathrm{kd}}
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
+\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}.
\end{aligned}
\]

The terms have distinct roles:

1. **All-seen-class classification**
   \[
   \mathcal L_{\mathrm{joint}}=
   \operatorname{CE}(s^t(x_i,\bar{\mathcal C}_t),y_i).
   \]
   Queries are drawn from the feature memory, including old exemplars and
   current K-shot support. This trains the new Prompt against all seen
   classes and includes old-vs-new competition.

2. **Local classification**
   \[
   \mathcal L_{\mathrm{local}}=
   \operatorname{CE}(s_l^t(x_i,\bar{\mathcal C}_t),y_i).
   \]
   This ensures that the trainable Prompt receives direct supervision through
   the local component pathway. The global pathway is frozen and acts as a
   stable score reference.

3. **Previous-session component consistency**
   \[
   \mathcal L_{\mathrm{prev}}=
   \frac{1}{|\mathcal M_{1:t-1}|}
   \sum_{i\in\mathcal M_{1:t-1}}
   D_\epsilon(Z_i^t,Z_i^{t-1}).
   \]
   This applies to previously learned incremental classes. Base classes are
   handled separately by the permanent Base reference, avoiding duplicate
   weighting in the first incremental session.

4. **Permanent Base component consistency**
   \[
   \mathcal L_{\mathrm{base-cons}}=
   \frac{1}{|\mathcal M_0|}
   \sum_{i\in\mathcal M_0}D_\epsilon(Z_i^t,Z_i^0).
   \]
   This limits accumulated drift of Base representations across sessions.

5. **Old-class logit distillation**
   \[
   \mathcal L_{\mathrm{kd}}=
   T_d^2\,
   \operatorname{KL}\left(
   \operatorname{softmax}(s^{t-1}_{old}/T_d)
   \,\middle\|\,
   \operatorname{softmax}(s^t_{old}/T_d)
   \right).
   \]
   Teacher logits are produced by the previous Prompt and previous prototypes
   on old-memory queries, with LOO prototypes. Student logits are restricted
   to old-class columns for this term. The all-class classification term
   separately protects old-vs-new decision boundaries.

6. **Prompt trust-region anchor**
   \[
   \mathcal L_{\mathrm{anchor}}=
   \frac{1}{Md}\|P_t-P_{t-1}\|_F^2.
   \]
   This limits a single-session parameter jump. It complements, rather than
   replaces, feature-space consistency.

7. **Component diversity**
   \[
   \mathcal L_{\mathrm{div}}=
   \frac{1}{nM(M-1)}
   \sum_i\sum_{a\ne b}
   \langle A_{i,a}^t,A_{i,b}^t\rangle.
   \]
   This discourages different Prompt queries from collapsing onto the same
   Patch distribution.

Only \(P_t\) is optimized. The query reconstructor and fusion coefficient
remain fixed. No test image or test label enters the memory or any loss.

## 8. Training algorithms

### Algorithm 1: Base training

1. Initialize the pretrained ViT and freeze it.
2. Train \(P,W_Q,W_K,W_V,W_g\), the query reconstructor and \(\alpha\) on
   real Base images with \(\mathcal L_{\mathrm{base}}\).
3. Select \(m\ge2\) deterministic training exemplars per Base class.
4. Extract and cache their frozen CLS and Patch features; derive and cache
   global features.
5. Save \(P_0\), Base-reference components, and Base prototypes.
6. Freeze every parameter except the shared Prompt for later sessions.

### Algorithm 2: Session \(t\) continual Prompt update

1. Receive only the labelled \(K\)-shot support set for the new classes.
2. Extract frozen CLS/Patch features and append them to the feature memory.
3. Snapshot \(P_{t-1}\); compute previous-teacher components and old-class
   teacher logits using LOO prototypes. Keep the permanent \(P_0\) Base
   components.
4. Enable gradients only for \(P_t\leftarrow P_{t-1}\).
5. For each class-balanced memory minibatch:
   - re-decompose all cached Patch features with current \(P_t\);
   - recompute all seen-class OT barycenters;
   - construct query-specific LOO candidate prototypes;
   - compute global, local and fused logits against all seen classes;
   - compute classification, consistency, distillation, anchor and diversity
     losses;
   - update only \(P_t\).
6. Recompute and store every seen-class local prototype using the final
   Prompt. Do not append stale local prototypes.
7. Evaluate on the test set of all seen classes; save the session state.
8. Continue to session \(t+1\) from the updated \(P_t\).

## 9. Module data flow

**Training / memory path**

Image -> frozen ViT -> CLS + Patch -> CPU feature memory.

Feature memory Patch -> frozen \(W_K,W_V\) -> shared Prompt queries
\(W_QP_t\) -> soft Prompt-to-Patch attention -> unordered components
\(Z_i^t\) -> class-wise OT barycenter -> current local prototypes.

**Query classification path**

Query image -> frozen ViT -> global feature + Patch + Prompt components.

- Global feature -> cosine score against cached global class prototypes.
- Query Patch + class local prototypes -> frozen Q/K/V reconstructor ->
  temporary query-conditioned class components -> balanced OT score.
- Global and local scores -> frozen convex fusion -> all-seen-class logits.

**Incremental optimization path**

Current Prompt -> all-seen feature re-decomposition -> LOO prototypes ->
all-seen classification, old-logit distillation, previous/base component
consistency, Prompt anchor and attention diversity -> gradient update of
shared Prompt only.

## 10. Protocol and reporting requirements

- Report the Base exemplar budget \(m\), incremental shot \(K\), cache precision,
  and total feature-cache size.
- Report that the method uses frozen-feature replay; do not describe it as
  raw-image-free/no-replay without qualification.
- Use the same session splits, frozen ViT checkpoint, preprocessing and
  evaluation protocol for the VSMR and Continual VSMR comparison.
- Select loss weights and epoch counts using Base-only validation or a
  predeclared protocol, never final incremental test labels.
- Report per-session class-average accuracy, sample accuracy, Base/novel
  accuracy, AA, PD, memory size, Prompt update norms, component drift, and
  old-class distillation loss.
- Run multiple seeds. The architecture is a testable hypothesis, not a
  guarantee of accuracy improvement.
