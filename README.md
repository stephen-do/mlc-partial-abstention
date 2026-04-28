# Understanding: Multilabel Classification with Partial Abstention
### Bayes-Optimal Prediction under Label Independence
> Nguyen & Hüllermeier — JAIR 2021 | A complete conceptual guide

---

## How to read this document

This README is structured as a **logical story**, not a reference manual.
Each section answers: *"Why does this exist? What problem does it solve?"*

```
Problem Setup
    ↓
Why Standard MLC is not enough
    ↓
The Framework: Generalized Loss + Abstention
    ↓
The Core Algorithm: Bayes-Optimal Prediction (BOP)
    ↓
The Key Assumption: Conditional Label Independence (CLI)
    ↓
Why CLI matters: Decomposable vs Non-decomposable Loss
    ↓
How BOP is computed for each loss type
    ↓
Classifiers used in experiments
    ↓
What the experiments actually test
```

---

## Part 1 — The Problem: What is Multilabel Classification?

### Standard classification
In regular classification, each input belongs to exactly **one** class.
A bank transaction is either "fraud" or "not fraud."

### Multilabel Classification (MLC)
In MLC, each input can belong to **multiple classes simultaneously**.
A single bank transaction can be flagged as:
- Money laundering ✓
- Tax evasion ✓
- Card fraud ✗
- Terrorism financing ✓

Formally, the model outputs a binary vector:
```
y = (y₁, y₂, ..., yₖ) ∈ {0, 1}ᴷ
```
where `K` is the total number of labels and `yₖ = 1` means label k is present.

### The problem with standard MLC

Standard MLC **forces** the model to predict all K labels, even when uncertain.

**Example:**
```
Transaction features: wire transfer $500k, 3am, new account, offshore
p(money laundering)      = 0.92  → confident → predict 1
p(tax evasion)           = 0.51  → barely above threshold → predict 1
p(terrorism financing)   = 0.48  → barely below threshold → predict 0
```

The model is nearly certain about money laundering, but almost completely uncertain about tax evasion and terrorism financing. Yet it's forced to give a hard yes/no on all three.

**In high-stakes domains** (banking, medical, legal), a wrong prediction can be far worse than admitting uncertainty. The paper's key question is: *"Can we let the model say 'I don't know' on uncertain labels?"*

---

## Part 2 — The Solution: Partial Abstention

Instead of predicting only `{0, 1}`, the model can now output three values:

```
ŷₖ = 1     →  label k is present (confident)
ŷₖ = 0     →  label k is absent  (confident)
ŷₖ = ⊥     →  abstain: "I'm not sure about this label"
```

**The same banking example with abstention:**
```
ŷ = (1, ⊥, 0, ⊥)
      ↑   ↑  ↑   ↑
      |   |  |   └── terrorism financing: abstain → human review
      |   |  └─── card fraud: predict 0 (confident)
      |   └─────── tax evasion: abstain → human review
      └─────────── money laundering: predict 1 (confident)
```

The model focuses its predictions on what it knows, and escalates the rest.

**Two important sets are defined:**
```
D(ŷ) = {k | ŷₖ ∈ {0,1}}   ← decided labels (model commits)
A(ŷ) = {k | ŷₖ = ⊥}       ← abstained labels (model refuses)

Always true: D(ŷ) ∪ A(ŷ) = {1, ..., K}
```

---

## Part 3 — The Framework: Generalized Loss Function

### Why do we need a new loss function?

When the model abstains, we need to measure performance differently. The standard MLC loss only compares predictions to ground truth labels — it doesn't handle `⊥`.

The paper proposes a **generalized loss** with two components:

```
L(y, ŷ) = ℓ(y_D, ŷ_D)  +  g(|A(ŷ)|)
           ─────────────    ──────────────
           loss on the      penalty for
           decided labels   abstaining on |A| labels
```

**Component 1 — `ℓ(y_D, ŷ_D)`:**
The standard MLC loss applied only to labels the model actually predicted.
If the model abstained on labels {2, 4}, then `y_D` and `ŷ_D` only cover labels {1, 3, 5, ...}.

**Component 2 — `g(|A(ŷ)|)`:**
A penalty for abstaining. Without this, the model would always abstain on everything (no risk, no usefulness). The penalty forces a trade-off.

**When no abstention:** `L(y, ŷ) = ℓ(y, ŷ)` — reduces to the original loss. ✓

### The two penalty functions studied

**SEP (Separate) — linear penalty:**
```
g₁(a) = a · c
```
Each abstained label costs exactly `c`. Abstaining on 3 labels costs `3c`.
Use when: each label has equal importance, simplicity matters.

**PAR (Parsimonious) — concave penalty:**
```
g₂(a) = (a · K · c) / (K + a)
```
The first abstention is expensive, but additional abstentions cost less and less.
Use when: batch abstentions are acceptable (e.g., recommendation systems).

**Visual intuition:**
```
Cost
 ↑
 │         SEP (straight line)
 │       /
 │     /   PAR (curves and flattens)
 │   / ___─────────────
 │ /__/
 └─────────────────────→ Number of abstentions
 0                      K
```

### Three desirable properties of a good generalized loss

**1. Monotonicity** — the loss should only increase when:
- A correct prediction becomes incorrect (bad)
- A correct prediction becomes abstention (worse but acceptable)
- An abstention becomes incorrect (even worse)

Formally: correct > abstention > incorrect (in terms of preference).

**2. Uncertainty-alignment** — abstentions should happen on the most uncertain labels.
If the model abstains on label i but predicts label j, then label i must be more uncertain than label j.
```
uₖ = 2·min(pₖ, 1-pₖ)       (uncertainty measure, ranges 0 to 1)
```
A prediction is uncertainty-aligned if: for all i ∈ A(ŷ), j ∈ D(ŷ): uᵢ ≥ uⱼ

**3. Semi-uncertainty-alignment** — a relaxed version.
After sorting labels by their marginal probability pₖ, the abstained labels form a contiguous "middle block":
```
Decision set: D(ŷ) = ⟪l, r⟫ = {1,...,l} ∪ {r,...,K}

Sorted by pₖ descending:
[high pₖ → predict 1] [middle pₖ → abstain ⊥] [low pₖ → predict 0]
        l                     gap                       r
```
This means: the model is confident at both extremes (very likely or very unlikely), and abstains in the uncertain middle.

---

## Part 4 — Bayes-Optimal Prediction (BOP)

### What is BOP?

BOP is the prediction `ŷ*` that **minimizes the expected loss**:

```
ŷ* = argmin_{ŷ ∈ Ω*} E[L(y, ŷ)]
   = argmin_{ŷ ∈ Ω*} Σ_{y∈Y} L(y, ŷ) · p(y|x)
```

**In plain English:** "Try every possible prediction ŷ. For each one, compute the average loss across all possible ground-truth outcomes weighted by their probability. Pick the ŷ with the smallest average loss."

**Why this is the right objective:**
The model doesn't know the true y at prediction time — it only knows p(y|x). BOP finds the prediction that is optimal *in expectation*, given all the uncertainty.

### The explosion problem

The prediction space with abstention is `Ω* = {0, ⊥, 1}ᴷ`.

```
K = 10  →  3¹⁰ =      59,049 candidates
K = 20  →  3²⁰ =  3,486,784,401 candidates
K = 174 →  3¹⁷⁴ ≈ 10⁸³ candidates   (CAL500 dataset)
```

Brute-force is completely infeasible. The paper's main theoretical contribution is showing that **BOP always has a specific structure** for each loss type — which allows efficient computation.

### The two BOP structures

**Structure 1: Uncertainty-aligned BOP**
Sort labels by uncertainty `uₖ`. Abstain on the most uncertain ones.
```
Algorithm:
1. Compute uₖ = 2·min(pₖ, 1-pₖ) for each label k
2. Sort labels: u_{π(1)} ≤ u_{π(2)} ≤ ... ≤ u_{π(K)}
3. Find optimal cutoff d*:
   d* = argmin_{d=0..K} [E[ℓ(y_D, ŷ_D)] + g(K-d)]
4. Predict on d* most certain labels, abstain on the rest
```

**Structure 2: Semi-uncertainty-aligned BOP**
After sorting by pₖ, predict on both ends, abstain in the middle.
```
Algorithm:
1. Sort labels by pₖ descending
2. Find optimal (l, r) pair:
   D*(ŷ) = ⟪l*, r*⟫ = {1,...,l*} ∪ {r*,...,K}
3. Only O(K²) pairs to check instead of 3ᴷ
```

**Why this reduces the search space so dramatically:**
Instead of checking all `3ᴷ` combinations, we only need to find the best `d` (for uncertainty-aligned) or the best `(l, r)` pair (for semi-uncertainty-aligned). This brings complexity down from exponential to polynomial.

---

## Part 5 — Conditional Label Independence (CLI)

### What is independence?

Two events A and B are independent if knowing A tells you nothing new about B:
```
P(B | A) = P(B)    ←→    P(A ∩ B) = P(A) × P(B)
```

### Unconditional vs Conditional independence

There are two very different notions:

**Unconditional label independence:**
```
P(Y₁, Y₂) = P(Y₁) × P(Y₂)
```
Looking at the entire dataset, ignoring features. This is almost always **false** in practice because labels often co-occur for structural reasons (e.g., "money laundering" and "tax evasion" both tend to appear together in the dataset overall).

**Conditional Label Independence (CLI):**
```
P(Y₁, Y₂, ..., Yₖ | x) = P(Y₁|x) × P(Y₂|x) × ... × P(Yₖ|x)
```
The labels are independent **given the specific transaction features x**. This is a weaker, more realistic assumption.

**Full CLI formula:**
```
p(y | x) = ∏ₖ  pₖ^{yₖ} · (1-pₖ)^{1-yₖ}
```
where `pₖ = p(Yₖ=1 | x)` is the marginal probability of label k.

### Why CLI is plausible — the causal intuition

Consider why "money laundering" and "tax evasion" co-occur in a dataset:

**Wrong mental model (CLI violation):**
```
Money laundering = 1  →  causes  →  Tax evasion = 1
```
If this were true, knowing Y₁ would always help predict Y₂, even after knowing x.

**Correct mental model (CLI holds):**
```
Customer profile x (professional tax criminal)
          ↙                    ↘
Money laundering = 1      Tax evasion = 1
```
The features x already encode the full profile. Once x is known, the two labels carry no additional information about each other. They are both consequences of x, not of each other.

**Concrete example:**
```
Without knowing x (whole dataset):
  P(Tax evasion=1 | Money laundering=1) = 80%   ≠   P(Tax evasion=1) = 50%
  → Labels are DEPENDENT globally

After knowing x = "offshore wire transfer at 3am, new account":
  P(Tax evasion=1 | x AND Money laundering=1) ≈ P(Tax evasion=1 | x) = 70%
  → x already explains the co-occurrence. Labels become INDEPENDENT given x.
```

### Why CLI is critical for the paper

CLI allows a massive simplification when computing `E[L(y, ŷ)]`:

**Without CLI:** Need to estimate `p(y₁, y₂, ..., yₖ | x)` — a joint distribution over `2ᴷ` outcomes.
```
K = 20  →  need to estimate 1,048,576 probabilities
```
This is impossible to learn reliably from finite data.

**With CLI:** Only need K marginal probabilities `p₁, p₂, ..., pₖ`.
```
K = 20  →  need to estimate 20 probabilities
```
And the joint distribution can be reconstructed exactly by multiplication.

**Example computation with CLI (K=3):**
```
p₁ = 0.85 (money laundering), p₂ = 0.52 (tax evasion), p₃ = 0.10 (card fraud)

p(1,1,0|x) = 0.85 × 0.52 × (1-0.10)
           = 0.85 × 0.52 × 0.90
           = 0.398
```
Without CLI, `p(1,1,0|x)` would have to be estimated directly from data — very unreliable.

---

## Part 6 — Decomposable vs Non-decomposable Loss

This is the most important distinction for understanding when CLI is needed.

### Decomposable loss

A loss is decomposable if it can be written as a **sum of per-label losses**:
```
ℓ(y, ŷ) = Σₖ ℓₖ(yₖ, ŷₖ)
```
Each label contributes independently. Label k being wrong does not affect how label j is scored.

**Example: Hamming loss**
```
ℓ_H(y, ŷ) = Σₖ 𝟙[yₖ ≠ ŷₖ]

y  = (1, 1, 0, 0)
ŷ  = (1, 0, 0, 1)
         ↑     ↑
         wrong wrong

Hamming = 0 + 1 + 0 + 1 = 2
```
Label 2 being wrong (contributed +1) has nothing to do with label 4 being wrong (+1). Completely separable.

**BOP for Hamming — does NOT need CLI:**
```
ŷₖ* = 1  if pₖ > 0.5
ŷₖ* = 0  if pₖ ≤ 0.5
```
Each label is optimized independently. No need to know the joint distribution.

### Non-decomposable loss

A loss is non-decomposable if it **cannot** be written as a sum of per-label losses.
To compute the loss, you must look at all labels simultaneously.

**Example 1: Subset 0/1 loss**
```
ℓ_S(y, ŷ) = 𝟙[y ≠ ŷ]

y  = (1, 1, 0, 0)
ŷ₁ = (1, 1, 0, 0)  →  loss = 0  (perfect)
ŷ₂ = (1, 0, 0, 0)  →  loss = 1  (one label wrong → all wrong)
ŷ₃ = (0, 0, 0, 0)  →  loss = 1  (two labels wrong → still all wrong)
```
It doesn't matter if 1 label is wrong or 4 labels are wrong — loss is always 1.
Cannot be decomposed per-label.

**Example 2: F-measure**
```
F₁(y, ŷ) = 2·tp / (2·tp + fn + fp)

where:
tp = Σₖ yₖ·ŷₖ    (true positives — sum across ALL labels)
fn = Σₖ yₖ·(1-ŷₖ) (false negatives — sum across ALL labels)
fp = Σₖ (1-yₖ)·ŷₖ (false positives — sum across ALL labels)
```

Concrete example:
```
y  = (1, 1, 0, 0)
ŷ  = (1, 0, 0, 0)
tp = 1·1 + 1·0 + 0·0 + 0·0 = 1
fn = 1·0 + 1·1 + 0·0 + 0·0 = 1
fp = 0·0 + 0·0 + 1·0 + 1·0 = 0
F₁ = 2×1 / (2×1 + 1 + 0) = 2/3 = 0.67
```

To know the F₁ when predicting label 2, you must already know what happened with labels 1, 3, 4. They are entangled.

**Example 3: Rank loss**
```
ℓ_R(y, π) = number of incorrectly ordered pairs
           = count of pairs (i,j) where yᵢ=1, yⱼ=0, but π ranks j above i
```

Concrete example:
```
y = (1, 1, 0, 0)    (money laundering and tax evasion are present)
Ranking π: Tax evasion > Money laundering > Card fraud > TF

Check pairs where yᵢ=1 and yⱼ=0:
  (Money laundering, Card fraud):   ML ranked above CF ✓  → 0 error
  (Money laundering, TF):           ML ranked above TF ✓  → 0 error
  (Tax evasion, Card fraud):        TE ranked above CF ✓  → 0 error
  (Tax evasion, TF):                TE ranked above TF ✓  → 0 error

Rank loss = 0  (perfect ranking)
```

To compute rank loss for label "money laundering", you must know how it is ranked *relative to* every other label. Labels are compared pairwise — they cannot be scored independently.

### Why non-decomposable loss needs CLI

For BOP, we need to compute `E[L(y, ŷ)]`:
```
E[L(y, ŷ)] = Σ_{y∈Y} L(y, ŷ) · p(y|x)
```

For decomposable losses (Hamming):
```
E[ℓ_H] = Σₖ E[ℓₖ(yₖ, ŷₖ)] = Σₖ  p(yₖ≠ŷₖ|x)
```
This only needs the K marginals `pₖ = p(yₖ=1|x)`. **No CLI needed.**

For non-decomposable losses (F-measure):
```
E[F₁] = Σ_{y∈Y} F₁(y, ŷ) · p(y|x)
```
`F₁(y, ŷ)` depends on `tp`, `fn`, `fp` which involve all labels simultaneously.
To sum over all `y`, we need `p(y₁, y₂, ..., yₖ|x)` — the full joint distribution.

**Without CLI:** Must estimate `2ᴷ` joint probabilities. Infeasible for large K.
**With CLI:** `p(y|x) = ∏ₖ pₖ^{yₖ}·(1-pₖ)^{1-yₖ}`. Only K marginals needed. ✓

**Summary table:**

| Loss | Decomposable? | Needs CLI for BOP? | BOP complexity |
|------|:---:|:---:|:---:|
| Hamming | Yes | No | O(K) |
| Rank loss | No | Yes | O(K log K) |
| Subset 0/1 | No | Yes | O(K log K) |
| F-measure | No | Yes | O(K³) |
| Jaccard | No | Yes | O(K³) |

---

## Part 7 — BOP Algorithms for Each Loss

### 7.1 Hamming Loss (Decomposable)

**Key insight:** Since loss decomposes per label, optimize each label independently.

**Uncertainty measure:**
```
uₖ = 2·min(pₖ, 1-pₖ)

Examples:
  pₖ = 0.50  →  uₖ = 1.00  (maximum uncertainty)
  pₖ = 0.90  →  uₖ = 0.20  (mostly certain: present)
  pₖ = 0.10  →  uₖ = 0.20  (mostly certain: absent)
  pₖ = 0.70  →  uₖ = 0.60  (fairly uncertain)
```

**Linear penalty algorithm — O(K):**
```
For each label k:
  if 2·min(pₖ, 1-pₖ) ≤ c:
    DECIDE: ŷₖ = 1 if pₖ > 0.5, else ŷₖ = 0
  else:
    ABSTAIN: ŷₖ = ⊥
```

**Why threshold is c:**
```
Cost of abstaining on label k  = c
Expected error if we predict   = min(pₖ, 1-pₖ)

Abstain when: c < expected error  ←→  uₖ > c
Decide when:  c ≥ expected error  ←→  uₖ ≤ c
```

**General penalty algorithm — O(K log K):**
```
1. Compute sₖ = min(pₖ, 1-pₖ) for each k
2. Sort labels: s_{π(1)} ≤ s_{π(2)} ≤ ... ≤ s_{π(K)}
3. For d = 0 to K:
     D_d = {π(1), ..., π(d)}    (d most certain labels)
     E_d = Σ_{k∈D_d} sₖ + g(K-d)
4. d* = argmin_d E_d
5. Predict on D_{d*}, abstain on rest
```

**BOP structure:** Uncertainty-aligned ✓

### 7.2 Rank Loss (Non-decomposable, needs CLI)

**What it measures:** The number of label pairs that are ranked in the wrong order.
A pair (i, j) is wrong if `yᵢ=1`, `yⱼ=0`, but label j is ranked higher than label i.

**Under CLI, expected rank loss for a selection D:**
```
E[ℓ_R(y, π_D)] = Σ_{i<j in D} (1-pᵢ)·pⱼ
```
(sum over all pairs in the decision set where i is ranked above j)

**Key theorem (Lemma 2 in paper):**
> If `D_d = ⟪l, r⟫` is an optimal d-selection, then at least one of `⟪l+1, r⟫` or `⟪l, r-1⟫` is an optimal (d+1)-selection.

This means we can build the optimal selection greedily, always extending from the current boundary — we never need to check all O(K²) pairs.

**Algorithm (O(K log K)):**
```
1. Sort labels by pₖ descending: p_{π(1)} ≥ p_{π(2)} ≥ ... ≥ p_{π(K)}
2. Initialize:
     D₀ = ∅,  E₀ = g(K)
     D₂ = ⟪1,K⟫,  l=1, r=K
3. For d = 3 to K:
     Left extension:  K_l = ⟪l+1, r⟫
     Right extension: K_r = ⟪l, r-1⟫
     Pick whichever has lower expected rank loss
     Update l or r accordingly
     Compute E_d = E[ℓ_R(π_{D_d})] + g(K-d)
4. d* = argmin_{d∈{0,2,...,K}} E_d
5. Output ranking π_{D_{d*}}
```

**BOP structure:** Semi-uncertainty-aligned (not fully uncertainty-aligned — proven by counter-example in paper)

### 7.3 Subset 0/1 Loss (Non-decomposable, needs CLI)

**What it measures:** 0 if all labels correct, 1 if any label wrong.

**Under CLI:**
```
E[ℓ_S(y, ŷ_D)] = 1 - p(ŷ_D | x) = 1 - ∏_{k∈D} max(pₖ, 1-pₖ)
```

**Algorithm (O(K log K)):**
```
1. Sort by uncertainty uₖ increasing (most certain first)
2. For d = 0 to K:
     D_d = d most certain labels
     E_d = [1 - ∏_{k∈D_d} max(pₖ, 1-pₖ)] + g(K-d)
3. d* = argmin E_d
4. Predict argmax_{y∈{0,1}} pₖ^y·(1-pₖ)^{1-y} on D_{d*}
```

**BOP structure:** Uncertainty-aligned ✓

### 7.4 F-measure (Non-decomposable, needs CLI)

**What it measures:**
```
F_β(y, ŷ) = (1+β²)·tp / [(1+β²)·tp + β²·fn + fp]
F₁ (β=1) = 2·tp / (2·tp + fn + fp)
```

**Why F-measure is important for banking:**
When positive labels are rare (most transactions are legitimate), Hamming loss is misleading — predicting "all zeros" gives near-zero Hamming loss. F-measure penalizes this by specifically measuring performance on the positive class.

**Dynamic programming preprocessing (O(K²)):**
```
Q(l, l₁) = P(exactly l₁ positives among first l labels | x)
P(r', r'₁) = P(exactly r'₁ positives among last r' labels | x)
```
Both computed using DP under CLI assumption.

**Main algorithm (O(K³)):**
```
1. Sort labels by pₖ descending
2. Compute Q and P matrices via DP
3. For each (l, r) pair:
     Compute E[F_β | D = ⟪l,r⟫] using Q, P
     Store F_β(l, r) - g(r-l-1)
4. (l*, r*) = argmax F_β(l, r) - g(r-l-1)
5. Output prediction with D = ⟪l*, r*⟫
```

**Why O(K³):** There are O(K²) pairs (l,r), and for each pair computing the expectation takes O(K) time via the recursive S matrix.

**BOP structure:** Semi-uncertainty-aligned ✓

### 7.5 Jaccard Index (Non-decomposable, needs CLI)

```
f_Jac(y, ŷ) = tp / (tp + fn + fp) = |y ∩ ŷ| / |y ∪ ŷ|
```

Same structure as F-measure, also O(K³). Differs in that it completely ignores true negatives (tn). Where F₁ treats false positives and false negatives symmetrically, Jaccard is stricter about the overlap between prediction and truth.

---

## Part 8 — The Classifiers

### Why three classifier families?

The BOP algorithms take marginal probabilities `p₁, ..., pₖ` as input. The classifiers are responsible for producing these probabilities. The paper tests whether abstention helps across different quality levels of probability estimates.

### Binary Relevance (BR) — assumes CLI by design

**The idea:** Train K independent binary classifiers, one per label.
```
Label 1: classifier₁(x) → p₁ = P(Y₁=1|x)
Label 2: classifier₂(x) → p₂ = P(Y₂=1|x)
...
Label K: classifierₖ(x) → pₖ = P(Yₖ=1|x)
```

**Why it works under CLI:**
If labels are conditionally independent given x, then training separate classifiers per label loses no information. Each classifier can focus entirely on its own label.

**Why it fails under label dependency:**
If Y₁ and Y₂ are correlated given x, then information about Y₁ helps predict Y₂. The BR classifiers cannot capture this — each classifier only sees its own label during training.

**Two variants:**
- `BR+LR`: Logistic Regression as base learner. Naturally outputs calibrated probabilities.
- `BR+SVM`: Support Vector Machine + **Platt Scaling**. SVMs output "scores" (not probabilities) — Platt Scaling fits a sigmoid on top to convert scores into calibrated probabilities. Without this step, the scores cannot be used in BOP algorithms.

### Ensemble of Classifier Chains (ECC) — captures label dependency

**The key idea:** Each classifier chain predicts labels sequentially, using the predictions of previous labels as additional input features.

**One chain (random label ordering):**
```
x → Classifier₁ → p(Y₃=1|x)
         ↓
x, Y₃ → Classifier₂ → p(Y₁=1|x, Y₃)
              ↓
x, Y₃, Y₁ → Classifier₃ → p(Y₂=1|x, Y₃, Y₁)
```

Each subsequent classifier is conditioned on all previous label predictions — it implicitly captures label dependencies.

**Why an ensemble (M=50 chains)?**
A single chain depends heavily on the arbitrary label ordering chosen. Different orderings produce different probability estimates. By training M=50 chains with different random orderings and averaging:
```
p̄ₖ = (1/M) × Σₘ pₖ,ₘ
```
The ensemble averages out the ordering bias and produces more stable estimates.

**Under CLI violation:** ECC compensates. Even if the underlying data violates CLI, ECC produces probability estimates that capture label correlations — making the BOP algorithms more accurate than if raw BR estimates were used.

### EVGG16 — for image data only

Ensemble of 5 VGG16 convolutional neural networks, pretrained on ImageNet and fine-tuned on the natural scene dataset (5 labels: desert, mountains, sea, sunset, trees).

**Architecture per model:**
```
Input (128×128×3)
   → VGG16 backbone (pretrained, frozen except last conv block)
   → Dense(128) hidden layer
   → Dense(5) + Sigmoid output
```

**Training:** SGD with Nesterov momentum, lr=10⁻⁶, 100 epochs, binary cross-entropy loss.
**Ensemble:** Average the 5 models' output probabilities → final marginals.
**Evaluation:** 3-fold CV (instead of 10-fold for tabular data, due to training cost).

---

## Part 9 — The Experiments

### The central question

> "Does allowing abstention actually reduce loss compared to always predicting?"

This is verified across 6 datasets, 5 loss functions, 5 classifier variants, and 2 penalty types.

### Experimental setup

**Datasets:**

| Dataset | Instances | Features | Labels | Domain |
|---------|----------:|:--------:|:------:|--------|
| CAL500 | 502 | 68 | 174 | Music annotations |
| EMOTIONS | 593 | 72 | 6 | Music emotions |
| SCENE | 2,407 | 294 | 6 | Image scenes |
| YEAST | 2,417 | 103 | 14 | Protein functions |
| MEDIAMILL | 43,907 | 120 | 101 | Video annotations |
| NUS-WIDE | 269,648 | 128 | 81 | Image tags |
| Natural Scene | 2,000 | 128×128×3 | 5 | Scene images |

**Baselines for comparison:**
- `MLC`: Standard prediction, no abstention (lower bound on loss, upper bound on coverage)
- `ABS`: Abstain on all labels (upper bound on safety, zero usefulness)

**Methods being tested:**
- `SEP`: BOP with linear penalty `g₁(a) = a·c`
- `PAR`: BOP with concave penalty `g₂(a) = a·K·c/(K+a)`

**Cost parameter ranges (vary to trace the trade-off curve):**

| Loss | c range (SEP) | Why this range? |
|------|:---:|---|
| Hamming | [0.05, 0.5] | Loss is in [0, K], so c in same scale |
| Rank | [0.1, 1] | Wrong pair can cost up to K-1 |
| Subset 0/1 | [0.25/K, 2.5/K] | Loss is {0,1}, so penalty must be tiny |
| F₁, Jaccard | [0.1/K, 1/K] | Same reasoning as Subset 0/1 |

### Experiment 1 — Loss vs Cost trade-off curves (Figures 1–6)

**What it does:**
For each value of `c`, compute the loss and abstention rate on the test set.

**What the plots show:**
- X-axis: cost of abstention `c` (increasing right = more expensive to abstain)
- Left Y-axis: average loss on decided labels
- Right Y-axis: abstention rate `|A|/K`

**Expected and observed pattern:**
```
Low c:   PAR/SEP abstain a lot → low loss, high abstention rate
High c:  PAR/SEP abstain less → converge toward MLC loss
Medium c: sweet spot — loss significantly below MLC, still useful
```

**Key findings:**
- SEP and PAR consistently outperform MLC (lower loss for same cost)
- PAR typically outperforms SEP — the concave penalty provides a better trade-off
- This holds across all 5 loss functions, all 6 datasets, both BR and ECC models
- Results for CNN (Figure 6) mirror tabular results — abstention helps even for deep models

### Experiment 2 — Meta-analysis: when does abstention help most? (Figure 7)

**What it does:**
Aggregate results across all 21 configurations (dataset × loss × classifier combinations).
For each configuration, compute:
- `x-axis`: the average loss of the standard MLC baseline (proxy for how hard the task is)
- `y-axis left`: average gain from abstention (MLC loss - abstention loss)
- `y-axis right`: average abstention rate

**The key finding:**
```
Harder datasets (higher MLC loss) → larger gain from abstention
```

| Dataset difficulty | MLC baseline loss | Abstention gain | Abstention rate |
|---|:---:|:---:|:---:|
| Easy (SCENE) | ~5% | ~0.5% | ~5% |
| Medium (YEAST) | ~15% | ~3% | ~25% |
| Hard (CAL500) | ~20% | ~6-7% | ~40% |

**What this means:**
When a model is already good, there's little room to gain — it rarely needs to abstain.
When a model struggles, abstention is most valuable — it can skip the labels it would likely get wrong, reducing its mistakes substantially.

The model learns to be cautious in proportion to its own uncertainty. This is exactly the desired behavior for high-stakes applications.

---

## Part 10 — Connecting Everything

Here is the complete logical flow of the paper:

```
Real-world need:
  "In banking, medical, legal domains — wrong prediction is worse
   than admitting uncertainty."
          ↓
Problem formulation:
  Standard MLC forces hard predictions on all K labels.
  Extension: allow ŷₖ = ⊥ (abstain).
          ↓
Framework design:
  Generalized loss: L(y, ŷ) = ℓ(y_D, ŷ_D) + g(|A|)
  Two penalties: SEP (linear) and PAR (concave)
  Desired properties: monotonicity, uncertainty-alignment
          ↓
Optimization goal:
  Find BOP = argmin E[L(y, ŷ)]
  Problem: 3ᴷ candidates → infeasible to brute-force
          ↓
Key theoretical insight:
  For each loss type, BOP has special structure:
    Decomposable  → uncertainty-aligned   → O(K) or O(K log K)
    Non-decomp.   → semi-uncertainty-aligned → O(K log K) to O(K³)
          ↓
Why non-decomposable needs CLI:
  To compute E[F₁], E[Rank loss], etc. need p(y₁,...,yₖ|x)
  Without CLI: 2ᴷ parameters → infeasible
  With CLI: p(y|x) = ∏ₖ pₖ^{yₖ}·(1-pₖ)^{1-yₖ} → K parameters ✓
          ↓
Experimental validation:
  - BOP with abstention beats standard MLC on all datasets
  - Most benefit when model is uncertain (hard datasets)
  - PAR > SEP in most configurations
  - ECC > BR because ECC better estimates marginals under label dependency
```

---

## Quick Reference

### Key formulas

| Concept | Formula |
|---------|---------|
| Generalized loss | `L(y,ŷ) = ℓ(y_D, ŷ_D) + g(\|A\|)` |
| Linear penalty | `g₁(a) = a·c` |
| Concave penalty | `g₂(a) = a·K·c / (K+a)` |
| Uncertainty | `uₖ = 2·min(pₖ, 1-pₖ)` |
| CLI | `p(y\|x) = ∏ₖ pₖ^{yₖ}·(1-pₖ)^{1-yₖ}` |
| BOP objective | `ŷ* = argmin E[L(y,ŷ)]` |
| Hamming | `ℓ_H = Σₖ 𝟙[yₖ≠ŷₖ]` |
| F₁ | `2tp/(2tp+fn+fp)` |
| Jaccard | `tp/(tp+fn+fp)` |

### When does each component matter?

| Situation | Relevant concept |
|-----------|-----------------|
| Labels co-occur frequently | CLI assumption — check if it holds |
| Using F-measure or Rank loss | CLI required for BOP |
| Using Hamming loss | CLI not needed — BOP works regardless |
| Model is uncertain on many labels | Abstention most beneficial |
| Need fastest BOP | Use Hamming O(K) |
| Need most accurate BOP | Use F-measure O(K³) |
| Labels have strong dependencies | Use ECC instead of BR |
| SVM as base learner | Must apply Platt Scaling |

---

*Paper: Nguyen, V-L. & Hüllermeier, E. (2021). Multilabel Classification with Partial Abstention: Bayes-Optimal Prediction under Label Independence. JAIR 72:613–665.*
