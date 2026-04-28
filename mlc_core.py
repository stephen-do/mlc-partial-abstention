"""
mlc_core.py
===========
Implementation of:
    "Multilabel Classification with Partial Abstention:
     Bayes-Optimal Prediction under Label Independence"
    Nguyen & Hüllermeier, JAIR 2021

File structure:
    1. Penalty functions g(a)           — Section 3.2.2
    2. Bayes-Optimal Predictors (BOP)   — Sections 4, 5, 6, 7
    3. Generalized Loss Functions       — Section 3.2
    4. Evaluation utilities
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ============================================================
# PART 1: PENALTY FUNCTIONS g(a)
# Section 3.2.2, Equations (30) and (31)
# ============================================================

def penalty_linear(a: int, K: int, c: float) -> float:
    """
    SEP — Linear penalty (Equation 30):
        g1(a) = a * c

    Every abstained label incurs the same cost c, regardless of
    whether it is the first or the tenth abstention.

    Args:
        a: number of abstained labels |A(ŷ)|
        K: total number of labels (unused here, kept for API consistency)
        c: per-label abstention cost
    """
    return a * c


def penalty_concave(a: int, K: int, c: float) -> float:
    """
    PAR — Concave penalty (Equation 31):
        g2(a) = (a * K * c) / (K + a)

    The first abstention is penalised more than subsequent ones.
    This is a concave function — marginal cost is decreasing.
    Encourages more abstentions than SEP at the same c.

    Note from paper (Section 9.1.4): at the same cost c,
    PAR is equivalent to SEP with c' = 2c.
    """
    return (a * K * c) / (K + a) if (K + a) > 0 else 0.0


def make_penalty(penalty_type: str, K: int, c: float) -> Callable[[int], float]:
    """
    Factory that returns a penalty function with K and c already bound.
    Returns g(a) accepting exactly one argument.

    Args:
        penalty_type: 'linear' (SEP) or 'concave' (PAR)
        K: total number of labels
        c: abstention cost coefficient
    """
    if penalty_type == 'linear':
        return lambda a: penalty_linear(a, K, c)
    elif penalty_type == 'concave':
        return lambda a: penalty_concave(a, K, c)
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}. Use 'linear' or 'concave'.")


# ============================================================
# PART 2: BAYES-OPTIMAL PREDICTORS
# ============================================================

# ----------------------------------------------------------
# 2.1 Hamming Loss BOP
# Section 4, Corollary 1 & 2
# ----------------------------------------------------------

def bop_hamming(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP for the Generalized Hamming Loss (Equation 38).

    Hamming loss is a decomposable loss (Equation 16):
        ℓ_H(y, ŷ) = Σ_k [y_k ≠ ŷ_k]

    With abstention each label is considered independently:
        - if p_k > 0.5: predict 1 (expected loss = 1 - p_k)
        - if p_k < 0.5: predict 0 (expected loss = p_k)
        - abstain when the abstention cost beats the prediction cost

    Corollary 1: BOP is uncertainty-aligned —
    abstain on labels closest to 0.5 (highest min(p_k, 1-p_k)),
    predict on labels farthest from 0.5.

    Algorithm (Corollary 1):
        1. Compute s_k = min(p_k, 1-p_k)  — minimum expected loss per label
        2. Sort labels by s_k ascending
        3. For each d = 0..K: compute expected total loss when predicting d labels
        4. Choose d* minimising expected loss + g(K-d)

    Complexity: O(K log K) due to sorting
    """
    K = len(probs)

    # s_k = minimum expected loss if we predict label k (Proposition 1):
    # s_k = min{p_k * ℓ(0,1), (1-p_k) * ℓ(1,0)}
    # For Hamming loss ℓ(0,1) = ℓ(1,0) = 1, so:
    s = np.minimum(probs, 1 - probs)  # shape (K,)

    # Sort ascending by s_k — most confident labels first
    sorted_idx = np.argsort(s)

    # Point-estimate prediction for each label
    best_pred = (probs >= 0.5).astype(float)

    best_d = 0
    best_expected = g(K)  # cost of full abstention (d=0)

    # Cumulative expected Hamming loss when predicting d labels
    cumulative_loss = 0.0
    for d in range(1, K + 1):
        k = sorted_idx[d - 1]
        cumulative_loss += s[k]
        total = cumulative_loss + g(K - d)
        if total < best_expected:
            best_expected = total
            best_d = d

    # Build prediction vector: D(ŷ) = {d labels with smallest s_k}
    pred = np.full(K, np.nan)  # NaN = ⊥ (abstain)
    decided_indices = sorted_idx[:best_d]
    pred[decided_indices] = best_pred[decided_indices]

    return pred


# ----------------------------------------------------------
# 2.2 Rank Loss BOP
# Section 5, Proposition 2, Algorithm 1
# ----------------------------------------------------------

def bop_rank(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP for the Generalized Rank Loss (Algorithm 1, Proposition 2).

    Rank loss counts misordered label pairs:
        ℓ_R(y, π) = Σ_{i<j} [y_{π(i)}=0 ∧ y_{π(j)}=1]

    Under CLI (Lemma 1, Equation 46), the expected rank loss of the
    optimal ranking on decision set D_d = ⟪l,r⟫ is:
        E[ℓ_R(y, π_{D_d})] = Σ_{i<j in D_d} (1-p_{π(i)}) * p_{π(j)}

    BOP is semi-uncertainty-aligned (Lemma 1):
        D_d = ⟪l,r⟫ = {1,...,l} ∪ {r,...,K}  (after sorting p descending)

    By Lemma 2: if ⟪l,r⟫ is an optimal d-selection, then at least one of
    ⟪l+1,r⟫ or ⟪l,r-1⟫ is an optimal (d+1)-selection → greedy construction.

    Algorithm 1 (O(K log K)):
        1. Sort p descending
        2. D₀=∅, D₂=⟪1,K⟫, l=1, r=K
        3. Greedily extend from d=3..K: pick the direction with lower E[ℓ_R]
        4. d* = argmin_{d∈{0,2,...,K}} E_d
        5. Output ranking π_{D_{d*}}

    Returns:
        pred: rank array (0.0 = highest rank, NaN = abstain).
              Label π(1) (largest p in D) gets rank 0, π(2) gets rank 1, …
    """
    K = len(probs)

    # Step 1: sort labels descending by p_k
    sorted_idx = np.argsort(-probs)   # sorted_idx[i] = original label at position i
    p = probs[sorted_idx]             # p[0] >= p[1] >= ... >= p[K-1]

    if K < 2:
        # K=1: rank loss is always 0; predict if it improves over full abstention
        pred = np.full(K, np.nan)
        if K == 1 and g(1) > g(0):
            pred[0] = 0.0
        return pred

    # best_total[d] = E_rank(D_d) + g(K-d)  for d = 0..K
    best_total = np.full(K + 1, np.inf)
    # track (l, r) 0-indexed at each d (left side: 0..l, right side: r..K-1)
    decisions = [None] * (K + 1)

    # d=0: full abstention
    best_total[0] = g(K)
    decisions[0] = (-1, K)  # empty set

    # d=2: D₂=⟪1,K⟫ → 0-indexed: l=0, r=K-1 (leftmost and rightmost labels)
    l, r = 0, K - 1
    # E_rank(⟪l,r⟫) with l=0, r=K-1: only one pair (0, K-1)
    E_rank = (1.0 - p[l]) * p[r]
    best_total[2] = E_rank + g(K - 2)
    decisions[2] = (l, r)

    # sum_right     = Σ p[j] for j in right side (r..K-1)
    # sum_left_neg  = Σ (1-p[i]) for i in left side (0..l)
    sum_right = float(np.sum(p[r:]))
    sum_left_neg = float(np.sum(1.0 - p[:l + 1]))

    for d in range(3, K + 1):
        # Try extending left: add position l+1 to left side
        if l + 1 < r:
            x = l + 1
            # New pairs: (x, j) for j in right side r..K-1, x ranked above j
            delta_left = (1.0 - p[x]) * sum_right
            E_left = E_rank + delta_left
        else:
            E_left = np.inf

        # Try extending right: add position r-1 to right side
        if l < r - 1:
            x = r - 1
            # New pairs: (i, x) for i in left side 0..l, i ranked above x
            delta_right = sum_left_neg * p[x]
            E_right = E_rank + delta_right
        else:
            E_right = np.inf

        if E_left <= E_right:
            l = l + 1
            E_rank = E_left
            sum_left_neg += (1.0 - p[l])
            # sum_right unchanged (r did not change)
        else:
            r = r - 1
            E_rank = E_right
            sum_right += p[r]
            # sum_left_neg unchanged (l did not change)

        best_total[d] = E_rank + g(K - d)
        decisions[d] = (l, r)

    # d* = argmin_{d ∈ {0,2,...,K}} best_total[d]
    # (d=1 is skipped: by Lemma 1 the BOP always has the form ⟪l,r⟫)
    d_star = 0
    for d in range(0, K + 1):
        if d == 1:
            continue
        if best_total[d] < best_total[d_star]:
            d_star = d

    # Build prediction
    pred = np.full(K, np.nan)
    if d_star == 0:
        return pred  # full abstention

    l_star, r_star = decisions[d_star]
    rank = 0.0
    for i in range(0, l_star + 1):
        pred[sorted_idx[i]] = rank
        rank += 1.0
    for i in range(r_star, K):
        pred[sorted_idx[i]] = rank
        rank += 1.0

    return pred


# ----------------------------------------------------------
# 2.3 Subset 0/1 Loss BOP
# Section 6, Proposition 3
# ----------------------------------------------------------

def bop_subset01(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP for the Generalized Subset 0/1 Loss (Equation 48).

    Subset 0/1 is a non-decomposable loss:
        ℓ_S(y, ŷ) = [y ≠ ŷ]  (one wrong label = entirely wrong)

    Requires CLI (Conditional Label Independence, Equation 15):
        p(y|x) = Π_k p_k^{y_k} * (1-p_k)^{1-y_k}

    Under CLI, the optimal per-label prediction is:
        ŷ_k^d = argmax_{ŷ_k ∈ {0,1}} p_k^{ŷ_k} * (1-p_k)^{1-ŷ_k}
               = round(p_k)  — predict the more probable value

    Key insight (Proposition 3): BOP is uncertainty-aligned —
    abstain on labels closest to 0.5 (highest u_k = 2*min(p_k, 1-p_k)).
    Same structure as Hamming BOP, different expected loss formula.

    Expected loss when predicting the d most certain labels:
        E[ℓ_S] = 1 - Π_{k in D} max(p_k, 1-p_k)

    Complexity: O(K log K)
    """
    K = len(probs)

    # u_k = 2*min(p_k, 1-p_k) — degree of uncertainty (Equation 24)
    uncertainty = 2 * np.minimum(probs, 1 - probs)

    # Sort ascending by uncertainty — most confident labels first
    sorted_idx = np.argsort(uncertainty)

    # max(p_k, 1-p_k) = probability of the best point prediction for label k
    max_probs = np.maximum(probs, 1 - probs)

    best_pred = (probs >= 0.5).astype(float)
    best_d = 0
    best_expected = g(K)  # full abstention

    # Accumulate log-product of max_probs:
    # E[ℓ_S | predict d labels] = 1 - Π_{k in D_d} max(p_k, 1-p_k)
    log_prod = 0.0
    for d in range(1, K + 1):
        k = sorted_idx[d - 1]
        log_prod += np.log(max_probs[k] + 1e-15)
        expected_loss = 1.0 - np.exp(log_prod)
        total = expected_loss + g(K - d)
        if total < best_expected:
            best_expected = total
            best_d = d

    pred = np.full(K, np.nan)
    pred[sorted_idx[:best_d]] = best_pred[sorted_idx[:best_d]]
    return pred


# ----------------------------------------------------------
# 2.4 F-measure BOP
# Section 7.3, Proposition 6, Algorithm 2
# ----------------------------------------------------------

def _compute_Q_matrix(probs_sorted: np.ndarray) -> np.ndarray:
    """
    Compute matrix Q(l, l1) — Lemma 4, Equation 57.

    Q(l, l1) = P(Σ_{k=1}^{l} y_{π(k)} = l1 | x)

    Probability of exactly l1 positives among the first l labels
    (after sorting by p_k descending).

    Dynamic programming recurrence:
        Q(l, l1) = p_{π(l)} * Q(l-1, l1-1) + (1-p_{π(l)}) * Q(l-1, l1)

    Complexity: O(K²)
    Output shape: (K+1, K+2)  — l in 0..K, l1 offset by +1 to avoid negative index
    """
    K = len(probs_sorted)
    # Q[l, l1+1] to avoid negative indices
    Q = np.zeros((K + 1, K + 2))

    # Base case: Q(0, 0) = 1 (no labels, no positives)
    Q[0, 0 + 1] = 1.0

    for l in range(1, K + 1):
        p = probs_sorted[l - 1]
        for l1 in range(0, l + 1):
            q_prev_pos = Q[l - 1, l1 - 1 + 1] if l1 > 0 else 0.0
            q_prev_neg = Q[l - 1, l1 + 1]
            Q[l, l1 + 1] = p * q_prev_pos + (1 - p) * q_prev_neg

    return Q


def _compute_P_matrix(probs_sorted: np.ndarray) -> np.ndarray:
    """
    Compute matrix P(r', r'1) — Lemma 4, Equation 58.

    P(r', r'1) = P(Σ_{k=r}^{K} y_{π(k)} = r'1 | x)
    where r' = K+1-r (number of labels in the tail starting at position r)

    Same DP as Q but computed from the end of the sorted list (lowest p first).
    Used for the right part of the decision set ⟪l,r⟫.
    """
    K = len(probs_sorted)
    P = np.zeros((K + 1, K + 2))
    P[0, 0 + 1] = 1.0

    for rp in range(1, K + 1):
        # Label from the tail: index K-rp in the descending-sorted array
        p = probs_sorted[K - rp]
        for rp1 in range(0, rp + 1):
            q_prev_pos = P[rp - 1, rp1 - 1 + 1] if rp1 > 0 else 0.0
            q_prev_neg = P[rp - 1, rp1 + 1]
            P[rp, rp1 + 1] = p * q_prev_pos + (1 - p) * q_prev_neg

    return P


def bop_fmeasure(probs: np.ndarray, g: Callable[[int], float],
                 beta: float = 1.0) -> np.ndarray:
    """
    BOP for the Generalized F_beta measure (Algorithm 2, Proposition 6).

    F_beta = (1+β²) * tp / [(1+β²)*tp + β²*fn + fp]

    Key results (Proposition 5, Lemma 3):
    - F_beta is semi-uncertainty-aligned
    - BOP has decision set ⟪l,r⟫: predict top-l labels (high p) and
      bottom-r' labels (low p), abstain in the middle
    - Found via dynamic programming, complexity O(K³)

    Decision set ⟪l,r⟫ = {1,...,l} ∪ {r,...,K}
    (indices after sorting p descending)
    Labels 1..l → predict 1, labels r..K → predict 0, middle → abstain.

    S_β recursion (Appendix, Proof of Prop 6):
        S_β(l, l1, r') = p_π(r) * S(l, l1+1, r'-1) + (1-p_π(r)) * S(l, l1, r'-1)
        S_β(l, l1, 0)  = 1 / (l*β⁻² + l1)   [boundary]

    Algorithm 2 lines 12-14 are the in-place form of this recursion:
        for i = 0 to K-l-r':  S(l,i) ← p_r*S(l,i+1) + (1-p_r)*S(l,i)

    Args:
        probs: marginal probabilities p_k, shape (K,)
        g: penalty function g(a) where a = number of abstentions
        beta: F_beta parameter (default 1.0 = F1)
    """
    K = len(probs)
    beta_sq = beta ** 2
    beta_inv2 = 1.0 / beta_sq   # β⁻²
    beta_prime = 1.0 + beta_inv2  # β' = 1 + β⁻²  (Appendix, Proof of Prop 6)

    # Sort labels descending by p_k (Remark 3, Proposition 6)
    sorted_idx = np.argsort(-probs)
    probs_sorted = probs[sorted_idx]

    # Compute Q matrix (Lemma 4) — P is not needed since S is updated in-place
    Q = _compute_Q_matrix(probs_sorted)

    best_val = -np.inf
    best_l, best_r = 0, K + 1

    # d=0: full abstention, F=0
    val_empty = 0.0 - g(K)
    if val_empty > best_val:
        best_val = val_empty
        best_l, best_r = 0, K + 1

    # Iterate over all pairs (l, r) with 1 <= l, r descending from K+1 to l+1
    # Algorithm 2 — O(K³)
    for l in range(1, K + 1):
        # Boundary: S(l, l1, r'=0) = 1 / (l*β⁻² + l1)
        # S[i] represents S(l, i, r') — r' increases as r decreases in the inner loop
        S = np.zeros(K + 2)
        for i in range(0, K + 1):
            denom = l * beta_inv2 + i
            if denom > 1e-10:
                S[i] = 1.0 / denom

        # F_β(l, K+1): r'=0, no tail labels, g(K-l) abstentions
        fb = beta_prime * sum(
            l1 * Q[l, l1 + 1] * S[l1] for l1 in range(0, l + 1)
        ) - g(K - l)
        if fb > best_val:
            best_val = fb
            best_l, best_r = l, K + 1

        # r decreases from K to l+1: each step adds label π(r) to the tail (r' += 1)
        # In-place S update per Algorithm 2 lines 12-14:
        #   for i = 0 to K-l-r': S(l,i) ← p_r*S(l,i+1) + (1-p_r)*S(l,i)
        # Iterate i ascending to avoid overwriting S[i+1] before it is read
        for r in range(K, l, -1):
            rp = K + 1 - r      # r' = number of labels added to the tail so far
            pr = probs_sorted[r - 1]
            limit = K - l - rp  # K - l - r'  (Algorithm 2 line 12)
            for i in range(0, limit + 1):
                S[i] = pr * S[i + 1] + (1.0 - pr) * S[i]

            fb = beta_prime * sum(
                l1 * Q[l, l1 + 1] * S[l1] for l1 in range(0, l + 1)
            ) - g(r - l - 1)
            if fb > best_val:
                best_val = fb
                best_l, best_r = l, r

    # Build prediction from (best_l, best_r):
    # top-l labels → predict 1, labels best_r..K → predict 0, middle → abstain
    pred = np.full(K, np.nan)
    for i in range(best_l):
        pred[sorted_idx[i]] = 1.0
    for i in range(best_r - 1, K):
        pred[sorted_idx[i]] = 0.0

    return pred


# ----------------------------------------------------------
# 2.5 Jaccard Measure BOP
# Section 7.3, Proposition 7, Algorithm 3
# ----------------------------------------------------------

def bop_jaccard(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP for the Generalized Jaccard measure (Algorithm 3, Proposition 7).

    Jaccard = tp / (tp + fn + fp)

    Similar to F-measure but simpler as there is no β parameter.
    Semi-uncertainty-aligned, decision set of the form ⟪l,r⟫.
    Complexity: O(K³)

    S_Jac depends only on (l, r'), not on l1 individually:
        F_Jac(l, r) = S_Jac(l, r') * Σ_{l1} l1*Q(l,l1) - g(r-l-1)

    Recursion (Appendix, Proof of Prop 7):
        S_Jac(l, r') = p_π(r) * S_Jac(l+1, r'-1) + (1-p_π(r)) * S_Jac(l, r'-1)
        S_Jac(l, 0)  = 1/l   [boundary]

    Algorithm 3 lines 12-14 are the in-place form:
        for i = l+r' downto l+1:  S(i) ← p_r*S(i) + (1-p_r)*S(i-1)
    Iterate i descending to avoid overwriting S[i-1] before it is read.
    """
    K = len(probs)
    sorted_idx = np.argsort(-probs)
    probs_sorted = probs[sorted_idx]

    Q = _compute_Q_matrix(probs_sorted)

    best_val = -np.inf
    best_l, best_r = 0, K + 1

    # d=0: full abstention, F=0
    val_empty = 0.0 - g(K)
    if val_empty > best_val:
        best_val = val_empty
        best_l, best_r = 0, K + 1

    for l in range(1, K + 1):
        # Precompute Σ_{l1} l1*Q(l,l1) once for all r in this outer loop
        sum_l1Q = sum(l1 * Q[l, l1 + 1] for l1 in range(0, l + 1))

        # Boundary: S_Jac(i, r'=0) = 1/i for i = l..K
        # S[i] represents S_Jac(i, r') — r' increases as r decreases
        S = np.zeros(K + 2)
        for i in range(l, K + 1):
            S[i] = 1.0 / i  # S_Jac(i, 0) = 1/i

        # F_Jac(l, K+1): r'=0, S_Jac = S[l] = 1/l
        fj = S[l] * sum_l1Q - g(K - l)
        if fj > best_val:
            best_val = fj
            best_l, best_r = l, K + 1

        # r decreases from K to l+1: each step adds label π(r) to the tail (r' += 1)
        # In-place S update per Algorithm 3 lines 12-14:
        #   for i = l+r' downto l+1:  S(i) ← p_r*S(i) + (1-p_r)*S(i-1)
        for r in range(K, l, -1):
            rp = K + 1 - r      # r' = number of labels added to the tail so far
            pr = probs_sorted[r - 1]
            # Iterate i descending to avoid overwriting S[i-1] before it is read
            for i in range(l + rp, l, -1):
                S[i] = pr * S[i] + (1.0 - pr) * S[i - 1]

            # S_Jac(l, r') is now stored at S[l + rp]
            fj = S[l + rp] * sum_l1Q - g(r - l - 1)
            if fj > best_val:
                best_val = fj
                best_l, best_r = l, r

    pred = np.full(K, np.nan)
    for i in range(best_l):
        pred[sorted_idx[i]] = 1.0
    for i in range(best_r - 1, K):
        pred[sorted_idx[i]] = 0.0

    return pred


# ============================================================
# PART 3: LOSS FUNCTIONS (for evaluation)
# Section 2.2
# ============================================================

ABSTAIN = np.nan  # symbol ⊥


def _get_decided(pred: np.ndarray) -> np.ndarray:
    """Return boolean mask of labels that were predicted (not abstained)."""
    return ~np.isnan(pred)


def loss_hamming(y_true: np.ndarray, pred: np.ndarray) -> float:
    """
    Generalized Hamming loss with abstention (Equation 7 + 29).
    Computed only over decided labels; abstentions are penalised separately by g(|A|).
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 0.0
    return float(np.sum(y_true[decided] != pred[decided]))


def loss_subset01(y_true: np.ndarray, pred: np.ndarray) -> float:
    """
    Generalized Subset 0/1 loss (Equation 8 + 29).
    Returns 0 if all decided labels are correct, 1 if any is wrong.
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 0.0
    return float(not np.all(y_true[decided] == pred[decided]))


def loss_fmeasure(y_true: np.ndarray, pred: np.ndarray,
                  beta: float = 1.0) -> float:
    """
    Generalized F_beta loss = 1 - F_beta(y_D, ŷ_D) (Equation 9 + 51).
    Uses 1-F because F is an accuracy measure (higher = better).
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 1.0  # no prediction = F=0, loss=1

    yt = y_true[decided]
    yp = pred[decided]

    tp = float(np.sum(yt * yp))
    fn = float(np.sum(yt * (1 - yp)))
    fp = float(np.sum((1 - yt) * yp))

    denom = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    if denom < 1e-10:
        return 1.0
    f = (1 + beta ** 2) * tp / denom
    return 1.0 - f


def loss_jaccard(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Jaccard loss = 1 - Jaccard(y_D, ŷ_D) (Table 2 row 6 + 51)."""
    decided = _get_decided(pred)
    if not np.any(decided):
        return 1.0

    yt = y_true[decided]
    yp = pred[decided]

    tp = float(np.sum(yt * yp))
    fn = float(np.sum(yt * (1 - yp)))
    fp = float(np.sum((1 - yt) * yp))

    denom = tp + fn + fp
    if denom < 1e-10:
        return 0.0
    return 1.0 - tp / denom


def loss_rank(y_true: np.ndarray, pred_ranking: np.ndarray) -> float:
    """
    Rank loss (Equation from Section 5).
    Counts misordered label pairs: λ_i better than λ_j but y_i=1, y_j=0.

    pred_ranking: rank array (0 = highest rank), NaN = abstain
    """
    decided = _get_decided(pred_ranking)
    if np.sum(decided) < 2:
        return 0.0

    yt = y_true[decided]
    ranks = pred_ranking[decided]

    loss = 0.0
    n = len(yt)
    for i in range(n):
        for j in range(i + 1, n):
            # If y_i=1 and y_j=0, rank_i must be < rank_j (i ranked higher)
            if yt[i] == 1 and yt[j] == 0:
                if ranks[i] > ranks[j]:  # wrong order
                    loss += 1.0
            elif yt[i] == 0 and yt[j] == 1:
                if ranks[i] < ranks[j]:  # wrong order
                    loss += 1.0
    return loss


def abstention_size(pred: np.ndarray) -> float:
    """
    Fraction of abstained labels: |A(ŷ)| / K
    Section 9.1.4 — comparison criterion.
    """
    return float(np.sum(np.isnan(pred))) / len(pred)


# ============================================================
# PART 4: BASELINES
# Section 9.1.4
# ============================================================

def predict_mlc(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Standard MLC — no abstention.
    Baseline: predict 1 if p_k >= threshold, 0 otherwise.
    """
    return (probs >= threshold).astype(float)


def predict_full_abstain(K: int) -> np.ndarray:
    """
    Full abstention — abstain on every label.
    ABS baseline from the paper.
    """
    return np.full(K, np.nan)
