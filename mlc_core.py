"""
mlc_core.py
===========
Implementation của bài báo:
    "Multilabel Classification with Partial Abstention:
     Bayes-Optimal Prediction under Label Independence"
    Nguyen & Hüllermeier, JAIR 2021

Cấu trúc file:
    1. Penalty functions g(a)           — Section 3.2.2
    2. Bayes-Optimal Predictors (BOP)   — Sections 4, 5, 6, 7
    3. Generalized Loss Functions       — Section 3.2
    4. Evaluation utilities
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


# ============================================================
# PHẦN 1: PENALTY FUNCTIONS g(a)
# Section 3.2.2, Equations (30) và (31)
# ============================================================

def penalty_linear(a: int, K: int, c: float) -> float:
    """
    SEP — Linear penalty (Equation 30):
        g1(a) = a * c

    Mỗi nhãn abstain bị phạt đều nhau, không phân biệt
    đây là nhãn abstain thứ nhất hay thứ mười.

    Args:
        a: Số nhãn abstain |A(ŷ)|
        K: Tổng số nhãn (không dùng ở đây, giữ để API nhất quán)
        c: Hệ số phạt mỗi nhãn
    """
    return a * c


def penalty_concave(a: int, K: int, c: float) -> float:
    """
    PAR — Concave penalty (Equation 31):
        g2(a) = (a * K * c) / (K + a)

    Nhãn abstain đầu tiên bị phạt nặng hơn nhãn thứ N.
    Đây là hàm lõm (concave) — marginal cost giảm dần.
    Khuyến khích mô hình abstain nhiều hơn so với SEP
    khi c bằng nhau.

    Note từ bài báo (Section 9.1.4): với cùng cost c,
    PAR thực ra tương đương với SEP nhưng với c' = 2c.
    """
    return (a * K * c) / (K + a) if (K + a) > 0 else 0.0


def make_penalty(penalty_type: str, K: int, c: float) -> Callable[[int], float]:
    """
    Factory tạo penalty function với K và c đã bind sẵn.
    Trả về hàm g(a) nhận đúng 1 argument.

    Args:
        penalty_type: 'linear' (SEP) hoặc 'concave' (PAR)
        K: Tổng số nhãn
        c: Hệ số phạt
    """
    if penalty_type == 'linear':
        return lambda a: penalty_linear(a, K, c)
    elif penalty_type == 'concave':
        return lambda a: penalty_concave(a, K, c)
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}. Use 'linear' or 'concave'.")


# ============================================================
# PHẦN 2: BAYES-OPTIMAL PREDICTORS
# ============================================================

# ----------------------------------------------------------
# 2.1 Hamming Loss BOP
# Section 4, Corollary 1 & 2
# ----------------------------------------------------------

def bop_hamming(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP cho Generalized Hamming Loss (Equation 38).

    Hamming loss là decomposable loss (Equation 16):
        ℓ_H(y, ŷ) = Σ_k [y_k ≠ ŷ_k]

    Với abstention, mỗi nhãn được xét độc lập:
        - Nếu p_k > 0.5: dự đoán 1 (expected loss = 1 - p_k)
        - Nếu p_k < 0.5: dự đoán 0 (expected loss = p_k)
        - Loss khi abstain = c (hệ số penalty)

    Corollary 1: BOP có dạng uncertainty-aligned —
    abstain những nhãn có min(p_k, 1-p_k) lớn nhất
    (tức gần 0.5 nhất), predict những nhãn xa 0.5.

    Thuật toán (Corollary 1):
        1. Tính s_k = min(p_k, 1-p_k)  — minimum expected loss per label
        2. Sắp xếp nhãn theo s_k tăng dần
        3. Với mỗi d = 0..K: tính expected total loss khi predict d nhãn đầu
        4. Chọn d* tối thiểu hóa expected loss + g(K-d)

    Độ phức tạp: O(K log K) do sorting
    """
    K = len(probs)

    # Tính score s_k = min expected loss nếu predict nhãn k
    # (Proposition 1): s_k = min{p_k * ℓ(0,1), (1-p_k) * ℓ(1,0)}
    # Với Hamming loss: ℓ(0,1) = ℓ(1,0) = 1, nên:
    s = np.minimum(probs, 1 - probs)  # shape (K,)

    # Sắp xếp tăng dần theo s_k → nhãn tự tin nhất trước
    sorted_idx = np.argsort(s)

    # Tính dự đoán tốt nhất khi quyết định predict d nhãn
    # (luôn chọn d nhãn có s_k nhỏ nhất = tự tin nhất)
    best_pred = (probs >= 0.5).astype(float)  # dự đoán point estimate

    best_d = 0
    best_expected = g(K)  # cost khi abstain hoàn toàn (d=0)

    # Tính cumulative expected Hamming loss khi predict d nhãn đầu
    cumulative_loss = 0.0
    for d in range(1, K + 1):
        k = sorted_idx[d - 1]
        cumulative_loss += s[k]
        total = cumulative_loss + g(K - d)
        if total < best_expected:
            best_expected = total
            best_d = d

    # Xây dựng prediction vector
    # D(ŷ) = {d nhãn có s_k nhỏ nhất}
    pred = np.full(K, np.nan)  # NaN = ⊥ (abstain)
    decided_indices = sorted_idx[:best_d]
    pred[decided_indices] = best_pred[decided_indices]

    return pred


# ----------------------------------------------------------
# 2.2 Subset 0/1 Loss BOP
# Section 6, Proposition 3
# ----------------------------------------------------------

def bop_subset01(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP cho Generalized Subset 0/1 Loss (Equation 48).

    Subset 0/1 là non-decomposable loss:
        ℓ_S(y, ŷ) = [y ≠ ŷ]  (sai 1 nhãn = sai toàn bộ)

    Cần giả định CLI (Conditional Label Independence, Equation 15):
        p(y|x) = Π_k p_k^{y_k} * (1-p_k)^{1-y_k}

    Dưới CLI, BOP cho subset 0/1 với abstention:
        ŷ_k^d = argmax_{ŷ_k ∈ {0,1}} p_k^{ŷ_k} * (1-p_k)^{1-ŷ_k}
             = round(p_k)  — predict nhãn có xác suất cao hơn

    Key insight (Proposition 3): BOP là uncertainty-aligned —
    abstain những nhãn gần 0.5 nhất (uncertainty u_k = 2*min(p_k, 1-p_k) cao nhất).
    Đây giống hệt Hamming, chỉ khác cách tính expected loss.

    Expected loss khi predict d nhãn có uncertainty nhỏ nhất:
        E[ℓ_S] = 1 - Π_{k in D} max(p_k, 1-p_k)

    Độ phức tạp: O(K log K)
    """
    K = len(probs)

    # u_k = 2*min(p_k, 1-p_k) — degree of uncertainty (Equation 24)
    uncertainty = 2 * np.minimum(probs, 1 - probs)

    # Sắp xếp tăng dần theo uncertainty → nhãn tự tin nhất trước
    sorted_idx = np.argsort(uncertainty)

    # max(p_k, 1-p_k) = xác suất của dự đoán tốt nhất cho nhãn k
    max_probs = np.maximum(probs, 1 - probs)

    best_pred = (probs >= 0.5).astype(float)
    best_d = 0
    best_expected = g(K)  # abstain hoàn toàn

    # Tích lũy product của max_probs
    # E[ℓ_S | predict d nhãn] = 1 - Π_{k in D_d} max(p_k, 1-p_k)
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
# 2.3 F-measure BOP
# Section 7.3, Proposition 6, Algorithm 2
# ----------------------------------------------------------

def _compute_Q_matrix(probs_sorted: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Q(l, l1) — Lemma 4, Equation 57.

    Q(l, l1) = P(Σ_{k=1}^{l} y_{π(k)} = l1 | x)

    Đây là xác suất có đúng l1 nhãn dương trong l nhãn đầu tiên
    (sau khi đã sắp xếp theo p_k giảm dần).

    Tính bằng dynamic programming:
        Q(l, l1) = p_{π(l)} * Q(l-1, l1-1) + (1-p_{π(l)}) * Q(l-1, l1)

    Độ phức tạp: O(K²)
    Shape output: (K+1, K+2)  — index l từ 0..K, l1 từ -1..K+1 (offset +1)
    """
    K = len(probs_sorted)
    # Q[l, l1+1] để tránh index âm (l1 có thể = -1 trong init)
    Q = np.zeros((K + 1, K + 2))

    # Base case: Q(0, 0) = 1 (không có nhãn nào, không có positive nào)
    Q[0, 0 + 1] = 1.0

    for l in range(1, K + 1):
        p = probs_sorted[l - 1]
        for l1 in range(0, l + 1):
            # Q(l, l1) = p_l * Q(l-1, l1-1) + (1-p_l) * Q(l-1, l1)
            q_prev_pos = Q[l - 1, l1 - 1 + 1] if l1 > 0 else 0.0
            q_prev_neg = Q[l - 1, l1 + 1]
            Q[l, l1 + 1] = p * q_prev_pos + (1 - p) * q_prev_neg

    return Q


def _compute_P_matrix(probs_sorted: np.ndarray) -> np.ndarray:
    """
    Tính ma trận P(r', r'1) — Lemma 4, Equation 58.

    P(r', r'1) = P(Σ_{k=r}^{K} y_{π(k)} = r'1 | x)
    với r' = K+1-r (số nhãn ở "đuôi" sau vị trí r)

    Tương tự Q nhưng tính từ cuối danh sách (nhãn có p nhỏ nhất).
    Dùng cho phần bên phải của decision set ⟪l,r⟫.
    """
    K = len(probs_sorted)
    P = np.zeros((K + 1, K + 2))
    P[0, 0 + 1] = 1.0

    for rp in range(1, K + 1):
        # Nhãn từ cuối lên: index K-rp trong mảng đã sort giảm dần
        p = probs_sorted[K - rp]
        for rp1 in range(0, rp + 1):
            q_prev_pos = P[rp - 1, rp1 - 1 + 1] if rp1 > 0 else 0.0
            q_prev_neg = P[rp - 1, rp1 + 1]
            P[rp, rp1 + 1] = p * q_prev_pos + (1 - p) * q_prev_neg

    return P


def bop_fmeasure(probs: np.ndarray, g: Callable[[int], float],
                 beta: float = 1.0) -> np.ndarray:
    """
    BOP cho Generalized F_beta measure (Algorithm 2, Proposition 6).

    F_beta = (1+β²) * tp / [(1+β²)*tp + β²*fn + fp]

    Key results (Proposition 5, Lemma 3):
    - F_beta là semi-uncertainty-aligned
    - BOP có dạng decision set ⟪l,r⟫ — predict l nhãn đầu (p cao)
      và r' nhãn cuối (p thấp), abstain phần giữa
    - Tìm bằng dynamic programming, độ phức tạp O(K³)

    Decision set ⟪l,r⟫ = {1,...,l} ∪ {r,...,K}
    (indices sau khi sort theo p giảm dần)
    Nhãn 1..l predict 1, nhãn r..K predict 0, phần giữa abstain.

    Args:
        probs: marginal probabilities p_k, shape (K,)
        g: penalty function g(a) với a = số nhãn abstain
        beta: tham số F_beta (default 1.0 = F1)
    """
    K = len(probs)
    beta_sq = beta ** 2
    beta_prime = 1.0 + 1.0 / beta_sq  # = (1+β²)/β² ... wait
    # Công thức trong paper: β' = 1 + β^{-2} (Appendix, Proof of Prop 6)
    # F_beta = β' * Σ l1*Q(l,l1)*S(l,l1) với S là hàm của P

    # Sắp xếp nhãn giảm dần theo p_k (Remark 3, Proposition 6)
    sorted_idx = np.argsort(-probs)
    probs_sorted = probs[sorted_idx]

    # Tính Q và P matrices (Lemma 4)
    Q = _compute_Q_matrix(probs_sorted)
    P = _compute_P_matrix(probs_sorted)

    best_val = -np.inf
    best_l, best_r = 0, K + 1

    # Xét trường hợp d=0: abstain hoàn toàn
    # F(0, K+1) = 0 - g(K) (không predict gì = F=0)
    val_empty = 0.0 - g(K)
    if val_empty > best_val:
        best_val = val_empty
        best_l, best_r = 0, K + 1

    # Duyệt tất cả cặp (l, r) với 0 <= l < r <= K+1
    # (Algorithm 2 — O(K³) nhờ cập nhật S incremental)
    for l in range(1, K + 1):
        # S(l, l1) ban đầu = 1 / (l*β^{-2} + l1) cho r=K+1 (không có phần đuôi)
        # Khi r giảm dần từ K+1 về l+1, update S theo công thức recursion

        # Initialize S cho r = K+1 (không có nhãn nào ở đuôi → P(0,0)=1)
        S = np.zeros(K + 2)
        for l1 in range(0, l + 1):
            denom = l * (1.0 / beta_sq) + l1
            if denom > 1e-10:
                S[l1] = 1.0 / denom  # S(l, l1, r'=0) = 1/(l/β² + l1)

        # F_beta(l, K+1): predict l nhãn, không có đuôi
        # = β' * Σ_{l1} l1 * Q(l,l1) * S(l,l1)
        fb = beta_prime * sum(
            l1 * Q[l, l1 + 1] * S[l1]
            for l1 in range(0, l + 1)
        ) - g(K - l)
        if fb > best_val:
            best_val = fb
            best_l, best_r = l, K + 1

        # Update S khi thêm nhãn vào đuôi (r giảm từ K về l+1)
        for r in range(K, l, -1):
            rp = K + 1 - r  # r' = số nhãn ở đuôi
            pr = probs_sorted[r - 1]

            # Recursion từ paper (Proof of Proposition 6):
            # S(l, l1, r') = pr * S(l, l1, r'-1)[shifted] + (1-pr) * S(l, l1, r'-1)
            # Simplified implementation:
            new_S = np.zeros(K + 2)
            for l1 in range(0, l + 1):
                total = 0.0
                for rp1 in range(0, rp + 1):
                    denom = l * (1.0 / beta_sq) + l1 + rp1
                    if denom > 1e-10:
                        total += P[rp, rp1 + 1] / denom
                if total > 0:
                    new_S[l1] = total
            S = new_S

            fb = beta_prime * sum(
                l1 * Q[l, l1 + 1] * S[l1]
                for l1 in range(0, l + 1)
            ) - g(r - l - 1)
            if fb > best_val:
                best_val = fb
                best_l, best_r = l, r

    # Xây dựng prediction từ (best_l, best_r)
    # predict 1 cho top-l nhãn, 0 cho bottom nhãn từ best_r..K, ⊥ cho giữa
    pred = np.full(K, np.nan)
    for i in range(best_l):
        orig_idx = sorted_idx[i]
        pred[orig_idx] = 1.0
    for i in range(best_r - 1, K):
        orig_idx = sorted_idx[i]
        pred[orig_idx] = 0.0

    return pred


# ----------------------------------------------------------
# 2.4 Jaccard Measure BOP
# Section 7.3, Proposition 7, Algorithm 3
# ----------------------------------------------------------

def bop_jaccard(probs: np.ndarray, g: Callable[[int], float]) -> np.ndarray:
    """
    BOP cho Generalized Jaccard measure (Algorithm 3, Proposition 7).

    Jaccard = tp / (tp + fn + fp)

    Tương tự F-measure nhưng đơn giản hơn vì không có β.
    Semi-uncertainty-aligned, decision set dạng ⟪l,r⟫.
    Độ phức tạp: O(K³)

    Công thức key (Proof of Proposition 7):
        S_Jac(l, r') = p_r * S_Jac(l+1, r'-1) + (1-p_r) * S_Jac(l, r'-1)
    với boundary: S_Jac(l, 0) = 1/l
    """
    K = len(probs)
    sorted_idx = np.argsort(-probs)
    probs_sorted = probs[sorted_idx]

    Q = _compute_Q_matrix(probs_sorted)

    best_val = -np.inf
    best_l, best_r = 0, K + 1

    # d=0: abstain hoàn toàn
    if -g(K) > best_val:
        best_val = -g(K)
        best_l, best_r = 0, K + 1

    for l in range(1, K + 1):
        # Initialize S_Jac cho r'=0 (không có đuôi)
        S = np.zeros(K + 2)
        for i in range(l, K + 1):
            S[i] = 1.0 / i if i > 0 else 0.0

        # F_Jac(l, K+1)
        fj = S[l] * sum(l1 * Q[l, l1 + 1] for l1 in range(0, l + 1)) - g(K - l)
        if fj > best_val:
            best_val = fj
            best_l, best_r = l, K + 1

        # Update S khi r giảm
        for r in range(K, l, -1):
            pr = probs_sorted[r - 1]
            rp = K + 1 - r

            # S_Jac(l, r') = p_r * S_Jac(l+1, r'-1) + (1-p_r) * S_Jac(l, r'-1)
            new_S = np.zeros(K + 2)
            for i in range(l + rp, K + 1):
                # Đây là S(l, r') tại index i = l + r'_1
                new_S[i] = pr * S[i] + (1 - pr) * (S[i - 1] if i > 0 else 0.0)
            S = new_S

            fj = S[l + rp] * sum(
                l1 * Q[l, l1 + 1] for l1 in range(0, l + 1)
            ) - g(r - l - 1)
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
# PHẦN 3: LOSS FUNCTIONS (để evaluate)
# Section 2.2
# ============================================================

ABSTAIN = np.nan  # ký hiệu ⊥


def _get_decided(pred: np.ndarray) -> np.ndarray:
    """Trả về mask của các nhãn đã dự đoán (không abstain)."""
    return ~np.isnan(pred)


def loss_hamming(y_true: np.ndarray, pred: np.ndarray) -> float:
    """
    Generalized Hamming loss với abstention (Equation 7 + 29).
    Chỉ tính loss trên phần đã predict, không tính abstain.
    Abstention sẽ được penalize riêng bởi g(|A|).
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 0.0
    return float(np.sum(y_true[decided] != pred[decided]))


def loss_subset01(y_true: np.ndarray, pred: np.ndarray) -> float:
    """
    Generalized Subset 0/1 loss (Equation 8 + 29).
    = 0 nếu tất cả nhãn đã predict đều đúng, 1 nếu có nhãn sai.
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 0.0
    return float(not np.all(y_true[decided] == pred[decided]))


def loss_fmeasure(y_true: np.ndarray, pred: np.ndarray,
                  beta: float = 1.0) -> float:
    """
    Generalized F_beta loss = 1 - F_beta(y_D, ŷ_D) (Equation 9 + 51).
    Dùng 1-F vì F là accuracy measure (higher = better).
    """
    decided = _get_decided(pred)
    if not np.any(decided):
        return 1.0  # không predict gì = F=0, loss=1

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
    Đếm số cặp nhãn bị xếp hạng sai:
        λ_i tốt hơn λ_j nhưng y_i=1, y_j=0 (hoặc ngược lại)

    pred_ranking: mảng rank (0 = highest rank), NaN = abstain
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
            # Nếu y_i=1 và y_j=0, rank_i phải < rank_j (i ranked higher)
            if yt[i] == 1 and yt[j] == 0:
                if ranks[i] > ranks[j]:  # wrong order
                    loss += 1.0
            elif yt[i] == 0 and yt[j] == 1:
                if ranks[i] < ranks[j]:  # wrong order
                    loss += 1.0
    return loss


def abstention_size(pred: np.ndarray) -> float:
    """
    Tỷ lệ nhãn abstain: |A(ŷ)| / K
    Section 9.1.4 — comparison criterion.
    """
    return float(np.sum(np.isnan(pred))) / len(pred)


# ============================================================
# PHẦN 4: BASELINES
# Section 9.1.4
# ============================================================

def predict_mlc(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    MLC thông thường — không abstain.
    Baseline: predict 1 nếu p_k >= threshold, 0 ngược lại.
    """
    return (probs >= threshold).astype(float)


def predict_full_abstain(K: int) -> np.ndarray:
    """
    Full abstention — abstain hết mọi nhãn.
    Baseline ABS trong paper.
    """
    return np.full(K, np.nan)
