# 📄 Multilabel Classification with Partial Abstention
### Bayes-Optimal Prediction under Label Independence
> Nguyen & Hüllermeier — JAIR 2021

---

## 📋 Mục lục

1. [Ý tưởng chính](#1-ý-tưởng-chính)
2. [Bài toán & Ký hiệu](#2-bài-toán--ký-hiệu)
3. [Framework toán học](#3-framework-toán-học)
4. [Các hàm Loss & Thuật toán BOP](#4-các-hàm-loss--thuật-toán-bop)
5. [Lộ trình Implementation](#5-lộ-trình-implementation)
6. [Experimental Setup](#6-experimental-setup)
7. [Bảng tra cứu công thức](#7-bảng-tra-cứu-công-thức)
8. [Lưu ý thực tế](#8-lưu-ý-thực-tế)

---

## 1. Ý tưởng chính

### Vấn đề
Phân loại đa nhãn (MLC) thông thường **bắt buộc** mô hình dự đoán tất cả K nhãn, dù có chắc chắn hay không.

### Giải pháp
Cho phép mô hình **từ chối trả lời (abstain)** trên những nhãn không chắc chắn:

```
Thay vì:  ŷ ∈ {0, 1}ᴷ         (bắt buộc dự đoán hết)
Thành:    ŷ ∈ {0, ⊥, 1}ᴷ      (có thể từ chối)
```

### Ví dụ thực tế (Ngân hàng)
```
Nhãn: {rửa tiền, trốn thuế, gian lận thẻ, tài trợ khủng bố}

Truyền thống:   [1,  0,   1,   0]       ← bắt buộc đoán hết
Với abstention: [1,  ⊥,   1,   ⊥]       ← từ chối khi không chắc
                         ↑              ← chuyển cho chuyên gia xem xét
```

### Tại sao quan trọng?
- **An toàn**: Sai trong y tế, tài chính, pháp lý → hậu quả nghiêm trọng
- **Trung thực**: Mô hình thừa nhận giới hạn của mình
- **Hiệu quả**: Tập trung nguồn lực vào phần không chắc chắn

---

## 2. Bài toán & Ký hiệu

### Bảng ký hiệu

| Ký hiệu | Ý nghĩa | Ví dụ |
|---------|---------|-------|
| `K` | Số nhãn | 10 |
| `𝒙` | Vector đặc trưng (input) | [0.3, 1.2, ...] |
| `𝒚 = (y₁,...,yₖ)` | Nhãn thật | (1, 0, 1, 0, ...) |
| `ŷ = (ŷ₁,...,ŷₖ)` | Nhãn dự đoán | (1, ⊥, 1, 0, ...) |
| `pₖ = p(yₖ=1\|𝒙)` | Xác suất nhãn k xuất hiện | 0.85 |
| `D(ŷ)` | Tập nhãn được dự đoán | {1, 3, 4} |
| `A(ŷ)` | Tập nhãn bị từ chối | {2, 5} |
| `ℓ(𝒚, ŷ)` | Hàm loss gốc | Hamming, F1... |
| `L(𝒚, ŷ)` | Hàm loss mở rộng (có abstention) | - |
| `g(\|A\|)` | Hàm phạt khi abstain | a·c |
| `BOP` | Bayes-Optimal Prediction | ŷ* tối ưu |
| `CLI` | Conditional Label Independence | Giả định độc lập |

### Định nghĩa tập quyết định
```
D(ŷ) = {k | ŷₖ ∈ {0, 1}}      ← Các nhãn được dự đoán
A(ŷ) = {k | ŷₖ = ⊥}           ← Các nhãn bị từ chối

Luôn đúng: D(ŷ) ∪ A(ŷ) = {1, ..., K}
```

---

## 3. Framework toán học

### 3.1 Hàm Loss tổng quát

**Công thức cốt lõi:**
```
L(𝒚, ŷ) = ℓ(𝒚_D, ŷ_D)  +  g(|A(ŷ)|)
           ───────────────    ──────────
           Loss trên phần     Phạt vì
           đã dự đoán         từ chối
```

**Ý nghĩa từng phần:**
- `ℓ(𝒚_D, ŷ_D)`: Chỉ tính loss trên phần **được dự đoán**
- `g(|A(ŷ)|)`: Phạt vì **không dự đoán** |A| nhãn
- Khi không abstain: `L = ℓ` (giống loss gốc)

### 3.2 Hai dạng hàm phạt

#### SEP — Phạt tuyến tính (Separate)
```
g₁(a) = a · c
```
| Đặc điểm | Giải thích |
|-----------|-----------|
| **Tuyến tính** | Mỗi nhãn abstain phạt đúng `c` |
| **Đơn giản** | Dễ giải thích |
| **Khi dùng** | Mỗi nhãn có tầm quan trọng như nhau |

#### PAR — Phạt lõm (Parsimonious)
```
g₂(a) = (a · K · c) / (K + a)
```
| Đặc điểm | Giải thích |
|-----------|-----------|
| **Concave** | Abstain thêm càng về sau càng ít tốn kém |
| **Thực tế hơn** | Nhiều ứng dụng chấp nhận abstain hàng loạt |
| **Khi dùng** | Abstain batch được chấp nhận |

**So sánh hình dạng:**
```
Cost
 |          SEP (thẳng)
 |        /
 |      /    PAR (cong)
 |    / ___-----------
 |  /__/
 |/
 +-------------------------> Số nhãn abstain
 0                          K
```

### 3.3 Tìm Bayes-Optimal Prediction (BOP)

**Mục tiêu:** Tìm dự đoán tối thiểu hóa kỳ vọng loss:
```
ŷ* = argmin_{ŷ ∈ Ω*} E[L(𝒚, ŷ)]
   = argmin_{ŷ ∈ Ω*} Σ_{𝒚∈𝒴} L(𝒚, ŷ) · p(𝒚|𝒙)
```

**Vấn đề tính toán:**
```
Ω* = {0, ⊥, 1}ᴷ  →  3ᴷ ứng viên
K = 20            →  3.5 tỷ ứng viên  ← KHÔNG thể brute-force!
```

**Giải pháp:** Khai thác cấu trúc của BOP cho từng loại loss → giảm xuống O(K) hoặc O(K²) ứng viên.

### 3.4 Giả định CLI (Conditional Label Independence)

**Định nghĩa:**
```
p(𝒚|𝒙) = ∏ₖ pₖ^{yₖ} · (1-pₖ)^{1-yₖ}
```

**Ý nghĩa:** Các nhãn **độc lập với nhau** khi đã biết đặc trưng 𝒙.

**Khi nào cần:** Bắt buộc với các loss **non-decomposable** (Rank, Subset 0/1, F-measure).

**Kiểm tra thực tế:**
```python
# Tính mutual information giữa các cặp nhãn
# Nếu MI cao → labels phụ thuộc → CLI bị vi phạm
from sklearn.metrics import mutual_info_score
MI = mutual_info_score(y[:, i], y[:, j])
```

### 3.5 Độ không chắc chắn của nhãn

```
uₖ = 2 · min(pₖ, 1-pₖ)
```

| pₖ | uₖ | Ý nghĩa |
|----|----|---------| 
| 0.5 | 1.0 | **Tối đa** không chắc chắn |
| 0.9 | 0.2 | Khá chắc chắn (label có) |
| 0.1 | 0.2 | Khá chắc chắn (label không) |
| 0.7 | 0.6 | Không chắc vừa |

### 3.6 Cấu trúc BOP

#### Uncertainty-Aligned
```
Nếu ŷᵢ = ⊥  và  ŷⱼ ≠ ⊥  thì  uᵢ ≥ uⱼ
```
→ **Luôn abstain nhãn bất định nhất trước**

#### Semi-Uncertainty-Aligned
Sau khi sắp xếp nhãn theo pₖ giảm dần:
```
D(ŷ) = ⟪l, r⟫ = {1,...,l} ∪ {r,...,K}

Trực quan:
[p cao ──── dự đoán 1] [⊥⊥⊥⊥ abstain ⊥⊥⊥⊥] [dự đoán 0 ──── p thấp]
         l                    giữa                     r
```
→ **Abstain ở vùng giữa** (xác suất không rõ ràng)

---

## 4. Các hàm Loss & Thuật toán BOP

### 4.1 Hamming Loss ✅ (Decomposable — Dễ nhất)

#### Định nghĩa
```
ℓ_H(𝒚, ŷ) = Σₖ 𝟙[yₖ ≠ ŷₖ]
```
Đếm số nhãn dự đoán sai.

#### Tính chất
- **Decomposable**: Tính từng nhãn độc lập → **không cần CLI**
- **Symmetric**: Sai kiểu nào cũng phạt như nhau
- **BOP structure**: Uncertainty-aligned

#### Thuật toán BOP — Linear Penalty O(K)

```
INPUT:
  p = (p₁, ..., pₖ)   # Marginal probabilities
  c                    # Abstention cost

ALGORITHM:
  For each k in [1..K]:
    uₖ = 2 · min(pₖ, 1-pₖ)
    
    if uₖ ≤ c:
      DECIDE: ŷₖ = 1 if pₖ > 0.5 else ŷₖ = 0
    else:
      ABSTAIN: ŷₖ = ⊥

OUTPUT: ŷ = (ŷ₁, ..., ŷₖ)
```

**Tại sao ngưỡng là c?**
```
Chi phí abstain label k   = c
Chi phí dự đoán label k   = min(pₖ, 1-pₖ)  ← xác suất bị sai

Abstain khi: c < min(pₖ, 1-pₖ)  ↔  uₖ > c
```

#### Thuật toán BOP — General Penalty O(K log K)

```
INPUT:
  p = (p₁, ..., pₖ)    # Marginal probabilities
  g(·)                  # Penalty function

ALGORITHM:
  1. Tính sₖ = min(pₖ, 1-pₖ) cho mỗi k
  2. Sắp xếp nhãn: s_{π(1)} ≤ s_{π(2)} ≤ ... ≤ s_{π(K)}
  3. For d = 0 to K:
       D_d = {π(1), ..., π(d)}          # d nhãn chắc chắn nhất
       E_d = Σ_{k∈D_d} sₖ + g(K-d)     # Expected total cost
  4. d* = argmin_d E_d
  5. Predict trên D_{d*}, abstain trên phần còn lại

OUTPUT: ŷ với D(ŷ) = D_{d*}
```

---

### 4.2 Rank Loss ⚡ (Non-Decomposable — Cần CLI)

#### Định nghĩa
```
ℓ_R(𝒚, π) = Σ_{i<j: yᵢ=1, yⱼ=0} 𝟙[π⁻¹(i) > π⁻¹(j)]
```
Đếm số cặp nhãn bị xếp sai thứ tự (nhãn có-nhưng-xếp-sau nhãn không-có).

**Ví dụ:**
```
True:    Núi=1,  Biển=1,  Sa mạc=0
Tốt:     Biển > Núi > Sa mạc        → 0 lỗi
Xấu:     Sa mạc > Biển > Núi        → 2 lỗi
```

#### Tính chất
- **Non-decomposable**: Phụ thuộc vào thứ tự tương đối giữa các nhãn
- **Cần CLI** để tính hiệu quả
- **BOP structure**: Semi-uncertainty-aligned

#### Thuật toán BOP O(K log K) — Algorithm 1

```
INPUT:
  p = (p₁, ..., pₖ)    # Marginal probabilities
  g(·)                  # Penalty function

STEP 1 — Sắp xếp:
  Sắp xếp p giảm dần: p_{π(1)} ≥ p_{π(2)} ≥ ... ≥ p_{π(K)}
  → Đây là ranking tối ưu khi không abstain

STEP 2 — Khởi tạo:
  D₀ = ∅,    E₀ = g(K)
  D₂ = ⟪1, K⟫,  E₂ = E[ℓ_R(𝒚, π_{⟪1,K⟫})] + g(K-2)
  l = 1,  r = K

STEP 3 — Mở rộng greedy từ d=3 đến K:
  Tại mỗi bước, xét hai lựa chọn:
    Mở rộng trái:  K_l = ⟪l+1, r⟫
    Mở rộng phải:  K_r = ⟪l, r-1⟫
  
  Chọn lựa có expected loss thấp hơn:
    if E[ℓ_R(K_l)] < E[ℓ_R(K_r)]:
      D_d = K_l,  l = l+1
    else:
      D_d = K_r,  r = r-1
  
  E_d = E[ℓ_R(D_d)] + g(K-d)

STEP 4 — Chọn d tối ưu:
  d* = argmin_{d∈{0,2,...,K}} E_d

OUTPUT: Ranking π_{D_{d*}}
```

**Tính Expected Loss dưới CLI:**
```
E[ℓ_R(𝒚, π_D)] = Σ_{i<j trong D} (1-pᵢ)·pⱼ
```

**Tại sao greedy hoạt động? (Lemma 2)**
```
Nếu D_d = ⟪l, r⟫ là lựa chọn tối ưu cho d nhãn,
thì ít nhất một trong ⟪l+1, r⟫ hoặc ⟪l, r-1⟫
là tối ưu cho d+1 nhãn.
→ Chỉ cần check 2 ứng viên mỗi bước!
```

---

### 4.3 Subset 0/1 Loss (Non-Decomposable — Cần CLI)

#### Định nghĩa
```
ℓ_S(𝒚, ŷ) = 𝟙[𝒚 ≠ ŷ]
```
**All-or-nothing**: Đúng hoàn toàn mới không phạt, sai bất kỳ nhãn nào = sai toàn bộ.

#### Tính chất
- **Strict**: Yêu cầu tất cả nhãn đều đúng
- **BOP structure**: Uncertainty-aligned (giống Hamming)
- **BOP formula**: Chọn `d*` nhãn chắc chắn nhất:
  ```
  E_d = [1 - ∏_{k∈D_d} max(pₖ, 1-pₖ)] + g(K-d)
  ```

#### Thuật toán BOP O(K log K)

```
INPUT:
  p = (p₁, ..., pₖ), g(·)

ALGORITHM:
  1. Sắp xếp theo uₖ tăng dần (chắc chắn nhất trước)
  2. For d = 0 to K:
       D_d = d nhãn chắc chắn nhất
       E_d = (1 - ∏_{k∈D_d} max(pₖ, 1-pₖ)) + g(K-d)
  3. d* = argmin E_d
  4. Predict trên D_{d*}
```

---

### 4.4 F-measure (Non-Decomposable — Cần CLI)

#### Định nghĩa
```
F_β(𝒚, ŷ) = (1+β²)·tp / [(1+β²)·tp + β²·fn + fp]
```

**Các thành phần:**
```
tp = Σₖ yₖ·ŷₖ          (Dự đoán đúng nhãn CÓ)
fn = Σₖ yₖ·(1-ŷₖ)      (Bỏ sót nhãn CÓ)
fp = Σₖ (1-yₖ)·ŷₖ      (Dự đoán sai nhãn KHÔNG có)
```

**F₁ (β=1):** Harmonic mean của precision và recall:
```
F₁ = 2·tp / (2·tp + fn + fp)
```

#### Tính chất
- **Imbalance-aware**: Tốt khi nhãn dương hiếm
- **BOP structure**: Semi-uncertainty-aligned `D = ⟪l, r⟫`
- **Phức tạp hơn**: Cần tính kỳ vọng qua DP

#### Thuật toán BOP O(K³) — Algorithm 2

**Preprocessing O(K²):**
```
Tính ma trận Q(l, l₁):
  Q(l, l₁) = p(có đúng l₁ nhãn dương trong l nhãn đầu)
  
Dùng dynamic programming:
  Q(l, l₁) = p_{π(l)} · Q(l-1, l₁-1) + (1-p_{π(l)}) · Q(l-1, l₁)
```

**Main DP O(K³):**
```
For l = 0 to K:
  For r = K+1 downto l+1:
    Tính E[F_β | D = ⟪l, r⟫]  # Dùng Q và S matrices
    Lưu F_β(l, r)

Output: ⟪l*, r*⟫ = argmax F_β(l, r) - g(r-l-1)
```

---

### 4.5 Jaccard Index (Non-Decomposable — Cần CLI)

#### Định nghĩa
```
f_Jac(𝒚, ŷ) = tp / (tp + fn + fp)
            = |𝒚 ∩ ŷ| / |𝒚 ∪ ŷ|
```
Intersection over Union (IoU) giữa nhãn thật và nhãn dự đoán.

#### Tính chất
- **Tương tự F₁** nhưng bỏ qua true negatives
- **BOP structure**: Semi-uncertainty-aligned
- **Complexity**: O(K³)

---

### Bảng tổng hợp các thuật toán

| Loss | Type | Cần CLI? | BOP Structure | Complexity |
|------|------|----------|---------------|------------|
| Hamming | Decomposable | ❌ Không | Uncertainty-aligned | **O(K)** hoặc O(K log K) |
| Rank | Non-decomposable | ✅ Cần | Semi-uncertainty-aligned | **O(K log K)** |
| Subset 0/1 | Non-decomposable | ✅ Cần | Uncertainty-aligned | **O(K log K)** |
| F-measure | Non-decomposable | ✅ Cần | Semi-uncertainty-aligned | **O(K³)** |
| Jaccard | Non-decomposable | ✅ Cần | Semi-uncertainty-aligned | **O(K³)** |

---

## 5. Lộ trình Implementation

### Phase 1 — Nền tảng

#### 1.1 Cấu trúc dữ liệu
```
PartialPrediction:
  - predictions: Dict[int → {0, 1}]    # Nhãn được dự đoán
  - abstentions: Set[int]              # Nhãn bị từ chối
  - K: int                             # Tổng số nhãn
```

#### 1.2 Probabilistic Model
```
Yêu cầu đầu ra:
  predict_marginals(X) → array (N, K)
    pₖ = p(yₖ=1|x) cho mỗi nhãn
  
Chú ý: Cần calibrated probabilities!
  → Dùng Platt scaling cho SVM
  → Logistic Regression tự động calibrated
```

### Phase 2 — Loss functions

#### 2.1 Standard Losses
```
Hamming:    sum(y_true != y_pred)
Rank:       count wrongly ordered pairs
Subset 0/1: int(any(y_true != y_pred))
F-measure:  2*tp / (2*tp + fn + fp)
Jaccard:    tp / (tp + fn + fp)
```

#### 2.2 Generalized Loss với Abstention
```
L(y, y_hat_partial):
  1. Tách D(ŷ) và A(ŷ) từ y_hat_partial
  2. Tính base_loss = ℓ(y[D], ŷ[D])
  3. Tính penalty = g(|A|)
  4. return base_loss + penalty
```

#### 2.3 Penalty Functions
```
SEP: g(a, c) = a * c
PAR: g(a, K, c) = (a * K * c) / (K + a)
```

### Phase 3 — BOP Algorithms

#### 3.1 Hamming BOP (bắt đầu từ đây!)
```
bop_hamming_linear(p, c):
  → O(K), không cần CLI
  → Dễ nhất, test trước

bop_hamming_general(p, g_func):
  → O(K log K)
  → Thêm flexibility
```

#### 3.2 Rank Loss BOP
```
bop_rank(p, g_func):
  → Cần CLI
  → Greedy algorithm (Algorithm 1)
  → O(K log K)
```

#### 3.3 F-measure BOP
```
bop_fmeasure(p, g_func, beta=1):
  → Cần CLI
  → Dynamic programming (Algorithm 2)
  → O(K³) — chậm với K lớn
```

### Phase 4 — Models

#### 4.1 Binary Relevance (BR)
```
BR_Abstention:
  fit(X_train, y_train):
    → Train K binary classifiers
    → Calibrate probabilities
  
  predict_with_abstention(X, loss_type, c):
    → Get marginals p
    → Apply BOP algorithm
    → Return partial predictions
```

#### 4.2 Ensemble Classifier Chains (ECC)
```
ECC_Abstention:
  fit(X_train, y_train, n_chains=50):
    → Train M classifier chains
    → Each with random label ordering
  
  predict_marginals(X):
    → Average marginals over all chains
    p̄ₖ = (1/M) Σₘ pₖ,ₘ
```

### Phase 5 — Evaluation

#### 5.1 Metrics cần track
```
Per prediction:
  - loss_on_decided: ℓ(y_D, ŷ_D)
  - abstention_rate: |A| / K
  - total_cost: L(y, ŷ)

Aggregate:
  - mean_loss vs cost_c curve
  - mean_abstention vs cost_c curve
  - gain vs baseline_loss scatter
```

#### 5.2 Protocol
```
For each dataset:
  10-fold CV
  For each c in cost_range:
    For each loss_type in [Hamming, Rank, Subset01, F1, Jaccard]:
      Compute metrics
      Compare: SEP vs PAR vs MLC baseline vs ABS baseline
```

### Phase 6 — Visualization

```
Plot 1: Loss vs Cost (như Figure 1-5 trong paper)
  X-axis: c (cost of abstention)
  Y-axis: Loss
  Lines: MLC, ABS, PAR, SEP

Plot 2: Abstention Rate vs Cost
  X-axis: c
  Y-axis: |A|/K (%)
  
Plot 3: Meta-analysis (như Figure 7)
  X-axis: MLC baseline loss
  Y-axis: Average gain from abstention
  → Expect: harder datasets → more gain
```

---

## 6. Experimental Setup

### 6.1 Datasets

| Dataset | #Instances | #Features | #Labels | Domain |
|---------|-----------|-----------|---------|--------|
| CAL500 | 502 | 68 | **174** | Âm nhạc |
| EMOTIONS | 593 | 72 | 6 | Cảm xúc âm nhạc |
| SCENE | 2,407 | 294 | 6 | Ảnh cảnh quan |
| YEAST | 2,417 | 103 | 14 | Protein function |
| MEDIAMILL | 43,907 | 120 | 101 | Video annotation |
| NUS-WIDE | 269,648 | 128 | 81 | Ảnh tags |
| Natural Scene | 2,000 | 128×128×3 | 5 | Ảnh (CNN) |

### 6.2 Models so sánh

| Model | Mô tả |
|-------|-------|
| BR+LR | Binary Relevance + Logistic Regression |
| BR+SVM | BR + SVM với Platt scaling |
| ECC+LR | Ensemble (M=50) Classifier Chains + LR |
| ECC+SVM | ECC + SVM |
| EVGG16 | Ensemble 5 VGG16 (chỉ cho ảnh) |

### 6.3 Baselines

| Baseline | Mô tả | Vai trò |
|----------|-------|---------|
| **MLC** | Dự đoán toàn bộ K nhãn | Lower bound loss |
| **ABS** | Từ chối toàn bộ K nhãn | Upper bound safety |
| **SEP** | Linear penalty g₁(a) = a·c | Method 1 |
| **PAR** | Concave penalty g₂(a) = aKc/(K+a) | Method 2 |

### 6.4 Cost ranges

| Loss | c range (SEP) | c' range (PAR) |
|------|--------------|----------------|
| Hamming | [0.05, 0.5] | [0.1, 1] |
| Rank | [0.1, 1] | [0.2, 2] |
| Subset 0/1 | [0.25/K, 2.5/K] | [0.5/K, 5/K] |
| F₁, Jaccard | [0.1/K, 1/K] | [0.2/K, 2/K] |

> **Lưu ý:** PAR c' = 2 × SEP c (vì PAR có full abstention cost gấp đôi)

### 6.5 Các thí nghiệm chính

#### Experiment 1: Loss-Abstention Trade-off (Figure 1-6)
```
Mục đích: Xem hiệu quả của partial abstention theo cost c
Setup:
  - Vary c từ thấp đến cao
  - Đo loss và abstention rate
  
Kỳ vọng:
  - c thấp → abstain nhiều → loss thấp
  - c cao → abstain ít → hội tụ về MLC
  - PAR thường tốt hơn SEP
```

#### Experiment 2: PAR vs SEP Comparison
```
Mục đích: So sánh hai loại penalty
Kỳ vọng: PAR linh hoạt hơn, trade-off tốt hơn
```

#### Experiment 3: Meta-Analysis (Figure 7)
```
Mục đích: Xem abstention có ích nhất khi nào
Setup:
  - X-axis: Loss của MLC baseline (độ khó dataset)
  - Y-axis: Average gain từ abstention

Phát hiện chính:
  Dataset càng khó → MLC loss càng cao → 
  Gain từ abstention càng lớn
  
  → Abstention hữu ích nhất khi model không chắc chắn
```

---

## 7. Bảng tra cứu công thức

### Công thức cốt lõi

| Khái niệm | Công thức |
|-----------|-----------|
| **Generalized loss** | `L(𝒚, ŷ) = ℓ(𝒚_D, ŷ_D) + g(\|A\|)` |
| **Linear penalty** | `g₁(a) = a · c` |
| **Concave penalty** | `g₂(a) = a·K·c / (K + a)` |
| **Uncertainty** | `uₖ = 2·min(pₖ, 1-pₖ)` |
| **CLI assumption** | `p(𝒚\|𝒙) = ∏ₖ pₖ^{yₖ}(1-pₖ)^{1-yₖ}` |
| **Hamming BOP** | Abstain nếu `uₖ > c` |
| **Rank expected loss** | `Σ_{i<j∈D} (1-pᵢ)·pⱼ` |
| **F₁** | `2tp / (2tp + fn + fp)` |
| **Jaccard** | `tp / (tp + fn + fp)` |

### Complexity

| Loss | Complexity | Bottleneck |
|------|-----------|-----------|
| Hamming (linear c) | **O(K)** | So sánh uₖ vs c |
| Hamming (general g) | **O(K log K)** | Sắp xếp |
| Rank | **O(K log K)** | Greedy + sort |
| Subset 0/1 | **O(K log K)** | Sort |
| F-measure | **O(K³)** | DP với Q, S matrices |
| Jaccard | **O(K³)** | DP |

---

## 8. Lưu ý thực tế

### ✅ Calibrated Probabilities là bắt buộc

```
SVM, tree-based → KHÔNG calibrated
→ Bắt buộc dùng Platt scaling hoặc isotonic regression

Logistic Regression, Neural Networks → Thường calibrated
→ Kiểm tra bằng reliability diagram
```

### ✅ Chọn c phù hợp

```
Heuristic 1: Cross-validation
  → Tìm c tối ưu trên validation set
  → Cẩn thận: Không tune trên test set!

Heuristic 2: Domain knowledge
  y tế/pháp lý:    c lớn (ưu tiên abstain)
  recommendation:  c nhỏ (chấp nhận abstain nhiều)

Heuristic 3: Target abstention rate
  → Calibrate c để đạt tỷ lệ abstain mong muốn
```

### ✅ Xử lý Label Imbalance

```
Nhãn hiếm → pₖ thường thấp → Model không chắc → Abstain nhiều
→ Có thể mất thông tin quan trọng

Giải pháp:
  1. Weighted penalty: Phạt nặng hơn khi abstain nhãn hiếm
  2. Class-weighted training
  3. Oversampling trước khi train
```

### ✅ Kiểm tra CLI assumption

```python
# Tính pairwise mutual information
for i in range(K):
    for j in range(i+1, K):
        MI = mutual_info_score(y_train[:, i], y_train[:, j])
        if MI > threshold:
            print(f"Label {i} và {j} có dependency!")
            # → CLI bị vi phạm
            # → Cần ECC thay vì BR
```

### ✅ Với K lớn (K > 100)

```
F-measure O(K³) → Rất chậm với K=174 (CAL500)

Tối ưu hóa:
  1. Early stopping trong DP
  2. Approximate với sampling
  3. Chỉ dùng Hamming/Rank (O(K log K))
```

### ✅ Pipeline Production (Ngân hàng)

```
Input: Transaction features
  ↓
Model: ECC với calibrated probabilities
  ↓  
BOP Algorithm (chọn loss phù hợp)
  ↓
Partial Prediction:
  ŷ = [1, 0, 1, ⊥, ⊥, 0, ...]
         ↑           ↑
      Confident   Uncertain
         ↓              ↓
  Auto decision    Human review
```

### ✅ Monitoring sau deployment

```
Track theo thời gian:
  - Abstention rate trend
  - Loss trên decided labels
  - Distribution của pₖ (model drift?)
  
Alert khi:
  - Abstention rate tăng đột biến → Có data drift
  - Loss tăng trên decided labels → Model suy giảm
```

---

## 📚 Tham chiếu Paper

| Nội dung | Section | Trang |
|----------|---------|-------|
| Setup tổng quan | Section 2 | 3-7 |
| Framework abstention | Section 3 | 8-12 |
| Hamming Loss | Section 4 | 12-13 |
| Rank Loss + Algorithm 1 | Section 5 | 13-16 |
| Subset 0/1 Loss | Section 6 | 16 |
| F-measure + Algorithm 2 | Section 7.3 | 18-19 |
| Jaccard + Algorithm 3 | Section 7.3 | 20 |
| Experiments | Section 9 | 21-29 |
| Proofs | Appendix A-D | 32-42 |

---

*Paper: Nguyen, V-L. & Hüllermeier, E. (2021). JAIR 72:613-665*
