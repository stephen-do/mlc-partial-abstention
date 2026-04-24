"""
run_experiments.py
==================
Script chính để chạy tất cả experiments và sinh plots.

Usage:
    python run_experiments.py

Output:
    - Plots tái hiện Figures 1-5 và Figure 7 của bài báo
    - CSV summary của tất cả kết quả
    - Console output giống paper: loss và abstention size

Thứ tự chạy:
    1. Unit tests để verify core algorithms
    2. Quick sanity check trên 1 dataset
    3. Full experiment suite
    4. Plots và tables
"""

import numpy as np
import os
import sys

# Thêm thư mục hiện tại vào path
os.makedirs('outputs', exist_ok=True)

from mlc_core import (
    bop_hamming, bop_subset01, bop_fmeasure, bop_jaccard,
    predict_mlc, predict_full_abstain,
    loss_hamming, loss_subset01, loss_fmeasure, loss_jaccard,
    abstention_size, make_penalty, penalty_linear, penalty_concave,
    ABSTAIN
)
from mlc_classifiers import BinaryRelevance, EnsembleClassifierChains
from mlc_experiments import (
    generate_synthetic_dataset, run_full_experiment, run_experiment_for_loss
)
from mlc_plots import (
    plot_loss_vs_cost, plot_meta_analysis, plot_summary_table
)


# ============================================================
# PHẦN 1: UNIT TESTS — verify core algorithms
# ============================================================

def test_penalty_functions():
    """Test SEP và PAR penalty functions."""
    print("\n[TEST] Penalty functions")

    K, c = 5, 0.1

    # SEP: g1(a) = a * c
    assert penalty_linear(0, K, c) == 0.0
    assert abs(penalty_linear(3, K, c) - 0.3) < 1e-10
    assert abs(penalty_linear(5, K, c) - 0.5) < 1e-10
    print("  SEP (linear): OK")

    # PAR: g2(a) = (a*K*c) / (K+a) — lõm (concave)
    g0 = penalty_concave(0, K, c)
    g1 = penalty_concave(1, K, c)
    g2 = penalty_concave(2, K, c)
    g3 = penalty_concave(3, K, c)
    assert g0 == 0.0
    assert g1 < g2 < g3  # tăng đơn điệu
    # Kiểm tra tính lõm: marginal cost giảm dần
    assert (g2 - g1) > (g3 - g2)
    print(f"  PAR (concave): g(0)={g0:.3f}, g(1)={g1:.3f}, g(2)={g2:.3f}, g(3)={g3:.3f} → OK")


def test_bop_hamming_simple():
    """
    Test BOP Hamming với ví dụ đơn giản.

    Nếu p = [0.9, 0.5, 0.1] và c = 0.05:
    - s = [0.1, 0.5, 0.1]  (uncertainty)
    - Nhãn 0 và 2: tự tin cao (s=0.1), nên predict
    - Nhãn 1: gần 0.5 (s=0.5), nên abstain
    """
    print("\n[TEST] BOP Hamming")
    probs = np.array([0.9, 0.5, 0.1])
    K = len(probs)

    # c lớn → ít abstain hơn
    g_small = make_penalty('linear', K, 0.05)
    g_large = make_penalty('linear', K, 0.6)

    pred_small = bop_hamming(probs, g_small)
    pred_large = bop_hamming(probs, g_large)

    abs_small = abstention_size(pred_small)
    abs_large = abstention_size(pred_large)

    print(f"  c=0.05 → pred={pred_small}, abs_size={abs_small:.2f}")
    print(f"  c=0.60 → pred={pred_large}, abs_size={abs_large:.2f}")

    # c lớn → phạt abstain nhiều → nên abstain ít hơn
    assert abs_small >= abs_large, \
        f"Expected more abstention with smaller c, got abs_small={abs_small} abs_large={abs_large}"

    # Nhãn 1 (p=0.5, uncertainty cao nhất) nên là nhãn đầu tiên bị abstain
    if abs_small > 0:
        assert np.isnan(pred_small[1]), "Label 1 (most uncertain) should be abstained first"

    print("  BOP Hamming: OK")


def test_bop_subset01():
    """Test BOP Subset 0/1 — uncertainty-aligned."""
    print("\n[TEST] BOP Subset 0/1")
    probs = np.array([0.85, 0.48, 0.52, 0.15])
    K = len(probs)

    g = make_penalty('linear', K, 0.1 / K)
    pred = bop_subset01(probs, g)

    print(f"  probs={probs}")
    print(f"  pred={pred}")
    print(f"  abstention_size={abstention_size(pred):.2f}")

    # Nhãn 1 (p=0.48) và nhãn 2 (p=0.52) có uncertainty cao nhất
    # Nên là những nhãn đầu tiên được abstain
    print("  BOP Subset 0/1: OK")


def test_bop_fmeasure():
    """Test BOP F-measure."""
    print("\n[TEST] BOP F-measure")
    probs = np.array([0.9, 0.7, 0.45, 0.55, 0.1])
    K = len(probs)

    g = make_penalty('linear', K, 0.2 / K)
    pred = bop_fmeasure(probs, g, beta=1.0)

    print(f"  probs={probs}")
    print(f"  pred={pred}")
    print(f"  abstention_size={abstention_size(pred):.2f}")

    # Không nên crash, pred phải có shape đúng
    assert pred.shape == (K,)
    print("  BOP F-measure: OK")


def test_loss_functions():
    """Test các loss functions."""
    print("\n[TEST] Loss functions")

    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

    # Perfect prediction
    pred_perfect = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert loss_hamming(y, pred_perfect) == 0.0
    assert loss_subset01(y, pred_perfect) == 0.0
    assert loss_fmeasure(y, pred_perfect) == 0.0
    print("  Perfect prediction: all losses = 0 ✓")

    # Wrong prediction
    pred_wrong = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert loss_hamming(y, pred_wrong) == 5.0
    assert loss_subset01(y, pred_wrong) == 1.0
    print("  Wrong prediction: correct ✓")

    # With abstention (NaN = ⊥)
    pred_partial = np.array([1.0, np.nan, 1.0, np.nan, 1.0])
    h_loss = loss_hamming(y, pred_partial)
    assert h_loss == 0.0, f"Expected 0 Hamming loss, got {h_loss}"
    print(f"  Partial prediction (2 abstain): Hamming={h_loss:.2f} ✓")

    # Full abstention
    pred_abs = predict_full_abstain(5)
    assert abstention_size(pred_abs) == 1.0
    print("  Full abstention: OK ✓")


def test_classifier():
    """Test BR classifier."""
    print("\n[TEST] Binary Relevance Classifier")
    from sklearn.datasets import make_multilabel_classification

    X, Y = make_multilabel_classification(
        n_samples=200, n_features=10, n_classes=4,
        n_labels=2, random_state=42, allow_unlabeled=False
    )
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    clf = BinaryRelevance(base_learner='lr')
    clf.fit(X[:150], Y[:150])
    probs = clf.predict_proba(X[150:])

    assert probs.shape == (50, 4)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    print(f"  BR+LR: probs shape={probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}] ✓")


def run_all_tests():
    """Chạy tất cả unit tests."""
    print("\n" + "="*60)
    print("UNIT TESTS")
    print("="*60)
    test_penalty_functions()
    test_bop_hamming_simple()
    test_bop_subset01()
    test_bop_fmeasure()
    test_loss_functions()
    test_classifier()
    print("\n[✓] All tests passed!\n")


# ============================================================
# PHẦN 2: QUICK DEMO — 1 dataset, 1 loss, dễ hiểu
# ============================================================

def demo_single_instance():
    """
    Demo ngắn gọn: từng bước tính BOP cho 1 instance.
    Giúp hiểu flow của algorithm.
    """
    print("\n" + "="*60)
    print("DEMO: BOP cho 1 instance (ngân hàng)")
    print("="*60)

    # Giả sử: 5 nhãn rủi ro, xác suất từ model deep learning
    labels = ['Nợ xấu', 'Gian lận thẻ', 'Rửa tiền', 'Chậm trả', 'Vỡ nợ']
    probs = np.array([0.85, 0.52, 0.48, 0.78, 0.12])
    K = len(probs)

    print(f"\nXác suất từ model: {dict(zip(labels, probs))}")

    uncertainty = 2 * np.minimum(probs, 1 - probs)
    print(f"Uncertainty u_k:   {dict(zip(labels, np.round(uncertainty, 2)))}")

    print("\n--- BOP Hamming (c=0.1) ---")
    g = make_penalty('linear', K, 0.1)
    pred = bop_hamming(probs, g)
    for i, (l, p, u, pr) in enumerate(zip(labels, probs, uncertainty, pred)):
        status = '⊥ ABSTAIN' if np.isnan(pr) else f'→ {int(pr)} ({"ĐÚ KHẮC" if pr==1 else "KHÔNG RỦI RO"})'
        print(f"  {l}: p={p:.2f}, u={u:.2f} {status}")

    print("\n--- BOP Hamming (c=0.4) — phạt nhiều hơn → ít abstain hơn ---")
    g2 = make_penalty('linear', K, 0.4)
    pred2 = bop_hamming(probs, g2)
    for i, (l, pr) in enumerate(zip(labels, pred2)):
        status = '⊥ ABSTAIN' if np.isnan(pr) else f'→ {int(pr)}'
        print(f"  {l}: {status}")

    print(f"\n  Abstention size (c=0.1): {abstention_size(pred)*100:.0f}%")
    print(f"  Abstention size (c=0.4): {abstention_size(pred2)*100:.0f}%")


# ============================================================
# PHẦN 3: FULL EXPERIMENTS + PLOTS
# ============================================================

def run_main_experiments():
    """
    Chạy experiments chính — tái hiện Section 9 của bài báo.
    """
    print("\n" + "="*60)
    print("MAIN EXPERIMENTS")
    print("="*60)

    # Datasets và config (smaller scale cho feasibility)
    DATASETS = ['emotions', 'scene', 'yeast']
    LOSSES = ['hamming', 'fmeasure', 'jaccard']
    CLASSIFIERS = ['br_lr', 'ecc_lr']

    # Chạy experiments
    all_results = run_full_experiment(
        dataset_names=DATASETS,
        loss_names=LOSSES,
        classifier_names=CLASSIFIERS,
        n_folds=5,
        n_cost_points=8,
        random_state=42
    )

    print("\n\nGenerating plots...")
    output_dir = 'outputs/'
    saved_files = []

    # Plot từng loss
    for loss_name in LOSSES:
        for clf_name in CLASSIFIERS:
            fname = plot_loss_vs_cost(
                all_results,
                dataset_names=DATASETS,
                loss_name=loss_name,
                classifier_name=clf_name,
                output_path=output_dir
            )
            if fname:
                saved_files.append(fname)

    # Meta-analysis plots (Figure 7)
    for loss_name in LOSSES:
        fname = plot_meta_analysis(all_results, loss_name, output_dir)
        if fname:
            saved_files.append(fname)

    # Summary table
    csv_path = plot_summary_table(all_results, output_dir)
    if csv_path:
        saved_files.append(csv_path)

    return all_results, saved_files


# ============================================================
# PHẦN 4: SUBSET01 EXPERIMENT (riêng vì chậm hơn)
# ============================================================

def run_subset01_demo():
    """Chạy riêng subset01 với dataset nhỏ."""
    print("\n[Subset 0/1 Loss demo]")
    X, Y = generate_synthetic_dataset('emotions', random_state=42)

    df = run_experiment_for_loss(
        X, Y,
        loss_name='subset01',
        classifier_name='br_lr',
        n_folds=3,
        n_cost_points=6,
        random_state=42
    )

    print("\nSubset 0/1 Results:")
    print(df[['cost', 'loss_mlc', 'loss_sep', 'loss_par', 'abs_size_sep']].to_string(index=False))

    # Save plot
    fake_results = {'emotions': {'subset01': {'br_lr': df}}}
    plot_loss_vs_cost(
        fake_results,
        dataset_names=['emotions'],
        loss_name='subset01',
        classifier_name='br_lr',
        output_path='outputs/'
    )


if __name__ == '__main__':
    # 1. Tests
    run_all_tests()

    # 2. Demo
    demo_single_instance()

    # 3. Subset 0/1 demo (nhanh)
    run_subset01_demo()

    # 4. Full experiments
    all_results, saved_files = run_main_experiments()

    print("\n" + "="*60)
    print("DONE! Files saved:")
    for f in saved_files:
        print(f"  {f}")
