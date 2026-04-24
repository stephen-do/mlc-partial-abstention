"""
mlc_experiments.py
==================
Tái hiện experiments từ Section 9 của bài báo.

Do không thể download MULAN datasets (network restricted),
ta tạo synthetic datasets mô phỏng các đặc điểm của chúng,
và thêm experiment với sklearn's built-in datasets.

Section 9 của bài báo dùng:
    - 6 benchmark datasets từ MULAN (Table 3)
    - 10-fold cross-validation
    - Classifiers: BR+LR, BR+SVM, ECC+LR, ECC+SVM
    - Loss functions: Hamming, Rank, Subset 0/1, F1, Jaccard
    - Penalty functions: SEP (linear), PAR (concave)
    - Cost ranges khác nhau cho từng loss (Section 9.1.4)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_multilabel_classification
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from mlc_core import (
    bop_hamming, bop_subset01, bop_fmeasure, bop_jaccard,
    predict_mlc, predict_full_abstain,
    loss_hamming, loss_subset01, loss_fmeasure, loss_jaccard,
    abstention_size, make_penalty, ABSTAIN
)
from mlc_classifiers import BinaryRelevance, EnsembleClassifierChains


# ============================================================
# DATASET GENERATION
# Section 9, Table 3
# ============================================================

def generate_synthetic_dataset(name: str, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo synthetic datasets mô phỏng đặc điểm của MULAN datasets.

    Các dataset trong Table 3:
        CAL500:    502 instances, 68 features, 174 labels
        EMOTIONS:  593 instances, 72 features, 6 labels
        SCENE:     2407 instances, 294 features, 6 labels
        YEAST:     2417 instances, 103 features, 14 labels
        MEDIAMILL: 43907 instances, 120 features, 101 labels
        NUS-WIDE:  269648 instances, 128 features, 81 labels

    Ta dùng tham số gần đúng để mô phỏng.
    """
    configs = {
        'emotions': {
            'n_samples': 593, 'n_features': 72, 'n_classes': 6,
            'n_labels': 2, 'n_informative': 6
        },
        'scene': {
            'n_samples': 800, 'n_features': 50, 'n_classes': 6,
            'n_labels': 2, 'n_informative': 8
        },
        'yeast': {
            'n_samples': 800, 'n_features': 40, 'n_classes': 14,
            'n_labels': 4, 'n_informative': 10
        },
        'cal500': {
            'n_samples': 502, 'n_features': 68, 'n_classes': 20,
            'n_labels': 5, 'n_informative': 10
        },
        'mediamill': {
            'n_samples': 1000, 'n_features': 50, 'n_classes': 30,
            'n_labels': 5, 'n_informative': 15
        },
    }

    cfg = configs.get(name, configs['emotions'])
    X, Y = make_multilabel_classification(
        n_samples=cfg['n_samples'],
        n_features=cfg['n_features'],
        n_classes=cfg['n_classes'],
        n_labels=cfg['n_labels'],
        random_state=random_state,
        allow_unlabeled=False
    )
    return X.astype(np.float64), Y.astype(np.float64)


# ============================================================
# COST RANGES
# Section 9.1.4
# ============================================================

def get_cost_range(loss_name: str, K: int, n_points: int = 10) -> np.ndarray:
    """
    Cost ranges cho từng loss function (Section 9.1.4).

    Lý do cost range khác nhau:
    - Hamming loss ∈ [0, K]  → cost range [0.05K, 0.5K] / K = [0.05, 0.5]
    - Rank loss cũng lớn     → c ∈ [0.1, 1.0]
    - Subset 0/1 loss ∈ [0,1]→ c ∈ [0.25/K, 2.5/K] (rất nhỏ)
    - F1, Jaccard ∈ [0,1]    → c ∈ [0.1/K, 1/K]
    """
    ranges = {
        'hamming': np.linspace(0.05, 0.5, n_points),
        'rank': np.linspace(0.1, 1.0, n_points),
        'subset01': np.linspace(0.25 / K, 2.5 / K, n_points),
        'fmeasure': np.linspace(0.1 / K, 1.0 / K, n_points),
        'jaccard': np.linspace(0.1 / K, 1.0 / K, n_points),
    }
    return ranges.get(loss_name, np.linspace(0.1, 1.0, n_points))


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment_for_loss(
    X: np.ndarray,
    Y: np.ndarray,
    loss_name: str,
    classifier_name: str = 'br_lr',
    n_folds: int = 5,          # Paper dùng 10, ta dùng 5 cho nhanh
    n_cost_points: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Chạy 1 experiment cho 1 loss function + 1 classifier.

    Với mỗi cost c trong cost_range:
        1. K-fold cross-validation
        2. Với mỗi fold: train classifier → predict probs → tính BOP
        3. Tính average loss và abstention size
        4. So sánh với MLC và ABS baselines

    Returns:
        DataFrame với columns: [cost, loss_mlc, loss_abs, loss_sep, loss_par,
                                  abs_size_sep, abs_size_par]
    """
    N, K = Y.shape
    cost_range = get_cost_range(loss_name, K, n_cost_points)

    # Chọn BOP function và loss function tương ứng
    bop_fn_map = {
        'hamming': bop_hamming,
        'subset01': bop_subset01,
        'fmeasure': bop_fmeasure,
        'jaccard': bop_jaccard,
    }
    loss_fn_map = {
        'hamming': loss_hamming,
        'subset01': loss_subset01,
        'fmeasure': loss_fmeasure,
        'jaccard': loss_jaccard,
    }

    bop_fn = bop_fn_map.get(loss_name)
    loss_fn = loss_fn_map.get(loss_name)

    if bop_fn is None:
        raise ValueError(f"Unknown loss: {loss_name}")

    # K-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scaler = StandardScaler()

    results = []

    print(f"  Running {loss_name} with {classifier_name} | K={K} labels | {n_folds}-fold CV")

    for cost_idx, c in enumerate(cost_range):
        fold_metrics = {
            'loss_mlc': [], 'loss_abs': [],
            'loss_sep': [], 'loss_par': [],
            'abs_sep': [], 'abs_par': []
        }

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Normalize features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train classifier
            clf = _make_classifier(classifier_name)
            clf.fit(X_train, Y_train)

            # Predict marginal probabilities
            probs_test = clf.predict_proba(X_test)  # shape (N_test, K)

            # Evaluate trên từng instance
            for i in range(len(X_test)):
                probs_i = probs_test[i]
                y_i = Y_test[i]

                # --- Baseline: MLC (predict tất cả, không abstain) ---
                pred_mlc = predict_mlc(probs_i)
                fold_metrics['loss_mlc'].append(loss_fn(y_i, pred_mlc))

                # --- Baseline: ABS (abstain tất cả) ---
                pred_abs = predict_full_abstain(K)
                fold_metrics['loss_abs'].append(loss_fn(y_i, pred_abs))

                # --- SEP: linear penalty ---
                g_sep = make_penalty('linear', K, c)
                pred_sep = bop_fn(probs_i, g_sep)
                fold_metrics['loss_sep'].append(loss_fn(y_i, pred_sep))
                fold_metrics['abs_sep'].append(abstention_size(pred_sep))

                # --- PAR: concave penalty ---
                # Paper dùng c' = 2c cho PAR để tương đương ABS cost với SEP
                g_par = make_penalty('concave', K, c * 2)
                pred_par = bop_fn(probs_i, g_par)
                fold_metrics['loss_par'].append(loss_fn(y_i, pred_par))
                fold_metrics['abs_par'].append(abstention_size(pred_par))

        # Average qua tất cả instances và folds
        results.append({
            'cost': c,
            'cost_pct': cost_idx + 1,   # 1..n_cost_points (x-axis trong paper)
            'loss_mlc': np.mean(fold_metrics['loss_mlc']),
            'loss_abs': np.mean(fold_metrics['loss_abs']),
            'loss_sep': np.mean(fold_metrics['loss_sep']),
            'loss_par': np.mean(fold_metrics['loss_par']),
            'abs_size_sep': np.mean(fold_metrics['abs_sep']),
            'abs_size_par': np.mean(fold_metrics['abs_par']),
        })

        print(f"    c={c:.4f} | MLC={results[-1]['loss_mlc']:.4f} | "
              f"SEP={results[-1]['loss_sep']:.4f} | PAR={results[-1]['loss_par']:.4f} | "
              f"AbsSize_SEP={results[-1]['abs_size_sep']:.3f}")

    return pd.DataFrame(results)


def _make_classifier(name: str):
    """Factory tạo classifier theo tên."""
    if name == 'br_lr':
        return BinaryRelevance(base_learner='lr')
    elif name == 'br_svm':
        return BinaryRelevance(base_learner='svm')
    elif name == 'ecc_lr':
        return EnsembleClassifierChains(base_learner='lr', n_chains=10)
    elif name == 'ecc_svm':
        return EnsembleClassifierChains(base_learner='svm', n_chains=5)
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ============================================================
# META-ANALYSIS: Average gain vs classifier performance
# Section 9.2, Figure 7
# ============================================================

def compute_average_gain(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Tính average gain của abstention so với MLC baseline.
    Figure 7 trong paper: gain tăng khi MLC loss cao hơn.

    Gain = E[loss_MLC - loss_SEP] (averaged over cost values)
    """
    gain_sep = np.mean(results_df['loss_mlc'] - results_df['loss_sep'])
    gain_par = np.mean(results_df['loss_mlc'] - results_df['loss_par'])
    avg_abs_sep = np.mean(results_df['abs_size_sep'])
    avg_abs_par = np.mean(results_df['abs_size_par'])
    avg_loss_mlc = np.mean(results_df['loss_mlc'])

    return {
        'gain_sep': gain_sep,
        'gain_par': gain_par,
        'avg_abs_size_sep': avg_abs_sep,
        'avg_abs_size_par': avg_abs_par,
        'avg_loss_mlc': avg_loss_mlc,
    }


# ============================================================
# FULL EXPERIMENT SUITE
# ============================================================

def run_full_experiment(
    dataset_names: List[str],
    loss_names: List[str],
    classifier_names: List[str],
    n_folds: int = 5,
    n_cost_points: int = 8,
    random_state: int = 42
) -> Dict:
    """
    Chạy toàn bộ experiment suite.

    Returns:
        Nested dict: results[dataset][loss][classifier] = DataFrame
    """
    all_results = {}

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        X, Y = generate_synthetic_dataset(ds_name, random_state)
        print(f"  Shape: X={X.shape}, Y={Y.shape}")
        print(f"  Label cardinality: {np.mean(np.sum(Y, axis=1)):.2f}")

        all_results[ds_name] = {}

        for loss_name in loss_names:
            all_results[ds_name][loss_name] = {}

            for clf_name in classifier_names:
                print(f"\n  [{ds_name}] Loss={loss_name} | Clf={clf_name}")
                try:
                    df = run_experiment_for_loss(
                        X, Y, loss_name, clf_name,
                        n_folds, n_cost_points, random_state
                    )
                    all_results[ds_name][loss_name][clf_name] = df
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    return all_results
