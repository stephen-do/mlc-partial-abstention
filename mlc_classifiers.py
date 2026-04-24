"""
mlc_classifiers.py
==================
Implement các MLC classifiers dùng trong experiments.
Section 9.1 của bài báo.

Classifiers:
    1. BinaryRelevance (BR) — Section 9.1.1
       Học K binary classifiers riêng biệt, phù hợp với
       giả định CLI (Conditional Label Independence).
       Được dùng với: Logistic Regression (BR+LR), SVM (BR+SVM)

    2. EnsembleClassifierChains (ECC) — Section 9.1.2
       Ensemble của M classifier chains, mỗi chain học
       các nhãn theo thứ tự ngẫu nhiên, có thể nắm bắt
       label dependencies.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from typing import Optional, Literal


# ============================================================
# BINARY RELEVANCE (BR)
# Section 9.1.1
# ============================================================

class BinaryRelevance:
    """
    Binary Relevance learning — Section 9.1.1.

    Ý tưởng: Với mỗi nhãn k, học một binary classifier h_k riêng biệt.
    Phù hợp với giả định CLI vì treat từng nhãn độc lập.

    Output: marginal probabilities p_k = p(Y_k=1 | x) cho mỗi nhãn k.
    Đây là input chính của tất cả các BOP algorithms.

    Paper dùng 2 variants:
        - BR+LR: base learner = Logistic Regression
        - BR+SVM: base learner = SVM + Platt scaling
    """

    def __init__(self, base_learner: str = 'lr', random_state: int = 42):
        """
        Args:
            base_learner: 'lr' (Logistic Regression) hoặc 'svm'
            random_state: random seed
        """
        self.base_learner = base_learner
        self.random_state = random_state
        self.classifiers_ = []

    def _make_classifier(self):
        """Tạo một binary classifier."""
        if self.base_learner == 'lr':
            # Section 9.1.1: Logistic Regression với regularization C=1 (sklearn default)
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )
        elif self.base_learner == 'svm':
            # Section 9.1.1: SVM + Platt scaling để convert scores → probabilities
            # (Lin et al., 2007; Platt, 1999) — Reference [38] và [46] trong bài báo
            svm = SVC(kernel='rbf', probability=False, random_state=self.random_state)
            return CalibratedClassifierCV(svm, method='sigmoid', cv=5)
        else:
            raise ValueError(f"Unknown base_learner: {self.base_learner}")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'BinaryRelevance':
        """
        Học K binary classifiers, một per nhãn.

        Args:
            X: features, shape (N, D)
            Y: labels, shape (N, K), binary {0, 1}
        """
        N, K = Y.shape
        self.classifiers_ = []
        self.K_ = K

        for k in range(K):
            clf = self._make_classifier()
            y_k = Y[:, k]

            # Xử lý trường hợp chỉ có 1 class (sẽ lỗi khi fit)
            if len(np.unique(y_k)) < 2:
                # Dummy classifier trả về constant probability
                from sklearn.dummy import DummyClassifier
                clf = DummyClassifier(strategy='most_frequent')

            clf.fit(X, y_k)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Trả về marginal probabilities p_k = P(Y_k=1 | x).

        Returns:
            probs: shape (N, K), mỗi cột là p_k của một nhãn
        """
        N = X.shape[0]
        probs = np.zeros((N, self.K_))

        for k, clf in enumerate(self.classifiers_):
            if hasattr(clf, 'predict_proba'):
                p = clf.predict_proba(X)
                # predict_proba trả về [P(y=0), P(y=1)], lấy cột 1
                if p.shape[1] == 2:
                    probs[:, k] = p[:, 1]
                else:
                    probs[:, k] = p[:, 0]
            else:
                # Fallback: dùng decision function + sigmoid
                scores = clf.decision_function(X)
                probs[:, k] = 1.0 / (1.0 + np.exp(-scores))

        return np.clip(probs, 1e-6, 1 - 1e-6)


# ============================================================
# ENSEMBLE CLASSIFIER CHAINS (ECC)
# Section 9.1.2
# ============================================================

class ClassifierChain:
    """
    Một Classifier Chain — component của ECC.

    Ý tưởng: Học nhãn theo một thứ tự (permutation) π.
    Khi học nhãn k, đưa vào features cả các nhãn đã predict trước đó.
    Điều này cho phép nắm bắt label dependencies.

    Equation (61) trong paper:
        p̄_k = (1/M) * Σ_{m=1}^{M} p_{k,m}
    """

    def __init__(self, base_learner: str = 'lr',
                 order: Optional[np.ndarray] = None,
                 random_state: int = 42):
        self.base_learner = base_learner
        self.order = order
        self.random_state = random_state
        self.classifiers_ = []

    def _make_classifier(self):
        if self.base_learner == 'lr':
            return LogisticRegression(C=1.0, max_iter=1000,
                                      random_state=self.random_state,
                                      solver='lbfgs')
        else:
            svm = SVC(kernel='rbf', probability=False,
                      random_state=self.random_state)
            return CalibratedClassifierCV(svm, method='sigmoid', cv=3)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'ClassifierChain':
        N, K = Y.shape
        self.K_ = K

        if self.order is None:
            rng = np.random.RandomState(self.random_state)
            self.order = rng.permutation(K)

        self.classifiers_ = []

        # Train theo thứ tự π
        X_aug = X.copy()
        for i, k in enumerate(self.order):
            clf = self._make_classifier()
            y_k = Y[:, k]

            if len(np.unique(y_k)) < 2:
                from sklearn.dummy import DummyClassifier
                clf = DummyClassifier(strategy='most_frequent')
            clf.fit(X_aug, y_k)
            self.classifiers_.append(clf)

            # Augment features: thêm nhãn k vào features cho nhãn tiếp theo
            X_aug = np.hstack([X_aug, y_k.reshape(-1, 1)])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict marginal probs (dependent marginals)."""
        N = X.shape[0]
        probs = np.zeros((N, self.K_))
        X_aug = X.copy()

        for i, k in enumerate(self.order):
            clf = self.classifiers_[i]
            if hasattr(clf, 'predict_proba'):
                p = clf.predict_proba(X_aug)
                if p.shape[1] == 2:
                    pk = p[:, 1]
                else:
                    pk = p[:, 0]
            else:
                scores = clf.decision_function(X_aug)
                pk = 1.0 / (1.0 + np.exp(-scores))

            probs[:, k] = np.clip(pk, 1e-6, 1 - 1e-6)
            # Augment với predicted probability (soft augmentation)
            X_aug = np.hstack([X_aug, pk.reshape(-1, 1)])

        return probs


class EnsembleClassifierChains:
    """
    Ensemble of Classifier Chains (ECC) — Section 9.1.2.

    Train M Classifier Chains với M permutations ngẫu nhiên khác nhau.
    Final probability = mean của M predictions (Equation 61):
        p̄_k = (1/M) * Σ_{m=1}^{M} p_{k,m}

    Paper dùng M=50 (Section 9.1.2).
    Có thể nắm bắt label dependence tốt hơn BR.
    """

    def __init__(self, base_learner: str = 'lr', n_chains: int = 10,
                 random_state: int = 42):
        """
        Args:
            base_learner: 'lr' hoặc 'svm'
            n_chains: số lượng chains M (paper dùng 50, ta dùng 10 cho nhanh)
            random_state: seed gốc
        """
        self.base_learner = base_learner
        self.n_chains = n_chains
        self.random_state = random_state
        self.chains_ = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'EnsembleClassifierChains':
        N, K = Y.shape
        self.K_ = K
        self.chains_ = []

        rng = np.random.RandomState(self.random_state)

        for m in range(self.n_chains):
            order = rng.permutation(K)
            chain = ClassifierChain(
                base_learner=self.base_learner,
                order=order,
                random_state=self.random_state + m
            )
            chain.fit(X, Y)
            self.chains_.append(chain)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Equation (61): p̄_k = (1/M) * Σ p_{k,m}
        """
        all_probs = np.stack(
            [chain.predict_proba(X) for chain in self.chains_],
            axis=0
        )  # shape (M, N, K)
        return np.mean(all_probs, axis=0)  # shape (N, K)
