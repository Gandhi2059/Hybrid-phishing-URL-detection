"""
src/logistic_scratch.py
------------------------
Binary logistic regression implemented from first principles using NumPy.

Purpose
-------
Provided as an educational reference for the IEEE paper, demonstrating that
the mathematical foundations of gradient descent and sigmoid activation are
transparent and auditable—unlike "black-box" sklearn estimators.

Algorithm
---------
Optimisation: Mini-batch gradient descent (full-batch by default).
Loss        : Binary cross-entropy, J = -1/m Σ[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient    : ∂J/∂w = 1/m · Xᵀ(ŷ − y),  ∂J/∂b = 1/m · Σ(ŷ − y)
"""

from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    """
    Logistic regression trained via gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size η for weight updates (default: 0.01).
    num_iterations : int
        Number of gradient descent steps (default: 1000).
    """

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate  = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None         = None
        self.cost_history: list[float]  = []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})."""
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _init_weights(self, n_features: int) -> None:
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """J = -1/m Σ[y log(ŷ) + (1−y) log(1−ŷ)]."""
        m   = len(y_true)
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-(1.0 / m) * np.sum(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        ))

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """
        Train the model.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Training features.
        y : ndarray, shape (m,)
            Binary labels {0, 1}.

        Returns
        -------
        self
        """
        m, n = X.shape
        self._init_weights(n)
        self.cost_history = []

        for i in range(self.num_iterations):
            y_hat = self._sigmoid(X @ self.weights + self.bias)

            dw = (1.0 / m) * (X.T @ (y_hat - y))
            db = (1.0 / m) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            cost = self._binary_cross_entropy(y, y_hat)
            self.cost_history.append(cost)

            if (i + 1) % 100 == 0:
                print(f"  Iter {i+1:>5}/{self.num_iterations}  cost={cost:.5f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | X) ∈ [0, 1]."""
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class labels {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_raw, y_raw = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=500)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"\nTest Accuracy (scratch LR): {acc * 100:.2f}%")
