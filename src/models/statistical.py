from __future__ import annotations

"""
Supervised statistical detector: TF-IDF features + Logistic Regression.

Faster than transformer-based detectors and useful as a trained baseline.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class StatisticalDetector:
    """
    Trained text detector using TF-IDF n-gram features and Logistic Regression.

    Parameters
    ----------
    max_features : vocabulary size cap for TF-IDF.
    ngram_range  : n-gram range passed to TfidfVectorizer.
    C            : inverse regularisation strength for LogisticRegression.
    """

    def __init__(
        self,
        max_features: int = 100_000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
    ) -> None:
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
            )),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1)),
        ])
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "StatisticalDetector":
        """Train on a list of texts and binary labels (1 = AI, 0 = human)."""
        self._pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability of being AI-generated, shape (n,)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._pipeline.predict_proba(texts)[:, 1].astype(float)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (1 = AI, 0 = human)."""
        return (self.predict_proba(texts) >= threshold).astype(int)
