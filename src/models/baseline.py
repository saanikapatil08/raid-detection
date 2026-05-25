from __future__ import annotations

"""
Baseline machine-generated text detectors.

Currently implemented:
- ZeroShotDetector: wraps any HuggingFace sequence-classification model,
  e.g. roberta-base-openai-detector or Hello-SimpleAI/chatgpt-detector-roberta.
"""

from typing import List

import numpy as np


class ZeroShotDetector:
    """
    Zero-shot classifier using a pretrained HuggingFace model.

    Parameters
    ----------
    model_name : HuggingFace model hub name or local path.
    device     : 'cpu', 'cuda', or 'mps'. Auto-detected if None.
    batch_size : texts per forward pass.
    """

    def __init__(
        self,
        model_name: str = "roberta-base-openai-detector",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = None
        self._device = device

    def _load(self) -> None:
        """Lazy-load the pipeline on first call."""
        import torch
        from transformers import pipeline

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=self._device,
            truncation=True,
            max_length=512,
        )

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Return per-text probability of being AI-generated.

        Returns
        -------
        1-D numpy array of shape (n_texts,) with values in [0, 1].
        """
        if self._pipeline is None:
            self._load()

        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results = self._pipeline(batch)
            for r in results:
                label = r["label"].upper()
                score = r["score"]
                # Different models use different label conventions
                if label in ("FAKE", "AI", "LABEL_1", "MACHINE"):
                    scores.append(score)
                elif label in ("REAL", "HUMAN", "LABEL_0"):
                    scores.append(1.0 - score)
                else:
                    scores.append(score)
        return np.array(scores, dtype=float)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions (1 = AI, 0 = human).

        Parameters
        ----------
        texts     : list of input strings.
        threshold : decision threshold on the AI-probability score.
        """
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(int)
