from __future__ import annotations

"""
Evaluation metrics for machine-generated text detection.

Primary metric for RAID: TPR @ 5% FPR (true positive rate at a fixed 5% false
positive rate), which is the standard used in the RAID paper.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tpr_at_fpr(
    y_true: list | np.ndarray,
    y_score: list | np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    Compute TPR at a fixed FPR threshold -- the primary RAID metric.

    Parameters
    ----------
    y_true      : binary ground-truth labels (1 = AI, 0 = human).
    y_score     : predicted probability of being AI-generated.
    target_fpr  : false positive rate at which to read off TPR (default 5%).

    Returns
    -------
    TPR (float) at the closest achievable FPR <= target_fpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    # Find the largest FPR that does not exceed the target
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_score: list | np.ndarray | None = None,
    target_fpr: float = 0.05,
) -> dict:
    """
    Compute a standard suite of detection metrics.

    Parameters
    ----------
    y_true     : binary ground-truth labels (1 = AI, 0 = human).
    y_pred     : binary predictions.
    y_score    : predicted probabilities (required for AUC and TPR@FPR).
    target_fpr : FPR level for TPR@FPR metric.

    Returns
    -------
    Dict with keys: accuracy, precision, recall, f1, auc, tpr_at_X_fpr.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        key = f"tpr_at_{int(target_fpr * 100)}pct_fpr"
        metrics[key] = tpr_at_fpr(y_true, y_score, target_fpr)

    return metrics


def per_domain_metrics(
    df,
    y_score_col: str,
    y_pred_col: str,
    label_col: str = "label",
    domain_col: str = "domain",
    target_fpr: float = 0.05,
) -> dict[str, dict]:
    """
    Compute metrics broken down by domain.

    Parameters
    ----------
    df          : DataFrame with predictions already added.
    y_score_col : column name for predicted probabilities.
    y_pred_col  : column name for binary predictions.
    label_col   : ground-truth label column.
    domain_col  : domain column name.
    target_fpr  : FPR level for TPR@FPR.

    Returns
    -------
    Dict mapping domain name -> metrics dict.
    """
    results = {}
    for domain, group in df.groupby(domain_col):
        results[domain] = compute_metrics(
            group[label_col].tolist(),
            group[y_pred_col].tolist(),
            group[y_score_col].tolist() if y_score_col in group.columns else None,
            target_fpr=target_fpr,
        )
    return results
