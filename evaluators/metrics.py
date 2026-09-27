"""
evaluators/metrics.py
Standard evaluation metrics for anomaly detection (AUROC, AP, and Execution Time).
Includes outlier identification via Top-K ranking.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    X_tem: np.ndarray = None,
    elapsed_time: float = None
) -> Dict[str, Any]:
    """Computes ROC-AUC and AP on the selected subset."""
    if X_tem is not None:
        y_eval = y_true[X_tem]
    else:
        y_eval = y_true

    if len(y_eval) != len(scores):
        raise ValueError(f"Length mismatch: y_eval ({len(y_eval)}) vs scores ({len(scores)})")

    if len(np.unique(y_eval)) < 2:
        raise ValueError("Evaluation subset contains only one class. AUROC cannot be computed.")

    auroc = float(roc_auc_score(y_eval, scores))
    ap = float(average_precision_score(y_eval, scores))

    results = {
        "auroc": round(auroc, 4),
        "ap": round(ap, 4)
    }

    if elapsed_time is not None:
        results["time_sec"] = round(elapsed_time, 4)

    return results


def identify_outliers(
    y_true: np.ndarray,
    scores: np.ndarray,
    X_tem: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies detected outliers (True Positives) and missed outliers (False Negatives)
    using Top-K anomaly score ranking where K is the true outlier count.

    Returns
    -------
    df_detected : pd.DataFrame
        Outliers successfully ranked in Top-K.
    df_missed : pd.DataFrame
        Outliers that fell outside Top-K.
    """
    y_eval = y_true[X_tem]
    k_outliers = int(np.sum(y_eval == 1))

    # Sort descending by score
    sorted_order = np.argsort(-scores)
    sorted_indices = X_tem[sorted_order]
    sorted_labels = y_eval[sorted_order]
    sorted_scores = scores[sorted_order]

    # Top-K cutoff
    predicted_outlier_mask = np.zeros(len(X_tem), dtype=bool)
    predicted_outlier_mask[:k_outliers] = True

    df_all = pd.DataFrame({
        "dataset_row_index": sorted_indices,
        "anomaly_score": sorted_scores,
        "score_rank": np.arange(1, len(X_tem) + 1),
        "true_label": sorted_labels,
        "predicted_in_top_k": predicted_outlier_mask
    })

    # Detected = True Outlier (1) and In Top-K
    df_detected = df_all[(df_all["true_label"] == 1) & (df_all["predicted_in_top_k"])].copy()
    df_detected.drop(columns=["predicted_in_top_k"], inplace=True)

    # Missed = True Outlier (1) and NOT in Top-K
    df_missed = df_all[(df_all["true_label"] == 1) & (~df_all["predicted_in_top_k"])].copy()
    df_missed.drop(columns=["predicted_in_top_k"], inplace=True)

    return df_detected, df_missed