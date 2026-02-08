"""
Metrics Calculation Module for Hate Speech Detection.

This module provides comprehensive metric calculation for binary classification
tasks, specifically designed for hate speech detection evaluation.

Metrics Implemented:
    - F1-Score: Harmonic mean of precision and recall (balanced metric)
    - F2-Score: Weighted F-score emphasizing recall (β=2)
              Rationale: In hate speech detection, missing actual hate speech (FN)
              is more costly than false alarms (FP), hence β=2 weights recall 2× higher.
    - MCC: Matthews Correlation Coefficient (range: [-1, 1])
              Robust to class imbalance, measures correlation between predictions and truth
    - Accuracy: Overall classification accuracy (interpretable baseline)
    - S-Score: Combined optimization metric (F2 + MCC)
              Rationale: Balances recall-optimized F2 with correlation-based MCC,
              providing a single scalar objective for prompt optimization.
              Range: approximately [-1, 2] (MCC ∈ [-1,1], F2 ∈ [0,1])

IMPORTANT - zero_division Parameter:
    We use zero_division=0 for F-scores (sklearn parameter). This means:
    - If the model predicts ALL negative (TP + FP = 0), then F-score = 0.0
    - This is a CONSERVATIVE choice: predicting all NOT-HS would be useless in
      production, so scoring it as 0.0 correctly represents complete failure.
    - Alternative zero_division=1 would dishonestly claim "perfect precision" when
      making no positive predictions at all.
    - This follows sklearn convention and ensures scientific honesty.

Author: Bachelor Thesis Project
Date: 2025
"""

from typing import Dict, List
from sklearn.metrics import fbeta_score, matthews_corrcoef, accuracy_score, f1_score


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for hate speech detection.

    Uses standard, peer-reviewed metrics suitable for imbalanced classification:
    - F1-Score: Balanced precision/recall metric
    - F2-Score: Recall-optimized metric (β=2 prioritizes catching hate speech)
    - MCC: Correlation-based metric robust to class imbalance
    - Accuracy: Overall correctness (interpretable baseline)
    - S-Score: Combined optimization metric (F2 + MCC) for prompt selection

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        Dictionary containing:
            - F1: F1-Score (harmonic mean of precision/recall)
            - F2: F2-Score (beta=2, emphasizes recall)
            - MCC: Matthews Correlation Coefficient (range: [-1, 1])
            - ACC: Accuracy (range: [0, 1])
            - S_SCORE: S-Score (F2 + MCC, combined optimization metric)

    Note on zero_division=0:
        If the model predicts all negative (TP+FP=0), F-scores are set to 0.0.
        This conservative approach ensures that complete model failure (predicting
        all NOT-HS) is scored as 0.0 rather than undefined or misleadingly high.
        This is scientifically honest and follows sklearn convention.

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0]
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"F2: {metrics['F2']:.4f}, MCC: {metrics['MCC']:.4f}, S-Score: {metrics['S_SCORE']:.4f}")
    """
    # Calculate F-scores with zero_division=0 (conservative approach)
    # If model predicts all negative: F1=F2=0.0 (correct failure mode)
    f1: float = f1_score(y_true, y_pred, average='binary', zero_division=0)
    f2: float = fbeta_score(y_true, y_pred, beta=2, average='binary', zero_division=0)

    # MCC and Accuracy don't have zero_division issues
    mcc: float = matthews_corrcoef(y_true, y_pred)
    acc: float = accuracy_score(y_true, y_pred)

    # S-Score: Combined optimization metric (F2 + MCC)
    # Balances recall-optimized F2 with correlation-based MCC
    # Higher is better (range: approximately [-1, 2])
    s_score: float = f2 + mcc

    return {
        "F1": round(f1, 4),
        "F2": round(f2, 4),
        "MCC": round(mcc, 4),
        "ACC": round(acc, 4),
        "S_SCORE": round(s_score, 4)
    }
