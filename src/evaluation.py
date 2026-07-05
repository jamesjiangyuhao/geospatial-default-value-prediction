"""Evaluation plots and summary metrics for geospatial classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate classifier with accuracy, F1, and precision metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
    }


def create_confusion_matrix_plot(y_test, y_pred):
    """Create a matplotlib confusion matrix figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Observed Dominant Class")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def create_feature_importance_plot(model, feature_names):
    """Create a matplotlib feature importance figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#2563eb")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def create_coverage_confidence_curve(prediction_df: pd.DataFrame):
    """Create a matplotlib coverage curve across confidence thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = summarize_threshold_tradeoffs(prediction_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["confidence_threshold"], curve["coverage_rate"], marker="o", color="#0f766e")
    ax.set_title("Coverage vs. Confidence Threshold")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Share of Hex Cells Covered")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def summarize_threshold_tradeoffs(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize count and share of usable recommendations by threshold."""
    rows = []
    total = len(prediction_df)
    for threshold in np.arange(0.40, 0.91, 0.05):
        covered = int((prediction_df["predicted_probability"] >= threshold).sum())
        rows.append(
            {
                "confidence_threshold": round(float(threshold), 2),
                "covered_hex_cells": covered,
                "total_hex_cells": total,
                "coverage_rate": covered / total if total else 0,
            }
        )
    return pd.DataFrame(rows)
