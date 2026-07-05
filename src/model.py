"""Model training and confidence-gated prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a random forest classifier for hex-level categorical prediction."""
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_confidence(model: RandomForestClassifier, X) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class and maximum predicted probability."""
    probabilities = model.predict_proba(X)
    classes = model.classes_
    best_idx = probabilities.argmax(axis=1)
    predictions = classes[best_idx]
    confidence = probabilities.max(axis=1)
    return predictions, confidence


def create_prediction_table(hex_df: pd.DataFrame, predictions, probabilities, confidence_threshold: float) -> pd.DataFrame:
    """Attach predictions and confidence-gated recommendation labels to hex rows."""
    output = hex_df.copy()
    output["predicted_class"] = predictions.astype(int)
    output["predicted_probability"] = probabilities
    output["confidence_label"] = np.select(
        [output["predicted_probability"] >= 0.70, output["predicted_probability"] >= 0.50],
        ["High Confidence", "Medium Confidence"],
        default="Low Confidence",
    )
    output["recommendation_status"] = np.select(
        [output["predicted_probability"] >= confidence_threshold, output["predicted_probability"] >= 0.50],
        ["Use Model Recommendation", "Review Manually"],
        default="Fallback Default",
    )
    return output
