"""Interactive visualization helpers for maps and charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


TEMPLATE = "plotly_white"


def create_prediction_map(prediction_df: pd.DataFrame):
    """Create a point-center map colored by predicted class."""
    return px.scatter_mapbox(
        prediction_df,
        lat="latitude_center",
        lon="longitude_center",
        color="predicted_class",
        size="record_count",
        hover_data=["h3_index", "record_count", "predicted_class", "predicted_probability", "confidence_label", "recommendation_status"],
        color_continuous_scale="Turbo",
        zoom=5.3,
        height=620,
        title="Predicted Default Class by Hex Cell",
    ).update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0))


def create_confidence_map(prediction_df: pd.DataFrame):
    """Create a point-center map colored by predicted probability."""
    return px.scatter_mapbox(
        prediction_df,
        lat="latitude_center",
        lon="longitude_center",
        color="predicted_probability",
        size="record_count",
        hover_data=["h3_index", "record_count", "predicted_class", "predicted_probability", "confidence_label", "recommendation_status"],
        color_continuous_scale="Viridis",
        zoom=5.3,
        height=620,
        title="Prediction Confidence by Hex Cell",
    ).update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0))


def create_class_distribution_chart(hex_df: pd.DataFrame):
    counts = hex_df["dominant_class"].value_counts().sort_index().reset_index()
    counts.columns = ["dominant_class", "hex_cells"]
    return px.bar(counts, x="dominant_class", y="hex_cells", template=TEMPLATE, title="Observed Dominant Class Distribution")


def create_confidence_distribution_chart(prediction_df: pd.DataFrame):
    return px.histogram(prediction_df, x="predicted_probability", nbins=20, template=TEMPLATE, title="Prediction Probability Distribution")


def create_volume_vs_confidence_chart(prediction_df: pd.DataFrame):
    return px.scatter(
        prediction_df,
        x="record_count",
        y="predicted_probability",
        color="confidence_label",
        template=TEMPLATE,
        title="Record Volume vs. Prediction Confidence",
    )
