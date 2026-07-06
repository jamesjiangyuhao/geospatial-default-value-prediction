"""Interactive visualization helpers for maps and charts."""

from __future__ import annotations

from pathlib import Path

import h3
import pandas as pd
import plotly.express as px


TEMPLATE = "plotly_white"
CONFIDENCE_COLOR_MAP = {
    "High Confidence": "#15803d",
    "Medium Confidence": "#f59e0b",
    "Low Confidence": "#b91c1c",
}


def _cell_boundary(h3_index: str) -> list[tuple[float, float]]:
    """Return H3 cell boundary coordinates as longitude/latitude pairs."""
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(h3_index)
    else:
        boundary = h3.h3_to_geo_boundary(h3_index)
    return [(lon, lat) for lat, lon in boundary]


def _prediction_geojson(prediction_df: pd.DataFrame) -> dict:
    """Convert H3 cells to a GeoJSON FeatureCollection for polygon maps."""
    features = []
    for row in prediction_df.itertuples(index=False):
        coords = _cell_boundary(row.h3_index)
        coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "id": row.h3_index,
                "properties": {"h3_index": row.h3_index},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def create_prediction_map(prediction_df: pd.DataFrame):
    """Create a hex polygon map colored by predicted class."""
    geojson = _prediction_geojson(prediction_df)
    center = {
        "lat": prediction_df["latitude_center"].mean(),
        "lon": prediction_df["longitude_center"].mean(),
    }
    fig = px.choropleth_mapbox(
        prediction_df,
        geojson=geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color="predicted_class",
        hover_data=["h3_index", "record_count", "predicted_class", "predicted_probability", "confidence_label", "recommendation_status"],
        color_continuous_scale="Turbo",
        center=center,
        zoom=5.3,
        height=620,
        title="Predicted Default Class by Hex Cell",
    )
    return fig.update_traces(marker_line_color="white", marker_line_width=0.6).update_layout(
        mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0)
    )


def create_confidence_map(prediction_df: pd.DataFrame):
    """Create a hex polygon map colored by confidence label."""
    geojson = _prediction_geojson(prediction_df)
    center = {
        "lat": prediction_df["latitude_center"].mean(),
        "lon": prediction_df["longitude_center"].mean(),
    }
    fig = px.choropleth_mapbox(
        prediction_df,
        geojson=geojson,
        locations="h3_index",
        featureidkey="properties.h3_index",
        color="confidence_label",
        hover_data=["h3_index", "record_count", "predicted_class", "predicted_probability", "confidence_label", "recommendation_status"],
        color_discrete_map=CONFIDENCE_COLOR_MAP,
        center=center,
        zoom=5.3,
        height=620,
        title="Prediction Confidence by Hex Cell",
        category_orders={"confidence_label": ["High Confidence", "Medium Confidence", "Low Confidence"]},
    )
    return fig.update_traces(marker_line_color="white", marker_line_width=0.6).update_layout(
        mapbox_style="open-street-map", margin=dict(l=0, r=0, t=45, b=0)
    )


def create_static_hex_confidence_map(prediction_df: pd.DataFrame, output_path: str | Path, max_cells: int = 160) -> Path:
    """Save a static portfolio image showing H3 cells colored by confidence and default class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    df = prediction_df.copy()
    center_lat = df["latitude_center"].median()
    center_lon = df["longitude_center"].median()
    df["center_distance"] = (df["latitude_center"] - center_lat).pow(2) + (df["longitude_center"] - center_lon).pow(2)
    low_confidence = df[df["confidence_label"] == "Low Confidence"].sort_values("center_distance").head(4)
    central_sample = df.sort_values(["center_distance", "record_count"], ascending=[True, False]).head(max_cells)
    sample = (
        pd.concat([central_sample, low_confidence], ignore_index=True)
        .drop_duplicates(subset=["h3_index"])
        .head(max_cells + len(low_confidence))
    )

    def color_for(row: pd.Series) -> str:
        if row["confidence_label"] == "Low Confidence" or row["recommendation_status"] == "Fallback Default":
            return "#b91c1c"
        if row["confidence_label"] == "Medium Confidence":
            return "#f59e0b"
        if int(row["predicted_class"]) <= 4:
            return "#15803d"
        if int(row["predicted_class"]) <= 7:
            return "#65a30d"
        return "#7c2d12"

    fig, ax = plt.subplots(figsize=(12, 7.6))
    for _, row in sample.iterrows():
        polygon = Polygon(
            _cell_boundary(row["h3_index"]),
            closed=True,
            facecolor=color_for(row),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
        )
        ax.add_patch(polygon)

    ax.set_xlim(sample["longitude_center"].min() - 0.12, sample["longitude_center"].max() + 0.12)
    ax.set_ylim(sample["latitude_center"].min() - 0.10, sample["latitude_center"].max() + 0.10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Hex Confidence Map\nGreen = high-confidence lower class; red = low-confidence review",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)

    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="High confidence, lower default class", markerfacecolor="#15803d", markersize=11),
        Line2D([0], [0], marker="s", color="w", label="High confidence, middle default class", markerfacecolor="#65a30d", markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Medium confidence", markerfacecolor="#f59e0b", markersize=11),
        Line2D([0], [0], marker="s", color="w", label="Low confidence / review", markerfacecolor="#b91c1c", markersize=11),
        Line2D([0], [0], marker="s", color="w", label="High confidence, higher default class", markerfacecolor="#7c2d12", markersize=11),
    ]
    ax.legend(handles=legend_items, frameon=True, facecolor="white", edgecolor="#d1d5db", loc="lower right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return output_path


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
