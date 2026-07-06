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

    x_min = sample["longitude_center"].min() - 0.12
    x_max = sample["longitude_center"].max() + 0.12
    y_min = sample["latitude_center"].min() - 0.10
    y_max = sample["latitude_center"].max() + 0.10

    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_facecolor("#eef2e6")

    # Lightweight map-style context for the Houston sample area. The basemap is
    # intentionally stylized so the project remains dependency-light and fully reproducible.
    water_polygons = [
        [(-95.08, 29.36), (-94.83, 29.36), (-94.83, 29.88), (-94.94, 29.83), (-95.02, 29.70)],
        [(-95.22, 29.48), (-95.05, 29.43), (-94.95, 29.56), (-95.13, 29.62)],
    ]
    for coords in water_polygons:
        ax.add_patch(Polygon(coords, closed=True, facecolor="#bfdbfe", edgecolor="#93c5fd", linewidth=0.8, alpha=0.75, zorder=0))

    roads = [
        ([-95.82, -95.02], [29.78, 29.77], "I-10"),
        ([-95.55, -95.31, -95.18], [30.18, 29.78, 29.42], "I-45"),
        ([-95.72, -95.37, -95.08], [29.55, 29.74, 30.02], "I-69"),
        ([-95.55, -95.18, -95.10, -95.34, -95.62, -95.66, -95.55], [29.92, 29.88, 29.63, 29.47, 29.57, 29.78, 29.92], "Beltway 8"),
    ]
    for xs, ys, label in roads:
        ax.plot(xs, ys, color="#d1d5db", linewidth=4.2, alpha=0.75, solid_capstyle="round", zorder=1)
        ax.plot(xs, ys, color="#ffffff", linewidth=1.4, alpha=0.9, solid_capstyle="round", zorder=2)
        ax.text(xs[len(xs) // 2], ys[len(ys) // 2] + 0.012, label, fontsize=8, color="#6b7280", ha="center", zorder=3, clip_on=True)

    city_labels = [
        ("Houston", -95.3698, 29.7604, 11),
        ("Sugar Land", -95.6349, 29.6197, 8),
        ("The Woodlands", -95.4613, 30.1658, 8),
        ("Baytown", -94.9774, 29.7355, 8),
    ]
    for label, lon, lat, size in city_labels:
        if x_min <= lon <= x_max and y_min <= lat <= y_max:
            ax.text(lon, lat, label, fontsize=size, color="#374151", weight="bold", ha="center", zorder=3, clip_on=True)

    for _, row in sample.iterrows():
        polygon = Polygon(
            _cell_boundary(row["h3_index"]),
            closed=True,
            facecolor=color_for(row),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.90,
            zorder=4,
        )
        ax.add_patch(polygon)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Houston Hex Confidence Map", loc="left", fontsize=17, weight="bold", pad=18)
    ax.text(
        0,
        1.01,
        "Synthetic H3 grid overlay: green = high-confidence lower class, red = low-confidence review",
        transform=ax.transAxes,
        fontsize=10,
        color="#4b5563",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, color="#ffffff", linewidth=0.7, alpha=0.75, zorder=0)

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
