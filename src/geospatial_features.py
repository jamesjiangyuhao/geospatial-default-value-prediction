"""Hex-grid aggregation and spatial neighborhood feature engineering."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

try:
    import h3
except ImportError:  # pragma: no cover
    h3 = None


def _h3_latlng_to_cell(lat: float, lon: float, resolution: int) -> str:
    if h3 is None:
        lat_bin = math.floor(lat * resolution * 4)
        lon_bin = math.floor(lon * resolution * 4)
        return f"fallback_{resolution}_{lat_bin}_{lon_bin}"
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)


def _h3_cell_to_latlng(cell: str, fallback_df: pd.DataFrame | None = None) -> tuple[float, float]:
    if h3 is not None and not str(cell).startswith("fallback_"):
        if hasattr(h3, "cell_to_latlng"):
            return h3.cell_to_latlng(cell)
        return h3.h3_to_geo(cell)
    if fallback_df is None or fallback_df.empty:
        return np.nan, np.nan
    return fallback_df["latitude"].mean(), fallback_df["longitude"].mean()


def assign_h3_index(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Assign each record to an H3 cell, with a deterministic fallback if needed."""
    output = df.copy()
    output["h3_index"] = [_h3_latlng_to_cell(lat, lon, resolution) for lat, lon in zip(output["latitude"], output["longitude"])]
    return output


def aggregate_to_hex(df: pd.DataFrame, h3_col: str = "h3_index") -> pd.DataFrame:
    """Aggregate point-level records into hex-level class share and numeric features."""
    rows = []
    for h3_index, group in df.groupby(h3_col):
        counts = group["categorical_attribute"].value_counts()
        dominant_class = int(counts.idxmax())
        lat_center, lon_center = _h3_cell_to_latlng(h3_index, group)
        row = {
            "h3_index": h3_index,
            "latitude_center": lat_center,
            "longitude_center": lon_center,
            "record_count": len(group),
            "dominant_class": dominant_class,
            "avg_distance_to_service_center": group["distance_to_service_center"].mean(),
            "avg_urban_density_score": group["urban_density_score"].mean(),
            "low_quality_share": (group["data_quality_flag"] == "Low Quality").mean(),
            "market_area": group["market_area"].mode().iat[0],
        }
        for klass in range(1, 11):
            row[f"class_{klass}_share"] = (group["categorical_attribute"] == klass).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def get_neighbor_hexes(h3_index: str, k: int = 1) -> list[str]:
    """Return neighboring H3 cells. Fallback cells use adjacent synthetic bins."""
    if str(h3_index).startswith("fallback_"):
        _, resolution, lat_bin, lon_bin = str(h3_index).split("_")
        neighbors = []
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                if dx == 0 and dy == 0:
                    continue
                neighbors.append(f"fallback_{resolution}_{int(lat_bin) + dx}_{int(lon_bin) + dy}")
        return neighbors
    if h3 is None:
        return []
    if hasattr(h3, "grid_disk"):
        return [cell for cell in h3.grid_disk(h3_index, k) if cell != h3_index]
    return [cell for cell in h3.k_ring(h3_index, k) if cell != h3_index]


def build_neighbor_features(hex_df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Add neighborhood record counts, average class, and class-share features."""
    output = hex_df.copy()
    lookup = output.set_index("h3_index")
    neighbor_rows = []
    for h3_index in output["h3_index"]:
        neighbors = [cell for cell in get_neighbor_hexes(h3_index, k) if cell in lookup.index]
        if not neighbors:
            row = {"h3_index": h3_index, "neighbor_record_count": 0, "neighbor_avg_class": np.nan}
            for klass in range(1, 11):
                row[f"neighbor_class_{klass}_share"] = 0.0
            neighbor_rows.append(row)
            continue
        neighbor_df = lookup.loc[neighbors]
        weights = neighbor_df["record_count"].to_numpy()
        row = {
            "h3_index": h3_index,
            "neighbor_record_count": int(weights.sum()),
            "neighbor_avg_class": float(np.average(neighbor_df["dominant_class"], weights=weights)),
        }
        for klass in range(1, 11):
            row[f"neighbor_class_{klass}_share"] = float(np.average(neighbor_df[f"class_{klass}_share"], weights=weights))
        neighbor_rows.append(row)
    neighbor_features = pd.DataFrame(neighbor_rows)
    output = output.merge(neighbor_features, on="h3_index", how="left")
    output["neighbor_avg_class"] = output["neighbor_avg_class"].fillna(output["dominant_class"])
    return output


def apply_minimum_volume_filter(hex_df: pd.DataFrame, min_records: int = 25) -> pd.DataFrame:
    """Keep hex cells with enough records for credible model training and review."""
    return hex_df[hex_df["record_count"] >= min_records].copy()


def prepare_modeling_dataset(hex_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and target from a hex-level feature table."""
    feature_cols = [
        "record_count",
        "avg_distance_to_service_center",
        "avg_urban_density_score",
        "low_quality_share",
        "neighbor_record_count",
        "neighbor_avg_class",
    ]
    feature_cols += [f"class_{klass}_share" for klass in range(1, 11)]
    feature_cols += [f"neighbor_class_{klass}_share" for klass in range(1, 11)]
    X = hex_df[feature_cols].fillna(0)
    y = hex_df["dominant_class"].astype(int)
    return X, y
