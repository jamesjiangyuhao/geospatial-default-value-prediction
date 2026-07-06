"""Generate synthetic location-tagged records for geospatial modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 45_000


def _bounded_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic points around a Houston, Texas metro bounding box."""
    lat_min, lat_max = 29.35, 30.25
    lon_min, lon_max = -95.90, -94.85

    centers = np.array(
        [
            [29.7604, -95.3698],  # Houston core
            [29.6197, -95.6349],  # Sugar Land
            [30.1658, -95.4613],  # The Woodlands
            [29.7355, -94.9774],  # Baytown
            [29.5636, -95.2860],  # Pearland
        ]
    )
    weights = np.array([0.50, 0.16, 0.13, 0.10, 0.11])
    cluster = rng.choice(len(centers), size=n, p=weights)
    lat = rng.normal(centers[cluster, 0], rng.uniform(0.035, 0.13, n))
    lon = rng.normal(centers[cluster, 1], rng.uniform(0.045, 0.18, n))

    boundary_mask = rng.random(n) < 0.12
    lat[boundary_mask] = rng.uniform(lat_min, lat_max, boundary_mask.sum())
    lon[boundary_mask] = rng.uniform(lon_min, lon_max, boundary_mask.sum())

    return np.clip(lat, lat_min, lat_max), np.clip(lon, lon_min, lon_max)


def build_synthetic_records(row_count: int = ROW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return deterministic synthetic records with spatially structured classes."""
    rng = np.random.default_rng(seed)
    latitude, longitude = _bounded_points(rng, row_count)

    urban_centers = np.array(
        [
            [29.7604, -95.3698],
            [29.6197, -95.6349],
            [30.1658, -95.4613],
        ]
    )
    distances = np.sqrt(((latitude[:, None] - urban_centers[:, 0]) * 69) ** 2 + ((longitude[:, None] - urban_centers[:, 1]) * 60) ** 2)
    nearest_urban = distances.min(axis=1)
    urban_density_score = np.clip(100 * np.exp(-nearest_urban / 38) + rng.normal(0, 8, row_count), 0, 100)
    distance_to_service_center = np.clip(nearest_urban + rng.gamma(1.6, 4.0, row_count), 0.2, 180)

    sparse_factor = np.clip((distance_to_service_center - 35) / 95, 0, 1)
    historical_record_count = np.maximum(1, rng.poisson(28 * (1 - sparse_factor) + 4)).astype(int)

    base_class = 8.4 - 0.055 * urban_density_score + 0.018 * distance_to_service_center
    base_class += 0.45 * np.sin((longitude + 95.35) * 9.5) + 0.35 * np.cos((latitude - 29.75) * 10.0)

    pockets = [
        (29.76, -95.37, -1.5, 0.08),
        (29.63, -95.62, -0.8, 0.07),
        (29.79, -95.02, 1.1, 0.08),
        (30.12, -95.30, 0.9, 0.09),
        (29.47, -95.18, 0.8, 0.07),
    ]
    for pocket_lat, pocket_lon, effect, scale in pockets:
        pocket_dist = np.sqrt((latitude - pocket_lat) ** 2 + (longitude - pocket_lon) ** 2)
        base_class += effect * np.exp(-(pocket_dist**2) / (2 * scale**2))

    categorical_attribute = np.rint(base_class + rng.normal(0, 0.9, row_count)).astype(int)
    categorical_attribute = np.clip(categorical_attribute, 1, 10)

    periods = pd.date_range("2022-01-01", "2025-12-01", freq="MS").strftime("%Y-%m")
    data_quality_flag = np.where(rng.random(row_count) < 0.075, "Low Quality", "Standard")
    noisy = data_quality_flag == "Low Quality"
    categorical_attribute[noisy] = np.clip(categorical_attribute[noisy] + rng.integers(-3, 4, noisy.sum()), 1, 10)

    market_area = np.select(
        [
            latitude >= 29.95,
            latitude <= 29.58,
            longitude >= -95.15,
            longitude <= -95.58,
        ],
        ["North", "South", "East", "West"],
        default="Central",
    )

    return pd.DataFrame(
        {
            "record_id": [f"REC{100000 + i}" for i in range(row_count)],
            "latitude": np.round(latitude, 6),
            "longitude": np.round(longitude, 6),
            "observation_period": rng.choice(periods, row_count),
            "region": "Synthetic Texas Metro Region",
            "market_area": market_area,
            "categorical_attribute": categorical_attribute,
            "record_source": rng.choice(["Source A", "Source B", "Source C"], row_count, p=[0.52, 0.31, 0.17]),
            "property_type": rng.choice(["Type 1", "Type 2", "Type 3"], row_count, p=[0.58, 0.28, 0.14]),
            "distance_to_service_center": np.round(distance_to_service_center, 3),
            "urban_density_score": np.round(urban_density_score, 3),
            "historical_record_count": historical_record_count,
            "data_quality_flag": data_quality_flag,
        }
    )


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_location_records.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_records()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} records to {output_path}")


if __name__ == "__main__":
    main()
