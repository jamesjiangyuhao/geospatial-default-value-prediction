"""Streamlit dashboard for synthetic geospatial default value prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

from evaluation import evaluate_classifier, summarize_threshold_tradeoffs
from generate_synthetic_data import main as generate_data
from geospatial_features import (
    aggregate_to_hex,
    apply_minimum_volume_filter,
    assign_h3_index,
    build_neighbor_features,
    prepare_modeling_dataset,
)
from model import create_prediction_table, predict_with_confidence, train_random_forest
from visualization import (
    create_class_distribution_chart,
    create_confidence_distribution_chart,
    create_confidence_map,
    create_prediction_map,
    create_volume_vs_confidence_chart,
)


DATA_PATH = ROOT / "data" / "synthetic_location_records.csv"
PREDICTION_PATH = ROOT / "data" / "hex_grid_predictions.csv"


st.set_page_config(page_title="Geospatial Default Value Prediction", layout="wide")


@st.cache_data(show_spinner=False)
def load_records() -> pd.DataFrame:
    if not DATA_PATH.exists():
        generate_data()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def build_pipeline(records: pd.DataFrame, resolution: int, min_records: int, neighbor_k: int, confidence_threshold: float):
    assigned = assign_h3_index(records, resolution)
    hex_df = aggregate_to_hex(assigned)
    hex_df = build_neighbor_features(hex_df, k=neighbor_k)
    credible = apply_minimum_volume_filter(hex_df, min_records=min_records)
    if credible["dominant_class"].nunique() < 2 or len(credible) < 20:
        raise ValueError("Not enough credible hex cells to train a classifier. Lower the minimum record threshold.")
    X, y = prepare_modeling_dataset(credible)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    clf = train_random_forest(X_train, y_train)
    metrics = evaluate_classifier(clf, X_test, y_test)
    predictions, probabilities = predict_with_confidence(clf, X)
    prediction_df = create_prediction_table(credible, predictions, probabilities, confidence_threshold)
    return assigned, hex_df, credible, prediction_df, metrics, clf, X.columns.tolist()


def pct(value: float) -> str:
    return f"{value:.1%}"


def build_insights(prediction_df: pd.DataFrame, feature_names: list[str], model, confidence_threshold: float) -> list[str]:
    insights = []
    low_conf = prediction_df[prediction_df["recommendation_status"] != "Use Model Recommendation"]
    if not low_conf.empty:
        area = low_conf.groupby("market_area").size().sort_values(ascending=False).index[0]
        insights.append(f"{area} has the largest count of cells requiring review or fallback under the current threshold.")
    sparse = prediction_df.assign(volume_band=pd.qcut(prediction_df["record_count"], q=3, duplicates="drop"))
    if sparse["volume_band"].nunique() > 1:
        band_summary = sparse.groupby("volume_band", observed=False)["predicted_probability"].mean()
        insights.append(f"Lower-volume cells average {band_summary.iloc[0]:.2f} confidence versus {band_summary.iloc[-1]:.2f} for higher-volume cells.")
    top_feature = feature_names[int(model.feature_importances_.argmax())]
    insights.append(f"The strongest model feature in this run is `{top_feature}`, indicating which signal most shaped the classifier.")
    coverage = (prediction_df["predicted_probability"] >= confidence_threshold).mean()
    insights.append(f"At the selected confidence threshold, {pct(coverage)} of credible hex cells receive a model recommendation.")
    if "neighbor_avg_class" in feature_names:
        insights.append("Neighbor-derived features are included so each hex cell can borrow spatial context from adjacent cells.")
    return insights[:6]


records = load_records()

st.title("Geospatial Default Value Prediction Dashboard")
st.caption(
    "Synthetic portfolio project demonstrating hex-grid feature engineering, random forest classification, "
    "confidence-gated recommendations, and interactive map review."
)

with st.sidebar:
    st.header("Controls")
    resolution = st.slider("H3 resolution", 4, 8, 6)
    min_records = st.slider("Minimum record count threshold", 5, 100, 25)
    neighbor_k = st.slider("Neighbor ring size", 1, 3, 1)
    confidence_threshold = st.slider("Confidence threshold", 0.40, 0.90, 0.70, 0.05)
    periods = st.multiselect("Observation period", sorted(records["observation_period"].unique()), default=sorted(records["observation_period"].unique()))
    areas = st.multiselect("Market area", sorted(records["market_area"].unique()), default=sorted(records["market_area"].unique()))
    sources = st.multiselect("Record source", sorted(records["record_source"].unique()), default=sorted(records["record_source"].unique()))
    types = st.multiselect("Property type", sorted(records["property_type"].unique()), default=sorted(records["property_type"].unique()))

filtered = records[
    records["observation_period"].isin(periods)
    & records["market_area"].isin(areas)
    & records["record_source"].isin(sources)
    & records["property_type"].isin(types)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

try:
    assigned, hex_df, credible, prediction_df, metrics, clf, feature_names = build_pipeline(
        filtered, resolution, min_records, neighbor_k, confidence_threshold
    )
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
prediction_df.to_csv(PREDICTION_PATH, index=False)

st.subheader("Project Overview")
st.write(
    "A location-based categorical attribute is sometimes missing or uncertain. This application uses nearby historical "
    "observations and spatial features to recommend a default value for each hex grid cell while flagging low-confidence "
    "areas for manual review."
)

st.subheader("KPI Summary")
cols = st.columns(7)
cols[0].metric("Raw Records", f"{len(filtered):,}")
cols[1].metric("Hex Cells", f"{len(hex_df):,}")
cols[2].metric("Credible Hex Cells", f"{len(credible):,}")
cols[3].metric("Accuracy", pct(metrics["accuracy"]))
cols[4].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
cols[5].metric("High Confidence", pct((prediction_df["confidence_label"] == "High Confidence").mean()))
cols[6].metric("Review/Fallback", pct((prediction_df["recommendation_status"] != "Use Model Recommendation").mean()))

st.subheader("Interactive Prediction Map")
st.plotly_chart(create_prediction_map(prediction_df), width="stretch")

st.subheader("Interactive Confidence Map")
st.plotly_chart(create_confidence_map(prediction_df), width="stretch")

st.subheader("Model Performance")
perf_cols = st.columns([1, 1])
with perf_cols[0]:
    st.write(
        {
            "Accuracy": round(metrics["accuracy"], 3),
            "Macro F1": round(metrics["macro_f1"], 3),
            "Weighted F1": round(metrics["weighted_f1"], 3),
            "Macro Precision": round(metrics["macro_precision"], 3),
        }
    )
    st.plotly_chart(create_class_distribution_chart(credible), width="stretch")
with perf_cols[1]:
    importance = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
    st.dataframe(importance.head(15), width="stretch")
    st.plotly_chart(create_confidence_distribution_chart(prediction_df), width="stretch")

st.subheader("Coverage Versus Confidence")
curve = summarize_threshold_tradeoffs(prediction_df)
st.line_chart(curve.set_index("confidence_threshold")["coverage_rate"])
st.dataframe(curve, width="stretch")

st.subheader("Segment Diagnostics")
for group_col in ["market_area", "record_source", "property_type"]:
    if group_col in assigned.columns:
        joined = assigned[["h3_index", group_col]].merge(prediction_df, on="h3_index", how="inner")
        grouped = joined.groupby(group_col).agg(
            record_count=("record_count", "sum"),
            average_predicted_probability=("predicted_probability", "mean"),
            high_confidence_share=("confidence_label", lambda x: (x == "High Confidence").mean()),
            most_common_predicted_class=("predicted_class", lambda x: x.mode().iat[0]),
        )
        st.write(f"By {group_col}")
        st.dataframe(grouped.reset_index(), width="stretch")

st.subheader("Business Interpretation")
for insight in build_insights(prediction_df, feature_names, clf, confidence_threshold):
    st.write(f"- {insight}")

st.subheader("Data Preview and Download")
tabs = st.tabs(["Raw Records", "Hex Features", "Prediction Table"])
tabs[0].dataframe(filtered.head(500), width="stretch")
tabs[1].dataframe(credible.head(500), width="stretch")
tabs[2].dataframe(prediction_df.head(500), width="stretch")
st.download_button(
    "Download prediction table",
    prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="hex_grid_predictions.csv",
    mime="text/csv",
)
