# Geospatial Default Value Prediction Using Hex Grids and Random Forests

## Overview

This repository contains a fully synthetic geospatial machine learning project that predicts a default categorical value for hexagonal geographic grid cells. It demonstrates spatial aggregation, neighborhood-based feature engineering, random forest classification, confidence-gated recommendations, and interactive map-based review.

## Business Problem

Some location-based operational workflows need a reasonable default categorical value when the true value is missing, uncertain, or slow to resolve. A static default is easy to maintain but often too blunt. This project shows how nearby historical observations can support a more localized default recommendation while still flagging low-confidence areas for manual review or fallback.

## Why This Project Matters

The workflow is designed around a real decision-support pattern: maximize useful coverage while avoiding overconfident recommendations in sparse areas. It combines model output, observation volume, and confidence thresholds so stakeholders can decide where to trust the model and where to review manually.

## Synthetic Data Design

The data is generated from random coordinates inside a realistic U.S. regional bounding box. It includes synthetic observation periods, market areas, record sources, property types, distance-to-service-center features, urban density scores, historical record counts, and quality flags.

Spatial patterns are intentionally built in:

- Nearby records often share similar categorical values.
- Urban areas tend to have lower class values.
- Rural or sparse areas tend to have higher class values.
- Local pockets deviate from the surrounding pattern.
- Boundary regions contain fewer observations.
- A small share of low-quality records adds noise.

## Methodology

1. Generate synthetic location-tagged records.
2. Assign records to H3 hexagonal grid cells.
3. Aggregate class distributions and numeric features at the hex level.
4. Add neighboring-cell features to capture spatial context.
5. Filter to credible hex cells using minimum record volume.
6. Train a random forest classifier to predict the dominant class.
7. Attach predicted probabilities and confidence labels.
8. Review predictions and confidence on interactive maps.

## Hex-Grid Feature Engineering

Each hex cell receives features such as record count, observed class shares, average distance to service center, average urban density score, low-quality share, neighbor record count, neighbor average class, and neighbor class shares.

## Model Design

The model uses `RandomForestClassifier` because it handles nonlinear interactions, mixed feature scales, and categorical targets with minimal preprocessing. The project evaluates out-of-sample accuracy, macro F1, weighted F1, macro precision, confusion matrix, and feature importance.

## Confidence-Gated Recommendations

Predictions are grouped into:

- High Confidence: probability >= 0.70
- Medium Confidence: probability >= 0.50 and < 0.70
- Low Confidence: probability < 0.50

Recommendation statuses are controlled by a configurable threshold:

- Use Model Recommendation
- Review Manually
- Fallback Default

## Dashboard Features

- H3 resolution, volume threshold, neighbor ring, and confidence threshold controls
- Raw record, hex-cell, and credible-cell KPI cards
- Interactive prediction and confidence maps
- Model performance summaries
- Coverage versus confidence curve
- Segment diagnostics by market area, source, and property type
- Automated business interpretation
- Prediction table download

## Repository Structure

```text
geospatial-default-value-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── synthetic_location_records.csv
│   └── hex_grid_predictions.csv
├── src/
│   ├── generate_synthetic_data.py
│   ├── geospatial_features.py
│   ├── model.py
│   ├── evaluation.py
│   └── visualization.py
├── notebooks/
│   └── geospatial_default_prediction_analysis.ipynb
└── outputs/
    ├── prediction_map.html
    ├── confidence_map.html
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── coverage_confidence_curve.png
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py
streamlit run app.py
```

The Streamlit app regenerates synthetic data automatically if the CSV file is missing.

## Example Insights

- Sparse hex cells tend to have lower prediction confidence.
- Neighbor-derived features help smooth noisy local observations.
- Higher confidence thresholds increase reliability but reduce coverage.
- Certain market areas require more manual review because their observation volume is lower.
- Feature importance can reveal whether local density, neighbor class patterns, or observed class shares are driving predictions.

## Skills Demonstrated

- Geospatial data engineering
- Hexagonal grid indexing
- Spatial feature engineering
- Supervised classification
- Random forest modeling
- Model evaluation
- Accuracy, F1, and confusion matrix interpretation
- Confidence threshold design
- Coverage versus confidence tradeoff analysis
- Interactive map visualization
- Streamlit dashboard development
- Product analytics and decision-support thinking
- Stakeholder-oriented storytelling

## Confidentiality Note

This project is built entirely with synthetic data and generic business logic. It does not contain any proprietary employer data, code, table names, rating logic, pricing logic, business rules, screenshots, URLs, customer information, or confidential information.

## Future Enhancements

- Add calibration analysis for predicted probabilities.
- Add polygon rendering for full H3 hex boundaries.
- Add drift monitoring across observation periods.
- Add threshold optimization based on business cost assumptions.
- Add unit tests for H3 assignment and feature engineering.
