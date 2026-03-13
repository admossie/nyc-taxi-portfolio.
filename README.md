# NYC Taxi Portfolio

End-to-end portfolio project for NYC taxi demand analysis, feature engineering, modeling, and policy-impact assessment.

Built an end-to-end forecasting and pricing analysis pipeline on recent NYC taxi trip data to estimate ride demand and fare behavior under congestion pricing conditions. Included feature engineering, model benchmarking, temporal validation, and explainability.

## Executive Summary (Non-Technical)

This project builds a production-style analytics pipeline on NYC taxi trip data to predict fares and surface demand patterns for operational planning.

It combines robust data preparation, model benchmarking (baseline + boosted models), temporal validation, and SHAP explainability to produce reliable and interpretable outputs.

Key value: strong predictive performance, clear business insights on peak demand timing, and a deployment-ready Streamlit app for interactive prediction and decision support.

## Structure

- `data/raw/`: raw ingested source files
- `data/processed/`: cleaned and feature-ready datasets
- `notebooks/01_eda_and_modeling.ipynb`: EDA + baseline modeling notebook
- `src/data_prep.py`: data ingestion/cleaning pipeline
- `src/features.py`: feature engineering logic
- `src/train.py`: model training and evaluation script
- `outputs/figures/`: plots and charts
- `outputs/models/`: trained model artifacts

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run data prep: `python src/data_prep.py`
3. Build features: `python src/features.py`
4. Train model: `python src/train.py`

## Prediction App (Deployment Starter)

1. Train and save artifact (auto-downloads parquet if needed):
   - `python src/train.py --download-if-missing`
2. Start interactive prediction app:
   - `streamlit run app.py`

Artifacts written to `outputs/models/`:
- `best_fare_model.joblib`
- `best_fare_model_metadata.json`
- `boosted_model_comparison.csv`

## Best Next-Step Upgrades

### Version 1: Solid Portfolio Baseline

- Predict `total_amount`
- Show feature importance
- Show peak demand by hour/day
- Write a clean README with business takeaways

### Version 2: Stronger and More Differentiated

- Forecast hourly trip counts by pickup zone
- Compare pre/post congestion-fee patterns
- Add weather as an external feature
- Build a small dashboard in Streamlit

### Version 3: Cracked Portfolio Version

- Use time-based validation instead of random split
- Train XGBoost/LightGBM and compare to baseline
- Add SHAP explanations
- Deploy an app where user selects hour/zone and sees expected demand/fare

## Business Takeaways

- Trip-level fare baseline (RandomForest on January 2025 trips) achieved MAE = 2.012 and R² = 0.944.
- Peak pickup hours were 18, 17, 15, 16, and 19, highlighting strong evening demand concentration.
- Peak weekdays were Thursday, Friday, and Wednesday, which can guide staffing and supply positioning.
- In the daily policy summary, average trips increased from 118,135 (pre) to 125,976 (post) in the current analysis window.

## Results Snapshot

- Trip-level model: RandomForestRegressor
- Metrics: MAE 2.012, R² 0.944
- Top pickup hours: 18 (236,572), 17 (229,363), 15 (204,511), 16 (203,122), 19 (200,647)
- Top weekdays: Thursday (571,819), Friday (544,288), Wednesday (537,189)
- Policy comparison (daily aggregate): Pre 118,135 avg trips vs Post 125,976 avg trips
- Time-based validation (chronological split): MAE 1.825, R² 0.954
- Random split benchmark (same feature set): MAE 1.875, R² 0.950
- Rolling-origin backtesting (4 folds, expanding window): MAE mean 1.923 (std 0.076), R² mean 0.945 (std 0.006)
- Boosted-model comparison (out-of-time split, 350k sample):
  - XGBoost: MAE 1.884, R² 0.944
  - LightGBM: MAE 1.900, R² 0.949
  - RandomForest: MAE 1.929, R² 0.948
  - Best by MAE: XGBoost
- SHAP explainability (XGBoost): top global drivers are `trip_distance`, `trip_minutes`, `payment_type_2`, `cbd_congestion_fee`, and `pickup_hour`.

## Model Limitations and Next Validation Steps

- Time-based validation and rolling-origin backtesting are implemented on January 2025 trip-level data.
- Reported policy comparison is descriptive and does not control for confounders (seasonality, weather shifts, events).
- Results are based on the current analysis window and one month of trip-level modeling; performance may vary across months.
- Next step: compare RandomForest against XGBoost/LightGBM on out-of-time test windows.
- Next step: deploy an interactive app (hour/zone inputs) to expose predictions and explanations to non-technical users.

## Three-Version Upgrade Comparison

| Version | Scope | Status | Comparison |
|---|---|---|---|
| Version 1 (Solid Baseline) | Trip-level fare prediction, feature importance, peak demand analysis, business write-up | ✅ Completed | Delivers strong baseline quality and clear operational insights. |
| Version 2 (Differentiated) | Zone-level hourly demand forecasting, weather fusion, Streamlit dashboard | 🟡 Planned | Adds product usability and richer business context compared with Version 1. |
| Version 3 (Advanced) | Temporal validation, boosted model comparison, SHAP explainability, deployment | 🟡 In Progress | Temporal validation, boosted-model comparison, and SHAP are complete; deployment is next. |

## Appendix: Detailed Metrics

### Artifact Training Run (src/train.py)

- XGBoost: MAE 1.662, RMSE 4.167, R² 0.951
- LightGBM: MAE 1.689, RMSE 4.145, R² 0.951
- RandomForest: MAE 1.769, RMSE 4.127, R² 0.952
- Selection rule in `src/train.py`: best by MAE (currently XGBoost)

### SHAP Local Check

- Local explanation check (sample row): base value + SHAP contributions matches prediction (27.335 + contributions = 14.522)

## Portfolio Summary

This portfolio demonstrates end-to-end analytics execution: public data ingestion, rigorous preparation, predictive modeling with temporal validation, explainability, and deployment-ready outputs for real operational decision support.
