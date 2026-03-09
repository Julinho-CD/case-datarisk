# Streamlit App Guide

## Purpose

The app is designed for public portfolio demonstration without shipping the original challenge datasets in Git.

- raw data comes from the official Datarisk repository at runtime;
- processed tables are cached locally after the first load;
- benchmark and explainability artifacts are read from the repository;
- no retraining happens in the deployment environment;
- model loading is MLflow-first for the scoring page.

## Main File

- Streamlit entrypoint for deployment: `streamlit_app.py`
- Internal app module: `app/streamlit_app.py`

## Data Loading Flow

1. Try local official CSVs under `data/raw/`.
2. If they do not exist, download the official files from the Datarisk repository.
3. Cache the downloaded raw files under `.cache/`.
4. Build processed train and test tables in memory and cache them.

## Required Repository Artifacts

- `reports/metrics/model_comparison.csv`
- `reports/metrics/model_info.json`
- `reports/metrics/best_run.json`
- `reports/metrics/validation_metrics.json`
- `reports/metrics/runs/val_predictions_56e3d7b850c24eb78a05b2a232f30fd5.csv`
- `reports/figures/runs/56e3d7b850c24eb78a05b2a232f30fd5/top_features.json`

## Environment Variables

- `DATARISK_DATA_SOURCE=auto|local|remote`
- `DATARISK_DATA_REFRESH=0|1`
- `MLFLOW_TRACKING_URI=http://...`
- `MLFLOW_MODEL_URI=models:/...` (optional)

## Recommended Public Mode

- `DATARISK_DATA_SOURCE=remote`
- `DATARISK_DATA_REFRESH=0`
- `MLFLOW_TRACKING_URI` configured in the deployment platform
