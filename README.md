# Datarisk Junior Data Scientist Case - Portfolio Extension

This repository extends the original Datarisk Junior Data Scientist technical challenge into a portfolio-ready project with a bilingual Streamlit app, business decision framing, MLflow-based model loading, and a public-demo data strategy.

## Original Challenge

- Official challenge repository: <https://github.com/datarisk-io/datarisk-case-ds-junior>
- Predictions must be generated for every record in `base_pagamentos_teste.csv`.
- In the challenge, delinquency is defined as a payment made `5` or more days after its due date.
- The original submission came first. This repository is the portfolio extension built on top of that solution.

## Public Data Policy

This repository does not redistribute the original challenge data.

- Source of truth for raw data: the official Datarisk repository.
- Public app mode: raw CSVs are downloaded at runtime from the official source.
- Local compatibility: if you already have the official CSVs under `data/raw/`, the project will use them first.
- Download cache: files are cached under `.cache/` to avoid repeated downloads.

See `data/README.md` for the expected local structure.

## Official Portfolio Artifact

The repository is intentionally consolidated around one official final run:

- Final model: `Random Forest`
- Variant: `No SMOTE`
- Official run ID: `56e3d7b850c24eb78a05b2a232f30fd5`
- Selection metric: `PR-AUC`
- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- Recommended threshold: `0.20`
- Precision at threshold: `0.6352`
- Recall at threshold: `0.7411`
- Positive rate at threshold: `0.0777`

## What The Project Demonstrates

- End-to-end machine learning workflow in Python.
- Temporal validation and benchmark comparison.
- Threshold tuning tied to operational trade-offs.
- Business communication for recruiters, technical reviewers, and managers.
- Streamlit application ready for public demonstration.

## Public Demo Mode

The app is structured for public deployment:

- raw data is downloaded from the official Datarisk repository at runtime;
- processed tables are cached locally after the first load;
- precomputed benchmark and explainability artifacts from this repository are reused;
- no retraining is required in the deployment environment;
- model binaries are expected to come from MLflow for the full prediction experience.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.make_dataset
python -m src.train
python -m src.evaluate
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Optional environment variables:

```bash
set DATARISK_DATA_SOURCE=auto
set DATARISK_DATA_REFRESH=0
set MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:5000
set MLFLOW_EXPERIMENT_NAME=datarisk-inadimplencia
```

## Deploy To Streamlit Community Cloud

- Main app file: `streamlit_app.py`
- Python runtime: `runtime.txt` -> `python-3.11`
- Dependencies: `requirements.txt`
- Sanity check command: `python -m src.sanity_check`

Recommended deployment flow:

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Select `streamlit_app.py` as the entrypoint.
4. Add `MLFLOW_TRACKING_URI` and, if needed, `MLFLOW_MODEL_URI` in the app secrets or environment settings.
5. Deploy and validate the app pages after the first data download completes.

## Public App Link

Add the public demo link here once available:

- `Streamlit Demo: <add-your-link-here>`

## Sanity Check Before Deploy

```bash
python -m src.sanity_check
```

The script validates:

- core imports;
- official Datarisk data availability;
- readability of required project artifacts;
- basic Streamlit entrypoint compilation.

## Repository Structure For Deploy

```text
.
+-- app/
+-- data/
|   +-- README.md
+-- docs/
+-- models/
+-- notebooks/
+-- reports/
+-- src/
+-- tests/
+-- requirements.txt
+-- runtime.txt
+-- streamlit_app.py
```

## Risks To Review Before Publishing

- Public demo prediction requires MLflow connectivity if you want the scoring page fully active.
- First load in Streamlit Cloud will be slower because the raw data is downloaded and cached.
- If the official Datarisk repository changes file names or paths, the remote loader must be updated.

## Additional Documents

- English model card: `MODEL_CARD.md`
- Portuguese model card: `MODEL_CARD.pt-BR.md`
- Executive summary in English: `docs/executive_summary.md`
- Executive summary in Portuguese: `docs/executive_summary.pt-BR.md`
