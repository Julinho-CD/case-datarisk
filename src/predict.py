import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import (
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    METRICS_DIR,
    MODELS_RUNS_DIR,
    TARGET_COL,
)
from src.data_access import load_processed_datasets
from src.features import build_features


def load_processed():
    return load_processed_datasets()


def resolve_model_path(run_id: str | None) -> Path:
    if run_id:
        p = MODELS_RUNS_DIR / f"model_{run_id}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model for run_id not found: {p}")
        return p
    return MODEL_PATH


def resolve_best_run_id() -> str | None:
    model_info_path = METRICS_DIR / "model_info.json"
    if model_info_path.exists():
        model_info = pd.read_json(model_info_path, typ="series")
        best_run_id = str(model_info.get("best_run_id", "")).strip()
        if best_run_id:
            return best_run_id

    if MODEL_COMPARISON_PATH.exists():
        comp = pd.read_csv(MODEL_COMPARISON_PATH)
        if not comp.empty:
            return str(comp.iloc[0]["run_id"]).strip()

    return None


def load_model(run_id: str | None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    resolved_run_id = run_id or resolve_best_run_id()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri:
        return mlflow.sklearn.load_model(model_uri)

    if tracking_uri and resolved_run_id:
        return mlflow.sklearn.load_model(f"runs:/{resolved_run_id}/model")

    return joblib.load(resolve_model_path(run_id))


def main(out_path: str, run_id: str | None = None):
    model = load_model(run_id)

    train_raw, test_raw = load_processed()
    _, test_fe = build_features(train_raw, test_raw)

    drop = [TARGET_COL, "DIAS_ATRASO", "DATA_PAGAMENTO"]
    X_test = test_fe.drop(columns=[c for c in drop if c in test_fe.columns], errors="ignore")

    prob = model.predict_proba(X_test)[:, 1]

    submission = test_fe[["ID_CLIENTE", "SAFRA_REF"]].copy()
    submission[TARGET_COL] = prob

    if len(submission) != len(test_raw):
        raise ValueError(
            "Prediction output row count does not match the processed version of base_pagamentos_teste.csv. "
            "The case expects one probability for each record from base_pagamentos_teste.csv."
        )

    submission.to_csv(out_path, index=False)
    print(f"OK: submission saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submissao_case.csv")
    parser.add_argument("--run-id", default=None, help="(Optional) MLflow run_id to use the model from that run")
    args = parser.parse_args()

    main(args.out, args.run_id)
