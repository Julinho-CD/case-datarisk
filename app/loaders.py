import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st

from src.config import (
    FIG_DIR,
    FIG_RUNS_DIR,
    METRICS_DIR,
    METRICS_RUNS_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_RUNS_DIR,
    PROJECT_ROOT,
)
from src.data_access import default_refresh_flag, load_processed_datasets, resolve_data_source
from src.features import build_features


def safe_read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def current_data_settings() -> tuple[str, bool]:
    source = resolve_data_source(os.getenv("DATARISK_DATA_SOURCE"))
    refresh = default_refresh_flag()
    return source, refresh


@st.cache_data
def load_processed(source: str | None = None, refresh: bool = False):
    try:
        return load_processed_datasets(source=resolve_data_source(source), refresh=refresh)
    except Exception:
        return None, None


@st.cache_data
def load_feature_data(source: str | None = None, refresh: bool = False):
    train, test = load_processed(source=source, refresh=refresh)
    if train is None or test is None:
        return None, None
    return build_features(train, test)


@st.cache_data
def load_model_comparison():
    if not MODEL_COMPARISON_PATH.exists():
        return None
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    numeric_cols = [
        "pr_auc",
        "roc_auc",
        "f1_at_05",
        "best_threshold",
        "f1_best_threshold",
        "precision_best_threshold",
        "recall_best_threshold",
        "positive_rate_best_threshold",
        "use_smote",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required_cols = [c for c in ["run_id", "model_name", "pr_auc", "roc_auc"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    if "pr_auc" in df.columns:
        df = df.sort_values(["pr_auc", "roc_auc"], ascending=False)
    return df.reset_index(drop=True)


@st.cache_data
def load_val_predictions_for_run(run_id: str):
    p = METRICS_RUNS_DIR / f"val_predictions_{run_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if not {"y_true", "y_prob"}.issubset(df.columns):
        return None
    return df


@st.cache_data
def load_json_artifact(path_like: str | Path, default=None):
    return safe_read_json(resolve_project_path(path_like), default=default)


def load_top_features(selected_row: dict | None, run_id: str | None):
    path_candidates: list[Path] = []

    if selected_row and selected_row.get("run_fig_dir"):
        path_candidates.append(resolve_project_path(str(selected_row["run_fig_dir"])) / "top_features.json")
    if run_id:
        path_candidates.append(FIG_RUNS_DIR / str(run_id) / "top_features.json")
    path_candidates.append(FIG_DIR / "top_features.json")

    for p in path_candidates:
        data = load_json_artifact(p, default=None)
        if data:
            return data
    return []


def _load_model_from_mlflow(run_id: str | None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_uri = os.getenv("MLFLOW_MODEL_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_uri, "mlflow_model_uri"

    if run_id:
        run_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(run_uri)
        return model, run_uri, "mlflow_run"

    raise FileNotFoundError("No MLflow model URI or run ID available.")


@st.cache_resource
def load_model_for_run(run_id: str | None):
    # If tracking URI is configured, prefer MLflow model loading first.
    if os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_MODEL_URI"):
        try:
            model, uri, source = _load_model_from_mlflow(run_id)
            return model, uri, False, source
        except Exception:
            pass

    if run_id:
        run_path = MODELS_RUNS_DIR / f"model_{run_id}.joblib"
        if run_path.exists():
            return joblib.load(run_path), str(run_path), False, "local_run"

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), str(MODEL_PATH), True, "local_fallback"

    try:
        model, uri, source = _load_model_from_mlflow(run_id)
        return model, uri, False, source
    except Exception as exc:
        raise FileNotFoundError(
            "No local model artifact found and MLflow loading failed. "
            "Set MLFLOW_TRACKING_URI and optionally MLFLOW_MODEL_URI, or run local training."
        ) from exc


@st.cache_data
def load_model_info():
    return load_json_artifact(METRICS_DIR / "model_info.json", default={})
