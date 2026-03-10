import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    BEST_MODEL_ARTIFACT_PATH,
    BEST_RUN_ARTIFACT_PATH,
    FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH,
    FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH,
    MODEL_COMPARISON_ARTIFACT_PATH,
    PR_CURVE_ARTIFACT_PATH,
    PROJECT_ROOT,
    ROC_CURVE_ARTIFACT_PATH,
    SHAP_SUMMARY_PNG_ARTIFACT_PATH,
    THRESHOLD_CURVE_ARTIFACT_PATH,
    VAL_PREDICTIONS_BEST_ARTIFACT_PATH,
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


def _read_csv_artifact(path: Path, required_cols: set[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if required_cols and not required_cols.issubset(df.columns):
        return None
    return df


@st.cache_data
def load_model_comparison():
    df = _read_csv_artifact(MODEL_COMPARISON_ARTIFACT_PATH)
    if df is None:
        return None

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
        "cv_roc_auc_mean",
        "cv_pr_auc_mean",
        "use_smote",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required_cols = [c for c in ["run_id", "model_name", "pr_auc", "roc_auc"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    if "pr_auc" in df.columns:
        df = df.sort_values(["pr_auc", "roc_auc"], ascending=False)
    return df.reset_index(drop=True)


@st.cache_data
def load_best_run():
    return safe_read_json(BEST_RUN_ARTIFACT_PATH, default={})


@st.cache_data
def load_val_predictions_best():
    return _read_csv_artifact(VAL_PREDICTIONS_BEST_ARTIFACT_PATH, required_cols={"y_true", "y_prob"})


@st.cache_data
def load_threshold_curve():
    return _read_csv_artifact(
        THRESHOLD_CURVE_ARTIFACT_PATH,
        required_cols={"threshold", "precision", "recall", "f1", "fpr", "tp", "tn", "fp", "fn"},
    )


@st.cache_data
def load_roc_curve():
    return _read_csv_artifact(ROC_CURVE_ARTIFACT_PATH, required_cols={"fpr", "tpr"})


@st.cache_data
def load_pr_curve():
    return _read_csv_artifact(PR_CURVE_ARTIFACT_PATH, required_cols={"precision", "recall"})


@st.cache_data
def load_feature_importance():
    df = _read_csv_artifact(FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH, required_cols={"feature", "importance"})
    if df is None:
        return None
    if "rank" not in df.columns:
        df = df.copy()
        df.insert(0, "rank", range(1, len(df) + 1))
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    return df.dropna(subset=["feature", "importance"]).reset_index(drop=True)


def load_top_features(_selected_row: dict | None = None, _run_id: str | None = None):
    df = load_feature_importance()
    if df is None or df.empty:
        return []
    return df[["feature", "importance"]].to_dict(orient="records")


@st.cache_resource
def load_public_model():
    if not BEST_MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing public model artifact: {BEST_MODEL_ARTIFACT_PATH}")
    return joblib.load(BEST_MODEL_ARTIFACT_PATH), str(BEST_MODEL_ARTIFACT_PATH)


def get_feature_importance_image_path() -> Path | None:
    return FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH if FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH.exists() else None


def get_shap_summary_image_path() -> Path | None:
    return SHAP_SUMMARY_PNG_ARTIFACT_PATH if SHAP_SUMMARY_PNG_ARTIFACT_PATH.exists() else None
