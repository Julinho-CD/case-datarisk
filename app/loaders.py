import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src import config as cfg
from src.data_access import default_refresh_flag, load_processed_datasets, resolve_data_source
from src.features import build_features

PROJECT_ROOT = cfg.PROJECT_ROOT
ARTIFACTS_DIR = cfg.ARTIFACTS_DIR
ARTIFACTS_RUNS_DIR = getattr(cfg, "ARTIFACTS_RUNS_DIR", ARTIFACTS_DIR / "runs")
BEST_MODEL_ARTIFACT_PATH = cfg.BEST_MODEL_ARTIFACT_PATH
BEST_RUN_ARTIFACT_PATH = cfg.BEST_RUN_ARTIFACT_PATH
FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH = cfg.FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH
FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH = cfg.FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH
FIG_RUNS_DIR = getattr(cfg, "FIG_RUNS_DIR", (cfg.REPORTS_DIR / "figures" / "runs"))
MODEL_COMPARISON_ARTIFACT_PATH = cfg.MODEL_COMPARISON_ARTIFACT_PATH
METRICS_RUNS_DIR = getattr(cfg, "METRICS_RUNS_DIR", (cfg.REPORTS_DIR / "metrics" / "runs"))
PR_CURVE_ARTIFACT_PATH = cfg.PR_CURVE_ARTIFACT_PATH
ROC_CURVE_ARTIFACT_PATH = cfg.ROC_CURVE_ARTIFACT_PATH
SHAP_SUMMARY_PNG_ARTIFACT_PATH = cfg.SHAP_SUMMARY_PNG_ARTIFACT_PATH
THRESHOLD_CURVE_ARTIFACT_PATH = cfg.THRESHOLD_CURVE_ARTIFACT_PATH
VAL_PREDICTIONS_BEST_ARTIFACT_PATH = cfg.VAL_PREDICTIONS_BEST_ARTIFACT_PATH


def safe_read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def best_run_id() -> str | None:
    payload = load_best_run()
    run_id = str(payload.get("run_id", "")).strip() if isinstance(payload, dict) else ""
    return run_id or None


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
        "split_test_size",
        "n_total_labeled",
        "n_train",
        "n_test",
        "train_share",
        "test_share",
        "train_positive_rate",
        "test_positive_rate",
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


def _run_metrics_path(run_id: str) -> Path:
    public_path = ARTIFACTS_RUNS_DIR / run_id / "val_predictions.csv"
    return public_path if public_path.exists() else (METRICS_RUNS_DIR / f"val_predictions_{run_id}.csv")


def _run_figures_dir(run_id: str) -> Path:
    public_path = ARTIFACTS_RUNS_DIR / run_id
    return public_path if public_path.exists() else (FIG_RUNS_DIR / run_id)


@st.cache_data
def load_run_val_predictions(run_id: str | None):
    if not run_id:
        return None
    df = _read_csv_artifact(_run_metrics_path(run_id), required_cols={"y_true", "y_prob"})
    if df is not None:
        return df
    if str(run_id).strip() == (best_run_id() or ""):
        return load_val_predictions_best()
    return None


@st.cache_data
def load_run_threshold_curve(run_id: str | None):
    val_df = load_run_val_predictions(run_id)
    if val_df is None or val_df.empty:
        return None

    y_true = val_df["y_true"].astype(int).to_numpy()
    y_prob = val_df["y_prob"].astype(float).to_numpy()
    rows = []
    for thr in [x / 1000 for x in range(50, 951, 5)]:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
                "positive_rate": float((y_pred == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_run_roc_curve(run_id: str | None):
    val_df = load_run_val_predictions(run_id)
    if val_df is None or val_df.empty:
        return None
    from sklearn.metrics import roc_curve

    y_true = val_df["y_true"].astype(int).to_numpy()
    y_prob = val_df["y_prob"].astype(float).to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


@st.cache_data
def load_run_pr_curve(run_id: str | None):
    val_df = load_run_val_predictions(run_id)
    if val_df is None or val_df.empty:
        return None
    from sklearn.metrics import precision_recall_curve

    y_true = val_df["y_true"].astype(int).to_numpy()
    y_prob = val_df["y_prob"].astype(float).to_numpy()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    padded_thresholds = list(thresholds) + [float("nan")]
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded_thresholds})


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


@st.cache_data
def load_run_feature_importance(run_id: str | None):
    if not run_id:
        return load_feature_importance()

    run_dir = _run_figures_dir(run_id)
    csv_path = run_dir / "feature_importance.csv"
    if csv_path.exists():
        df = _read_csv_artifact(csv_path, required_cols={"feature", "importance"})
    else:
        json_path = run_dir / "top_features.json"
        if not json_path.exists():
            if str(run_id).strip() == (best_run_id() or ""):
                return load_feature_importance()
            return None
        df = pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))

    if df is None or df.empty:
        return None
    df = df.copy()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["feature", "importance"]).sort_values("importance", ascending=False).reset_index(drop=True)
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


def load_top_features(_selected_row: dict | None = None, _run_id: str | None = None):
    df = load_run_feature_importance(_run_id)
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
