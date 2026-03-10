import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_ARTIFACT_PATH,
    BEST_RUN_ARTIFACT_PATH,
    FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH,
    FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH,
    FIG_DIR,
    FIG_RUNS_DIR,
    METRICS_DIR,
    METRICS_RUNS_DIR,
    MODEL_COMPARISON_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_RUNS_DIR,
    PR_CURVE_ARTIFACT_PATH,
    ROC_CURVE_ARTIFACT_PATH,
    SHAP_SUMMARY_PNG_ARTIFACT_PATH,
    THRESHOLD_CURVE_ARTIFACT_PATH,
    VAL_PREDICTIONS_BEST_ARTIFACT_PATH,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUBLIC_COMPARISON_COLUMNS = [
    "run_id",
    "model_key",
    "model_name",
    "use_smote",
    "roc_auc",
    "pr_auc",
    "f1_at_05",
    "best_threshold",
    "f1_best_threshold",
    "precision_best_threshold",
    "recall_best_threshold",
    "positive_rate_best_threshold",
    "cv_roc_auc_mean",
    "cv_pr_auc_mean",
]
MODEL_COMPRESSION = ("xz", 9)


def normalize_feature_importance_df(feature_importance: pd.DataFrame | list[dict] | None) -> pd.DataFrame:
    if feature_importance is None:
        return pd.DataFrame(columns=["rank", "feature", "importance"])

    df = pd.DataFrame(feature_importance).copy()
    if df.empty:
        return pd.DataFrame(columns=["rank", "feature", "importance"])

    available = [c for c in ["feature", "importance"] if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame(columns=["rank", "feature", "importance"])

    df = df[available].copy()
    df["feature"] = df["feature"].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["feature", "importance"]).sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def load_feature_importance_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["rank", "feature", "importance"])
    return normalize_feature_importance_df(json.loads(path.read_text(encoding="utf-8")))


def compute_threshold_curve_df(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for thr in np.linspace(0.05, 0.95, 181):
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


def compute_roc_curve_df(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def compute_pr_curve_df(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    padded_thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded_thresholds})


def copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def create_summary_figure(feature_df: pd.DataFrame, out_path: Path, title: str):
    if feature_df.empty:
        return
    plot_df = feature_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#1D4ED8")
    plt.title(title)
    plt.xlabel("importance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def export_public_artifacts(
    best_run_payload: dict[str, Any],
    comparison_df: pd.DataFrame,
    val_predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame | list[dict] | None,
    best_fig_dir: Path | None = None,
    best_model: Any | None = None,
    best_model_path: Path | None = None,
) -> dict[str, bool]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_comparison = comparison_df.copy()
    available_columns = [c for c in PUBLIC_COMPARISON_COLUMNS if c in clean_comparison.columns]
    if available_columns:
        clean_comparison = clean_comparison[available_columns].copy()
    clean_comparison.to_csv(MODEL_COMPARISON_ARTIFACT_PATH, index=False)

    BEST_RUN_ARTIFACT_PATH.write_text(
        json.dumps(best_run_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    val_df = val_predictions_df.copy()
    required_cols = {"y_true", "y_prob"}
    if not required_cols.issubset(val_df.columns):
        raise ValueError(f"Validation predictions must contain {sorted(required_cols)}.")
    val_df.to_csv(VAL_PREDICTIONS_BEST_ARTIFACT_PATH, index=False)

    y_true = val_df["y_true"].astype(int).to_numpy()
    y_prob = val_df["y_prob"].astype(float).to_numpy()
    compute_threshold_curve_df(y_true, y_prob).to_csv(THRESHOLD_CURVE_ARTIFACT_PATH, index=False)
    compute_roc_curve_df(y_true, y_prob).to_csv(ROC_CURVE_ARTIFACT_PATH, index=False)
    compute_pr_curve_df(y_true, y_prob).to_csv(PR_CURVE_ARTIFACT_PATH, index=False)

    clean_feature_importance = normalize_feature_importance_df(feature_importance_df)
    clean_feature_importance.to_csv(FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH, index=False)

    feature_importance_created = False
    shap_summary_created = False
    if best_fig_dir is not None:
        feature_importance_created = copy_if_exists(
            best_fig_dir / "feature_importance.png",
            FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH,
        )
        shap_summary_created = copy_if_exists(best_fig_dir / "shap_summary.png", SHAP_SUMMARY_PNG_ARTIFACT_PATH)

    if not feature_importance_created and not clean_feature_importance.empty:
        create_summary_figure(clean_feature_importance, FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH, "Feature Importance")
        feature_importance_created = FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH.exists()

    if not shap_summary_created and not clean_feature_importance.empty:
        create_summary_figure(
            clean_feature_importance,
            SHAP_SUMMARY_PNG_ARTIFACT_PATH,
            "Explainability Summary (feature importance proxy)",
        )
        shap_summary_created = SHAP_SUMMARY_PNG_ARTIFACT_PATH.exists()

    model_exported = False
    if best_model is not None:
        joblib.dump(best_model, BEST_MODEL_ARTIFACT_PATH, compress=MODEL_COMPRESSION)
        model_exported = True
    elif best_model_path is not None and best_model_path.exists():
        BEST_MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(joblib.load(best_model_path), BEST_MODEL_ARTIFACT_PATH, compress=MODEL_COMPRESSION)
        model_exported = True

    return {
        "best_model_exported": model_exported,
        "feature_importance_png": feature_importance_created,
        "shap_summary_png": shap_summary_created,
    }


def resolve_best_run_paths(run_id: str) -> tuple[Path, Path, Path | None]:
    best_fig_dir = FIG_RUNS_DIR / run_id
    if not best_fig_dir.exists():
        best_fig_dir = FIG_DIR

    top_features_path = best_fig_dir / "top_features.json"
    if not top_features_path.exists():
        top_features_path = FIG_DIR / "top_features.json"

    model_candidates = [
        MODELS_RUNS_DIR / f"model_{run_id}.joblib",
        MODEL_PATH,
    ]
    best_model_path = next((path for path in model_candidates if path.exists()), None)
    return best_fig_dir, top_features_path, best_model_path


def export_public_artifacts_from_reports() -> dict[str, bool]:
    best_run_path = METRICS_DIR / "best_run.json"
    if not best_run_path.exists():
        raise FileNotFoundError("Missing reports/metrics/best_run.json. Run local training first.")
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError("Missing reports/metrics/model_comparison.csv. Run local training first.")

    best_run_payload = json.loads(best_run_path.read_text(encoding="utf-8"))
    best_run_id = str(best_run_payload.get("run_id", "")).strip()
    if not best_run_id:
        raise ValueError("best_run.json does not contain a valid run_id.")

    val_pred_path = METRICS_RUNS_DIR / f"val_predictions_{best_run_id}.csv"
    if not val_pred_path.exists():
        raise FileNotFoundError(f"Missing validation predictions for run {best_run_id}.")

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    val_predictions_df = pd.read_csv(val_pred_path)
    best_fig_dir, top_features_path, best_model_path = resolve_best_run_paths(best_run_id)
    feature_importance_df = load_feature_importance_json(top_features_path)

    return export_public_artifacts(
        best_run_payload=best_run_payload,
        comparison_df=comparison_df,
        val_predictions_df=val_predictions_df,
        feature_importance_df=feature_importance_df,
        best_fig_dir=best_fig_dir,
        best_model_path=best_model_path,
    )


def main():
    results = export_public_artifacts_from_reports()
    print(f"OK: exported public artifacts to {ARTIFACTS_DIR}")
    if not results["best_model_exported"]:
        print("WARNING: best_model.joblib was not exported because no local model artifact was found.")


if __name__ == "__main__":
    main()
