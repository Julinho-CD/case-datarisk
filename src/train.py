"""
Train and compare models (with local MLflow tracking).

This script:
- loads processed datasets from the central data access layer
- builds consistent features for train/test
- performs temporal validation (last 2 monthly cohorts)
- trains multiple models (LogReg, RandomForest, LightGBM) and applies SMOTE only to LogReg/RF when it makes sense
- logs everything to local MLflow
- exports artifacts to reports/ for Streamlit

Run:
  python -m src.train
  mlflow ui
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    ARTIFACTS_DIR,
    FIG_DIR,
    FIG_RUNS_DIR,
    METRICS_DIR,
    METRICS_RUNS_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_DIR,
    MODELS_RUNS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COL,
    DROP_COLS,
)
from src.data_access import load_processed_datasets
from src.features import build_features, cast_to_str
from src.public_artifacts import export_public_artifacts


# ---------------------------
# Training config
# ---------------------------
N_VAL_SAFRAS = 2
RUN_CV = True          # 3-fold CV on train (non-temporal) for stability; disable if needed for speed
CV_SPLITS = 3
BEST_SELECTION_METRIC = "pr_auc"  # "pr_auc" (recommended for imbalance) or "roc_auc"


# ---------------------------
# Types
# ---------------------------
@dataclass
class RunMetrics:
    roc_auc: float
    pr_auc: float
    f1_at_05: float
    best_threshold: float
    f1_best_threshold: float
    precision_best_threshold: float
    recall_best_threshold: float
    positive_rate_best_threshold: float
    cv_roc_auc_mean: Optional[float] = None
    cv_pr_auc_mean: Optional[float] = None


def to_project_relative(path: Path) -> str:
    """
    Store paths relative to the repository root to keep artifacts portable
    across machines and operating systems.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def from_project_path(path_like: str | Path) -> Path:
    """
    Resolve both relative (preferred) and absolute paths for backward
    compatibility with older metadata files.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ---------------------------
# IO
# ---------------------------
def load_processed_csv():
    return load_processed_datasets()


# ---------------------------
# Split / Preprocess
# ---------------------------
def temporal_split(df: pd.DataFrame, n_val_safras: int = 2):
    safras = sorted(df["SAFRA_REF"].dropna().unique())
    if len(safras) <= n_val_safras:
        raise ValueError(
            f"Too few monthly cohorts ({len(safras)}) for temporal split with n_val_safras={n_val_safras}."
        )
    val_safras = safras[-n_val_safras:]
    is_val = df["SAFRA_REF"].isin(val_safras)
    return df.loc[~is_val].copy(), df.loc[is_val].copy(), val_safras


def select_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    return [c for c in train_df.columns if c not in DROP_COLS and c in test_df.columns]


def infer_column_groups(features: List[str]) -> Tuple[List[str], List[str]]:
    cat_candidates = ["DDD", "FLAG_PF", "SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL", "PORTE", "CEP_2_DIG"]
    cat_cols = [c for c in cat_candidates if c in features]
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(cast_to_str)),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def build_smote_input_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessor for the SMOTE path.
    Keeps categorical variables as ordinal-encoded columns so SMOTENC can run
    before One-Hot Encoding.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_str", FunctionTransformer(cast_to_str)),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    return ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def build_post_smote_preprocessor(n_num: int, n_cat: int) -> ColumnTransformer:
    """
    Applies final modeling transformations after SMOTENC.
    Numeric columns are scaled; categorical columns are one-hot encoded.
    """
    num_idx = list(range(n_num))
    cat_idx = list(range(n_num, n_num + n_cat))

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    num_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(transformers=[("num", num_pipe, num_idx), ("cat", cat_pipe, cat_idx)])


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue

        last = transformer
        if isinstance(transformer, Pipeline):
            last = transformer.steps[-1][1]

        if hasattr(last, "get_feature_names_out"):
            input_features = [str(c) for c in cols]
            fn = last.get_feature_names_out(input_features)
            names.extend(fn.tolist())
        else:
            names.extend([str(c) for c in cols])

    return names


# ---------------------------
# Metrics
# ---------------------------
def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    positive_rate = float((y_pred == 1).mean())

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(positive_rate),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, y_pred)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return float(best_t), float(best_f)


def cv_scores(
    pipe: Any,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits: int = 3,
) -> Tuple[float, float]:
    """
    Simple CV (non-temporal) to stabilize benchmark estimates.
    Note: the final holdout is temporal; this CV is only complementary.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    roc_scores = []
    pr_scores = []
    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        roc_scores.append(roc_auc_score(yte, prob))
        pr_scores.append(average_precision_score(yte, prob))

    return float(np.mean(roc_scores)), float(np.mean(pr_scores))


# ---------------------------
# Plots
# ---------------------------
def save_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.savefig(out_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(out_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (thr={threshold:.2f})")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig(out_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()


def save_feature_importance(
    model_pipe: Any,
    feature_names: List[str],
    out_dir: Path,
    title: str,
    top_n: int = 20,
) -> List[Dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = model_pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_)).ravel().astype(float)
    else:
        return []

    idx = np.argsort(imp)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values = imp[idx]

    plt.figure(figsize=(8, 5))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(title)
    plt.xlabel("importance")
    plt.savefig(out_dir / "feature_importance.png", bbox_inches="tight")
    plt.close()

    top_list = [{"feature": top_features[i], "importance": float(top_values[i])} for i in range(len(top_features))]
    (out_dir / "top_features.json").write_text(json.dumps(top_list, indent=2), encoding="utf-8")
    return top_list


# ---------------------------
# Local MLflow
# ---------------------------
def setup_mlflow_local():
    """
    MLflow setup.
    - If MLFLOW_TRACKING_URI is defined, use remote/local URI from environment.
    - Otherwise, default to local file store under ./mlruns.
    """
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlruns_dir = PROJECT_ROOT / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "datarisk-inadimplencia")
    mlflow.set_experiment(experiment_name)


def bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def should_export_local_models() -> bool:
    """
    Keep local .joblib exports optional to avoid duplicating heavy artifacts
    already stored in MLflow.
    """
    return bool_env("EXPORT_LOCAL_MODELS", default=False)


# ---------------------------
# Builders
# ---------------------------
def should_enable_smote(y: pd.Series) -> bool:
    """
    Apply SMOTE only when there is meaningful class imbalance
    and enough minority-class examples.
    """
    vc = y.value_counts(dropna=False)
    if len(vc) < 2:
        return False

    minority = int(vc.min())
    majority = int(vc.max())
    total = minority + majority
    minority_rate = minority / total if total else 0.0
    imbalance_ratio = majority / minority if minority else np.inf

    enough_minority_examples = minority >= 200
    strong_imbalance = minority_rate <= 0.35 and imbalance_ratio >= 1.5
    return bool(enough_minority_examples and strong_imbalance)


def build_model_grid(seed: int, enable_smote_variants: bool) -> List[Dict[str, Any]]:
    """Define which models to run and whether to use SMOTE."""
    grid: List[Dict[str, Any]] = []

    # Logistic Regression
    base_lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    grid.append({"model_key": "logreg", "use_smote": False, "estimator": base_lr})
    if enable_smote_variants:
        grid.append({"model_key": "logreg", "use_smote": True, "estimator": base_lr})

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    grid.append({"model_key": "rf", "use_smote": False, "estimator": rf})
    if enable_smote_variants:
        grid.append({"model_key": "rf", "use_smote": True, "estimator": rf})

    # LightGBM
    lgbm = LGBMClassifier(
        class_weight="balanced",
        random_state=seed,
        n_estimators=700,
        learning_rate=0.05,
        max_depth=8,
        n_jobs=-1,
    )
    # For LightGBM, this pipeline does not use SMOTE.
    grid.append({"model_key": "lgbm", "use_smote": False, "estimator": lgbm})

    return grid


def build_pipeline(num_cols: List[str], cat_cols: List[str], estimator: Any, use_smote: bool, seed: int):
    if use_smote:
        pre_smote = build_smote_input_preprocessor(num_cols, cat_cols)
        n_num, n_cat = len(num_cols), len(cat_cols)
        cat_idx = list(range(n_num, n_num + n_cat))
        smote_step = SMOTENC(categorical_features=cat_idx, random_state=seed)
        pre_model = build_post_smote_preprocessor(n_num=n_num, n_cat=n_cat)
        return ImbPipeline(steps=[("pre_smote", pre_smote), ("smote", smote_step), ("pre_model", pre_model), ("clf", estimator)])

    preproc = build_preprocessor(num_cols, cat_cols)
    return ImbPipeline(steps=[("pre", preproc), ("smote", "passthrough"), ("clf", estimator)])


def model_title(model_key: str) -> str:
    return {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "lgbm": "LightGBM",
    }.get(model_key, model_key)


# ---------------------------
# Main
# ---------------------------
def main():
    export_local_models = should_export_local_models()
    if export_local_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow_local()

    train_raw, test_raw = load_processed_csv()
    train_fe, test_fe = build_features(train_raw, test_raw)

    if TARGET_COL not in train_fe.columns:
        raise ValueError(f"Target {TARGET_COL} not found in train. Run `python -m src.make_dataset`.")

    features = select_features(train_fe, test_fe)
    num_cols, cat_cols = infer_column_groups(features)
    # Temporal holdout
    train_df, val_df, val_safras = temporal_split(train_fe, n_val_safras=N_VAL_SAFRAS)
    X_tr, y_tr = train_df[features], train_df[TARGET_COL].astype(int)
    X_val, y_val = val_df[features], val_df[TARGET_COL].astype(int)

    enable_smote_variants = should_enable_smote(y_tr)
    grid = build_model_grid(RANDOM_SEED, enable_smote_variants=enable_smote_variants)
    comparison_rows: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Any] = {}
    val_prediction_frames: Dict[str, pd.DataFrame] = {}
    feature_importance_tables: Dict[str, List[Dict[str, float]]] = {}
    print(f"SMOTE variants enabled for LogReg/RF: {int(enable_smote_variants)}")

    for cfg in grid:
        mkey = cfg["model_key"]
        use_smote = bool(cfg["use_smote"])
        est = cfg["estimator"]
        run_name = f"{mkey}|smote={int(use_smote)}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            pipe = build_pipeline(num_cols, cat_cols, est, use_smote, RANDOM_SEED)

            # Fit + temporal validation
            pipe.fit(X_tr, y_tr)
            val_prob = pipe.predict_proba(X_val)[:, 1]

            roc_auc = float(roc_auc_score(y_val, val_prob))
            pr_auc = float(average_precision_score(y_val, val_prob))
            f1_05 = float(f1_score(y_val, (val_prob >= 0.5).astype(int)))
            best_thr, f1_best = best_threshold_by_f1(y_val.values, val_prob)
            best_stats = metrics_at_threshold(y_val.values, val_prob, best_thr)

            cv_roc, cv_pr = None, None
            if RUN_CV:
                cv_pipe = build_pipeline(num_cols, cat_cols, est, use_smote, RANDOM_SEED)
                cv_roc, cv_pr = cv_scores(cv_pipe, X_tr, y_tr, seed=RANDOM_SEED, n_splits=CV_SPLITS)

            run_metrics = RunMetrics(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                f1_at_05=f1_05,
                best_threshold=float(best_thr),
                f1_best_threshold=float(f1_best),
                precision_best_threshold=float(best_stats["precision"]),
                recall_best_threshold=float(best_stats["recall"]),
                positive_rate_best_threshold=float(best_stats["positive_rate"]),
                cv_roc_auc_mean=cv_roc,
                cv_pr_auc_mean=cv_pr,
            )

            # Directories per run
            run_fig_dir = FIG_RUNS_DIR / run_id
            run_fig_dir.mkdir(parents=True, exist_ok=True)

            # Predictions per run (for the Streamlit threshold slider)
            run_val_pred_path = METRICS_RUNS_DIR / f"val_predictions_{run_id}.csv"
            run_val_predictions_df = pd.DataFrame(
                {
                    "y_true": y_val.values.astype(int),
                    "y_prob": val_prob.astype(float),
                    "SAFRA_REF": val_df["SAFRA_REF"].astype(str).values,
                }
            )
            run_val_predictions_df.to_csv(run_val_pred_path, index=False)

            # Artifacts per run
            save_curves(y_val.values, val_prob, run_fig_dir)
            save_confusion_matrix(y_val.values, val_prob, best_thr, run_fig_dir)

            pre_name = "pre_model" if use_smote else "pre"
            feature_names = get_feature_names(pipe.named_steps[pre_name])
            title = f"Feature Importance ({model_title(mkey)})"
            top_features = save_feature_importance(pipe, feature_names, run_fig_dir, title=title, top_n=20)

            # Optional local export (disabled by default to keep repo lightweight).
            run_model_path: Optional[Path] = None
            if export_local_models:
                run_model_path = MODELS_RUNS_DIR / f"model_{run_id}.joblib"
                joblib.dump(pipe, run_model_path)

            # MLflow params/metrics
            mlflow.log_params(
                {
                    "model_key": mkey,
                    "model_name": model_title(mkey),
                    "use_smote": int(use_smote),
                    "smote_variants_enabled": int(enable_smote_variants),
                    "seed": int(RANDOM_SEED),
                    "val_safras": ",".join([str(x) for x in val_safras]),
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)),
                    "n_features": int(len(features)),
                }
            )

            if hasattr(est, "get_params"):
                p = est.get_params()
                keys = ["C", "n_estimators", "max_depth", "learning_rate", "num_leaves", "min_samples_leaf"]
                mlflow.log_params({k: p.get(k) for k in keys if k in p and p.get(k) is not None})

            mlflow.log_metrics(
                {
                    "roc_auc": run_metrics.roc_auc,
                    "pr_auc": run_metrics.pr_auc,
                    "f1_at_05": run_metrics.f1_at_05,
                    "best_threshold": run_metrics.best_threshold,
                    "f1_best_threshold": run_metrics.f1_best_threshold,
                    "precision_best_threshold": run_metrics.precision_best_threshold,
                    "recall_best_threshold": run_metrics.recall_best_threshold,
                    "positive_rate_best_threshold": run_metrics.positive_rate_best_threshold,
                }
            )

            if run_metrics.cv_roc_auc_mean is not None:
                mlflow.log_metric("cv_roc_auc_mean", float(run_metrics.cv_roc_auc_mean))
            if run_metrics.cv_pr_auc_mean is not None:
                mlflow.log_metric("cv_pr_auc_mean", float(run_metrics.cv_pr_auc_mean))

            mlflow.log_artifacts(str(run_fig_dir), artifact_path="figures")
            mlflow.log_artifact(str(run_val_pred_path), artifact_path="metrics")
            if run_model_path is not None:
                mlflow.log_artifact(str(run_model_path), artifact_path="exported_model")

            try:
                mlflow.sklearn.log_model(pipe, name="model")
            except TypeError:
                mlflow.sklearn.log_model(pipe, artifact_path="model")

            row = {
                "run_id": run_id,
                "model_key": mkey,
                "model_name": model_title(mkey),
                "use_smote": int(use_smote),
                "roc_auc": run_metrics.roc_auc,
                "pr_auc": run_metrics.pr_auc,
                "f1_at_05": run_metrics.f1_at_05,
                "best_threshold": run_metrics.best_threshold,
                "f1_best_threshold": run_metrics.f1_best_threshold,
                "precision_best_threshold": run_metrics.precision_best_threshold,
                "recall_best_threshold": run_metrics.recall_best_threshold,
                "positive_rate_best_threshold": run_metrics.positive_rate_best_threshold,
                "cv_roc_auc_mean": run_metrics.cv_roc_auc_mean,
                "cv_pr_auc_mean": run_metrics.cv_pr_auc_mean,
                "run_fig_dir": to_project_relative(run_fig_dir),
                "run_val_pred_path": to_project_relative(run_val_pred_path),
                "run_model_path": (to_project_relative(run_model_path) if run_model_path is not None else ""),
            }
            comparison_rows.append(row)
            trained_pipelines[run_id] = pipe
            val_prediction_frames[run_id] = run_val_predictions_df
            feature_importance_tables[run_id] = top_features

            print(
                f"{run_name} | ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
                f"F1@0.5={f1_05:.4f} | best_thr={best_thr:.2f} | run_id={run_id}"
            )

    # Save benchmark comparison
    comp = pd.DataFrame(comparison_rows)
    if BEST_SELECTION_METRIC == "roc_auc":
        comp = comp.sort_values(["roc_auc", "pr_auc"], ascending=False)
    else:
        comp = comp.sort_values(["pr_auc", "roc_auc"], ascending=False)

    comp.to_csv(MODEL_COMPARISON_PATH, index=False)

    # Select best run
    best = comp.iloc[0].to_dict()
    best_run_id = best["run_id"]

    # Optional local fallback model
    best_model_ref = str(best.get("run_model_path", "")).strip()
    if export_local_models and best_model_ref:
        best_model_path = from_project_path(best_model_ref)
        joblib.dump(joblib.load(best_model_path), MODEL_PATH)

    # Copy best artifacts to reports/figures (shortcut)
    best_fig_dir = from_project_path(str(best["run_fig_dir"]))
    for fname in ["roc_curve.png", "pr_curve.png", "confusion_matrix.png", "feature_importance.png", "top_features.json"]:
        src = best_fig_dir / fname
        if src.exists():
            shutil.copyfile(src, FIG_DIR / fname)

    best_metrics_payload = {
        "roc_auc": float(best["roc_auc"]),
        "pr_auc": float(best["pr_auc"]),
        "f1_at_05": float(best["f1_at_05"]),
        "best_threshold": float(best["best_threshold"]),
        "f1_best_threshold": float(best["f1_best_threshold"]),
        "precision_best_threshold": float(best["precision_best_threshold"]),
        "recall_best_threshold": float(best["recall_best_threshold"]),
        "positive_rate_best_threshold": float(best["positive_rate_best_threshold"]),
        "cv_roc_auc_mean": None if pd.isna(best.get("cv_roc_auc_mean")) else float(best.get("cv_roc_auc_mean")),
        "cv_pr_auc_mean": None if pd.isna(best.get("cv_pr_auc_mean")) else float(best.get("cv_pr_auc_mean")),
    }

    (METRICS_DIR / "validation_metrics.json").write_text(
        json.dumps(best_metrics_payload, indent=2), encoding="utf-8"
    )

    model_info = {
        "best_model": best["model_name"],
        "best_model_key": best["model_key"],
        "best_run_id": best_run_id,
        "best_use_smote": int(best["use_smote"]),
        "local_model_exported": int(export_local_models),
        "smote_variants_enabled": int(enable_smote_variants),
        "smote_policy": "SMOTE only for Logistic Regression and Random Forest when there is relevant imbalance.",
        "selection_metric": BEST_SELECTION_METRIC,
        "val_safras": [str(x) for x in val_safras],
        "n_models_tested": int(len(comp)),
    }
    (METRICS_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    best_run_payload = {
        "run_id": best_run_id,
        "run_name": f"{best['model_key']}|smote={int(best['use_smote'])}",
        "model_name": best["model_name"],
        "variant": "smote" if int(best["use_smote"]) == 1 else "nosmote",
        "use_smote": bool(best["use_smote"]),
        "roc_auc": float(best["roc_auc"]),
        "pr_auc": float(best["pr_auc"]),
        "f1_at_05": float(best["f1_at_05"]),
        "best_threshold": float(best["best_threshold"]),
        "f1_best_threshold": float(best["f1_best_threshold"]),
        "precision_best_threshold": float(best["precision_best_threshold"]),
        "recall_best_threshold": float(best["recall_best_threshold"]),
        "positive_rate_best_threshold": float(best["positive_rate_best_threshold"]),
        "cv_roc_auc_mean": None if pd.isna(best.get("cv_roc_auc_mean")) else float(best.get("cv_roc_auc_mean")),
        "cv_pr_auc_mean": None if pd.isna(best.get("cv_pr_auc_mean")) else float(best.get("cv_pr_auc_mean")),
        "selection_metric": BEST_SELECTION_METRIC,
        "is_best": True,
        "val_safras": [str(x) for x in val_safras],
    }
    (METRICS_DIR / "best_run.json").write_text(json.dumps(best_run_payload, indent=2), encoding="utf-8")

    export_results = export_public_artifacts(
        best_run_payload=best_run_payload,
        comparison_df=comp,
        val_predictions_df=val_prediction_frames[best_run_id],
        feature_importance_df=feature_importance_tables.get(best_run_id, []),
        best_fig_dir=best_fig_dir,
        best_model=trained_pipelines.get(best_run_id),
    )

    # Mark best run in MLflow (tag)
    try:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("best_run", "true")
            mlflow.set_tag("selection_metric", BEST_SELECTION_METRIC)
    except Exception:
        pass

    print("\n=== FINAL RESULT ===")
    print(f"Best (by {BEST_SELECTION_METRIC}): {best['model_name']} | smote={best['use_smote']} | run_id={best_run_id}")
    print(f"Comparison saved to: {MODEL_COMPARISON_PATH}")
    print(f"Public artifacts saved to: {ARTIFACTS_DIR}")
    if export_local_models:
        print(f"Best model saved to: {MODEL_PATH}")
    else:
        print("Local model export disabled (EXPORT_LOCAL_MODELS=0). Public artifacts remain available under artifacts/.")
    if not export_results["best_model_exported"]:
        print("WARNING: public best_model.joblib was not exported.")
    print("To inspect runs, execute `mlflow ui` at the project root")


if __name__ == "__main__":
    main()

