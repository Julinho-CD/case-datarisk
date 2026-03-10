from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_CACHE_DIR = CACHE_DIR / "datarisk_case"
RAW_CACHE_DIR = DATA_CACHE_DIR / "raw"
PROCESSED_CACHE_DIR = DATA_CACHE_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "best_model.joblib"
BEST_RUN_ARTIFACT_PATH = ARTIFACTS_DIR / "best_run.json"
MODEL_COMPARISON_ARTIFACT_PATH = ARTIFACTS_DIR / "model_comparison.csv"
VAL_PREDICTIONS_BEST_ARTIFACT_PATH = ARTIFACTS_DIR / "val_predictions_best.csv"
THRESHOLD_CURVE_ARTIFACT_PATH = ARTIFACTS_DIR / "threshold_curve.csv"
ROC_CURVE_ARTIFACT_PATH = ARTIFACTS_DIR / "roc_curve.csv"
PR_CURVE_ARTIFACT_PATH = ARTIFACTS_DIR / "pr_curve.csv"
FEATURE_IMPORTANCE_CSV_ARTIFACT_PATH = ARTIFACTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG_ARTIFACT_PATH = ARTIFACTS_DIR / "feature_importance.png"
SHAP_SUMMARY_PNG_ARTIFACT_PATH = ARTIFACTS_DIR / "shap_summary.png"

# Subdirectories for multiple runs/models
FIG_RUNS_DIR = FIG_DIR / "runs"
METRICS_RUNS_DIR = METRICS_DIR / "runs"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_RUNS_DIR = MODELS_DIR / "runs"

# CSV separator
SEP = ";"

# Raw files
CADASTRAL_CSV = RAW_DIR / "base_cadastral.csv"
INFO_CSV = RAW_DIR / "base_info.csv"
PAG_DEV_CSV = RAW_DIR / "base_pagamentos_desenvolvimento.csv"
PAG_TEST_CSV = RAW_DIR / "base_pagamentos_teste.csv"

# Processed files
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# Artifacts (best model)
MODEL_PATH = MODELS_DIR / "model.joblib"

# ML
RANDOM_SEED = 42
TARGET_COL = "PROBABILIDADE_INADIMPLENCIA"
DIAS_ATRASO_COL = "DIAS_ATRASO"

# Columns that should never be used as features
DROP_COLS = [
    "ID_CLIENTE",
    "SAFRA_REF",
    "DATA_CADASTRO",
    "DATA_EMISSAO_DOCUMENTO",
    "DATA_VENCIMENTO",
    "DATA_PAGAMENTO",
    DIAS_ATRASO_COL,
    TARGET_COL,
]

# Date columns used in the pipeline
DATE_COLS_TREINO = [
    "DATA_CADASTRO",
    "DATA_EMISSAO_DOCUMENTO",
    "DATA_VENCIMENTO",
    "DATA_PAGAMENTO",
]
DATE_COLS_TESTE = ["DATA_CADASTRO", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO"]
