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
