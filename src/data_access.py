import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

from src.config import (
    CADASTRAL_CSV,
    DATA_CACHE_DIR,
    DIAS_ATRASO_COL,
    INFO_CSV,
    PAG_DEV_CSV,
    PAG_TEST_CSV,
    PROCESSED_CACHE_DIR,
    PROCESSED_DIR,
    RAW_CACHE_DIR,
    SEP,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
)

DATA_SOURCE_AUTO = "auto"
DATA_SOURCE_LOCAL = "local"
DATA_SOURCE_REMOTE = "remote"
DEFAULT_TIMEOUT_SECONDS = 30

OFFICIAL_REPOSITORY_URL = "https://github.com/datarisk-io/datarisk-case-ds-junior"
OFFICIAL_RAW_URL_TEMPLATES = (
    "https://raw.githubusercontent.com/datarisk-io/datarisk-case-ds-junior/main/data/{filename}",
    "https://raw.githubusercontent.com/datarisk-io/datarisk-case-ds-junior/master/data/{filename}",
)

RAW_DATA_FILES: Dict[str, Path] = {
    "base_cadastral": CADASTRAL_CSV,
    "base_info": INFO_CSV,
    "base_pagamentos_desenvolvimento": PAG_DEV_CSV,
    "base_pagamentos_teste": PAG_TEST_CSV,
}

RAW_DATA_FILENAMES: Dict[str, str] = {
    key: path.name for key, path in RAW_DATA_FILES.items()
}

DATE_COLUMNS = [
    "SAFRA_REF",
    "DATA_CADASTRO",
    "DATA_EMISSAO_DOCUMENTO",
    "DATA_VENCIMENTO",
    "DATA_PAGAMENTO",
]


def bool_env(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_data_source(source: str | None = None) -> str:
    resolved = (source or os.getenv("DATARISK_DATA_SOURCE", DATA_SOURCE_AUTO)).strip().lower()
    if resolved not in {DATA_SOURCE_AUTO, DATA_SOURCE_LOCAL, DATA_SOURCE_REMOTE}:
        raise ValueError(
            f"Unsupported data source '{resolved}'. Use one of: "
            f"{DATA_SOURCE_AUTO}, {DATA_SOURCE_LOCAL}, {DATA_SOURCE_REMOTE}."
        )
    return resolved


def default_refresh_flag(refresh: bool | None = None) -> bool:
    if refresh is not None:
        return refresh
    return bool_env("DATARISK_DATA_REFRESH", default=False)


def ensure_cache_dirs():
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def official_file_urls(file_name: str) -> list[str]:
    return [template.format(filename=file_name) for template in OFFICIAL_RAW_URL_TEMPLATES]


def _download_file(urls: list[str], dest_path: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    ensure_cache_dirs()
    errors: list[str] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(response.content)
                return dest_path
            errors.append(f"{url} -> HTTP {response.status_code}")
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")

    raise FileNotFoundError(
        "Unable to download official Datarisk data file. "
        f"Tried: {', '.join(errors)}"
    )


def download_official_file(logical_name: str, refresh: bool | None = None) -> Path:
    if logical_name not in RAW_DATA_FILENAMES:
        raise KeyError(f"Unknown dataset '{logical_name}'. Expected one of: {sorted(RAW_DATA_FILENAMES)}")

    refresh = default_refresh_flag(refresh)
    filename = RAW_DATA_FILENAMES[logical_name]
    cached_path = RAW_CACHE_DIR / filename

    if cached_path.exists() and not refresh:
        return cached_path

    return _download_file(official_file_urls(filename), cached_path)


def resolve_raw_path(logical_name: str, source: str | None = None, refresh: bool | None = None) -> Path:
    if logical_name not in RAW_DATA_FILES:
        raise KeyError(f"Unknown dataset '{logical_name}'. Expected one of: {sorted(RAW_DATA_FILES)}")

    source = resolve_data_source(source)
    refresh = default_refresh_flag(refresh)
    local_path = RAW_DATA_FILES[logical_name]

    if source in {DATA_SOURCE_AUTO, DATA_SOURCE_LOCAL} and local_path.exists():
        return local_path

    if source == DATA_SOURCE_LOCAL:
        raise FileNotFoundError(
            f"Local dataset not found: {local_path}. "
            "Provide the official CSV locally under data/raw/ or switch DATARISK_DATA_SOURCE to 'remote' or 'auto'."
        )

    return download_official_file(logical_name, refresh=refresh)


def read_case_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=SEP, low_memory=False)


def load_raw_dataframe(logical_name: str, source: str | None = None, refresh: bool | None = None) -> pd.DataFrame:
    path = resolve_raw_path(logical_name, source=source, refresh=refresh)
    try:
        return read_case_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset '{logical_name}' from {path}: {exc}") from exc


def load_raw_case_data(source: str | None = None, refresh: bool | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_raw_dataframe("base_cadastral", source=source, refresh=refresh),
        load_raw_dataframe("base_info", source=source, refresh=refresh),
        load_raw_dataframe("base_pagamentos_desenvolvimento", source=source, refresh=refresh),
        load_raw_dataframe("base_pagamentos_teste", source=source, refresh=refresh),
    )


def clean_types(base_cadastral, base_info, base_pag_dev, base_pag_test):
    for df in (base_info, base_pag_dev, base_pag_test):
        if "SAFRA_REF" in df.columns:
            df["SAFRA_REF"] = pd.to_datetime(df["SAFRA_REF"], format="%Y-%m", errors="coerce")

    if "DATA_CADASTRO" in base_cadastral.columns:
        base_cadastral["DATA_CADASTRO"] = pd.to_datetime(base_cadastral["DATA_CADASTRO"], errors="coerce")

    for col in ["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "DATA_PAGAMENTO"]:
        if col in base_pag_dev.columns:
            base_pag_dev[col] = pd.to_datetime(base_pag_dev[col], errors="coerce")
        if col in base_pag_test.columns:
            base_pag_test[col] = pd.to_datetime(base_pag_test[col], errors="coerce")

    if "DDD" in base_cadastral.columns:
        base_cadastral["DDD"] = base_cadastral["DDD"].astype(str).str.zfill(2)
    if "CEP_2_DIG" in base_cadastral.columns:
        base_cadastral["CEP_2_DIG"] = base_cadastral["CEP_2_DIG"].astype(str).str.zfill(2)
    if "FLAG_PF" in base_cadastral.columns:
        base_cadastral["FLAG_PF"] = base_cadastral["FLAG_PF"].apply(
            lambda x: "PF" if pd.notna(x) and str(x).strip() != "" else "PJ"
        )

    return base_cadastral, base_info, base_pag_dev, base_pag_test


def deduplicate(base_pag_dev, base_pag_test):
    return base_pag_dev.drop_duplicates(), base_pag_test.drop_duplicates()


def create_target(base_pag_dev: pd.DataFrame) -> pd.DataFrame:
    base_pag_dev = base_pag_dev.copy()
    base_pag_dev[DIAS_ATRASO_COL] = (base_pag_dev["DATA_PAGAMENTO"] - base_pag_dev["DATA_VENCIMENTO"]).dt.days
    base_pag_dev[TARGET_COL] = base_pag_dev[DIAS_ATRASO_COL].apply(lambda x: 1 if x >= 5 else 0)
    return base_pag_dev


def validate_case_alignment(
    base_pag_dev: pd.DataFrame,
    base_pag_test: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    if len(train) != len(base_pag_dev):
        raise ValueError(
            "Training merge changed the number of rows from base_pagamentos_desenvolvimento.csv. "
            "Check join keys and duplicate records before training."
        )

    if len(test) != len(base_pag_test):
        raise ValueError(
            "Test merge changed the number of rows from base_pagamentos_teste.csv. "
            "The case expects one prediction for each original payment record."
        )

    recalculated_target = (base_pag_dev[DIAS_ATRASO_COL] >= 5).astype(int)
    if not recalculated_target.equals(base_pag_dev[TARGET_COL].astype(int)):
        raise ValueError("Target definition mismatch. Delinquency must be 1 when payment delay is >= 5 days.")


def build_train_test(base_cadastral, base_info, base_pag_dev, base_pag_test):
    train = base_pag_dev.merge(base_cadastral, on="ID_CLIENTE", how="left")
    train = train.merge(base_info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")

    test = base_pag_test.merge(base_cadastral, on="ID_CLIENTE", how="left")
    test = test.merge(base_info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
    return train, test


def build_processed_datasets(source: str | None = None, refresh: bool | None = None):
    base_cadastral, base_info, base_pag_dev, base_pag_test = load_raw_case_data(source=source, refresh=refresh)
    base_cadastral, base_info, base_pag_dev, base_pag_test = clean_types(
        base_cadastral, base_info, base_pag_dev, base_pag_test
    )
    base_pag_dev, base_pag_test = deduplicate(base_pag_dev, base_pag_test)
    base_pag_dev = create_target(base_pag_dev)
    train, test = build_train_test(base_cadastral, base_info, base_pag_dev, base_pag_test)
    validate_case_alignment(base_pag_dev, base_pag_test, train, test)
    return train, test


def _cache_key(source: str) -> str:
    return source


def cache_processed_datasets(train: pd.DataFrame, test: pd.DataFrame, source: str):
    ensure_cache_dirs()
    cache_dir = PROCESSED_CACHE_DIR / _cache_key(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(cache_dir / "train.csv", sep=SEP, index=False)
    test.to_csv(cache_dir / "test.csv", sep=SEP, index=False)


def save_processed_datasets(train: pd.DataFrame, test: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, sep=SEP, index=False)
    test.to_csv(TEST_PATH, sep=SEP, index=False)


def _parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_cached_processed_datasets(source: str):
    cache_dir = PROCESSED_CACHE_DIR / _cache_key(source)
    train_path = cache_dir / "train.csv"
    test_path = cache_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return None, None
    return _parse_datetime_columns(read_case_csv(train_path)), _parse_datetime_columns(read_case_csv(test_path))


def load_processed_datasets(
    source: str | None = None,
    refresh: bool | None = None,
    prefer_local_processed: bool = True,
):
    source = resolve_data_source(source)
    refresh = default_refresh_flag(refresh)

    if prefer_local_processed and source in {DATA_SOURCE_AUTO, DATA_SOURCE_LOCAL}:
        if TRAIN_PATH.exists() and TEST_PATH.exists() and not refresh:
            return _parse_datetime_columns(read_case_csv(TRAIN_PATH)), _parse_datetime_columns(read_case_csv(TEST_PATH))

    if not refresh:
        cached_train, cached_test = load_cached_processed_datasets(source)
        if cached_train is not None and cached_test is not None:
            return cached_train, cached_test

    train, test = build_processed_datasets(source=source, refresh=refresh)
    cache_processed_datasets(train, test, source)
    return _parse_datetime_columns(train), _parse_datetime_columns(test)


def check_official_data_availability(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, str]:
    status: dict[str, str] = {}
    for logical_name, filename in RAW_DATA_FILENAMES.items():
        urls = official_file_urls(filename)
        last_error = "unreachable"
        for url in urls:
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    status[logical_name] = url
                    last_error = ""
                    break
                last_error = f"{url} -> HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = f"{url} -> {exc}"
        if logical_name not in status:
            raise RuntimeError(
                f"Official Datarisk dataset '{filename}' is not accessible. Last error: {last_error}"
            )
    return status
