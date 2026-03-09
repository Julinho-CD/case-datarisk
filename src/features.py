# src/features.py
import numpy as np
import pandas as pd

from src.config import TARGET_COL


def cast_to_str(x):
    """
    Usado no Pipeline (FunctionTransformer) para garantir que variáveis categóricas
    sejam tratadas como string antes do OneHotEncoder.
    """
    return x.astype(str)


def _ensure_datetime(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df, ["DATA_CADASTRO", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "SAFRA_REF"])

    df["TEMPO_CADASTRO"] = (df["DATA_EMISSAO_DOCUMENTO"] - df["DATA_CADASTRO"]).dt.days
    df["PRAZO_EMISSAO_VENCIMENTO"] = (df["DATA_VENCIMENTO"] - df["DATA_EMISSAO_DOCUMENTO"]).dt.days

    renda = df["RENDA_MES_ANTERIOR"].replace({0: np.nan})
    df["VALOR_RELATIVO_RENDA"] = df["VALOR_A_PAGAR"] / renda

    df["MES"] = df["SAFRA_REF"].dt.month

    df["TEMPO_CADASTRO"] = df["TEMPO_CADASTRO"].clip(lower=0)
    df["PRAZO_EMISSAO_VENCIMENTO"] = df["PRAZO_EMISSAO_VENCIMENTO"].clip(lower=0)

    return df


def add_ticket_medio_anterior(train: pd.DataFrame, test: pd.DataFrame):
    """
    TICKET_MEDIO_ANT = média histórica de VALOR_A_PAGAR por cliente, usando apenas períodos anteriores.
    (não usa target, então pode concatenar treino+teste com segurança)
    """
    train = train.copy()
    test = test.copy()

    all_df = pd.concat([train.assign(_is_train=1), test.assign(_is_train=0)], ignore_index=True)
    all_df = _ensure_datetime(all_df, ["SAFRA_REF"])
    all_df = all_df.sort_values(["ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO"], na_position="last")

    all_df["TICKET_MEDIO_ANT"] = (
        all_df.groupby("ID_CLIENTE")["VALOR_A_PAGAR"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    train_out = all_df.loc[all_df["_is_train"] == 1].drop(columns=["_is_train"])
    test_out = all_df.loc[all_df["_is_train"] == 0].drop(columns=["_is_train"])
    return train_out, test_out


def add_qtde_atrasos_anterior(train: pd.DataFrame, test: pd.DataFrame):
    """
    QTDE_ATRASOS_ANT (treino): cumulativo de inadimplências anteriores (shift).
    QTDE_ATRASOS_ANT (teste): total histórico de inadimplências no treino por cliente (sem leakage).
    """
    train = train.copy()
    test = test.copy()

    train = _ensure_datetime(train, ["SAFRA_REF"])
    train = train.sort_values(["ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO"], na_position="last")

    # Shift must be applied within each customer to avoid cross-customer leakage.
    train["QTDE_ATRASOS_ANT"] = (
        train.groupby("ID_CLIENTE")[TARGET_COL]
        .transform(lambda s: s.cumsum().shift(1).fillna(0))
    )

    hist_total = (
        train.groupby("ID_CLIENTE")[TARGET_COL]
        .sum()
        .reset_index()
        .rename(columns={TARGET_COL: "QTDE_ATRASOS_ANT"})
    )

    test = test.merge(hist_total, on="ID_CLIENTE", how="left")
    test["QTDE_ATRASOS_ANT"] = test["QTDE_ATRASOS_ANT"].fillna(0)

    return train, test


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    train = add_basic_features(train)
    test = add_basic_features(test)

    train, test = add_ticket_medio_anterior(train, test)
    train, test = add_qtde_atrasos_anterior(train, test)

    for df in (train, test):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_values = df[numeric_cols]
            df[numeric_cols] = numeric_values.mask(np.isinf(numeric_values), np.nan)

    return train, test
