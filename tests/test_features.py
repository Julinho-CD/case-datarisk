import pandas as pd

from src.config import TARGET_COL
from src.features import add_basic_features, add_qtde_atrasos_anterior


def test_add_basic_features_builds_columns_and_clips_negative_values():
    df = pd.DataFrame(
        {
            "DATA_CADASTRO": pd.to_datetime(["2024-01-10"]),
            "DATA_EMISSAO_DOCUMENTO": pd.to_datetime(["2024-01-05"]),
            "DATA_VENCIMENTO": pd.to_datetime(["2024-01-03"]),
            "SAFRA_REF": pd.to_datetime(["2024-02-01"]),
            "RENDA_MES_ANTERIOR": [0.0],
            "VALOR_A_PAGAR": [100.0],
        }
    )

    out = add_basic_features(df)

    assert "TEMPO_CADASTRO" in out.columns
    assert "PRAZO_EMISSAO_VENCIMENTO" in out.columns
    assert "VALOR_RELATIVO_RENDA" in out.columns
    assert "MES" in out.columns
    assert out.loc[0, "TEMPO_CADASTRO"] == 0
    assert out.loc[0, "PRAZO_EMISSAO_VENCIMENTO"] == 0
    assert pd.isna(out.loc[0, "VALOR_RELATIVO_RENDA"])
    assert int(out.loc[0, "MES"]) == 2


def test_add_qtde_atrasos_anterior_uses_train_history_only_for_test():
    train = pd.DataFrame(
        {
            "ID_CLIENTE": [1, 1, 2],
            "SAFRA_REF": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
            "DATA_EMISSAO_DOCUMENTO": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-01-03"]),
            TARGET_COL: [0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "ID_CLIENTE": [1, 2, 3],
            "SAFRA_REF": pd.to_datetime(["2024-03-01", "2024-02-01", "2024-01-01"]),
            "DATA_EMISSAO_DOCUMENTO": pd.to_datetime(["2024-03-02", "2024-02-02", "2024-01-05"]),
        }
    )

    train_out, test_out = add_qtde_atrasos_anterior(train, test)

    # In train, value is shifted cumulative sum.
    assert train_out["QTDE_ATRASOS_ANT"].tolist() == [0.0, 0.0, 0.0]
    # In test, each client receives total delinquency count observed in train.
    assert test_out["QTDE_ATRASOS_ANT"].tolist() == [1.0, 1.0, 0.0]
