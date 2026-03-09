import pandas as pd

from src.config import DIAS_ATRASO_COL, TARGET_COL
from src.make_dataset import create_target


def test_create_target_marks_delinquency_from_delay_days():
    df = pd.DataFrame(
        {
            "DATA_VENCIMENTO": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-10"]),
            "DATA_PAGAMENTO": pd.to_datetime(["2024-01-10", "2024-01-14", "2024-01-20"]),
        }
    )

    out = create_target(df)

    assert out[DIAS_ATRASO_COL].tolist() == [0, 4, 10]
    assert out[TARGET_COL].tolist() == [0, 0, 1]
