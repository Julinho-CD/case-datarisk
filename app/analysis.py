import numpy as np
import pandas as pd


def missing_report(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    miss = df.isna().sum()
    out = pd.DataFrame(
        {"column": miss.index, "missing_n": miss.values, "missing_pct": (miss.values / len(df) * 100.0)}
    )
    return out.sort_values("missing_pct", ascending=False).head(top_n).reset_index(drop=True)


def compute_threshold_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
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


def threshold_row(thr_df: pd.DataFrame, thr: float) -> pd.DataFrame:
    idx = (thr_df["threshold"] - float(thr)).abs().idxmin()
    return thr_df.loc[[idx]].copy()


def infer_base_feature(raw_feature: str, candidate_features: list[str]) -> str | None:
    cleaned = str(raw_feature).replace("num__", "").replace("cat__", "")
    if cleaned in candidate_features:
        return cleaned
    for base in sorted(candidate_features, key=len, reverse=True):
        if cleaned.startswith(f"{base}_") or cleaned == base:
            return base
    return None


def select_story_features(top_feats: list[dict], train_fe: pd.DataFrame, top_n: int = 5) -> list[dict]:
    selected = []
    candidates = [c for c in train_fe.columns]
    for i, item in enumerate(top_feats):
        feature = str(item.get("feature", ""))
        base = infer_base_feature(feature, candidates)
        if not base or base in [x["base_feature"] for x in selected]:
            continue
        selected.append(
            {
                "rank": i + 1,
                "raw_feature": feature,
                "base_feature": base,
                "importance": float(item.get("importance", 0.0)),
            }
        )
        if len(selected) >= top_n:
            break
    return selected


def build_numeric_story(df: pd.DataFrame, feature: str) -> tuple[str, pd.DataFrame]:
    tmp = df[[feature, "PROBABILIDADE_INADIMPLENCIA"]].dropna().copy()
    if tmp.empty:
        return f"No data for `{feature}`.", pd.DataFrame()

    try:
        tmp["bucket"] = pd.qcut(tmp[feature], q=10, duplicates="drop")
    except ValueError:
        bins = max(2, min(10, int(tmp[feature].nunique())))
        tmp["bucket"] = pd.cut(tmp[feature], bins=bins)

    agg = (
        tmp.groupby("bucket", observed=False)["PROBABILIDADE_INADIMPLENCIA"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "volume"})
    )
    agg["bucket"] = agg["bucket"].astype(str)
    if agg.empty:
        return f"No signal for `{feature}`.", agg

    first_rate = float(agg["rate"].iloc[0])
    last_rate = float(agg["rate"].iloc[-1])
    direction = "up" if last_rate > first_rate else "down"
    return f"Risk tends to move {direction} across `{feature}` buckets.", agg


def build_categorical_story(df: pd.DataFrame, feature: str) -> tuple[str, pd.DataFrame]:
    tmp = df[[feature, "PROBABILIDADE_INADIMPLENCIA"]].dropna().copy()
    if tmp.empty:
        return f"No data for `{feature}`.", pd.DataFrame()

    tmp[feature] = tmp[feature].astype(str)
    agg = (
        tmp.groupby(feature, observed=False)["PROBABILIDADE_INADIMPLENCIA"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "count": "volume"})
        .sort_values("rate", ascending=False)
        .head(15)
    )
    return f"`{feature}` segments customer risk meaningfully.", agg
