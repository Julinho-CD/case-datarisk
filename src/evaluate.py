import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIG_DIR, FIG_RUNS_DIR, METRICS_DIR, MODEL_COMPARISON_PATH, PROJECT_ROOT, TARGET_COL
from src.data_access import load_processed_datasets
from src.features import build_features

STORY_FIG_DIR = FIG_DIR / "story"
TOP_K_STORY = 5


def to_project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_processed():
    return load_processed_datasets()


def map_ohe_feature_to_base(feature_name: str, base_candidates: List[str]) -> Optional[str]:
    cleaned = str(feature_name).replace("num__", "").replace("cat__", "")
    if cleaned in base_candidates:
        return cleaned
    for base in sorted(base_candidates, key=len, reverse=True):
        if cleaned.startswith(f"{base}_") or cleaned == base:
            return base
    return None


def pick_top_base_features(top_features: List[Dict[str, float]], base_candidates: List[str], k: int) -> List[str]:
    out: List[str] = []
    for item in top_features:
        feature = item.get("feature")
        if not feature:
            continue
        base = map_ohe_feature_to_base(str(feature), base_candidates)
        if base and base not in out:
            out.append(base)
        if len(out) >= k:
            break
    return out


def rate_by_quantiles(df: pd.DataFrame, feature: str, bins: int = 10) -> pd.Series:
    tmp = df[[feature, TARGET_COL]].dropna().copy()
    if tmp.empty:
        return pd.Series(dtype=float)
    try:
        tmp["bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    except ValueError:
        tmp["bin"] = pd.cut(tmp[feature], bins=min(bins, max(2, tmp[feature].nunique())))
    return tmp.groupby("bin", observed=False)[TARGET_COL].mean()


def plot_numeric_story(df: pd.DataFrame, feature: str, out_path: Path, bins: int = 10) -> Dict[str, str]:
    rate = rate_by_quantiles(df, feature, bins=bins)
    if rate.empty:
        return {
            "eda_summary": f"Sem dados suficientes para analisar `{feature}`.",
            "business_message": f"`{feature}` não teve suporte amostral suficiente para narrativa robusta.",
        }

    first = float(rate.iloc[0])
    last = float(rate.iloc[-1])
    direction = "aumenta" if last > first else "diminui"
    delta = abs(last - first)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(rate)), rate.values, marker="o")
    plt.ylim(0, 1)
    plt.title(f"Taxa de inadimplência por quantis de {feature}")
    plt.xlabel("Faixa (quantil)")
    plt.ylabel("Inadimplência média")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    eda_summary = (
        f"Na EDA, a taxa de inadimplência {direction} ao longo dos quantis de {feature}. "
        f"Ela vai de ~{first:.1%} para ~{last:.1%}, diferença de ~{delta:.1%}."
    )
    business_message = (
        f"{feature} é um sinal útil: quando {feature} muda, o risco tende a {direction}. "
        "Isso ajuda a priorizar cobranças com maior chance de inadimplência."
    )
    return {"eda_summary": eda_summary, "business_message": business_message}


def plot_categorical_story(df: pd.DataFrame, feature: str, out_path: Path, top_n: int = 12) -> Dict[str, str]:
    tmp = df[[feature, TARGET_COL]].copy()
    tmp[feature] = tmp[feature].astype(str).fillna("NaN")

    top_cats = tmp[feature].value_counts().head(top_n).index
    tmp = tmp[tmp[feature].isin(top_cats)]
    rate = tmp.groupby(feature)[TARGET_COL].mean().sort_values()
    if rate.empty:
        return {
            "eda_summary": f"Sem dados suficientes para analisar `{feature}`.",
            "business_message": f"`{feature}` não teve suporte amostral suficiente para narrativa robusta.",
        }

    plt.figure(figsize=(9, 5))
    plt.barh(rate.index, rate.values)
    plt.xlim(0, 1)
    plt.title(f"Taxa de inadimplência por {feature} (top {top_n})")
    plt.xlabel("Inadimplência média")
    plt.grid(True, axis="x", alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    lowest_cat = str(rate.index[0])
    highest_cat = str(rate.index[-1])
    lowest_val = float(rate.iloc[0])
    highest_val = float(rate.iloc[-1])
    eda_summary = (
        f"Na EDA, {feature} separa perfis com risco distinto: "
        f"~{lowest_val:.1%} ({lowest_cat}) até ~{highest_val:.1%} ({highest_cat})."
    )
    business_message = (
        f"{feature} ajuda a segmentar ações: perfis com maior taxa podem receber "
        "estratégias de cobrança mais proativas."
    )
    return {"eda_summary": eda_summary, "business_message": business_message}


def load_best_run_meta() -> Dict[str, str]:
    model_info_path = METRICS_DIR / "model_info.json"
    if model_info_path.exists() and MODEL_COMPARISON_PATH.exists():
        model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        comp = pd.read_csv(MODEL_COMPARISON_PATH)
        best_run_id = str(model_info.get("best_run_id", "")).strip()
        if best_run_id:
            row = comp[comp["run_id"].astype(str) == best_run_id]
            if not row.empty:
                r = row.iloc[0]
                return {
                    "run_id": str(r["run_id"]),
                    "model_name": str(r.get("model_name", model_info.get("best_model", "Model"))),
                    "variant": "smote" if int(r.get("use_smote", 0)) == 1 else "nosmote",
                }
        if not comp.empty:
            r = comp.iloc[0]
            return {
                "run_id": str(r["run_id"]),
                "model_name": str(r.get("model_name", "Model")),
                "variant": "smote" if int(r.get("use_smote", 0)) == 1 else "nosmote",
            }

    best_run_path = METRICS_DIR / "best_run.json"
    if best_run_path.exists():
        best = json.loads(best_run_path.read_text(encoding="utf-8"))
        run_id = str(best.get("run_id", "")).strip()
        if run_id:
            return {
                "run_id": run_id,
                "model_name": str(best.get("model_name", "Model")),
                "variant": str(best.get("variant", "nosmote")),
            }

    raise FileNotFoundError("Nenhum best run encontrado. Rode `python -m src.train` antes.")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    STORY_FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_processed()
    train_fe, _ = build_features(train_raw, test_raw)

    best_meta = load_best_run_meta()
    best_run_id = best_meta["run_id"]
    best_model_name = best_meta["model_name"]
    best_variant = best_meta["variant"]

    run_fig_dir = FIG_RUNS_DIR / best_run_id
    top_path = run_fig_dir / "top_features.json"
    if not top_path.exists():
        fallback_top = FIG_DIR / "top_features.json"
        if fallback_top.exists():
            top_path = fallback_top
        else:
            raise FileNotFoundError(f"top_features.json não encontrado para o run {best_run_id}.")

    top_feats = json.loads(top_path.read_text(encoding="utf-8"))

    base_candidates = [c for c in train_fe.columns if c not in {TARGET_COL, "ID_CLIENTE", "DIAS_ATRASO"}]
    top_base = pick_top_base_features(top_feats, base_candidates, k=TOP_K_STORY)

    stories: List[Dict[str, str]] = []
    for feature in top_base:
        out_path = STORY_FIG_DIR / f"story_{feature}.png"
        series = train_fe[feature]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
            story = plot_numeric_story(train_fe, feature, out_path)
            ftype = "numeric"
        else:
            story = plot_categorical_story(train_fe, feature, out_path)
            ftype = "categorical"

        stories.append(
            {
                "feature": feature,
                "type": ftype,
                "image": to_project_relative(out_path),
                "eda_summary": story["eda_summary"],
                "business_message": story["business_message"],
            }
        )

    payload = {
        "best_run_id": best_run_id,
        "best_model": best_model_name,
        "best_variant": best_variant,
        "top_k": TOP_K_STORY,
        "stories": stories,
    }

    (METRICS_DIR / "stakeholder_story.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"OK: stakeholder_story.json gerado com top {TOP_K_STORY} features")
    print(f"OK: imagens em {STORY_FIG_DIR}")


if __name__ == "__main__":
    main()
