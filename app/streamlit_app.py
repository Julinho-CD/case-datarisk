import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import (
    load_best_run,
    load_model_comparison,
    load_pr_curve,
    load_roc_curve,
    load_threshold_curve,
    load_val_predictions_best,
)
from app.pages import eda, executive, explainability, modeling, prediction

ENABLE_CUSTOM_CHROME_CSS = False


def cast_to_str(x):
    return x.astype(str)


def debug_log(message: str):
    print(f"[streamlit-app] {message}", flush=True)


def safe_render(label: str, render_fn, *args, **kwargs):
    try:
        return render_fn(*args, **kwargs)
    except Exception as exc:
        debug_log(f"{label} failed: {exc!r}")
        st.error(f"{label} failed.")
        st.exception(exc)
        return None


def render_style(enabled: bool = False):
    base_css = """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(2,6,23,0.72));
        border: 1px solid rgba(59,130,246,0.18);
        border-radius: 16px;
        padding: 0.45rem 0.7rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    div[data-testid="stAlert"] {
        border: 1px solid rgba(148,163,184,0.35) !important;
        border-radius: 16px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(15,23,42,0.86), rgba(2,6,23,0.72));
        box-shadow: 0 16px 32px rgba(2,6,23,0.22);
        overflow: hidden;
    }
    div[data-testid="stExpander"] details {
        background: transparent;
    }
    div[data-testid="stExpander"] summary {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    div[data-testid="stExpander"] summary p {
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    div[data-testid="stExpander"] details[open] {
        border-color: rgba(59,130,246,0.26);
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        padding-top: 0.2rem;
    }
    </style>
    """

    chrome_css = """
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[kind="header"][aria-label="Open sidebar"] {
        display: none !important;
    }
    button[kind="header"][aria-label="Close sidebar"] {
        display: none !important;
    }
    </style>
    """

    if not enabled:
        debug_log("custom chrome CSS disabled for debugging")

    st.markdown(base_css + (chrome_css if enabled else ""), unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Datarisk - Delinquency Portfolio App",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    def tr(en_text: str, pt_text: str) -> str:
        return en_text if st.session_state.lang == "en" else pt_text

    def render_language_switcher():
        col1, col2, col3 = st.columns([8, 1, 1])
        with col1:
            st.caption(tr("Language", "Idioma"))
        if col2.button("English", width="stretch"):
            st.session_state.lang = "en"
            st.rerun()
        if col3.button("Português", width="stretch"):
            st.session_state.lang = "pt"
            st.rerun()

    debug_log("app started")

    render_style(enabled=ENABLE_CUSTOM_CHROME_CSS)
    render_language_switcher()

    st.title(tr("Datarisk Delinquency App", "App Datarisk de Inadimplência"))
    st.caption(
        tr(
            "Original challenge solution + portfolio extension: executive summary, EDA, modeling, explainability, and prioritization.",
            "Solução original do case + extensão de portfólio: resumo executivo, análise exploratória, modelagem, explicabilidade e priorização.",
        )
    )

    comparison = None
    best_run = {}
    selected_run_id = None
    selected_row = None
    val_pred: pd.DataFrame | None = None
    threshold_curve = None
    roc_curve_df = None
    pr_curve_df = None

    try:
        comparison = load_model_comparison()
    except Exception as exc:
        debug_log(f"load_model_comparison failed: {exc!r}")
        st.error("Failed to load model comparison artifact.")
        st.exception(exc)

    try:
        best_run = load_best_run()
    except Exception as exc:
        debug_log(f"load_best_run failed: {exc!r}")
        st.error("Failed to load best run artifact.")
        st.exception(exc)

    try:
        val_pred = load_val_predictions_best()
    except Exception as exc:
        debug_log(f"load_val_predictions_best failed: {exc!r}")
        st.error("Failed to load validation predictions artifact.")
        st.exception(exc)

    try:
        threshold_curve = load_threshold_curve()
    except Exception as exc:
        debug_log(f"load_threshold_curve failed: {exc!r}")
        st.error("Failed to load threshold curve artifact.")
        st.exception(exc)

    try:
        roc_curve_df = load_roc_curve()
    except Exception as exc:
        debug_log(f"load_roc_curve failed: {exc!r}")
        st.error("Failed to load ROC curve artifact.")
        st.exception(exc)

    try:
        pr_curve_df = load_pr_curve()
    except Exception as exc:
        debug_log(f"load_pr_curve failed: {exc!r}")
        st.error("Failed to load PR curve artifact.")
        st.exception(exc)

    if comparison is None or len(comparison) == 0:
        st.warning(
            tr(
                "No public benchmark artifact found. Run `python -m src.train` and commit the generated `artifacts/` directory.",
                "Nenhum artefato público de benchmark foi encontrado. Rode `python -m src.train` e versione a pasta `artifacts/` gerada.",
            )
        )
    else:
        best_run_id = str(best_run.get("run_id", "")).strip()
        if best_run_id and "run_id" in comparison.columns:
            official_row = comparison[comparison["run_id"].astype(str) == best_run_id]
            if not official_row.empty:
                selected_row = official_row.iloc[0].to_dict()

        if selected_row is None:
            selected_row = comparison.iloc[0].to_dict()

    if selected_row is None and best_run:
        selected_row = dict(best_run)

    if selected_row is not None:
        selected_run_id = str(selected_row.get("run_id", best_run.get("run_id", ""))).strip() or None
        model_name = str(selected_row.get("model_name", best_run.get("model_name", "Model")))
        pr_auc = float(selected_row.get("pr_auc", best_run.get("pr_auc", 0.0)))
        st.caption(
            tr(
                f"Official portfolio artifact: {model_name} | Run {selected_run_id or '-'} | PR-AUC {pr_auc:.4f}",
                f"Artefato oficial do portfólio: {model_name} | Run {selected_run_id or '-'} | PR-AUC {pr_auc:.4f}",
            )
        )

    debug_log(
        "artifacts: "
        f"comparison_loaded={comparison is not None and len(comparison) > 0}, "
        f"best_run_loaded={bool(best_run)}, "
        f"val_pred_loaded={val_pred is not None}, "
        f"threshold_curve_loaded={threshold_curve is not None}, "
        f"roc_curve_loaded={roc_curve_df is not None}, "
        f"pr_curve_loaded={pr_curve_df is not None}"
    )

    safe_render("executive.render_top_summary", executive.render_top_summary, selected_row, val_pred, tr)

    page_labels = {
        "executive": tr("Executive", "Executivo"),
        "eda": tr("Data and exploratory analysis", "Dados e análise exploratória"),
        "modeling": tr("Modeling", "Modelagem"),
        "explainability": tr("Explainability", "Explicabilidade"),
        "prediction": tr("Prediction", "Predição"),
    }

    if "active_page" not in st.session_state:
        st.session_state.active_page = "executive"

    active_page = st.radio(
        tr("Section", "Seção"),
        options=list(page_labels.keys()),
        format_func=lambda key: page_labels[key],
        horizontal=True,
        key="active_page",
        label_visibility="collapsed",
    )

    debug_log(f"active_page={active_page}")

    page_dispatch = {
        "executive": ("executive.render_page", executive.render_page, (selected_row, comparison, val_pred, tr)),
        "eda": ("eda.render_page", eda.render_page, (selected_row, selected_run_id, tr)),
        "modeling": (
            "modeling.render_page",
            modeling.render_page,
            (comparison, selected_row, threshold_curve, roc_curve_df, pr_curve_df, tr),
        ),
        "explainability": ("explainability.render_page", explainability.render_page, (selected_row, selected_run_id, tr)),
        "prediction": ("prediction.render_page", prediction.render_page, (selected_row, tr)),
    }

    page_label, page_fn, page_args = page_dispatch[active_page]
    safe_render(page_label, page_fn, *page_args)
