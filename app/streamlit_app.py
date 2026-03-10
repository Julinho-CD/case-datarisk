import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import load_model_comparison, load_model_info, load_val_predictions_for_run
from app.pages import eda, executive, explainability, modeling, prediction

ENABLE_CUSTOM_CHROME_CSS = False


def cast_to_str(x):
    # Backward compatibility for models serialized with __main__.cast_to_str.
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
    if not enabled:
        debug_log("custom chrome CSS disabled for debugging")
        return

    st.markdown(
        """
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
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        div[data-testid="stAlert"] {
            border: 1px solid rgba(148,163,184,0.35) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
            "Solução original do case + extensão de portfólio: resumo executivo, EDA, modelagem, explicabilidade e priorização.",
        )
    )

    comparison = None
    model_info = {}
    selected_run_id = None
    selected_row = None
    val_pred: pd.DataFrame | None = None

    try:
        comparison = load_model_comparison()
    except Exception as exc:
        debug_log(f"load_model_comparison failed: {exc!r}")
        st.error("Failed to load comparison artifact.")
        st.exception(exc)

    try:
        model_info = load_model_info()
    except Exception as exc:
        debug_log(f"load_model_info failed: {exc!r}")
        st.error("Failed to load model info artifact.")
        st.exception(exc)

    if comparison is None or len(comparison) == 0:
        st.warning(
            tr(
                "No benchmark artifact found. Run `python -m src.train` to generate model outputs.",
                "Nenhum artefato de benchmark encontrado. Rode `python -m src.train` para gerar as saídas de modelo.",
            )
        )
    else:
        best_run_id = str(model_info.get("best_run_id", "")).strip()
        if best_run_id:
            official_row = comparison[comparison["run_id"].astype(str) == best_run_id]
            if not official_row.empty:
                selected_row = official_row.iloc[0].to_dict()
                selected_run_id = best_run_id

        if selected_row is None:
            selected_row = comparison.iloc[0].to_dict()
            selected_run_id = str(selected_row["run_id"])

        if selected_run_id:
            try:
                val_pred = load_val_predictions_for_run(selected_run_id)
            except Exception as exc:
                debug_log(f"load_val_predictions_for_run failed: {exc!r}")
                st.error("Failed to load validation predictions.")
                st.exception(exc)

        st.caption(
            tr(
                f"Official portfolio artifact: {selected_row['model_name']} | Run {selected_run_id} | PR-AUC {float(selected_row['pr_auc']):.4f}",
                f"Artefato oficial do portfólio: {selected_row['model_name']} | Run {selected_run_id} | PR-AUC {float(selected_row['pr_auc']):.4f}",
            )
        )

    debug_log(
        "artifacts: "
        f"comparison_loaded={comparison is not None and len(comparison) > 0}, "
        f"model_info_loaded={bool(model_info)}, "
        f"val_pred_loaded={val_pred is not None}"
    )

    safe_render("executive.render_top_summary", executive.render_top_summary, selected_row, val_pred, tr)

    page_labels = {
        "executive": tr("Executive", "Executivo"),
        "eda": tr("Data and Exploratory Data Analysis", "Dados e Análise Exploratória"),
        "modeling": tr("Modeling", "Modelagem"),
        "explainability": tr("Explainability", "Explicabilidade"),
        "prediction": tr("Prediction", "Predição"),
    }

    if "active_page" not in st.session_state:
        st.session_state.active_page = "executive"

    active_page = st.radio(
        tr("Section", "Seção"),
        options=list(page_labels.keys()),
        format_func=lambda k: page_labels[k],
        horizontal=True,
        key="active_page",
        label_visibility="collapsed",
    )

    debug_log(f"active_page={active_page}")

    page_dispatch = {
        "executive": ("executive.render_page", executive.render_page, (selected_row, comparison, val_pred, tr)),
        "eda": ("eda.render_page", eda.render_page, (selected_row, selected_run_id, tr)),
        "modeling": ("modeling.render_page", modeling.render_page, (comparison, selected_row, val_pred, tr)),
        "explainability": ("explainability.render_page", explainability.render_page, (selected_row, selected_run_id, tr)),
        "prediction": ("prediction.render_page", prediction.render_page, (selected_run_id, selected_row, tr)),
    }

    page_label, page_fn, page_args = page_dispatch[active_page]
    safe_render(page_label, page_fn, *page_args)
