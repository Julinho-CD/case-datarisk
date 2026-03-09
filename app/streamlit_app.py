import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import load_model_comparison, load_model_info, load_val_predictions_for_run
from app.pages import eda, executive, explainability, modeling, prediction


def cast_to_str(x):
    # Backward compatibility for models serialized with __main__.cast_to_str.
    return x.astype(str)


st.set_page_config(
    page_title="Datarisk - Delinquency Portfolio App",
    layout="wide",
    initial_sidebar_state="collapsed",
)


if "lang" not in st.session_state:
    st.session_state.lang = "en"


def tr(en_text: str, pt_text: str) -> str:
    return en_text if st.session_state.lang == "en" else pt_text


def render_style():
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


def render_language_switcher():
    col1, col2, col3 = st.columns([8, 1, 1])
    with col1:
        st.caption(tr("Language", "Idioma"))
    if col2.button("English", use_container_width=True):
        st.session_state.lang = "en"
        st.rerun()
    if col3.button("Português", use_container_width=True):
        st.session_state.lang = "pt"
        st.rerun()


render_style()
render_language_switcher()

st.title(tr("Datarisk Delinquency App", "App Datarisk de Inadimplência"))
st.caption(
    tr(
        "Original challenge solution + portfolio extension: executive summary, EDA, modeling, explainability, and prioritization.",
        "Solução original do case + extensão de portfólio: resumo executivo, EDA, modelagem, explicabilidade e priorização.",
    )
)

comparison = load_model_comparison()
model_info = load_model_info()
selected_run_id = None
selected_row = None
val_pred: pd.DataFrame | None = None

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

    val_pred = load_val_predictions_for_run(selected_run_id)

    st.caption(
        tr(
            f"Official portfolio artifact: {selected_row['model_name']} | Run {selected_run_id} | PR-AUC {float(selected_row['pr_auc']):.4f}",
            f"Artefato oficial do portfólio: {selected_row['model_name']} | Run {selected_run_id} | PR-AUC {float(selected_row['pr_auc']):.4f}",
        )
    )

executive.render_top_summary(selected_row, val_pred, tr)

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

if active_page == "executive":
    executive.render_page(selected_row, comparison, val_pred, tr)
elif active_page == "eda":
    eda.render_page(selected_row, selected_run_id, tr)
elif active_page == "modeling":
    modeling.render_page(comparison, selected_row, val_pred, tr)
elif active_page == "explainability":
    explainability.render_page(selected_row, selected_run_id, tr)
elif active_page == "prediction":
    prediction.render_page(selected_run_id, selected_row, tr)
