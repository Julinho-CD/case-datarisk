import pandas as pd
import streamlit as st

from app.analysis import compute_threshold_table, threshold_row


def render_top_summary(selected_row: dict | None, val_pred: pd.DataFrame | None, tr):
    st.markdown(f"### {tr('Executive Summary', 'Resumo Executivo')}")

    if not selected_row:
        st.warning(tr("No Benchmark Model Is Available.", "Nenhum Modelo De Benchmark Está Disponível."))
        return

    model_name = str(selected_row.get("model_name", "-"))
    pr_auc = float(selected_row.get("pr_auc", 0.0))
    roc_auc = float(selected_row.get("roc_auc", 0.0))
    thr = float(selected_row.get("best_threshold", 0.5))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Official Model", "Modelo Oficial"), model_name)
    c2.metric("PR-AUC", f"{pr_auc:.3f}")
    c3.metric("ROC-AUC", f"{roc_auc:.3f}")
    c4.metric(tr("Recommended Threshold", "Threshold Recomendado"), f"{thr:.2f}")

    if val_pred is not None:
        y_true = val_pred["y_true"].astype(int).values
        y_prob = val_pred["y_prob"].astype(float).values
        mm = threshold_row(compute_threshold_table(y_true, y_prob), thr).iloc[0]
        st.caption(
            tr(
                f"At threshold {thr:.2f}: precision={mm['precision']:.3f}, recall={mm['recall']:.3f}, positive-rate={mm['positive_rate']:.3f}.",
                f"No threshold {thr:.2f}: precisão={mm['precision']:.3f}, recall={mm['recall']:.3f}, taxa-positiva={mm['positive_rate']:.3f}.",
            )
        )

    with st.expander(tr("How This Supports Business Decisions", "Como Isso Apoia A Decisão De Negócio")):
        st.markdown(
            tr(
                "- Use score + threshold to prioritize collection/risk review queues.\n"
                "- Lower threshold increases recall and workload.\n"
                "- Higher threshold increases selectivity and usually precision.",
                "- Use score + threshold para priorizar filas de cobrança/risco.\n"
                "- Threshold menor aumenta recall e carga operacional.\n"
                "- Threshold maior aumenta seletividade e tende a elevar precisão.",
            )
        )


def render_page(selected_row: dict | None, comparison: pd.DataFrame | None, val_pred: pd.DataFrame | None, tr):
    st.subheader(tr("Executive View", "Visão Executiva"))

    if selected_row is None:
        st.info(tr("Run Training First: `python -m src.train`.", "Rode O Treino Primeiro: `python -m src.train`."))
        return

    render_top_summary(selected_row, val_pred, tr)

    st.markdown(f"**{tr('Project Positioning', 'Posicionamento Do Projeto')}**")
    st.markdown(
        tr(
            "- Original technical challenge solution delivered first.\n"
            "- This app is a portfolio extension focused on decision support.\n"
            "- Built for recruiters, technical reviewers, and business managers.",
            "- Solução original do case técnico entregue primeiro.\n"
            "- Este app é uma extensão de portfólio focada em suporte à decisão.\n"
            "- Construído para recrutadores, avaliadores técnicos e gestores.",
        )
    )

    if comparison is not None and len(comparison) > 0:
        best3 = comparison[["model_name", "pr_auc", "roc_auc", "best_threshold"]].head(3).copy()
        best3.columns = [
            tr("Model", "Modelo"),
            "PR-AUC",
            "ROC-AUC",
            tr("Best Threshold", "Melhor Threshold"),
        ]
        st.markdown(f"**{tr('Top Benchmark Runs', 'Top Runs Do Benchmark')}**")
        st.dataframe(best3, width="stretch")

    st.markdown(f"**{tr('Business Decision Framework', 'Framework De Decisão De Negócio')}**")
    if selected_row and val_pred is not None:
        thr = float(selected_row.get("best_threshold", 0.5))
        y_true = val_pred["y_true"].astype(int).values
        y_prob = val_pred["y_prob"].astype(float).values
        mm = threshold_row(compute_threshold_table(y_true, y_prob), thr).iloc[0]
        st.caption(
            tr(
                f"Recommended policy at threshold {thr:.2f}: action queue covers {mm['positive_rate']*100:.1f}% of cases, "
                f"with precision {mm['precision']:.3f} and recall {mm['recall']:.3f}.",
                f"Política recomendada no threshold {thr:.2f}: fila de ação cobre {mm['positive_rate']*100:.1f}% dos casos, "
                f"com precisão {mm['precision']:.3f} e recall {mm['recall']:.3f}.",
            )
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{tr('Operational Use', 'Uso Operacional')}**")
        st.markdown(
            tr(
                "- Score all invoices/customers.\n"
                "- Apply threshold to create the high-risk queue.\n"
                "- Prioritize top scores when capacity is limited.",
                "- Pontuar todas as faturas/clientes.\n"
                "- Aplicar threshold para criar fila de alto risco.\n"
                "- Priorizar maiores scores quando a capacidade for limitada.",
            )
        )
    with col_b:
        st.markdown(f"**{tr('Error-Cost Trade-Off', 'Trade-Off De Custo De Erro')}**")
        st.markdown(
            tr(
                "- False Negative (high-risk missed): delayed action and potential loss.\n"
                "- False Positive (low-risk flagged): unnecessary operational effort.\n"
                "- Threshold should balance expected loss vs team workload.",
                "- Falso Negativo (alto risco não sinalizado): atraso na ação e potencial perda.\n"
                "- Falso Positivo (baixo risco sinalizado): esforço operacional desnecessário.\n"
                "- O threshold deve equilibrar perda esperada vs capacidade do time.",
            )
        )
