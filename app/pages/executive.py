import pandas as pd
import streamlit as st

from app.analysis import compute_threshold_table, threshold_row


def render_top_summary(selected_row: dict | None, val_pred: pd.DataFrame | None, tr):
    if not selected_row:
        st.warning(tr("No benchmark model is available.", "Nenhum modelo de benchmark está disponível."))
        return

    model_name = str(selected_row.get("model_name", "-"))
    pr_auc = float(selected_row.get("pr_auc", 0.0))
    roc_auc = float(selected_row.get("roc_auc", 0.0))
    thr = float(selected_row.get("best_threshold", 0.5))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Official model", "Modelo oficial"), model_name)
    c2.metric("PR-AUC", f"{pr_auc:.3f}")
    c3.metric("ROC-AUC", f"{roc_auc:.3f}")
    c4.metric(tr("Recommended threshold", "Threshold recomendado"), f"{thr:.2f}")

    if val_pred is not None:
        y_true = val_pred["y_true"].astype(int).values
        y_prob = val_pred["y_prob"].astype(float).values
        mm = threshold_row(compute_threshold_table(y_true, y_prob), thr).iloc[0]
        st.caption(
            tr(
                f"At threshold {thr:.2f}: precision={mm['precision']:.3f}, recall={mm['recall']:.3f}, positive-rate={mm['positive_rate']:.3f}.",
                f"No threshold {thr:.2f}: precisão={mm['precision']:.3f}, recall={mm['recall']:.3f}, taxa positiva={mm['positive_rate']:.3f}.",
            )
        )
    else:
        st.info(
            tr(
                "Validation predictions are unavailable for this run. Summary metrics remain visible, but threshold diagnostics are hidden.",
                "As predições de validação não estão disponíveis para esta run. As métricas-resumo continuam visíveis, mas os diagnósticos de threshold foram ocultados.",
            )
        )

    with st.expander(tr("How this supports business decisions", "Como isso apoia a decisão de negócio")):
        st.markdown(
            tr(
                "- Use score + threshold to prioritize collection and risk-review queues.\n"
                "- Lower threshold increases recall and workload.\n"
                "- Higher threshold increases selectivity and usually precision.",
                "- Use score + threshold para priorizar filas de cobrança e revisão de risco.\n"
                "- Threshold menor aumenta recall e carga operacional.\n"
                "- Threshold maior aumenta seletividade e tende a elevar a precisão.",
            )
        )


def render_page(selected_row: dict | None, comparison: pd.DataFrame | None, val_pred: pd.DataFrame | None, tr):
    if selected_row is None:
        st.info(tr("Run training first: `python -m src.train`.", "Rode o treino primeiro: `python -m src.train`."))
        return

    with st.expander(tr("Project positioning", "Posicionamento do projeto")):
        st.markdown(
            tr(
                "- Original technical challenge solution delivered first.\n"
                "- This app is a portfolio extension focused on decision support.\n"
                "- Built for recruiters, technical reviewers, and business managers.",
                "- A solução original do case técnico foi entregue primeiro.\n"
                "- Este app é uma extensão de portfólio focada em suporte à decisão.\n"
                "- Foi construído para recrutadores, avaliadores técnicos e gestores.",
            )
        )

    if comparison is not None and len(comparison) > 0:
        best3 = comparison[["model_name", "pr_auc", "roc_auc", "best_threshold"]].head(3).copy()
        best3.columns = [
            tr("Model", "Modelo"),
            "PR-AUC",
            "ROC-AUC",
            tr("Best threshold", "Melhor threshold"),
        ]
        st.markdown(f"**{tr('Top benchmark runs', 'Melhores runs do benchmark')}**")
        st.dataframe(best3, width="stretch")

    st.markdown(f"**{tr('Business decision framework', 'Framework de decisão de negócio')}**")
    if val_pred is not None:
        thr = float(selected_row.get("best_threshold", 0.5))
        y_true = val_pred["y_true"].astype(int).values
        y_prob = val_pred["y_prob"].astype(float).values
        mm = threshold_row(compute_threshold_table(y_true, y_prob), thr).iloc[0]
        st.caption(
            tr(
                f"Recommended policy at threshold {thr:.2f}: action queue covers {mm['positive_rate']*100:.1f}% of cases, "
                f"with precision {mm['precision']:.3f} and recall {mm['recall']:.3f}.",
                f"Política recomendada no threshold {thr:.2f}: a fila de ação cobre {mm['positive_rate']*100:.1f}% dos casos, "
                f"com precisão {mm['precision']:.3f} e recall {mm['recall']:.3f}.",
            )
        )
    else:
        st.info(
            tr(
                "Validation predictions are unavailable for this run, so the threshold policy preview cannot be rendered.",
                "As predições de validação não estão disponíveis para esta run, então a prévia da política de threshold não pode ser exibida.",
            )
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{tr('Operational use', 'Uso operacional')}**")
        st.markdown(
            tr(
                "- Score all invoices or customers.\n"
                "- Apply the threshold to build the high-risk queue.\n"
                "- Prioritize the highest scores when capacity is limited.",
                "- Pontue todas as faturas ou clientes.\n"
                "- Aplique o threshold para formar a fila de alto risco.\n"
                "- Priorize os maiores scores quando a capacidade for limitada.",
            )
        )
    with col_b:
        st.markdown(f"**{tr('Error-cost trade-off', 'Trade-off entre custos de erro')}**")
        st.markdown(
            tr(
                "- False Negative (high-risk missed): delayed action and potential loss.\n"
                "- False Positive (low-risk flagged): unnecessary operational effort.\n"
                "- Threshold should balance expected loss vs. team workload.",
                "- Falso negativo (alto risco não sinalizado): atraso na ação e perda potencial.\n"
                "- Falso positivo (baixo risco sinalizado): esforço operacional desnecessário.\n"
                "- O threshold deve equilibrar perda esperada e capacidade do time.",
            )
        )
