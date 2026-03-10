import pandas as pd
import streamlit as st

from app.analysis import threshold_row
from app.charts import chart_f1, chart_pr, chart_roc


def render_page(
    comparison: pd.DataFrame | None,
    selected_row: dict | None,
    threshold_curve: pd.DataFrame | None,
    roc_curve_df: pd.DataFrame | None,
    pr_curve_df: pd.DataFrame | None,
    tr,
):
    st.subheader(tr("Modeling And Threshold", "Modelagem e Threshold"))

    if comparison is None or len(comparison) == 0:
        st.warning(
            tr(
                "No benchmark artifact was found. Run `python -m src.train` and export public artifacts.",
                "Nenhum artefato de benchmark foi encontrado. Rode `python -m src.train` e exporte os artefatos públicos.",
            )
        )
        return

    view = comparison.copy()
    view.insert(0, "rank", range(1, len(view) + 1))
    cols = [
        "rank",
        "model_name",
        "use_smote",
        "pr_auc",
        "roc_auc",
        "f1_best_threshold",
        "precision_best_threshold",
        "recall_best_threshold",
        "best_threshold",
    ]
    available = [c for c in cols if c in view.columns]
    view = view[available].copy()
    view = view.rename(
        columns={
            "rank": tr("Rank", "Rank"),
            "model_name": tr("Model", "Modelo"),
            "use_smote": tr("SMOTE", "SMOTE"),
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "f1_best_threshold": "F1@best",
            "precision_best_threshold": tr("Precision@Best", "Precisão@Best"),
            "recall_best_threshold": "Recall@best",
            "best_threshold": tr("Best Threshold", "Melhor Threshold"),
        }
    )
    st.dataframe(view, width="stretch", height=260)
    st.caption(
        tr(
            "Benchmark table shows all tested models. The charts below use the exported final artifact.",
            "A tabela mostra todos os modelos testados. Os gráficos abaixo usam o artefato final exportado.",
        )
    )

    if not selected_row:
        return

    st.markdown(f"**{tr('Selected Model Summary', 'Resumo do modelo selecionado')}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", f"{float(selected_row.get('pr_auc', 0.0)):.4f}")
    m2.metric("ROC-AUC", f"{float(selected_row.get('roc_auc', 0.0)):.4f}")
    m3.metric("F1@0.5", f"{float(selected_row.get('f1_at_05', 0.0)):.4f}")
    m4.metric("F1@best", f"{float(selected_row.get('f1_best_threshold', 0.0)):.4f}")

    if threshold_curve is None or threshold_curve.empty:
        st.info(
            tr(
                "Threshold diagnostics were not exported. Add `artifacts/threshold_curve.csv` to enable the interactive modeling view.",
                "Os diagnósticos de threshold não foram exportados. Adicione `artifacts/threshold_curve.csv` para habilitar a visão interativa de modelagem.",
            )
        )
        return

    default_thr = float(selected_row.get("best_threshold", 0.5))
    thr = st.slider(
        tr("Interactive Threshold", "Threshold interativo"),
        min_value=0.05,
        max_value=0.95,
        value=float(min(0.95, max(0.05, default_thr))),
        step=0.01,
    )
    marker = threshold_row(threshold_curve, thr)

    g1, g2 = st.columns(2)
    with g1:
        if pr_curve_df is None or pr_curve_df.empty:
            st.info(
                tr(
                    "PR curve artifact is unavailable.",
                    "O artefato da curva PR não está disponível.",
                )
            )
        else:
            pr_marker = marker[["recall", "precision"]].copy()
            st.altair_chart(chart_pr(pr_curve_df, pr_marker), use_container_width=True)
    with g2:
        if roc_curve_df is None or roc_curve_df.empty:
            st.info(
                tr(
                    "ROC curve artifact is unavailable.",
                    "O artefato da curva ROC não está disponível.",
                )
            )
        else:
            roc_marker = marker.assign(tpr=marker["recall"])
            st.altair_chart(chart_roc(roc_curve_df, roc_marker[["fpr", "tpr"]]), use_container_width=True)
    st.altair_chart(chart_f1(threshold_curve, marker), use_container_width=True)

    mm = marker.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{mm['precision']:.3f}")
    k2.metric("Recall", f"{mm['recall']:.3f}")
    k3.metric("F1", f"{mm['f1']:.3f}")
    k4.metric(tr("Positive Rate", "Taxa Positiva"), f"{mm['positive_rate']:.3f}")

    cm_df = pd.DataFrame(
        [[int(mm["tn"]), int(mm["fp"])], [int(mm["fn"]), int(mm["tp"])]],
        index=[tr("Actual 0", "Real 0"), tr("Actual 1", "Real 1")],
        columns=[tr("Pred 0", "Pred 0"), tr("Pred 1", "Pred 1")],
    )
    st.markdown(f"**{tr('Confusion Matrix At Active Threshold', 'Matriz de confusão no threshold ativo')}**")
    st.dataframe(cm_df, width="stretch")

    with st.expander(tr("How To Decide Threshold", "Como decidir o threshold")):
        st.markdown(
            tr(
                "- Lower threshold: more cases flagged, higher recall, more workload.\n"
                "- Higher threshold: fewer cases flagged, more selectivity, usually higher precision.\n"
                "- Match threshold to team capacity and business cost of false negatives.",
                "- Threshold menor: mais casos sinalizados, maior recall, mais carga operacional.\n"
                "- Threshold maior: menos casos sinalizados, mais seletividade e geralmente maior precisão.\n"
                "- Ajuste o threshold conforme a capacidade do time e o custo de falso negativo.",
            )
        )
