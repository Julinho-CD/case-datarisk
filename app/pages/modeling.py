import pandas as pd
import streamlit as st

from app.analysis import threshold_row
from app.charts import chart_f1, chart_pr, chart_roc
from app.loaders import load_run_pr_curve, load_run_roc_curve, load_run_threshold_curve


def _model_option_label(row: pd.Series, tr) -> str:
    smote_value = pd.to_numeric(row.get("use_smote", 0), errors="coerce")
    smote_enabled = bool(int(smote_value)) if pd.notna(smote_value) else False
    smote_label = tr("with SMOTE", "com SMOTE") if smote_enabled else tr("without SMOTE", "sem SMOTE")
    model_name = str(row.get("model_name", "Model"))
    pr_auc = float(row.get("pr_auc", 0.0) or 0.0)
    return f"{model_name} | {smote_label} | PR-AUC {pr_auc:.4f}"


def _select_model_row(comparison: pd.DataFrame, selected_row: dict | None, tr) -> dict:
    run_ids = comparison["run_id"].astype(str).tolist()
    label_map = {
        str(row["run_id"]): _model_option_label(row, tr)
        for _, row in comparison.iterrows()
    }

    default_run_id = None
    if selected_row is not None:
        default_run_id = str(selected_row.get("run_id", "")).strip() or None
    if not default_run_id and run_ids:
        default_run_id = run_ids[0]

    current_run_id = st.session_state.get("analysis_run_id", default_run_id)
    if current_run_id not in run_ids:
        current_run_id = default_run_id
        st.session_state["analysis_run_id"] = default_run_id

    selected_run_id = st.selectbox(
        tr("Model for analysis", "Modelo para análise"),
        options=run_ids,
        index=run_ids.index(current_run_id) if current_run_id in run_ids else 0,
        format_func=lambda run_id: label_map.get(run_id, run_id),
        key="analysis_run_id",
    )

    return comparison.loc[comparison["run_id"].astype(str) == str(selected_run_id)].iloc[0].to_dict()


def render_page(
    comparison: pd.DataFrame | None,
    selected_row: dict | None,
    tr,
):
    st.subheader(tr("Modeling and threshold", "Modelagem e threshold"))

    if comparison is None or len(comparison) == 0:
        st.warning(
            tr(
                "No benchmark artifact was found. Run `python -m src.train` and export public artifacts.",
                "Nenhum artefato de benchmark foi encontrado. Rode `python -m src.train` e exporte os artefatos públicos.",
            )
        )
        return

    active_row = _select_model_row(comparison, selected_row, tr)
    active_run_id = str(active_row.get("run_id", "")).strip()

    n_train = pd.to_numeric(active_row.get("n_train"), errors="coerce")
    n_test = pd.to_numeric(active_row.get("n_test"), errors="coerce")
    train_share = pd.to_numeric(active_row.get("train_share"), errors="coerce")
    test_share = pd.to_numeric(active_row.get("test_share"), errors="coerce")
    if pd.notna(n_train) and pd.notna(n_test):
        train_share_pct = float(train_share) * 100 if pd.notna(train_share) else 0.0
        test_share_pct = float(test_share) * 100 if pd.notna(test_share) else 0.0
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(tr("Rows used for train", "Linhas usadas para treino"), f"{int(n_train):,}".replace(",", "."))
        d2.metric(tr("Rows used for test", "Linhas usadas para teste"), f"{int(n_test):,}".replace(",", "."))
        d3.metric(tr("Train share", "% treino"), f"{train_share_pct:.1f}%")
        d4.metric(tr("Test share", "% teste"), f"{test_share_pct:.1f}%")
        st.caption(
            tr(
                "Evaluation split: stratified 80/20 holdout on the labeled development base.",
                "Split de avaliação: holdout estratificado 80/20 na base rotulada de desenvolvimento.",
            )
        )

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
            "best_threshold": tr("Best threshold", "Melhor threshold"),
        }
    )
    st.dataframe(view, width="stretch", height=260)

    st.markdown(f"**{tr('Selected model summary', 'Resumo do modelo selecionado')}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", f"{float(active_row.get('pr_auc', 0.0)):.4f}")
    m2.metric("ROC-AUC", f"{float(active_row.get('roc_auc', 0.0)):.4f}")
    m3.metric("F1@0.5", f"{float(active_row.get('f1_at_05', 0.0)):.4f}")
    m4.metric("F1@best", f"{float(active_row.get('f1_best_threshold', 0.0)):.4f}")
    st.caption(
        tr(
            f"Run {active_run_id or '-'} loaded for the interactive diagnostics below.",
            f"Run {active_run_id or '-'} carregada para os diagnósticos interativos abaixo.",
        )
    )

    threshold_curve = load_run_threshold_curve(active_run_id)
    roc_curve_df = load_run_roc_curve(active_run_id)
    pr_curve_df = load_run_pr_curve(active_run_id)
    if threshold_curve is None or threshold_curve.empty:
        st.info(
            tr(
                "Threshold diagnostics were not exported for this run.",
                "Os diagnósticos de threshold não foram exportados para esta run.",
            )
        )
        return

    default_thr = float(active_row.get("best_threshold", 0.5))
    thr = st.slider(
        tr("Interactive threshold", "Threshold interativo"),
        min_value=0.05,
        max_value=0.95,
        value=float(min(0.95, max(0.05, default_thr))),
        step=0.01,
    )
    marker = threshold_row(threshold_curve, thr)

    g1, g2 = st.columns(2)
    with g1:
        if pr_curve_df is None or pr_curve_df.empty:
            st.info(tr("PR-curve artifact is unavailable.", "O artefato da curva PR não está disponível."))
        else:
            pr_marker = marker[["recall", "precision"]].copy()
            st.altair_chart(chart_pr(pr_curve_df, pr_marker), use_container_width=True)
    with g2:
        if roc_curve_df is None or roc_curve_df.empty:
            st.info(tr("ROC-curve artifact is unavailable.", "O artefato da curva ROC não está disponível."))
        else:
            roc_marker = marker.assign(tpr=marker["recall"])
            st.altair_chart(chart_roc(roc_curve_df, roc_marker[["fpr", "tpr"]]), use_container_width=True)
    st.altair_chart(chart_f1(threshold_curve, marker), use_container_width=True)

    mm = marker.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{mm['precision']:.3f}")
    k2.metric("Recall", f"{mm['recall']:.3f}")
    k3.metric("F1", f"{mm['f1']:.3f}")
    k4.metric(tr("Positive rate", "Taxa positiva"), f"{mm['positive_rate']:.3f}")

    cm_df = pd.DataFrame(
        [[int(mm["tn"]), int(mm["fp"])], [int(mm["fn"]), int(mm["tp"])]],
        index=[tr("Actual 0", "Real 0"), tr("Actual 1", "Real 1")],
        columns=[tr("Pred 0", "Pred 0"), tr("Pred 1", "Pred 1")],
    )
    st.markdown(f"**{tr('Confusion matrix at the active threshold', 'Matriz de confusão no threshold ativo')}**")
    st.dataframe(cm_df, width="stretch")

    with st.expander(tr("How to choose the threshold", "Como escolher o threshold")):
        st.markdown(
            tr(
                "- Lower threshold: more cases flagged, higher recall, more workload.\n"
                "- Higher threshold: fewer cases flagged, more selectivity, usually higher precision.\n"
                "- Match threshold to team capacity and business cost of false negatives.",
                "- Threshold menor: mais casos sinalizados, maior recall e mais carga operacional.\n"
                "- Threshold maior: menos casos sinalizados, mais seletividade e geralmente maior precisão.\n"
                "- Ajuste o threshold conforme a capacidade do time e o custo de falso negativo.",
            )
        )
