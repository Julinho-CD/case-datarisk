import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.charts import AXIS_X, PALETTE
from app.loaders import current_data_settings, load_feature_data, load_model_for_run
from src.config import TARGET_COL


def render_page(selected_run_id: str | None, selected_row: dict | None, tr):
    st.subheader(tr("Prediction And Prioritization", "Predicao e priorizacao"))
    st.caption(
        tr(
            "Use this page to simulate operational queues by threshold and risk ranking.",
            "Use esta pagina para simular filas operacionais por threshold e ranking de risco.",
        )
    )

    data_source, refresh = current_data_settings()
    train_fe, test_fe = load_feature_data(data_source, refresh)
    if test_fe is None:
        st.warning(
            tr(
                "Data could not be loaded. Provide the official case CSVs locally or allow the app to download them from the official Datarisk repository.",
                "Os dados nao puderam ser carregados. Forneca os CSVs oficiais do case localmente ou permita que o app os baixe do repositorio oficial da Datarisk.",
            )
        )
        return

    try:
        model, model_source_ref, used_fallback_model, model_source_kind = load_model_for_run(selected_run_id)
    except FileNotFoundError:
        st.warning(
            tr(
                "No local model artifact was found. Load from MLflow artifacts or registry, or configure MLflow for the public demo.",
                "Nenhum artefato local de modelo foi encontrado. Carregue via artefatos ou registry do MLflow, ou configure o MLflow para a demonstracao publica.",
            )
        )
        return

    if model_source_kind.startswith("mlflow"):
        st.success(
            tr(
                f"Model Loaded From MLflow (`{model_source_ref}`).",
                f"Modelo carregado do MLflow (`{model_source_ref}`).",
            )
        )
    elif used_fallback_model and selected_run_id:
        st.info(
            tr(
                "Selected Run Model Is Not Available Locally. Using Fallback `models/model.joblib`.",
                "O modelo da run selecionada nao esta disponivel localmente. Usando fallback `models/model.joblib`.",
            )
        )

    drop_cols = [TARGET_COL, "DIAS_ATRASO", "DATA_PAGAMENTO"]
    X_test = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns], errors="ignore")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories in columns .* encoded as all zeros",
            category=UserWarning,
        )
        test_prob = model.predict_proba(X_test)[:, 1]

    base_pred_cols = [
        c
        for c in [
            "ID_CLIENTE",
            "SAFRA_REF",
            "VALOR_A_PAGAR",
            "VALOR_RELATIVO_RENDA",
            "QTDE_ATRASOS_ANT",
            "TICKET_MEDIO_ANT",
        ]
        if c in test_fe.columns
    ]

    pred_df = test_fe[base_pred_cols].copy()
    if "ID_CLIENTE" not in pred_df.columns:
        pred_df["ID_CLIENTE"] = np.arange(1, len(pred_df) + 1)
    if "SAFRA_REF" not in pred_df.columns:
        pred_df["SAFRA_REF"] = pd.NaT

    pred_df[TARGET_COL] = test_prob
    pred_df["SAFRA_REF"] = pd.to_datetime(pred_df["SAFRA_REF"], errors="coerce").astype(str)

    default_thr = float(selected_row.get("best_threshold", 0.5)) if selected_row else 0.5
    thr = st.slider(
        tr("Active Prioritization Threshold", "Threshold ativo de priorizacao"),
        min_value=0.05,
        max_value=0.95,
        value=float(min(0.95, max(0.05, default_thr))),
        step=0.01,
    )
    pred_df["risk_classified"] = (pred_df[TARGET_COL] >= thr).astype(int)

    n_rows = len(pred_df)
    above_thr_n = int(pred_df["risk_classified"].sum())
    above_thr_pct = (above_thr_n / n_rows * 100.0) if n_rows else 0.0
    p90 = float(pred_df[TARGET_COL].quantile(0.9))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Rows Scored", "Linhas pontuadas"), f"{n_rows:,}".replace(",", "."))
    c2.metric(tr("Average Score", "Score medio"), f"{pred_df[TARGET_COL].mean():.4f}")
    c3.metric("P90", f"{p90:.4f}")
    c4.metric(tr("% Above Threshold", "% acima do threshold"), f"{above_thr_pct:.2f}%")

    with st.expander(tr("Business Decision Policy", "Politica de decisao de negocio")):
        st.markdown(
            tr(
                "- Keep threshold close to the recommended value as default policy.\n"
                "- Lower threshold when retention or risk prevention is priority.\n"
                "- Raise threshold when team capacity is constrained.\n"
                "- Review queue size weekly against SLA and collection outcomes.",
                "- Manter o threshold proximo ao recomendado como politica padrao.\n"
                "- Reduzir o threshold quando retencao ou prevencao de risco for prioridade.\n"
                "- Aumentar o threshold quando a capacidade operacional estiver restrita.\n"
                "- Revisar o tamanho da fila semanalmente conforme SLA e resultados de cobranca.",
            )
        )

    hist_df = pred_df.assign(bin=pd.cut(pred_df[TARGET_COL], bins=np.linspace(0, 1, 21), include_lowest=True))
    hist_df = hist_df["bin"].value_counts().sort_index().reset_index(name="count")
    hist_df["label"] = hist_df["bin"].astype(str)
    st.altair_chart(
        alt.Chart(hist_df)
        .mark_bar(color=PALETTE["blue"], cornerRadiusEnd=4)
        .encode(
            y=alt.Y("label:N", sort=hist_df["label"].tolist(), axis=AXIS_X, title=tr("Score Range", "Faixa de score")),
            x=alt.X("count:Q", title=tr("Count", "Quantidade")),
            tooltip=["label:N", "count:Q"],
        )
        .properties(height=420),
        use_container_width=True,
    )

    st.markdown(f"**{tr('Prioritization Table', 'Tabela de priorizacao')}**")
    top_n = st.selectbox(tr("Top N Highest-Risk Cases", "Top N casos de maior risco"), [20, 50, 100, 200], index=0)
    prio_cols = [
        "ID_CLIENTE",
        "SAFRA_REF",
        TARGET_COL,
        "risk_classified",
        "VALOR_A_PAGAR",
        "VALOR_RELATIVO_RENDA",
        "QTDE_ATRASOS_ANT",
        "TICKET_MEDIO_ANT",
    ]
    prio_cols = [c for c in prio_cols if c in pred_df.columns]

    top_df = pred_df.sort_values(TARGET_COL, ascending=False).head(int(top_n))[prio_cols].copy()
    st.dataframe(top_df, use_container_width=True, height=420)

    csv_bytes = top_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=tr("Download Top Prioritized Cases", "Baixar top casos priorizados"),
        data=csv_bytes,
        file_name="top_prioritized_cases.csv",
        mime="text/csv",
        use_container_width=True,
    )
