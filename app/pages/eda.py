import altair as alt
import pandas as pd
import streamlit as st

from app.analysis import build_categorical_story, build_numeric_story, missing_report, select_story_features
from app.charts import AXIS_X, PALETTE, pearson_heatmap, story_chart_categorical, story_chart_numeric
from app.loaders import current_data_settings, load_feature_data, load_processed, load_top_features
from src.config import TARGET_COL


def render_page(selected_row: dict | None, selected_run_id: str | None, tr):
    st.subheader(tr("Data and EDA", "Dados e EDA"))

    data_source, refresh = current_data_settings()
    train, test = load_processed(data_source, refresh)
    train_fe, _ = load_feature_data(data_source, refresh)
    if train is None or test is None or train_fe is None:
        st.warning(
            tr(
                "Data could not be loaded. Provide the official case CSVs locally or allow the app to download them from the official Datarisk repository.",
                "Os dados nao puderam ser carregados. Forneca os CSVs oficiais do case localmente ou permita que o app os baixe do repositorio oficial da Datarisk.",
            )
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Rows (train)", "Linhas (train)"), f"{train.shape[0]:,}".replace(",", "."))
    c2.metric(tr("Rows (test)", "Linhas (test)"), f"{test.shape[0]:,}".replace(",", "."))
    c3.metric(tr("Columns (train)", "Colunas (train)"), f"{train.shape[1]}")
    c4.metric(tr("Delinquency Rate", "Taxa De Inadimplencia"), f"{train[TARGET_COL].mean() * 100:.2f}%")

    st.markdown(f"**{tr('Data quality', 'Qualidade dos dados')}**")
    miss = missing_report(train, 15).rename(
        columns={
            "column": tr("column", "coluna"),
            "missing_n": tr("missing_n", "faltantes_n"),
            "missing_pct": tr("missing_%", "faltantes_%"),
        }
    )
    st.dataframe(miss, width="stretch")

    st.markdown(f"**{tr('Target Distribution', 'Distribuicao do target')}**")
    vc = train[TARGET_COL].value_counts().rename_axis("target").reset_index(name="count")
    vc["class"] = vc["target"].map({0: tr("Current (0)", "Adimplente (0)"), 1: tr("Delinquent (1)", "Inadimplente (1)")})
    st.altair_chart(
        alt.Chart(vc)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("class:N", axis=AXIS_X, title=None),
            y=alt.Y("count:Q", title=tr("Count", "Quantidade")),
            color=alt.Color("class:N", legend=None, scale=alt.Scale(range=[PALETTE["blue"], PALETTE["orange"]])),
        )
        .properties(height=220),
        use_container_width=True,
    )

    st.markdown(f"**{tr('Delinquency Over Time', 'Inadimplencia ao longo do tempo')}**")
    tmp = train.dropna(subset=["SAFRA_REF"]).copy()
    tmp["REF_MONTH"] = tmp["SAFRA_REF"].dt.to_period("M").dt.to_timestamp()
    rate = tmp.groupby("REF_MONTH", as_index=False)[TARGET_COL].mean()
    st.altair_chart(
        alt.Chart(rate)
        .mark_line(color=PALETTE["blue"], strokeWidth=3, point=True)
        .encode(
            x=alt.X("REF_MONTH:T", axis=AXIS_X, title=tr("Reference Month", "Mes de referencia")),
            y=alt.Y(f"{TARGET_COL}:Q", scale=alt.Scale(domain=[0, 1]), title=tr("Delinquency Rate", "Taxa de inadimplencia")),
        )
        .properties(height=260),
        use_container_width=True,
    )

    st.markdown(f"**{tr('Correlation View', 'Visao de correlacao')}**")
    numeric_cols = [c for c in train_fe.columns if pd.api.types.is_numeric_dtype(train_fe[c]) and c not in {"ID_CLIENTE"}]
    default_corr = [c for c in [TARGET_COL, "VALOR_A_PAGAR", "TAXA", "QTDE_ATRASOS_ANT", "TICKET_MEDIO_ANT"] if c in numeric_cols]
    corr_features = st.multiselect(
        tr("Features For Pearson Heatmap", "Features para heatmap de Pearson"),
        options=numeric_cols,
        default=default_corr if len(default_corr) >= 3 else numeric_cols[:8],
    )
    if len(corr_features) >= 2:
        corr_chart, _ = pearson_heatmap(train_fe, corr_features)
        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.info(tr("Select At Least 2 Features.", "Selecione ao menos 2 features."))

    st.markdown(f"**{tr('Top Feature Behavior In EDA', 'Comportamento das top features na EDA')}**")
    top_feats = load_top_features(selected_row, selected_run_id)
    story_n = st.radio(tr("How Many Features", "Quantas Features"), [3, 5], index=1, horizontal=True, key="eda_story_n")
    stories = select_story_features(top_feats, train_fe, top_n=story_n)

    for s in stories:
        feat = s["base_feature"]
        st.markdown(f"**{feat}**")
        if pd.api.types.is_numeric_dtype(train_fe[feat]):
            _, agg = build_numeric_story(train_fe, feat)
            st.caption(tr("Risk Trend By Bucket For This Feature.", "Tendencia de risco por faixa desta feature."))
            if not agg.empty:
                st.altair_chart(story_chart_numeric(agg, feat), use_container_width=True)
        else:
            _, agg = build_categorical_story(train_fe, feat)
            st.caption(
                tr(
                    "Risk segmentation by category for this feature.",
                    "Segmentacao de risco por categoria para esta feature.",
                )
            )
            if not agg.empty:
                st.altair_chart(story_chart_categorical(agg, feat), use_container_width=True)
