import altair as alt
import pandas as pd
import streamlit as st

from app.analysis import build_categorical_story, build_numeric_story, select_story_features
from app.charts import PALETTE, story_chart_categorical, story_chart_numeric
from app.loaders import current_data_settings, load_feature_data, load_top_features


def render_page(selected_row: dict | None, selected_run_id: str | None, tr):
    st.subheader(tr("Explainability", "Explicabilidade"))

    data_source, refresh = current_data_settings()
    train_fe, _ = load_feature_data(data_source, refresh)
    if train_fe is None:
        st.warning(
            tr(
                "Data could not be loaded. Provide the official case CSVs locally or allow the app to download them from the official Datarisk repository.",
                "Os dados nao puderam ser carregados. Forneca os CSVs oficiais do case localmente ou permita que o app os baixe do repositorio oficial da Datarisk.",
            )
        )
        return

    top_feats = load_top_features(selected_row, selected_run_id)
    if not top_feats:
        st.info(tr("Feature Importance Data Was Not Found.", "Dados de importancia de features nao foram encontrados."))
        return

    imp_df = pd.DataFrame(top_feats).head(20).copy()
    if imp_df.empty:
        st.info(tr("Feature Importance Data Was Not Found.", "Dados de importancia de features nao foram encontrados."))
        return

    imp_df["rank"] = range(1, len(imp_df) + 1)
    st.altair_chart(
        alt.Chart(imp_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title=tr("Importance", "Importancia")),
            y=alt.Y("feature:N", sort="-x", title=tr("Feature", "Feature")),
            tooltip=["rank:Q", "feature:N", alt.Tooltip("importance:Q", format=".4f")],
            color=alt.value(PALETTE["blue"]),
        )
        .properties(height=460),
        width="stretch",
    )

    with st.expander(tr("Interpretation Notes", "Notas de interpretacao")):
        st.markdown(
            tr(
                "- Importance indicates contribution to model ranking, not causality.\n"
                "- Use this with EDA evidence and business context.\n"
                "- Review data quality and category stability before decisions.",
                "- Importancia indica contribuicao para o ranking do modelo, nao causalidade.\n"
                "- Use junto com evidencias de EDA e contexto de negocio.\n"
                "- Revise qualidade dos dados e estabilidade de categorias antes da decisao.",
            )
        )

    st.markdown(f"**{tr('Top Feature Stories', 'Historias das top features')}**")
    story_n = st.radio(tr("How Many Features", "Quantas Features"), [3, 5], index=1, horizontal=True, key="exp_story_n")
    stories = select_story_features(top_feats, train_fe, top_n=story_n)

    for s in stories:
        feat = s["base_feature"]
        st.markdown(f"**{feat}**")
        if pd.api.types.is_numeric_dtype(train_fe[feat]):
            _, agg = build_numeric_story(train_fe, feat)
            st.caption(tr("Observed Behavior In Numeric Buckets.", "Comportamento observado em faixas numericas."))
            if not agg.empty:
                st.altair_chart(story_chart_numeric(agg, feat), width="stretch")
        else:
            _, agg = build_categorical_story(train_fe, feat)
            st.caption(tr("Observed Behavior Across Categories.", "Comportamento observado entre categorias."))
            if not agg.empty:
                st.altair_chart(story_chart_categorical(agg, feat), width="stretch")
