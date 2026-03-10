import altair as alt
import pandas as pd
import streamlit as st

from app.analysis import build_categorical_story, build_numeric_story, select_story_features
from app.charts import PALETTE, story_chart_categorical, story_chart_numeric
from app.loaders import (
    current_data_settings,
    get_feature_importance_image_path,
    get_shap_summary_image_path,
    load_feature_data,
    load_feature_importance,
    load_top_features,
)


def render_page(selected_row: dict | None, selected_run_id: str | None, tr):
    st.subheader(tr("Explainability", "Explicabilidade"))

    imp_df = load_feature_importance()
    if imp_df is None or imp_df.empty:
        st.info(
            tr(
                "Feature-importance artifacts were not found. Export `artifacts/feature_importance.csv` and the related figures.",
                "Os artefatos de importância de features não foram encontrados. Exporte `artifacts/feature_importance.csv` e as figuras relacionadas.",
            )
        )
        return

    st.altair_chart(
        alt.Chart(imp_df.head(20))
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title=tr("Importance", "Importância")),
            y=alt.Y("feature:N", sort="-x", title=tr("Feature", "Feature")),
            tooltip=["rank:Q", "feature:N", alt.Tooltip("importance:Q", format=".4f")],
            color=alt.value(PALETTE["blue"]),
        )
        .properties(height=460),
        use_container_width=True,
    )

    feature_importance_image = get_feature_importance_image_path()
    shap_summary_image = get_shap_summary_image_path()
    if feature_importance_image or shap_summary_image:
        st.markdown(f"**{tr('Saved explainability figures', 'Figuras salvas de explicabilidade')}**")
        col1, col2 = st.columns(2)
        with col1:
            if feature_importance_image:
                st.image(
                    str(feature_importance_image),
                    caption=tr("Feature-importance figure", "Figura de importância de features"),
                    use_container_width=True,
                )
            else:
                st.info(tr("Feature-importance image not available.", "A imagem de importância de features não está disponível."))
        with col2:
            if shap_summary_image:
                st.image(
                    str(shap_summary_image),
                    caption=tr("Explainability-summary figure", "Figura-resumo de explicabilidade"),
                    use_container_width=True,
                )
            else:
                st.info(tr("Explainability-summary image not available.", "A imagem-resumo de explicabilidade não está disponível."))

    with st.expander(tr("Interpretation notes", "Notas de interpretação")):
        st.markdown(
            tr(
                "- Importance indicates contribution to model ranking, not causality.\n"
                "- Use this with EDA evidence and business context.\n"
                "- Review data quality and category stability before decisions.",
                "- Importância indica contribuição para o ranking do modelo, não causalidade.\n"
                "- Use isso em conjunto com as evidências de EDA e o contexto de negócio.\n"
                "- Revise a qualidade dos dados e a estabilidade das categorias antes de decidir.",
            )
        )

    data_source, refresh = current_data_settings()
    train_fe, _ = load_feature_data(data_source, refresh)
    if train_fe is None:
        st.info(
            tr(
                "Feature stories depend on the processed training data. The saved explainability artifacts are still available above.",
                "As histórias das features dependem dos dados processados de treino. Os artefatos salvos de explicabilidade continuam disponíveis acima.",
            )
        )
        return

    st.markdown(f"**{tr('Top feature stories', 'Histórias das principais features')}**")
    top_feats = load_top_features(selected_row, selected_run_id)
    story_n = st.radio(tr("How many features", "Quantas features"), [3, 5], index=1, horizontal=True, key="exp_story_n")
    stories = select_story_features(top_feats, train_fe, top_n=story_n)

    for story in stories:
        feat = story["base_feature"]
        st.markdown(f"**{feat}**")
        if pd.api.types.is_numeric_dtype(train_fe[feat]):
            _, agg = build_numeric_story(train_fe, feat)
            st.caption(tr("Observed behavior in numeric buckets.", "Comportamento observado em faixas numéricas."))
            if not agg.empty:
                st.altair_chart(story_chart_numeric(agg, feat), use_container_width=True)
        else:
            _, agg = build_categorical_story(train_fe, feat)
            st.caption(tr("Observed behavior across categories.", "Comportamento observado entre categorias."))
            if not agg.empty:
                st.altair_chart(story_chart_categorical(agg, feat), use_container_width=True)
