import altair as alt
import pandas as pd
import streamlit as st

from app import loaders
from app.analysis import build_categorical_story, build_numeric_story, select_story_features
from app.charts import PALETTE, story_chart_categorical, story_chart_numeric


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
    st.session_state["analysis_run_id"] = selected_run_id

    return comparison.loc[comparison["run_id"].astype(str) == str(selected_run_id)].iloc[0].to_dict()


def render_page(comparison: pd.DataFrame | None, selected_row: dict | None, tr):
    st.subheader(tr("Explainability", "Explicabilidade"))

    if comparison is None or len(comparison) == 0:
        st.info(
            tr(
                "Feature-importance artifacts were not found. Run `python -m src.train` to regenerate them.",
                "Os artefatos de importância de features não foram encontrados. Rode `python -m src.train` para gerá-los novamente.",
            )
        )
        return

    active_row = _select_model_row(comparison, selected_row, tr)
    active_run_id = str(active_row.get("run_id", "")).strip()
    imp_loader = getattr(loaders, "load_run_feature_importance", loaders.load_feature_importance)
    imp_df = imp_loader(active_run_id) if imp_loader is not loaders.load_feature_importance else imp_loader()
    if imp_df is None or imp_df.empty:
        st.info(
            tr(
                "Feature-importance artifacts were not found for the selected run.",
                "Os artefatos de importância de features não foram encontrados para a run selecionada.",
            )
        )
        return

    st.caption(
        tr(
            f"Run {active_run_id or '-'} loaded for the explainability view.",
            f"Run {active_run_id or '-'} carregada para a visão de explicabilidade.",
        )
    )

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

    data_source, refresh = loaders.current_data_settings()
    train_fe, _ = loaders.load_feature_data(data_source, refresh)
    if train_fe is None:
        st.info(
            tr(
                "Feature stories depend on the processed training data.",
                "As histórias das features dependem dos dados processados de treino.",
            )
        )
        return

    st.markdown(f"**{tr('Top feature stories', 'Histórias das principais features')}**")
    top_feats = loaders.load_top_features(active_row, active_run_id)
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
