import altair as alt
import pandas as pd
import streamlit as st

from app.analysis import build_categorical_story, build_numeric_story, missing_report, select_story_features
from app.charts import AXIS_X, AXIS_X_VERTICAL, PALETTE, pearson_heatmap, story_chart_categorical, story_chart_numeric
from app.loaders import current_data_settings, load_feature_data, load_processed, load_top_features
from src.config import TARGET_COL


def _is_datetime_col(df: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_datetime64_any_dtype(df[column])


def _target_distribution_chart(train: pd.DataFrame, tr):
    vc = train[TARGET_COL].value_counts().rename_axis("target").reset_index(name="count")
    vc["class"] = vc["target"].map({0: tr("Current (0)", "Adimplente (0)"), 1: tr("Delinquent (1)", "Inadimplente (1)")})
    vc["pct"] = vc["count"] / vc["count"].sum()
    base = alt.Chart(vc).encode(
        x=alt.X("class:N", axis=AXIS_X, title=None),
        y=alt.Y("pct:Q", axis=alt.Axis(format=".0%"), title=tr("Share", "Percentual")),
        color=alt.Color("class:N", legend=None, scale=alt.Scale(range=[PALETTE["blue"], PALETTE["orange"]])),
        tooltip=[
            "class:N",
            alt.Tooltip("count:Q", title=tr("Count", "Quantidade"), format=","),
            alt.Tooltip("pct:Q", title=tr("Share", "Percentual"), format=".2%"),
        ],
    )
    return (
        base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        + base.mark_text(dy=-10, color="white").encode(text=alt.Text("pct:Q", format=".1%"))
    ).properties(height=240)


def _delinquency_over_time_chart(train: pd.DataFrame, tr):
    tmp = train.dropna(subset=["SAFRA_REF"]).copy()
    tmp["REF_MONTH"] = tmp["SAFRA_REF"].dt.to_period("M").dt.to_timestamp()
    rate = tmp.groupby("REF_MONTH", as_index=False)[TARGET_COL].mean()
    return (
        alt.Chart(rate)
        .mark_line(color=PALETTE["blue"], strokeWidth=3, point=True)
        .encode(
            x=alt.X("REF_MONTH:T", axis=AXIS_X, title=tr("Reference month", "Mês de referência")),
            y=alt.Y(f"{TARGET_COL}:Q", scale=alt.Scale(domain=[0, 1]), title=tr("Delinquency rate", "Taxa de inadimplência")),
            tooltip=[alt.Tooltip("REF_MONTH:T", title=tr("Month", "Mês")), alt.Tooltip(f"{TARGET_COL}:Q", format=".2%")],
        )
        .properties(height=260)
    )


def _numeric_histogram(df: pd.DataFrame, feature: str, tr):
    plot_df = df[[feature]].dropna().copy()
    if plot_df.empty:
        return None
    return (
        alt.Chart(plot_df)
        .mark_bar(color=PALETTE["blue"], opacity=0.85)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=feature),
            y=alt.Y("count():Q", title=tr("Count", "Quantidade")),
            tooltip=[alt.Tooltip("count():Q", title=tr("Count", "Quantidade"))],
        )
        .properties(height=240)
    )


def _numeric_boxplot(df: pd.DataFrame, feature: str, tr):
    plot_df = df[[feature, TARGET_COL]].dropna().copy()
    if plot_df.empty:
        return None
    plot_df["target_label"] = plot_df[TARGET_COL].map({0: tr("Current (0)", "Adimplente (0)"), 1: tr("Delinquent (1)", "Inadimplente (1)")})
    return (
        alt.Chart(plot_df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("target_label:N", title=None, axis=AXIS_X),
            y=alt.Y(f"{feature}:Q", title=feature),
            color=alt.Color("target_label:N", legend=None, scale=alt.Scale(range=[PALETTE["gray"], PALETTE["orange"]])),
        )
        .properties(height=240)
    )


def _categorical_count_chart(df: pd.DataFrame, feature: str, top_n: int, tr):
    plot_df = df[[feature]].copy()
    plot_df[feature] = plot_df[feature].fillna(tr("Missing", "Ausente")).astype(str)
    top_values = plot_df[feature].value_counts().head(top_n).index.tolist()
    plot_df[feature] = plot_df[feature].apply(lambda value: value if value in top_values else tr("Other", "Outros"))
    counts = plot_df[feature].value_counts().reset_index()
    counts.columns = [feature, "count"]
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title=tr("Count", "Quantidade")),
            y=alt.Y(f"{feature}:N", sort="-x", axis=AXIS_X, title=feature),
            color=alt.value(PALETTE["teal"]),
            tooltip=[f"{feature}:N", "count:Q"],
        )
        .properties(height=320)
    )


def _date_frequency_chart(df: pd.DataFrame, feature: str, tr):
    plot_df = df[[feature]].dropna().copy()
    if plot_df.empty:
        return None
    plot_df["month"] = pd.to_datetime(plot_df[feature], errors="coerce").dt.to_period("M").dt.to_timestamp()
    counts = plot_df.groupby("month", as_index=False).size().rename(columns={"size": "count"})
    return (
        alt.Chart(counts)
        .mark_bar(color=PALETTE["blue"])
        .encode(
            x=alt.X("month:T", title=tr("Month", "Mês")),
            y=alt.Y("count:Q", title=tr("Count", "Quantidade")),
            tooltip=[alt.Tooltip("month:T", title=tr("Month", "Mês")), "count:Q"],
        )
        .properties(height=260)
    )


def _numeric_by_category_chart(df: pd.DataFrame, numeric_feature: str, categorical_feature: str, top_n: int):
    tmp = df[[numeric_feature, categorical_feature]].dropna().copy()
    if tmp.empty:
        return None
    tmp[categorical_feature] = tmp[categorical_feature].astype(str)
    top_values = tmp[categorical_feature].value_counts().head(top_n).index.tolist()
    tmp = tmp[tmp[categorical_feature].isin(top_values)].copy()
    return (
        alt.Chart(tmp)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{categorical_feature}:N", sort=top_values, axis=AXIS_X_VERTICAL, title=categorical_feature),
            y=alt.Y(f"{numeric_feature}:Q", title=numeric_feature),
            color=alt.value(PALETTE["orange"]),
        )
        .properties(height=320)
    )


def _categorical_pair_heatmap(df: pd.DataFrame, cat_a: str, cat_b: str, top_n: int):
    tmp = df[[cat_a, cat_b, TARGET_COL]].dropna().copy()
    if tmp.empty:
        return None
    tmp[cat_a] = tmp[cat_a].astype(str)
    tmp[cat_b] = tmp[cat_b].astype(str)
    top_a = tmp[cat_a].value_counts().head(top_n).index.tolist()
    top_b = tmp[cat_b].value_counts().head(top_n).index.tolist()
    tmp = tmp[tmp[cat_a].isin(top_a) & tmp[cat_b].isin(top_b)].copy()
    agg = (
        tmp.groupby([cat_a, cat_b], observed=False)[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "rate"})
    )
    return (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X(f"{cat_b}:N", axis=AXIS_X_VERTICAL, title=cat_b, sort=top_b),
            y=alt.Y(f"{cat_a}:N", axis=AXIS_X, title=cat_a, sort=top_a),
            color=alt.Color("rate:Q", scale=alt.Scale(scheme="oranges", domain=[0, agg["rate"].max() or 1]), title="rate"),
            tooltip=[f"{cat_a}:N", f"{cat_b}:N", alt.Tooltip("rate:Q", format=".2%")],
        )
        .properties(height=340)
    )


def render_page(selected_row: dict | None, selected_run_id: str | None, tr):
    st.subheader(tr("Data and EDA", "Dados e análise exploratória"))

    data_source, refresh = current_data_settings()
    train, test = load_processed(data_source, refresh)
    train_fe, _ = load_feature_data(data_source, refresh)
    if train is None or test is None or train_fe is None:
        st.warning(
            tr(
                "Data could not be loaded. Provide the official case CSVs locally or allow the app to download them from the official Datarisk repository.",
                "Os dados não puderam ser carregados. Forneça os CSVs oficiais do case localmente ou permita que o app os baixe do repositório oficial da Datarisk.",
            )
        )
        return

    feature_columns = [c for c in train.columns if c != TARGET_COL]
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(train[c])]
    categorical_features = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(train[c]) and not _is_datetime_col(train, c)]
    date_features = [c for c in train.columns if _is_datetime_col(train, c)]
    train_fe_numeric = [c for c in train_fe.columns if pd.api.types.is_numeric_dtype(train_fe[c]) and c not in {TARGET_COL, "ID_CLIENTE"}]
    train_fe_categorical = [c for c in train_fe.columns if c not in train_fe_numeric + [TARGET_COL] and not _is_datetime_col(train_fe, c)]
    monthly_cohorts = train["SAFRA_REF"].dropna().dt.to_period("M").nunique() if "SAFRA_REF" in train.columns else 0
    duplicate_rows = int(train.duplicated().sum())
    unique_clients = int(train["ID_CLIENTE"].nunique()) if "ID_CLIENTE" in train.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Columns in the base", "Colunas da base"), f"{train.shape[1]}")
    c2.metric(tr("Numeric features", "Features numéricas"), f"{len(numeric_features)}")
    c3.metric(tr("Categorical features", "Features categóricas"), f"{len(categorical_features)}")
    c4.metric(tr("Reference cohorts", "Safras analisadas"), f"{monthly_cohorts}")

    c5, c6 = st.columns(2)
    c5.metric(tr("Unique clients", "Clientes únicos"), f"{unique_clients:,}".replace(",", "."))
    c6.metric(tr("Duplicate rows", "Linhas duplicadas"), f"{duplicate_rows:,}".replace(",", "."))

    active_run_id = str(st.session_state.get("analysis_run_id", selected_run_id or "")).strip() or selected_run_id
    tabs = st.tabs(
        [
            tr("Overview", "Visão geral"),
            tr("Numerical", "Numéricas"),
            tr("Categorical", "Categóricas"),
            tr("Dates", "Datas"),
            tr("Combined analyses", "Análises combinadas"),
        ]
    )

    with tabs[0]:
        st.markdown(f"**{tr('Data quality', 'Qualidade dos dados')}**")
        miss = missing_report(train, 15).rename(
            columns={
                "column": tr("column", "coluna"),
                "missing_n": tr("missing_n", "faltantes_n"),
                "missing_pct": tr("missing_%", "faltantes_%"),
            }
        )
        st.dataframe(miss, width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"**{tr('Target distribution', 'Distribuição do target')}**")
            st.altair_chart(_target_distribution_chart(train, tr), use_container_width=True)
        with g2:
            st.markdown(f"**{tr('Delinquency over time', 'Inadimplência ao longo do tempo')}**")
            st.altair_chart(_delinquency_over_time_chart(train, tr), use_container_width=True)

        st.markdown(f"**{tr('Top-feature behavior in EDA', 'Comportamento das principais features na EDA')}**")
        top_feats = load_top_features(selected_row, active_run_id)
        story_n = st.radio(tr("How many features", "Quantas features"), [3, 5], index=1, horizontal=True, key="eda_story_n")
        stories = select_story_features(top_feats, train_fe, top_n=story_n)

        for story in stories:
            feat = story["base_feature"]
            st.markdown(f"**{feat}**")
            if pd.api.types.is_numeric_dtype(train_fe[feat]):
                _, agg = build_numeric_story(train_fe, feat)
                st.caption(tr("Risk trend by bucket for this feature.", "Tendência de risco por faixa para esta feature."))
                if not agg.empty:
                    st.altair_chart(story_chart_numeric(agg, feat), use_container_width=True)
            else:
                _, agg = build_categorical_story(train_fe, feat)
                st.caption(
                    tr(
                        "Risk segmentation by category for this feature.",
                        "Segmentação de risco por categoria para esta feature.",
                    )
                )
                if not agg.empty:
                    st.altair_chart(story_chart_categorical(agg, feat), use_container_width=True)

    with tabs[1]:
        if not train_fe_numeric:
            st.info(tr("No numeric features are available.", "Nenhuma feature numérica está disponível."))
        else:
            numeric_feature = st.selectbox(
                tr("Numeric feature", "Feature numérica"),
                options=train_fe_numeric,
                key="eda_numeric_feature",
            )
            q1, q2 = st.columns(2)
            with q1:
                st.markdown(f"**{tr('Distribution', 'Distribuição')}**")
                hist_chart = _numeric_histogram(train_fe, numeric_feature, tr)
                if hist_chart is not None:
                    st.altair_chart(hist_chart, use_container_width=True)
            with q2:
                st.markdown(f"**{tr('Boxplot by target', 'Boxplot por target')}**")
                box_chart = _numeric_boxplot(train_fe, numeric_feature, tr)
                if box_chart is not None:
                    st.altair_chart(box_chart, use_container_width=True)

            summary = train_fe[[numeric_feature]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            st.dataframe(summary, width="stretch")

            _, agg = build_numeric_story(train_fe, numeric_feature)
            if not agg.empty:
                st.markdown(f"**{tr('Delinquency by bucket', 'Inadimplência por faixa')}**")
                st.altair_chart(story_chart_numeric(agg, numeric_feature), use_container_width=True)

    with tabs[2]:
        if not train_fe_categorical:
            st.info(tr("No categorical features are available.", "Nenhuma feature categórica está disponível."))
        else:
            categorical_feature = st.selectbox(
                tr("Categorical feature", "Feature categórica"),
                options=train_fe_categorical,
                key="eda_categorical_feature",
            )
            top_n = st.slider(tr("Top categories", "Top categorias"), min_value=5, max_value=20, value=10, step=1, key="eda_top_categories")
            cplot1, cplot2 = st.columns(2)
            with cplot1:
                st.markdown(f"**{tr('Category frequency', 'Frequência das categorias')}**")
                st.altair_chart(_categorical_count_chart(train_fe, categorical_feature, top_n, tr), use_container_width=True)
            with cplot2:
                st.markdown(f"**{tr('Delinquency by category', 'Inadimplência por categoria')}**")
                _, agg = build_categorical_story(train_fe, categorical_feature)
                if not agg.empty:
                    st.altair_chart(story_chart_categorical(agg.head(top_n), categorical_feature), use_container_width=True)

            top_table = (
                train_fe[categorical_feature]
                .fillna(tr("Missing", "Ausente"))
                .astype(str)
                .value_counts(dropna=False)
                .head(top_n)
                .rename_axis(categorical_feature)
                .reset_index(name=tr("count", "quantidade"))
            )
            st.dataframe(top_table, width="stretch")

    with tabs[3]:
        if not date_features:
            st.info(tr("No date columns are available.", "Nenhuma coluna de data está disponível."))
        else:
            date_feature = st.selectbox(tr("Date column", "Coluna de data"), options=date_features, key="eda_date_feature")
            series = pd.to_datetime(train[date_feature], errors="coerce")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(tr("Non-null rows", "Linhas não nulas"), f"{int(series.notna().sum()):,}".replace(",", "."))
            d2.metric(tr("Unique dates", "Datas únicas"), f"{int(series.nunique()):,}".replace(",", "."))
            d3.metric(tr("Min date", "Data mínima"), "-" if series.dropna().empty else str(series.min().date()))
            d4.metric(tr("Max date", "Data máxima"), "-" if series.dropna().empty else str(series.max().date()))
            date_chart = _date_frequency_chart(train, date_feature, tr)
            if date_chart is not None:
                st.altair_chart(date_chart, use_container_width=True)

    with tabs[4]:
        st.markdown(f"**{tr('Correlation view', 'Visão de correlação')}**")
        default_corr = [c for c in [TARGET_COL, "VALOR_A_PAGAR", "TAXA", "QTDE_ATRASOS_ANT", "TICKET_MEDIO_ANT"] if c in train_fe_numeric + [TARGET_COL]]
        corr_source_cols = [c for c in train_fe.columns if pd.api.types.is_numeric_dtype(train_fe[c]) and c not in {"ID_CLIENTE"}]
        corr_features = st.multiselect(
            tr("Features for the Pearson heatmap", "Features para o heatmap de Pearson"),
            options=corr_source_cols,
            default=default_corr if len(default_corr) >= 3 else corr_source_cols[:8],
            key="eda_corr_features",
        )
        if len(corr_features) >= 2:
            corr_chart, _ = pearson_heatmap(train_fe, corr_features)
            st.altair_chart(corr_chart, use_container_width=True)
        else:
            st.info(tr("Select at least 2 features.", "Selecione ao menos 2 features."))

        st.markdown(f"**{tr('Numeric vs categorical', 'Numérica vs categórica')}**")
        if train_fe_numeric and train_fe_categorical:
            num_col, cat_col = st.columns(2)
            with num_col:
                selected_num = st.selectbox(tr("Numeric variable", "Variável numérica"), train_fe_numeric, key="eda_combo_num")
            with cat_col:
                selected_cat = st.selectbox(tr("Categorical variable", "Variável categórica"), train_fe_categorical, key="eda_combo_cat")
            combo_chart = _numeric_by_category_chart(train_fe, selected_num, selected_cat, top_n=10)
            if combo_chart is not None:
                st.altair_chart(combo_chart, use_container_width=True)
        else:
            st.info(tr("Numeric/categorical combinations are unavailable.", "As combinações numérica/categórica não estão disponíveis."))

        st.markdown(f"**{tr('Categorical pair heatmap', 'Heatmap entre categóricas')}**")
        if len(train_fe_categorical) >= 2:
            cat1_col, cat2_col = st.columns(2)
            with cat1_col:
                cat_a = st.selectbox(tr("First categorical", "Primeira categórica"), train_fe_categorical, index=0, key="eda_cat_a")
            with cat2_col:
                cat_b_options = [c for c in train_fe_categorical if c != cat_a]
                cat_b = st.selectbox(tr("Second categorical", "Segunda categórica"), cat_b_options, index=0, key="eda_cat_b")
            heatmap = _categorical_pair_heatmap(train_fe, cat_a, cat_b, top_n=8)
            if heatmap is not None:
                st.altair_chart(heatmap, use_container_width=True)
        else:
            st.info(tr("Categorical pair analysis is unavailable.", "A análise entre duas categóricas não está disponível."))
