import altair as alt
import pandas as pd

PALETTE = {
    "teal": "#0F766E",
    "blue": "#1D4ED8",
    "orange": "#F97316",
    "gray": "#94A3B8",
    "green": "#16A34A",
}

AXIS_X = alt.Axis(labelAngle=0, labelLimit=220)


def chart_pr(thr_df: pd.DataFrame, marker_df: pd.DataFrame):
    return (
        alt.Chart(thr_df)
        .mark_line(color=PALETTE["blue"], strokeWidth=3)
        .encode(x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision"))
        + alt.Chart(marker_df).mark_point(color=PALETTE["orange"], size=120).encode(x="recall:Q", y="precision:Q")
    ).properties(height=260)


def chart_roc(thr_df: pd.DataFrame, marker_df: pd.DataFrame):
    return (
        alt.Chart(thr_df)
        .mark_line(color=PALETTE["teal"], strokeWidth=3)
        .encode(x=alt.X("fpr:Q", title="False Positive Rate"), y=alt.Y("recall:Q", title="True Positive Rate"))
        + alt.Chart(marker_df).mark_point(color=PALETTE["orange"], size=120).encode(x="fpr:Q", y="recall:Q")
    ).properties(height=260)


def chart_f1(thr_df: pd.DataFrame, marker_df: pd.DataFrame):
    return (
        alt.Chart(thr_df)
        .mark_line(color=PALETTE["green"], strokeWidth=3)
        .encode(x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("f1:Q", title="F1"))
        + alt.Chart(marker_df).mark_point(color=PALETTE["orange"], size=120).encode(x="threshold:Q", y="f1:Q")
    ).properties(height=240)


def pearson_heatmap(df: pd.DataFrame, columns: list[str]):
    corr = df[columns].corr(method="pearson").reset_index().melt("index")
    corr.columns = ["feature_x", "feature_y", "corr"]

    chart = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("feature_x:N", axis=AXIS_X, title=None),
            y=alt.Y("feature_y:N", axis=AXIS_X, title=None),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["feature_x:N", "feature_y:N", alt.Tooltip("corr:Q", format=".3f")],
        )
        .properties(height=360)
    )
    corr_df = df[columns].corr(method="pearson")
    return chart, corr_df


def story_chart_numeric(agg: pd.DataFrame, feature: str):
    return (
        alt.Chart(agg)
        .mark_line(color=PALETTE["blue"], strokeWidth=3, point=True)
        .encode(
            x=alt.X("bucket:N", sort=None, axis=AXIS_X, title=f"Buckets of {feature}"),
            y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title="Delinquency rate"),
            tooltip=["bucket:N", alt.Tooltip("rate:Q", format=".2%"), "volume:Q"],
        )
        .properties(height=230)
    )


def story_chart_categorical(agg: pd.DataFrame, feature: str):
    return (
        alt.Chart(agg)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), title="Delinquency rate"),
            y=alt.Y(f"{feature}:N", sort="-x", axis=AXIS_X, title=feature),
            color=alt.Color("volume:Q", scale=alt.Scale(scheme="blues"), title="Volume"),
            tooltip=[f"{feature}:N", alt.Tooltip("rate:Q", format=".2%"), "volume:Q"],
        )
        .properties(height=260)
    )
