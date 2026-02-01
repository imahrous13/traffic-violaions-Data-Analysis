import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Traffic Stops Findings Dashboard", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in ["stop_datetime", "stop_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    bool_cols = ["is_arrested", "search_conducted", "drugs_related_stop"]
    bool_map = {
        "True": True,
        "False": False,
        True: True,
        False: False,
        1: True,
        0: False,
    }

    for c in bool_cols:
        if c in df.columns and df[c].dtype != "bool":
            df[c] = df[c].map(bool_map)

    if (
        "stop_hour" in df.columns
        and df["stop_hour"].dtype != "int64"
        and df["stop_hour"].dtype != "int32"
    ):
        df["stop_hour"] = pd.to_numeric(df["stop_hour"], errors="coerce")

    return df


def safe_rate(series: pd.Series):
    if series is None or series.empty:
        return None
    if series.dtype == "bool":
        return float(series.mean())
    s = series.dropna()
    if s.empty:
        return None
    if s.dtype == "object":
        m = {"true": True, "false": False, "1": True, "0": False}
        s2 = s.astype(str).str.lower().map(m)
        if s2.notna().any():
            return float(s2.mean())
    return None


st.title("Traffic Stops: Findings Dashboard")
st.caption(
    "Interactive EDA dashboard using Plotly. Hover any chart to see counts and rates."
)

file_path = st.sidebar.text_input(
    "Dataset path", value="cleaned_traffic_violations.csv"
)
df = load_data(file_path)

st.sidebar.header("Filters")

date_range = None
if "stop_datetime" in df.columns and df["stop_datetime"].notna().any():
    min_date = df["stop_datetime"].min().date()
    max_date = df["stop_datetime"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

v_sel = "All"
if "violation_raw" in df.columns:
    v_all = ["All"] + sorted(df["violation_raw"].dropna().unique().tolist())
    v_sel = st.sidebar.selectbox("Violation", v_all)

g_sel = "All"
if "driver_gender" in df.columns:
    g_all = ["All"] + sorted(df["driver_gender"].dropna().unique().tolist())
    g_sel = st.sidebar.selectbox("Gender", g_all)

r_sel = "All"
if "driver_race" in df.columns:
    r_all = ["All"] + sorted(df["driver_race"].dropna().unique().tolist())
    r_sel = st.sidebar.selectbox("Race", r_all)

p_sel = "All"
if "period" in df.columns:
    p_order = ["Morning", "Afternoon", "Evening", "Night"]
    uniq = df["period"].dropna().unique().tolist()
    p_all = (
        ["All"]
        + [p for p in p_order if p in uniq]
        + [p for p in sorted(uniq) if p not in p_order]
    )
    p_sel = st.sidebar.selectbox("Period", p_all)

fdf = df.copy()

if (
    date_range
    and isinstance(date_range, tuple)
    and len(date_range) == 2
    and "stop_datetime" in fdf.columns
):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf["stop_datetime"] >= start) & (fdf["stop_datetime"] <= end)]

if v_sel != "All" and "violation_raw" in fdf.columns:
    fdf = fdf[fdf["violation_raw"] == v_sel]

if g_sel != "All" and "driver_gender" in fdf.columns:
    fdf = fdf[fdf["driver_gender"] == g_sel]

if r_sel != "All" and "driver_race" in fdf.columns:
    fdf = fdf[fdf["driver_race"] == r_sel]

if p_sel != "All" and "period" in fdf.columns:
    fdf = fdf[fdf["period"] == p_sel]

st.sidebar.subheader("Export")
csv_bytes = fdf.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "Download filtered CSV",
    data=csv_bytes,
    file_name="filtered_traffic.csv",
    mime="text/csv",
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Stops", f"{len(fdf):,}")

ar = safe_rate(fdf["is_arrested"]) if "is_arrested" in fdf.columns else None
sr = safe_rate(fdf["search_conducted"]) if "search_conducted" in fdf.columns else None
dr = (
    safe_rate(fdf["drugs_related_stop"])
    if "drugs_related_stop" in fdf.columns
    else None
)

k2.metric("Arrest Rate", f"{ar:.2%}" if ar is not None else "N/A")
k3.metric("Search Rate", f"{sr:.2%}" if sr is not None else "N/A")
k4.metric("Drugs-Related Rate", f"{dr:.2%}" if dr is not None else "N/A")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Findings Summary", "Arrest Analysis", "Search & Drugs", "Time Patterns"]
)

with tab1:
    st.subheader("Key Findings (Auto-generated from current filters)")
    insights = []

    if ar is not None:
        insights.append(
            f"- Arrests represent a minority of stops. Current arrest rate: **{ar:.2%}**."
        )
    if sr is not None:
        insights.append(
            f"- Searches indicate higher enforcement intensity. Current search rate: **{sr:.2%}**."
        )
    if dr is not None:
        insights.append(
            f"- Drug-related stops are a smaller subset but show stronger enforcement. Current drug-related rate: **{dr:.2%}**."
        )

    if "violation_raw" in fdf.columns and "is_arrested" in fdf.columns and len(fdf) > 0:
        top = (
            fdf.groupby("violation_raw")["is_arrested"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        if len(top) > 0:
            items = ", ".join([f"{idx} ({val:.2%})" for idx, val in top.items()])
            insights.append(
                f"- Highest arrest rates by violation (top 3): **{items}**."
            )

    if "stop_weekday" in fdf.columns and len(fdf) > 0:
        vc = fdf["stop_weekday"].value_counts()
        if not vc.empty:
            insights.append(
                f"- Day with most violations: **{vc.idxmax()}** ({int(vc.max()):,} stops)."
            )

    if len(insights) == 0:
        st.info("No insights available for the selected filters.")
    else:
        st.markdown("\n".join(insights))

    st.subheader("Filtered Data Preview")
    st.dataframe(fdf, use_container_width=True)

with tab2:
    st.subheader("Arrest Distribution")
    if "is_arrested" in fdf.columns and len(fdf) > 0:
        counts = (
            fdf["is_arrested"]
            .value_counts(dropna=False)
            .rename_axis("is_arrested")
            .reset_index(name="count")
        )
        fig = px.bar(
            counts,
            x="is_arrested",
            y="count",
            hover_data={"count": True},
            title="Arrest vs Non-Arrest",
        )
        fig.update_layout(xaxis_title="Is Arrested", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'is_arrested' column.")

    st.subheader("Arrest Rate by Violation ")
    if "violation_raw" in fdf.columns and "is_arrested" in fdf.columns and len(fdf) > 0:
        plot_df = (
            fdf.groupby("violation_raw")
            .agg(
                arrest_rate=("is_arrested", "mean"),
                total_stops=("is_arrested", "count"),
                arrests=("is_arrested", "sum"),
            )
            .reset_index()
            .sort_values("arrest_rate", ascending=False)
        )
        fig = px.bar(
            plot_df,
            x="arrest_rate",
            y="violation_raw",
            orientation="h",
            hover_data={"total_stops": True, "arrests": True, "arrest_rate": ":.2%"},
            title="Arrest Rate by Violation",
        )
        fig.update_layout(xaxis_title="Arrest Rate", yaxis_title="Violation Type")
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'violation_raw' and 'is_arrested' columns.")

    st.subheader("Arrest Rate by Gender / Race ")
    c1, c2 = st.columns(2)

    with c1:
        if (
            "driver_gender" in fdf.columns
            and "is_arrested" in fdf.columns
            and len(fdf) > 0
        ):
            plot_df = (
                fdf.groupby("driver_gender")
                .agg(
                    arrest_rate=("is_arrested", "mean"),
                    total_stops=("is_arrested", "count"),
                    arrests=("is_arrested", "sum"),
                )
                .reset_index()
            )
            fig = px.bar(
                plot_df,
                x="driver_gender",
                y="arrest_rate",
                hover_data={
                    "total_stops": True,
                    "arrests": True,
                    "arrest_rate": ":.2%",
                },
                title="Arrest Rate by Gender",
            )
            fig.update_layout(xaxis_title="Gender", yaxis_title="Arrest Rate")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 'driver_gender' and 'is_arrested' columns.")

    with c2:
        if (
            "driver_race" in fdf.columns
            and "is_arrested" in fdf.columns
            and len(fdf) > 0
        ):
            plot_df = (
                fdf.groupby("driver_race")
                .agg(
                    arrest_rate=("is_arrested", "mean"),
                    total_stops=("is_arrested", "count"),
                    arrests=("is_arrested", "sum"),
                )
                .reset_index()
                .sort_values("arrest_rate", ascending=False)
            )
            fig = px.bar(
                plot_df,
                y="driver_race",
                x="arrest_rate",
                orientation="h",
                hover_data={
                    "total_stops": True,
                    "arrests": True,
                    "arrest_rate": ":.2%",
                },
                title="Arrest Rate by Race",
            )
            fig.update_layout(xaxis_title="Arrest Rate", yaxis_title="Race")
            fig.update_xaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 'driver_race' and 'is_arrested' columns.")

with tab3:
    st.subheader("Search Rate by Drugs-Related Stop (Hover shows counts)")
    if (
        "drugs_related_stop" in fdf.columns
        and "search_conducted" in fdf.columns
        and len(fdf) > 0
    ):
        plot_df = (
            fdf.groupby("drugs_related_stop")
            .agg(
                search_rate=("search_conducted", "mean"),
                total_stops=("search_conducted", "count"),
                searches=("search_conducted", "sum"),
            )
            .reset_index()
        )
        fig = px.bar(
            plot_df,
            x="drugs_related_stop",
            y="search_rate",
            hover_data={"total_stops": True, "searches": True, "search_rate": ":.2%"},
            title="Search Rate by Drugs-Related Stop",
        )
        fig.update_layout(xaxis_title="Drugs Related Stop", yaxis_title="Search Rate")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'drugs_related_stop' and 'search_conducted' columns.")

    st.subheader("Arrest Rate by Search Conducted (Hover shows counts)")
    if (
        "search_conducted" in fdf.columns
        and "is_arrested" in fdf.columns
        and len(fdf) > 0
    ):
        plot_df = (
            fdf.groupby("search_conducted")
            .agg(
                arrest_rate=("is_arrested", "mean"),
                total_stops=("is_arrested", "count"),
                arrests=("is_arrested", "sum"),
            )
            .reset_index()
        )
        fig = px.bar(
            plot_df,
            x="search_conducted",
            y="arrest_rate",
            hover_data={"total_stops": True, "arrests": True, "arrest_rate": ":.2%"},
            title="Arrest Rate by Search Conducted",
        )
        fig.update_layout(xaxis_title="Search Conducted", yaxis_title="Arrest Rate")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'search_conducted' and 'is_arrested' columns.")

    st.subheader("Arrest Rate by Drugs-Related Stop (Hover shows counts)")
    if (
        "drugs_related_stop" in fdf.columns
        and "is_arrested" in fdf.columns
        and len(fdf) > 0
    ):
        plot_df = (
            fdf.groupby("drugs_related_stop")
            .agg(
                arrest_rate=("is_arrested", "mean"),
                total_stops=("is_arrested", "count"),
                arrests=("is_arrested", "sum"),
            )
            .reset_index()
        )
        fig = px.bar(
            plot_df,
            x="drugs_related_stop",
            y="arrest_rate",
            hover_data={"total_stops": True, "arrests": True, "arrest_rate": ":.2%"},
            title="Arrest Rate by Drugs-Related Stop",
        )
        fig.update_layout(xaxis_title="Drugs Related Stop", yaxis_title="Arrest Rate")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'drugs_related_stop' and 'is_arrested' columns.")

with tab4:
    st.subheader("Violations by Day of Week (Hover shows counts)")
    if "stop_weekday" in fdf.columns and len(fdf) > 0:
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        vc = fdf["stop_weekday"].value_counts().reindex(weekday_order)
        plot_df = vc.dropna().reset_index()
        plot_df.columns = ["stop_weekday", "count"]
        fig = px.bar(
            plot_df,
            x="stop_weekday",
            y="count",
            hover_data={"count": True},
            title="Traffic Violations by Weekday",
        )
        fig.update_layout(xaxis_title="Weekday", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'stop_weekday' column.")

    st.subheader("Arrest Rate by Hour (Hover shows counts)")
    if "stop_hour" in fdf.columns and "is_arrested" in fdf.columns and len(fdf) > 0:
        plot_df = (
            fdf.groupby("stop_hour")
            .agg(
                arrest_rate=("is_arrested", "mean"),
                total_stops=("is_arrested", "count"),
                arrests=("is_arrested", "sum"),
            )
            .reset_index()
            .sort_values("stop_hour")
        )
        fig = px.line(
            plot_df,
            x="stop_hour",
            y="arrest_rate",
            markers=True,
            hover_data={"total_stops": True, "arrests": True, "arrest_rate": ":.2%"},
            title="Arrest Rate by Hour of Day",
        )
        fig.update_layout(xaxis_title="Hour", yaxis_title="Arrest Rate")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'stop_hour' and 'is_arrested' columns.")

    st.subheader("Arrest Rate by Period (Hover shows counts)")
    if "period" in fdf.columns and "is_arrested" in fdf.columns and len(fdf) > 0:
        p_order = ["Morning", "Afternoon", "Evening", "Night"]
        plot_df = (
            fdf.groupby("period")
            .agg(
                arrest_rate=("is_arrested", "mean"),
                total_stops=("is_arrested", "count"),
                arrests=("is_arrested", "sum"),
            )
            .reset_index()
        )
        plot_df["period"] = pd.Categorical(
            plot_df["period"], categories=p_order, ordered=True
        )
        plot_df = plot_df.sort_values("period")
        fig = px.bar(
            plot_df,
            x="period",
            y="arrest_rate",
            hover_data={"total_stops": True, "arrests": True, "arrest_rate": ":.2%"},
            title="Arrest Rate by Time Period",
        )
        fig.update_layout(xaxis_title="Period", yaxis_title="Arrest Rate")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need 'period' and 'is_arrested' columns.")
