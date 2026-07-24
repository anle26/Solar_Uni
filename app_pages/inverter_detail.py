"""Inverter drill-down: actual vs. expected power with fault markers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import (
    build_fault_events,
    inverter_summary,
    inverter_timeseries,
    load_master_data,
)
from app.ui import render_page_header, render_upper_bound_note


CASE_STUDY_INVERTER = "bvBOhCH3iADSZry"
EVENT_COLORS = {"TOTAL_LOSS": "#d94a4a", "PARTIAL_LOSS": "#f4a621"}


try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Operations / Drill-down",
        "Inverter Detail",
        "Actual-versus-expected power, fault markers, and downtime for one inverter.",
    )
    st.error(str(error))
    st.stop()

plant_options = sorted(data["PLANT_NAME"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### View controls")
    selected_plant = st.selectbox("Plant", options=plant_options)
    inverter_options = sorted(
        data.loc[data["PLANT_NAME"].eq(selected_plant), "SOURCE_KEY"]
        .dropna()
        .unique()
        .tolist()
    )
    default_index = (
        inverter_options.index(CASE_STUDY_INVERTER)
        if CASE_STUDY_INVERTER in inverter_options
        else 0
    )
    selected_inverter = st.selectbox(
        "Inverter", options=inverter_options, index=default_index
    )

render_page_header(
    "Operations / Drill-down",
    "Inverter Detail",
    (
        "Actual versus expected AC power for one inverter, with TOTAL_LOSS and "
        "PARTIAL_LOSS periods marked directly on the timeline."
    ),
)

events = build_fault_events(data)
series = inverter_timeseries(data, selected_inverter)
summary = inverter_summary(events, selected_inverter)
inverter_events = events.loc[events["SOURCE_KEY"].eq(selected_inverter)].sort_values(
    "start_time"
)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Masked energy loss", f"{summary['masked_loss_kwh']:,.1f} kWh")
kpi_2.metric("Fault events", f"{summary['event_count']:,}")
kpi_3.metric(
    "Dispatch / inspect",
    f"{summary['total_loss_events']} / {summary['partial_loss_events']}",
)
kpi_4.metric("Downtime", f"{summary['downtime_hours']:,.1f} h")

st.markdown("### Actual vs. expected AC power")
power_chart = go.Figure()
power_chart.add_trace(
    go.Scatter(
        x=series["DATE_TIME"],
        y=series["EXPECTED_AC_POWER"],
        name="Expected AC power",
        line=dict(color="#93c01f", width=1.5, dash="dot"),
    )
)
power_chart.add_trace(
    go.Scatter(
        x=series["DATE_TIME"],
        y=series["AC_POWER"],
        name="Actual AC power",
        line=dict(color="#0d766e", width=1.8),
    )
)
for _, event in inverter_events.iterrows():
    power_chart.add_vrect(
        x0=event["start_time"],
        x1=event["end_time"] + pd.Timedelta(minutes=15),
        fillcolor=EVENT_COLORS[event["ANOMALY_CLASS"]],
        opacity=0.18,
        line_width=0,
    )
power_chart.update_layout(
    height=380,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=10, t=45, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
power_chart.update_yaxes(title="kW", gridcolor="#e6eeeb")
power_chart.update_xaxes(gridcolor="#eef3f1")
st.plotly_chart(power_chart, width="stretch", config={"displayModeBar": False})
st.caption(
    "Shaded bands mark TOTAL_LOSS (red) and PARTIAL_LOSS (orange) intervals "
    "for this inverter."
)

st.markdown("### Power loss over time")
loss_chart = go.Figure(
    go.Bar(
        x=series["DATE_TIME"],
        y=series["POWER_LOSS_KW"],
        marker_color="#d94a4a",
    )
)
loss_chart.update_layout(
    height=220,
    margin=dict(l=0, r=10, t=10, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
loss_chart.update_yaxes(title="kW", gridcolor="#e6eeeb", rangemode="tozero")
loss_chart.update_xaxes(gridcolor="#eef3f1")
st.plotly_chart(loss_chart, width="stretch", config={"displayModeBar": False})

st.markdown("### Fault events for this inverter")
display = inverter_events[
    ["ANOMALY_CLASS", "start_time", "end_time", "duration_minutes", "masked_loss_kwh"]
].rename(
    columns={
        "ANOMALY_CLASS": "Fault class",
        "start_time": "Start",
        "end_time": "End",
        "duration_minutes": "Duration (min)",
        "masked_loss_kwh": "Energy loss (kWh)",
    }
)
st.dataframe(
    display.style.format(
        {
            "Start": "{:%d %b %Y %H:%M}",
            "End": "{:%d %b %Y %H:%M}",
            "Duration (min)": "{:,.0f}",
            "Energy loss (kWh)": "{:,.1f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

render_upper_bound_note()
