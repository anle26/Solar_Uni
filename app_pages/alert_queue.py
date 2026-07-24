"""Alert queue: fault intervals collapsed into prioritized operator events."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import (
    FAULT_CLASSES,
    build_fault_events,
    dataset_meta,
    load_master_data,
)
from app.ui import render_page_header, render_upper_bound_note


try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Operations / Alerts",
        "Alert Queue",
        "A date-aware worklist that turns historical fault intervals into operator actions.",
    )
    st.error(str(error))
    st.stop()

meta = dataset_meta(data)
plant_options = sorted(data["PLANT_NAME"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### View controls")
    selected_plants = st.multiselect(
        "Plant", options=plant_options, default=plant_options
    )
    selected_classes = st.multiselect(
        "Fault class",
        options=list(FAULT_CLASSES),
        default=list(FAULT_CLASSES),
    )
    selected_range = st.slider(
        "Event date range",
        min_value=meta.start_date.date(),
        max_value=meta.end_date.date(),
        value=(meta.start_date.date(), meta.end_date.date()),
    )

render_page_header(
    "Operations / Alerts",
    "Alert Queue",
    (
        "Consecutive fault intervals per inverter, collapsed into events and "
        "ranked by masked energy loss so the worst issue is always on top."
    ),
)

events = build_fault_events(data)
range_start = pd.Timestamp(selected_range[0])
range_end = pd.Timestamp(selected_range[1]) + pd.Timedelta(days=1)

filtered = events.loc[
    events["PLANT_NAME"].isin(selected_plants)
    & events["ANOMALY_CLASS"].isin(selected_classes)
    & events["start_time"].ge(range_start)
    & events["start_time"].lt(range_end)
]

if not selected_plants or not selected_classes:
    st.warning("Select at least one plant and fault class to populate the queue.")
    st.stop()

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Open events", f"{len(filtered):,}")
kpi_2.metric(
    "Dispatch now",
    f"{int((filtered['ANOMALY_CLASS'] == 'TOTAL_LOSS').sum()):,}",
)
kpi_3.metric(
    "Inspect within 24 h",
    f"{int((filtered['ANOMALY_CLASS'] == 'PARTIAL_LOSS').sum()):,}",
)
kpi_4.metric(
    "Masked loss in view",
    f"{filtered['masked_loss_kwh'].sum():,.1f} kWh",
)

st.markdown("### Prioritized worklist")
st.caption(
    "Sorted by masked kWh loss, worst first. Click a column header to re-sort."
)

display = filtered[
    [
        "SOURCE_KEY",
        "PLANT_NAME",
        "ANOMALY_CLASS",
        "start_time",
        "end_time",
        "duration_minutes",
        "masked_loss_kwh",
        "co2_loss_kg",
        "recommended_action",
    ]
].rename(
    columns={
        "SOURCE_KEY": "Inverter",
        "PLANT_NAME": "Plant",
        "ANOMALY_CLASS": "Fault class",
        "start_time": "Start",
        "end_time": "End",
        "duration_minutes": "Duration (min)",
        "masked_loss_kwh": "Energy loss (kWh)",
        "co2_loss_kg": "CO₂ (kg)",
        "recommended_action": "Recommended action",
    }
)

st.dataframe(
    display.style.format(
        {
            "Start": "{:%d %b %Y %H:%M}",
            "End": "{:%d %b %Y %H:%M}",
            "Duration (min)": "{:,.0f}",
            "Energy loss (kWh)": "{:,.1f}",
            "CO₂ (kg)": "{:,.1f}",
        }
    ),
    hide_index=True,
    width="stretch",
    height=520,
)

render_upper_bound_note()
