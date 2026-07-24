"""Loss & Carbon Impact: decomposition, trend, and inverter ranking."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.data import (
    build_fault_events,
    daily_loss_trend,
    filter_plants,
    inverter_ranking,
    load_master_data,
)
from app.ui import render_page_header, render_upper_bound_note


COLOR_MAP = {
    "TOTAL_LOSS": "#d94a4a",
    "PARTIAL_LOSS": "#f4a621",
    "Plant 1": "#0d766e",
    "Plant 2": "#93c01f",
}


try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Operations / Impact",
        "Loss & Carbon Impact",
        "Break down upper-bound energy loss and its associated carbon impact.",
    )
    st.error(str(error))
    st.stop()

plant_options = sorted(data["PLANT_NAME"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### View controls")
    selected_plants = st.multiselect(
        "Plant", options=plant_options, default=plant_options
    )
    top_n = st.slider("Inverters in ranking", min_value=5, max_value=30, value=10)

render_page_header(
    "Operations / Impact",
    "Loss & Carbon Impact",
    "Decompose masked energy loss by plant, fault class, and time, then rank inverters by impact.",
)

if not selected_plants:
    st.warning("Select at least one plant to populate the breakdown.")
    st.stop()

filtered = filter_plants(data, selected_plants)
total_loss = filtered["TRUE_ENERGY_LOSS_KWH"].sum()
total_co2_tonnes = filtered["TRUE_CO2_LOSS_KG"].sum() / 1000

kpi_1, kpi_2, kpi_3 = st.columns(3)
kpi_1.metric("Masked energy loss", f"{total_loss:,.1f} kWh")
kpi_2.metric("Carbon impact", f"{total_co2_tonnes:,.1f} tCO₂")
kpi_3.metric(
    "Emission factor",
    "0.703 kg/kWh",
    help="CEA India, FY2020-21 grid emission factor applied to every masked kWh.",
)

class_col, trend_col = st.columns([1, 1.4], gap="large")

with class_col:
    st.markdown("### Loss by plant and fault class")
    plant_class = filtered.loc[
        filtered["ANOMALY_CLASS"].isin(("TOTAL_LOSS", "PARTIAL_LOSS")),
    ]
    breakdown = (
        plant_class.groupby(["PLANT_NAME", "ANOMALY_CLASS"], observed=True)[
            "TRUE_ENERGY_LOSS_KWH"
        ]
        .sum()
        .reset_index()
    )
    bar = px.bar(
        breakdown,
        x="PLANT_NAME",
        y="TRUE_ENERGY_LOSS_KWH",
        color="ANOMALY_CLASS",
        color_discrete_map=COLOR_MAP,
        labels={
            "PLANT_NAME": "",
            "TRUE_ENERGY_LOSS_KWH": "Energy loss (kWh)",
            "ANOMALY_CLASS": "Fault class",
        },
    )
    bar.update_layout(
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=10, t=45, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    bar.update_yaxes(gridcolor="#e6eeeb")
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

with trend_col:
    st.markdown("### Daily loss trend")
    trend = daily_loss_trend(filtered)
    line = px.line(
        trend,
        x="DATE",
        y="masked_loss_kwh",
        color="PLANT_NAME",
        color_discrete_map=COLOR_MAP,
        labels={
            "DATE": "",
            "masked_loss_kwh": "Energy loss (kWh)",
            "PLANT_NAME": "Plant",
        },
    )
    line.update_layout(
        height=340,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=10, t=45, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    line.update_yaxes(gridcolor="#e6eeeb", rangemode="tozero")
    line.update_xaxes(gridcolor="#eef3f1")
    st.plotly_chart(line, width="stretch", config={"displayModeBar": False})

st.markdown(f"### Top {top_n} inverters by masked loss")
ranking = inverter_ranking(filtered)
event_counts = (
    build_fault_events(filtered).groupby("SOURCE_KEY").size().rename("event_count")
)
ranking = ranking.merge(event_counts, on="SOURCE_KEY", how="left").fillna(
    {"event_count": 0}
)
ranking["event_count"] = ranking["event_count"].astype(int)
ranking.insert(0, "Rank", range(1, len(ranking) + 1))

display = ranking.head(top_n)[
    [
        "Rank",
        "SOURCE_KEY",
        "PLANT_NAME",
        "masked_loss_kwh",
        "co2_loss_tonnes",
        "event_count",
    ]
].rename(
    columns={
        "SOURCE_KEY": "Inverter",
        "PLANT_NAME": "Plant",
        "masked_loss_kwh": "Energy loss (kWh)",
        "co2_loss_tonnes": "CO₂ (t)",
        "event_count": "Fault events",
    }
)
st.dataframe(
    display.style.format(
        {"Energy loss (kWh)": "{:,.1f}", "CO₂ (t)": "{:,.2f}"}
    ),
    hide_index=True,
    width="stretch",
)

render_upper_bound_note()
