"""How It Works: the seven-step fault diagnosis pipeline, explained."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.data import (
    MODEL_FEATURES,
    MODEL_PERFORMANCE,
    RAW_LOSS_KWH,
    inverter_window,
    load_master_data,
)
from app.ui import render_page_header, render_step_flow, render_upper_bound_note


CASE_STUDY_INVERTER = "bvBOhCH3iADSZry"
LEAKAGE_WINDOW = ("2020-06-14 08:00:00", "2020-06-14 16:00:00")

PIPELINE_STEPS = [
    ("Preprocess", "Correct the DC power scale factor and align every plant to 15-minute intervals."),
    ("Engineer features", f"Keep only {len(MODEL_FEATURES)} weather/time features — DC power is deliberately excluded."),
    ("Train a global model", "Fit one XGBoost regressor per plant across all of its inverters, not one model per inverter."),
    ("Predict expected power", "Score every interval to get EXPECTED_AC_POWER — what the plant should have produced."),
    ("Apply the fault taxonomy", "Compare actual vs. expected to classify NORMAL / PARTIAL_LOSS / TOTAL_LOSS."),
    ("Mask the loss", "Sum kWh loss only on confirmed-fault intervals, not on every positive residual."),
    ("Quantify carbon", "Convert masked kWh to CO2 using the 0.703 kgCO2/kWh grid emission factor."),
]

try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Methodology / Pipeline",
        "How It Works",
        "A visual explanation of the seven-step fault diagnosis workflow.",
    )
    st.error(str(error))
    st.stop()

render_page_header(
    "Methodology / Pipeline",
    "How It Works",
    "Seven steps turn raw 15-minute SCADA readings into a defensible, upper-bound loss estimate.",
)

render_step_flow(PIPELINE_STEPS)

st.markdown("### Why DC power is excluded from the feature set")
st.write(
    "If the model could see DC power, it would learn `AC ≈ 0.98 × DC` and stop "
    "there. When an inverter's DC input drops — a failed string, a tripped "
    "breaker — the model would predict a low AC power too, and the residual "
    "would collapse to zero: the fault disappears instead of being detected. "
    "The chart below is a real occurrence for inverter "
    f"`{CASE_STUDY_INVERTER}` on 14 June 2020: actual AC power and corrected "
    "DC power fall to zero together at 11:15, while EXPECTED_AC_POWER — "
    "predicted from weather alone — stays high. That gap is the signal a "
    "DC-aware model would have erased."
)

window = inverter_window(
    data,
    CASE_STUDY_INVERTER,
    LEAKAGE_WINDOW[0],
    LEAKAGE_WINDOW[1],
)
leakage_chart = go.Figure()
leakage_chart.add_trace(
    go.Scatter(
        x=window["DATE_TIME"],
        y=window["EXPECTED_AC_POWER"],
        name="Expected AC power (weather-only model)",
        line=dict(color="#93c01f", width=1.8, dash="dot"),
    )
)
leakage_chart.add_trace(
    go.Scatter(
        x=window["DATE_TIME"],
        y=window["AC_POWER"],
        name="Actual AC power",
        line=dict(color="#0d766e", width=2),
    )
)
leakage_chart.add_trace(
    go.Scatter(
        x=window["DATE_TIME"],
        y=window["DC_POWER_CORRECTED"],
        name="Corrected DC power (excluded feature)",
        line=dict(color="#d94a4a", width=1.6, dash="dash"),
    )
)
leakage_chart.update_layout(
    height=340,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=10, t=45, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
leakage_chart.update_yaxes(title="kW", gridcolor="#e6eeeb")
leakage_chart.update_xaxes(gridcolor="#eef3f1")
st.plotly_chart(leakage_chart, width="stretch", config={"displayModeBar": False})

st.markdown("### Global model vs. local model: why a lower R² can be correct")
st.write(
    "A model trained per-inverter (*local*) fits each unit's own history so "
    "closely that a slow degradation gets absorbed as \"normal\" behavior — "
    "the fault is masked by the model itself. A single model trained across "
    "the whole plant (*global*) has no such memory: an inverter that drifts "
    "from the plant-wide pattern shows up as prediction error. Plant 2's "
    "lower test R² is not a worse model — it is the global model correctly "
    "refusing to explain away Plant 2's heavier degradation."
)

r2_chart = go.Figure()
for plant, color in (("Plant 1", "#0d766e"), ("Plant 2", "#93c01f")):
    plant_rows = MODEL_PERFORMANCE.loc[MODEL_PERFORMANCE["plant"].eq(plant)]
    r2_chart.add_trace(
        go.Bar(
            x=plant_rows["split"],
            y=plant_rows["r2"],
            name=plant,
            marker_color=color,
        )
    )
r2_chart.update_layout(
    height=300,
    barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=10, t=45, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
r2_chart.update_yaxes(title="R²", gridcolor="#e6eeeb", range=[0, 1])
st.plotly_chart(r2_chart, width="stretch", config={"displayModeBar": False})

st.markdown("### Masked loss aggregation: avoiding positive drift bias")
st.write(
    "Summing `max(expected − actual, 0)` across every interval accumulates "
    "ordinary forecast noise in one direction, inflating the total. Counting "
    "loss only where the taxonomy confirms a fault removes that bias."
)
raw_col, masked_col = st.columns(2)
raw_col.metric("Raw loss (before masking) — not used for reporting", f"{RAW_LOSS_KWH:,.1f} kWh")
masked_col.metric(
    "Masked loss (taxonomy-confirmed) — used everywhere in this app",
    f"{data['TRUE_ENERGY_LOSS_KWH'].sum():,.1f} kWh",
)

render_upper_bound_note()
