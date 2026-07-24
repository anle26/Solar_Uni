"""Model & Performance: metrics, live SHAP, and a what-if simulator."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.data import (
    MODEL_PERFORMANCE,
    TRAIN_TEST_CUTOFF,
    load_master_data,
)
from app.models import PLANT_MODEL_PATHS, predict_single, shap_importance
from app.ui import render_page_header, render_upper_bound_note


try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Methodology / Model",
        "Model & Performance",
        "Expected-power model metrics, live explanations, and what-if simulation.",
    )
    st.error(str(error))
    st.stop()

render_page_header(
    "Methodology / Model",
    "Model & Performance",
    "How well the weather-only XGBoost model predicts expected AC power, and what drives its predictions.",
)

st.markdown("### Regression performance")
performance_display = MODEL_PERFORMANCE.rename(
    columns={
        "plant": "Plant",
        "split": "Split",
        "r2": "R²",
        "mae_kw": "MAE (kW)",
        "rmse_kw": "RMSE (kW)",
        "mape_pct": "MAPE (%)",
    }
)
st.dataframe(
    performance_display.style.format(
        {"R²": "{:.4f}", "MAE (kW)": "{:.2f}", "RMSE (kW)": "{:.2f}", "MAPE (%)": "{:.2f}"}
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Plant 2's lower R² is expected, not a modeling failure: a global model "
    "measures every inverter against one plant-wide standard, so it can't "
    "quietly absorb Plant 2's heavier degradation the way a per-inverter "
    "model would."
)

st.markdown("### Actual vs. expected AC power")
plant_options = sorted(data["PLANT_NAME"].dropna().unique().tolist())
scatter_plant = st.selectbox("Plant", options=plant_options, key="scatter_plant")

day_frame = data.loc[data["PLANT_NAME"].eq(scatter_plant) & data["IS_DAY"]].copy()
day_frame["Split"] = day_frame["DATE_TIME"].lt(TRAIN_TEST_CUTOFF).map(
    {True: "Train (in-sample)", False: "Test (out-of-sample)"}
)
scatter_sample = day_frame.sample(
    n=min(3000, len(day_frame)), random_state=42
)
scatter = px.scatter(
    scatter_sample,
    x="EXPECTED_AC_POWER",
    y="AC_POWER",
    color="Split",
    color_discrete_map={
        "Train (in-sample)": "#0d766e",
        "Test (out-of-sample)": "#f4a621",
    },
    opacity=0.45,
    labels={"EXPECTED_AC_POWER": "Expected AC power (kW)", "AC_POWER": "Actual AC power (kW)"},
)
axis_max = float(
    max(scatter_sample["EXPECTED_AC_POWER"].max(), scatter_sample["AC_POWER"].max())
)
scatter.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=axis_max,
    y1=axis_max,
    line=dict(color="#617478", width=1, dash="dot"),
)
scatter.update_layout(
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=10, t=45, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
scatter.update_xaxes(gridcolor="#eef3f1")
scatter.update_yaxes(gridcolor="#e6eeeb")
st.plotly_chart(scatter, width="stretch", config={"displayModeBar": False})
st.caption(
    "Daytime intervals only, sampled for readability. Points on the dotted "
    "line are perfect predictions; points below it are under-performing "
    "inverters — including confirmed faults."
)

st.markdown("### Live SHAP feature importance")
shap_plant = st.selectbox("Plant", options=plant_options, key="shap_plant")
shap_pool = data.loc[data["PLANT_NAME"].eq(shap_plant) & data["IS_DAY"]]
shap_sample = shap_pool.sample(n=min(500, len(shap_pool)), random_state=42)
with st.spinner("Computing SHAP values..."):
    importance = shap_importance(shap_plant, shap_sample)
shap_chart = px.bar(
    importance,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    color_discrete_sequence=["#0d766e"],
    labels={"mean_abs_shap": "Mean |SHAP value| (kW)", "feature": ""},
)
shap_chart.update_layout(
    height=300,
    margin=dict(l=0, r=10, t=10, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
shap_chart.update_yaxes(categoryorder="total ascending")
shap_chart.update_xaxes(gridcolor="#eef3f1")
st.plotly_chart(shap_chart, width="stretch", config={"displayModeBar": False})
st.caption(f"Computed live from `{PLANT_MODEL_PATHS[shap_plant].name}` on a 500-row sample.")

st.markdown("### What-if simulator")
st.write(
    "Adjust the weather inputs to see how the model's expected AC power "
    "responds. This is the same five-feature model used everywhere else in "
    "this app — DC power is never one of the inputs."
)
sim_plant = st.selectbox("Plant", options=plant_options, key="sim_plant")
slider_col, result_col = st.columns([1.4, 1], gap="large")
with slider_col:
    irradiation = st.slider("Irradiation", 0.0, 1.2, 0.5, step=0.01)
    ambient_temperature = st.slider("Ambient temperature (°C)", 20.0, 40.0, 27.0, step=0.5)
    module_temperature = st.slider("Module temperature (°C)", 18.0, 67.0, 32.0, step=0.5)
    hour = st.slider("Hour of day", 0, 23, 12)

predicted = predict_single(
    sim_plant, irradiation, ambient_temperature, module_temperature, hour
)
with result_col:
    st.metric("Predicted expected AC power", f"{predicted:,.1f} kW")
    st.caption(
        "If a real inverter reports far below this at the same weather "
        "conditions, the taxonomy would flag it as a fault."
    )

render_upper_bound_note()
