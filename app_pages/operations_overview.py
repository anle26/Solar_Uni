"""Operations overview for historical plant monitoring."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import (
    FAULT_CLASSES,
    dataset_meta,
    filter_plants,
    load_master_data,
    overview_aggregates,
    snapshot_summary,
)
from app.ui import render_page_header, render_upper_bound_note


COLOR_MAP = {
    "TOTAL_LOSS": "#d94a4a",
    "PARTIAL_LOSS": "#f4a621",
    "Plant 1": "#0d766e",
    "Plant 2": "#93c01f",
}


def _format_int(value: float) -> str:
    return f"{value:,.0f}"


def _format_one(value: float) -> str:
    return f"{value:,.1f}"


try:
    data = load_master_data()
except (FileNotFoundError, ValueError) as error:
    render_page_header(
        "Operations / Overview",
        "Plant operations at a glance",
        "Fault activity, masked energy loss, and associated carbon impact.",
    )
    st.error(str(error))
    st.stop()

meta = dataset_meta(data)
plant_options = sorted(data["PLANT_NAME"].dropna().unique().tolist())
fault_dates = data.loc[
    data["ANOMALY_CLASS"].isin(FAULT_CLASSES), "DATE_TIME"
].dt.date
default_operating_date = (
    fault_dates.max() if not fault_dates.empty else meta.end_date.date()
)

with st.sidebar:
    st.markdown("### View controls")
    selected_plants = st.multiselect(
        "Plant",
        options=plant_options,
        default=plant_options,
        help="Historical KPIs and the daily snapshot use this plant filter.",
    )
    selected_date_value = st.slider(
        "Simulated operating date",
        min_value=meta.start_date.date(),
        max_value=meta.end_date.date(),
        value=default_operating_date,
        help=(
            "The dataset is historical. This control simulates the day an "
            "operator would have been monitoring."
        ),
    )
    st.caption(
        f"{meta.start_date:%d %b %Y} – {meta.end_date:%d %b %Y} · "
        f"{meta.row_count:,} records"
    )

render_page_header(
    "Operations / Overview",
    "Plant operations at a glance",
    (
        "Prioritize inverter attention while tracking masked energy loss and "
        "the associated grid-carbon impact."
    ),
)

if not selected_plants:
    st.warning("Select at least one plant to populate the dashboard.")
    st.stop()

filtered = filter_plants(data, selected_plants)
selected_date = pd.Timestamp(selected_date_value)
snapshot = snapshot_summary(filtered, selected_date)

# KPIs and the plant comparison table are cumulative through the simulated
# date — an operator on that day couldn't see loss from days that haven't
# happened yet. The timeline chart keeps the full history for context.
cumulative = filtered.loc[filtered["DATE"].le(selected_date)]
plant_summary, _ = overview_aggregates(cumulative)
_, daily_faults = overview_aggregates(filtered)

total_loss = cumulative["TRUE_ENERGY_LOSS_KWH"].sum()
total_co2_tonnes = cumulative["TRUE_CO2_LOSS_KG"].sum() / 1000
total_actual_energy = cumulative["ENERGY_KWH_INTERVAL"].sum()
loss_rate = (
    total_loss / total_actual_energy * 100 if total_actual_energy else 0.0
)

st.markdown(
    f"""
    <div class="status-strip">
      <span class="status-dot"></span>
      <div>
        <strong>Simulated snapshot · {selected_date:%d %B %Y}</strong><br>
        <span>{snapshot["affected_inverters"]} inverter(s) showed at least one
        actionable fault interval across the selected plant scope.</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric(
    "Masked energy loss",
    f"{_format_one(total_loss)} kWh",
    help=(
        "Sum of TRUE_ENERGY_LOSS_KWH from the start of the dataset through "
        "the simulated date — non-fault residuals are masked out."
    ),
)
kpi_2.metric(
    "Carbon impact",
    f"{_format_one(total_co2_tonnes)} tCO₂",
    help="Masked loss through the simulated date × 0.703 kgCO₂/kWh.",
)
kpi_3.metric(
    "Loss rate",
    f"{loss_rate:.2f}%",
    help=(
        "Masked loss through the simulated date divided by actual generated "
        "energy over the same window, not expected energy."
    ),
)
kpi_4.metric(
    "Affected inverters",
    _format_int(snapshot["affected_inverters"]),
    help="Unique inverters with TOTAL_LOSS or PARTIAL_LOSS on the simulated day.",
)

st.markdown("### Action required")
alert_1, alert_2 = st.columns(2)
with alert_1:
    st.error(
        f"**Dispatch now · {snapshot['total_loss_inverters']} inverter(s)**  \n"
        "TOTAL_LOSS: AC power was zero under sufficient irradiation."
    )
with alert_2:
    st.warning(
        f"**Inspect within 24 h · {snapshot['partial_loss_inverters']} inverter(s)**  \n"
        "PARTIAL_LOSS: output fell below 50% of the plant peer average."
    )

chart_col, table_col = st.columns([1.65, 1], gap="large")

with chart_col:
    st.markdown("### Daily fault activity")
    selected_daily = daily_faults.loc[
        daily_faults["PLANT_NAME"].isin(selected_plants)
    ]
    daily_plot = (
        selected_daily.groupby(
            ["DATE", "ANOMALY_CLASS"], observed=True
        )["affected_inverters"]
        .sum()
        .reset_index()
    )

    all_dates = pd.DataFrame(
        {"DATE": pd.date_range(meta.start_date.normalize(), meta.end_date.normalize())}
    )
    all_classes = pd.DataFrame({"ANOMALY_CLASS": list(FAULT_CLASSES)})
    chart_grid = all_dates.merge(all_classes, how="cross")
    daily_plot = chart_grid.merge(
        daily_plot, on=["DATE", "ANOMALY_CLASS"], how="left"
    ).fillna({"affected_inverters": 0})

    timeline = px.area(
        daily_plot,
        x="DATE",
        y="affected_inverters",
        color="ANOMALY_CLASS",
        color_discrete_map=COLOR_MAP,
        category_orders={"ANOMALY_CLASS": list(FAULT_CLASSES)},
        labels={
            "DATE": "",
            "affected_inverters": "Affected inverters",
            "ANOMALY_CLASS": "Fault class",
        },
    )
    timeline.add_vline(
        x=selected_date.timestamp() * 1000,
        line_dash="dot",
        line_color="#132a2f",
        opacity=0.55,
    )
    timeline.update_layout(
        height=355,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        margin=dict(l=0, r=10, t=45, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    timeline.update_yaxes(gridcolor="#e6eeeb", rangemode="tozero")
    timeline.update_xaxes(gridcolor="#eef3f1")
    st.plotly_chart(
        timeline, width="stretch", config={"displayModeBar": False}
    )
    st.caption(
        "Counts are unique affected inverters per day, not confirmed maintenance events."
    )

with table_col:
    st.markdown("### Plant comparison")
    st.caption(f"Cumulative through {selected_date:%d %b %Y}.")
    display_summary = plant_summary[
        [
            "PLANT_NAME",
            "masked_loss_kwh",
            "loss_rate_pct",
            "co2_loss_tonnes",
            "affected_inverters",
        ]
    ].rename(
        columns={
            "PLANT_NAME": "Plant",
            "masked_loss_kwh": "Loss (kWh)",
            "loss_rate_pct": "Loss rate (%)",
            "co2_loss_tonnes": "CO₂ (t)",
            "affected_inverters": "Affected units",
        }
    )
    st.dataframe(
        display_summary.style.format(
            {
                "Loss (kWh)": "{:,.1f}",
                "Loss rate (%)": "{:.2f}",
                "CO₂ (t)": "{:,.1f}",
                "Affected units": "{:,.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=180,
    )

    if len(plant_summary) > 1 and total_loss > 0:
        leading = plant_summary.nlargest(1, "masked_loss_kwh").iloc[0]
        leading_share = leading["masked_loss_kwh"] / total_loss * 100
        st.info(
            f"**{leading['PLANT_NAME']} contributes {leading_share:.1f}%** "
            "of masked loss in the current plant scope."
        )

render_upper_bound_note()
