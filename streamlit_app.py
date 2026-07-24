"""Entry point for the Solar PV Fault Diagnosis Streamlit application."""

from __future__ import annotations

import streamlit as st

from app.assistant import render_floating_widget
from app.ui import apply_app_shell

st.set_page_config(
    page_title="Solar PV Operations",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_shell()

navigation = st.navigation(
    {
        "OPERATIONS": [
            st.Page(
                "app_pages/operations_overview.py",
                title="Operations Overview",
                icon=":material/space_dashboard:",
                default=True,
            ),
            st.Page(
                "app_pages/alert_queue.py",
                title="Alert Queue",
                icon=":material/notification_important:",
            ),
            st.Page(
                "app_pages/inverter_detail.py",
                title="Inverter Detail",
                icon=":material/electric_meter:",
            ),
            st.Page(
                "app_pages/loss_carbon.py",
                title="Loss & Carbon Impact",
                icon=":material/co2:",
            ),
        ],
        "METHODOLOGY": [
            st.Page(
                "app_pages/how_it_works.py",
                title="How It Works",
                icon=":material/account_tree:",
            ),
            st.Page(
                "app_pages/model_performance.py",
                title="Model & Performance",
                icon=":material/model_training:",
            ),
            st.Page(
                "app_pages/evaluation.py",
                title="Evaluation & Baselines",
                icon=":material/analytics:",
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <span class="brand-mark">SP</span>
          <div>
            <strong>Solar Pulse</strong>
            <small>PV fault intelligence</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

navigation.run()

render_floating_widget()
