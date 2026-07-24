"""Gemini-backed AI assistant, grounded in this project's methodology and live KPIs."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from app.data import (
    BASELINE_COMPARISON,
    EMISSION_FACTOR_KG_PER_KWH,
    MODEL_FEATURES,
    MODEL_PERFORMANCE,
    RAW_LOSS_KWH,
    load_master_data,
    overview_aggregates,
)

EXAMPLE_QUESTIONS = (
    "Why was DC power excluded from the model?",
    "Why is Plant 2's R² lower, and is that a problem?",
    "What are the limitations of the reported loss numbers?",
)

MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_PROMPT_TEMPLATE = """\
You are the methodology assistant embedded in a Streamlit app for a Solar PV \
Fault Diagnosis and Carbon Loss research project.
Answer only using the facts below. If a question falls outside this context, \
say you don't have that information rather than guessing. Be concise and \
technically precise; you are talking to operators, reviewers, and researchers.

PROBLEM
A solar plant has many inverters. When one degrades, output drops, but this is \
hard to detect because output naturally varies with weather. The pipeline:
1. Trains an XGBoost model to predict the AC power a plant SHOULD produce \
given weather conditions only.
2. Compares actual vs. expected power to classify each 15-minute interval as \
NORMAL / PARTIAL_LOSS / TOTAL_LOSS.
3. Converts confirmed-fault loss into kWh and tonnes of CO2.

THREE CORE DESIGN DECISIONS
(a) DC power is excluded from the feature set (only {features} are used). \
Including DC power would let the model learn AC ≈ 0.98 × DC, so when an \
inverter's DC input drops from a real fault, predicted AC would drop too and \
the fault would vanish (target leakage).
(b) One GLOBAL model is trained per plant (not one LOCAL model per inverter). \
A local model would fit a degraded inverter's history so closely it would \
learn the degradation as "normal," masking the fault. A global model has no \
such memory, so a drifting inverter shows up as prediction error. This is why \
Plant 2's test R² (0.54) is LOWER than Plant 1's (0.96) — it is correctly \
exposing heavier real degradation, not a worse model.
(c) Masked Loss Aggregation: summing max(expected − actual, 0) across every \
interval accumulates ordinary forecast noise in one direction (Positive Drift \
Bias), inflating raw loss to {raw_loss:,.1f} kWh. Loss is only counted when the \
taxonomy confirms a fault, giving the correct masked total below.

FAULT TAXONOMY
- TOTAL_LOSS: AC power = 0 AND irradiation > 0.2 → dispatch a crew now.
- PARTIAL_LOSS: 0 < AC power < 50% of the plant's peer-average inverter, AND \
irradiation > 0.2 → inspect within 24 hours.
- NORMAL: everything else, including all nighttime intervals.

DATA
136,476 rows, 15-minute intervals, 15 May – 17 June 2020 (34 days), two Indian \
plants, from the public Kaggle "Solar Power Generation Data" dataset. Split by \
day: first 27 days train, last 7 days test (cutoff 11 June 2020). Inverter \
`bvBOhCH3iADSZry` (Plant 1) was deliberately excluded from training so the \
model wouldn't learn its faulty state as normal; it has confirmed TOTAL_LOSS \
events around 7 and 14 June 2020, used as the drill-down case study.

CURRENT HEADLINE NUMBERS (computed live from the loaded dataset)
- Total masked energy loss: {total_loss:,.1f} kWh
- Total carbon impact: {total_co2_t:,.1f} tCO2 (emission factor {emission_factor} kgCO2/kWh, CEA India FY2020-21)
- System-wide loss rate (masked loss / actual generated energy): {loss_rate:.2f}%
- Raw (unmasked) loss — NEVER cite this as the headline number: {raw_loss:,.1f} kWh
By plant:
{plant_lines}

MODEL PERFORMANCE (R² / MAE / RMSE / MAPE)
{perf_lines}
SHAP feature ranking (Plant 1): IRRADIATION > MODULE_TEMPERATURE > HOUR_COS > \
AMBIENT_TEMPERATURE > HOUR_SIN.

EVALUATION VS. UNSUPERVISED BASELINES (physics-informed synthetic fault injection, N=20 seeds)
{baseline_lines}

LIMITATIONS — be upfront about these if asked about validity or trustworthiness
- All headline loss/CO2 numbers are an UPPER-BOUND ESTIMATE, not field-verified \
loss. The global model cannot fully separate a genuinely faulty inverter from \
one that is legitimately different (position, wiring, orientation), and no \
O&M log exists to confirm real faults.
- Partial circularity in evaluation: the TOTAL_LOSS rule (AC power near zero \
under real irradiance) closely mirrors how total-loss faults are synthetically \
injected, so the reported precision of 1.00 is a consistency check, not proof \
of detection power on independent real-world faults.
- No field-verified fault records were available; all evaluation metrics are \
against synthetic faults. Validation on real annotated faults is future work.
"""


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_system_context(frame: pd.DataFrame) -> str:
    plant_totals, _ = overview_aggregates(frame)
    plant_lines = "\n".join(
        f"- {row.PLANT_NAME}: masked loss {row.masked_loss_kwh:,.1f} kWh, "
        f"loss rate {row.loss_rate_pct:.2f}%, CO2 {row.co2_loss_tonnes:,.1f} t, "
        f"{row.affected_inverters} affected inverters"
        for row in plant_totals.itertuples()
    )
    perf_lines = "\n".join(
        f"- {row.plant} {row.split}: R²={row.r2:.4f}, MAE={row.mae_kw:.2f} kW, "
        f"RMSE={row.rmse_kw:.2f} kW, MAPE={row.mape_pct:.2f}%"
        for row in MODEL_PERFORMANCE.itertuples()
    )
    baseline_lines = "\n".join(
        f"- {row.method}: precision={row.precision:.2f}, recall={row.recall:.2f}, "
        f"F1={row.f1:.2f}, kappa={row.kappa:.2f}"
        for row in BASELINE_COMPARISON.itertuples()
    )
    total_loss = float(frame["TRUE_ENERGY_LOSS_KWH"].sum())
    total_co2_t = float(frame["TRUE_CO2_LOSS_KG"].sum() / 1000)
    total_actual = float(frame["ENERGY_KWH_INTERVAL"].sum())
    return SYSTEM_PROMPT_TEMPLATE.format(
        features=", ".join(MODEL_FEATURES),
        raw_loss=RAW_LOSS_KWH,
        total_loss=total_loss,
        total_co2_t=total_co2_t,
        emission_factor=EMISSION_FACTOR_KG_PER_KWH,
        loss_rate=total_loss / total_actual * 100 if total_actual else 0.0,
        plant_lines=plant_lines,
        perf_lines=perf_lines,
        baseline_lines=baseline_lines,
    )


def start_chat(api_key: str, system_context: str):
    client = _client(api_key)
    return client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_context,
            temperature=0.2,
        ),
    )


def ask(chat, question: str) -> str:
    response = chat.send_message(question)
    return response.text


def _answer(question: str) -> str:
    try:
        return ask(st.session_state.assistant_chat, question)
    except Exception as error:
        return f"Something went wrong calling the Gemini API: {error}"


def render_floating_widget() -> None:
    """A chat bubble that floats over every page via fixed-position CSS.

    Two mutually-exclusive keyed containers (``chat_bubble`` / ``chat_panel``)
    are pinned to the viewport corner by the CSS in ``app/ui.py`` — only one
    renders per run, driven by ``st.session_state.chat_open``.
    """

    st.session_state.setdefault("chat_open", False)
    st.session_state.setdefault("assistant_history", [])

    if not st.session_state.chat_open:
        with st.container(key="chat_bubble"):
            if st.button("💬", key="chat_bubble_button", help="Ask the AI assistant"):
                st.session_state.chat_open = True
                st.rerun()
        return

    with st.container(key="chat_panel"):
        with st.container(key="chat_panel_header"):
            header_col, close_col = st.columns([5, 1])
            header_col.markdown("**AI Assistant**")
            with close_col:
                with st.container(key="chat_close"):
                    if st.button("✕", key="chat_close_button", help="Collapse"):
                        st.session_state.chat_open = False
                        st.rerun()

        api_key = get_api_key()
        if not api_key:
            st.caption(
                "No Gemini API key configured. Copy `secrets.toml.example` to "
                "`.streamlit/secrets.toml` and set `GOOGLE_API_KEY`."
            )
            return

        try:
            data = load_master_data()
        except (FileNotFoundError, ValueError) as error:
            st.caption(str(error))
            return

        if "assistant_chat" not in st.session_state:
            st.session_state.assistant_chat = start_chat(
                api_key, build_system_context(data)
            )

        pending_question = None
        message_area = st.container(height=320)
        with message_area:
            if not st.session_state.assistant_history:
                st.caption("Try asking:")
                for example in EXAMPLE_QUESTIONS:
                    if st.button(example, key=f"chat_example_{example}", width="stretch"):
                        pending_question = example
            for role, text in st.session_state.assistant_history:
                with st.chat_message(role):
                    st.write(text)

        typed_question = st.chat_input("Ask a question...", key="floating_chat_input")
        question = pending_question or typed_question

        if question:
            st.session_state.assistant_history.append(("user", question))
            with message_area:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = _answer(question)
                    st.write(answer)
            st.session_state.assistant_history.append(("assistant", answer))
