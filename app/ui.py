"""Visual shell and small presentation helpers."""

from __future__ import annotations

import streamlit as st


def apply_app_shell() -> None:
    st.markdown(
        """
        <style>
        :root {
          --solar-ink: #132a2f;
          --solar-muted: #617478;
          --solar-green: #0d766e;
          --solar-lime: #93c01f;
          --solar-gold: #f4a621;
          --solar-red: #d94a4a;
          --solar-surface: #f4f8f6;
        }

        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at 88% 2%, rgba(147, 192, 31, .11), transparent 24rem),
            #fbfdfc;
        }

        [data-testid="stSidebar"] {
          border-right: 1px solid #dfe9e5;
          background: #f5f9f7;
        }

        [data-testid="stSidebarNav"] {
          padding-top: .25rem;
        }

        .block-container {
          padding-top: 5rem;
          padding-bottom: 4rem;
          max-width: 1440px;
        }

        h1, h2, h3 {
          color: var(--solar-ink);
          letter-spacing: -.02em;
        }

        .eyebrow {
          color: var(--solar-green);
          font-size: .76rem;
          font-weight: 750;
          letter-spacing: .13em;
          margin-bottom: .35rem;
          text-transform: uppercase;
        }

        .page-subtitle {
          color: var(--solar-muted);
          font-size: 1rem;
          margin-top: -.45rem;
          max-width: 850px;
        }

        .sidebar-brand {
          align-items: center;
          display: flex;
          gap: .7rem;
          margin: .2rem 0 1.2rem;
        }

        .sidebar-brand .brand-mark {
          align-items: center;
          background: var(--solar-ink);
          border-radius: 10px;
          color: #fff;
          display: flex;
          font-size: .75rem;
          font-weight: 800;
          height: 36px;
          justify-content: center;
          letter-spacing: .06em;
          width: 36px;
        }

        .sidebar-brand strong,
        .sidebar-brand small {
          display: block;
        }

        .sidebar-brand small {
          color: var(--solar-muted);
          font-size: .72rem;
        }

        [data-testid="stMetric"] {
          background: rgba(255, 255, 255, .88);
          border: 1px solid #e0e9e6;
          border-radius: 14px;
          box-shadow: 0 8px 24px rgba(21, 58, 55, .05);
          min-height: 120px;
          padding: 1rem 1.1rem;
        }

        [data-testid="stMetricLabel"] {
          color: var(--solar-muted);
          font-weight: 650;
        }

        [data-testid="stMetricValue"] {
          color: var(--solar-ink);
        }

        .status-strip {
          align-items: center;
          background: #edf7f3;
          border: 1px solid #d3e9e1;
          border-radius: 13px;
          display: flex;
          gap: .8rem;
          margin: .75rem 0 1.25rem;
          padding: .85rem 1rem;
        }

        .status-dot {
          background: var(--solar-green);
          border-radius: 50%;
          box-shadow: 0 0 0 5px rgba(13, 118, 110, .12);
          flex: 0 0 auto;
          height: 9px;
          width: 9px;
        }

        .method-note {
          background: #fff8e8;
          border-left: 4px solid var(--solar-gold);
          border-radius: 4px 12px 12px 4px;
          color: #594619;
          margin-top: 1.25rem;
          padding: .9rem 1rem;
        }

        div[data-testid="stDataFrame"] {
          border: 1px solid #e0e9e6;
          border-radius: 12px;
          overflow: hidden;
        }

        .step-flow {
          display: grid;
          gap: .6rem;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          margin: 1rem 0 1.5rem;
        }

        .step-card {
          background: rgba(255, 255, 255, .88);
          border: 1px solid #e0e9e6;
          border-radius: 12px;
          padding: .85rem .9rem;
        }

        .step-card .step-number {
          background: var(--solar-green);
          border-radius: 50%;
          color: #fff;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: .74rem;
          font-weight: 800;
          height: 22px;
          width: 22px;
          margin-bottom: .5rem;
        }

        .step-card .step-title {
          color: var(--solar-ink);
          display: block;
          font-weight: 700;
          margin-bottom: .2rem;
        }

        .step-card .step-body {
          color: var(--solar-muted);
          font-size: .84rem;
          line-height: 1.35;
        }

        .st-key-chat_bubble {
          position: fixed !important;
          bottom: 22px;
          right: 22px;
          width: fit-content !important;
          z-index: 999999;
        }

        .st-key-chat_bubble button {
          background: var(--solar-green) !important;
          border: none !important;
          border-radius: 50% !important;
          box-shadow: 0 8px 24px rgba(21, 58, 55, .32) !important;
          height: 56px !important;
          width: 56px !important;
          font-size: 1.4rem !important;
          padding: 0 !important;
        }

        .st-key-chat_panel {
          position: fixed !important;
          bottom: 22px;
          right: 22px;
          width: 380px !important;
          max-width: calc(100vw - 32px);
          z-index: 999999;
          background: #fbfdfc;
          border: 1px solid #dfe9e5;
          border-radius: 18px;
          box-shadow: 0 16px 48px rgba(21, 58, 55, .28);
          padding: .85rem .95rem 1rem;
        }

        .st-key-chat_panel [data-testid="stChatInput"] {
          margin-top: .4rem;
        }

        .st-key-chat_panel_header {
          align-items: center;
          display: flex;
          justify-content: space-between;
          margin-bottom: .3rem;
        }

        .st-key-chat_panel_header strong {
          color: var(--solar-ink);
        }

        .st-key-chat_close button {
          background: transparent !important;
          border: none !important;
          color: var(--solar-muted) !important;
          height: 28px !important;
          min-height: 28px !important;
          padding: 0 !important;
          width: 28px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_step_flow(steps: list[tuple[str, str]]) -> None:
    """Render a numbered pipeline as a wrapping row of cards.

    ``steps`` is a list of ``(title, body)`` tuples; enumeration supplies the
    step number so callers don't need to track it themselves.
    """

    # Built as single-line HTML: st.markdown runs CommonMark first, and an
    # indented line following a blank line is read as a code block, which
    # silently swallows every card after the first if this is pretty-printed.
    cards = "".join(
        f'<div class="step-card"><span class="step-number">{index}</span>'
        f'<span class="step-title">{title}</span>'
        f'<div class="step-body">{body}</div></div>'
        for index, (title, body) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="step-flow">{cards}</div>', unsafe_allow_html=True)


def render_page_header(
    eyebrow: str, title: str, subtitle: str
) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_upper_bound_note() -> None:
    st.markdown(
        """
        <div class="method-note">
          <strong>Interpretation note.</strong>
          Masked loss is an <strong>upper-bound estimate</strong>, not a
          field-verified loss. The global model cannot fully separate a faulty
          inverter from legitimate unit-level differences, and no O&amp;M log
          is available for confirmation.
        </div>
        """,
        unsafe_allow_html=True,
    )
