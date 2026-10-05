"""Scientific environmental colour palette and CSS injection."""
from __future__ import annotations

import streamlit as st

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "navy":    "#0d1b2a",
    "teal":    "#0a9396",
    "teal_lt": "#94d2bd",
    "blue":    "#4895ef",
    "red":     "#e63946",
    "amber":   "#f4a261",
    "green":   "#2dc653",
    "gray":    "#8d99ae",
    "violet":  "#7b2d8b",
    "bg":      "#f8f9fa",
    "card":    "#ffffff",
    "border":  "#dee2e6",
    "text":    "#1a1a2e",
    "muted":   "#6c757d",
}

# Plotly series colours
SERIES_COLORS = [
    COLORS["blue"], COLORS["red"], COLORS["green"],
    COLORS["amber"], COLORS["violet"], COLORS["teal"],
    "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3",
]

ANOMALY_COLOR = COLORS["red"]
IMPUTED_COLOR = COLORS["green"]
MISSING_COLOR = COLORS["gray"]
OBSERVED_COLOR = COLORS["blue"]


def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── Global ─────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: {COLORS['bg']};
        color: {COLORS['text']};
    }}
    /* ── Metric cards ──────────────────────────────── */
    .metric-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow .2s;
    }}
    .metric-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.10); }}
    .metric-value {{ font-size: 2rem; font-weight: 700; color: {COLORS['navy']}; line-height: 1.1; }}
    .metric-label {{ font-size: 0.78rem; font-weight: 600; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }}
    .metric-delta-pos {{ font-size: 0.82rem; color: {COLORS['green']}; }}
    .metric-delta-neg {{ font-size: 0.82rem; color: {COLORS['red']}; }}
    /* ── Status badges ─────────────────────────────── */
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }}
    .badge-ok      {{ background: #d4edda; color: #155724; }}
    .badge-warn    {{ background: #fff3cd; color: #856404; }}
    .badge-error   {{ background: #f8d7da; color: #721c24; }}
    .badge-info    {{ background: #cce5ff; color: #004085; }}
    .badge-skip    {{ background: #e2e3e5; color: #383d41; }}
    /* ── Section headers ────────────────────────────── */
    .section-header {{ border-bottom: 2px solid {COLORS['teal']}; padding-bottom: 6px; margin-bottom: 18px; color: {COLORS['navy']}; font-weight: 700; font-size: 1.15rem; }}
    /* ── Info box ───────────────────────────────────── */
    .info-box {{ background: #e8f4f8; border-left: 4px solid {COLORS['teal']}; border-radius: 6px; padding: 12px 16px; margin: 10px 0; font-size: 0.9rem; }}
    .warn-box {{ background: #fff8e1; border-left: 4px solid {COLORS['amber']}; border-radius: 6px; padding: 12px 16px; margin: 10px 0; font-size: 0.9rem; }}
    .danger-box {{ background: #fdecea; border-left: 4px solid {COLORS['red']}; border-radius: 6px; padding: 12px 16px; margin: 10px 0; font-size: 0.9rem; }}
    /* ── Sidebar ────────────────────────────────────── */
    [data-testid="stSidebar"] {{ background: {COLORS['navy']}; }}
    [data-testid="stSidebar"] * {{ color: #e0e0e0 !important; }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}
    /* ── Plotly charts ──────────────────────────────── */
    .js-plotly-plot .plotly {{ border-radius: 8px; }}
    /* ── DataFrames ─────────────────────────────────── */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    /* ── Buttons ────────────────────────────────────── */
    .stButton > button {{
        background: {COLORS['teal']};
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
        transition: background .2s;
    }}
    .stButton > button:hover {{ background: {COLORS['teal_lt']}; color: {COLORS['navy']}; }}
    /* ── Empty state ────────────────────────────────── */
    .empty-state {{ text-align: center; padding: 48px 24px; color: {COLORS['muted']}; }}
    .empty-state-icon {{ font-size: 3rem; margin-bottom: 12px; }}
    .empty-state-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }}
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", color: str = COLORS["teal"],
                tooltip: str = "", icon: str = "") -> str:
    delta_html = ""
    if delta:
        cls = "metric-delta-pos" if not delta.startswith("-") else "metric-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    return f"""
    <div class="metric-card" title="{tooltip}">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def info_box(text: str, kind: str = "info") -> None:
    cls = {"info": "info-box", "warn": "warn-box", "danger": "danger-box"}.get(kind, "info-box")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

