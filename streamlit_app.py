from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="FairEval Dashboard",
    page_icon="⚖️",
    layout="wide",
)


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def safe_read_text(path: Path) -> Optional[str]:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def safe_float_text(path: Path) -> Optional[float]:
    text = safe_read_text(path)
    if text is None or text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            padding: 0.9rem 0.2rem 0.4rem 0.2rem;
            border-radius: 12px;
        ">
            <div style="font-size: 0.95rem; opacity: 0.9;">{label}</div>
            <div style="font-size: 2rem; font-weight: 700; line-height: 1.2;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_metric(x, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def show_figure(path: Path, caption: str):
    if path.exists():
        try:
            img = Image.open(path)
            st.image(img, width="stretch")
            st.caption(caption)
        except Exception as e:
            st.warning(f"Could not load figure {path.name}: {e}")
    else:
        st.info(f"Figure not found: {path.name}")


# -----------------------------
# Theme / appearance CSS
# -----------------------------
def inject_theme_css(mode: str):
    if mode == "Light":
        bg = "#f6f8fc"
        panel = "#ffffff"
        text = "#111827"
        subtext = "#374151"
        border = "#d1d5db"
        sidebar_bg = "#eef2f7"
        sidebar_box = "#ffffff"
        hover = "#e5e7eb"
        selected = "#dbeafe"
    else:
        # Dark + Auto both use the same stable custom theme inside app
        bg = "#060816"
        panel = "#0b1020"
        text = "#f5f7fb"
        subtext = "#cbd5e1"
        border = "#374151"
        sidebar_bg = "linear-gradient(180deg, #1f2230 0%, #0b1020 100%)"
        sidebar_box = "#0b1020"
        hover = "#1a2238"
        selected = "#24304d"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 24%),
                radial-gradient(circle at top right, rgba(168,85,247,0.10), transparent 24%),
                {bg};
            color: {text};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        h1, h2, h3, h4, h5, h6, p, li, label, div, span {{
            color: {text};
        }}

        section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {{
            color: {text} !important;
        }}

        /* Closed selectbox */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: {sidebar_box} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
            min-height: 42px !important;
        }}

        /* Text inside closed selectbox */
        section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
            color: {text} !important;
        }}

        /* Dropdown popup */
        div[role="listbox"] {{
            background-color: {sidebar_box} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
            color: {text} !important;
        }}

        /* Each option */
        div[role="option"] {{
            background-color: {sidebar_box} !important;
            color: {text} !important;
        }}

        /* Hovered option */
        div[role="option"]:hover {{
            background-color: {hover} !important;
            color: {text} !important;
        }}

        /* Selected / focused option */
        div[aria-selected="true"] {{
            background-color: {selected} !important;
            color: {text} !important;
        }}

        /* Arrow icon */
        section[data-testid="stSidebar"] svg {{
            fill: {text} !important;
        }}

        /* Dataframe / table container */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* Metric spacing */
        [data-testid="stMetric"] {{
            background: transparent;
            border-radius: 12px;
            padding: 0.25rem 0.1rem;
        }}

        /* Horizontal rule */
        hr {{
            border: none;
            border-top: 1px solid {border};
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        /* Code blocks */
        code {{
            color: {text};
            background-color: rgba(127,127,127,0.12);
            padding: 0.12rem 0.35rem;
            border-radius: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## Dashboard Settings")

appearance = st.sidebar.selectbox(
    "Appearance",
    ["Auto", "Dark", "Light"],
    index=0,
)

inject_theme_css(appearance)

results_df = safe_read_csv(OUTPUTS_DIR / "results.csv")
tool_df = safe_read_csv(OUTPUTS_DIR / "tool_comparison.csv")
agreement_df = safe_read_csv(OUTPUTS_DIR / "agreement_kendall_tau.csv")
judge_summary_df = safe_read_csv(OUTPUTS_DIR / "fairness_judge_summary.csv")
crows_df = safe_read_csv(OUTPUTS_DIR / "crows_pairs_llm_eval.csv")
bbq_df = safe_read_csv(OUTPUTS_DIR / "bbq_llm_eval.csv")
alpha_value = safe_float_text(OUTPUTS_DIR / "agreement_krippendorff_alpha.txt")

if results_df is not None and not results_df.empty:
    dataset_options = ["All"] + sorted(results_df["Dataset"].dropna().astype(str).unique().tolist())
    model_options = ["All"] + sorted(results_df["Model"].dropna().astype(str).unique().tolist())
else:
    dataset_options = ["All"]
    model_options = ["All"]

st.sidebar.markdown("## Result Filters")
dataset_filter = st.sidebar.selectbox("Dataset filter", dataset_options, index=0)
model_filter = st.sidebar.selectbox("Model filter", model_options, index=0)


# -----------------------------
# Apply filters
# -----------------------------
filtered_results = results_df.copy() if results_df is not None else None

if filtered_results is not None and not filtered_results.empty:
    if dataset_filter != "All":
        filtered_results = filtered_results[filtered_results["Dataset"].astype(str) == dataset_filter]
    if model_filter != "All":
        filtered_results = filtered_results[filtered_results["Model"].astype(str) == model_filter]


# -----------------------------
# Header
# -----------------------------
st.title("FairEval Dashboard")
st.caption(
    "Integrated fairness benchmarking, agreement analysis, LLM bias evaluation, and dashboard presentation for Milestone 6."
)


# -----------------------------
# Executive summary
# -----------------------------
st.header("Executive Summary")
st.markdown(
    """
- **FairEval benchmarks multiple machine learning models** across Adult, COMPAS, and German Credit.
- The project **compares multiple fairness metrics** and shows that fairness conclusions can differ depending on the metric used.
- The framework also includes **optional LLM-based bias evaluation** using CrowS-Pairs, BBQ, and a fairness-judge experiment.
"""
)

st.header("Execution Modes")
st.markdown(
    """
- **Core mode:** fully reproducible local package with tabular datasets, fairness metrics, agreement analysis, generated figures, and dashboard outputs.
- **Extended mode:** optional API-backed LLM evaluation for CrowS-Pairs, BBQ, and fairness-judge style experiments.
"""
)


# -----------------------------
# Core benchmark summary
# -----------------------------
st.header("Core Benchmark Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Krippendorff's alpha", format_metric(alpha_value, 6) if alpha_value is not None else "N/A")
with col2:
    metric_card(
        "Datasets",
        str(results_df["Dataset"].nunique()) if results_df is not None and not results_df.empty else "0",
    )
with col3:
    metric_card(
        "Models",
        str(results_df["Model"].nunique()) if results_df is not None and not results_df.empty else "0",
    )
with col4:
    metric_card("Fairness metrics", "3")


# -----------------------------
# Optional LLM/API summary
# -----------------------------
st.header("Optional LLM/API Extension Summary")
c1, c2, c3, c4 = st.columns(4)

crows_acc = None
if crows_df is not None and not crows_df.empty and "correct" in crows_df.columns:
    crows_acc = float(crows_df["correct"].mean())

bbq_acc = None
if bbq_df is not None and not bbq_df.empty and "correct" in bbq_df.columns:
    bbq_acc = float(bbq_df["correct"].mean())

judge_agreement = None
if judge_summary_df is not None and not judge_summary_df.empty and "benchmark_judge_agreement" in judge_summary_df.columns:
    judge_agreement = float(judge_summary_df.iloc[0]["benchmark_judge_agreement"])

llm_mode = "Not available"
if crows_df is not None and not crows_df.empty and "mode" in crows_df.columns:
    llm_mode = str(crows_df.iloc[0]["mode"])

with c1:
    metric_card("CrowS-Pairs accuracy", format_metric(crows_acc))
with c2:
    metric_card("BBQ accuracy", format_metric(bbq_acc))
with c3:
    metric_card("Judge agreement", format_metric(judge_agreement))
with c4:
    metric_card("LLM mode", llm_mode)


# -----------------------------
# Tabular results
# -----------------------------
st.header("Tabular Benchmark Results")
if filtered_results is not None and not filtered_results.empty:
    st.dataframe(filtered_results, width="stretch", hide_index=True)
else:
    st.info("No benchmark results available yet.")


# -----------------------------
# Agreement + Judge summary
# -----------------------------
left, right = st.columns(2)

with left:
    st.header("Agreement Matrix")
    if agreement_df is not None and not agreement_df.empty:
        st.dataframe(agreement_df, width="stretch", hide_index=True)
        st.markdown(
            "**Interpretation:** High positive Kendall τ values indicate that two metrics rank model outcomes similarly. "
            "Lower or negative values indicate disagreement, which supports the main idea that fairness conclusions can depend on the metric chosen."
        )
    else:
        st.info("Agreement matrix not available.")

with right:
    st.header("Fairness-Judge Evaluation Summary")
    if judge_summary_df is not None and not judge_summary_df.empty:
        st.dataframe(judge_summary_df, width="stretch", hide_index=True)
        st.markdown(
            "**Interpretation:** The fairness-judge experiment is intended as a complementary evaluation. "
            "Limited agreement with benchmark labels suggests that natural-language judging may differ from formal benchmark expectations."
        )
    else:
        st.info("Fairness-judge summary not available.")


# -----------------------------
# Tool comparison
# -----------------------------
st.header("Feature Comparison with Existing Fairness Toolkits")
if tool_df is not None and not tool_df.empty:
    st.dataframe(tool_df, width="stretch", hide_index=True)
    st.markdown(
        "**Takeaway:** FairEval combines classical ML fairness benchmarking, metric agreement analysis, optional LLM evaluation, "
        "a fairness-judge experiment, and a reproducible dashboard workflow in one framework."
    )
else:
    st.info("Tool comparison table not available.")


# -----------------------------
# Figures
# -----------------------------
st.header("Generated Figures")

fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    show_figure(
        FIGURES_DIR / "figure1_accuracy_by_model_dataset.png",
        "Figure 1 compares predictive accuracy across models and datasets.",
    )
with fig_col2:
    show_figure(
        FIGURES_DIR / "figure2_demographic_parity_by_model_dataset.png",
        "Figure 2 compares demographic parity differences across models and datasets.",
    )

fig_col3, fig_col4 = st.columns(2)
with fig_col3:
    show_figure(
        FIGURES_DIR / "figure3_equalized_odds_vs_predictive_parity.png",
        "Figure 3 shows how equalized odds and predictive parity can give different fairness rankings.",
    )
with fig_col4:
    show_figure(
        FIGURES_DIR / "figure4_kendall_tau_heatmap.png",
        "Figure 4 summarizes agreement strength among evaluation metrics using Kendall τ.",
    )


# -----------------------------
# Reproducibility section
# -----------------------------
st.header("Reproducibility")
st.markdown("This project is organized as a reproducible local package.")

st.markdown("**Core commands**")
st.markdown(
    """
- `python main.py`
- `python scripts/generate_figures.py`
- `PYTHONPATH=. python scripts/run_crows_pairs.py`
- `python scripts/run_bbq.py`
- `python scripts/run_fairness_judge.py`
- `python scripts/build_dashboard.py`
"""
)

st.markdown("**Notes**")
st.markdown(
    """
- Core tabular benchmarking works locally from packaged datasets and scripts.
- LLM/API-backed evaluation is optional and environment-variable controlled.
- Outputs are saved into the `outputs/`, `figures/`, and `dashboard/` folders.
"""
)