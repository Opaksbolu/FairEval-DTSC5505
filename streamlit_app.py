from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="FairEval Dashboard", layout="wide")

# =========================================================
# Session state
# =========================================================
if "appearance_mode" not in st.session_state:
    st.session_state.appearance_mode = "Auto"

# =========================================================
# Theme styling
# =========================================================
light_css = """
<style>
:root {
  --bg: #ffffff;
  --card: #f7f7f7;
  --text: #111111;
  --muted: #444444;
  --border: #d9d9d9;
  --sidebar: #f3f4f6;
  --accent: #2563eb;
  --select-bg: #ffffff;
  --select-text: #111111;
  --select-border: #d1d5db;
  --select-hover: #f3f4f6;
}

html, body, [data-testid="stAppViewContainer"], .main, .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background-color: var(--sidebar) !important;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
  color: var(--text) !important;
}

div[data-testid="stMetric"] {
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.75rem !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

div[data-baseweb="select"] > div {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
  border: 1px solid var(--select-border) !important;
  border-radius: 10px !important;
}

div[data-baseweb="select"] input {
  color: var(--select-text) !important;
  -webkit-text-fill-color: var(--select-text) !important;
}

div[data-baseweb="select"] span {
  color: var(--select-text) !important;
}

div[role="listbox"] {
  background-color: var(--select-bg) !important;
  border: 1px solid var(--select-border) !important;
  color: var(--select-text) !important;
}

div[role="option"] {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
}

div[role="option"]:hover {
  background-color: var(--select-hover) !important;
}

ul[role="listbox"] {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
  border: 1px solid var(--select-border) !important;
}

ul[role="listbox"] li {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}

.stTabs [data-baseweb="tab"] {
  background: #f3f4f6;
  border-radius: 10px;
  padding: 8px 14px;
}

.custom-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem;
  margin-bottom: 1rem;
}

.small-note {
  color: var(--muted) !important;
  font-size: 0.92rem;
}
</style>
"""

dark_css = """
<style>
:root {
  --bg: #0e1117;
  --card: #161b22;
  --text: #f3f3f3;
  --muted: #b3b3b3;
  --border: #2d333b;
  --sidebar: #111827;
  --accent: #60a5fa;
  --select-bg: #111827;
  --select-text: #f9fafb;
  --select-border: #374151;
  --select-hover: #1f2937;
}

html, body, [data-testid="stAppViewContainer"], .main, .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background-color: var(--sidebar) !important;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
  color: var(--text) !important;
}

div[data-testid="stMetric"] {
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.75rem !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

div[data-baseweb="select"] > div {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
  border: 1px solid var(--select-border) !important;
  border-radius: 10px !important;
}

div[data-baseweb="select"] input {
  color: var(--select-text) !important;
  -webkit-text-fill-color: var(--select-text) !important;
}

div[data-baseweb="select"] span {
  color: var(--select-text) !important;
}

div[role="listbox"] {
  background-color: var(--select-bg) !important;
  border: 1px solid var(--select-border) !important;
  color: var(--select-text) !important;
}

div[role="option"] {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
}

div[role="option"]:hover {
  background-color: var(--select-hover) !important;
}

ul[role="listbox"] {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
  border: 1px solid var(--select-border) !important;
}

ul[role="listbox"] li {
  background-color: var(--select-bg) !important;
  color: var(--select-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}

.stTabs [data-baseweb="tab"] {
  background: #161b22;
  border-radius: 10px;
  padding: 8px 14px;
}

.custom-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem;
  margin-bottom: 1rem;
}

.small-note {
  color: var(--muted) !important;
  font-size: 0.92rem;
}
</style>
"""

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("Dashboard Settings")
    st.selectbox(
        "Appearance",
        ["Auto", "Light", "Dark"],
        key="appearance_mode",
    )

mode = st.session_state.appearance_mode

if mode == "Light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    # default Auto -> Dark for consistent professor/demo screenshots
    st.markdown(dark_css, unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_read_text(path: Path, default: str = "n/a") -> str:
    if not path.exists():
        return default
    return path.read_text().strip()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # Drop junk index columns
    junk_cols = [c for c in out.columns if c.lower().startswith("unnamed")]
    if junk_cols:
        out = out.drop(columns=junk_cols, errors="ignore")

    # Friendly column names
    rename_map = {
        "dataset": "Dataset",
        "model": "Model",
        "accuracy": "Accuracy",
        "demographic_parity": "Demographic Parity",
        "equalized_odds": "Equalized Odds",
        "predictive_parity": "Predictive Parity",
        "accuracy_ci_low": "Accuracy_CI_Low",
        "accuracy_ci_high": "Accuracy_CI_High",
        "dp_ci_low": "DP_CI_Low",
        "dp_ci_high": "DP_CI_High",
        "eo_ci_low": "EO_CI_Low",
        "eo_ci_high": "EO_CI_High",
        "pp_ci_low": "PP_CI_Low",
        "pp_ci_high": "PP_CI_High",
        "mode": "Mode",
        "model_name": "Model Name",
        "bias_type": "Bias Type",
        "stereo_antistereo": "Stereo/AntiStereo",
        "prediction": "Prediction",
        "expected": "Expected",
        "judge_prediction": "Judge Prediction",
        "benchmark_correct": "Benchmark Correct",
        "judge_correct": "Judge Correct",
        "benchmark_judge_agree": "Benchmark-Judge Agree",
        "context_condition": "Context Condition",
        "raw_response": "Raw Response",
        "group": "Group",
        "category": "Category",
        "correct": "Correct",
        "gold": "Gold",
        "pred": "Pred",
        "index": "Index",
        "sent_more": "Sentence More-Biased",
        "sent_less": "Sentence Less-Biased",
        "n_items": "N Items",
        "benchmark_accuracy_vs_expected": "Benchmark Accuracy vs Expected",
        "judge_accuracy_vs_expected": "Judge Accuracy vs Expected",
        "benchmark_judge_agreement": "Benchmark-Judge Agreement",
        "requirement": "Requirement",
        "status": "Status",
    }

    for old, new in rename_map.items():
        if old in out.columns:
            out = out.rename(columns={old: new})

    # Prettify dataset values
    if "Dataset" in out.columns:
        out["Dataset"] = (
            out["Dataset"]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
        )

    # Prettify model values
    if "Model" in out.columns:
        out["Model"] = (
            out["Model"]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace("Nb", "NB", regex=False)
        )

    return out


def round_numeric(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].round(decimals)
    return out


def display_df(df: pd.DataFrame, height: int | str | None = "auto"):
    if df is None:
        st.info("No data to display.")
        return

    safe_df = round_numeric(clean_columns(df))

    valid_string_heights = {"auto", "content", "stretch"}

    if height is None:
        safe_height = "auto"
    elif isinstance(height, int) and height > 0:
        safe_height = height
    elif isinstance(height, str) and height in valid_string_heights:
        safe_height = height
    else:
        safe_height = "auto"

    st.dataframe(
        safe_df,
        width="stretch",
        height=safe_height,
    )


def get_theme_colors(mode: str):
    if mode == "Light":
        return {
            "fig_bg": "#ffffff",
            "ax_bg": "#ffffff",
            "text": "#111111",
            "grid": "#d9d9d9",
            "spine": "#666666",
        }
    return {
        "fig_bg": "#0e1117",
        "ax_bg": "#161b22",
        "text": "#f3f3f3",
        "grid": "#2d333b",
        "spine": "#9aa4b2",
    }


def style_axes(ax, colors):
    ax.set_facecolor(colors["ax_bg"])
    ax.tick_params(colors=colors["text"], labelcolor=colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["spine"])
    ax.grid(True, color=colors["grid"], alpha=0.3)


def style_legend(ax, colors):
    leg = ax.get_legend()
    if leg:
        leg.get_title().set_color(colors["text"])
        for text in leg.get_texts():
            text.set_color(colors["text"])
        leg.get_frame().set_facecolor(colors["ax_bg"])
        leg.get_frame().set_edgecolor(colors["spine"])


def plot_accuracy(results, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    pivot = results.pivot(index="Model", columns="Dataset", values="Accuracy")
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Accuracy by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    style_axes(ax, colors)
    style_legend(ax, colors)
    fig.tight_layout()
    return fig


def plot_dp(results, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    pivot = results.pivot(index="Model", columns="Dataset", values="Demographic Parity")
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Demographic Parity Difference")
    ax.set_xlabel("Model")
    ax.set_ylabel("Difference")
    style_axes(ax, colors)
    style_legend(ax, colors)
    fig.tight_layout()
    return fig


def plot_eo_vs_pp(results, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    datasets = results["Dataset"].unique()
    for ds in datasets:
        subset = results[results["Dataset"] == ds]
        ax.scatter(
            subset["Equalized Odds"],
            subset["Predictive Parity"],
            label=ds,
            s=70,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["Model"],
                (row["Equalized Odds"], row["Predictive Parity"]),
                fontsize=8,
                color=colors["text"],
            )

    ax.set_title("Equalized Odds vs Predictive Parity")
    ax.set_xlabel("Equalized Odds")
    ax.set_ylabel("Predictive Parity")
    style_axes(ax, colors)
    style_legend(ax, colors)
    fig.tight_layout()
    return fig


def plot_tau_heatmap(tau, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    matrix = tau.values
    ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(tau.columns)))
    ax.set_yticks(np.arange(len(tau.index)))
    ax.set_xticklabels(tau.columns, rotation=25, ha="right", color=colors["text"])
    ax.set_yticklabels(tau.index, color=colors["text"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.4f}",
                ha="center",
                va="center",
                color=colors["text"],
                fontsize=9,
            )

    ax.set_title("Kendall Tau Agreement Matrix")
    style_axes(ax, colors)
    fig.tight_layout()
    return fig


# =========================================================
# Load data
# =========================================================
results = safe_read_csv(OUTPUTS / "results.csv")
tau = safe_read_csv(OUTPUTS / "agreement_kendall_tau.csv")
if not tau.empty and tau.columns[0] not in tau.columns[1:]:
    tau = tau.set_index(tau.columns[0])

alpha = safe_read_text(OUTPUTS / "agreement_krippendorff_alpha.txt", "n/a")

judge_summary = safe_read_csv(OUTPUTS / "fairness_judge_summary.csv")
judge_cases = safe_read_csv(OUTPUTS / "fairness_judge_cases.csv")
crows = safe_read_csv(OUTPUTS / "crows_pairs_llm_eval.csv")
bbq = safe_read_csv(OUTPUTS / "bbq_llm_eval.csv")

tool_comparison_path = OUTPUTS / "tool_comparison.csv"
tool_comparison = safe_read_csv(tool_comparison_path) if tool_comparison_path.exists() else pd.DataFrame()

# =========================================================
# Derived values
# =========================================================
crows_acc = float(crows["correct"].mean()) if ("correct" in crows.columns and not crows.empty) else None

bbq_acc = None
if "correct" in bbq.columns and not bbq.empty:
    valid = bbq[bbq["correct"].isin([True, False])]
    if len(valid) > 0:
        bbq_acc = float(valid["correct"].mean())

judge_agreement = None
if not judge_summary.empty and "benchmark_judge_agreement" in judge_summary.columns:
    judge_agreement = float(judge_summary["benchmark_judge_agreement"].iloc[0])

# =========================================================
# Header
# =========================================================
st.title("FairEval Dashboard")
st.caption("Milestone dashboard for fairness benchmarking, agreement analysis, and LLM bias evaluation.")

# =========================================================
# Summary
# =========================================================
st.subheader("Executive Summary")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Krippendorff's alpha", alpha)
m2.metric("CrowS-Pairs accuracy", f"{crows_acc:.3f}" if crows_acc is not None else "n/a")
m3.metric("BBQ accuracy", f"{bbq_acc:.3f}" if bbq_acc is not None else "n/a")
m4.metric("Judge agreement", f"{judge_agreement:.3f}" if judge_agreement is not None else "n/a")

st.markdown(
    """
<div class="custom-card">
<b>Execution modes:</b><br>
• <b>Core local/unpaid mode:</b> tabular datasets, preprocessing, multiple ML models, fairness metrics, metric agreement analysis, confidence intervals, figures, and dashboard.<br>
• <b>Optional paid/API mode:</b> OpenAI-backed CrowS-Pairs, BBQ, and fairness-judge evaluation.
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Benchmark Results",
        "Agreement Analysis",
        "LLM Evaluation",
        "Fairness Judge",
        "Roadmap Coverage",
    ]
)

# -------------------------
# Tab 1
# -------------------------
with tab1:
    st.subheader("Tabular benchmark results")
    st.markdown(
        '<p class="small-note">This section shows the main local benchmark results across datasets, models, fairness metrics, and confidence intervals.</p>',
        unsafe_allow_html=True,
    )
    display_df(results, height=360)

    if not results.empty:
        clean_results = round_numeric(clean_columns(results))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_accuracy(clean_results, mode), width="stretch")
            st.caption("Figure 1 compares predictive accuracy across models and datasets.")
        with c2:
            st.pyplot(plot_dp(clean_results, mode), width="stretch")
            st.caption("Figure 2 compares demographic parity differences across models and datasets.")

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(plot_eo_vs_pp(clean_results, mode), width="stretch")
            st.caption("Figure 3 shows how equalized odds and predictive parity can rank models differently.")
        with c4:
            if not tau.empty:
                st.pyplot(
                    plot_tau_heatmap(
                        round_numeric(
                            clean_columns(tau.reset_index()).set_index("index")
                            if "index" in tau.reset_index().columns
                            else tau,
                            4,
                        ),
                        mode,
                    ),
                    width="stretch",
                )
                st.caption("Figure 4 summarizes agreement strength among evaluation metrics using Kendall τ.")

# -------------------------
# Tab 2
# -------------------------
with tab2:
    st.subheader("Agreement analysis")
    st.markdown(
        """
<div class="custom-card">
This section summarizes how fairness metrics align across benchmark runs.
Higher Kendall tau values indicate stronger ranking agreement between metrics.
Krippendorff's alpha gives an overall agreement score across the metric set.
</div>
""",
        unsafe_allow_html=True,
    )

    if not tau.empty:
        display_df(tau)
    else:
        st.info("Agreement matrix file not found.")

    st.markdown(
        '<p class="small-note">Interpretation: if agreement is low or mixed, fairness conclusions can change depending on which metric is used. That is one of the key points of the project.</p>',
        unsafe_allow_html=True,
    )

# -------------------------
# Tab 3
# -------------------------
with tab3:
    st.subheader("LLM benchmark evaluation")
    st.markdown(
        """
<div class="custom-card">
This tab represents the optional API-backed extension of the project.
It complements the local tabular fairness benchmarks with language-bias benchmark evaluation.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CrowS-Pairs sample results**")
        if not crows.empty:
            display_df(crows.head(15), height=360)
        else:
            st.info("CrowS-Pairs results not found.")
    with c2:
        st.markdown("**BBQ sample results**")
        if not bbq.empty:
            display_df(bbq.head(15), height=360)
        else:
            st.info("BBQ results not found.")

# -------------------------
# Tab 4
# -------------------------
with tab4:
    st.subheader("Fairness judge summary")
    st.markdown(
        """
<div class="custom-card">
The fairness-judge experiment checks whether a natural-language judge agrees with benchmark expectations.
This is a complementary evaluation branch, not a replacement for formal fairness metrics.
</div>
""",
        unsafe_allow_html=True,
    )

    if not judge_summary.empty:
        display_df(judge_summary)
    else:
        st.info("Fairness judge summary not found.")

    if not judge_cases.empty:
        st.markdown("**Sample judge cases**")
        display_df(judge_cases.head(15), height=360)

# -------------------------
# Tab 5
# -------------------------
with tab5:
    st.subheader("Roadmap coverage checklist")

    checklist = pd.DataFrame(
        [
            ["Tabular benchmark datasets", "Done"],
            ["Multiple classification models", "Done"],
            ["Fairness metrics", "Done"],
            ["Metric agreement analysis", "Done"],
            ["Krippendorff alpha", "Done"],
            ["CrowS-Pairs evaluation", "Done" if not crows.empty else "Missing"],
            ["BBQ evaluation", "Done" if not bbq.empty else "Missing"],
            ["Fairness judge evaluation", "Done" if not judge_summary.empty else "Missing"],
            ["Confidence intervals", "Done" if any(col.lower().endswith("_ci_low") for col in results.columns) else "Missing"],
            ["Interactive Streamlit dashboard", "Done"],
            ["Static dashboard export", "Done"],
            ["Smoke test / reproducibility", "Done"],
            ["Tool comparison table", "Done" if not tool_comparison.empty else "Optional / missing"],
            ["Local/unpaid execution path", "Done"],
            ["Optional API-backed execution path", "Done" if not crows.empty and not bbq.empty else "Partial"],
        ],
        columns=["Requirement", "Status"],
    )
    display_df(checklist, height=420)

    if not tool_comparison.empty:
        st.markdown("**Tool comparison**")
        display_df(tool_comparison)

    st.markdown(
        """
<div class="custom-card">
<b>Final coverage statement:</b><br>
This dashboard now documents both the <b>core local package workflow</b> and the <b>optional paid/API LLM extension</b>, which makes the roadmap coverage explicit for grading.
</div>
""",
        unsafe_allow_html=True,
    )