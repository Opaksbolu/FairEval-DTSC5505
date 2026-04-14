from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="FairEval Dashboard", layout="wide")

# ----------------------------
# Appearance toggle
# ----------------------------
if "appearance_mode" not in st.session_state:
    st.session_state.appearance_mode = "Auto"

with st.sidebar:
    st.header("Dashboard Settings")
    st.selectbox(
        "Appearance",
        ["Auto", "Light", "Dark"],
        key="appearance_mode",
    )

mode = st.session_state.appearance_mode

light_css = """
<style>
:root {
  --bg: #ffffff;
  --card: #f7f7f7;
  --text: #111111;
  --muted: #444444;
  --border: #d9d9d9;
}

html, body, [data-testid="stAppViewContainer"], .main {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background-color: #f3f4f6 !important;
  color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
  color: var(--text) !important;
}

div[data-testid="stMetric"] {
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.5rem !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
  color: var(--text) !important;
}

div[data-testid="stDataFrame"] *,
table, th, td {
  color: var(--text) !important;
}

.custom-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1rem;
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
}

html, body, [data-testid="stAppViewContainer"], .main {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background-color: #111827 !important;
  color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
  color: var(--text) !important;
}

div[data-testid="stMetric"] {
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.5rem !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
  color: var(--text) !important;
}

div[data-testid="stDataFrame"] *,
table, th, td {
  color: var(--text) !important;
}

.custom-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1rem;
}
</style>
"""

# ----------------------------
# Theme-aware figure helpers
# ----------------------------
def get_theme_colors(mode: str):
    if mode == "Dark":
        return {
            "fig_bg": "#0e1117",
            "ax_bg": "#161b22",
            "text": "#f3f3f3",
            "grid": "#2d333b",
            "spine": "#9aa4b2",
        }
    return {
        "fig_bg": "#ffffff",
        "ax_bg": "#ffffff",
        "text": "#111111",
        "grid": "#d9d9d9",
        "spine": "#666666",
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


def plot_accuracy(results, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    pivot = results.pivot(index="Model", columns="Dataset", values="Accuracy")
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Figure 1. Accuracy by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    style_axes(ax, colors)

    leg = ax.get_legend()
    if leg:
        leg.get_title().set_color(colors["text"])
        for text in leg.get_texts():
            text.set_color(colors["text"])
        leg.get_frame().set_facecolor(colors["ax_bg"])
        leg.get_frame().set_edgecolor(colors["spine"])

    fig.tight_layout()
    return fig


def plot_dp(results, mode):
    colors = get_theme_colors(mode)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(colors["fig_bg"])

    pivot = results.pivot(index="Model", columns="Dataset", values="Demographic Parity")
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Figure 2. Demographic Parity Difference by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Demographic Parity Difference (abs)")
    style_axes(ax, colors)

    leg = ax.get_legend()
    if leg:
        leg.get_title().set_color(colors["text"])
        for text in leg.get_texts():
            text.set_color(colors["text"])
        leg.get_frame().set_facecolor(colors["ax_bg"])
        leg.get_frame().set_edgecolor(colors["spine"])

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

    ax.set_title("Figure 3. Equalized Odds vs Predictive Parity")
    ax.set_xlabel("Equalized Odds Difference")
    ax.set_ylabel("Predictive Parity Difference")
    style_axes(ax, colors)

    leg = ax.get_legend()
    if leg:
        leg.get_title().set_color(colors["text"])
        for text in leg.get_texts():
            text.set_color(colors["text"])
        leg.get_frame().set_facecolor(colors["ax_bg"])
        leg.get_frame().set_edgecolor(colors["spine"])

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

    ax.set_title("Figure 4. Agreement Between Fairness Metrics (Kendall τ)")
    style_axes(ax, colors)
    fig.tight_layout()
    return fig


if mode == "Light":
    st.markdown(light_css, unsafe_allow_html=True)
elif mode == "Dark":
    st.markdown(dark_css, unsafe_allow_html=True)

st.title("FairEval Dashboard")
st.caption("Interactive Streamlit dashboard for Milestone 6 inspection and presentation.")

# ----------------------------
# Load outputs
# ----------------------------
results = pd.read_csv(OUTPUTS / "results.csv")
tau = pd.read_csv(OUTPUTS / "agreement_kendall_tau.csv", index_col=0)
alpha = (OUTPUTS / "agreement_krippendorff_alpha.txt").read_text().strip()

judge_summary = pd.read_csv(OUTPUTS / "fairness_judge_summary.csv")
crows = pd.read_csv(OUTPUTS / "crows_pairs_llm_eval.csv")
bbq = pd.read_csv(OUTPUTS / "bbq_llm_eval.csv")
tool_comparison = pd.read_csv(OUTPUTS / "tool_comparison.csv")

crows_acc = float(crows["correct"].mean()) if "correct" in crows.columns else None

bbq_acc = None
if "correct" in bbq.columns:
    valid = bbq[crows.columns.intersection(["correct"]).tolist() or ["correct"]]
    valid = bbq[bbq["correct"].isin([True, False])]
    if len(valid) > 0:
        bbq_acc = float(valid["correct"].mean())

st.subheader("Core benchmark summary")

c1, c2, c3 = st.columns(3)
c1.metric("Krippendorff's alpha", alpha)
c2.metric(
    "CrowS-Pairs proxy accuracy",
    f"{crows_acc:.3f}" if crows_acc is not None else "n/a",
)
c3.metric(
    "BBQ proxy accuracy",
    f"{bbq_acc:.3f}" if bbq_acc is not None else "n/a",
)

st.subheader("Tabular benchmark results")
st.dataframe(results, use_container_width=True)

c4, c5 = st.columns(2)

with c4:
    st.subheader("Agreement matrix")
    st.dataframe(tau, use_container_width=True)

with c5:
    st.subheader("Fairness-judge summary")
    st.dataframe(judge_summary, use_container_width=True)

st.subheader("FairEval vs Existing Tools")
st.dataframe(tool_comparison, use_container_width=True)

st.subheader("Generated figures")

fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.pyplot(plot_accuracy(results, mode), use_container_width=True)
with fig_col2:
    st.pyplot(plot_dp(results, mode), use_container_width=True)

fig_col3, fig_col4 = st.columns(2)
with fig_col3:
    st.pyplot(plot_eo_vs_pp(results, mode), use_container_width=True)
with fig_col4:
    st.pyplot(plot_tau_heatmap(tau, mode), use_container_width=True)