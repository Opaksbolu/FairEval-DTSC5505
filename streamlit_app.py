from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

st.set_page_config(page_title="FairEval Dashboard", layout="wide")

# ----------------------------
# Appearance toggle
# ----------------------------
if "appearance_mode" not in st.session_state:
    st.session_state.appearance_mode = "Auto"

with st.sidebar:
    st.header("Dashboard Settings")
    appearance = st.selectbox(
        "Appearance",
        ["Auto", "Light", "Dark"],
        index=["Auto", "Light", "Dark"].index(st.session_state.appearance_mode),
    )
    st.session_state.appearance_mode = appearance

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
.block-container {
  background-color: var(--bg);
  color: var(--text);
}
div[data-testid="stMetric"], div[data-testid="stDataFrame"], .stMarkdown, .stTable {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] {
  background-color: #111827;
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

crows_acc = float(crows["correct"].mean()) if "correct" in crows.columns else None

bbq_acc = None
if "correct" in bbq.columns:
    valid = bbq[bbq["correct"].isin([True, False])]
    if len(valid) > 0:
        bbq_acc = float(valid["correct"].mean())

st.subheader("Core benchmark summary")

c1, c2, c3 = st.columns(3)
c1.metric("Krippendorff's alpha", alpha)
c2.metric("CrowS-Pairs proxy accuracy", f"{crows_acc:.3f}" if crows_acc is not None else "n/a")
c3.metric("BBQ proxy accuracy", f"{bbq_acc:.3f}" if bbq_acc is not None else "n/a")

st.subheader("Tabular benchmark results")
st.dataframe(results, use_container_width=True)

c4, c5 = st.columns(2)

with c4:
    st.subheader("Agreement matrix")
    st.dataframe(tau, use_container_width=True)

with c5:
    st.subheader("Fairness-judge summary")
    st.dataframe(judge_summary, use_container_width=True)

st.subheader("Generated figures")

fig_paths = [
    FIGURES / "fig1_accuracy_by_model_dataset.png",
    FIGURES / "fig2_demographic_parity.png",
    FIGURES / "fig3_eo_vs_pp.png",
    FIGURES / "fig4_kendall_tau_heatmap.png",
]

cols = st.columns(2)
for i, fig_path in enumerate(fig_paths):
    with cols[i % 2]:
        st.image(str(fig_path), caption=fig_path.name, use_container_width=True)