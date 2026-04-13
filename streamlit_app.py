from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

st.set_page_config(page_title="FairEval Dashboard", layout="wide")

st.title("FairEval Dashboard")
st.caption("Interactive Streamlit dashboard for Milestone 6 inspection and presentation.")

# Load outputs
results = pd.read_csv(OUTPUTS / "results.csv")
tau = pd.read_csv(OUTPUTS / "agreement_kendall_tau.csv", index_col=0)
alpha = (OUTPUTS / "agreement_krippendorff_alpha.txt").read_text().strip()

judge_summary = pd.read_csv(OUTPUTS / "fairness_judge_summary.csv")
crows = pd.read_csv(OUTPUTS / "crows_pairs_llm_eval.csv")
bbq = pd.read_csv(OUTPUTS / "bbq_llm_eval.csv")

# Summary metrics
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
