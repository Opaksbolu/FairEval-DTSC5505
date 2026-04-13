#!/usr/bin/env python3
"""Generate Milestone-6 figures from outputs/results.csv and outputs/agreement_kendall_tau.csv.

Usage:
  python scripts/generate_figures.py

This script assumes you already ran:
  python main.py
which creates outputs/results.csv and outputs/agreement_kendall_tau.csv.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
out_dir = ROOT / "outputs"
fig_dir = ROOT / "figures"
fig_dir.mkdir(exist_ok=True)

results_path = out_dir / "results.csv"
tau_path = out_dir / "agreement_kendall_tau.csv"

if not results_path.exists():
    raise FileNotFoundError(f"Missing {results_path}. Run `python main.py` first.")
if not tau_path.exists():
    raise FileNotFoundError(f"Missing {tau_path}. Run `python main.py` first.")

df = pd.read_csv(results_path)
tau = pd.read_csv(tau_path, index_col=0)

# --- Robust column handling ---
# Different runs / versions may write slightly different column names
# (e.g., "model" vs "Model", "dataset" vs "Dataset").

def _canon(col: str) -> str:
    return str(col).strip().lower().replace("_", " ")

canon_to_actual = {_canon(c): c for c in df.columns}

def _pick(*candidates: str):
    """Return the first matching column (case-insensitive) from candidates."""
    for cand in candidates:
        key = _canon(cand)
        if key in canon_to_actual:
            return canon_to_actual[key]
    return None

COL_MODEL = _pick("Model", "model", "model name", "model_name")
COL_DATASET = _pick("Dataset", "dataset")
COL_ACC = _pick("Accuracy", "accuracy", "acc")
COL_DP = _pick("Demographic Parity", "demographic parity", "demographic_parity")
COL_EO = _pick("Equalized Odds", "equalized odds", "equalized_odds")
COL_PP = _pick("Predictive Parity", "predictive parity", "predictive_parity")

required_for_fig1 = [COL_MODEL, COL_DATASET, COL_ACC]
if any(c is None for c in required_for_fig1):
    raise KeyError(
        "results.csv is missing required columns for figures. "
        f"Found columns: {list(df.columns)}\n"
        "Expected (any casing): Model, Dataset, Accuracy."
    )


# Figure 1: Accuracy by dataset/model
pivot_acc = df.pivot(index=COL_MODEL, columns=COL_DATASET, values=COL_ACC)
ax = pivot_acc.plot(kind="bar")
ax.set_ylabel("Accuracy")
ax.set_title("Figure 1. Accuracy by Model and Dataset")
plt.tight_layout()
plt.savefig(fig_dir / "fig1_accuracy_by_model_dataset.png", dpi=200)
plt.close()

# Figure 2: Fairness metrics by dataset/model (Demographic Parity)
if COL_DP is not None:
    pivot_dp = df.pivot(index=COL_MODEL, columns=COL_DATASET, values=COL_DP)
    ax = pivot_dp.plot(kind="bar")
    ax.set_ylabel("Demographic Parity Difference (abs)")
    ax.set_title("Figure 2. Demographic Parity Difference by Model and Dataset")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_demographic_parity.png", dpi=200)
    plt.close()

else:
    print("Skipping fig2_demographic_parity: missing Demographic Parity column in results.csv")
# Figure 3: Equalized Odds vs Predictive Parity scatter
# Figure 3: Equalized Odds vs Predictive Parity scatter
if (COL_EO is not None) and (COL_PP is not None):
    ax = plt.gca()
    for ds in df[COL_DATASET].unique():
        sub = df[df[COL_DATASET] == ds]
        ax.scatter(sub[COL_EO], sub[COL_PP], label=ds)
        for _, r in sub.iterrows():
            ax.annotate(str(r[COL_MODEL]), (r[COL_EO], r[COL_PP]), fontsize=7)
    ax.set_xlabel("Equalized Odds Difference")
    ax.set_ylabel("Predictive Parity Difference")
    ax.set_title("Figure 3. Equalized Odds vs Predictive Parity (per Dataset)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_eo_vs_pp.png", dpi=200)
    plt.close()

else:
    print("Skipping fig3_eo_vs_pp: missing Equalized Odds and/or Predictive Parity column in results.csv")
# Figure 4: Kendall tau agreement heatmap (simple image)
# Figure 4: Kendall tau agreement heatmap (simple image)
fig, ax = plt.subplots()
im = ax.imshow(tau.values)
ax.set_xticks(range(len(tau.columns)))
ax.set_yticks(range(len(tau.index)))
ax.set_xticklabels(tau.columns, rotation=30, ha="right")
ax.set_yticklabels(tau.index)
for i in range(tau.shape[0]):
    for j in range(tau.shape[1]):
        ax.text(j, i, f"{tau.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Figure 4. Agreement Between Fairness Metrics (Kendall τ)")
plt.tight_layout()
plt.savefig(fig_dir / "fig4_kendall_tau_heatmap.png", dpi=200)
plt.close()

print("Saved figures to:", fig_dir)

# FIG2_GUARD_INSERTED
