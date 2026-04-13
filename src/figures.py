from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def generate_figures(results: pd.DataFrame, kendall: pd.DataFrame, out_dir: str = "figures") -> None:
    """
    Generates:
      Figure 1: Accuracy by Model and Dataset
      Figure 2: Demographic Parity Difference by Model and Dataset (abs)
      Figure 3: Equalized Odds vs Predictive Parity scatter (per dataset)
      Figure 4: Kendall tau heatmap (rendered with matplotlib)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Accuracy ----
    fig = plt.figure()
    ax = plt.gca()

    pivot = results.pivot(index="Model", columns="Dataset", values="Accuracy")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Figure 1. Accuracy by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(out / "fig1_accuracy_by_model_dataset.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: Demographic Parity (absolute) ----
    fig = plt.figure()
    ax = plt.gca()

    dp = results.copy()
    dp["Demographic Parity"] = dp["Demographic Parity"].abs()
    pivot = dp.pivot(index="Model", columns="Dataset", values="Demographic Parity")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Figure 2. Demographic Parity Difference by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Demographic Parity Difference (abs)")
    plt.tight_layout()
    fig.savefig(out / "fig2_demographic_parity.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: EO vs PP scatter ----
    fig = plt.figure()
    ax = plt.gca()

    for ds, sub in results.groupby("Dataset"):
        ax.scatter(sub["Equalized Odds"], sub["Predictive Parity"], label=ds)
        for _, r in sub.iterrows():
            ax.annotate(r["Model"], (r["Equalized Odds"], r["Predictive Parity"]))

    ax.set_title("Figure 3. Equalized Odds vs Predictive Parity (per Dataset)")
    ax.set_xlabel("Equalized Odds Difference")
    ax.set_ylabel("Predictive Parity Difference")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "fig3_eo_vs_pp.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4: Kendall tau heatmap (matplotlib) ----
    fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(kendall.values)
    ax.set_xticks(range(len(kendall.columns)))
    ax.set_xticklabels(kendall.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(kendall.index)))
    ax.set_yticklabels(kendall.index)
    ax.set_title("Figure 4. Agreement Between Fairness Metrics (Kendall τ)")

    for i in range(kendall.shape[0]):
        for j in range(kendall.shape[1]):
            ax.text(j, i, f"{kendall.iloc[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out / "fig4_kendall_tau_heatmap.png", dpi=200)
    plt.close(fig)