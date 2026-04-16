from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_adult, load_compas, load_german_credit, load_bbq_optional
from src.preprocessing import (
    preprocess_adult,
    preprocess_compas,
    preprocess_german_credit,
)
from src.experiments import run_tabular_experiments
from src.agreement import metric_agreement_kendall_tau, krippendorff_alpha
from src.llm_eval import run_crows_pairs_eval
from src.bbq_eval import run_bbq_llm_eval, summarize_bbq_results


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def ensure_outputs_dir() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_outputs_dir()

    print("Loading datasets...")
    adult_df = load_adult()
    compas_df = load_compas()
    german_df = load_german_credit()

    print("Preprocessing Adult dataset...")
    adult_processed = preprocess_adult(adult_df)

    print("Preprocessing COMPAS dataset...")
    compas_processed = preprocess_compas(compas_df)

    print("Preprocessing German Credit dataset...")
    german_processed = preprocess_german_credit(german_df)

    datasets = {
        "adult": adult_processed,
        "compas": compas_processed,
        "german": german_processed,
    }

    print("Running models + fairness metrics...")
    results = run_tabular_experiments(
        datasets=datasets,
        output_dir=OUTPUTS,
    )

    results_path = OUTPUTS / "results.csv"
    print(f"Saved results to {results_path}")

    metric_cols = [
        "Accuracy",
        "Demographic Parity",
        "Equalized Odds",
        "Predictive Parity",
    ]

    print("\n=== Metric agreement (Kendall tau) ===")
    tau = metric_agreement_kendall_tau(results, metric_cols)
    print(tau)

    tau_path = OUTPUTS / "agreement_kendall_tau.csv"
    tau.to_csv(tau_path)
    print(f"Saved Kendall tau matrix to {tau_path}")

    alpha_value = krippendorff_alpha(results, metric_cols)
    alpha_path = OUTPUTS / "agreement_krippendorff_alpha.txt"
    alpha_path.write_text(f"{alpha_value:.6f}")

    print(f"\nKrippendorff alpha: {alpha_value:.6f}")
    print(f"Saved Krippendorff alpha to {alpha_path}")

    print("\n[LLM] Running CrowS-Pairs evaluation (first 50 items)...")
    try:
        crows_df = run_crows_pairs_eval(
            csv_path=str(ROOT / "datasets" / "crows_pairs.csv"),
            output_path=str(OUTPUTS / "crows_pairs_llm_eval.csv"),
            model="gpt-5",
            use_api=True,
            max_items=50,
        )
        if "correct" in crows_df.columns:
            acc = float(crows_df["correct"].mean())
            print(f"[LLM] CrowS-Pairs accuracy: {acc:.3f}")
        print(f"[LLM] Saved to {OUTPUTS / 'crows_pairs_llm_eval.csv'}")
    except Exception as e:
        print(f"[LLM] CrowS-Pairs eval skipped/failed: {e}")

    print("\n[LLM] Running BBQ evaluation (first 200 items)...")
    try:
        bbq_df = load_bbq_optional()
        if bbq_df is None:
            print("[LLM] BBQ dataset not found. Skipping BBQ evaluation.")
        else:
            bbq_eval_df = run_bbq_llm_eval(
                bbq_df,
                out_path=str(OUTPUTS / "bbq_llm_eval.csv"),
                max_items=200,
                model="gpt-5",
            )
            bbq_summary = summarize_bbq_results(bbq_eval_df)
            print(f"[LLM] BBQ summary: {bbq_summary}")
            print(f"[LLM] Saved to {OUTPUTS / 'bbq_llm_eval.csv'}")
    except Exception as e:
        print(f"[LLM] BBQ eval skipped/failed: {e}")


if __name__ == "__main__":
    main()