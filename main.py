from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import (
    load_adult,
    load_compas,
    load_german_credit,
    load_crows_pairs,
    load_bbq_optional,
)
from src.preprocessing import preprocess_adult, preprocess_compas, preprocess_german_credit
from src.experiments import run_tabular_experiments
from src.agreement import kendall_tau_matrix, krippendorff_alpha_metrics
from src.llm_eval import run_crows_pairs_llm_eval
from src.bbq_eval import run_bbq_llm_eval, summarize_bbq_results


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    _ensure_dir("outputs")

    print("Loading datasets...")
    adult_df = load_adult()
    compas_df = load_compas()
    german_df = load_german_credit()
    crows_df = load_crows_pairs()
    bbq_df = load_bbq_optional()

    print("Preprocessing Adult dataset...")
    adult_data = preprocess_adult(adult_df)

    print("Preprocessing COMPAS dataset...")
    compas_data = preprocess_compas(compas_df)

    print("Preprocessing German Credit dataset...")
    german_data = preprocess_german_credit(german_df)

    print("Running models + fairness metrics...")
    datasets = {
    "adult": adult_data,
    "compas": compas_data,
    "german": german_data,
    }

    results_df = run_tabular_experiments(datasets)
    results_path = "outputs/results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # Agreement (Kendall tau matrix)
    # Keep metric column names aligned with `src/experiments.py` output.
    # (Older drafts used snake_case; the saved CSV uses Title Case.)
    metric_cols = ["Demographic Parity", "Equalized Odds", "Predictive Parity"]
    tau_df = kendall_tau_matrix(results_df, metric_cols)

    print("\n=== Metric agreement (Kendall tau) ===")
    pretty = tau_df.copy()
    pretty.index = ["Demographic Parity", "Equalized Odds", "Predictive Parity"]
    pretty.columns = ["Demographic Parity", "Equalized Odds", "Predictive Parity"]
    print(pretty)

    tau_out = "outputs/agreement_kendall_tau.csv"
    tau_df.to_csv(tau_out)
    print(f"Saved Kendall tau matrix to {tau_out}")

    # Krippendorff alpha
    alpha = krippendorff_alpha_metrics(results_df, metric_cols)
    print(f"\nKrippendorff alpha: {alpha:.6f}")
    alpha_out = "outputs/agreement_krippendorff_alpha.txt"
    Path(alpha_out).write_text(f"{alpha:.6f}\n")
    print(f"Saved Krippendorff alpha to {alpha_out}")

    # (1) CrowS-Pairs LLM eval
    print("\n[LLM] Running CrowS-Pairs evaluation (first 50 items)...")
    try:
        run_crows_pairs_llm_eval(crows_df, out_path="outputs/crows_pairs_llm_eval.csv", max_items=50)
        print("[LLM] Saved to outputs/crows_pairs_llm_eval.csv")
    except Exception as e:
        print(f"[LLM] CrowS-Pairs eval skipped/failed: {e}")

    # (2) BBQ text-audit eval (offline proxy or API-backed)
    if bbq_df is not None:
        print(f"\n[LLM] Running BBQ evaluation (first {min(200, len(bbq_df))} items)...")
        try:
            bbq_out = run_bbq_llm_eval(
                bbq_df,
                out_path="outputs/bbq_llm_eval.csv",
                max_items=200,
                model="gpt-4o-mini",
            )
        except Exception as e:
            print(f"[LLM] BBQ eval skipped/failed: {e}")
            bbq_out = None
        if bbq_out is None:
            print("[LLM] BBQ eval skipped (schema mismatch or missing key columns).")
        else:
            summary = summarize_bbq_results(bbq_out)
            print("[LLM] BBQ summary:", summary)
            print("[LLM] Saved to outputs/bbq_llm_eval.csv")
    else:
        print("\n[LLM] BBQ dataset not found (datasets/bbq.csv). Skipping BBQ eval.")


if __name__ == "__main__":
    main()
