from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_adult, load_compas, load_german_credit, load_bbq_optional
from src.preprocessing import (
    preprocess_adult,
    preprocess_compas,
    preprocess_german_credit,
)
from src.fairness_metrics import evaluate_model_on_split
from src.agreement import metric_agreement_kendall_tau, krippendorff_alpha
from src.llm_eval import run_crows_pairs_eval
from src.bbq_eval import run_bbq_llm_eval, summarize_bbq_results


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def ensure_outputs_dir() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "GaussianNB": GaussianNB(),
    }


def evaluate_dataset(dataset_name: str, processed: dict) -> list[dict]:
    models = build_models()
    rows: list[dict] = []

    for model_label, model in models.items():
        metrics = evaluate_model_on_split(
            model=model,
            X_train=processed["X_train"],
            X_test=processed["X_test"],
            y_train=processed["y_train"],
            y_test=processed["y_test"],
            sensitive_train=processed.get("sensitive_train"),
            sensitive_test=processed["sensitive_test"],
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_label,
                "Accuracy": metrics["accuracy"],
                "Demographic Parity": metrics["demographic_parity"],
                "Equalized Odds": metrics["equalized_odds"],
                "Predictive Parity": metrics["predictive_parity"],
            }
        )

    return rows


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

    print("Running models + fairness metrics...")
    all_rows: list[dict] = []
    all_rows.extend(evaluate_dataset("Adult", adult_processed))
    all_rows.extend(evaluate_dataset("COMPAS", compas_processed))
    all_rows.extend(evaluate_dataset("German Credit", german_processed))

    results = pd.DataFrame(all_rows)
    results_path = OUTPUTS / "results.csv"
    results.to_csv(results_path, index=False)
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
            csv_path="datasets/crows_pairs.csv",
            output_path=str(OUTPUTS / "crows_pairs_llm_eval.csv"),
            model="gpt-5",
            use_api=True,
            max_items=50,
        )
        crows_acc = float(crows_df["correct"].mean()) if "correct" in crows_df.columns else 0.0
        print(f"[LLM] CrowS-Pairs accuracy: {crows_acc:.3f}")
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
                bbq_df=bbq_df,
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