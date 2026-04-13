from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from .models import get_models
from .fairness_metrics import evaluate_model_on_split


def run_tabular_experiments(
    datasets: Dict[str, Dict[str, Any]],
    output_dir: Path | str = "outputs",
) -> pd.DataFrame:
    """
    Run all configured benchmark models across all prepared datasets.
    Each dataset bundle is expected to contain:
      X_train, X_test, y_train, y_test, sensitive_train, sensitive_test
    """
    output_dir = Path(output_dir)
    rows: List[Dict[str, Any]] = []

    models = get_models()

    for dataset_name, bundle in datasets.items():
        X_train = bundle["X_train"]
        X_test = bundle["X_test"]
        y_train = bundle["y_train"]
        y_test = bundle["y_test"]
        sensitive_train = bundle.get("sensitive_train")
        sensitive_test = bundle["sensitive_test"]

        for model_name, model in models.items():
            metrics = evaluate_model_on_split(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
            )

            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Demographic Parity": metrics["demographic_parity"],
                    "Equalized Odds": metrics["equalized_odds"],
                    "Predictive Parity": metrics["predictive_parity"],
                }
            )

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)
    return results