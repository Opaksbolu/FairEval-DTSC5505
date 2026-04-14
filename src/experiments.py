from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .models import get_models
from .fairness_metrics import evaluate_model_on_split, compute_metrics


def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    sensitive,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap 95% confidence intervals for accuracy and fairness metrics.
    """
    rng = np.random.default_rng(random_state)

    if hasattr(y_true, "to_numpy"):
        y_true = y_true.to_numpy()
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()
    if hasattr(sensitive, "to_numpy"):
        sensitive = sensitive.to_numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    n = len(y_true)

    acc_vals = []
    dp_vals = []
    eo_vals = []
    pp_vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        yt = y_true[idx]
        yp = y_pred[idx]
        st = sensitive[idx]

        m = compute_metrics(yt, yp, st)
        acc_vals.append(m["accuracy"])
        dp_vals.append(m["demographic_parity"])
        eo_vals.append(m["equalized_odds"])
        pp_vals.append(m["predictive_parity"])

    def bounds(vals):
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    acc_low, acc_high = bounds(acc_vals)
    dp_low, dp_high = bounds(dp_vals)
    eo_low, eo_high = bounds(eo_vals)
    pp_low, pp_high = bounds(pp_vals)

    return {
        "Accuracy_CI_Low": acc_low,
        "Accuracy_CI_High": acc_high,
        "DP_CI_Low": dp_low,
        "DP_CI_High": dp_high,
        "EO_CI_Low": eo_low,
        "EO_CI_High": eo_high,
        "PP_CI_Low": pp_low,
        "PP_CI_High": pp_high,
    }


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

            # regenerate predictions for bootstrap CIs
            X_test_eval = X_test
            X_train_eval = X_train
            if model.__class__.__name__ in {"GaussianNB"}:
                if hasattr(X_train_eval, "toarray"):
                    X_train_eval = X_train_eval.toarray()
                if hasattr(X_test_eval, "toarray"):
                    X_test_eval = X_test_eval.toarray()

            model.fit(X_train_eval, y_train)
            y_pred = model.predict(X_test_eval)

            ci = bootstrap_confidence_intervals(
                y_true=y_test,
                y_pred=y_pred,
                sensitive=sensitive_test,
                n_bootstrap=200,
                random_state=42,
            )

            row = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Demographic Parity": metrics["demographic_parity"],
                "Equalized Odds": metrics["equalized_odds"],
                "Predictive Parity": metrics["predictive_parity"],
            }
            row.update(ci)
            rows.append(row)

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False)
    return results