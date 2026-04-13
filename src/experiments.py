# src/experiments.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .fairness_metrics import evaluate_model_on_split
from .models import get_models


DatasetParts = Dict[str, object]
DatasetsDict = Dict[str, DatasetParts]


def _normalize_parts(parts: Any) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test, sensitive_train, sensitive_test).

    Preprocessing returns a 6-tuple, but earlier versions used a dict.
    This helper supports both so main.py can stay simple.
    """

    # Newer style: 6-tuple
    if isinstance(parts, (list, tuple)) and len(parts) == 6:
        X_train, X_test, y_train, y_test, s_train, s_test = parts
        return X_train, X_test, y_train, y_test, s_train, s_test

    # Older style: dict
    if isinstance(parts, dict):
        X_train = parts.get("X_train")
        X_test = parts.get("X_test")
        y_train = parts.get("y_train")
        y_test = parts.get("y_test")
        s_train = parts.get("sensitive_train", parts.get("s_train"))
        s_test = parts.get("sensitive_test", parts.get("s_test"))
        return X_train, X_test, y_train, y_test, s_train, s_test

    raise TypeError(
        "Expected dataset parts as a dict of train/test arrays or a 6-tuple "
        "(X_train, X_test, y_train, y_test, sensitive_train, sensitive_test)."
    )


def run_tabular_experiments(
    datasets_or_adult: Union[DatasetsDict, DatasetParts],
    compas_parts: Optional[DatasetParts] = None,
    german_parts: Optional[DatasetParts] = None,
    output_csv: str = "outputs/results.csv",
) -> pd.DataFrame:
    """Run tabular experiments across multiple datasets and models.

    Preferred calling style:
        run_tabular_experiments(datasets)

    where `datasets` is a dict mapping dataset name -> dataset parts dict.

    Backwards-compatible calling style:
        run_tabular_experiments(adult_parts, compas_parts, german_parts)

    where each `*_parts` dict contains keys:
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test
    """

    # If the caller passed 3 positional args, stitch them into the datasets dict.
    if compas_parts is not None or german_parts is not None:
        if compas_parts is None or german_parts is None:
            raise TypeError(
                "run_tabular_experiments(adult_parts, compas_parts, german_parts) requires 3 dataset-part dicts"
            )
        datasets: DatasetsDict = {
            "Adult": datasets_or_adult,  # type: ignore[assignment]
            "COMPAS": compas_parts,
            "German": german_parts,
        }
    else:
        datasets = datasets_or_adult  # type: ignore[assignment]

    results: List[Dict] = []
    models = get_models()

    for dataset_name, parts in datasets.items():
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = _normalize_parts(parts)

        for model_name, model in models.items():
            metrics = evaluate_model_on_split(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_test,
            )
            # Standardize output column names across the project.
            results.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy"),
                    "Demographic Parity": metrics.get("demographic_parity"),
                    "Equalized Odds": metrics.get("equalized_odds"),
                    "Predictive Parity": metrics.get("predictive_parity"),
                }
            )

    df = pd.DataFrame(results)

    # Stable ordering for readability and deterministic diffs.
    if not df.empty and set(["Dataset", "Model"]).issubset(df.columns):
        df = df.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    return df
