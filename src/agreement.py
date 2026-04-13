from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

def _resolve_metric_column(df: pd.DataFrame, name: str) -> str | None:
    """Resolve a requested metric name to an actual dataframe column."""
    if name in df.columns:
        return name
    # Common transforms
    # snake_case -> Title Case with spaces
    if '_' in name and name.lower() == name:
        title = name.replace('_', ' ').title()
        if title in df.columns:
            return title
    # Title Case / spaces -> snake_case
    snake = name.lower().replace(' ', '_')
    if snake in df.columns:
        return snake
    # Try case-insensitive exact match
    lower_map = {c.lower(): c for c in df.columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    return None
import krippendorff
from scipy.stats import kendalltau


def metric_agreement_kendall_tau(results_df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Compute Kendall tau agreement between metric rankings of models (averaged across datasets).
    Returns a square DataFrame with one row/col per metric.
    """
    # Support either legacy lowercase columns or standardized Title Case.
    model_col = "Model" if "Model" in results_df.columns else "model"
    model_order = sorted(results_df[model_col].unique())

    ranks = {}
    for m in metric_cols:
        # Allow callers to pass Title Case metric names even if the DataFrame
        # contains snake_case columns (or vice versa).
        use_metric = _resolve_metric_column(results_df, m)
        if not use_metric:
            # Skip missing metrics gracefully
            continue
        avg_scores = results_df.groupby(model_col)[use_metric].mean().reindex(model_order)
        # lower is "better" for fairness gaps, higher is "better" for accuracy.
        # We'll rank by ascending for fairness metrics and descending for accuracy.
        if m.lower() == "accuracy":
            ranks[m] = avg_scores.rank(ascending=False, method="average").values
        else:
            ranks[m] = avg_scores.rank(ascending=True, method="average").values

    matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)
    for i, m1 in enumerate(metric_cols):
        for j, m2 in enumerate(metric_cols):
            if i == j:
                matrix.loc[m1, m2] = 1.0
            else:
                tau, _ = kendalltau(ranks[m1], ranks[m2])
                matrix.loc[m1, m2] = float(tau) if tau is not None else float("nan")

    return matrix


def krippendorff_alpha(results_df: pd.DataFrame, metric_cols: List[str]) -> float:
    """
    Compute Krippendorff's alpha over metric-based model rankings (averaged across datasets).
    Each metric is treated as a "rater" ranking the models.
    """
    model_col = "Model" if "Model" in results_df.columns else "model"
    model_order = sorted(results_df[model_col].unique())

    all_ranks = []
    for m in metric_cols:
        use_metric = _resolve_metric_column(results_df, m)
        if not use_metric:
            # Skip missing metrics gracefully
            continue
        avg_scores = results_df.groupby(model_col)[use_metric].mean().reindex(model_order)
        if m.lower() == "accuracy":
            r = avg_scores.rank(ascending=False, method="average").values
        else:
            r = avg_scores.rank(ascending=True, method="average").values
        all_ranks.append(r)

    data = np.array(all_ranks)  # shape: (n_raters, n_items)
    alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
    return float(alpha)


# -------------------------------------------------------------------
# Backwards-compatible names (what your main.py imports)
# -------------------------------------------------------------------
def kendall_tau_matrix(results_df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    return metric_agreement_kendall_tau(results_df, metric_cols)


def krippendorff_alpha_metrics(results_df: pd.DataFrame, metric_cols: List[str]) -> float:
    return krippendorff_alpha(results_df, metric_cols)