from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def _to_1d_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        arr = x.values
    else:
        arr = np.asarray(x)
    return arr.reshape(-1)


def demographic_parity_difference(y_pred, sensitive_features) -> float:
    """
    Demographic parity difference = max_group_positive_rate - min_group_positive_rate
    """
    y_pred = _to_1d_array(y_pred)
    s = _to_1d_array(sensitive_features)

    groups = pd.Series(s).astype(str)
    pred = pd.Series(y_pred)

    rates = pred.groupby(groups).mean()
    if len(rates) <= 1:
        return 0.0
    return float(rates.max() - rates.min())


def equalized_odds_difference(y_true, y_pred, sensitive_features) -> float:
    """
    Equalized odds difference = max over groups of |TPR_g - TPR_ref| and |FPR_g - FPR_ref|.
    Implemented as (max TPR - min TPR) and (max FPR - min FPR), then take max of those.
    """
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)
    s = _to_1d_array(sensitive_features)

    groups = pd.Series(s).astype(str)

    tprs = []
    fprs = []

    for g, idx in groups.groupby(groups).groups.items():
        yt = y_true[list(idx)]
        yp = y_pred[list(idx)]

        pos = (yt == 1)
        neg = (yt == 0)

        # TPR = P(ŷ=1 | y=1)
        if pos.sum() == 0:
            tpr = np.nan
        else:
            tpr = float((yp[pos] == 1).mean())

        # FPR = P(ŷ=1 | y=0)
        if neg.sum() == 0:
            fpr = np.nan
        else:
            fpr = float((yp[neg] == 1).mean())

        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs, dtype=float)
    fprs = np.array(fprs, dtype=float)

    # Drop NaNs if a group has no positives/negatives
    tprs = tprs[~np.isnan(tprs)]
    fprs = fprs[~np.isnan(fprs)]

    tpr_gap = float(tprs.max() - tprs.min()) if tprs.size >= 2 else 0.0
    fpr_gap = float(fprs.max() - fprs.min()) if fprs.size >= 2 else 0.0

    return max(tpr_gap, fpr_gap)


def predictive_parity_difference(y_true, y_pred, sensitive_features) -> float:
    """
    Predictive parity difference = max_group_PPV - min_group_PPV
    PPV = P(y=1 | ŷ=1)
    """
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)
    s = _to_1d_array(sensitive_features)

    groups = pd.Series(s).astype(str)

    ppvs = []
    for g, idx in groups.groupby(groups).groups.items():
        yt = y_true[list(idx)]
        yp = y_pred[list(idx)]

        pred_pos = (yp == 1)
        if pred_pos.sum() == 0:
            ppv = np.nan
        else:
            ppv = float((yt[pred_pos] == 1).mean())
        ppvs.append(ppv)

    ppvs = np.array(ppvs, dtype=float)
    ppvs = ppvs[~np.isnan(ppvs)]

    if ppvs.size <= 1:
        return 0.0
    return float(ppvs.max() - ppvs.min())


def evaluate_model_on_split(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_train: pd.Series,
    sensitive_test: pd.Series,
) -> Dict[str, float]:
    """
    Fits model, predicts, and returns:
      - accuracy
      - demographic_parity (difference)
      - equalized_odds (difference)
      - predictive_parity (difference)
    Keys must match what src/experiments.py expects.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    dp = demographic_parity_difference(y_pred, sensitive_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_test)
    pp = predictive_parity_difference(y_test, y_pred, sensitive_test)

    return {
        "accuracy": acc,
        "demographic_parity": float(dp),
        "equalized_odds": float(eo),
        "predictive_parity": float(pp),
    }