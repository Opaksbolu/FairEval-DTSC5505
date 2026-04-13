from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def _to_numpy(x):
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x).ravel()


def demographic_parity_difference(y_pred, sensitive) -> float:
    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0

    rates = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            continue
        rates.append(float(np.mean(y_pred[mask] == 1)))

    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def equalized_odds_difference(y_true, y_pred, sensitive) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0

    stats = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        stats.append((tpr, fpr))

    if len(stats) < 2:
        return 0.0

    tprs = [x[0] for x in stats]
    fprs = [x[1] for x in stats]
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def predictive_parity_difference(y_true, y_pred, sensitive) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0

    ppvs = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        predicted_positive = yp == 1
        if predicted_positive.sum() == 0:
            ppv = 0.0
        else:
            ppv = float(np.mean(yt[predicted_positive] == 1))

        ppvs.append(ppv)

    if len(ppvs) < 2:
        return 0.0
    return float(max(ppvs) - min(ppvs))


def compute_metrics(y_true, y_pred, sensitive) -> Dict[str, float]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive = _to_numpy(sensitive)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "demographic_parity": demographic_parity_difference(y_pred, sensitive),
        "equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive),
        "predictive_parity": predictive_parity_difference(y_true, y_pred, sensitive),
    }


def evaluate_model_on_split(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    sensitive_train=None,
    sensitive_test=None,
) -> Dict[str, Any]:
    """
    Train a model on the provided split and return benchmark metrics.
    Converts sparse matrices to dense for models that require dense input.
    """
    if sensitive_test is None:
        raise ValueError("sensitive_test must be provided")

    model_name = model.__class__.__name__
    dense_required_models = {"GaussianNB"}

    X_train_fit = X_train
    X_test_fit = X_test

    if model_name in dense_required_models:
        if hasattr(X_train_fit, "toarray"):
            X_train_fit = X_train_fit.toarray()
        if hasattr(X_test_fit, "toarray"):
            X_test_fit = X_test_fit.toarray()

    model.fit(X_train_fit, y_train)
    y_pred = model.predict(X_test_fit)

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        sensitive=sensitive_test,
    )

    return {
        "accuracy": metrics["accuracy"],
        "demographic_parity": metrics["demographic_parity"],
        "equalized_odds": metrics["equalized_odds"],
        "predictive_parity": metrics["predictive_parity"],
    }