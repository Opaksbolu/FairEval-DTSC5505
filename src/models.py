from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelSpec:
    name: str
    estimator: object


def get_models(random_state: int = 42) -> Dict[str, object]:
    """Two solid baselines that run fast."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=None,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
