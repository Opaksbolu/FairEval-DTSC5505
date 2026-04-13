from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Return the benchmark model collection.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=False,
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "GaussianNB": GaussianNB(),
    }