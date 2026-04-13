from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on object columns; helps stabilize categories."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()
    return out


def _split_and_transform(X: pd.DataFrame, y: np.ndarray, s: np.ndarray) -> dict:
    X = X.copy()
    # Drop columns that are entirely missing to avoid sklearn imputer warnings/errors
    X = X.dropna(axis=1, how="all")

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipelines
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    # Use numpy arrays for y and s to avoid pandas/arrow indexing edge cases
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(str)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.25, random_state=42, stratify=y
    )

    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    # Standardized payload used across the project.
    return {
        "X_train": X_train_t,
        "X_test": X_test_t,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": s_train,
        "sensitive_test": s_test,
    }


def preprocess_adult(df: pd.DataFrame):
    df = _clean_strings(df)

    # Label: income column (OpenML)
    if "income" not in df.columns:
        # fallback if someone uses UCI naming
        for cand in ["class", "target", "label"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "income"})
                break
    if "income" not in df.columns:
        raise ValueError("Adult dataset missing label column (expected income).")

    y = df["income"].astype(str).str.contains(">50K").astype(int).to_numpy()

    # Sensitive attribute (typical choice): sex
    if "sex" in df.columns:
        s = df["sex"].astype(str).str.lower().to_numpy()
    elif "gender" in df.columns:
        s = df["gender"].astype(str).str.lower().to_numpy()
    else:
        s = np.array(["group"] * len(df), dtype=str)

    X = df.drop(columns=["income"])
    return _split_and_transform(X, y, s)


def preprocess_compas(df: pd.DataFrame):
    df = _clean_strings(df)

    # Label
    if "two_year_recid" not in df.columns:
        for cand in ["is_recid", "recidivism", "recid"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "two_year_recid"})
                break
    if "two_year_recid" not in df.columns:
        raise ValueError("COMPAS dataset missing label column (expected two_year_recid).")

    y = pd.to_numeric(df["two_year_recid"], errors="coerce").fillna(0).astype(int).to_numpy()

    # Sensitive attribute
    if "race" in df.columns:
        s = df["race"].astype(str).str.lower().to_numpy()
    else:
        s = np.array(["group"] * len(df), dtype=str)

    drop_cols = [c for c in ["two_year_recid", "name"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Drop columns entirely missing (prevents warnings like violent_recid all-NaN)
    X = X.dropna(axis=1, how="all")

    return _split_and_transform(X, y, s)


def preprocess_german_credit(df: pd.DataFrame):
    df = _clean_strings(df)

    label_col = None
    for cand in ["target", "class"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("German Credit dataset missing label column (expected target or class).")

    y_raw = df[label_col].astype(str).str.lower()
    y = y_raw.map(lambda v: 1 if "good" in v or v == "1" else 0).astype(int).to_numpy()

    # Sensitive attribute: personal_status -> male/female
    if "personal_status" in df.columns:
        ps = df["personal_status"].astype(str).str.lower()
        s = ps.map(lambda v: "female" if "female" in v else "male").astype(str).to_numpy()
    elif "sex" in df.columns:
        s = df["sex"].astype(str).str.lower().to_numpy()
    elif "age" in df.columns:
        s = pd.to_numeric(df["age"], errors="coerce").fillna(0).map(lambda a: "age>=25" if a >= 25 else "age<25").astype(str).to_numpy()
    else:
        s = np.array(["group"] * len(df), dtype=str)

    X = df.drop(columns=[label_col])
    return _split_and_transform(X, y, s)