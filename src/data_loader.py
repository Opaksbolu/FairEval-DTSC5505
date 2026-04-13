from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p.resolve()}")
    return pd.read_csv(p)


def load_adult() -> pd.DataFrame:
    return _csv("datasets/adult.csv")


def load_compas() -> pd.DataFrame:
    return _csv("datasets/compas.csv")


def load_german_credit() -> pd.DataFrame:
    return _csv("datasets/german_credit.csv")


def load_crows_pairs() -> pd.DataFrame:
    return _csv("datasets/crows_pairs.csv")


def load_bbq_optional() -> Optional[pd.DataFrame]:
    """
    BBQ is optional for Milestone 5.
    If datasets/bbq.csv exists, return it; else return None (do not crash).
    """
    p = Path("datasets/bbq.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)