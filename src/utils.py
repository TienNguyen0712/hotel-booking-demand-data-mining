from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory if not exists and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """Reproducibility for python/random/numpy."""
    random.seed(seed)
    np.random.seed(seed)


def load_csv(path: PathLike, **kwargs) -> pd.DataFrame:
    """Load CSV with pandas."""
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: PathLike, index: bool = False, **kwargs) -> None:
    """Save dataframe to CSV. Creates parent dir if needed."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index, **kwargs)


_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def month_str_to_num(month: str) -> Optional[int]:
    """Convert month name to number. Returns None if unknown."""
    if month is None:
        return None
    month = str(month).strip()
    return _MONTH_MAP.get(month)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division to avoid ZeroDivisionError."""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default
