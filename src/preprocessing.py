from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import month_str_to_num


@dataclass
class PreprocessConfig:
    target: str = "is_canceled"
    fill_country: str = "Unknown"
    fill_categorical: str = "Unknown"
    fill_agent_company: int = 0
    drop_cols: Optional[List[str]] = None


def clean_missing_values(df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Handle missing values with simple, explainable rules."""
    out = df.copy()

    if "children" in out.columns:
        out["children"] = out["children"].fillna(0)

    if "country" in out.columns:
        out["country"] = out["country"].fillna(cfg.fill_country)

    # agent/company often missing a lot -> fill with 0 (unknown id)
    for col in ["agent", "company"]:
        if col in out.columns:
            out[col] = out[col].fillna(cfg.fill_agent_company)

    # other categoricals: fill "Unknown"
    cat_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        out[col] = out[col].fillna(cfg.fill_categorical)

    return out


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are logically invalid or clearly erroneous."""
    out = df.copy()

    # Invalid: no guests
    guest_cols = [c for c in ["adults", "children", "babies"] if c in out.columns]
    if guest_cols:
        total = out[guest_cols].sum(axis=1)
        out = out[total > 0]

    # Optional: remove negative adr (if exists)
    if "adr" in out.columns:
        out = out[out["adr"] >= 0]

    return out.reset_index(drop=True)


def handle_outliers_iqr(df: pd.DataFrame, cols: Iterable[str], k: float = 1.5) -> pd.DataFrame:
    """Clip outliers using IQR rule (winsorization)."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out[col] = s.clip(lo, hi)
    return out


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful features for EDA/ML."""
    out = df.copy()

    # total guests / nights
    if set(["adults", "children", "babies"]).issubset(out.columns):
        out["total_guests"] = out["adults"] + out["children"] + out["babies"]

    if set(["stays_in_weekend_nights", "stays_in_week_nights"]).issubset(out.columns):
        out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]

    # build a month number from month name if present
    if "arrival_date_month" in out.columns:
        out["arrival_month_num"] = out["arrival_date_month"].apply(month_str_to_num)

    # create a sortable date key (monthly). Many versions of this dataset lack arrival day -> monthly is safe.
    if set(["arrival_date_year", "arrival_month_num"]).issubset(out.columns):
        out["arrival_year_month"] = (
            out["arrival_date_year"].astype(int).astype(str)
            + "-"
            + out["arrival_month_num"].astype(int).astype(str).str.zfill(2)
        )
        out["arrival_year_month"] = pd.to_datetime(out["arrival_year_month"] + "-01")

    return out


def build_time_series(df: pd.DataFrame, by: str = "arrival_year_month") -> pd.DataFrame:
    """
    Build monthly time-series table for time series notebook.
    Output columns:
      date, total_bookings, canceled_bookings, cancel_rate, avg_adr, hotel_type(optional)
    """
    if by not in df.columns:
        raise ValueError(f"Missing '{by}'. Ensure feature_engineering() created it.")

    base_cols = [by]
    if "hotel" in df.columns:
        base_cols.append("hotel")

    # aggregate
    group_cols = base_cols
    agg = {
        "is_canceled": ["count", "sum"],
    }
    if "adr" in df.columns:
        agg["adr"] = ["mean"]

    g = df.groupby(group_cols).agg(agg)
    g.columns = ["_".join([c for c in tup if c]) for tup in g.columns.to_flat_index()]
    g = g.reset_index()

    g = g.rename(columns={
        by: "date",
        "is_canceled_count": "total_bookings",
        "is_canceled_sum": "canceled_bookings",
        "adr_mean": "avg_adr",
    })
    g["cancel_rate"] = np.where(g["total_bookings"] > 0, g["canceled_bookings"] / g["total_bookings"], 0.0)

    # hotel is optional
    if "hotel" in g.columns:
        g = g.rename(columns={"hotel": "hotel_type"})

    g = g.sort_values(["hotel_type", "date"] if "hotel_type" in g.columns else ["date"]).reset_index(drop=True)
    return g


def build_model_matrix(
    df: pd.DataFrame,
    cfg: PreprocessConfig = PreprocessConfig(),
    scale_numeric: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[ColumnTransformer]]:
    """
    Prepare X, y for classification/regression models.
    - One-hot encode categoricals
    - Optional scaling for numeric columns
    Returns: X, y, fitted preprocessor (ColumnTransformer) if used
    """
    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' not found.")

    drop_cols = set(cfg.drop_cols or [])
    drop_cols.add(cfg.target)

    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[cfg.target].astype(int)

    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if num_cols:
        if scale_numeric:
            transformers.append(("num", StandardScaler(), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_arr = preprocessor.fit_transform(X_raw)

    # build feature names
    feature_names: List[str] = []
    if cat_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)
    if num_cols:
        feature_names.extend(num_cols)

    X = pd.DataFrame(X_arr, columns=feature_names)
    return X, y, preprocessor
