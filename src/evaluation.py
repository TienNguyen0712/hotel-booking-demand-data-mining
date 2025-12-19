from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)


# ---------- Classification ----------

def eval_classification(y_true, y_pred) -> Dict[str, float]:
    """Return standard classification metrics (rubric-aligned)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def get_confusion(y_true, y_pred) -> np.ndarray:
    """Confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    results = {
      "LogReg": {"accuracy":..., "precision":..., "recall":..., "f1":...},
      ...
    }
    """
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    # sort by f1 then accuracy
    if "f1" in df.columns:
        df = df.sort_values(["f1", "accuracy"], ascending=False)
    return df


# ---------- Clustering ----------

def eval_clustering(X, labels) -> Dict[str, float]:
    """Return silhouette score (rubric)."""
    # Need at least 2 clusters and not all noise
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"silhouette": float("nan")}
    return {"silhouette": float(silhouette_score(X, labels))}


def silhouette_by_k(X, k_list: List[int], kmeans_factory) -> pd.DataFrame:
    """
    Helper to compute silhouette score for different k.
    kmeans_factory: function(k) -> fitted KMeans-like object with .fit_predict(X)
    """
    rows = []
    for k in k_list:
        model = kmeans_factory(k)
        labels = model.fit_predict(X)
        score = eval_clustering(X, labels)["silhouette"]
        inertia = getattr(model, "inertia_", np.nan)
        rows.append({"k": k, "silhouette": score, "inertia": float(inertia)})
    return pd.DataFrame(rows)


# ---------- Association Rules ----------

def summarize_rules(rules_df: pd.DataFrame, top_n: int = 10, sort_by: str = "lift") -> pd.DataFrame:
    """
    rules_df expected columns: antecedents, consequents, support, confidence, lift
    """
    needed = {"support", "confidence", "lift"}
    if not needed.issubset(set(rules_df.columns)):
        raise ValueError(f"rules_df must include columns: {needed}")

    out = rules_df.sort_values(sort_by, ascending=False).head(top_n).copy()
    return out.reset_index(drop=True)


def filter_rules(
    rules_df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.3,
    min_lift: float = 1.1
) -> pd.DataFrame:
    out = rules_df.copy()
    if "support" in out.columns:
        out = out[out["support"] >= min_support]
    if "confidence" in out.columns:
        out = out[out["confidence"] >= min_confidence]
    if "lift" in out.columns:
        out = out[out["lift"] >= min_lift]
    return out.reset_index(drop=True)


# ---------- Time Series (lightweight) ----------

def train_test_split_ts(df: pd.DataFrame, date_col: str = "date", test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series by time order."""
    out = df.sort_values(date_col).reset_index(drop=True)
    n = len(out)
    cut = int(np.floor((1 - test_ratio) * n))
    return out.iloc[:cut].copy(), out.iloc[cut:].copy()


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))
