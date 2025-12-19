from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def save_figure(fig: plt.Figure, filename: str, out_dir: str = "reports/figures", dpi: int = 200) -> Path:
    """Save figure to reports/figures with consistent settings."""
    out_path = ensure_dir(out_dir) / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


# ----------- EDA plots -----------

def plot_missing_values(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    miss = df.isna().sum().sort_values(ascending=False).head(top_n)
    fig = plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Missing Values by Column")
    plt.ylabel("Missing count")
    return fig


def plot_cancel_rate(df: pd.DataFrame, target: str = "is_canceled") -> plt.Figure:
    counts = df[target].value_counts().sort_index()
    fig = plt.figure()
    plt.bar(["Not canceled (0)", "Canceled (1)"], counts.values)
    plt.title("Cancellation Distribution")
    plt.ylabel("Count")
    return fig


def plot_hist(df: pd.DataFrame, col: str, bins: int = 30) -> plt.Figure:
    fig = plt.figure()
    plt.hist(df[col].dropna().values, bins=bins)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    return fig


def plot_box(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    fig = plt.figure()
    groups = [g[y].dropna().values for _, g in df.groupby(x)]
    labels = [str(k) for k in df[x].dropna().unique()]
    plt.boxplot(groups, labels=labels)
    plt.title(f"Boxplot {y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    return fig


def plot_corr_heatmap(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> plt.Figure:
    data = df[cols].copy() if cols is not None else df.select_dtypes(include=[np.number]).copy()
    corr = data.corr(numeric_only=True)

    fig = plt.figure()
    plt.imshow(corr.values)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (numeric)")
    return fig


# ----------- Classification plots -----------

def plot_confusion_matrix(cm: np.ndarray, labels=("0", "1")) -> plt.Figure:
    fig = plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    return fig


def plot_model_comparison(df_metrics: pd.DataFrame, metric: str = "f1") -> plt.Figure:
    fig = plt.figure()
    plt.bar(df_metrics["model"].astype(str), df_metrics[metric].astype(float))
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Model Comparison by {metric.upper()}")
    plt.ylabel(metric)
    return fig


# ----------- Clustering plots -----------

def plot_elbow(ks: Sequence[int], inertias: Sequence[float]) -> plt.Figure:
    fig = plt.figure()
    plt.plot(list(ks), list(inertias), marker="o")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    return fig


def plot_silhouette(ks: Sequence[int], scores: Sequence[float]) -> plt.Figure:
    fig = plt.figure()
    plt.plot(list(ks), list(scores), marker="o")
    plt.title("Silhouette Scores by k")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    return fig


def plot_clusters_2d(X2d: np.ndarray, labels: np.ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.scatter(X2d[:, 0], X2d[:, 1], c=labels)
    plt.title("Clusters (2D projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    return fig


# ----------- Association rules -----------

def plot_top_rules(rules_df: pd.DataFrame, metric: str = "lift", top_n: int = 10) -> plt.Figure:
    top = rules_df.sort_values(metric, ascending=False).head(top_n).copy()
    # label = antecedents -> consequents (stringify)
    def _to_str(x):
        return ", ".join(list(x)) if not isinstance(x, str) else x

    labels = [
        f"{_to_str(a)} → {_to_str(c)}"
        for a, c in zip(top["antecedents"], top["consequents"])
    ]

    fig = plt.figure()
    plt.barh(range(len(top)), top[metric].astype(float).values)
    plt.yticks(range(len(top)), labels)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Rules by {metric}")
    plt.xlabel(metric)
    return fig


# ----------- Time series -----------

def plot_time_series(ts_df: pd.DataFrame, y: str = "total_bookings", x: str = "date") -> plt.Figure:
    fig = plt.figure()
    plt.plot(ts_df[x], ts_df[y])
    plt.title(f"Time Series: {y}")
    plt.xlabel("Date")
    plt.ylabel(y)
    return fig


def plot_cancel_rate_ts(ts_df: pd.DataFrame, x: str = "date", y: str = "cancel_rate") -> plt.Figure:
    fig = plt.figure()
    plt.plot(ts_df[x], ts_df[y])
    plt.title("Cancellation Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cancel rate")
    return fig
