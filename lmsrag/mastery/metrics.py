"""Prediction metrics (review C2): AUC, ACC, RMSE, log-loss, ECE, and the reliability table behind the diagram."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with ``bins`` equal-width bins on [0, 1]."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    conf = np.bincount(b, weights=p, minlength=bins)
    acc = np.bincount(b, weights=y, minlength=bins)
    m = n > 0
    return float(np.abs(acc[m] - conf[m]).sum() / len(y))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    b = np.minimum((p * bins).astype(int), bins - 1)
    d = pd.DataFrame({"bin": b, "y": y, "p": p}).groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"),
                                                                     frac_correct=("y", "mean"))
    return d.reset_index()


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return {"n": int(len(y)), "AUC": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
            "ACC": float(((p >= 0.5) == (y == 1)).mean()), "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
            "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "ECE": ece(y, p)}
