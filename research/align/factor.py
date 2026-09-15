"""From aligned signals to a tradeable daily factor.

Two reductions. First, collapse multiple signals on the same ticker-day
into one confidence-weighted score. Second, make scores comparable across
tickers on a given day via winsorization + cross-sectional ranking — a
factor is only meaningful *relative to the cross-section that day*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_daily(aligned: pd.DataFrame) -> pd.DataFrame:
    """Confidence-weighted mean sentiment per (date, ticker)."""
    if aligned.empty:
        return aligned

    def _wmean(g: pd.DataFrame) -> float:
        w = g["confidence"].to_numpy()
        x = g["sentiment"].to_numpy()
        return float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())

    agg = (
        aligned.groupby(["date", "ticker"])
        .apply(lambda g: pd.Series({
            "raw_signal": _wmean(g),
            "fwd_ret": g["fwd_ret"].iloc[0],   # same for a ticker-day
        }))
        .reset_index()
    )
    return agg


def _winsorize(s: pd.Series, limit: float = 0.02) -> pd.Series:
    lo, hi = s.quantile(limit), s.quantile(1 - limit)
    return s.clip(lo, hi)


def cross_sectional_factor(daily: pd.DataFrame) -> pd.DataFrame:
    """Winsorize then rank-standardize each day's signals to [-1, 1]."""
    if daily.empty:
        return daily
    out = []
    for date, g in daily.groupby("date"):
        g = g.copy()
        g["w_signal"] = _winsorize(g["raw_signal"])
        # Rank within the day, then map to [-1, 1]. A factor is a bet on
        # relative order, not absolute magnitude.
        r = g["w_signal"].rank(method="average")
        n = len(g)
        g["factor"] = 0.0 if n <= 1 else (r - 1) / (n - 1) * 2.0 - 1.0
        out.append(g)
    return pd.concat(out, ignore_index=True)
