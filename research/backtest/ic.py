"""The information coefficient: does the factor rank-predict returns?

IC is the Spearman rank correlation, per day, between the factor and the
realized forward return. Positive and stable across days means the factor
carries predictive information. This is the first number a researcher
looks at — before any P&L — because it measures the *signal*, not a
particular trading rule laid on top of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation = Pearson correlation of the ranks."""
    if len(a) < 2:
        return np.nan
    ra = a.rank()
    rb = b.rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan          # no dispersion -> correlation undefined
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_ic(factor_df: pd.DataFrame) -> pd.Series:
    """One IC per date: rank-corr(factor, fwd_ret) across that day's names."""
    ics = {}
    for date, g in factor_df.groupby("date"):
        ics[date] = _spearman(g["factor"], g["fwd_ret"])
    return pd.Series(ics, name="ic").dropna().sort_index()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean IC, its volatility, and the IC information ratio (mean/std,
    annualized by sqrt of periods per year)."""
    if ic.empty:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_days": 0}
    mean_ic = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    ir = 0.0 if ic_std == 0 else mean_ic / ic_std * np.sqrt(252)
    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ir,
        "n_days": int(len(ic)),
    }
