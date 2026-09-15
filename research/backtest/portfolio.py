"""A minimal long/short decile backtest.

Each day: rank the cross-section by factor, go long the top decile and
short the bottom, equal-weight, dollar-neutral. The daily return is the
mean forward return of the longs minus that of the shorts, less a simple
turnover-based cost. This is the P&L companion to the IC — same signal,
now expressed as a tradeable rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_PER_TURN = 0.0010   # 10 bps round-trip cost assumption


def _daily_ls_return(g: pd.DataFrame, decile: float) -> float:
    g = g.sort_values("factor")
    n = len(g)
    if n < 10:
        return 0.0                      # too few names to form deciles
    k = max(1, int(n * decile))
    shorts = g.head(k)["fwd_ret"].mean()   # lowest factor
    longs = g.tail(k)["fwd_ret"].mean()    # highest factor
    gross = longs - shorts
    # Assume we fully rebalance both legs each day: 2 legs turned over.
    cost = 2.0 * COST_PER_TURN
    return float(gross - cost)


def backtest(factor_df: pd.DataFrame, decile: float = 0.1) -> pd.Series:
    """Daily dollar-neutral long/short return series."""
    rets = {}
    for date, g in factor_df.groupby("date"):
        rets[date] = _daily_ls_return(g, decile)
    return pd.Series(rets, name="ret").sort_index()


def perf_summary(ret: pd.Series) -> dict[str, float]:
    if ret.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "hit_rate": 0.0}
    ann_return = float(ret.mean() * 252)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else 0.0
    sharpe = 0.0 if ann_vol == 0 else ann_return / ann_vol
    hit_rate = float((ret > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
    }
