"""Build the prediction target: forward returns per ticker per day.

The signal predicts *the future*, so the label must be a return that
happens strictly after the news. This module produces, for each ticker
and date, the next-day forward return we'll try to predict.
"""
from __future__ import annotations

import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """prices: columns [date, ticker, close]. Returns [date, ticker, fwd_ret]
    where fwd_ret is the close-to-close return over the next `horizon_days`,
    stamped on the date the *information* would be acted on."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["fwd_close"] = df.groupby("ticker")["close"].shift(-horizon_days)
    df["fwd_ret"] = df["fwd_close"] / df["close"] - 1.0
    # Drop the tail rows where no forward price exists yet.
    return df.dropna(subset=["fwd_ret"])[["date", "ticker", "fwd_ret"]]
