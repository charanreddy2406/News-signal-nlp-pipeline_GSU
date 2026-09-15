"""As-of alignment of signals to forward returns with a hard embargo.

Each signal fires at a `published_at` instant. We map it to a trading date
using an embargo rule: news public before the cutoff trades that day; news
after the cutoff trades the next day. Then we join to that date's forward
return. Get this join wrong and you have lookahead — the single most common
way a research backtest lies.
"""
from __future__ import annotations

from datetime import datetime, time, timezone

import pandas as pd

from research.extract.signal import ExtractedSignal

# News published at/after this UTC time trades the NEXT session, not today's.
CUTOFF = time(20, 0)   # 20:00 UTC ~ US market close


def trade_date(published_at: datetime) -> pd.Timestamp:
    """Map an event time to the trading date its signal is actionable on."""
    pub = published_at.astimezone(timezone.utc)
    d = pd.Timestamp(pub.date())
    if pub.time() >= CUTOFF:
        d = d + pd.Timedelta(days=1)   # embargo: too late for today's move
    return d


def align_signals(
    signals: list[ExtractedSignal],
    published_at: dict[str, datetime],
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join signals to forward returns on (trade_date, ticker).

    targets: [date, ticker, fwd_ret]. Returns one row per actionable signal
    with its sentiment, event, and the return it is trying to predict."""
    rows = []
    for s in signals:
        if not s.is_actionable():
            continue
        td = trade_date(published_at[s.doc_id])
        rows.append({
            "date": td,
            "ticker": s.primary_ticker,
            "sentiment": s.sentiment,
            "event_type": s.event_type,
            "confidence": s.confidence,
        })
    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        return sig_df
    # Inner join: a signal with no matching forward return is unusable.
    return sig_df.merge(targets, on=["date", "ticker"], how="inner")
