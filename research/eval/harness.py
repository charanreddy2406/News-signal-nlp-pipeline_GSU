"""One command that runs a hypothesis end-to-end and scores it.

This is the paper-to-production loop compressed into a function: raw news
in, a verdict out. A researcher proposes an idea (an extractor, an
embedding choice, a horizon), runs the harness, and gets back the IC, the
Sharpe, and a rule-based verdict — accept, refine, or kill — instead of a
week of ad-hoc notebook archaeology. Fast, repeatable measurement is the
whole discipline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from research.common.schema import Document
from research.ingest.main import ingest
from research.extract.extractor import extract
from research.extract.fallback import extract_rules
from research.extract.signal import ExtractedSignal
from research.align.align import align_signals
from research.align.factor import aggregate_daily, cross_sectional_factor
from research.backtest.ic import daily_ic, ic_summary
from research.backtest.portfolio import backtest, perf_summary


@dataclass
class Thresholds:
    """The bar a signal must clear. These encode the research team's
    standard for 'worth pursuing' — tune them, but tune them deliberately."""
    min_mean_ic: float = 0.02      # a real but modest edge
    min_ic_ir: float = 0.5         # consistent, not a lucky sample
    min_sharpe: float = 0.5        # survives after costs


@dataclass
class Verdict:
    decision: str                  # 'accept' | 'refine' | 'kill'
    reason: str
    metrics: dict


def _extract_all(
    docs: list[Document],
    complete: Optional[Callable[[str], str]],
) -> tuple[list[ExtractedSignal], dict[str, datetime]]:
    """Run extraction over every doc, falling back to rules when the LLM
    is absent or its output failed validation. The pipeline never stalls
    on a single extraction miss."""
    signals: list[ExtractedSignal] = []
    published: dict[str, datetime] = {}
    for d in docs:
        sig = extract(d, complete) if complete else None
        if sig is None:
            sig = extract_rules(d)         # deterministic floor
        signals.append(sig)
        published[d.doc_id] = d.published_at
    return signals, published


def run_eval(
    raw_records: list[dict],
    prices: "pd.DataFrame",
    targets: "pd.DataFrame",
    complete: Optional[Callable[[str], str]] = None,
    thresholds: Thresholds = Thresholds(),
) -> Verdict:
    # 1) ingest + dedup  2) extract  3) align  4) factor  5) score.
    docs = list(ingest(raw_records))
    signals, published = _extract_all(docs, complete)
    aligned = align_signals(signals, published, targets)
    if aligned.empty:
        return Verdict("kill", "no signals aligned to a return", {})

    factor = cross_sectional_factor(aggregate_daily(aligned))
    ic = ic_summary(daily_ic(factor))
    perf = perf_summary(backtest(factor))
    metrics = {**ic, **perf}
    return decide(metrics, thresholds)


def decide(metrics: dict, t: Thresholds) -> Verdict:
    """The verdict rubric — three tiers, checked in order."""
    mean_ic = metrics.get("mean_ic", 0.0)
    ic_ir = metrics.get("ic_ir", 0.0)
    sharpe = metrics.get("sharpe", 0.0)

    # ACCEPT: real edge, consistent, and it survives costs.
    if mean_ic >= t.min_mean_ic and ic_ir >= t.min_ic_ir and sharpe >= t.min_sharpe:
        return Verdict("accept", "clears IC, consistency, and net-Sharpe bars", metrics)

    # KILL: no directional edge at all — refining won't rescue a zero.
    if mean_ic <= 0:
        return Verdict("kill", "non-positive mean IC: no predictive signal", metrics)

    # REFINE: there's a pulse, but it misses a bar — worth another iteration.
    return Verdict(
        "refine",
        "positive IC but below a threshold (consistency or net-Sharpe)",
        metrics,
    )
