"""A deterministic, model-free extractor.

Two uses: a baseline to measure the LLM against, and a fallback when the
LLM is unavailable or its output failed validation. A rules extractor is
transparent and free — you always want a floor you understand.
"""
from __future__ import annotations

from research.common.schema import Document
from research.extract.signal import ExtractedSignal

_BULLISH = {"beats", "surges", "raises", "upgrade", "record", "approval"}
_BEARISH = {"misses", "plunges", "cuts", "downgrade", "probe", "recall", "lawsuit"}


def extract_rules(doc: Document) -> ExtractedSignal:
    text = (doc.title + " " + doc.body).lower()
    pos = sum(w in text for w in _BULLISH)
    neg = sum(w in text for w in _BEARISH)
    total = pos + neg
    sentiment = 0.0 if total == 0 else (pos - neg) / total
    return ExtractedSignal(
        doc_id=doc.doc_id,
        primary_ticker=doc.tickers[0] if doc.tickers else "",
        event_type="other",
        sentiment=sentiment,
        # No hits -> low confidence; the more lexicon matches, the surer.
        confidence=min(1.0, total / 5.0),
    )
