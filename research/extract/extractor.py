"""LLM extraction with a hard validation boundary.

The model is prompted to return JSON matching ExtractedSignal. We never
trust that it did. `validate` clamps ranges, checks the event vocabulary,
and rejects anything unrepairable — so a hallucinated event type or a
sentiment of 7.4 becomes a dropped signal, not a corrupted factor.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from research.common.schema import Document
from research.extract.signal import ExtractedSignal, VALID_EVENTS

PROMPT = """You are a financial-news analyst. Read the article and return ONLY JSON:
{{"event_type": <one of {events}>, "sentiment": <float -1..1>, "confidence": <float 0..1>}}
sentiment: -1 very bearish for the primary company, +1 very bullish.
Article title: {title}
Article body: {body}
"""


def build_prompt(doc: Document) -> str:
    return PROMPT.format(
        events=sorted(VALID_EVENTS),
        title=doc.title,
        body=doc.body[:4000],   # bound the context; ledes carry the signal
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def validate(doc: Document, raw: dict[str, Any]) -> Optional[ExtractedSignal]:
    """Turn a model dict into a signal, or None if it can't be repaired."""
    event = str(raw.get("event_type", "")).lower().strip()
    if event not in VALID_EVENTS:
        event = "other"                       # repair: unknown -> 'other'
    try:
        sentiment = _clamp(float(raw["sentiment"]), -1.0, 1.0)
        confidence = _clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)
    except (KeyError, TypeError, ValueError):
        return None                           # unrepairable: no usable number
    ticker = doc.tickers[0] if doc.tickers else ""
    return ExtractedSignal(
        doc_id=doc.doc_id,
        primary_ticker=ticker,
        event_type=event,
        sentiment=sentiment,
        confidence=confidence,
    )


def extract(
    doc: Document,
    complete: Callable[[str], str],
) -> Optional[ExtractedSignal]:
    """`complete` is any text->text LLM call. We parse, then validate.

    A JSON parse error is not fatal — it's just an extraction miss for this
    document, handled the same as an unrepairable field: return None and
    let the caller fall back or skip.
    """
    prompt = build_prompt(doc)
    try:
        raw = json.loads(complete(prompt))
    except (json.JSONDecodeError, TypeError):
        return None
    return validate(doc, raw)
