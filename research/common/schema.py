"""Canonical document model for the news-to-signal pipeline.

Every source — a newswire, a filing scraper, a press-release feed — is
normalized into this one shape at the boundary. Extraction, embedding,
alignment, and backtest all downstream code speak `Document`, never a
source-specific dict. One schema is what keeps six stages from drifting.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase for hashing/near-dup comparison.

    We hash the *normalized* body, not the raw one, so a wire and its
    reprint with different spacing collapse to the same identity.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(title: str, body: str) -> str:
    norm = _normalize_text(title) + "\x1f" + _normalize_text(body)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass
class Document:
    """One news item / filing, source-agnostic.

    `published_at` is the exchange-relevant event time in UTC — the instant
    the market could first have known this. Everything about lookahead
    hinges on that field, so it is required and always UTC.
    """

    doc_id: str                 # stable id = content_hash
    source: str                 # 'newswire' | 'filing' | 'press'
    tickers: list[str]          # symbols the story references
    title: str
    body: str
    published_at: datetime      # event time, UTC, tz-aware
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def build(source: str, tickers: list[str], title: str, body: str,
              published_at: datetime) -> "Document":
        if published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware UTC")
        return Document(
            doc_id=content_hash(title, body),
            source=source,
            tickers=[t.upper() for t in tickers],
            title=title,
            body=body,
            published_at=published_at.astimezone(timezone.utc),
        )
