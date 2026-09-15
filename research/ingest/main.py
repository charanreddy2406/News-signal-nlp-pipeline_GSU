"""Ingest: normalize raw source records into Documents and drop duplicates.

The output is a clean, deduplicated document stream — the only thing the
rest of the pipeline is allowed to see.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Iterator

from research.common.schema import Document
from research.ingest.dedup import Deduper


def ingest(raw_records: Iterable[dict]) -> Iterator[Document]:
    deduper = Deduper()
    for rec in raw_records:
        published = datetime.fromisoformat(rec["published_at"])
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        doc = Document.build(
            source=rec["source"],
            tickers=rec.get("tickers", []),
            title=rec["title"],
            body=rec["body"],
            published_at=published,
        )
        if deduper.is_duplicate(doc):
            continue
        yield doc
