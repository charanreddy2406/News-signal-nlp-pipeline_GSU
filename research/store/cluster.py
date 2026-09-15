"""Collapse an event's duplicate coverage using retrieval.

Near-dup ingest kills verbatim and lightly-edited reprints. This catches
the subtler case: ten *distinct* articles about the same event, written
independently. They cluster tightly in embedding space; we group them so
the event contributes one signal, not ten.
"""
from __future__ import annotations

from research.store.vector_store import VectorStore


def event_cluster(
    store: VectorStore,
    seed_vector: list[float],
    radius: float = 0.85,
    k: int = 25,
) -> list[str]:
    """doc_ids whose cosine to the seed is >= radius: one story-event."""
    hits = store.search(seed_vector, k=k)
    return [h.doc_id for h in hits if h.score >= radius]
