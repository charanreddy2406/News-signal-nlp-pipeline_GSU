"""Two-layer dedup: exact identity, then near-duplicate by shingle overlap.

Newsflow is heavily duplicated — the same story crosses ten wires in a
minute. If we let each copy through, the signal is counted ten times and
the backtest is a lie. We suppress duplicates at the door, before any LLM
or embedding cost is paid.
"""
from __future__ import annotations

from research.common.schema import Document, _normalize_text


def _shingles(text: str, k: int = 5) -> set[str]:
    """k-word shingles: overlapping word windows. Two stories that share
    most shingles are the same story with light edits."""
    words = _normalize_text(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class Deduper:
    def __init__(self, near_threshold: float = 0.8) -> None:
        self._seen_ids: set[str] = set()          # exact identity
        self._shingle_cache: list[set[str]] = []  # for near-dup checks
        self._near_threshold = near_threshold

    def is_duplicate(self, doc: Document) -> bool:
        # Layer 1: exact content hash — free, catches verbatim reprints.
        if doc.doc_id in self._seen_ids:
            return True
        # Layer 2: near-duplicate — same story, reworded headline/lede.
        sh = _shingles(doc.title + " " + doc.body)
        for prior in self._shingle_cache:
            if jaccard(sh, prior) >= self._near_threshold:
                return True
        self._seen_ids.add(doc.doc_id)
        self._shingle_cache.append(sh)
        return False
