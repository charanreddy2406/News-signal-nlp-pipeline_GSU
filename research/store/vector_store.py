"""A small, exact in-memory vector store.

Real research runs on FAISS / a managed vector DB, but the *interface* is
what matters: upsert by a stable id, search top-k by cosine. Because our
vectors are already L2-normalized, cosine is a single matrix-vector dot,
so exact search over a day of news is trivially fast and 100% recall —
the right default before you reach for an approximate index.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Hit:
    doc_id: str
    score: float   # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}          # doc_id -> row index
        self._matrix = torch.empty((0, dim))

    def upsert(self, doc_id: str, vector: list[float]) -> None:
        """Insert or overwrite by doc_id. Re-embedding a document updates
        its row in place instead of duplicating it — the same idempotency
        the content hash gives us upstream, preserved in the index."""
        row = torch.tensor(vector).unsqueeze(0)
        if doc_id in self._pos:
            self._matrix[self._pos[doc_id]] = row
            return
        self._pos[doc_id] = len(self._ids)
        self._ids.append(doc_id)
        self._matrix = torch.cat([self._matrix, row], dim=0)

    def search(self, query: list[float], k: int = 10) -> list[Hit]:
        if not self._ids:
            return []
        q = torch.tensor(query)
        # Vectors are unit-norm, so this dot product is cosine similarity.
        scores = self._matrix @ q
        k = min(k, len(self._ids))
        top = torch.topk(scores, k)
        return [
            Hit(self._ids[i], float(s))
            for s, i in zip(top.values, top.indices)
        ]
