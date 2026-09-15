"""Batch documents through the encoder and hand back id->vector rows.

Batching is the difference between a research loop that finishes and one
that doesn't: one forward pass over 64 documents instead of 64 passes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from research.common.schema import Document
from research.embed.encoder import DocumentEncoder


@dataclass
class EmbeddedDoc:
    doc_id: str
    vector: list[float]


def _embed_text(doc: Document) -> str:
    # Title carries most of the signal; prepend it, then a bounded body.
    return doc.title + ". " + doc.body[:1000]


class EmbeddingService:
    def __init__(self, batch_size: int = 64) -> None:
        self.encoder = DocumentEncoder()
        self.batch_size = batch_size

    def embed(self, docs: list[Document]) -> list[EmbeddedDoc]:
        out: list[EmbeddedDoc] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            texts = [_embed_text(d) for d in batch]
            vecs: torch.Tensor = self.encoder.encode(texts)
            for d, v in zip(batch, vecs):
                out.append(EmbeddedDoc(doc_id=d.doc_id, vector=v.tolist()))
        return out
