"""A small PyTorch text encoder producing L2-normalized document vectors.

We wrap a pretrained transformer, mean-pool its token outputs into one
vector per document, and normalize so that a dot product IS cosine
similarity. The rest of the pipeline only ever sees fixed-width vectors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


class DocumentEncoder:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()   # inference only — no dropout, no grad graph

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average token vectors, ignoring padding via the attention mask."""
        mask = mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)   # never divide by zero
        return summed / counts

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**toks)
        pooled = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
        # L2-normalize so cosine similarity == dot product.
        return F.normalize(pooled, p=2, dim=1)


if __name__ == "__main__":
    enc = DocumentEncoder()
    v = enc.encode(["Acme beats earnings", "Acme misses earnings"])
    print(v.shape, float(v[0] @ v[1]))   # (2, 384) and their cosine
