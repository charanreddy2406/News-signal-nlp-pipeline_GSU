"""The structured signal we pull from each document.

This is the contract between messy language and clean math. Extraction
emits exactly these fields; alignment and backtest consume exactly these
fields. Anything the model returns that doesn't fit is rejected or
repaired here, never passed downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventType = Literal[
    "earnings", "guidance", "mna", "litigation",
    "management_change", "product", "macro", "other",
]

VALID_EVENTS = set(EventType.__args__)  # type: ignore[attr-defined]


@dataclass
class ExtractedSignal:
    doc_id: str
    primary_ticker: str
    event_type: str          # one of VALID_EVENTS
    sentiment: float         # bounded to [-1.0, +1.0]
    confidence: float        # [0.0, 1.0]; extractor's self-reported certainty

    def is_actionable(self, min_confidence: float = 0.35) -> bool:
        """A signal we'd let into the factor: known event, real ticker,
        confidence above the floor."""
        return (
            self.event_type in VALID_EVENTS
            and bool(self.primary_ticker)
            and self.confidence >= min_confidence
        )
