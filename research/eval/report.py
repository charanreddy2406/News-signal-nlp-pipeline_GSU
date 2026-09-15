"""Render a verdict for a human. The harness returns structure; this makes
it readable in a terminal or a research log."""
from __future__ import annotations

from research.eval.harness import Verdict


def format_verdict(v: Verdict) -> str:
    lines = [f"VERDICT: {v.decision.upper()} — {v.reason}", "-" * 48]
    for key in ("mean_ic", "ic_ir", "sharpe", "ann_return", "hit_rate", "n_days"):
        if key in v.metrics:
            lines.append(f"  {key:<12} {v.metrics[key]:.4f}")
    return "\n".join(lines)
