"""Dependency-light metrics for the pinned upstream lens evaluations."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable


def token_variants(tokenizer: Any, text: str) -> list[int]:
    """Return unique single-token IDs for common spacing/case variants."""

    variants = {text, text.lower(), text.capitalize()}
    token_ids: set[int] = set()
    for value in variants:
        for candidate in (value, f" {value}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
    return sorted(token_ids)


def best_rank(logits_by_layer: Iterable[Any], token_ids: Iterable[int]) -> int | None:
    """Return the best one-indexed rank over layers and token variants."""

    identifiers = tuple(token_ids)
    best: int | None = None
    for logits in logits_by_layer:
        for token_id in identifiers:
            score = logits[token_id]
            rank = int((logits > score).sum().item()) + 1
            best = rank if best is None else min(best, rank)
    return best


def pass_at(records: Iterable[dict[str, Any]], method: str, k: int) -> float:
    """Compute the official item-macro pass@k metric.

    Each intermediate counts in its item's denominator. An intermediate with
    no single-token variant is therefore a failure, not a silently dropped row.
    """

    by_item: dict[str, list[int | None]] = defaultdict(list)
    for record in records:
        by_item[str(record["item"])].append(record[method])
    if not by_item:
        return 0.0
    item_scores = [
        sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
        for ranks in by_item.values()
    ]
    return statistics.fmean(item_scores)
