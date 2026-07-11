"""Intention-to-treat and omitted-subset paired outcome analysis."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from jlens_panel.experiment import Condition, OutcomeRecord

DEFAULT_COMPARISONS: tuple[tuple[Condition, Condition], ...] = (
    (Condition.JLENS_TARGETED, Condition.BEST_NON_J),
    (Condition.JLENS_TARGETED, Condition.GENERIC),
    (Condition.BEST_NON_J, Condition.GENERIC),
    (Condition.ORACLE, Condition.GENERIC),
    (Condition.ORACLE, Condition.JLENS_TARGETED),
)


class AnalysisError(ValueError):
    """Raised when paired outcomes violate the analysis contract."""


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """The minimal immutable row needed for paired exact-match analysis."""

    item_id: str
    seed: int
    condition: Condition
    eligible_omitted: bool
    exact_match: bool

    @classmethod
    def from_outcome(cls, outcome: OutcomeRecord) -> AnalysisRow:
        """Project a full outcome onto the analysis fields."""

        return cls(
            item_id=outcome.item_id,
            seed=outcome.seed,
            condition=outcome.condition,
            eligible_omitted=outcome.eligible_omitted,
            exact_match=outcome.exact_match,
        )


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """Exact-match accuracy for one condition in one analysis subset."""

    condition: Condition
    n_rows: int
    n_items: int
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """An item-clustered paired exact-match difference and percentile CI."""

    treatment: Condition
    comparator: Condition
    n_pairs: int
    n_items: int
    difference: float | None
    ci_low: float | None
    ci_high: float | None
    bootstrap_standard_error: float | None


@dataclass(frozen=True, slots=True)
class SubsetSummary:
    """Condition accuracies and paired contrasts for one prespecified subset."""

    subset: str
    n_rows: int
    n_item_seeds: int
    n_items: int
    conditions: tuple[ConditionSummary, ...]
    comparisons: tuple[PairedBootstrapResult, ...]


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """The required ITT analysis and pre-treatment omitted-subset analysis."""

    itt: SubsetSummary
    omitted: SubsetSummary

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""

        return asdict(self)


RowInput = AnalysisRow | OutcomeRecord | Mapping[str, object]


def paired_cluster_bootstrap(
    rows: Iterable[RowInput],
    *,
    treatment: Condition | str,
    comparator: Condition | str,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260710,
    require_complete: bool = True,
) -> PairedBootstrapResult:
    """Estimate a paired accuracy difference by resampling item clusters.

    Pairing occurs within ``(item_id, seed)``.  Differences are first averaged
    across seeds within each item, making the item—not the stochastic seed—the
    independent unit resampled by the bootstrap.
    """

    treatment_condition = Condition(treatment)
    comparator_condition = Condition(comparator)
    if treatment_condition is comparator_condition:
        raise AnalysisError("treatment and comparator must differ")
    if n_resamples <= 0:
        raise AnalysisError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise AnalysisError("confidence must be between zero and one")

    materialized = _coerce_and_validate(rows)
    by_pair: dict[tuple[str, int], dict[Condition, bool]] = defaultdict(dict)
    for row in materialized:
        pair = (row.item_id, row.seed)
        by_pair[pair]
        if row.condition not in {treatment_condition, comparator_condition}:
            continue
        by_pair[pair][row.condition] = row.exact_match

    incomplete = [
        pair
        for pair, outcomes in by_pair.items()
        if treatment_condition not in outcomes or comparator_condition not in outcomes
    ]
    if incomplete and require_complete:
        preview = ", ".join(f"{item}/{run_seed}" for item, run_seed in incomplete[:3])
        raise AnalysisError(f"incomplete paired conditions for: {preview}")

    item_differences: dict[str, list[float]] = defaultdict(list)
    n_pairs = 0
    for (item_id, _), outcomes in by_pair.items():
        if treatment_condition not in outcomes or comparator_condition not in outcomes:
            continue
        difference = float(outcomes[treatment_condition]) - float(
            outcomes[comparator_condition]
        )
        item_differences[item_id].append(difference)
        n_pairs += 1

    cluster_effects = [
        statistics.fmean(differences) for differences in item_differences.values()
    ]
    if not cluster_effects:
        return PairedBootstrapResult(
            treatment=treatment_condition,
            comparator=comparator_condition,
            n_pairs=0,
            n_items=0,
            difference=None,
            ci_low=None,
            ci_high=None,
            bootstrap_standard_error=None,
        )

    point = statistics.fmean(cluster_effects)
    generator = random.Random(seed)
    cluster_count = len(cluster_effects)
    draws = [
        statistics.fmean(
            cluster_effects[generator.randrange(cluster_count)]
            for _ in range(cluster_count)
        )
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    ordered = sorted(draws)
    standard_error = statistics.stdev(draws) if len(draws) > 1 else 0.0
    return PairedBootstrapResult(
        treatment=treatment_condition,
        comparator=comparator_condition,
        n_pairs=n_pairs,
        n_items=cluster_count,
        difference=point,
        ci_low=_quantile(ordered, alpha),
        ci_high=_quantile(ordered, 1.0 - alpha),
        bootstrap_standard_error=standard_error,
    )


def analyze_itt_and_omitted(
    rows: Iterable[RowInput],
    *,
    comparisons: Sequence[tuple[Condition | str, Condition | str]] = (
        DEFAULT_COMPARISONS
    ),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260710,
    require_complete: bool = True,
) -> ExperimentSummary:
    """Return the full ITT and gold-bridge-omitted subset summaries."""

    materialized = _coerce_and_validate(rows)
    return ExperimentSummary(
        itt=_summarize_subset(
            "itt",
            materialized,
            comparisons=comparisons,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            require_complete=require_complete,
        ),
        omitted=_summarize_subset(
            "omitted",
            [row for row in materialized if row.eligible_omitted],
            comparisons=comparisons,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            require_complete=require_complete,
        ),
    )


def _summarize_subset(
    name: str,
    rows: list[AnalysisRow],
    *,
    comparisons: Sequence[tuple[Condition | str, Condition | str]],
    n_resamples: int,
    confidence: float,
    seed: int,
    require_complete: bool,
) -> SubsetSummary:
    condition_summaries = []
    for condition in Condition:
        condition_rows = [row for row in rows if row.condition is condition]
        condition_summaries.append(
            ConditionSummary(
                condition=condition,
                n_rows=len(condition_rows),
                n_items=len({row.item_id for row in condition_rows}),
                accuracy=(
                    statistics.fmean(float(row.exact_match) for row in condition_rows)
                    if condition_rows
                    else None
                ),
            )
        )

    paired_summaries = []
    for index, (treatment, comparator) in enumerate(comparisons):
        paired_summaries.append(
            paired_cluster_bootstrap(
                rows,
                treatment=treatment,
                comparator=comparator,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + index,
                require_complete=require_complete,
            )
        )

    return SubsetSummary(
        subset=name,
        n_rows=len(rows),
        n_item_seeds=len({(row.item_id, row.seed) for row in rows}),
        n_items=len({row.item_id for row in rows}),
        conditions=tuple(condition_summaries),
        comparisons=tuple(paired_summaries),
    )


def _coerce_and_validate(rows: Iterable[RowInput]) -> list[AnalysisRow]:
    materialized = [_coerce_row(row) for row in rows]
    seen: set[tuple[str, int, Condition]] = set()
    eligibility: dict[tuple[str, int], bool] = {}
    for row in materialized:
        key = (row.item_id, row.seed, row.condition)
        if key in seen:
            raise AnalysisError(f"duplicate outcome row: {key}")
        seen.add(key)
        pair = (row.item_id, row.seed)
        if pair in eligibility and eligibility[pair] != row.eligible_omitted:
            raise AnalysisError(
                f"eligibility is inconsistent across branches for {pair}"
            )
        eligibility[pair] = row.eligible_omitted
    return materialized


def _coerce_row(row: RowInput) -> AnalysisRow:
    if isinstance(row, AnalysisRow):
        return row
    if isinstance(row, OutcomeRecord):
        return AnalysisRow.from_outcome(row)
    try:
        exact_value = row["exact_match"]
        eligible_value = row["eligible_omitted"]
        if not isinstance(exact_value, bool) or not isinstance(eligible_value, bool):
            raise AnalysisError("exact_match and eligible_omitted must be booleans")
        return AnalysisRow(
            item_id=str(row["item_id"]),
            seed=int(row["seed"]),
            condition=Condition(str(row["condition"])),
            eligible_omitted=eligible_value,
            exact_match=exact_value,
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f"invalid analysis row: {row!r}") from error


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        raise AnalysisError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
