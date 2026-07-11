#!/usr/bin/env python3
"""Analyze paired dirty-run outcomes and interpret the preregistered gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jlens_panel.analysis import (  # noqa: E402
    ExperimentSummary,
    PairedBootstrapResult,
    analyze_itt_and_omitted,
)
from jlens_panel.config import load_config  # noqa: E402
from jlens_panel.experiment import Condition, JsonlResultStore  # noqa: E402
from jlens_panel.provenance import write_json_atomic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute ITT and pre-treatment omitted-subset paired summaries."
    )
    parser.add_argument("--config", type=Path, default=Path("config/dirty_run.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze available pairs; strict complete-pair validation is the default.",
    )
    return parser


def analyze(args: argparse.Namespace) -> dict[str, object]:
    """Load outcomes, run paired analyses, and build gate interpretation."""

    config = load_config(args.config)
    outcomes = JsonlResultStore(args.input).all_outcomes()
    if not outcomes:
        raise ValueError("result JSONL contains no outcomes")
    summary = analyze_itt_and_omitted(
        outcomes,
        n_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.bootstrap_seed,
        require_complete=not args.allow_incomplete,
    )
    payload = {
        "schema_version": "dirty-clarification-analysis-v1",
        "input": str(args.input.resolve()),
        "analysis": summary.to_dict(),
        "interpretation": build_gate_interpretation(summary, config["gates"]),
    }
    write_json_atomic(args.output, payload)
    return payload


def build_gate_interpretation(
    summary: ExperimentSummary, gates: dict[str, Any]
) -> dict[str, object]:
    """Interpret the oracle gate and clearly label targeted contrasts exploratory."""

    oracle_threshold = float(gates["min_oracle_accuracy_lift"])
    targeted_threshold = float(gates["min_targeted_accuracy_lift"])
    minimum_omitted_items = int(gates["min_omitted_items"])
    omitted_oracle = _find_comparison(
        summary.omitted.comparisons, Condition.ORACLE, Condition.GENERIC
    )
    primary_gate = _threshold_interpretation(omitted_oracle, oracle_threshold)
    enough_omitted_items = summary.omitted.n_items >= minimum_omitted_items
    primary_gate.update(
        {
            "name": "oracle_lift",
            "primary_subset": "omitted",
            "minimum_omitted_items": minimum_omitted_items,
            "observed_omitted_items": summary.omitted.n_items,
            "enough_omitted_items": enough_omitted_items,
            "status": (
                "not_evaluable"
                if omitted_oracle.difference is None or not enough_omitted_items
                else "pass" if omitted_oracle.difference >= oracle_threshold else "fail"
            ),
        }
    )

    by_subset: dict[str, object] = {}
    for subset in (summary.itt, summary.omitted):
        oracle = _find_comparison(
            subset.comparisons, Condition.ORACLE, Condition.GENERIC
        )
        jlens_vs_non_j = _find_comparison(
            subset.comparisons,
            Condition.JLENS_TARGETED,
            Condition.BEST_NON_J,
        )
        jlens_vs_generic = _find_comparison(
            subset.comparisons,
            Condition.JLENS_TARGETED,
            Condition.GENERIC,
        )
        by_subset[subset.subset] = {
            "oracle_vs_generic": _threshold_interpretation(oracle, oracle_threshold),
            "exploratory_jlens_vs_best_non_j": _contrast_dict(jlens_vs_non_j),
            "exploratory_jlens_vs_generic": _threshold_interpretation(
                jlens_vs_generic, targeted_threshold
            ),
        }

    return {
        "oracle_lift_gate": primary_gate,
        "targeted_contrasts_are_exploratory": True,
        "subsets": by_subset,
    }


def _find_comparison(
    comparisons: Sequence[PairedBootstrapResult],
    treatment: Condition,
    comparator: Condition,
) -> PairedBootstrapResult:
    for comparison in comparisons:
        if comparison.treatment is treatment and comparison.comparator is comparator:
            return comparison
    raise ValueError(f"missing comparison: {treatment.value} vs {comparator.value}")


def _contrast_dict(comparison: PairedBootstrapResult) -> dict[str, object]:
    return {
        "treatment": comparison.treatment.value,
        "comparator": comparison.comparator.value,
        "n_pairs": comparison.n_pairs,
        "n_items": comparison.n_items,
        "difference": comparison.difference,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "bootstrap_standard_error": comparison.bootstrap_standard_error,
    }


def _threshold_interpretation(
    comparison: PairedBootstrapResult, threshold: float
) -> dict[str, object]:
    result = _contrast_dict(comparison)
    result["threshold"] = threshold
    result["meets_point_estimate_threshold"] = (
        None if comparison.difference is None else comparison.difference >= threshold
    )
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = analyze(args)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    gate = payload["interpretation"]["oracle_lift_gate"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "oracle_lift_gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
