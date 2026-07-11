from __future__ import annotations

import pytest

from jlens_panel.analysis import (
    AnalysisError,
    AnalysisRow,
    analyze_itt_and_omitted,
    paired_cluster_bootstrap,
)
from jlens_panel.experiment import Condition


def _rows() -> list[AnalysisRow]:
    outcomes = {
        "a": {
            Condition.GENERIC: False,
            Condition.JLENS_TARGETED: True,
            Condition.BEST_NON_J: False,
            Condition.ORACLE: True,
        },
        "b": {
            Condition.GENERIC: True,
            Condition.JLENS_TARGETED: True,
            Condition.BEST_NON_J: True,
            Condition.ORACLE: True,
        },
        "c": {
            Condition.GENERIC: False,
            Condition.JLENS_TARGETED: False,
            Condition.BEST_NON_J: True,
            Condition.ORACLE: True,
        },
    }
    rows = []
    for item_id, conditions in outcomes.items():
        for condition, correct in conditions.items():
            rows.append(
                AnalysisRow(
                    item_id=item_id,
                    seed=17,
                    condition=condition,
                    eligible_omitted=item_id != "c",
                    exact_match=correct,
                )
            )
    return rows


def test_item_cluster_paired_bootstrap_is_deterministic() -> None:
    first = paired_cluster_bootstrap(
        _rows(),
        treatment=Condition.JLENS_TARGETED,
        comparator=Condition.BEST_NON_J,
        n_resamples=500,
        seed=7,
    )
    second = paired_cluster_bootstrap(
        _rows(),
        treatment=Condition.JLENS_TARGETED,
        comparator=Condition.BEST_NON_J,
        n_resamples=500,
        seed=7,
    )

    assert first == second
    assert first.n_pairs == 3
    assert first.n_items == 3
    assert first.difference == pytest.approx(0.0)
    assert first.ci_low is not None
    assert first.ci_high is not None


def test_bootstrap_averages_seeds_within_item_before_resampling() -> None:
    rows = [
        AnalysisRow("a", 1, Condition.JLENS_TARGETED, True, True),
        AnalysisRow("a", 1, Condition.BEST_NON_J, True, False),
        AnalysisRow("a", 2, Condition.JLENS_TARGETED, True, False),
        AnalysisRow("a", 2, Condition.BEST_NON_J, True, False),
        AnalysisRow("b", 1, Condition.JLENS_TARGETED, True, False),
        AnalysisRow("b", 1, Condition.BEST_NON_J, True, True),
    ]

    result = paired_cluster_bootstrap(
        rows,
        treatment="jlens_targeted",
        comparator="best_non_j",
        n_resamples=100,
    )

    assert result.n_pairs == 3
    assert result.n_items == 2
    assert result.difference == pytest.approx(-0.25)


def test_analysis_reports_itt_and_pre_treatment_omitted_subset() -> None:
    summary = analyze_itt_and_omitted(_rows(), n_resamples=250, seed=11)

    assert summary.itt.n_items == 3
    assert summary.itt.n_item_seeds == 3
    assert summary.omitted.n_items == 2
    assert summary.omitted.n_item_seeds == 2
    primary_itt = summary.itt.comparisons[0]
    primary_omitted = summary.omitted.comparisons[0]
    assert primary_itt.difference == pytest.approx(0.0)
    assert primary_omitted.difference == pytest.approx(0.5)


def test_analysis_rejects_inconsistent_branch_eligibility() -> None:
    rows = [
        AnalysisRow("a", 1, Condition.JLENS_TARGETED, True, True),
        AnalysisRow("a", 1, Condition.BEST_NON_J, False, False),
    ]

    with pytest.raises(AnalysisError, match="eligibility is inconsistent"):
        analyze_itt_and_omitted(rows, n_resamples=10)


def test_paired_analysis_rejects_incomplete_pairs_by_default() -> None:
    rows = [AnalysisRow("a", 1, Condition.GENERIC, True, True)]

    with pytest.raises(AnalysisError, match="incomplete paired conditions"):
        paired_cluster_bootstrap(
            rows,
            treatment=Condition.JLENS_TARGETED,
            comparator=Condition.BEST_NON_J,
            n_resamples=10,
        )
