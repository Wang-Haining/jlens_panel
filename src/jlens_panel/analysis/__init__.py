"""Paired, item-clustered analysis for clarification outcomes."""

from .paired import (
    DEFAULT_COMPARISONS,
    AnalysisError,
    AnalysisRow,
    ConditionSummary,
    ExperimentSummary,
    PairedBootstrapResult,
    SubsetSummary,
    analyze_itt_and_omitted,
    paired_cluster_bootstrap,
)

__all__ = [
    "DEFAULT_COMPARISONS",
    "AnalysisError",
    "AnalysisRow",
    "ConditionSummary",
    "ExperimentSummary",
    "PairedBootstrapResult",
    "SubsetSummary",
    "analyze_itt_and_omitted",
    "paired_cluster_bootstrap",
]
