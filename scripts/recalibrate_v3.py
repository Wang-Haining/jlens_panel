#!/usr/bin/env python3
"""Center existing v3 candidate scores using train-only score means."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jlens_panel.config import load_config
from jlens_panel.data import DEFAULT_BRIDGE_CANDIDATES
from jlens_panel.provenance import (
    build_manifest,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.readouts import (
    CandidateSet,
    ReadoutMethod,
    ReadoutRecord,
    aggregate_metrics,
    evaluate_record,
)
from jlens_panel.readouts.artifacts import (
    EXTRACTION_MANIFEST_NAME,
    LoadFunction,
    discover_split_artifacts,
    load_artifact,
    load_extraction_manifest,
    path_fingerprint,
)

SCHEMA_VERSION = "jlens-panel-recalibration-v1"
METHODS = ("jlens", "logit_lens", "next_token")
SPLITS = ("train", "dev")
S0_TOP1_THRESHOLD = 0.25
G_CONST_MAX_SHARE = 0.50
G_CONST_MIN_ENTROPY_BITS = 1.5
EXPECTED_EXTRACTION_SHA256 = (
    "1b80f78fefda0714563dfc646ceb5d69e1758967b48f7ba66431e357d63e6767"
)
EXPECTED_TRAIN_COUNT = 1000
EXPECTED_DEV_COUNT = 200
EXPECTED_LAYER = 13


class RecalibrationError(ValueError):
    """Raised when an S0 input or output violates the frozen contract."""


def validate_v3_source_contract(
    manifest: Mapping[str, object], *, config_sha256: str
) -> None:
    """Bind S0 to the exact frozen v3 extraction rather than a lookalike tree."""

    if manifest.get("extraction_sha256") != EXPECTED_EXTRACTION_SHA256:
        raise RecalibrationError("artifact tree is not the frozen v3 extraction")
    if manifest.get("canonical_layer") != EXPECTED_LAYER:
        raise RecalibrationError("frozen v3 canonical layer changed")
    expected_candidates = tuple(sorted(DEFAULT_BRIDGE_CANDIDATES))
    if tuple(manifest.get("candidates", ())) != expected_candidates:
        raise RecalibrationError("frozen v3 candidate inventory changed")
    counts = manifest.get("expected_counts")
    if not isinstance(counts, Mapping):
        raise RecalibrationError("frozen v3 manifest counts are missing")
    if counts.get("train") != EXPECTED_TRAIN_COUNT:
        raise RecalibrationError("frozen v3 train count must be 1000")
    if counts.get("dev") != EXPECTED_DEV_COUNT:
        raise RecalibrationError("frozen v3 dev count must be 200")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RecalibrationError("frozen v3 provenance is missing")
    if provenance.get("config_sha256") != config_sha256:
        raise RecalibrationError("config hash disagrees with frozen v3 provenance")


def validate_output_paths(output: Path, manifest: Path) -> None:
    """Reject collisions and overwrites before either atomic write begins."""

    if output.resolve() == manifest.resolve():
        raise RecalibrationError("summary and manifest paths must be distinct")
    if output.exists() or manifest.exists():
        raise RecalibrationError(
            "refusing to overwrite an existing S0 output or manifest"
        )


def s0_checkpoint(top1_accuracy: float) -> dict[str, object]:
    """Interpret the frozen checkpoint while surfacing exact equality."""

    if not math.isfinite(top1_accuracy):
        raise RecalibrationError("S0 checkpoint accuracy must be finite")
    if top1_accuracy == S0_TOP1_THRESHOLD:
        status = "ambiguous_equal_threshold"
        short_circuit: bool | None = None
    elif top1_accuracy > S0_TOP1_THRESHOLD:
        status = "pass"
        short_circuit = True
    else:
        status = "fail"
        short_circuit = False
    return {
        "method": "jlens",
        "metric": "centered_dev_top1_accuracy",
        "threshold": S0_TOP1_THRESHOLD,
        "observed": top1_accuracy,
        "status": status,
        "short_circuit": short_circuit,
    }


def load_train_dev_artifacts(
    root: str | Path,
    *,
    load_fn: LoadFunction | None = None,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Load exactly train and dev artifacts; never discover the test directory."""

    artifact_root = Path(root)
    manifest = load_extraction_manifest(artifact_root)
    candidates = tuple(manifest["candidates"])
    token_ids = dict(manifest["candidate_token_ids"])
    extraction = str(manifest["extraction_sha256"])
    layer = int(manifest["canonical_layer"])
    expected_counts = manifest["expected_counts"]
    if not isinstance(expected_counts, Mapping):
        raise RecalibrationError("extraction manifest counts must be a mapping")
    if len(candidates) != 16:
        raise RecalibrationError("S0 requires exactly 16 candidate labels")

    artifacts: dict[str, list[dict[str, object]]] = {}
    seen_ids: set[str] = set()
    for split in SPLITS:
        paths = discover_split_artifacts(artifact_root, split)
        expected = expected_counts.get(split)
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise RecalibrationError(f"manifest has no integer count for {split}")
        if len(paths) != expected:
            raise RecalibrationError(
                f"{split} artifact count {len(paths)} does not match {expected}"
            )
        split_artifacts: list[dict[str, object]] = []
        for path in paths:
            artifact = load_artifact(path, load_fn=load_fn)
            _validate_artifact_contract(
                artifact,
                split=split,
                candidates=candidates,
                candidate_token_ids=token_ids,
                extraction_sha256=extraction,
                layer=layer,
            )
            example_id = str(artifact["example_id"])
            if example_id in seen_ids:
                raise RecalibrationError(f"duplicate example id: {example_id}")
            seen_ids.add(example_id)
            split_artifacts.append(artifact)
        artifacts[split] = split_artifacts
    return artifacts, manifest


def _validate_artifact_contract(
    artifact: Mapping[str, object],
    *,
    split: str,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    extraction_sha256: str,
    layer: int,
) -> None:
    if artifact.get("split") != split:
        raise RecalibrationError(f"artifact is stored in the wrong split: {split}")
    if tuple(artifact.get("candidates", ())) != tuple(candidates):
        raise RecalibrationError("artifact candidate order changed")
    raw_token_ids = artifact.get("candidate_token_ids")
    if not isinstance(raw_token_ids, Mapping) or dict(raw_token_ids) != dict(
        candidate_token_ids
    ):
        raise RecalibrationError("artifact candidate token ids changed")
    if artifact.get("layer") != layer:
        raise RecalibrationError("artifact layer changed")
    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise RecalibrationError("artifact fingerprints are missing")
    if fingerprints.get("extraction") != extraction_sha256:
        raise RecalibrationError("artifact extraction fingerprint changed")
    _validated_scores(artifact, candidates)


def _validated_scores(
    artifact: Mapping[str, object], candidates: Sequence[str]
) -> dict[str, dict[str, float]]:
    scores = artifact.get("scores")
    if not isinstance(scores, Mapping):
        raise RecalibrationError("artifact scores must be a mapping")
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        method_scores = scores.get(method)
        if not isinstance(method_scores, Mapping) or set(method_scores) != set(
            candidates
        ):
            raise RecalibrationError(
                f"{method} scores must match the 16 candidates exactly"
            )
        normalized: dict[str, float] = {}
        for candidate in candidates:
            value = method_scores[candidate]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecalibrationError(f"{method} scores must be numeric")
            score = float(value)
            if not math.isfinite(score):
                raise RecalibrationError(f"{method} scores must be finite")
            normalized[candidate] = score
        result[method] = normalized
    return result


def compute_train_means(
    train_artifacts: Sequence[Mapping[str, object]],
    candidates: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Estimate one context-independent candidate mean from train only."""

    if not train_artifacts:
        raise RecalibrationError("train artifacts cannot be empty")
    totals = {
        method: {candidate: [] for candidate in candidates} for method in METHODS
    }
    for artifact in train_artifacts:
        if artifact.get("split") != "train":
            raise RecalibrationError("candidate means may use train artifacts only")
        scores = _validated_scores(artifact, candidates)
        for method in METHODS:
            for candidate in candidates:
                totals[method][candidate].append(scores[method][candidate])
    count = len(train_artifacts)
    return {
        method: {
            candidate: math.fsum(totals[method][candidate]) / count
            for candidate in candidates
        }
        for method in METHODS
    }


def evaluate_dev(
    dev_artifacts: Sequence[Mapping[str, object]],
    *,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    layer: int,
    candidate_means: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, object]]:
    """Evaluate raw or centered candidate scores on dev only."""

    if not dev_artifacts:
        raise RecalibrationError("dev artifacts cannot be empty")
    candidate_set = CandidateSet.from_pairs(
        (candidate_token_ids[candidate], candidate) for candidate in candidates
    )
    metrics: dict[str, list[Any]] = {method: [] for method in METHODS}
    predictions: dict[str, list[str]] = {method: [] for method in METHODS}
    for artifact in dev_artifacts:
        if artifact.get("split") != "dev":
            raise RecalibrationError("S0 evaluation may use dev artifacts only")
        scores = _validated_scores(artifact, candidates)
        gold_token_id = artifact.get("gold_token_id")
        gold_bridge = artifact.get("gold_bridge")
        if gold_bridge not in candidates:
            raise RecalibrationError("dev gold bridge is not a candidate")
        if gold_token_id != candidate_token_ids[gold_bridge]:
            raise RecalibrationError("dev gold token id disagrees with gold bridge")
        for method in METHODS:
            means = candidate_means.get(method) if candidate_means is not None else None
            if means is not None and set(means) != set(candidates):
                raise RecalibrationError(f"{method} means do not match candidates")
            logits = tuple(
                scores[method][candidate]
                - (float(means[candidate]) if means is not None else 0.0)
                for candidate in candidates
            )
            if not all(math.isfinite(value) for value in logits):
                raise RecalibrationError(f"{method} produced non-finite logits")
            record = ReadoutRecord(
                example_id=str(artifact["example_id"]),
                method=ReadoutMethod(method),
                candidates=candidate_set,
                logits=logits,
                layer=layer if method != "next_token" else None,
                metadata={"split": "dev"},
            )
            metrics[method].append(evaluate_record(record, int(gold_token_id)))
            winner = min(
                range(len(logits)),
                key=lambda index: (-logits[index], index),
            )
            predictions[method].append(candidates[winner])

    return {
        method: {
            "metrics": aggregate_metrics(metrics[method]),
            "prediction_distribution": _prediction_distribution(
                predictions[method], candidates
            ),
        }
        for method in METHODS
    }


def _prediction_distribution(
    predictions: Sequence[str], candidates: Sequence[str]
) -> dict[str, object]:
    if not predictions:
        raise RecalibrationError("prediction distribution cannot be empty")
    counts = Counter(predictions)
    if not set(counts).issubset(candidates):
        raise RecalibrationError("prediction distribution contains an unknown label")
    total = len(predictions)
    proportions = {
        candidate: counts[candidate] / total for candidate in candidates
    }
    entropy = -math.fsum(
        probability * math.log2(probability)
        for probability in proportions.values()
        if probability > 0.0
    )
    maximum = max(proportions.values())
    return {
        "counts": {candidate: counts[candidate] for candidate in candidates},
        "proportions": proportions,
        "max_prediction_share": maximum,
        "prediction_entropy_bits": entropy,
    }


def build_summary(
    artifacts: Mapping[str, Sequence[Mapping[str, object]]],
    manifest: Mapping[str, object],
    *,
    source: Mapping[str, object],
) -> dict[str, object]:
    """Build the complete S0 summary from a validated train/dev collection."""

    if set(artifacts) != set(SPLITS):
        raise RecalibrationError("S0 requires exactly train and dev artifact splits")
    candidates = tuple(manifest["candidates"])
    candidate_token_ids = dict(manifest["candidate_token_ids"])
    layer = int(manifest["canonical_layer"])
    means = compute_train_means(artifacts["train"], candidates)
    raw = evaluate_dev(
        artifacts["dev"],
        candidates=candidates,
        candidate_token_ids=candidate_token_ids,
        layer=layer,
        candidate_means=None,
    )
    centered = evaluate_dev(
        artifacts["dev"],
        candidates=candidates,
        candidate_token_ids=candidate_token_ids,
        layer=layer,
        candidate_means=means,
    )
    jlens_top1 = float(centered["jlens"]["metrics"]["top1_accuracy"])
    jlens_distribution = centered["jlens"]["prediction_distribution"]
    max_share = float(jlens_distribution["max_prediction_share"])
    entropy = float(jlens_distribution["prediction_entropy_bits"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "S0",
        "calibration": "train_candidate_mean_centering",
        "splits_used": ["train", "dev"],
        "test_or_smoke_read": False,
        "n_train": len(artifacts["train"]),
        "n_dev": len(artifacts["dev"]),
        "layer": layer,
        "candidates": list(candidates),
        "candidate_token_ids": candidate_token_ids,
        "candidate_means": means,
        "methods": {
            method: {"raw_dev": raw[method], "centered_dev": centered[method]}
            for method in METHODS
        },
        "gates": {
            "s0_checkpoint": s0_checkpoint(jlens_top1),
            "g_const": {
                "method": "jlens",
                "calibration": "center",
                "maximum_prediction_share_threshold": G_CONST_MAX_SHARE,
                "prediction_entropy_bits_threshold": G_CONST_MIN_ENTROPY_BITS,
                "observed_max_prediction_share": max_share,
                "observed_prediction_entropy_bits": entropy,
                "pass": max_share <= G_CONST_MAX_SHARE
                and entropy > G_CONST_MIN_ENTROPY_BITS,
            },
        },
        "source": dict(source),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train-mean calibration on stored v3 train/dev artifacts."
    )
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/dirty_v3")
    parser.add_argument(
        "--output",
        default="analysis/recalibrated_v3_summary.json",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--project-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_config(args.config)
    config_sha256 = sha256_file(args.config)
    artifact_root = Path(args.artifact_dir)
    output = Path(args.output)
    output_manifest = (
        Path(args.manifest)
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    try:
        validate_output_paths(output, output_manifest)
    except RecalibrationError as error:
        raise SystemExit(f"error: {error}") from error

    artifacts, extraction_manifest = load_train_dev_artifacts(artifact_root)
    validate_v3_source_contract(
        extraction_manifest,
        config_sha256=config_sha256,
    )
    extraction_manifest_path = artifact_root / EXTRACTION_MANIFEST_NAME
    source = {
        "artifact_dir": str(artifact_root.resolve()),
        "extraction_manifest": str(extraction_manifest_path.resolve()),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "extraction_sha256": extraction_manifest["extraction_sha256"],
        "config_sha256": config_sha256,
        "train_artifacts_sha256": path_fingerprint(artifact_root / "train"),
        "dev_artifacts_sha256": path_fingerprint(artifact_root / "dev"),
    }
    summary = build_summary(artifacts, extraction_manifest, source=source)
    manifest = build_manifest(
        config_path=args.config,
        project_root=args.project_root,
        extra={
            "task": "S0",
            "source": source,
            "splits_used": ["train", "dev"],
            "test_or_smoke_read": False,
        },
    )
    # Capture the clean code state before the untracked analysis output exists.
    write_json_atomic(output, summary)
    summary_sha256 = sha256_file(output)
    manifest_extra = manifest.get("extra")
    if not isinstance(manifest_extra, dict):  # pragma: no cover - helper invariant
        raise RecalibrationError("run manifest has no mutable extra mapping")
    manifest_extra.update(
        {"summary": str(output.resolve()), "summary_sha256": summary_sha256}
    )
    write_json_atomic(output_manifest, manifest)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "output_sha256": summary_sha256,
                "manifest": str(output_manifest.resolve()),
                "gates": summary["gates"],
                "methods": summary["methods"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
