#!/usr/bin/env python3
"""Fit the train-only residual probe and score all five frozen readouts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jlens_panel.config import load_config
from jlens_panel.provenance import (
    build_manifest,
    git_revision,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.readouts import (
    JLensReadout,
    LogitLensReadout,
    NextTokenReadout,
    RawResidualProbeReadout,
    ReadoutMethod,
    ReadoutPanel,
    ReadoutRequest,
    TextOnlyReadout,
    aggregate_metrics,
)
from jlens_panel.readouts.artifacts import (
    SCORE_SCHEMA,
    SPLITS,
    ArtifactError,
    LoadFunction,
    discover_split_artifacts,
    load_artifact,
    load_extraction_manifest,
    path_fingerprint,
    residual_row,
    stable_fingerprint,
    write_jsonl_atomic,
)
from jlens_panel.readouts.core import CandidateSet

NON_J_TIE_ORDER = ("logit_lens", "raw_probe", "next_token", "text_only")
ProbeFactory = Callable[[int], Any]


def load_artifact_splits(
    root: str | Path,
    *,
    limit: int | None = None,
    load_fn: LoadFunction | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Load required train/dev/test artifacts in deterministic path order."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    result: dict[str, list[dict[str, object]]] = {}
    manifest = load_extraction_manifest(root)
    expected_counts = manifest["expected_counts"]
    assert isinstance(expected_counts, Mapping)
    for split in SPLITS:
        paths = discover_split_artifacts(root, split)
        if len(paths) != expected_counts[split]:
            raise ArtifactError(
                f"{split} artifact count {len(paths)} does not match extraction "
                f"manifest count {expected_counts[split]}"
            )
        selected = paths[:limit] if limit is not None else paths
        payloads = [load_artifact(path, load_fn=load_fn) for path in selected]
        if any(payload["split"] != split for payload in payloads):
            raise ArtifactError(f"artifact split does not match {split} directory")
        result[split] = payloads
    candidates, token_ids, layer = validate_scoring_inventory(result)
    if (
        list(candidates) != manifest["candidates"]
        or token_ids != manifest["candidate_token_ids"]
        or layer != manifest["canonical_layer"]
    ):
        raise ArtifactError("loaded artifacts do not match extraction manifest")
    first_fingerprints = result["train"][0]["fingerprints"]
    assert isinstance(first_fingerprints, Mapping)
    if first_fingerprints["extraction"] != manifest["extraction_sha256"]:
        raise ArtifactError("artifact extraction fingerprint disagrees with manifest")
    return result


def validate_scoring_inventory(
    artifacts: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[tuple[str, ...], dict[str, int], int]:
    """Require one candidate/token/layer/extraction contract across all splits."""

    if set(artifacts) != set(SPLITS):
        raise ArtifactError("scoring requires train, dev, and test artifact splits")
    flattened = [artifact for split in SPLITS for artifact in artifacts[split]]
    if not flattened or any(not artifacts[split] for split in SPLITS):
        raise ArtifactError("every scoring split must contain at least one artifact")
    reference = flattened[0]
    candidates = tuple(reference["candidates"])
    token_ids = dict(reference["candidate_token_ids"])
    layer = int(reference["layer"])
    fingerprints = reference["fingerprints"]
    assert isinstance(fingerprints, Mapping)
    extraction = fingerprints["extraction"]

    seen_ids: set[str] = set()
    for artifact in flattened:
        example_id = str(artifact["example_id"])
        if example_id in seen_ids:
            raise ArtifactError(f"duplicate artifact example_id: {example_id}")
        seen_ids.add(example_id)
        artifact_fingerprints = artifact["fingerprints"]
        assert isinstance(artifact_fingerprints, Mapping)
        if (
            tuple(artifact["candidates"]) != candidates
            or dict(artifact["candidate_token_ids"]) != token_ids
            or int(artifact["layer"]) != layer
            or artifact_fingerprints["extraction"] != extraction
        ):
            raise ArtifactError(f"artifact contract mismatch at example {example_id!r}")
    return candidates, token_ids, layer


def select_strongest_non_j(
    dev_method_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Select on DEV log loss only, with a frozen deterministic tie order."""

    missing = [method for method in NON_J_TIE_ORDER if method not in dev_method_metrics]
    if missing:
        raise ArtifactError(
            "DEV metrics are missing non-J methods: " + ", ".join(missing)
        )
    losses: dict[str, float] = {}
    for method in NON_J_TIE_ORDER:
        loss = float(dev_method_metrics[method]["log_loss"])
        if not math.isfinite(loss):
            raise ArtifactError(f"DEV log loss for {method} is not finite")
        losses[method] = loss
    selected = min(
        NON_J_TIE_ORDER,
        key=lambda method: (losses[method], NON_J_TIE_ORDER.index(method)),
    )
    return {
        "schema_version": SCORE_SCHEMA,
        "selection_split": "dev",
        "criterion": "minimum_mean_log_loss",
        "tie_order": list(NON_J_TIE_ORDER),
        "selected_method": selected,
        "dev_log_loss": losses,
        "test_metrics_used_for_selection": False,
    }


def score_artifact_splits(
    artifacts: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    probe_factory: ProbeFactory = RawResidualProbeReadout,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Fit on TRAIN, select on DEV, and only then evaluate TEST."""

    candidates, token_ids, layer = validate_scoring_inventory(artifacts)
    candidate_set = CandidateSet.from_pairs(
        (token_ids[candidate], candidate) for candidate in candidates
    )

    probe = probe_factory(layer=layer)
    train_rows = [residual_row(artifact["residual"]) for artifact in artifacts["train"]]
    train_targets = [int(artifact["gold_token_id"]) for artifact in artifacts["train"]]
    # The only estimator fit in this pipeline sees TRAIN artifacts exclusively.
    probe.fit(train_rows, train_targets, layer=layer)

    records: list[dict[str, object]] = []
    split_metrics: dict[str, dict[str, list[Any]]] = {
        split: {method.value: [] for method in ReadoutMethod} for split in SPLITS
    }

    # Score TRAIN and DEV, freeze the non-J selection, then touch TEST outcomes.
    for split in ("train", "dev"):
        _score_split(
            artifacts[split],
            split=split,
            candidate_set=candidate_set,
            probe=probe,
            output_records=records,
            metric_store=split_metrics[split],
        )
    summaries: dict[str, dict[str, dict[str, float]]] = {
        split: {
            method: aggregate_metrics(rows)
            for method, rows in split_metrics[split].items()
        }
        for split in ("train", "dev")
    }
    selection = select_strongest_non_j(summaries["dev"])

    _score_split(
        artifacts["test"],
        split="test",
        candidate_set=candidate_set,
        probe=probe,
        output_records=records,
        metric_store=split_metrics["test"],
    )
    summaries["test"] = {
        method: aggregate_metrics(rows)
        for method, rows in split_metrics["test"].items()
    }
    summary: dict[str, object] = {
        "schema_version": SCORE_SCHEMA,
        "canonical_layer": layer,
        "candidates": list(candidates),
        "candidate_token_ids": token_ids,
        "selected_non_j_method": selection["selected_method"],
        "splits": {
            split: {
                "examples": len(artifacts[split]),
                "methods": summaries[split],
            }
            for split in SPLITS
        },
    }
    return records, summary, selection


def _score_split(
    artifacts: Sequence[Mapping[str, object]],
    *,
    split: str,
    candidate_set: CandidateSet,
    probe: Any,
    output_records: list[dict[str, object]],
    metric_store: dict[str, list[Any]],
) -> None:
    for artifact in artifacts:
        raw_scores = artifact["scores"]
        assert isinstance(raw_scores, Mapping)
        text_scores = raw_scores["text_only"]
        panel = ReadoutPanel(
            [
                JLensReadout(),
                LogitLensReadout(),
                probe,
                NextTokenReadout(),
                TextOnlyReadout(lambda _text, _candidates, values=text_scores: values),
            ]
        )
        fingerprints = artifact["fingerprints"]
        assert isinstance(fingerprints, Mapping)
        request = ReadoutRequest(
            example_id=str(artifact["example_id"]),
            candidates=candidate_set,
            layer=int(artifact["layer"]),
            jlens_logits=raw_scores["jlens"],
            logit_lens_logits=raw_scores["logit_lens"],
            residual=artifact["residual"],
            next_token_logits=raw_scores["next_token"],
            # Text is a lookup key only: extraction already ran the text classifier.
            text=str(artifact["example_id"]),
            metadata={
                "split": split,
                "artifact_extraction_sha256": fingerprints["extraction"],
            },
        )
        result = panel.score(request)
        metrics = result.evaluate(int(artifact["gold_token_id"]))
        metric_by_method = {metric.method: metric for metric in metrics}
        for record in result.records:
            metric = metric_by_method[record.method]
            metric_store[record.method.value].append(metric)
            row = record.to_dict()
            row.update(
                {
                    "schema_version": SCORE_SCHEMA,
                    "split": split,
                    "gold_bridge": artifact["gold_bridge"],
                    "gold_token_id": artifact["gold_token_id"],
                    "top1_accuracy": metric.top1_accuracy,
                    "mean_reciprocal_rank": metric.mean_reciprocal_rank,
                    "log_loss": metric.log_loss,
                }
            )
            output_records.append(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score extracted readouts with train/dev/test isolation."
    )
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug limit applied independently to each split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    load_config(args.config)
    artifact_tree_sha256 = path_fingerprint(args.artifact_dir)
    extraction_manifest_path = Path(args.artifact_dir) / "extraction_manifest.json"
    extraction_manifest_sha256 = sha256_file(extraction_manifest_path)
    artifacts = load_artifact_splits(args.artifact_dir, limit=args.limit)
    records, summary, selection = score_artifact_splits(artifacts)
    scored_counts = {split: len(artifacts[split]) for split in SPLITS}
    score_bundle_sha256 = stable_fingerprint(
        {
            "schema_version": SCORE_SCHEMA,
            "config_sha256": sha256_file(args.config),
            "artifact_tree_sha256": artifact_tree_sha256,
            "extraction_manifest_sha256": extraction_manifest_sha256,
            "git_revision": git_revision(args.project_root),
            "scored_counts": scored_counts,
            "limit": args.limit,
        }
    )
    for record in records:
        record["score_bundle_sha256"] = score_bundle_sha256
    summary["score_bundle_sha256"] = score_bundle_sha256
    selection["score_bundle_sha256"] = score_bundle_sha256
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = write_jsonl_atomic(output / "readout_scores.jsonl", records)
    write_json_atomic(output / "readout_summary.json", summary)
    write_json_atomic(output / "non_j_selection.json", selection)
    score_manifest = build_manifest(
        config_path=args.config,
        project_root=args.project_root,
        extra={
            "artifact_dir": str(Path(args.artifact_dir).resolve()),
            "artifact_tree_sha256": artifact_tree_sha256,
            "extraction_manifest_sha256": extraction_manifest_sha256,
            "score_bundle_sha256": score_bundle_sha256,
            "scored_counts": scored_counts,
            "record_count": len(records),
            "selected_non_j_method": selection["selected_method"],
            "selection_split": "dev",
            "test_metrics_used_for_selection": False,
        },
    )
    write_json_atomic(output / "score_manifest.json", score_manifest)
    print(
        json.dumps(
            {
                "records": str(records_path.resolve()),
                "summary": str((output / "readout_summary.json").resolve()),
                "selection": str((output / "non_j_selection.json").resolve()),
                "manifest": str((output / "score_manifest.json").resolve()),
                "selected_non_j_method": selection["selected_method"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
