#!/usr/bin/env python3
"""Capture train/dev position sweeps and produce calibrated probe analyses."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens_panel.calibration import load_calibrations
from jlens_panel.config import load_config
from jlens_panel.data import DEFAULT_BRIDGE_CANDIDATES, SyntheticBridgeExample, read_jsonl
from jlens_panel.modeling import (
    chat_template_fingerprint,
    load_model_bundle,
    resolve_candidate_token_ids,
)
from jlens_panel.provenance import (
    build_manifest,
    git_is_dirty,
    git_revision,
    installed_versions,
    package_source_fingerprint,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.readouts.artifacts import (
    RunByteBudget,
    stable_fingerprint,
)
from jlens_panel.runtime import H100RuntimeError, configure_h100_runtime
from jlens_panel.storage import build_disk_guard, exclusive_run_lock
from jlens_panel.sweep.capture import (
    CAPTURE_METHODS,
    CAPTURE_SCHEMA,
    build_capture_artifact,
    capture_artifact_path,
    capture_task_example,
    load_capture_artifact,
    save_capture_artifact_atomic,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES
from jlens_panel.sweep.probe_sweep import (
    RESULT_COLUMNS,
    run_probe_sweep,
)

CAPTURE_RUN_SCHEMA = "jlens-panel-sweep-run-v1"
CAPTURE_INDEX_SCHEMA = "jlens-panel-sweep-index-v1"
CAPTURE_PROGRESS_SCHEMA = "jlens-panel-sweep-progress-v1"
ANALYSIS_MANIFEST_SCHEMA = "jlens-panel-sweep-analysis-v1"
CAPTURE_MANIFEST_NAME = "capture_manifest.json"
CAPTURE_INDEX_NAME = "capture_index.json"
CAPTURE_PROGRESS_NAME = "capture_progress.json"
_INDEX_RESERVE_BYTES = 10_000_000


class SweepRunError(RuntimeError):
    """Raised when sweep execution conflicts with frozen inputs or artifacts."""


def _load_json_mapping(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepRunError(f"cannot decode {name}: {path}") from error
    if not isinstance(value, dict):
        raise SweepRunError(f"{name} must be a JSON mapping")
    return value


def _load_train_dev(
    train_path: Path,
    dev_path: Path,
    *,
    expected_train: int,
    expected_dev: int,
) -> tuple[tuple[SyntheticBridgeExample, ...], tuple[SyntheticBridgeExample, ...]]:
    """Read only the two authorized splits and validate their boundaries."""

    train = tuple(read_jsonl(train_path))
    dev = tuple(read_jsonl(dev_path))
    if len(train) != expected_train or len(dev) != expected_dev:
        raise SweepRunError(
            f"train/dev counts changed: {len(train)}/{len(dev)} "
            f"!= {expected_train}/{expected_dev}"
        )
    if any(example.split != "train" for example in train) or any(
        example.split != "dev" for example in dev
    ):
        raise SweepRunError("input JSONL split labels are not exactly train/dev")
    all_examples = (*train, *dev)
    ids = tuple(example.example_id for example in all_examples)
    if len(ids) != len(set(ids)):
        raise SweepRunError("train/dev example IDs are not unique")
    inventory = tuple(sorted(DEFAULT_BRIDGE_CANDIDATES))
    if any(tuple(sorted(example.candidate_bridges)) != inventory for example in all_examples):
        raise SweepRunError("train/dev candidate inventory changed")
    return train, dev


def _runtime_identity(
    config: Mapping[str, object],
    tokenizer: object,
    *,
    execution: Mapping[str, object],
) -> dict[str, object]:
    versions = installed_versions()
    transformers_version = versions["transformers"]
    jlens_version = versions["jlens"]
    if not transformers_version or not jlens_version:
        raise SweepRunError("GPU runtime packages are not installed")
    return {
        "upstream_commit": config["lens"]["upstream_commit"],  # type: ignore[index]
        "jlens_source_sha256": package_source_fingerprint("jlens"),
        "transformers_version": transformers_version,
        "jlens_version": jlens_version,
        "chat_template_sha256": chat_template_fingerprint(tokenizer),
        **dict(execution),
    }


def _require_clean_revision(project_root: Path) -> str:
    revision = git_revision(project_root)
    if revision is None or git_is_dirty(project_root) is not False:
        raise SweepRunError("position sweep requires a clean Git revision")
    return revision


def _capture_identity(
    *,
    config_sha256: str,
    revision: str,
    model_name: str,
    model_revision: str,
    lens_sha256: str,
    input_sha256: Mapping[str, str],
    dataset_sha256: str,
    candidate_token_ids: Mapping[str, int],
    layers: Sequence[int],
    hidden_size: int,
    max_seq_len: int,
    runtime: Mapping[str, object],
    eos_policy: str,
) -> dict[str, object]:
    return {
        "config_sha256": config_sha256,
        "git_revision": revision,
        "model_name": model_name,
        "model_revision": model_revision,
        "lens_sha256": lens_sha256,
        "input_sha256": dict(input_sha256),
        "dataset_sha256": dataset_sha256,
        "candidate_token_ids": dict(candidate_token_ids),
        "positions": list(ALL_POSITION_NAMES),
        "layers": list(layers),
        "hidden_size": hidden_size,
        "max_seq_len": max_seq_len,
        "decode_steps": [1, 2, 4, 8],
        "decode_state_semantics": "generated_token",
        "eos_policy": eos_policy,
        "runtime": dict(runtime),
        "test_or_smoke_read": False,
    }


def _ensure_capture_manifest(
    path: Path,
    *,
    identity: Mapping[str, object],
    expected_counts: Mapping[str, int],
    config_path: Path,
    project_root: Path,
) -> dict[str, object]:
    expected_extra = {
        "schema_version": CAPTURE_RUN_SCHEMA,
        "artifact_schema": CAPTURE_SCHEMA,
        "capture_sha256": stable_fingerprint(identity),
        "identity": dict(identity),
        "expected_counts": dict(expected_counts),
    }
    if path.exists():
        existing = _load_json_mapping(path, name="capture manifest")
        if existing.get("extra") != expected_extra:
            raise SweepRunError("existing capture manifest identity changed")
        return existing
    manifest = build_manifest(
        config_path=config_path,
        project_root=project_root,
        extra=expected_extra,
    )
    write_json_atomic(path, manifest)
    return manifest


def _validate_capture_index(
    index_path: Path,
    *,
    capture_manifest_sha256: str,
    capture_sha256: str,
    progress_sha256: str,
    expected_relative_paths: Sequence[str],
) -> dict[str, str] | None:
    if not index_path.exists():
        return None
    index = _load_json_mapping(index_path, name="capture index")
    if set(index) != {
        "schema_version",
        "capture_manifest_sha256",
        "capture_sha256",
        "progress_sha256",
        "artifacts",
        "counts",
        "test_or_smoke_read",
    }:
        raise SweepRunError("capture index fields changed")
    if (
        index["schema_version"] != CAPTURE_INDEX_SCHEMA
        or index["capture_manifest_sha256"] != capture_manifest_sha256
        or index["capture_sha256"] != capture_sha256
        or index["progress_sha256"] != progress_sha256
        or index["test_or_smoke_read"] is not False
    ):
        raise SweepRunError("capture index identity changed")
    artifacts = index["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        expected_relative_paths
    ):
        raise SweepRunError("capture index artifact inventory changed")
    hashes = dict(artifacts)
    if any(
        not isinstance(digest, str) or len(digest) != 64 for digest in hashes.values()
    ):
        raise SweepRunError("capture index contains an invalid SHA-256")
    return hashes  # type: ignore[return-value]


def _load_capture_progress(
    path: Path,
    *,
    capture_manifest_sha256: str,
    capture_sha256: str,
    expected_relative_paths: Sequence[str],
) -> dict[str, str] | None:
    if not path.exists():
        return None
    progress = _load_json_mapping(path, name="capture progress")
    if set(progress) != {
        "schema_version",
        "capture_manifest_sha256",
        "capture_sha256",
        "artifacts",
        "test_or_smoke_read",
    } or (
        progress["schema_version"] != CAPTURE_PROGRESS_SCHEMA
        or progress["capture_manifest_sha256"] != capture_manifest_sha256
        or progress["capture_sha256"] != capture_sha256
        or progress["test_or_smoke_read"] is not False
    ):
        raise SweepRunError("capture progress identity changed")
    artifacts = progress["artifacts"]
    if not isinstance(artifacts, Mapping) or not set(artifacts).issubset(
        expected_relative_paths
    ):
        raise SweepRunError("capture progress artifact inventory changed")
    hashes = dict(artifacts)
    if any(
        not isinstance(digest, str) or len(digest) != 64 for digest in hashes.values()
    ):
        raise SweepRunError("capture progress contains an invalid SHA-256")
    return hashes  # type: ignore[return-value]


def _write_capture_progress(
    path: Path,
    *,
    capture_manifest_sha256: str,
    capture_sha256: str,
    artifact_hashes: Mapping[str, str],
    expected_relative_paths: Sequence[str],
) -> None:
    ordered = {
        relative: artifact_hashes[relative]
        for relative in expected_relative_paths
        if relative in artifact_hashes
    }
    write_json_atomic(
        path,
        {
            "schema_version": CAPTURE_PROGRESS_SCHEMA,
            "capture_manifest_sha256": capture_manifest_sha256,
            "capture_sha256": capture_sha256,
            "artifacts": ordered,
            "test_or_smoke_read": False,
        },
    )


def _estimate_artifact_bytes(
    *,
    layers: int,
    hidden_size: int,
    candidates: int,
) -> int:
    raw = (
        len(ALL_POSITION_NAMES) * layers * hidden_size * 2
        + len(CAPTURE_METHODS)
        * len(ALL_POSITION_NAMES)
        * layers
        * candidates
        * 4
    )
    return math.ceil(raw * 1.25) + 1_000_000


def _validate_capture_against_run(
    payload: Mapping[str, object],
    *,
    example: SyntheticBridgeExample,
    identity: Mapping[str, object],
    capture_manifest_sha256: str,
    capture_sha256: str,
) -> None:
    """Require one internally valid artifact to match the immutable run identity."""

    candidate_token_ids = identity["candidate_token_ids"]
    expected_provenance = {
        "capture_manifest_sha256": capture_manifest_sha256,
        "capture_sha256": capture_sha256,
        "git_revision": identity["git_revision"],
        "config_sha256": identity["config_sha256"],
        "test_or_smoke_read": False,
    }
    fingerprints = payload["fingerprints"]
    if not isinstance(candidate_token_ids, Mapping) or not isinstance(
        fingerprints, Mapping
    ):
        raise SweepRunError("capture run identity is malformed")
    if (
        payload["example_id"] != example.example_id
        or payload["split"] != example.split
        or payload["gold_bridge"] != example.gold_bridge
        or payload["gold_token_id"] != candidate_token_ids[example.gold_bridge]
        or payload["candidates"] != sorted(candidate_token_ids)
        or payload["candidate_token_ids"] != dict(candidate_token_ids)
        or payload["layers"] != identity["layers"]
        or payload["hidden_size"] != identity["hidden_size"]
        or fingerprints["example"] != stable_fingerprint(example.to_dict())
        or fingerprints["dataset"] != identity["dataset_sha256"]
        or fingerprints["capture"] != capture_sha256
        or payload["provenance"] != expected_provenance
    ):
        raise SweepRunError(f"capture artifact changed: {example.example_id}")


def run_capture(args: argparse.Namespace) -> dict[str, object]:
    """Run or resume bounded train/dev-only GPU capture."""

    config_path = Path(args.config)
    project_root = Path(args.project_root)
    output_dir = Path(args.artifact_dir)
    train_path = Path(args.train)
    dev_path = Path(args.dev)
    lens_path = Path(args.lens)
    config = load_config(config_path)
    sweep = config["sweep"]
    revision = _require_clean_revision(project_root)
    disk_check = build_disk_guard(
        config,
        project_root=project_root,
        environment=args.environment,
    )
    disk_check()
    train, dev = _load_train_dev(
        train_path,
        dev_path,
        expected_train=int(config["data"]["train_size"]),
        expected_dev=int(config["data"]["dev_size"]),
    )
    input_sha256 = {
        str(train_path.resolve()): sha256_file(train_path),
        str(dev_path.resolve()): sha256_file(dev_path),
    }
    dataset_sha256 = stable_fingerprint(
        {
            "schema_version": "jlens-panel-train-dev-input-v1",
            "train_sha256": input_sha256[str(train_path.resolve())],
            "dev_sha256": input_sha256[str(dev_path.resolve())],
        }
    )
    try:
        execution_identity = configure_h100_runtime()
    except H100RuntimeError as error:
        raise SweepRunError(str(error)) from error
    model_config = config["model"]
    bundle = load_model_bundle(
        model_name=model_config["name"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        device_map=model_config["device_map"],
        lens_path=lens_path,
    )
    candidate_token_ids = resolve_candidate_token_ids(
        bundle.tokenizer,
        DEFAULT_BRIDGE_CANDIDATES,
    )
    chat_sha256 = chat_template_fingerprint(bundle.tokenizer)
    if chat_sha256 != sweep["calibration"]["chat_template_sha256"]:
        raise SweepRunError("pinned chat-template fingerprint changed")
    runtime = _runtime_identity(
        config,
        bundle.tokenizer,
        execution=execution_identity,
    )
    layers = tuple(sorted(set(int(layer) for layer in bundle.lens.source_layers)))
    hidden_size = int(bundle.hf_model.config.hidden_size)
    identity = _capture_identity(
        config_sha256=sha256_file(config_path),
        revision=revision,
        model_name=model_config["name"],
        model_revision=model_config["revision"],
        lens_sha256=sha256_file(lens_path),
        input_sha256=input_sha256,
        dataset_sha256=dataset_sha256,
        candidate_token_ids=candidate_token_ids,
        layers=layers,
        hidden_size=hidden_size,
        max_seq_len=int(sweep["max_seq_len"]),
        runtime=runtime,
        eos_policy=sweep["decode"]["eos_policy"],
    )
    capture_sha256 = stable_fingerprint(identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / CAPTURE_MANIFEST_NAME
    index_path = output_dir / CAPTURE_INDEX_NAME
    progress_path = output_dir / CAPTURE_PROGRESS_NAME
    expected_counts = {"train": len(train), "dev": len(dev)}
    _ensure_capture_manifest(
        manifest_path,
        identity=identity,
        expected_counts=expected_counts,
        config_path=config_path,
        project_root=project_root,
    )
    capture_manifest_sha256 = sha256_file(manifest_path)
    examples = (*train, *dev)
    expected_paths = {
        example.example_id: capture_artifact_path(
            output_dir,
            example.split,
            example.example_id,
        )
        for example in examples
    }
    expected_relative_paths = tuple(
        expected_paths[example.example_id].relative_to(output_dir).as_posix()
        for example in examples
    )
    registered_files = set(expected_relative_paths) | {
        CAPTURE_MANIFEST_NAME,
        CAPTURE_INDEX_NAME,
        CAPTURE_PROGRESS_NAME,
    }
    unexpected_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.relative_to(output_dir).as_posix() not in registered_files
    }
    if unexpected_files:
        raise SweepRunError(
            "capture directory contains unregistered files: "
            + ", ".join(sorted(unexpected_files))
        )
    progress_hashes = _load_capture_progress(
        progress_path,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_sha256=capture_sha256,
        expected_relative_paths=expected_relative_paths,
    )
    indexed_hashes = _validate_capture_index(
        index_path,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_sha256=capture_sha256,
        progress_sha256=(sha256_file(progress_path) if progress_hashes is not None else ""),
        expected_relative_paths=expected_relative_paths,
    )
    if indexed_hashes is not None and progress_hashes != indexed_hashes:
        raise SweepRunError("capture index and progress hashes disagree")
    if progress_hashes is not None:
        for relative in progress_hashes:
            if not (output_dir / relative).is_file():
                raise SweepRunError(
                    f"capture progress references a missing artifact: {relative}"
                )

    hard_limit = int(float(config["storage"]["run_hard_limit_gb"]) * 10**9)
    if hard_limit <= _INDEX_RESERVE_BYTES:
        raise SweepRunError("run hard limit is too small for manifest reserve")
    budget = RunByteBudget.inspect(output_dir, hard_limit - _INDEX_RESERVE_BYTES)
    artifact_provenance = {
        "capture_manifest_sha256": capture_manifest_sha256,
        "capture_sha256": capture_sha256,
        "git_revision": revision,
        "config_sha256": sha256_file(config_path),
        "test_or_smoke_read": False,
    }
    artifact_hashes: dict[str, str] = {}
    adopted_orphan = False
    missing: list[SyntheticBridgeExample] = []
    for example in examples:
        path = expected_paths[example.example_id]
        relative = path.relative_to(output_dir).as_posix()
        if not path.exists():
            if indexed_hashes is not None or (
                progress_hashes is not None and relative in progress_hashes
            ):
                raise SweepRunError(f"indexed capture artifact is missing: {relative}")
            missing.append(example)
            continue
        expected_digest = (
            indexed_hashes[relative]
            if indexed_hashes is not None
            else (progress_hashes or {}).get(relative)
        )
        if expected_digest is None:
            expected_digest = sha256_file(path)
            adopted_orphan = True
        payload = load_capture_artifact(
            path,
            expected_sha256=expected_digest,
        )
        _validate_capture_against_run(
            payload,
            example=example,
            identity=identity,
            capture_manifest_sha256=capture_manifest_sha256,
            capture_sha256=capture_sha256,
        )
        artifact_hashes[relative] = sha256_file(path)

    if adopted_orphan:
        _write_capture_progress(
            progress_path,
            capture_manifest_sha256=capture_manifest_sha256,
            capture_sha256=capture_sha256,
            artifact_hashes=artifact_hashes,
            expected_relative_paths=expected_relative_paths,
        )

    estimated = _estimate_artifact_bytes(
        layers=len(layers),
        hidden_size=hidden_size,
        candidates=len(candidate_token_ids),
    )
    budget.check_addition(len(missing) * estimated)
    for index, example in enumerate(missing):
        if index and index % int(config["storage"]["disk_check_interval"]) == 0:
            disk_check()
        residuals, scores, captured_layers = capture_task_example(
            bundle,
            example,
            candidate_token_ids=candidate_token_ids,
            max_seq_len=int(sweep["max_seq_len"]),
        )
        if captured_layers != layers:
            raise SweepRunError("capture source layers changed during the run")
        artifact = build_capture_artifact(
            example_id=example.example_id,
            split=example.split,
            candidates=DEFAULT_BRIDGE_CANDIDATES,
            candidate_token_ids=candidate_token_ids,
            gold_bridge=example.gold_bridge,
            layers=layers,
            hidden_size=hidden_size,
            residuals=residuals,
            scores=scores,
            example_fingerprint=stable_fingerprint(example.to_dict()),
            dataset_fingerprint=dataset_sha256,
            capture_fingerprint=capture_sha256,
            provenance=artifact_provenance,
        )
        path = expected_paths[example.example_id]
        saved, digest = save_capture_artifact_atomic(
            path,
            artifact,
            run_root=output_dir,
            byte_budget=budget,
        )
        artifact_hashes[saved.relative_to(output_dir).as_posix()] = digest
        _write_capture_progress(
            progress_path,
            capture_manifest_sha256=capture_manifest_sha256,
            capture_sha256=capture_sha256,
            artifact_hashes=artifact_hashes,
            expected_relative_paths=expected_relative_paths,
        )
    disk_check()
    if set(artifact_hashes) != set(expected_relative_paths):
        raise SweepRunError("capture artifact index is incomplete")
    ordered_hashes = {
        relative: artifact_hashes[relative] for relative in expected_relative_paths
    }
    if not progress_path.exists():
        _write_capture_progress(
            progress_path,
            capture_manifest_sha256=capture_manifest_sha256,
            capture_sha256=capture_sha256,
            artifact_hashes=ordered_hashes,
            expected_relative_paths=expected_relative_paths,
        )
    progress_sha256 = sha256_file(progress_path)
    index = {
        "schema_version": CAPTURE_INDEX_SCHEMA,
        "capture_manifest_sha256": capture_manifest_sha256,
        "capture_sha256": capture_sha256,
        "progress_sha256": progress_sha256,
        "artifacts": ordered_hashes,
        "counts": expected_counts,
        "test_or_smoke_read": False,
    }
    if index_path.exists():
        if _load_json_mapping(index_path, name="capture index") != index:
            raise SweepRunError("existing capture index changed")
    else:
        write_json_atomic(index_path, index)
    if sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()) > (
        hard_limit
    ):
        raise SweepRunError("capture run exceeded the 20 GB hard limit")
    return {
        "capture_sha256": capture_sha256,
        "written": len(missing),
        "reused": len(examples) - len(missing),
        "artifacts": len(artifact_hashes),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": capture_manifest_sha256,
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "progress": str(progress_path.resolve()),
        "progress_sha256": progress_sha256,
    }


def _load_calibration_collection(
    calibration_dir: Path,
    *,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    layers: Sequence[int],
    config_sha256: str,
    revision: str,
    capture_identity: Mapping[str, object],
) -> tuple[dict[tuple[str, str, int], object], str]:
    manifest_path = calibration_dir / "calibration_manifest.json"
    manifest = _load_json_mapping(manifest_path, name="calibration manifest")
    if manifest.get("config_sha256") != config_sha256:
        raise SweepRunError("calibration and sweep config SHA disagree")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        raise SweepRunError("calibration manifest extra is invalid")
    identity = extra.get("identity")
    artifact_hashes = extra.get("artifacts")
    if not isinstance(identity, Mapping) or not isinstance(artifact_hashes, Mapping):
        raise SweepRunError("calibration manifest identity or artifacts are invalid")
    capture_runtime = capture_identity.get("runtime")
    if not isinstance(capture_runtime, Mapping):
        raise SweepRunError("capture runtime identity is invalid")
    cross_identity = {
        "model_name": capture_identity.get("model_name"),
        "model_revision": capture_identity.get("model_revision"),
        "lens_sha256": capture_identity.get("lens_sha256"),
        "max_seq_len": capture_identity.get("max_seq_len"),
        "eos_policy": capture_identity.get("eos_policy"),
        "upstream_commit": capture_runtime.get("upstream_commit"),
        "jlens_source_sha256": capture_runtime.get("jlens_source_sha256"),
        "transformers_version": capture_runtime.get("transformers_version"),
        "jlens_version": capture_runtime.get("jlens_version"),
        "chat_template_sha256": capture_runtime.get("chat_template_sha256"),
        "torch_version": capture_runtime.get("torch_version"),
        "cuda_runtime": capture_runtime.get("cuda_runtime"),
        "cuda_driver_version": capture_runtime.get("cuda_driver_version"),
        "gpu_name": capture_runtime.get("gpu_name"),
        "gpu_compute_capability": capture_runtime.get("gpu_compute_capability"),
        "deterministic_algorithms": capture_runtime.get(
            "deterministic_algorithms"
        ),
        "allow_tf32": capture_runtime.get("allow_tf32"),
        "cublas_workspace_config": capture_runtime.get(
            "cublas_workspace_config"
        ),
    }
    if (
        identity.get("git_revision") != revision
        or identity.get("candidate_token_ids") != dict(candidate_token_ids)
        or identity.get("source_layers") != list(layers)
        or identity.get("null_prompt_count") != 200
        or extra.get("test_or_smoke_read") is not False
        or any(identity.get(key) != value for key, value in cross_identity.items())
    ):
        raise SweepRunError("calibration collection disagrees with sweep identity")
    expected_names = {
        f"{method}__{position}__layer_{layer:02d}.json"
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in layers
    }
    if set(artifact_hashes) != expected_names:
        raise SweepRunError("calibration artifact cell inventory changed")
    paths = tuple(calibration_dir / name for name in sorted(expected_names))
    expected_sha = {
        str(path.resolve()): artifact_hashes[path.name] for path in paths
    }
    calibrations = load_calibrations(paths, expected_sha256=expected_sha)
    if tuple(sorted(candidates)) != next(iter(calibrations.values())).candidates:
        raise SweepRunError("calibration candidate ordering changed")
    if any(
        calibration.to_payload()["provenance"] != dict(identity)
        for calibration in calibrations.values()
    ):
        raise SweepRunError("calibration artifacts disagree with manifest identity")
    return calibrations, sha256_file(manifest_path)


def _load_capture_arrays(
    artifact_dir: Path,
    *,
    train: Sequence[SyntheticBridgeExample],
    dev: Sequence[SyntheticBridgeExample],
) -> tuple[
    dict[tuple[str, int], object],
    dict[tuple[str, int], object],
    dict[tuple[str, str, int], object],
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, int],
    tuple[int, ...],
    str,
    str,
]:
    import torch

    manifest_path = artifact_dir / CAPTURE_MANIFEST_NAME
    index_path = artifact_dir / CAPTURE_INDEX_NAME
    progress_path = artifact_dir / CAPTURE_PROGRESS_NAME
    manifest = _load_json_mapping(manifest_path, name="capture manifest")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or extra.get("schema_version") != (
        CAPTURE_RUN_SCHEMA
    ):
        raise SweepRunError("capture manifest schema changed")
    identity = extra["identity"]
    if not isinstance(identity, Mapping):
        raise SweepRunError("capture identity is invalid")
    capture_sha256 = extra["capture_sha256"]
    layers = tuple(identity["layers"])
    candidates = tuple(sorted(identity["candidate_token_ids"]))
    candidate_token_ids = dict(identity["candidate_token_ids"])
    hidden_size = int(identity["hidden_size"])
    examples = (*train, *dev)
    relative_paths = tuple(
        capture_artifact_path(
            artifact_dir,
            example.split,
            example.example_id,
        )
        .relative_to(artifact_dir)
        .as_posix()
        for example in examples
    )
    progress_hashes = _load_capture_progress(
        progress_path,
        capture_manifest_sha256=sha256_file(manifest_path),
        capture_sha256=str(capture_sha256),
        expected_relative_paths=relative_paths,
    )
    if progress_hashes is None:
        raise SweepRunError("capture analysis requires a complete progress index")
    hashes = _validate_capture_index(
        index_path,
        capture_manifest_sha256=sha256_file(manifest_path),
        capture_sha256=str(capture_sha256),
        progress_sha256=sha256_file(progress_path),
        expected_relative_paths=relative_paths,
    )
    if hashes is None:
        raise SweepRunError("capture analysis requires a complete artifact index")
    if hashes != progress_hashes:
        raise SweepRunError("capture progress and final index disagree")

    split_residuals = {
        "train": torch.empty(
            (len(train), len(ALL_POSITION_NAMES), len(layers), hidden_size),
            dtype=torch.float16,
        ),
        "dev": torch.empty(
            (len(dev), len(ALL_POSITION_NAMES), len(layers), hidden_size),
            dtype=torch.float16,
        ),
    }
    dev_scores = torch.empty(
        (
            len(CAPTURE_METHODS),
            len(dev),
            len(ALL_POSITION_NAMES),
            len(layers),
            len(candidates),
        ),
        dtype=torch.float32,
    )
    targets: dict[str, list[int]] = {"train": [], "dev": []}
    split_offsets = {"train": 0, "dev": 0}
    for example in examples:
        split = example.split
        row_index = split_offsets[split]
        split_offsets[split] += 1
        path = capture_artifact_path(artifact_dir, split, example.example_id)
        relative = path.relative_to(artifact_dir).as_posix()
        artifact = load_capture_artifact(path, expected_sha256=hashes[relative])
        _validate_capture_against_run(
            artifact,
            example=example,
            identity=identity,
            capture_manifest_sha256=sha256_file(manifest_path),
            capture_sha256=str(capture_sha256),
        )
        targets[split].append(int(artifact["gold_token_id"]))
        for position_index, position in enumerate(ALL_POSITION_NAMES):
            split_residuals[split][row_index, position_index].copy_(
                artifact["residuals"][position]
            )
            if split == "dev":
                for method_index, method in enumerate(CAPTURE_METHODS):
                    dev_scores[method_index, row_index, position_index].copy_(
                        artifact["scores"][method][position]
                    )

    train_cells = {
        (position, layer): split_residuals["train"][:, position_index, layer_index]
        .float()
        .numpy()
        for position_index, position in enumerate(ALL_POSITION_NAMES)
        for layer_index, layer in enumerate(layers)
    }
    dev_cells = {
        (position, layer): split_residuals["dev"][:, position_index, layer_index]
        .float()
        .numpy()
        for position_index, position in enumerate(ALL_POSITION_NAMES)
        for layer_index, layer in enumerate(layers)
    }
    score_cells = {
        (method, position, layer): dev_scores[
            method_index, :, position_index, layer_index
        ].numpy()
        for method_index, method in enumerate(CAPTURE_METHODS)
        for position_index, position in enumerate(ALL_POSITION_NAMES)
        for layer_index, layer in enumerate(layers)
    }
    return (
        train_cells,
        dev_cells,
        score_cells,
        tuple(targets["train"]),
        tuple(targets["dev"]),
        tuple(example.example_id for example in dev),
        candidates,
        candidate_token_ids,
        layers,
        sha256_file(manifest_path),
        sha256_file(index_path),
    )


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_capture_file_hashes(
    artifact_dir: Path,
    *,
    examples: Sequence[SyntheticBridgeExample],
    capture_manifest_sha256: str,
    capture_sha256: str,
) -> tuple[str, str]:
    relative_paths = tuple(
        capture_artifact_path(
            artifact_dir,
            example.split,
            example.example_id,
        )
        .relative_to(artifact_dir)
        .as_posix()
        for example in examples
    )
    progress_path = artifact_dir / CAPTURE_PROGRESS_NAME
    progress = _load_capture_progress(
        progress_path,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_sha256=capture_sha256,
        expected_relative_paths=relative_paths,
    )
    if progress is None:
        raise SweepRunError("capture progress is missing")
    index_path = artifact_dir / CAPTURE_INDEX_NAME
    hashes = _validate_capture_index(
        index_path,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_sha256=capture_sha256,
        progress_sha256=sha256_file(progress_path),
        expected_relative_paths=relative_paths,
    )
    if hashes is None or hashes != progress:
        raise SweepRunError("capture progress/index is incomplete")
    for relative, expected_sha256 in hashes.items():
        if sha256_file(artifact_dir / relative) != expected_sha256:
            raise SweepRunError(f"capture artifact SHA changed: {relative}")
    return sha256_file(index_path), sha256_file(progress_path)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Load complete capture/calibration artifacts and produce frozen dev outputs."""

    config_path = Path(args.config)
    project_root = Path(args.project_root)
    config = load_config(config_path)
    revision = _require_clean_revision(project_root)
    train, dev = _load_train_dev(
        Path(args.train),
        Path(args.dev),
        expected_train=int(config["data"]["train_size"]),
        expected_dev=int(config["data"]["dev_size"]),
    )
    results_path = Path(args.results)
    heatmap_path = Path(args.heatmap)
    analysis_manifest_path = Path(args.analysis_manifest)
    capture_manifest_path = Path(args.artifact_dir) / CAPTURE_MANIFEST_NAME
    capture_index_path = Path(args.artifact_dir) / CAPTURE_INDEX_NAME
    calibration_manifest_path = (
        Path(args.calibration_dir) / "calibration_manifest.json"
    )
    capture_manifest = _load_json_mapping(
        capture_manifest_path,
        name="capture manifest",
    )
    capture_extra = capture_manifest.get("extra")
    if not isinstance(capture_extra, Mapping) or not isinstance(
        capture_extra.get("identity"), Mapping
    ):
        raise SweepRunError("capture manifest identity is invalid")
    capture_identity = capture_extra["identity"]
    current_inputs = {
        str(Path(args.train).resolve()): sha256_file(args.train),
        str(Path(args.dev).resolve()): sha256_file(args.dev),
    }
    if (
        capture_identity.get("git_revision") != revision
        or capture_identity.get("config_sha256") != sha256_file(config_path)
        or capture_identity.get("input_sha256") != current_inputs
        or capture_identity.get("test_or_smoke_read") is not False
    ):
        raise SweepRunError("capture identity disagrees with current analysis inputs")
    current_capture_manifest_sha256 = sha256_file(capture_manifest_path)
    current_capture_index_sha256 = sha256_file(capture_index_path)
    current_calibration_manifest_sha256 = sha256_file(calibration_manifest_path)
    if analysis_manifest_path.exists():
        manifest = _load_json_mapping(
            analysis_manifest_path,
            name="sweep analysis manifest",
        )
        extra = manifest.get("extra")
        if not isinstance(extra, Mapping) or extra.get("schema_version") != (
            ANALYSIS_MANIFEST_SCHEMA
        ):
            raise SweepRunError("existing analysis manifest schema changed")
        if (
            manifest.get("config_sha256") != sha256_file(config_path)
            or manifest.get("git_revision") != revision
            or extra.get("capture_manifest_sha256")
            != current_capture_manifest_sha256
            or extra.get("capture_index_sha256") != current_capture_index_sha256
            or extra.get("calibration_manifest_sha256")
            != current_calibration_manifest_sha256
            or extra.get("test_or_smoke_read") is not False
            or sha256_file(results_path) != extra.get("results_sha256")
            or sha256_file(heatmap_path) != extra.get("heatmap_sha256")
        ):
            raise SweepRunError("existing sweep analysis outputs changed")
        _verify_capture_file_hashes(
            Path(args.artifact_dir),
            examples=(*train, *dev),
            capture_manifest_sha256=current_capture_manifest_sha256,
            capture_sha256=str(capture_extra["capture_sha256"]),
        )
        reuse_candidates = tuple(sorted(capture_identity["candidate_token_ids"]))
        reuse_candidate_token_ids = dict(capture_identity["candidate_token_ids"])
        reuse_layers = tuple(capture_identity["layers"])
        _load_calibration_collection(
            Path(args.calibration_dir),
            candidates=reuse_candidates,
            candidate_token_ids=reuse_candidate_token_ids,
            layers=reuse_layers,
            config_sha256=sha256_file(config_path),
            revision=revision,
            capture_identity=capture_identity,
        )
        heatmap = _load_json_mapping(heatmap_path, name="sweep heatmap")
        if heatmap.get("gates") != extra.get("gates"):
            raise SweepRunError("analysis manifest gates disagree with heatmap")
        with results_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                raise SweepRunError("existing sweep CSV header changed")
            row_count = sum(1 for _row in reader)
        if row_count != extra.get("row_count"):
            raise SweepRunError("analysis manifest row count disagrees with CSV")
        return {
            "reused": True,
            "results": str(results_path.resolve()),
            "results_sha256": extra["results_sha256"],
            "heatmap": str(heatmap_path.resolve()),
            "heatmap_sha256": extra["heatmap_sha256"],
            "manifest": str(analysis_manifest_path.resolve()),
            "manifest_sha256": sha256_file(analysis_manifest_path),
            "gates": extra["gates"],
        }
    (
        train_residuals,
        dev_residuals,
        dev_scores,
        train_targets,
        dev_targets,
        dev_ids,
        candidates,
        candidate_token_ids,
        layers,
        capture_manifest_sha256,
        capture_index_sha256,
    ) = _load_capture_arrays(
        Path(args.artifact_dir),
        train=train,
        dev=dev,
    )
    calibrations, calibration_manifest_sha256 = _load_calibration_collection(
        Path(args.calibration_dir),
        candidates=candidates,
        candidate_token_ids=candidate_token_ids,
        layers=layers,
        config_sha256=sha256_file(config_path),
        revision=revision,
        capture_identity=capture_identity,
    )
    rows, heatmap, gates = run_probe_sweep(
        candidates=candidates,
        candidate_token_ids=candidate_token_ids,
        layers=layers,
        train_residuals=train_residuals,
        dev_residuals=dev_residuals,
        dev_scores=dev_scores,
        train_target_token_ids=train_targets,
        dev_target_token_ids=dev_targets,
        dev_example_ids=dev_ids,
        calibrations=calibrations,  # type: ignore[arg-type]
        c_values=config["sweep"]["probe"]["c_values"],
        gates=config["sweep"]["gates"],
    )
    serialized_rows = tuple(row.to_dict() for row in rows)
    _write_csv_atomic(results_path, serialized_rows)
    write_json_atomic(heatmap_path, heatmap)
    manifest = build_manifest(
        config_path=config_path,
        project_root=project_root,
        extra={
            "schema_version": ANALYSIS_MANIFEST_SCHEMA,
            "capture_manifest_sha256": capture_manifest_sha256,
            "capture_index_sha256": capture_index_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "results": str(results_path.resolve()),
            "results_sha256": sha256_file(results_path),
            "heatmap": str(heatmap_path.resolve()),
            "heatmap_sha256": sha256_file(heatmap_path),
            "row_count": len(serialized_rows),
            "gates": gates,
            "test_or_smoke_read": False,
        },
    )
    write_json_atomic(analysis_manifest_path, manifest)
    return {
        "reused": False,
        "results": str(results_path.resolve()),
        "results_sha256": sha256_file(results_path),
        "heatmap": str(heatmap_path.resolve()),
        "heatmap_sha256": sha256_file(heatmap_path),
        "manifest": str(analysis_manifest_path.resolve()),
        "manifest_sha256": sha256_file(analysis_manifest_path),
        "gates": gates,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and analyze the train/dev position-by-layer sweep."
    )
    parser.add_argument("--phase", choices=("capture", "analyze", "all"), default="all")
    parser.add_argument("--config", default="config/sprint.yaml")
    parser.add_argument("--train", default="data/generated/dirty_v3/train.jsonl")
    parser.add_argument("--dev", default="data/generated/dirty_v3/dev.jsonl")
    parser.add_argument(
        "--lens",
        default="checkpoints/qwen2.5-7b-instruct-jlens.pt",
    )
    parser.add_argument("--artifact-dir", default="artifacts/sweep_v3")
    parser.add_argument("--calibration-dir", default="artifacts/calibration")
    parser.add_argument("--results", default="analysis/sweep_results.csv")
    parser.add_argument("--heatmap", default="analysis/sweep_heatmap.json")
    parser.add_argument(
        "--analysis-manifest",
        default="analysis/sweep_manifest.json",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", choices=("local", "tempest"), default="local")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    lock_path = artifact_dir.parent / f".{artifact_dir.name}.lock"
    output: dict[str, object] = {}
    with exclusive_run_lock(lock_path):
        if args.phase in ("capture", "all"):
            output["capture"] = run_capture(args)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        if args.phase in ("analyze", "all"):
            output["analysis"] = run_analysis(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
