#!/usr/bin/env python3
"""Capture and persist first-class null calibrations for every sweep cell."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens_panel.calibration import (
    NULL_RENDERING_POLICY,
    POSITION_RESOLVER_SCHEMA,
    CalibrationArtifactError,
    load_calibrations,
)
from jlens_panel.config import load_config
from jlens_panel.corpus import (
    load_prompts,
    sample_prompts,
    sampled_prompt_fingerprint,
    sampled_prompt_manifest,
)
from jlens_panel.data import DEFAULT_BRIDGE_CANDIDATES
from jlens_panel.modeling import load_model_bundle, resolve_candidate_token_ids
from jlens_panel.provenance import (
    build_manifest,
    git_revision,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.storage import build_disk_guard
from jlens_panel.sweep.capture import (
    CAPTURE_METHODS,
    capture_null_calibrations,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES, DECODE_STEPS

COLLECTION_SCHEMA = "jlens-panel-null-calibration-collection-v1"


class NullCalibrationRunError(RuntimeError):
    """Raised when a null-calibration run conflicts with immutable state."""


def calibration_filename(method: str, position: str, layer: int) -> str:
    """Return the stable artifact filename for one calibration cell."""

    if method not in CAPTURE_METHODS or position not in ALL_POSITION_NAMES:
        raise NullCalibrationRunError("invalid calibration cell name")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise NullCalibrationRunError("calibration layer must be non-negative")
    return f"{method}__{position}__layer_{layer:02d}.json"


def expected_calibration_filenames(layers: Sequence[int]) -> tuple[str, ...]:
    """Enumerate every required method-position-layer artifact."""

    normalized_layers = tuple(sorted(set(layers)))
    if not normalized_layers:
        raise NullCalibrationRunError("source layers cannot be empty")
    return tuple(
        calibration_filename(method, position, layer)
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in normalized_layers
    )


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NullCalibrationRunError(f"cannot decode manifest: {path}") from error
    if not isinstance(value, dict):
        raise NullCalibrationRunError("calibration manifest must be a mapping")
    return value


def _reuse_complete_collection(
    manifest_path: Path,
    *,
    config_sha256: str,
    output_dir: Path,
    expected_static_identity: Mapping[str, object],
    sample_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifest = _load_json_mapping(manifest_path)
    if manifest.get("config_sha256") != config_sha256:
        raise NullCalibrationRunError("existing calibration config SHA changed")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or set(extra) != {
        "schema_version",
        "identity",
        "sample",
        "artifacts",
        "test_or_smoke_read",
    }:
        raise NullCalibrationRunError("existing calibration manifest fields changed")
    if extra["schema_version"] != COLLECTION_SCHEMA:
        raise NullCalibrationRunError("existing calibration collection schema changed")
    if extra["test_or_smoke_read"] is not False:
        raise NullCalibrationRunError("existing calibration touched forbidden inputs")
    if extra["sample"] != list(sample_records):
        raise NullCalibrationRunError("existing null-prompt sample changed")
    identity = extra["identity"]
    if not isinstance(identity, Mapping):
        raise NullCalibrationRunError("existing calibration identity is invalid")
    for key, expected in expected_static_identity.items():
        if identity.get(key) != expected:
            raise NullCalibrationRunError(
                f"existing calibration identity changed at {key}"
            )
    layers = identity.get("source_layers")
    if (
        isinstance(layers, (str, bytes))
        or not isinstance(layers, Sequence)
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
    ):
        raise NullCalibrationRunError("existing source layers are invalid")
    expected_names = expected_calibration_filenames(layers)
    artifact_hashes = extra["artifacts"]
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(
        expected_names
    ):
        raise NullCalibrationRunError("existing calibration artifact inventory changed")
    paths = tuple(output_dir / name for name in expected_names)
    expected_sha = {
        str(path.resolve()): artifact_hashes[path.name] for path in paths
    }
    calibrations = load_calibrations(paths, expected_sha256=expected_sha)
    if len(calibrations) != len(expected_names) or any(
        calibration.n_null_prompts != 200 for calibration in calibrations.values()
    ):
        raise NullCalibrationRunError("existing calibration collection is incomplete")
    return {
        "reused": True,
        "artifacts": len(calibrations),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture 200-prompt null calibrations for all sweep cells."
    )
    parser.add_argument("--config", default="config/sprint.yaml")
    parser.add_argument("--prompts")
    parser.add_argument(
        "--lens",
        default="checkpoints/qwen2.5-7b-instruct-jlens.pt",
    )
    parser.add_argument("--output-dir", default="artifacts/calibration")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", choices=("local", "tempest"), default="local")
    parser.add_argument("--disk-check-every", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    sweep = config["sweep"]
    calibration_config = sweep["calibration"]
    prompts_path = Path(args.prompts or calibration_config["corpus"])
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "calibration_manifest.json"
    project_root = Path(args.project_root)
    disk_check_every = (
        args.disk_check_every
        if args.disk_check_every is not None
        else int(config["storage"]["disk_check_interval"])
    )
    if disk_check_every < 1:
        raise NullCalibrationRunError("disk check interval must be positive")

    disk_check = build_disk_guard(
        config,
        project_root=project_root,
        environment=args.environment,
    )
    disk_check()
    prompts = load_prompts(prompts_path)
    sampled = sample_prompts(
        prompts,
        count=int(calibration_config["null_prompts"]),
        seed=int(calibration_config["sample_seed"]),
    )
    sample_records = sampled_prompt_manifest(sampled)
    sample_sha256 = sampled_prompt_fingerprint(sampled)
    config_sha256 = sha256_file(args.config)
    lens_sha256 = sha256_file(args.lens)
    corpus_sha256 = sha256_file(prompts_path)
    revision = git_revision(project_root)
    if revision is None:
        raise NullCalibrationRunError("null calibration requires a Git revision")
    static_identity = {
        "model_name": config["model"]["name"],
        "model_revision": config["model"]["revision"],
        "config_sha256": config_sha256,
        "git_revision": revision,
        "lens_sha256": lens_sha256,
        "corpus_sha256": corpus_sha256,
        "sample_sha256": sample_sha256,
        "sample_seed": int(calibration_config["sample_seed"]),
        "null_prompt_count": len(sampled),
        "max_seq_len": int(sweep["max_seq_len"]),
        "decode_steps": list(DECODE_STEPS),
        "ddof": int(calibration_config["ddof"]),
        "resolver_schema": POSITION_RESOLVER_SCHEMA,
        "rendering_policy": dict(NULL_RENDERING_POLICY),
    }
    if manifest_path.exists():
        result = _reuse_complete_collection(
            manifest_path,
            config_sha256=config_sha256,
            output_dir=output_dir,
            expected_static_identity=static_identity,
            sample_records=sample_records,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NullCalibrationRunError(
            "calibration output is partial and has no manifest; use a new output path"
        )

    model_config = config["model"]
    bundle = load_model_bundle(
        model_name=model_config["name"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        device_map=model_config["device_map"],
        lens_path=args.lens,
    )
    candidate_token_ids = resolve_candidate_token_ids(
        bundle.tokenizer,
        DEFAULT_BRIDGE_CANDIDATES,
    )
    source_layers = sorted(set(int(layer) for layer in bundle.lens.source_layers))
    identity = {
        **static_identity,
        "candidate_token_ids": candidate_token_ids,
        "source_layers": source_layers,
    }
    calibrations = capture_null_calibrations(
        bundle,
        sampled,
        candidate_token_ids=candidate_token_ids,
        max_seq_len=int(sweep["max_seq_len"]),
        provenance=identity,
        disk_check_every=disk_check_every,
        disk_check=disk_check,
    )
    expected_names = expected_calibration_filenames(source_layers)
    if len(calibrations) != len(expected_names):
        raise NullCalibrationRunError("captured calibration cell count changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes: dict[str, str] = {}
    for calibration, name in zip(calibrations, expected_names, strict=True):
        artifact_hashes[name] = calibration.save(output_dir / name)
    manifest = build_manifest(
        config_path=args.config,
        project_root=project_root,
        extra={
            "schema_version": COLLECTION_SCHEMA,
            "identity": identity,
            "sample": list(sample_records),
            "artifacts": artifact_hashes,
            "test_or_smoke_read": False,
        },
    )
    write_json_atomic(manifest_path, manifest)
    disk_check()
    result = {
        "reused": False,
        "artifacts": len(artifact_hashes),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
