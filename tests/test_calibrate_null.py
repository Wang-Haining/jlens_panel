import importlib.util
import json
from pathlib import Path

import pytest

from jlens_panel.calibration import (
    NULL_RENDERING_POLICY,
    POSITION_RESOLVER_SCHEMA,
    NullCalibration,
)
from jlens_panel.provenance import write_json_atomic
from jlens_panel.sweep.capture import CAPTURE_METHODS
from jlens_panel.sweep.positions import ALL_POSITION_NAMES

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_null_for_test",
    ROOT / "scripts" / "calibrate_null.py",
)
assert SPEC is not None and SPEC.loader is not None
CALIBRATE_NULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATE_NULL)


def provenance() -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a" * 40,
        "config_sha256": "1" * 64,
        "git_revision": "2" * 40,
        "upstream_commit": "3" * 40,
        "jlens_source_sha256": "4" * 64,
        "transformers_version": "5.13.0",
        "jlens_version": "0.1.0",
        "torch_version": "2.9.1+cu128",
        "cuda_runtime": "12.8",
        "cuda_driver_version": "570.00",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_compute_capability": [9, 0],
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "chat_template_sha256": "5" * 64,
        "eos_policy": "mask_eos_for_tokens_1_through_7",
        "lens_sha256": "b" * 64,
        "corpus_sha256": "c" * 64,
        "sample_sha256": "d" * 64,
        "sample_seed": 20260711,
        "null_prompt_count": 200,
        "candidate_token_ids": {"amber": 10, "bamboo": 11},
        "source_layers": [4, 7],
        "max_seq_len": 512,
        "decode_steps": [1, 2, 4, 8],
        "ddof": 0,
        "resolver_schema": POSITION_RESOLVER_SCHEMA,
        "rendering_policy": dict(NULL_RENDERING_POLICY),
    }


def test_expected_calibration_inventory_is_complete_and_stable() -> None:
    names = CALIBRATE_NULL.expected_calibration_filenames([7, 4, 7])

    assert len(names) == 2 * 8 * 2
    assert names[0] == "jlens__template_tail__layer_04.json"
    assert names[-1] == "logit_lens__decode_8__layer_07.json"
    with pytest.raises(CALIBRATE_NULL.NullCalibrationRunError, match="layer"):
        CALIBRATE_NULL.calibration_filename("jlens", "template_tail", True)


def test_complete_collection_reuse_verifies_every_artifact_sha(tmp_path: Path) -> None:
    output = tmp_path / "calibration"
    output.mkdir()
    identity = provenance()
    names = CALIBRATE_NULL.expected_calibration_filenames([4, 7])
    hashes: dict[str, str] = {}
    calibrations = (
        NullCalibration(
            method=method,
            position_type=position,
            layer=layer,
            candidate_means={"amber": 1.0, "bamboo": 2.0},
            candidate_stds={"amber": 0.5, "bamboo": 0.75},
            n_null_prompts=200,
            provenance=identity,
        )
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in (4, 7)
    )
    for name, calibration in zip(names, calibrations, strict=True):
        hashes[name] = calibration.save(output / name)
    sample = [{"sample_index": 0, "corpus_index": 7, "text_sha256": "e" * 64}]
    manifest_path = output / "calibration_manifest.json"
    checkpoint_path = output / CALIBRATE_NULL.CHECKPOINT_NAME
    write_json_atomic(checkpoint_path, {"completed_prompts": 200})
    write_json_atomic(
        manifest_path,
        {
            "config_sha256": "1" * 64,
            "extra": {
                "schema_version": CALIBRATE_NULL.COLLECTION_SCHEMA,
                "identity": identity,
                "sample": sample,
                "artifacts": hashes,
                "checkpoint": {
                    "path": CALIBRATE_NULL.CHECKPOINT_NAME,
                    "sha256": CALIBRATE_NULL.sha256_file(checkpoint_path),
                },
                "test_or_smoke_read": False,
            },
        },
    )

    result = CALIBRATE_NULL._reuse_complete_collection(
        manifest_path,
        config_sha256="1" * 64,
        output_dir=output,
        expected_static_identity=identity,
        sample_records=sample,
    )

    assert result["reused"] is True
    assert result["artifacts"] == 32
    mismatched_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatched_manifest["extra"]["identity"]["model_name"] = "claimed/model"
    write_json_atomic(manifest_path, mismatched_manifest)
    claimed_identity = dict(identity)
    claimed_identity["model_name"] = "claimed/model"
    with pytest.raises(
        CALIBRATE_NULL.NullCalibrationRunError,
        match="artifacts disagree",
    ):
        CALIBRATE_NULL._reuse_complete_collection(
            manifest_path,
            config_sha256="1" * 64,
            output_dir=output,
            expected_static_identity=claimed_identity,
            sample_records=sample,
        )
    mismatched_manifest["extra"]["identity"] = identity
    write_json_atomic(manifest_path, mismatched_manifest)
    first = output / names[0]
    first.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(Exception, match="SHA-256|decode"):
        CALIBRATE_NULL._reuse_complete_collection(
            manifest_path,
            config_sha256="1" * 64,
            output_dir=output,
            expected_static_identity=identity,
            sample_records=sample,
        )


def test_checkpoint_binds_identity_and_state_sha(tmp_path: Path) -> None:
    path = tmp_path / CALIBRATE_NULL.CHECKPOINT_NAME
    identity_sha256 = "a" * 64
    state = {"completed_prompts": 25, "accumulators": {}}

    CALIBRATE_NULL._write_checkpoint(
        path,
        identity_sha256=identity_sha256,
        state=state,
    )

    assert (
        CALIBRATE_NULL._load_checkpoint(
            path,
            identity_sha256=identity_sha256,
        )
        == state
    )
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["state"]["completed_prompts"] = 26
    write_json_atomic(path, checkpoint)
    with pytest.raises(
        CALIBRATE_NULL.NullCalibrationRunError,
        match="state SHA",
    ):
        CALIBRATE_NULL._load_checkpoint(
            path,
            identity_sha256=identity_sha256,
        )
