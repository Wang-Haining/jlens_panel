import importlib.util
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
    write_json_atomic(
        manifest_path,
        {
            "config_sha256": "1" * 64,
            "extra": {
                "schema_version": CALIBRATE_NULL.COLLECTION_SCHEMA,
                "identity": identity,
                "sample": sample,
                "artifacts": hashes,
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
