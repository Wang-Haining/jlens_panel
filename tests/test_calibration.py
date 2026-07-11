import json
from pathlib import Path

import pytest

from jlens_panel.calibration import (
    NULL_RENDERING_POLICY,
    POSITION_RESOLVER_SCHEMA,
    CalibrationArtifactError,
    CalibrationError,
    NullCalibration,
    NullScoreAccumulator,
    calibrate,
    load_calibrations,
)


def provenance(
    *,
    n_null_prompts: int = 200,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict[str, object]:
    return {
        "model_name": model_name,
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
        "null_prompt_count": n_null_prompts,
        "candidate_token_ids": {"amber": 10, "bamboo": 11},
        "source_layers": [4, 7],
        "max_seq_len": 512,
        "decode_steps": [1, 2, 4, 8],
        "ddof": 0,
        "resolver_schema": POSITION_RESOLVER_SCHEMA,
        "rendering_policy": dict(NULL_RENDERING_POLICY),
    }


def calibration(**overrides: object) -> NullCalibration:
    values: dict[str, object] = {
        "method": "jlens",
        "position_type": "template_tail",
        "layer": 4,
        "candidate_means": {"amber": 1.0, "bamboo": -2.0},
        "candidate_stds": {"amber": 2.0, "bamboo": 0.0},
        "n_null_prompts": 200,
        "provenance": provenance(),
    }
    values.update(overrides)
    return NullCalibration(**values)  # type: ignore[arg-type]


def test_center_and_zscore_use_exact_support_and_std_floor() -> None:
    cal = calibration()

    assert calibrate({"amber": 5.0, "bamboo": 1.0}, cal) == {
        "amber": 4.0,
        "bamboo": 3.0,
    }
    assert calibrate(
        {"amber": 5.0, "bamboo": -1.999999}, cal, mode="zscore"
    ) == pytest.approx({"amber": 2.0, "bamboo": 1.0})

    assert calibrate({"bamboo": 1.0, "amber": 5.0}, cal) == {
        "amber": 4.0,
        "bamboo": 3.0,
    }
    with pytest.raises(CalibrationError, match="unsupported calibration mode"):
        calibrate({"amber": 5.0, "bamboo": 1.0}, cal, mode="scale")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "next_token"}, "unsupported calibration method"),
        ({"position_type": "unknown"}, "unsupported calibration position"),
        ({"layer": True}, "layer must be an integer"),
        ({"n_null_prompts": 0}, "positive integer"),
        (
            {"candidate_stds": {"amber": 2.0, "bamboo": -1.0}},
            "non-negative",
        ),
        (
            {"candidate_means": {"amber": 1.0, "bamboo": float("nan")}},
            "finite",
        ),
        ({"candidate_stds": {"amber": 2.0, "copper": 1.0}}, "disagree"),
        ({"provenance": {}}, "provenance fields"),
    ],
)
def test_null_calibration_fails_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(CalibrationError, match=message):
        calibration(**overrides)


@pytest.mark.parametrize(
    "changes",
    [
        {"null_prompt_count": True},
        {"ddof": False},
        {"ddof": 0.0},
        {"decode_steps": [True, 2, 4, 8]},
        {"decode_steps": [1, 2.0, 4, 8]},
    ],
)
def test_provenance_rejects_bool_and_float_integer_aliases(
    changes: dict[str, object],
) -> None:
    invalid = provenance(n_null_prompts=1)
    invalid.update(changes)

    with pytest.raises(CalibrationError, match="null_prompt_count|ddof|decode_steps"):
        calibration(n_null_prompts=1, provenance=invalid)


def test_accumulator_builds_population_moments() -> None:
    accumulator = NullScoreAccumulator(("amber", "bamboo"))
    accumulator.update({"amber": 1.0, "bamboo": 4.0})
    accumulator.update({"amber": 3.0, "bamboo": 8.0})

    cal = accumulator.build(
        method="logit_lens",
        position_type="decode_2",
        layer=7,
        provenance=provenance(n_null_prompts=2),
    )

    assert cal.candidate_means == {"amber": 2.0, "bamboo": 6.0}
    assert cal.candidate_stds == {"amber": 1.0, "bamboo": 2.0}
    assert cal.n_null_prompts == 2
    accumulator.update({"bamboo": 1.0, "amber": 2.0})


def test_accumulator_state_round_trip_and_tamper_detection() -> None:
    accumulator = NullScoreAccumulator(("bamboo", "amber"))
    accumulator.update({"amber": 1.0, "bamboo": 4.0})
    accumulator.update({"amber": 3.0, "bamboo": 8.0})

    restored = NullScoreAccumulator.from_state(accumulator.to_state())

    assert restored.to_state() == accumulator.to_state()
    invalid = accumulator.to_state()
    invalid["m2"]["amber"] = -1.0
    with pytest.raises(CalibrationError, match="non-negative"):
        NullScoreAccumulator.from_state(invalid)

    single = NullScoreAccumulator(("amber", "bamboo"))
    single.update({"amber": 1.0, "bamboo": 2.0})
    impossible = single.to_state()
    impossible["m2"]["amber"] = 1.0
    with pytest.raises(CalibrationError, match="single-item.*m2"):
        NullScoreAccumulator.from_state(impossible)


def test_artifact_round_trip_sha_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "jlens_template_tail_layer04.json"
    cal = calibration()

    file_sha256 = cal.save(path)

    assert NullCalibration.load(path, expected_sha256=file_sha256) == cal
    with pytest.raises(CalibrationArtifactError, match="overwrite"):
        cal.save(path)
    with pytest.raises(CalibrationArtifactError, match="file SHA-256 mismatch"):
        NullCalibration.load(path, expected_sha256="0" * 64)


def test_artifact_round_trip_canonicalizes_candidate_order(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    cal = calibration(
        candidate_means={"bamboo": -2.0, "amber": 1.0},
        candidate_stds={"bamboo": 0.0, "amber": 2.0},
    )

    cal.save(path)
    loaded = NullCalibration.load(path)

    assert loaded.candidates == ("amber", "bamboo")
    assert calibrate({"bamboo": 1.0, "amber": 5.0}, loaded) == {
        "amber": 4.0,
        "bamboo": 3.0,
    }


def test_frozen_calibration_does_not_expose_mutable_state() -> None:
    cal = calibration()

    with pytest.raises(TypeError):
        cal.candidate_means["amber"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        cal.provenance["sample_seed"] = 1  # type: ignore[index]
    rendering_policy = cal.provenance["rendering_policy"]
    assert isinstance(rendering_policy, dict) is False
    with pytest.raises(TypeError):
        rendering_policy["clue_last"] = "changed"  # type: ignore[index]

    payload = cal.to_payload()
    payload["candidate_means"]["amber"] = 99.0
    assert cal.candidate_means["amber"] == 1.0
    with pytest.raises(TypeError):
        NULL_RENDERING_POLICY["clue_last"] = "changed"  # type: ignore[index]


def test_failed_save_never_publishes_empty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration.json"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr("jlens_panel.calibration.write_json_atomic", fail_write)
    with pytest.raises(OSError, match="interrupted"):
        calibration().save(path)

    assert not path.exists()


def test_artifact_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    calibration().save(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["calibration"]["candidate_means"]["amber"] = 99.0
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(CalibrationArtifactError, match="payload SHA-256 mismatch"):
        NullCalibration.load(path)


def test_load_calibrations_rejects_duplicate_cell(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    calibration().save(first)
    calibration().save(second)

    with pytest.raises(CalibrationArtifactError, match="duplicate calibration cell"):
        load_calibrations([first, second])


def test_load_calibrations_requires_exact_sha_coverage(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    digest = calibration().save(path)

    loaded = load_calibrations(
        [path],
        expected_sha256={str(path.resolve()): digest},
    )
    assert len(loaded) == 1
    with pytest.raises(CalibrationArtifactError, match="exactly cover"):
        load_calibrations([path], expected_sha256={"typo.json": digest})
    alias = f"{path.parent}/./{path.name}"
    with pytest.raises(CalibrationArtifactError, match="path aliases"):
        load_calibrations(
            [path],
            expected_sha256={str(path): "0" * 64, alias: digest},
        )
    with pytest.raises(CalibrationArtifactError, match="malformed digest"):
        load_calibrations([path], expected_sha256={str(path): "bad"})


def test_load_calibrations_rejects_mixed_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    calibration(position_type="template_tail").save(first)
    calibration(
        method="logit_lens",
        position_type="template_tail",
        provenance=provenance(model_name="different/model"),
    ).save(second)

    with pytest.raises(CalibrationArtifactError, match="incompatible provenance"):
        load_calibrations([first, second])
