import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest

from jlens_panel.readouts.artifacts import RunByteBudget, RunHardLimitError
from jlens_panel.sweep.capture import (
    CAPTURE_METHODS,
    SweepCaptureConflictError,
    SweepCaptureError,
    build_capture_artifact,
    capture_artifact_path,
    load_capture_artifact,
    save_capture_artifact_atomic,
    validate_capture_artifact,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"


def fake_residuals() -> dict[str, FakeTensor]:
    return {
        position: FakeTensor((2, 3), "float16")
        for position in ALL_POSITION_NAMES
    }


def fake_scores() -> dict[str, dict[str, FakeTensor]]:
    return {
        method: {
            position: FakeTensor((2, 2), "float32")
            for position in ALL_POSITION_NAMES
        }
        for method in CAPTURE_METHODS
    }


def artifact() -> dict[str, object]:
    return build_capture_artifact(
        example_id="bridge-train-000001",
        split="train",
        candidates=("bamboo", "amber"),
        candidate_token_ids={"amber": 10, "bamboo": 11},
        gold_bridge="amber",
        layers=(0, 2),
        hidden_size=3,
        residuals=fake_residuals(),
        scores=fake_scores(),
        example_fingerprint="a" * 64,
        dataset_fingerprint="b" * 64,
        capture_fingerprint="c" * 64,
        provenance={"model_revision": "d" * 40},
    )


def pickle_save(value: dict[str, object], path: Path) -> None:
    path.write_bytes(pickle.dumps(value))


def pickle_load(raw_bytes: bytes) -> dict[str, object]:
    value = pickle.loads(raw_bytes)
    assert isinstance(value, dict)
    return value


def test_capture_artifact_is_train_dev_only_and_candidate_restricted() -> None:
    payload = artifact()

    validate_capture_artifact(payload)

    assert payload["split"] == "train"
    assert payload["candidates"] == ["amber", "bamboo"]
    assert payload["positions"] == list(ALL_POSITION_NAMES)
    assert set(payload["scores"]) == {"jlens", "logit_lens"}
    assert "input_ids" not in payload
    assert "generated_token_ids" not in payload
    with pytest.raises(SweepCaptureError, match="train or dev"):
        capture_artifact_path("artifacts", "test", "bridge-test-000001")


def test_capture_artifact_fails_closed_on_tensor_and_support_drift() -> None:
    payload = artifact()
    payload["positions"] = list(reversed(ALL_POSITION_NAMES))
    with pytest.raises(SweepCaptureError, match="position inventory"):
        validate_capture_artifact(payload)

    payload = artifact()
    payload["residuals"]["template_tail"].dtype = "float32"
    with pytest.raises(SweepCaptureError, match="dtype float16"):
        validate_capture_artifact(payload)

    with pytest.raises(SweepCaptureError, match="token IDs do not match"):
        build_capture_artifact(
            example_id="bridge-dev-000001",
            split="dev",
            candidates=("amber", "bamboo"),
            candidate_token_ids={"amber": 10},
            gold_bridge="amber",
            layers=(0, 2),
            hidden_size=3,
            residuals=fake_residuals(),
            scores=fake_scores(),
            example_fingerprint="a" * 64,
            dataset_fingerprint="b" * 64,
            capture_fingerprint="c" * 64,
            provenance={"model_revision": "d" * 40},
        )


def test_capture_save_load_is_atomic_hashed_and_budgeted(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    path = capture_artifact_path(root, "train", "bridge-train-000001")
    budget = RunByteBudget.inspect(root, 1_000_000)

    saved, digest = save_capture_artifact_atomic(
        path,
        artifact(),
        run_root=root,
        byte_budget=budget,
        save_fn=pickle_save,
    )

    assert saved == path
    assert load_capture_artifact(
        path,
        expected_sha256=digest,
        load_fn=pickle_load,
    )["example_id"] == "bridge-train-000001"
    with pytest.raises(SweepCaptureConflictError, match="overwrite"):
        save_capture_artifact_atomic(
            path,
            artifact(),
            run_root=root,
            byte_budget=budget,
            save_fn=pickle_save,
        )
    with pytest.raises(SweepCaptureError, match="SHA-256 mismatch"):
        load_capture_artifact(
            path,
            expected_sha256="0" * 64,
            load_fn=pickle_load,
        )


def test_capture_save_does_not_publish_failed_or_oversize_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    failed = capture_artifact_path(root, "train", "failed")
    budget = RunByteBudget.inspect(root, 1_000_000)

    def fail_save(value: dict[str, object], path: Path) -> None:
        raise OSError("simulated failure")

    with pytest.raises(OSError, match="simulated"):
        save_capture_artifact_atomic(
            failed,
            artifact(),
            run_root=root,
            byte_budget=budget,
            save_fn=fail_save,
        )
    assert not failed.exists()

    oversize = capture_artifact_path(root, "train", "oversize")
    tiny_budget = RunByteBudget.inspect(root, 1)
    with pytest.raises(RunHardLimitError, match="exceed run hard limit"):
        save_capture_artifact_atomic(
            oversize,
            artifact(),
            run_root=root,
            byte_budget=tiny_budget,
            save_fn=pickle_save,
        )
    assert not oversize.exists()
