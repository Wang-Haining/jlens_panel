import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from jlens_panel.calibration import NullScoreAccumulator
from jlens_panel.readouts.artifacts import RunByteBudget, RunHardLimitError
from jlens_panel.sweep.capture import (
    CAPTURE_METHODS,
    SweepCaptureConflictError,
    SweepCaptureError,
    _decode_residuals,
    _forward_activations,
    build_capture_artifact,
    candidate_only_logits,
    capture_artifact_path,
    capture_task_example,
    load_capture_artifact,
    restore_null_accumulators,
    save_capture_artifact_atomic,
    serialize_null_accumulators,
    validate_capture_artifact,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"


def fake_residuals() -> dict[str, FakeTensor]:
    return {position: FakeTensor((2, 3), "float16") for position in ALL_POSITION_NAMES}


def fake_scores() -> dict[str, dict[str, FakeTensor]]:
    return {
        method: {
            position: FakeTensor((2, 2), "float32") for position in ALL_POSITION_NAMES
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

    payload = artifact()
    payload["residuals"]["template_tail"].dtype = "torch.bfloat16"
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
    assert (
        load_capture_artifact(
            path,
            expected_sha256=digest,
            load_fn=pickle_load,
        )["example_id"]
        == "bridge-train-000001"
    )
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


def test_task_capture_rejects_test_before_importing_gpu_runtime() -> None:
    with pytest.raises(SweepCaptureError, match="train/dev"):
        capture_task_example(
            SimpleNamespace(),
            SimpleNamespace(split="test"),
            candidate_token_ids={"amber": 1, "bamboo": 2},
            max_seq_len=512,
        )


class FakeResidual:
    def __init__(self, value: int) -> None:
        self.value = value

    def float(self) -> "FakeResidual":
        return self


class FakeActivation:
    def __init__(self, value: int) -> None:
        self.value = value

    def __getitem__(self, key: object) -> FakeResidual:
        return FakeResidual(self.value)


def test_decode_indices_are_generated_token_states_and_eos_is_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_calls: list[tuple[int, ...]] = []
    generated = iter((10, 11, 12, 13, 14, 15, 16, 99))

    def fake_greedy(
        bundle: object,
        residual: object,
        *,
        forbidden_token_ids: tuple[int, ...] = (),
    ) -> int:
        forbidden_calls.append(tuple(forbidden_token_ids))
        return next(generated)

    def fake_forward(
        bundle: object,
        input_ids: tuple[int, ...] | list[int],
        layers: tuple[int, ...],
    ) -> dict[int, FakeActivation]:
        return {layer: FakeActivation(len(input_ids)) for layer in set(layers)}

    monkeypatch.setattr("jlens_panel.sweep.capture._greedy_token", fake_greedy)
    monkeypatch.setattr("jlens_panel.sweep.capture._forward_activations", fake_forward)
    bundle = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=99))

    captured = _decode_residuals(
        bundle,
        initial_ids=(1, 2),
        initial_activations={0: FakeActivation(2), 1: FakeActivation(2)},
        layers=(0,),
        final_layer=1,
        max_seq_len=32,
    )

    assert captured["decode_1"][0].value == 3
    assert captured["decode_8"][0].value == 10
    assert forbidden_calls == [(99,)] * 7 + [()]


def test_candidate_only_logits_matches_full_linear_unembed() -> None:
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(4, 7, bias=True)
    norm = torch.nn.LayerNorm(4)
    model = SimpleNamespace(
        _lm_head=head,
        _final_norm=norm,
        _logit_softcap=None,
    )
    residual = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    ids = (1, 5, 6)

    actual = candidate_only_logits(model, residual, ids)
    expected = head(norm(residual)).index_select(-1, torch.tensor(ids))

    assert torch.equal(actual, expected)


def test_forward_activations_preserves_exact_input_length() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jlens")

    class TinyModel:
        def __init__(self) -> None:
            self.input_device = torch.device("cpu")
            self.embedding = torch.nn.Embedding(20, 4)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )

        def forward(self, input_ids: object) -> object:
            hidden = self.embedding(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return hidden

    bundle = SimpleNamespace(lens_model=TinyModel())

    activations = _forward_activations(bundle, (1, 2, 3), (0, 1))

    assert tuple(activations[0].shape) == (1, 3, 4)
    assert tuple(activations[1].shape) == (1, 3, 4)


def test_null_accumulator_checkpoint_round_trip_is_cell_complete() -> None:
    candidates = ("amber", "bamboo")
    accumulators = {
        (method, position, 0): NullScoreAccumulator(candidates)
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
    }
    for accumulator in accumulators.values():
        accumulator.update({"amber": 1.0, "bamboo": 2.0})
    state = serialize_null_accumulators(
        accumulators,
        layers=(0,),
        completed_prompts=1,
    )

    completed, restored = restore_null_accumulators(
        state,
        candidates=candidates,
        layers=(0,),
    )

    assert completed == 1
    assert set(restored) == set(accumulators)
    assert all(accumulator.count == 1 for accumulator in restored.values())
    state["cells"] = state["cells"][:-1]
    with pytest.raises(SweepCaptureError, match="incomplete"):
        restore_null_accumulators(state, candidates=candidates, layers=(0,))
