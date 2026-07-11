from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import jlens_panel.readouts.artifacts as artifact_module
from jlens_panel.readouts import ReadoutMethod, ReadoutRecord
from jlens_panel.readouts.artifacts import (
    ExtractionExample,
    artifact_path,
    build_artifact,
    build_extraction_manifest,
    ensure_extraction_manifest,
    load_artifact,
    load_extraction_manifest,
    pickle_load,
    pickle_save,
    save_artifact_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXTRACT = load_script("extract_readouts")
SCORE = load_script("score_readouts")
extract_examples = EXTRACT.extract_examples
score_artifact_splits = SCORE.score_artifact_splits
select_strongest_non_j = SCORE.select_strongest_non_j


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeVocabularyLogits:
    ndim = 1

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> FakeScalar:
        return FakeScalar(self.values[index])


@dataclass
class FakeProbe:
    layer: int
    method = ReadoutMethod.RAW_PROBE
    fit_targets: tuple[int, ...] = ()

    def fit(
        self,
        residuals: object,
        target_token_ids: object,
        *,
        layer: int,
    ) -> FakeProbe:
        assert layer == self.layer
        assert all(len(row) == 2 for row in residuals)
        self.fit_targets = tuple(target_token_ids)
        return self

    def score(self, request: object) -> ReadoutRecord:
        assert request.layer == self.layer
        return ReadoutRecord(
            example_id=request.example_id,
            method=self.method,
            candidates=request.candidates,
            logits=tuple(float(index) for index in range(len(request.candidates))),
            layer=self.layer,
            metadata=request.metadata,
        )


def extraction_example(
    example_id: str,
    split: str,
    gold: str = "Mars",
) -> ExtractionExample:
    return ExtractionExample.from_mapping(
        {
            "example_id": example_id,
            "split": split,
            "candidate_bridges": ["Venus", "Mars", "Jupiter"],
            "gold_bridge": gold,
            "agent_a_prompt": f"Natural handoff prompt for {example_id}",
            "agent_a_probe_prompt": f"Visible classifier evidence for {example_id}",
        }
    )


def test_extraction_captures_once_restricts_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = extraction_example("bridge-train-000001", "train")
    candidates = example.candidates
    token_ids = {"Jupiter": 7, "Mars": 2, "Venus": 5}
    bundle = SimpleNamespace(tokenizer=object())
    calls = {"pre": 0, "text": 0, "disk": 0, "size_scan": 0}
    real_directory_size = artifact_module.directory_size

    def counted_directory_size(path: object) -> int:
        calls["size_scan"] += 1
        return real_directory_size(path)

    monkeypatch.setattr(artifact_module, "directory_size", counted_directory_size)

    def fake_render(tokenizer: object, messages: object) -> str:
        assert tokenizer is bundle.tokenizer
        assert messages == [{"role": "user", "content": example.agent_a_prompt}]
        return "<chat>natural prompt</chat>"

    def fake_pre(bundle_arg: object, prompt: str, **kwargs: int) -> object:
        calls["pre"] += 1
        assert bundle_arg is bundle
        assert prompt.startswith("<chat>")
        assert kwargs == {"layer": 12, "max_seq_len": 256}
        return SimpleNamespace(
            layer=12,
            residual=[0.25, -0.5],
            jlens_logits=FakeVocabularyLogits([0, 0, 2, 0, 0, 5, 0, 7]),
            logit_lens_logits=FakeVocabularyLogits([0, 0, 3, 0, 0, 1, 0, 2]),
            next_token_logits=FakeVocabularyLogits([0, 0, 1, 0, 0, 2, 0, 3]),
        )

    def fake_text(bundle_arg: object, prompt: str, **kwargs: int) -> object:
        calls["text"] += 1
        assert bundle_arg is bundle
        assert prompt.endswith("Answer:")
        assert all(candidate in prompt for candidate in candidates)
        assert kwargs == {"max_seq_len": 256}
        return FakeVocabularyLogits([0, 0, 8, 0, 0, 4, 0, 2])

    def disk_check() -> None:
        calls["disk"] += 1

    arguments = {
        "output_dir": tmp_path,
        "bundle": bundle,
        "layer": 12,
        "candidate_token_ids": token_ids,
        "extraction_fingerprint": "b" * 64,
        "provenance": {"model": "fake"},
        "max_seq_len": 256,
        "hard_limit_bytes": 1_000_000,
        "disk_check": disk_check,
        "capture_pre_speech_fn": fake_pre,
        "capture_next_token_logits_fn": fake_text,
        "render_chat_fn": fake_render,
        "residual_converter": lambda residual: residual,
        "save_fn": pickle_save,
        "load_fn": pickle_load,
    }
    first = extract_examples([example], **arguments)
    second = extract_examples([example], **arguments)

    assert first == {"requested": 1, "written": 1, "skipped": 0}
    assert second == {"requested": 1, "written": 0, "skipped": 1}
    assert calls["pre"] == 1
    assert calls["text"] == 1
    assert calls["disk"] == 4
    assert calls["size_scan"] == 2  # Once per invocation, not once per artifact.
    manifest = load_extraction_manifest(tmp_path)
    assert manifest["expected_counts"] == {"train": 1, "dev": 0, "test": 0}
    artifact = load_artifact(
        artifact_path(tmp_path, "train", example.example_id),
        load_fn=pickle_load,
    )
    assert artifact["candidates"] == list(candidates)
    assert artifact["scores"]["jlens"] == {
        "Jupiter": 7.0,
        "Mars": 2.0,
        "Venus": 5.0,
    }
    assert all(set(scores) == set(candidates) for scores in artifact["scores"].values())


def test_extraction_identity_binds_inputs_and_git_revision() -> None:
    values = {
        "config_sha256": "a" * 64,
        "lens_sha256": "b" * 64,
        "model_name": "fake-model",
        "model_revision": "main",
        "layer": 12,
        "max_seq_len": 256,
        "candidates": ["Jupiter", "Mars", "Venus"],
        "candidate_token_ids": {"Jupiter": 7, "Mars": 2, "Venus": 5},
        "input_sha256": {"train.jsonl": "c" * 64},
        "git_revision_value": "d" * 40,
    }

    original = EXTRACT.extraction_identity(**values)
    changed_input = EXTRACT.extraction_identity(
        **{**values, "input_sha256": {"train.jsonl": "e" * 64}}
    )
    changed_revision = EXTRACT.extraction_identity(
        **{**values, "git_revision_value": "f" * 40}
    )

    assert original != changed_input
    assert original != changed_revision


def test_canonical_jacobian_is_moved_once_to_input_device() -> None:
    class FakeJacobian:
        def __init__(self) -> None:
            self.destination: str | None = None

        def to(self, *, device: str) -> "FakeJacobian":
            self.destination = device
            return self

    jacobian = FakeJacobian()
    bundle = SimpleNamespace(
        lens=SimpleNamespace(jacobians={12: jacobian}),
        lens_model=SimpleNamespace(input_device="cuda:0"),
    )

    EXTRACT.move_canonical_jacobian_to_input_device(bundle, 12)

    assert bundle.lens.jacobians[12] is jacobian
    assert jacobian.destination == "cuda:0"


def scoring_artifact(example_id: str, split: str, gold: str) -> dict[str, object]:
    example = extraction_example(example_id, split, gold)
    scores = {
        "jlens": {"Jupiter": 0.0, "Mars": 4.0, "Venus": 1.0},
        "logit_lens": {"Jupiter": 0.0, "Mars": 3.0, "Venus": 1.0},
        "next_token": {"Jupiter": 1.0, "Mars": 0.0, "Venus": 2.0},
        "text_only": {"Jupiter": 0.0, "Mars": 2.0, "Venus": 1.0},
    }
    return build_artifact(
        example=example,
        candidate_token_ids={"Jupiter": 7, "Mars": 2, "Venus": 5},
        layer=12,
        residual=[0.1, 0.2],
        scores=scores,
        extraction_fingerprint="c" * 64,
        provenance={"model": "fake"},
    )


def test_scoring_fits_probe_on_train_and_selects_on_dev_only() -> None:
    artifacts = {
        "train": [
            scoring_artifact("train-1", "train", "Mars"),
            scoring_artifact("train-2", "train", "Venus"),
        ],
        "dev": [scoring_artifact("dev-1", "dev", "Mars")],
        "test": [scoring_artifact("test-1", "test", "Jupiter")],
    }
    probes: list[FakeProbe] = []

    def factory(*, layer: int) -> FakeProbe:
        probe = FakeProbe(layer)
        probes.append(probe)
        return probe

    records, summary, selection = score_artifact_splits(
        artifacts,
        probe_factory=factory,
    )

    assert probes[0].fit_targets == (2, 5)
    assert len(records) == 4 * 5
    assert selection["selection_split"] == "dev"
    assert selection["test_metrics_used_for_selection"] is False
    assert selection["selected_method"] in {
        "logit_lens",
        "raw_probe",
        "next_token",
        "text_only",
    }
    assert summary["selected_non_j_method"] == selection["selected_method"]
    assert summary["splits"]["test"]["examples"] == 1


def test_non_j_selection_uses_frozen_tie_order() -> None:
    tied = {
        method: {
            "log_loss": 1.0,
            "top1_accuracy": 0.0,
            "mean_reciprocal_rank": 0.5,
        }
        for method in ("text_only", "next_token", "raw_probe", "logit_lens")
    }

    selection = select_strongest_non_j(tied)

    assert selection["selected_method"] == "logit_lens"
    assert selection["tie_order"] == [
        "logit_lens",
        "raw_probe",
        "next_token",
        "text_only",
    ]


def test_split_loader_validates_top_level_manifest(tmp_path: Path) -> None:
    examples = [
        extraction_example("train-1", "train", "Mars"),
        extraction_example("dev-1", "dev", "Mars"),
        extraction_example("test-1", "test", "Jupiter"),
    ]
    token_ids = {"Jupiter": 7, "Mars": 2, "Venus": 5}
    manifest = build_extraction_manifest(
        examples=examples,
        candidate_token_ids=token_ids,
        layer=12,
        extraction_fingerprint="c" * 64,
        provenance={"model": "fake"},
    )
    ensure_extraction_manifest(tmp_path, manifest)
    for example in examples:
        save_artifact_atomic(
            artifact_path(tmp_path, example.split, example.example_id),
            scoring_artifact(
                example.example_id,
                example.split,
                example.gold_bridge,
            ),
            run_root=tmp_path,
            hard_limit_bytes=1_000_000,
            save_fn=pickle_save,
        )

    loaded = SCORE.load_artifact_splits(tmp_path, load_fn=pickle_load)

    assert {split: len(values) for split, values in loaded.items()} == {
        "train": 1,
        "dev": 1,
        "test": 1,
    }


def test_score_cli_writes_provenance_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "scores"
    artifact_dir.mkdir()
    (artifact_dir / "extraction_manifest.json").write_text(
        '{"fixture":true}\n', encoding="utf-8"
    )
    fake_artifacts = {split: [{}] for split in ("train", "dev", "test")}
    fake_selection = {
        "selected_method": "logit_lens",
        "selection_split": "dev",
        "test_metrics_used_for_selection": False,
    }
    monkeypatch.setattr(
        SCORE,
        "load_artifact_splits",
        lambda *args, **kwargs: fake_artifacts,
    )
    monkeypatch.setattr(
        SCORE,
        "score_artifact_splits",
        lambda artifacts: (
            [{"schema_version": "fixture", "method": "jlens"}],
            {"schema_version": "fixture"},
            fake_selection,
        ),
    )

    result = SCORE.main(
        [
            "--config",
            str(ROOT / "config" / "dirty_run.yaml"),
            "--project-root",
            str(ROOT),
            "--artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads(
        (output_dir / "score_manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (output_dir / "non_j_selection.json").read_text(encoding="utf-8")
    )
    score_record = json.loads(
        (output_dir / "readout_scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert result == 0
    assert manifest["config_sha256"]
    assert "git_revision" in manifest
    assert manifest["extra"]["artifact_tree_sha256"]
    assert manifest["extra"]["scored_counts"] == {
        "train": 1,
        "dev": 1,
        "test": 1,
    }
    assert manifest["extra"]["record_count"] == 1
    assert (
        score_record["score_bundle_sha256"]
        == selection["score_bundle_sha256"]
        == manifest["extra"]["score_bundle_sha256"]
    )
