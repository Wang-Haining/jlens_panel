from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import pytest

from jlens_panel.data import DEFAULT_BRIDGE_CANDIDATES, generate_dataset, write_jsonl
from jlens_panel.experiment import (
    BranchPlan,
    Condition,
    ExperimentItem,
    InitialMessage,
    JsonlResultStore,
)
from jlens_panel.experiment.adapters import (
    AdapterError,
    FrozenReadoutSelector,
    GenerationSettings,
    HFReceiverAdapter,
    HFSenderAdapter,
)
from jlens_panel.storage import DiskStatus

ROOT = Path(__file__).resolve().parents[1]
SCORE_BUNDLE = "a" * 64


class RecordingCompletion:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        bundle: object,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.95,
    ) -> str:
        self.calls.append(
            {
                "bundle": bundle,
                "messages": list(messages),
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return self.responses.pop(0)


def _item() -> ExperimentItem:
    return ExperimentItem(
        item_id="example-1",
        context="Agent A private relations and public question.",
        question=(
            "AGENT B RELATIONS:\n- Mars unlocks red.\n\n"
            "AGENT A MESSAGE:\n{agent_a_message}\n\n"
            "QUESTION: Which destination?"
        ),
        gold_bridge="Mars",
        gold_answer="red",
    )


def _initial() -> InitialMessage:
    item = _item()
    return InitialMessage(
        item_id=item.item_id,
        seed=17,
        text="I found a useful relation.",
        gold_bridge=item.gold_bridge,
        eligible_omitted=True,
        item_fingerprint=item.fingerprint,
    )


def test_hf_adapters_share_fixed_clarification_budget_and_blind_receiver() -> None:
    settings = GenerationSettings(
        initial_max_tokens=40,
        clarification_max_tokens=19,
        receiver_max_tokens=7,
    )
    sender_completion = RecordingCompletion(
        ["common message", "generic fact", "Mars maps to the lookup"]
    )
    sender = HFSenderAdapter(
        bundle=object(),  # type: ignore[arg-type]
        settings=settings,
        completion=sender_completion,
    )
    item = _item()
    assert sender.initial_message(item, seed=3) == "common message"

    generic = BranchPlan(
        condition=Condition.GENERIC,
        order_index=0,
        branch_seed=11,
        target_concept=None,
    )
    targeted = BranchPlan(
        condition=Condition.JLENS_TARGETED,
        order_index=1,
        branch_seed=12,
        target_concept="Mars",
    )
    sender.clarification(item, _initial(), generic)
    sender.clarification(item, _initial(), targeted)

    assert sender_completion.calls[0]["max_new_tokens"] == 40
    assert sender_completion.calls[1]["max_new_tokens"] == 19
    assert sender_completion.calls[2]["max_new_tokens"] == 19
    assert sender_completion.calls[1]["temperature"] == 0.7
    assert sender_completion.calls[2]["temperature"] == 0.7
    targeted_prompt = sender_completion.calls[2]["messages"][-1]["content"]  # type: ignore[index]
    assert 'Candidate concept: "Mars"' in targeted_prompt
    assert "reply exactly: irrelevant" in targeted_prompt

    receiver_completion = RecordingCompletion(["red"])
    receiver = HFReceiverAdapter(
        bundle=object(),  # type: ignore[arg-type]
        settings=settings,
        completion=receiver_completion,
    )
    assert receiver.answer(item, _initial(), "Mars maps onward.", seed=99) == "red"
    receiver_prompt = receiver_completion.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert "I found a useful relation." in receiver_prompt
    assert "Mars maps onward." in receiver_prompt
    assert "jlens_targeted" not in receiver_prompt
    assert "{agent_a_message}" not in receiver_prompt
    assert receiver_completion.calls[0]["max_new_tokens"] == 7
    assert receiver_completion.calls[0]["temperature"] == 0.0


def test_frozen_selector_uses_jlens_and_dev_selected_non_j_without_gold(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(
            {
                "example_id": "example-1",
                "score_bundle_sha256": SCORE_BUNDLE,
                "gold_bridge": "Venus",
                "records": [
                    {
                        "method": "jlens",
                        "candidates": [{"text": "Mars"}, {"text": "Venus"}],
                        "logits": [4.0, 1.0],
                    },
                    {
                        "method": "logit_lens",
                        "candidates": [{"text": "Mars"}, {"text": "Venus"}],
                        "logits": [0.0, 3.0],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "best.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "best_non_j_method": "logit_lens",
                "score_bundle_sha256": SCORE_BUNDLE,
            }
        ),
        encoding="utf-8",
    )
    selector = FrozenReadoutSelector.from_files(
        score_jsonl=scores,
        best_non_j_json=selection,
    )

    assert selector.jlens_target(_item(), _initial()) == "Mars"
    assert selector.best_non_j_target(_item(), _initial()) == "Venus"

    selection.write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "best_non_j_method": "logit_lens",
                "score_bundle_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdapterError, match="different bundles"):
        FrozenReadoutSelector.from_files(
            score_jsonl=scores,
            best_non_j_json=selection,
        )

    selection.write_text(
        json.dumps({"selection_split": "test", "best_non_j_method": "jlens"}),
        encoding="utf-8",
    )
    with pytest.raises(AdapterError):
        FrozenReadoutSelector.from_files(
            score_jsonl=scores,
            best_non_j_json=selection,
        )


class DirtyRunCompletion:
    def __init__(self, *, gold_bridge: str, gold_answer: str) -> None:
        self.gold_bridge = gold_bridge
        self.gold_answer = gold_answer
        self.calls = 0

    def __call__(
        self,
        bundle: object,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.95,
    ) -> str:
        del bundle, seed, max_new_tokens, temperature, top_p
        self.calls += 1
        prompt = messages[-1]["content"]
        if "Return only the final answer" in prompt:
            clarification = prompt.split("CLARIFICATION FROM AGENT A:\n", 1)[1]
            if clarification.startswith(f"Use bridge {self.gold_bridge}."):
                return self.gold_answer
            return "wrong"
        if len(messages) == 3:
            if f'Candidate concept: "{self.gold_bridge}"' in prompt:
                return f"Use bridge {self.gold_bridge}."
            return "irrelevant"
        return "I found a bridge relation but did not name it."


def test_dirty_run_and_analysis_clis_are_gpu_free_with_injection(
    tmp_path: Path,
) -> None:
    run_module = _load_script("run_dirty_experiment.py", "dirty_run_script")
    analyze_module = _load_script("analyze_dirty_run.py", "dirty_analysis_script")
    dataset = generate_dataset(
        candidates=DEFAULT_BRIDGE_CANDIDATES,
        seed=55,
        train_size=16,
        dev_size=16,
        test_size=16,
    )
    example = dataset["dev"][0]
    data_path = write_jsonl(tmp_path / "dev.jsonl", [example])
    distractor = next(
        candidate
        for candidate in example.candidate_bridges
        if candidate != example.gold_bridge
    )
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(
            {
                "example_id": example.example_id,
                "score_bundle_sha256": SCORE_BUNDLE,
                "predictions": {
                    "jlens": example.gold_bridge,
                    "logit_lens": distractor,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "dev",
                "best_non_j_method": "logit_lens",
                "score_bundle_sha256": SCORE_BUNDLE,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run" / "outcomes.jsonl"
    run_args = run_module.build_parser().parse_args(
        [
            "--config",
            str(ROOT / "config" / "dirty_run.yaml"),
            "--data",
            str(data_path),
            "--scores",
            str(scores),
            "--best-non-j",
            str(selection),
            "--output",
            str(output),
            "--project-root",
            str(ROOT),
            "--limit",
            "1",
            "--seed",
            "17",
        ]
    )
    completion = DirtyRunCompletion(
        gold_bridge=example.gold_bridge,
        gold_answer=example.final_answer,
    )
    bundle_loads = []

    def fake_loader(**kwargs: object) -> object:
        bundle_loads.append(kwargs)
        return object()

    first = run_module.run(
        run_args,
        bundle_loader=fake_loader,
        completion=completion,
        disk_inspector=_safe_disk_status,
    )
    assert first["completed_outcomes"] == 4
    assert first["expected_outcomes"] == 4
    assert len(bundle_loads) == 1
    assert completion.calls == 9
    assert output.with_suffix(".manifest.json").is_file()
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    generation = manifest["extra"]["run_spec"]["generation"]
    assert generation == {
        "clarification_max_tokens": 24,
        "initial_max_tokens": 32,
        "receiver_max_tokens": 16,
        "receiver_temperature": 0.0,
        "sender_temperature": 0.7,
        "top_p": 0.95,
    }
    outcomes = JsonlResultStore(output).all_outcomes()
    assert len(outcomes) == 4
    assert all(outcome.eligible_omitted for outcome in outcomes)
    assert sum(outcome.exact_match for outcome in outcomes) == 2

    resumed_completion = DirtyRunCompletion(
        gold_bridge=example.gold_bridge,
        gold_answer=example.final_answer,
    )

    def fail_loader(**kwargs: object) -> object:
        raise AssertionError(f"complete resume should not load a model: {kwargs}")

    second = run_module.run(
        run_args,
        bundle_loader=fail_loader,
        completion=resumed_completion,
        disk_inspector=_safe_disk_status,
    )
    assert second == first
    assert resumed_completion.calls == 0

    analysis_output = tmp_path / "analysis.json"
    analysis_args = analyze_module.build_parser().parse_args(
        [
            "--config",
            str(ROOT / "config" / "dirty_run.yaml"),
            "--input",
            str(output),
            "--output",
            str(analysis_output),
            "--bootstrap-resamples",
            "100",
        ]
    )
    payload = analyze_module.analyze(analysis_args)
    gate = payload["interpretation"]["oracle_lift_gate"]
    assert gate["status"] == "not_evaluable"
    assert gate["primary_subset"] == "omitted"
    assert gate["enough_omitted_items"] is False
    assert gate["meets_point_estimate_threshold"] is True
    assert analysis_output.is_file()


def _load_script(filename: str, module_name: str) -> ModuleType:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_disk_status(path: str | Path) -> DiskStatus:
    del path
    tebibyte = 1024**4
    return DiskStatus(
        total_bytes=10 * tebibyte,
        used_bytes=tebibyte,
        free_bytes=9 * tebibyte,
        project_bytes=1024,
    )
