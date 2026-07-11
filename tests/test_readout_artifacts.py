from pathlib import Path

import pytest

from jlens_panel.readouts.artifacts import (
    ArtifactError,
    ExtractionExample,
    RunHardLimitError,
    artifact_path,
    build_artifact,
    build_text_only_prompt,
    load_artifact,
    pickle_load,
    pickle_save,
    save_artifact_atomic,
    validate_shared_inventory,
)

CANDIDATES = ("Jupiter", "Mars", "Venus")
TOKEN_IDS = {"Jupiter": 7, "Mars": 2, "Venus": 5}


def raw_example(example_id: str = "bridge-train-000001") -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": "train",
        "candidate_bridges": ["Venus", "Jupiter", "Mars"],
        "gold_bridge": "Mars",
        "final_answer": "dock 7",
        "agent_a_prompt": "Privately determine the bridge and brief Agent B.",
        "agent_a_probe_prompt": "Which bridge follows from the private relations?",
        "agent_b_prompt_template": "Message: {agent_a_message}",
        "facts_metadata": {"ignored": True},
    }


def artifact_payload(example: ExtractionExample) -> dict[str, object]:
    scores = {
        method: {candidate: float(index) for index, candidate in enumerate(CANDIDATES)}
        for method in ("jlens", "logit_lens", "next_token", "text_only")
    }
    return build_artifact(
        example=example,
        candidate_token_ids=TOKEN_IDS,
        layer=12,
        residual=[0.25, -0.5],
        scores=scores,
        extraction_fingerprint="a" * 64,
        provenance={"model": "fake", "seed": 17},
    )


def test_stable_synthetic_schema_is_normalized() -> None:
    example = ExtractionExample.from_mapping(raw_example())

    assert example.example_id == "bridge-train-000001"
    assert example.candidates == CANDIDATES
    assert example.gold_bridge == "Mars"
    assert len(example.fingerprint) == 64


def test_inventory_mismatch_and_unsafe_paths_fail_closed(tmp_path: Path) -> None:
    first = ExtractionExample.from_mapping(raw_example("first"))
    changed = raw_example("second")
    changed["candidate_bridges"] = ["Mars", "Venus", "Saturn"]
    second = ExtractionExample.from_mapping(changed)

    with pytest.raises(ArtifactError, match="inventory mismatch"):
        validate_shared_inventory([first, second])
    with pytest.raises(ArtifactError, match="unsafe example_id"):
        artifact_path(tmp_path, "train", "../escape")


def test_text_only_prompt_is_plain_candidate_prompt() -> None:
    prompt = build_text_only_prompt("Visible evidence only.", CANDIDATES)

    assert all(candidate in prompt for candidate in CANDIDATES)
    assert prompt.startswith("Visible evidence only.")
    assert prompt.endswith("Answer:")


def test_pickle_fixture_round_trip_is_atomic_and_validated(tmp_path: Path) -> None:
    example = ExtractionExample.from_mapping(raw_example())
    payload = artifact_payload(example)
    path = artifact_path(tmp_path, example.split, example.example_id)

    save_artifact_atomic(
        path,
        payload,
        run_root=tmp_path,
        hard_limit_bytes=1_000_000,
        save_fn=pickle_save,
    )
    loaded = load_artifact(path, load_fn=pickle_load)

    assert loaded["example_id"] == example.example_id
    assert loaded["scores"] == payload["scores"]
    assert not any(path.parent.glob(f".{path.name}.*"))


def test_atomic_save_refuses_run_hard_limit_overshoot(tmp_path: Path) -> None:
    example = ExtractionExample.from_mapping(raw_example())
    payload = artifact_payload(example)
    path = artifact_path(tmp_path, example.split, example.example_id)

    with pytest.raises(RunHardLimitError, match="exceed run hard limit"):
        save_artifact_atomic(
            path,
            payload,
            run_root=tmp_path,
            hard_limit_bytes=1,
            save_fn=pickle_save,
        )

    assert not path.exists()
