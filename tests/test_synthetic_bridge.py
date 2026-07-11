import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from jlens_panel.data.synthetic_bridge import (
    BRIDGE_CLUES,
    DEFAULT_BRIDGE_CANDIDATES,
    SCHEMA_VERSION,
    AnswerContract,
    BridgeDataError,
    SyntheticBridgeExample,
    bridge_clue,
    contains_candidate_word,
    dataset_fingerprint,
    generate_dataset,
    gold_bridge_counts,
    read_jsonl,
    render_agent_b_prompt,
    validate_candidates,
    validate_dataset,
    validate_example,
    write_dataset_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def _small_dataset(seed: int = 17):
    return generate_dataset(
        candidates=DEFAULT_BRIDGE_CANDIDATES,
        seed=seed,
        train_size=33,
        dev_size=35,
        test_size=37,
    )


def test_candidate_contract_is_exact_and_case_insensitive() -> None:
    assert validate_candidates(DEFAULT_BRIDGE_CANDIDATES) == (DEFAULT_BRIDGE_CANDIDATES)

    with pytest.raises(BridgeDataError, match="exactly 16"):
        validate_candidates(DEFAULT_BRIDGE_CANDIDATES[:-1])

    duplicates = list(DEFAULT_BRIDGE_CANDIDATES)
    duplicates[-1] = duplicates[0].upper()
    with pytest.raises(BridgeDataError, match="unique ignoring case"):
        validate_candidates(duplicates)

    invalid = list(DEFAULT_BRIDGE_CANDIDATES)
    invalid[-1] = " padded "
    with pytest.raises(BridgeDataError, match="surrounding whitespace"):
        validate_candidates(invalid)


def test_generation_is_seeded_reproducible_and_seed_sensitive() -> None:
    first = _small_dataset(seed=291)
    second = _small_dataset(seed=291)
    changed = _small_dataset(seed=292)

    assert first == second
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
    assert dataset_fingerprint(first) != dataset_fingerprint(changed)


def test_splits_are_balanced_and_use_disjoint_families_and_entities() -> None:
    dataset = _small_dataset()
    validate_dataset(dataset, candidates=DEFAULT_BRIDGE_CANDIDATES)

    template_sets = {
        split: {example.template_family for example in examples}
        for split, examples in dataset.items()
    }
    family_sets = {
        split: {example.entity_family for example in examples}
        for split, examples in dataset.items()
    }
    entity_sets = {
        split: {
            value
            for example in examples
            for chain in [example.gold_chain, *example.distractor_chains]
            for value in (chain.source_entity, chain.final_answer)
        }
        for split, examples in dataset.items()
    }

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert template_sets[left].isdisjoint(template_sets[right])
        assert family_sets[left].isdisjoint(family_sets[right])
        assert entity_sets[left].isdisjoint(entity_sets[right])

    for examples in dataset.values():
        counts = gold_bridge_counts(examples, DEFAULT_BRIDGE_CANDIDATES)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_example_has_explicit_chain_and_no_cross_agent_answer_leak() -> None:
    example = _small_dataset()["train"][0]
    validate_example(example)

    assert 6 <= example.distractor_count <= 10
    assert len(example.agent_a_facts) == example.distractor_count + 1
    assert len(example.agent_b_facts) == example.distractor_count + 1
    assert example.gold_bridge in example.candidate_bridges
    assert example.gold_chain.bridge_concept == example.gold_bridge
    assert example.gold_chain.final_answer == example.final_answer
    assert len(example.candidate_bridges) == 16
    assert example.schema_version == SCHEMA_VERSION == "synthetic-bridge-v3"
    assert "CANDIDATE BRIDGE CONCEPTS" not in example.agent_a_prompt
    assert "PUBLIC END-TO-END QUESTION" in example.agent_a_prompt
    assert "Agent B has a separate private lookup table" in example.agent_a_prompt
    assert "concise, helpful handoff sentence" in example.agent_a_prompt
    assert "BEHAVIORAL COMPETENCE PROBE" in example.agent_a_probe_prompt
    assert "Reply with only" in example.agent_a_probe_prompt
    assert example.final_answer not in example.agent_a_prompt
    assert example.final_answer not in example.agent_a_probe_prompt
    assert any(example.final_answer in fact for fact in example.agent_b_facts)
    assert example.source_entity not in example.agent_b_prompt_template
    assert bridge_clue(example.gold_bridge, example.split) in example.agent_a_prompt
    assert not any(
        contains_candidate_word(example.agent_a_prompt, candidate)
        or contains_candidate_word(example.agent_a_probe_prompt, candidate)
        for candidate in example.candidate_bridges
    )
    chains = [example.gold_chain, *example.distractor_chains]
    source_units = {chain.source_entity.rsplit("-", 1)[1] for chain in chains}
    answer_units = {chain.final_answer.rsplit("-", 1)[1] for chain in chains}
    assert source_units.isdisjoint(answer_units)

    rendered = render_agent_b_prompt(example, example.gold_bridge)
    assert "{agent_a_message}" not in rendered
    assert example.gold_bridge in rendered


def test_all_frozen_clues_are_split_specific_and_candidate_free() -> None:
    assert set(BRIDGE_CLUES) == set(DEFAULT_BRIDGE_CANDIDATES)
    for candidate, clues in BRIDGE_CLUES.items():
        assert set(clues) == {"train", "dev", "test"}
        assert len(set(clues.values())) == 3
        for clue in clues.values():
            assert not any(
                contains_candidate_word(clue, label)
                for label in DEFAULT_BRIDGE_CANDIDATES
            ), (candidate, clue)


def test_final_answers_include_exact_template_types_in_receiver_relations() -> None:
    dataset = _small_dataset()
    expected_types = {
        "train": {"destination", "station", "chamber"},
        "dev": {"berth", "greenhouse", "gallery"},
        "test": {"terminus", "stage", "bay"},
    }

    for split, examples in dataset.items():
        observed_types = {example.final_answer.split(" ", 1)[0] for example in examples}
        assert observed_types == expected_types[split]
        for example in examples:
            chains = [example.gold_chain, *example.distractor_chains]
            assert example.accepted_answers == (
                example.final_answer,
                example.answer_contract.answer_id,
            )
            assert all(
                any(chain.final_answer in fact for fact in example.agent_b_facts)
                for chain in chains
            )


def test_semantic_validation_rejects_tampered_gold_answer() -> None:
    example = _small_dataset()["dev"][0]
    answer_prefix = example.answer_contract.answer_id.split("-", 1)[0]
    tampered = replace(
        example,
        answer_contract=AnswerContract.build(
            answer_id=f"{answer_prefix}-999999-199",
            answer_type=example.answer_contract.answer_type,
        ),
    )

    with pytest.raises(BridgeDataError, match="agent_b_facts"):
        validate_example(tampered)

    payload = example.to_dict()
    payload["gold_chain"]["bridge_concept"] = "not-the-gold"
    with pytest.raises(BridgeDataError, match="explicit gold fields"):
        SyntheticBridgeExample.from_dict(payload)


def test_jsonl_round_trip_and_bytes_are_reproducible(tmp_path: Path) -> None:
    dataset = _small_dataset(seed=88)
    first_paths = write_dataset_jsonl(tmp_path / "first", dataset)
    second_paths = write_dataset_jsonl(tmp_path / "second", dataset)

    for split in ("train", "dev", "test"):
        assert first_paths[split].read_bytes() == second_paths[split].read_bytes()
        assert read_jsonl(first_paths[split]) == dataset[split]
        lines = first_paths[split].read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(dataset[split])
        assert all(json.loads(line)["split"] == split for line in lines)


def test_cli_writes_all_splits_and_reproducible_summary(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_dirty_data.py"),
        "--output-dir",
        str(output),
        "--seed",
        "101",
        "--train-size",
        "16",
        "--dev-size",
        "16",
        "--test-size",
        "16",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)

    assert summary["dataset_sha256"]
    for split in ("train", "dev", "test"):
        assert summary["splits"][split]["examples"] == 16
        assert (output / f"{split}.jsonl").is_file()
