from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jlens_panel.experiment import (
    CONDITIONS,
    BranchPlan,
    ClarificationExperiment,
    Condition,
    ExperimentItem,
    InitialMessage,
    JsonlResultStore,
    condition_order,
    exact_match,
    gold_bridge_is_absent,
)


class FakeSender:
    def __init__(self, initial_text: str = "I found a useful relation.") -> None:
        self.initial_text = initial_text
        self.initial_calls = 0
        self.initial_objects: list[InitialMessage] = []
        self.plans: list[BranchPlan] = []

    def initial_message(self, item: ExperimentItem, *, seed: int) -> str:
        self.initial_calls += 1
        assert seed >= 0
        return self.initial_text

    def clarification(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        plan: BranchPlan,
    ) -> str:
        self.initial_objects.append(initial_message)
        self.plans.append(plan)
        return plan.target_concept or "Please reconsider the evidence."


class FakeSelector:
    def jlens_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        return item.gold_bridge

    def best_non_j_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        return "irrelevant"


class FakeReceiver:
    def answer(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        clarification: str,
        *,
        seed: int,
    ) -> str:
        assert seed >= 0
        if clarification == item.gold_bridge:
            return f"The {item.gold_answer}."
        return "incorrect"


def _item() -> ExperimentItem:
    return ExperimentItem(
        item_id="item-1",
        context="A private supporting fact.",
        question="What is the answer?",
        gold_bridge="Mars",
        gold_answer="red planet",
    )


def test_runner_shares_one_immutable_initial_message_across_branches() -> None:
    sender = FakeSender()
    runner = ClarificationExperiment(
        sender=sender,
        selector=FakeSelector(),
        receiver=FakeReceiver(),
    )

    outcomes = runner.run_item(_item(), seed=17)

    assert sender.initial_calls == 1
    assert len(outcomes) == len(CONDITIONS)
    assert {outcome.condition for outcome in outcomes} == set(CONDITIONS)
    assert len({id(message) for message in sender.initial_objects}) == 1
    assert all(outcome.eligible_omitted for outcome in outcomes)
    assert [outcome.condition for outcome in outcomes] == list(
        condition_order("item-1", 17)
    )
    assert len({plan.branch_seed for plan in sender.plans}) == len(CONDITIONS)
    with pytest.raises(FrozenInstanceError):
        sender.initial_objects[0].text = "mutated"  # type: ignore[misc]


def test_eligibility_uses_common_pre_treatment_message() -> None:
    assert gold_bridge_is_absent("I only know its color.", "Mars")
    assert not gold_bridge_is_absent("It concerns MARS.", "Mars")
    assert not gold_bridge_is_absent("New-York is relevant", "New York")
    assert gold_bridge_is_absent("Marshall is relevant", "Mars")


def test_normalized_exact_match() -> None:
    assert exact_match("The Red Planet.", "red planet")
    assert exact_match(
        "zenith-000000-151",
        "bay zenith-000000-151",
        aliases=("zenith-000000-151",),
    )
    assert exact_match(
        "Bay Zenith-000000-151.",
        "bay zenith-000000-151",
        aliases=("zenith-000000-151",),
    )
    assert not exact_match("blue planet", "red planet")
    assert not exact_match(
        "zenith-000000",
        "bay zenith-000000-151",
        aliases=("zenith-000000-151",),
    )
    assert not exact_match(
        "stage zenith-000000-151",
        "bay zenith-000000-151",
        aliases=("zenith-000000-151",),
    )


def test_jsonl_append_and_resume_reuses_common_message(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    first_sender = FakeSender()
    store = JsonlResultStore(path)
    first_runner = ClarificationExperiment(
        sender=first_sender,
        selector=FakeSelector(),
        receiver=FakeReceiver(),
        store=store,
    )
    first = first_runner.run_item(_item(), seed=29)

    second_sender = FakeSender(initial_text="a different message")
    second_runner = ClarificationExperiment(
        sender=second_sender,
        selector=FakeSelector(),
        receiver=FakeReceiver(),
        store=JsonlResultStore(path),
    )
    second = second_runner.run_item(_item(), seed=29)

    assert first == second
    assert first_sender.initial_calls == 1
    assert second_sender.initial_calls == 0
    assert second_sender.plans == []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 + len(CONDITIONS)
    assert len({record["resume_key"] for record in records}) == len(records)
    assert records[0]["record_type"] == "initial_message"


def test_resume_only_runs_missing_conditions(tmp_path: Path) -> None:
    complete_path = tmp_path / "complete.jsonl"
    complete_store = JsonlResultStore(complete_path)
    runner = ClarificationExperiment(
        sender=FakeSender(),
        selector=FakeSelector(),
        receiver=FakeReceiver(),
        store=complete_store,
    )
    runner.run_item(_item(), seed=43)

    lines = complete_path.read_text().splitlines()
    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
    resumed_sender = FakeSender()
    resumed = ClarificationExperiment(
        sender=resumed_sender,
        selector=FakeSelector(),
        receiver=FakeReceiver(),
        store=JsonlResultStore(partial_path),
    ).run_item(_item(), seed=43)

    assert resumed_sender.initial_calls == 0
    assert len(resumed_sender.plans) == 2
    assert len(resumed) == len(CONDITIONS)
    assert len(partial_path.read_text().splitlines()) == 1 + len(CONDITIONS)


def test_oracle_and_jlens_targets_are_scored_by_exact_match() -> None:
    outcomes = ClarificationExperiment(
        sender=FakeSender(),
        selector=FakeSelector(),
        receiver=FakeReceiver(),
    ).run_item(_item(), seed=17)
    by_condition = {outcome.condition: outcome for outcome in outcomes}

    assert by_condition[Condition.ORACLE].exact_match
    assert by_condition[Condition.JLENS_TARGETED].exact_match
    assert not by_condition[Condition.BEST_NON_J].exact_match
    assert not by_condition[Condition.GENERIC].exact_match
