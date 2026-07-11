"""Causally clean orchestration for the paired clarification experiment.

The sender's initial message is generated exactly once for an item/seed pair and
is represented by a frozen object.  Every clarification branch receives that
same object.  Model implementations live behind protocols so this module has no
dependency on a model library or API client.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .jsonl import JsonlResultStore


class Condition(StrEnum):
    """The four preregistered clarification branches."""

    GENERIC = "generic"
    JLENS_TARGETED = "jlens_targeted"
    BEST_NON_J = "best_non_j"
    ORACLE = "oracle"


CONDITIONS: tuple[Condition, ...] = tuple(Condition)


@dataclass(frozen=True, slots=True)
class ExperimentItem:
    """One distributed-evidence item used by sender and receiver adapters."""

    item_id: str
    context: str
    question: str
    gold_bridge: str
    gold_answer: str

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.gold_bridge.strip():
            raise ValueError("gold_bridge must be non-empty")
        if not self.gold_answer.strip():
            raise ValueError("gold_answer must be non-empty")

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint that detects changed item contents."""

        payload = json.dumps(
            {
                "context": self.context,
                "gold_answer": self.gold_answer,
                "gold_bridge": self.gold_bridge,
                "item_id": self.item_id,
                "question": self.question,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InitialMessage:
    """The immutable common pre-treatment message shared by all branches."""

    item_id: str
    seed: int
    text: str
    gold_bridge: str
    eligible_omitted: bool
    item_fingerprint: str

    @property
    def resume_key(self) -> str:
        """Return the collision-safe JSONL key for the common message."""

        return initial_resume_key(self.item_id, self.seed)

    def to_dict(self) -> dict[str, object]:
        """Serialize this message as a versioned JSONL record."""

        return {
            "schema_version": 1,
            "record_type": "initial_message",
            "resume_key": self.resume_key,
            "item_id": self.item_id,
            "seed": self.seed,
            "text": self.text,
            "gold_bridge": self.gold_bridge,
            "eligible_omitted": self.eligible_omitted,
            "item_fingerprint": self.item_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> InitialMessage:
        """Deserialize and validate one common-message record."""

        message = cls(
            item_id=str(value["item_id"]),
            seed=int(value["seed"]),
            text=str(value["text"]),
            gold_bridge=str(value["gold_bridge"]),
            eligible_omitted=_require_bool(value["eligible_omitted"]),
            item_fingerprint=str(value["item_fingerprint"]),
        )
        if value.get("resume_key") != message.resume_key:
            raise ValueError("initial-message resume key does not match its fields")
        if message.eligible_omitted != gold_bridge_is_absent(
            message.text, message.gold_bridge
        ):
            raise ValueError("stored eligibility disagrees with the common message")
        return message


@dataclass(frozen=True, slots=True)
class BranchPlan:
    """A deterministic condition assignment passed to orchestration adapters."""

    condition: Condition
    order_index: int
    branch_seed: int
    target_concept: str | None


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """One condition outcome suitable for append-only JSONL storage."""

    item_id: str
    seed: int
    condition: Condition
    condition_index: int
    branch_seed: int
    initial_message: str
    eligible_omitted: bool
    target_concept: str | None
    clarification: str
    predicted_answer: str
    gold_answer: str
    exact_match: bool
    item_fingerprint: str

    @property
    def resume_key(self) -> str:
        """Return the collision-safe branch resume key."""

        return outcome_resume_key(self.item_id, self.seed, self.condition)

    def to_dict(self) -> dict[str, object]:
        """Serialize this outcome as a versioned JSONL record."""

        return {
            "schema_version": 1,
            "record_type": "outcome",
            "resume_key": self.resume_key,
            "item_id": self.item_id,
            "seed": self.seed,
            "condition": self.condition.value,
            "condition_index": self.condition_index,
            "branch_seed": self.branch_seed,
            "initial_message": self.initial_message,
            "eligible_omitted": self.eligible_omitted,
            "target_concept": self.target_concept,
            "clarification": self.clarification,
            "predicted_answer": self.predicted_answer,
            "gold_answer": self.gold_answer,
            "exact_match": self.exact_match,
            "item_fingerprint": self.item_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> OutcomeRecord:
        """Deserialize and validate one branch outcome."""

        target = value.get("target_concept")
        record = cls(
            item_id=str(value["item_id"]),
            seed=int(value["seed"]),
            condition=Condition(str(value["condition"])),
            condition_index=int(value["condition_index"]),
            branch_seed=int(value["branch_seed"]),
            initial_message=str(value["initial_message"]),
            eligible_omitted=_require_bool(value["eligible_omitted"]),
            target_concept=None if target is None else str(target),
            clarification=str(value["clarification"]),
            predicted_answer=str(value["predicted_answer"]),
            gold_answer=str(value["gold_answer"]),
            exact_match=_require_bool(value["exact_match"]),
            item_fingerprint=str(value["item_fingerprint"]),
        )
        if value.get("resume_key") != record.resume_key:
            raise ValueError("outcome resume key does not match its fields")
        if record.exact_match != exact_match(
            record.predicted_answer, record.gold_answer
        ):
            raise ValueError("stored exact-match outcome disagrees with answer text")
        expected_order = condition_order(record.item_id, record.seed)
        if not 0 <= record.condition_index < len(expected_order):
            raise ValueError("stored condition index is out of range")
        if expected_order[record.condition_index] is not record.condition:
            raise ValueError(
                "stored condition index disagrees with deterministic order"
            )
        expected_seed = derive_seed(
            record.item_id, record.seed, f"branch:{record.condition.value}"
        )
        if record.branch_seed != expected_seed:
            raise ValueError(
                "stored branch seed disagrees with deterministic assignment"
            )
        return record


class SenderAdapter(Protocol):
    """Adapter contract for initial and clarification message generation."""

    def initial_message(self, item: ExperimentItem, *, seed: int) -> str:
        """Generate the single common pre-treatment message."""

        ...

    def clarification(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        plan: BranchPlan,
    ) -> str:
        """Generate one condition-specific clarification."""

        ...


class TargetSelectorAdapter(Protocol):
    """Adapter contract for J-lens and strongest non-J target selection."""

    def jlens_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        """Choose the concept surfaced by the J-lens condition."""

        ...

    def best_non_j_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        """Choose the concept surfaced by the strongest non-J baseline."""

        ...


class ReceiverAdapter(Protocol):
    """Adapter contract for the condition-blinded receiver answer."""

    def answer(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        clarification: str,
        *,
        seed: int,
    ) -> str:
        """Return an answer without receiving the condition or target label."""

        ...


class ClarificationExperiment:
    """Run all paired branches while enforcing a common initial message."""

    def __init__(
        self,
        *,
        sender: SenderAdapter,
        selector: TargetSelectorAdapter,
        receiver: ReceiverAdapter,
        store: JsonlResultStore | None = None,
    ) -> None:
        self.sender = sender
        self.selector = selector
        self.receiver = receiver
        self.store = store

    def run_item(self, item: ExperimentItem, *, seed: int) -> list[OutcomeRecord]:
        """Run or resume the four branches for one item/seed pair.

        The returned outcomes follow the deterministic assigned order.  If a
        store is supplied, existing branches are not regenerated.
        """

        initial = self._common_initial_message(item, seed)
        order = condition_order(item.item_id, seed)
        outcomes: dict[Condition, OutcomeRecord] = {}

        if self.store is not None:
            for existing in self.store.outcomes_for(item.item_id, seed):
                self._validate_existing_outcome(existing, item, initial)
                outcomes[existing.condition] = existing

        for index, condition in enumerate(order):
            if condition in outcomes:
                continue
            plan = BranchPlan(
                condition=condition,
                order_index=index,
                branch_seed=derive_seed(
                    item.item_id, seed, f"branch:{condition.value}"
                ),
                target_concept=self._target(condition, item, initial),
            )
            clarification = self.sender.clarification(item, initial, plan)
            predicted = self.receiver.answer(
                item, initial, clarification, seed=plan.branch_seed
            )
            outcome = OutcomeRecord(
                item_id=item.item_id,
                seed=seed,
                condition=condition,
                condition_index=index,
                branch_seed=plan.branch_seed,
                initial_message=initial.text,
                eligible_omitted=initial.eligible_omitted,
                target_concept=plan.target_concept,
                clarification=clarification,
                predicted_answer=predicted,
                gold_answer=item.gold_answer,
                exact_match=exact_match(predicted, item.gold_answer),
                item_fingerprint=item.fingerprint,
            )
            if self.store is None:
                outcomes[condition] = outcome
            else:
                self.store.append_outcome(outcome)
                stored = self.store.outcome_for(item.item_id, seed, condition)
                if stored is None:  # pragma: no cover - defensive disk invariant
                    raise RuntimeError("outcome disappeared after JSONL append")
                self._validate_existing_outcome(stored, item, initial)
                outcomes[condition] = stored

        return [outcomes[condition] for condition in order]

    def _common_initial_message(
        self, item: ExperimentItem, seed: int
    ) -> InitialMessage:
        if self.store is not None:
            existing = self.store.initial_message_for(item.item_id, seed)
            if existing is not None:
                self._validate_existing_initial(existing, item)
                return existing

        initial_seed = derive_seed(item.item_id, seed, "initial")
        text = self.sender.initial_message(item, seed=initial_seed)
        candidate = InitialMessage(
            item_id=item.item_id,
            seed=seed,
            text=text,
            gold_bridge=item.gold_bridge,
            eligible_omitted=gold_bridge_is_absent(text, item.gold_bridge),
            item_fingerprint=item.fingerprint,
        )
        if self.store is None:
            return candidate
        stored = self.store.get_or_append_initial(candidate)
        self._validate_existing_initial(stored, item)
        return stored

    def _target(
        self,
        condition: Condition,
        item: ExperimentItem,
        initial: InitialMessage,
    ) -> str | None:
        if condition is Condition.GENERIC:
            return None
        if condition is Condition.JLENS_TARGETED:
            target = self.selector.jlens_target(item, initial)
        elif condition is Condition.BEST_NON_J:
            target = self.selector.best_non_j_target(item, initial)
        else:
            target = item.gold_bridge
        if not target.strip():
            raise ValueError(f"{condition.value} target must be non-empty")
        return target

    @staticmethod
    def _validate_existing_initial(
        initial: InitialMessage, item: ExperimentItem
    ) -> None:
        if initial.item_fingerprint != item.fingerprint:
            raise ValueError(
                f"item {item.item_id!r} changed after its initial message was stored"
            )
        if initial.gold_bridge != item.gold_bridge:
            raise ValueError("stored gold bridge differs from the current item")

    @staticmethod
    def _validate_existing_outcome(
        outcome: OutcomeRecord,
        item: ExperimentItem,
        initial: InitialMessage,
    ) -> None:
        if outcome.item_fingerprint != item.fingerprint:
            raise ValueError(
                f"item {item.item_id!r} changed after an outcome was stored"
            )
        if outcome.initial_message != initial.text:
            raise ValueError("condition branches do not share one initial message")
        if outcome.eligible_omitted != initial.eligible_omitted:
            raise ValueError("condition branches disagree on omission eligibility")


def derive_seed(item_id: str, seed: int, namespace: str) -> int:
    """Derive a stable non-negative 63-bit seed from item, seed, and phase."""

    payload = json.dumps(
        [item_id, int(seed), namespace],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def condition_order(item_id: str, seed: int) -> tuple[Condition, ...]:
    """Return a stable counterbalanced order keyed only by item and seed."""

    return tuple(
        sorted(
            CONDITIONS,
            key=lambda condition: derive_seed(
                item_id, seed, f"condition-order:{condition.value}"
            ),
        )
    )


def initial_resume_key(item_id: str, seed: int) -> str:
    """Return the JSONL resume key for a common initial message."""

    return json.dumps(
        ["initial_message", item_id, int(seed)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def outcome_resume_key(item_id: str, seed: int, condition: Condition | str) -> str:
    """Return the JSONL resume key for one condition outcome."""

    condition_value = Condition(condition).value
    return json.dumps(
        ["outcome", item_id, int(seed), condition_value],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def gold_bridge_is_absent(message: str, gold_bridge: str) -> bool:
    """Return whether the normalized gold bridge is lexically absent.

    This eligibility rule is deliberately pre-treatment and deterministic.  It
    handles case, Unicode, whitespace, and punctuation variants but does not
    pretend to be a semantic-omission annotation.
    """

    bridge = _normalize_lexical(gold_bridge)
    if not bridge:
        raise ValueError("gold_bridge must contain an alphanumeric token")
    normalized_message = _normalize_lexical(message)
    return f" {bridge} " not in f" {normalized_message} "


def exact_match(predicted: str, gold: str) -> bool:
    """Return standard normalized exact match for a receiver answer."""

    return _normalize_answer(predicted) == _normalize_answer(gold)


def _normalize_lexical(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [character if character.isalnum() else " " for character in normalized]
    return " ".join("".join(characters).split())


def _normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        character
        for character in normalized
        if character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"expected a JSON boolean, got {value!r}")
    return value
