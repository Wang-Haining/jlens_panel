"""Concrete HuggingFace and frozen-readout adapters for the dirty run.

Importing this module does not import Torch or Transformers.  Generation is
delegated to :func:`jlens_panel.modeling.generate_completion`, which loads its
heavy dependency only when an adapter is actually called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from jlens_panel.io import read_jsonl
from jlens_panel.modeling import ModelBundle, generate_completion
from jlens_panel.readouts import ReadoutMethod

from .core import BranchPlan, Condition, ExperimentItem, InitialMessage

AGENT_A_MESSAGE_PLACEHOLDER = "{agent_a_message}"


class AdapterError(ValueError):
    """Raised when a prompt or frozen readout artifact is invalid."""


class CompletionFunction(Protocol):
    """Dependency-injectable signature of ``generate_completion``."""

    def __call__(
        self,
        bundle: ModelBundle,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.95,
    ) -> str:
        """Generate one assistant continuation."""

        ...


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Fixed decoding budgets shared across paired branches."""

    initial_max_tokens: int = 64
    clarification_max_tokens: int = 24
    receiver_max_tokens: int = 24
    sender_temperature: float = 0.7
    receiver_temperature: float = 0.0
    top_p: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "initial_max_tokens",
            "clarification_max_tokens",
            "receiver_max_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AdapterError(f"{name} must be a positive integer")
        if self.sender_temperature < 0:
            raise AdapterError("sender_temperature must be non-negative")
        if self.receiver_temperature < 0:
            raise AdapterError("receiver_temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise AdapterError("top_p must be in (0, 1]")


@dataclass(slots=True)
class HFSenderAdapter:
    """Generate the common handoff and fixed-budget clarification messages."""

    bundle: ModelBundle
    settings: GenerationSettings
    completion: CompletionFunction = generate_completion

    def initial_message(self, item: ExperimentItem, *, seed: int) -> str:
        """Generate Agent A's common pre-treatment handoff exactly once."""

        return _require_completion(
            self.completion(
                self.bundle,
                [{"role": "user", "content": item.context}],
                seed=seed,
                max_new_tokens=self.settings.initial_max_tokens,
                temperature=self.settings.sender_temperature,
                top_p=self.settings.top_p,
            ),
            phase="initial handoff",
        )

    def clarification(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        plan: BranchPlan,
    ) -> str:
        """Generate a generic or candidate-targeted clarification.

        Every condition receives the same decoding budget.  All targeted
        conditions use one prompt template, so only the selected candidate
        differs between J-lens, best-non-J, and oracle branches.
        """

        instruction = _clarification_instruction(plan)
        messages = [
            {"role": "user", "content": item.context},
            {"role": "assistant", "content": initial_message.text},
            {"role": "user", "content": instruction},
        ]
        return _require_completion(
            self.completion(
                self.bundle,
                messages,
                seed=plan.branch_seed,
                max_new_tokens=self.settings.clarification_max_tokens,
                temperature=self.settings.sender_temperature,
                top_p=self.settings.top_p,
            ),
            phase=f"{plan.condition.value} clarification",
        )


@dataclass(slots=True)
class HFReceiverAdapter:
    """Generate a final answer from a condition-blinded Agent B prompt."""

    bundle: ModelBundle
    settings: GenerationSettings
    completion: CompletionFunction = generate_completion

    def answer(
        self,
        item: ExperimentItem,
        initial_message: InitialMessage,
        clarification: str,
        *,
        seed: int,
    ) -> str:
        """Safely insert the observed handoff and request answer-only output."""

        if item.question.count(AGENT_A_MESSAGE_PLACEHOLDER) != 1:
            raise AdapterError(
                "receiver prompt must contain exactly one literal "
                f"{AGENT_A_MESSAGE_PLACEHOLDER!r} placeholder"
            )
        handoff = (
            f"{initial_message.text.strip()}\n\n"
            "CLARIFICATION FROM AGENT A:\n"
            f"{clarification.strip()}"
        )
        # ``replace`` is intentional: model text is data, never a format string.
        prompt = item.question.replace(AGENT_A_MESSAGE_PLACEHOLDER, handoff, 1)
        prompt += (
            "\n\nReturn only the final answer requested by the QUESTION. "
            "Do not explain your reasoning."
        )
        return _require_completion(
            self.completion(
                self.bundle,
                [{"role": "user", "content": prompt}],
                seed=seed,
                max_new_tokens=self.settings.receiver_max_tokens,
                temperature=self.settings.receiver_temperature,
                top_p=self.settings.top_p,
            ),
            phase="receiver answer",
        )


@dataclass(frozen=True, slots=True)
class FrozenReadoutSelector:
    """Select targets from frozen per-item scores without reading gold labels."""

    predictions: Mapping[str, Mapping[str, str]]
    best_non_j_method: str
    score_bundle_sha256: str

    def __post_init__(self) -> None:
        method = _method_name(self.best_non_j_method)
        if method == ReadoutMethod.JLENS.value:
            raise AdapterError("best_non_j_method cannot be jlens")
        if not _is_sha256(self.score_bundle_sha256):
            raise AdapterError("score_bundle_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "best_non_j_method", method)

    @classmethod
    def from_files(
        cls,
        *,
        score_jsonl: str | Path,
        best_non_j_json: str | Path,
    ) -> FrozenReadoutSelector:
        """Load frozen top predictions and the globally selected dev method."""

        predictions, score_bundle_sha256 = _load_predictions(score_jsonl)
        selection = _load_json_object(best_non_j_json)
        selected_method = _selected_method(selection)
        split = selection.get("selection_split", selection.get("split"))
        if split is None:
            raise AdapterError("best-non-J selection must record its dev split")
        if str(split).casefold() != "dev":
            raise AdapterError("best-non-J method must be selected on the dev split")
        selection_bundle = selection.get("score_bundle_sha256")
        if selection_bundle != score_bundle_sha256:
            raise AdapterError(
                "score JSONL and best-non-J selection come from different bundles"
            )
        return cls(
            predictions=predictions,
            best_non_j_method=selected_method,
            score_bundle_sha256=score_bundle_sha256,
        )

    def jlens_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        """Return the frozen J-lens top prediction for this item."""

        del initial_message
        return self._prediction(item.item_id, ReadoutMethod.JLENS.value)

    def best_non_j_target(
        self, item: ExperimentItem, initial_message: InitialMessage
    ) -> str:
        """Return the frozen dev-selected non-J top prediction for this item."""

        del initial_message
        return self._prediction(item.item_id, self.best_non_j_method)

    def _prediction(self, item_id: str, method: str) -> str:
        try:
            prediction = self.predictions[item_id][method]
        except KeyError as error:
            raise AdapterError(
                f"missing frozen {method!r} prediction for item {item_id!r}"
            ) from error
        if not isinstance(prediction, str) or not prediction.strip():
            raise AdapterError(
                f"invalid frozen {method!r} prediction for item {item_id!r}"
            )
        return prediction.strip()


def _clarification_instruction(plan: BranchPlan) -> str:
    if plan.condition is Condition.GENERIC:
        if plan.target_concept is not None:
            raise AdapterError("generic clarification cannot have a target concept")
        return (
            "Give one concise clarification sentence containing the single most "
            "important fact or relation omitted from your previous handoff. Do "
            "not guess the final answer."
        )

    if plan.target_concept is None or not plan.target_concept.strip():
        raise AdapterError(f"{plan.condition.value} requires a target concept")
    candidate = json.dumps(plan.target_concept.strip(), ensure_ascii=False)
    return (
        f"Candidate concept: {candidate}. If this exact candidate is supported "
        "by your private relations, state one concise sentence giving its "
        "relevant relation for the public question. Otherwise reply exactly: "
        "irrelevant"
    )


def _require_completion(value: str, *, phase: str) -> str:
    text = value.strip()
    if not text:
        raise AdapterError(f"model returned an empty {phase}")
    return text


def _load_predictions(
    path: str | Path,
) -> tuple[dict[str, dict[str, str]], str]:
    predictions: dict[str, dict[str, str]] = {}
    score_bundle_sha256: str | None = None
    for line_number, value in enumerate(read_jsonl(path), start=1):
        record_bundle = value.get("score_bundle_sha256")
        if not _is_sha256(record_bundle):
            raise AdapterError(
                f"score JSONL line {line_number} has no valid score bundle id"
            )
        if score_bundle_sha256 is None:
            score_bundle_sha256 = record_bundle
        elif record_bundle != score_bundle_sha256:
            raise AdapterError("score JSONL mixes records from different bundles")
        item_id = value.get("example_id", value.get("item_id"))
        if not isinstance(item_id, str) or not item_id:
            raise AdapterError(f"score JSONL line {line_number} has no item id")
        extracted = _predictions_from_score_record(value, line_number=line_number)
        if not extracted:
            raise AdapterError(f"score JSONL line {line_number} has no top predictions")
        item_predictions = predictions.setdefault(item_id, {})
        for method, prediction in extracted.items():
            if method in item_predictions and item_predictions[method] != prediction:
                raise AdapterError(
                    f"conflicting {method!r} predictions for item {item_id!r}"
                )
            item_predictions[method] = prediction
    if not predictions:
        raise AdapterError("score JSONL is empty")
    assert score_bundle_sha256 is not None
    return predictions, score_bundle_sha256


def _predictions_from_score_record(
    value: Mapping[str, Any], *, line_number: int
) -> dict[str, str]:
    for key in ("top_predictions", "predictions"):
        mapped = value.get(key)
        if isinstance(mapped, Mapping):
            return {
                _method_name(str(method)): _require_prediction(
                    prediction, line_number=line_number
                )
                for method, prediction in mapped.items()
            }

    score_mapping = value.get("scores")
    if isinstance(score_mapping, Mapping) and "method" not in value:
        result: dict[str, str] = {}
        for method_value, candidate_scores in score_mapping.items():
            method = _method_name(str(method_value))
            if not isinstance(candidate_scores, Mapping) or not candidate_scores:
                raise AdapterError(
                    f"score JSONL line {line_number} has invalid {method!r} scores"
                )
            names = [
                _require_prediction(candidate, line_number=line_number)
                for candidate in candidate_scores
            ]
            scores = [
                _finite_float(score, line_number=line_number)
                for score in candidate_scores.values()
            ]
            result[method] = names[max(range(len(scores)), key=scores.__getitem__)]
        return result

    records = value.get("records")
    if isinstance(records, list):
        result: dict[str, str] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise AdapterError(
                    f"score JSONL line {line_number} has a non-object panel record"
                )
            method, prediction = _prediction_from_method_record(
                record, line_number=line_number
            )
            if method in result:
                raise AdapterError(
                    f"score JSONL line {line_number} repeats method {method!r}"
                )
            result[method] = prediction
        return result

    if "method" in value:
        method, prediction = _prediction_from_method_record(
            value, line_number=line_number
        )
        return {method: prediction}
    return {}


def _prediction_from_method_record(
    value: Mapping[str, Any], *, line_number: int
) -> tuple[str, str]:
    method_value = value.get("method")
    if not isinstance(method_value, str):
        raise AdapterError(f"score JSONL line {line_number} has no method")
    method = _method_name(method_value)
    for key in (
        "top_prediction",
        "prediction",
        "predicted_bridge",
        "top_candidate",
        "top1",
    ):
        if key in value:
            return method, _require_prediction(value[key], line_number=line_number)

    mapped_scores = value.get("scores")
    if isinstance(mapped_scores, Mapping) and mapped_scores:
        names = [
            _require_prediction(candidate, line_number=line_number)
            for candidate in mapped_scores
        ]
        scores = [
            _finite_float(score, line_number=line_number)
            for score in mapped_scores.values()
        ]
        return method, names[max(range(len(scores)), key=scores.__getitem__)]

    candidates = value.get("candidates")
    scores = value.get("logits", value.get("probabilities"))
    if not isinstance(candidates, list) or not isinstance(scores, list):
        raise AdapterError(
            f"score JSONL line {line_number} needs a prediction or candidates/scores"
        )
    if len(candidates) != len(scores) or not candidates:
        raise AdapterError(
            f"score JSONL line {line_number} has misaligned candidates/scores"
        )
    names = [
        _candidate_text(candidate, line_number=line_number) for candidate in candidates
    ]
    numeric_scores = [_finite_float(score, line_number=line_number) for score in scores]
    best_index = max(range(len(numeric_scores)), key=numeric_scores.__getitem__)
    return method, names[best_index]


def _candidate_text(value: object, *, line_number: int) -> str:
    if isinstance(value, Mapping):
        value = value.get("text")
    return _require_prediction(value, line_number=line_number)


def _require_prediction(value: object, *, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(
            f"score JSONL line {line_number} has an invalid top prediction"
        )
    return value.strip()


def _finite_float(value: object, *, line_number: int) -> float:
    import math

    if isinstance(value, bool):
        raise AdapterError(f"score JSONL line {line_number} has a non-numeric score")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AdapterError(
            f"score JSONL line {line_number} has a non-numeric score"
        ) from error
    if not math.isfinite(result):
        raise AdapterError(f"score JSONL line {line_number} has a non-finite score")
    return result


def _method_name(value: str) -> str:
    try:
        return ReadoutMethod(value).value
    except ValueError as error:
        raise AdapterError(f"unknown readout method: {value!r}") from error


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot load best-non-J selection: {path}") from error
    if not isinstance(value, dict):
        raise AdapterError("best-non-J selection JSON must be an object")
    return value


def _selected_method(value: Mapping[str, Any]) -> str:
    for key in (
        "best_non_j_method",
        "selected_method",
        "best_non_j",
        "method",
    ):
        if key in value:
            selected = value[key]
            if isinstance(selected, Mapping):
                selected = selected.get("method", selected.get("selected_method"))
            if not isinstance(selected, str):
                raise AdapterError(f"{key} must be a readout method string")
            return _method_name(selected)
    raise AdapterError("best-non-J selection JSON has no selected method")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
