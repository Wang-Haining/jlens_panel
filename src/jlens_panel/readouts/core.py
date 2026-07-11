"""Dependency-light types and metrics for candidate-token readouts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable


class ReadoutError(ValueError):
    """Raised when a readout request or score vector violates the contract."""


class LayerMismatchError(ReadoutError):
    """Raised when intermediate-state readouts do not use the same layer."""


class MissingReadoutInputError(ReadoutError):
    """Raised when a scorer's required request input is absent."""


class OptionalDependencyError(ImportError):
    """Raised when fitting a readout requires an unavailable optional extra."""


class ReadoutMethod(str, Enum):
    """Stable method names used in configurations and serialized outputs."""

    JLENS = "jlens"
    LOGIT_LENS = "logit_lens"
    RAW_PROBE = "raw_probe"
    NEXT_TOKEN = "next_token"
    TEXT_ONLY = "text_only"


LAYER_BOUND_METHODS = frozenset(
    {
        ReadoutMethod.JLENS,
        ReadoutMethod.LOGIT_LENS,
        ReadoutMethod.RAW_PROBE,
    }
)
FIVE_READOUT_METHODS = frozenset(ReadoutMethod)


@dataclass(frozen=True, slots=True)
class CandidateToken:
    """One candidate answer represented by a unique vocabulary token."""

    token_id: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, int):
            raise ReadoutError("candidate token_id must be an integer")
        if self.token_id < 0:
            raise ReadoutError("candidate token_id must be non-negative")
        if not isinstance(self.text, str) or not self.text:
            raise ReadoutError("candidate text must be a non-empty string")

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible representation."""

        return {"token_id": self.token_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """An ordered, immutable set defining the shared comparison support."""

    items: tuple[CandidateToken, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if len(self.items) < 2:
            raise ReadoutError("at least two candidate tokens are required")

        token_ids = [candidate.token_id for candidate in self.items]
        texts = [candidate.text for candidate in self.items]
        if len(token_ids) != len(set(token_ids)):
            raise ReadoutError("candidate token_ids must be unique")
        if len(texts) != len(set(texts)):
            raise ReadoutError("candidate texts must be unique")

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, str]]) -> CandidateSet:
        """Build a candidate set from ``(token_id, text)`` pairs."""

        return cls(tuple(CandidateToken(token_id, text) for token_id, text in pairs))

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return candidate token ids in comparison order."""

        return tuple(candidate.token_id for candidate in self.items)

    @property
    def texts(self) -> tuple[str, ...]:
        """Return candidate token strings in comparison order."""

        return tuple(candidate.text for candidate in self.items)

    def index_for_token(self, token_id: int) -> int:
        """Return the comparison index for token_id."""

        try:
            return self.token_ids.index(token_id)
        except ValueError as error:
            raise ReadoutError(f"target token {token_id} is not a candidate") from error


ScoreScalar: TypeAlias = int | float | Any
ScoreInput: TypeAlias = Sequence[ScoreScalar] | Mapping[object, ScoreScalar] | Any


@dataclass(frozen=True, slots=True)
class ReadoutRequest:
    """All optional inputs needed to score one example with the five methods.

    Model-specific code should compute tensors upstream and place them here. The
    readout package only relies on small tensor-like conventions such as
    ``tolist()`` and ``item()``; importing it never imports a tensor framework.
    """

    example_id: str
    candidates: CandidateSet
    layer: int | None = None
    jlens_logits: ScoreInput | None = field(default=None, repr=False)
    logit_lens_logits: ScoreInput | None = field(default=None, repr=False)
    residual: Any | None = field(default=None, repr=False)
    next_token_logits: ScoreInput | None = field(default=None, repr=False)
    text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ReadoutError("example_id must be a non-empty string")
        if self.layer is not None and (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ReadoutError("layer must be a non-negative integer or None")
        if self.text is not None and not isinstance(self.text, str):
            raise ReadoutError("text must be a string or None")
        _validate_json_mapping(self.metadata, name="request metadata")


def _as_float(value: object, *, name: str) -> float:
    """Convert a Python or tensor scalar to a finite float."""

    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ReadoutError(f"{name} must contain numeric scalar scores") from error
    if not math.isfinite(result):
        raise ReadoutError(f"{name} must contain only finite scores")
    return result


def _tolist(value: object) -> object:
    """Detach a tensor-like value and convert it to Python containers."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _as_1d_scores(scores: object, *, name: str) -> tuple[float, ...]:
    """Normalize a one-dimensional score vector without NumPy or Torch imports."""

    values = _tolist(scores)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ReadoutError(f"{name} must be a one-dimensional score sequence")
    if values and isinstance(values[0], (list, tuple)):
        raise ReadoutError(f"{name} must be one-dimensional, not batched")
    return tuple(_as_float(value, name=name) for value in values)


def restrict_candidate_scores(
    scores: ScoreInput,
    candidates: CandidateSet,
    *,
    name: str = "scores",
) -> tuple[float, ...]:
    """Restrict full-vocabulary or keyed scores to a fixed candidate support.

    A mapping may be keyed by integer token id, decimal token-id string, or
    candidate text. A sequence is interpreted as a full vocabulary vector and
    indexed by token id.
    """

    if isinstance(scores, Mapping):
        restricted: list[float] = []
        for candidate in candidates:
            keys = (candidate.token_id, str(candidate.token_id), candidate.text)
            key = next((item for item in keys if item in scores), None)
            if key is None:
                raise ReadoutError(
                    f"{name} has no score for candidate {candidate.text!r} "
                    f"(token {candidate.token_id})"
                )
            restricted.append(_as_float(scores[key], name=name))
        return tuple(restricted)

    full_scores = _as_1d_scores(scores, name=name)
    maximum_id = max(candidates.token_ids)
    if maximum_id >= len(full_scores):
        raise ReadoutError(
            f"{name} length {len(full_scores)} does not include token {maximum_id}"
        )
    return tuple(full_scores[token_id] for token_id in candidates.token_ids)


def aligned_candidate_scores(
    scores: ScoreInput,
    candidates: CandidateSet,
    *,
    name: str = "scores",
) -> tuple[float, ...]:
    """Normalize scores already aligned to candidates, or restrict keyed scores."""

    if isinstance(scores, Mapping):
        return restrict_candidate_scores(scores, candidates, name=name)
    values = _as_1d_scores(scores, name=name)
    if len(values) != len(candidates):
        raise ReadoutError(
            f"{name} returned {len(values)} scores for {len(candidates)} candidates"
        )
    return values


def stable_softmax(logits: Sequence[object]) -> tuple[float, ...]:
    """Compute a numerically stable softmax over finite logits."""

    values = tuple(_as_float(value, name="logits") for value in logits)
    if not values:
        raise ReadoutError("softmax requires at least one logit")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = math.fsum(exponentials)
    if denominator <= 0.0:
        raise ReadoutError("softmax denominator is zero")
    return tuple(value / denominator for value in exponentials)


def multiclass_log_loss(logits: Sequence[object], target_index: int) -> float:
    """Return stable single-example multiclass cross-entropy from logits."""

    values = tuple(_as_float(value, name="logits") for value in logits)
    if not values:
        raise ReadoutError("log loss requires at least one logit")
    if isinstance(target_index, bool) or not 0 <= target_index < len(values):
        raise ReadoutError("target_index is outside the logit vector")
    maximum = max(values)
    log_partition = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in values)
    )
    return log_partition - values[target_index]


def stable_rank(logits: Sequence[object], target_index: int) -> int:
    """Return a deterministic one-based rank, breaking ties by candidate order."""

    values = tuple(_as_float(value, name="logits") for value in logits)
    if isinstance(target_index, bool) or not 0 <= target_index < len(values):
        raise ReadoutError("target_index is outside the logit vector")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return order.index(target_index) + 1


@dataclass(frozen=True, slots=True)
class ReadoutRecord:
    """Candidate-restricted logits from one readout method."""

    example_id: str
    method: ReadoutMethod
    candidates: CandidateSet
    logits: tuple[float, ...]
    layer: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, ReadoutMethod):
            object.__setattr__(self, "method", ReadoutMethod(self.method))
        normalized = aligned_candidate_scores(
            self.logits,
            self.candidates,
            name=f"{self.method.value} logits",
        )
        object.__setattr__(self, "logits", normalized)
        if self.method in LAYER_BOUND_METHODS and self.layer is None:
            raise ReadoutError(f"{self.method.value} record requires a layer")
        if self.layer is not None and (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ReadoutError("record layer must be non-negative or None")
        _validate_json_mapping(self.metadata, name="record metadata")

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Return candidate-normalized probabilities."""

        return stable_softmax(self.logits)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record using only built-in containers."""

        return {
            "example_id": self.example_id,
            "method": self.method.value,
            "layer": self.layer,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReadoutMetrics:
    """Per-example metrics aligned with the names in the project config."""

    example_id: str
    method: ReadoutMethod
    target_token_id: int
    top1_accuracy: float
    mean_reciprocal_rank: float
    log_loss: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-compatible metric row."""

        return {
            "example_id": self.example_id,
            "method": self.method.value,
            "target_token_id": self.target_token_id,
            "top1_accuracy": self.top1_accuracy,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "log_loss": self.log_loss,
        }


def evaluate_record(record: ReadoutRecord, target_token_id: int) -> ReadoutMetrics:
    """Evaluate one candidate-restricted record against its target token."""

    target_index = record.candidates.index_for_token(target_token_id)
    rank = stable_rank(record.logits, target_index)
    return ReadoutMetrics(
        example_id=record.example_id,
        method=record.method,
        target_token_id=target_token_id,
        top1_accuracy=float(rank == 1),
        mean_reciprocal_rank=1.0 / rank,
        log_loss=multiclass_log_loss(record.logits, target_index),
    )


def aggregate_metrics(metrics: Iterable[ReadoutMetrics]) -> dict[str, float]:
    """Macro-average per-example metrics for one homogeneous collection."""

    rows = tuple(metrics)
    if not rows:
        raise ReadoutError("cannot aggregate an empty metric collection")
    count = len(rows)
    return {
        "top1_accuracy": math.fsum(row.top1_accuracy for row in rows) / count,
        "mean_reciprocal_rank": math.fsum(row.mean_reciprocal_rank for row in rows)
        / count,
        "log_loss": math.fsum(row.log_loss for row in rows) / count,
    }


def validate_same_layer(records: Iterable[ReadoutRecord]) -> int:
    """Validate that all present intermediate-state readouts share one layer."""

    bounded = tuple(
        record for record in records if record.method in LAYER_BOUND_METHODS
    )
    if not bounded:
        raise LayerMismatchError("no layer-bound readout records were provided")
    layers = {record.layer for record in bounded}
    if len(layers) != 1:
        details = ", ".join(
            f"{record.method.value}={record.layer}" for record in bounded
        )
        raise LayerMismatchError(f"layer-bound readouts must match: {details}")
    layer = next(iter(layers))
    assert layer is not None
    return layer


@runtime_checkable
class CandidateReadout(Protocol):
    """Common interface implemented by all five scorer adapters."""

    method: ReadoutMethod

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Score one request on its fixed candidate support."""


@dataclass(frozen=True, slots=True)
class ReadoutPanelResult:
    """Validated five-readout records for one example."""

    records: tuple[ReadoutRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if not self.records:
            raise ReadoutError("a panel result requires at least one record")
        example_ids = {record.example_id for record in self.records}
        candidate_sets = {record.candidates for record in self.records}
        methods = [record.method for record in self.records]
        if len(example_ids) != 1:
            raise ReadoutError("panel records must share an example_id")
        if len(candidate_sets) != 1:
            raise ReadoutError("panel records must share candidate order and support")
        if len(methods) != len(set(methods)):
            raise ReadoutError("panel records must have unique methods")
        validate_same_layer(self.records)

    @property
    def example_id(self) -> str:
        """Return the shared example id."""

        return self.records[0].example_id

    def evaluate(self, target_token_id: int) -> tuple[ReadoutMetrics, ...]:
        """Evaluate every method against the same candidate target."""

        return tuple(
            evaluate_record(record, target_token_id) for record in self.records
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible panel record."""

        return {
            "example_id": self.example_id,
            "layer": validate_same_layer(self.records),
            "records": [record.to_dict() for record in self.records],
        }


def _validate_json_mapping(value: Mapping[str, object], *, name: str) -> None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReadoutError(f"{name} must be a mapping with string keys")
    try:
        json.dumps(dict(value), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReadoutError(f"{name} must be JSON-serializable") from error
