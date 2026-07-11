"""Five candidate-token scorer adapters with lazy optional dependencies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias

from .core import (
    FIVE_READOUT_METHODS,
    CandidateSet,
    MissingReadoutInputError,
    OptionalDependencyError,
    ReadoutError,
    ReadoutMethod,
    ReadoutPanelResult,
    ReadoutRecord,
    ReadoutRequest,
    ScoreInput,
    _as_1d_scores,
    _tolist,
    aligned_candidate_scores,
    restrict_candidate_scores,
)


class JLensReadout:
    """Restrict precomputed Jacobian-lens vocabulary logits to candidates."""

    method = ReadoutMethod.JLENS

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Score candidate tokens from request.jlens_logits."""

        if request.jlens_logits is None:
            raise MissingReadoutInputError("jlens_logits are required")
        if request.layer is None:
            raise ReadoutError("J-lens scoring requires request.layer")
        return _vocabulary_record(
            request,
            method=self.method,
            scores=request.jlens_logits,
            layer=request.layer,
        )


class LogitLensReadout:
    """Restrict precomputed same-layer logit-lens logits to candidates."""

    method = ReadoutMethod.LOGIT_LENS

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Score candidate tokens from request.logit_lens_logits."""

        if request.logit_lens_logits is None:
            raise MissingReadoutInputError("logit_lens_logits are required")
        if request.layer is None:
            raise ReadoutError("logit-lens scoring requires request.layer")
        return _vocabulary_record(
            request,
            method=self.method,
            scores=request.logit_lens_logits,
            layer=request.layer,
        )


class NextTokenReadout:
    """Restrict final next-token vocabulary logits to candidates."""

    method = ReadoutMethod.NEXT_TOKEN

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Score candidate tokens from request.next_token_logits."""

        if request.next_token_logits is None:
            raise MissingReadoutInputError("next_token_logits are required")
        return _vocabulary_record(
            request,
            method=self.method,
            scores=request.next_token_logits,
            layer=None,
        )


TextPrediction: TypeAlias = ScoreInput
TextPredictor: TypeAlias = Callable[[str, CandidateSet], TextPrediction]


class TextOnlyReadout:
    """Adapt a text-only callable to the common candidate readout interface.

    The callable receives ``(text, candidates)`` and may return either a score
    sequence already aligned to candidate order or a mapping keyed by token id,
    decimal token-id string, or candidate text.
    """

    method = ReadoutMethod.TEXT_ONLY

    def __init__(self, predictor: TextPredictor) -> None:
        if not callable(predictor):
            raise ReadoutError("text-only predictor must be callable")
        self.predictor = predictor

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Call the predictor and normalize its candidate-aligned scores."""

        if request.text is None:
            raise MissingReadoutInputError("text is required by text-only readout")
        predicted = self.predictor(request.text, request.candidates)
        logits = aligned_candidate_scores(
            predicted,
            request.candidates,
            name="text-only predictor scores",
        )
        return ReadoutRecord(
            example_id=request.example_id,
            method=self.method,
            candidates=request.candidates,
            logits=logits,
            layer=None,
            metadata=request.metadata,
        )


class _DecisionEstimator(Protocol):
    classes_: object

    def decision_function(self, values: object) -> object:
        """Return binary or multiclass decision scores."""


class RawResidualProbeReadout:
    """Multinomial linear probe over one canonical residual-stream layer.

    Scikit-learn is imported only inside :meth:`fit`. A compatible pre-fitted
    estimator can be injected, which keeps scoring usable in lightweight jobs.
    Labels are vocabulary token ids so probe classes can be restricted to the
    exact same candidate support as the other four methods.
    """

    method = ReadoutMethod.RAW_PROBE

    def __init__(
        self,
        *,
        layer: int,
        estimator: _DecisionEstimator | None = None,
        c: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 0,
    ) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ReadoutError("probe layer must be a non-negative integer")
        if c <= 0:
            raise ReadoutError("probe regularization parameter c must be positive")
        if max_iter < 1:
            raise ReadoutError("probe max_iter must be positive")
        self.layer = layer
        self.estimator = estimator
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    @property
    def is_fitted(self) -> bool:
        """Return whether a fitted or injected estimator is available."""

        return self.estimator is not None and hasattr(self.estimator, "classes_")

    def fit(
        self,
        residuals: object,
        target_token_ids: Sequence[int],
        *,
        layer: int | None = None,
    ) -> RawResidualProbeReadout:
        """Fit sklearn LogisticRegression, importing sklearn only on demand."""

        if layer is not None and layer != self.layer:
            raise ReadoutError(
                f"probe was configured for layer {self.layer}, got layer {layer}"
            )
        labels = tuple(int(token_id) for token_id in target_token_ids)
        if len(labels) < 2 or len(set(labels)) < 2:
            raise ReadoutError("probe fitting requires at least two target classes")
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as error:  # pragma: no cover - environment dependent
            raise OptionalDependencyError(
                "raw-probe fitting requires the 'analysis' extra: "
                "pip install -e '.[analysis]'"
            ) from error

        estimator = LogisticRegression(
            C=self.c,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        estimator.fit(residuals, labels)
        self.estimator = estimator
        return self

    def score(self, request: ReadoutRequest) -> ReadoutRecord:
        """Apply the fitted probe and restrict its class logits to candidates."""

        if request.layer != self.layer:
            raise ReadoutError(
                f"raw probe requires layer {self.layer}, got {request.layer}"
            )
        if request.residual is None:
            raise MissingReadoutInputError("residual is required by raw probe")
        if not self.is_fitted:
            raise ReadoutError("raw residual probe has not been fitted")
        assert self.estimator is not None

        row = _single_sample_batch(request.residual)
        raw_scores = self.estimator.decision_function(row)
        classes = tuple(
            int(value)
            for value in _as_1d_scores(self.estimator.classes_, name="classes")
        )
        class_logits = _decision_logits(raw_scores, class_count=len(classes))
        keyed = dict(zip(classes, class_logits, strict=True))
        logits = restrict_candidate_scores(
            keyed,
            request.candidates,
            name="raw probe logits",
        )
        return ReadoutRecord(
            example_id=request.example_id,
            method=self.method,
            candidates=request.candidates,
            logits=logits,
            layer=self.layer,
            metadata=request.metadata,
        )

    def score_batch_logits(
        self,
        residuals: object,
        candidates: CandidateSet,
    ) -> tuple[tuple[float, ...], ...]:
        """Score a residual matrix in one estimator call."""

        if not self.is_fitted:
            raise ReadoutError("raw residual probe has not been fitted")
        assert self.estimator is not None
        classes = tuple(
            int(value)
            for value in _as_1d_scores(self.estimator.classes_, name="classes")
        )
        score_rows = _batch_decision_logits(
            self.estimator.decision_function(residuals),
            class_count=len(classes),
        )
        return tuple(
            restrict_candidate_scores(
                dict(zip(classes, row, strict=True)),
                candidates,
                name="raw probe logits",
            )
            for row in score_rows
        )


class ReadoutPanel:
    """Run and validate a complete, fixed-support five-readout comparison."""

    def __init__(self, readouts: Sequence[object], *, require_all: bool = True) -> None:
        scorers = tuple(readouts)
        methods = tuple(getattr(readout, "method", None) for readout in scorers)
        if not scorers or any(method not in FIVE_READOUT_METHODS for method in methods):
            raise ReadoutError(
                "every panel member must implement a known readout method"
            )
        if len(methods) != len(set(methods)):
            raise ReadoutError("panel readout methods must be unique")
        if require_all and set(methods) != FIVE_READOUT_METHODS:
            missing = sorted(
                method.value for method in FIVE_READOUT_METHODS - set(methods)
            )
            raise ReadoutError(f"complete panel is missing: {', '.join(missing)}")
        self.readouts = scorers
        self.require_all = require_all

    def score(self, request: ReadoutRequest) -> ReadoutPanelResult:
        """Score an example and enforce support, uniqueness, and same-layer rules."""

        records = tuple(readout.score(request) for readout in self.readouts)
        return ReadoutPanelResult(records)


def _vocabulary_record(
    request: ReadoutRequest,
    *,
    method: ReadoutMethod,
    scores: ScoreInput,
    layer: int | None,
) -> ReadoutRecord:
    logits = restrict_candidate_scores(
        scores,
        request.candidates,
        name=f"{method.value} logits",
    )
    return ReadoutRecord(
        example_id=request.example_id,
        method=method,
        candidates=request.candidates,
        logits=logits,
        layer=layer,
        metadata=request.metadata,
    )


def _single_sample_batch(residual: object) -> object:
    """Wrap one vector for sklearn while preserving tensor/array inputs."""

    shape = getattr(residual, "shape", None)
    if shape is not None:
        dimensions = tuple(int(value) for value in shape)
        converted = _tolist(residual)
        if len(dimensions) == 1:
            return [converted]
        if len(dimensions) == 2 and dimensions[0] == 1:
            return converted
        raise ReadoutError("residual must be one vector or a one-row batch")

    if isinstance(residual, (str, bytes)) or not isinstance(residual, Sequence):
        raise ReadoutError("residual must be a one-dimensional numeric vector")
    if residual and isinstance(residual[0], (list, tuple)):
        if len(residual) != 1:
            raise ReadoutError("residual batch must contain exactly one row")
        return residual
    return [residual]


def _decision_logits(scores: object, *, class_count: int) -> tuple[float, ...]:
    """Normalize sklearn's binary and multiclass decision_function shapes."""

    values = scores
    shape = getattr(values, "shape", None)
    if shape is not None:
        dimensions = tuple(int(value) for value in shape)
        if len(dimensions) == 2:
            if dimensions[0] != 1:
                raise ReadoutError("probe returned scores for more than one sample")
            values = values[0]
    elif (
        isinstance(values, Sequence) and values and isinstance(values[0], (list, tuple))
    ):
        if len(values) != 1:
            raise ReadoutError("probe returned scores for more than one sample")
        values = values[0]

    normalized = _as_1d_scores(values, name="probe decision scores")
    if class_count == 2 and len(normalized) == 1:
        margin = normalized[0]
        return (-margin / 2.0, margin / 2.0)
    if len(normalized) != class_count:
        raise ReadoutError(
            f"probe returned {len(normalized)} scores for {class_count} classes"
        )
    return normalized


def _batch_decision_logits(
    scores: object,
    *,
    class_count: int,
) -> tuple[tuple[float, ...], ...]:
    """Normalize sklearn decision scores for multiple samples."""

    values = _tolist(scores)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ReadoutError("probe batch scores must be a sequence")
    if not values:
        raise ReadoutError("probe batch scores cannot be empty")
    if class_count == 2 and not isinstance(values[0], Sequence):
        return tuple(
            _decision_logits([float(margin)], class_count=class_count)
            for margin in values
        )
    rows: list[tuple[float, ...]] = []
    for row in values:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ReadoutError("probe batch scores must contain row sequences")
        rows.append(_decision_logits([row], class_count=class_count))
    return tuple(rows)
