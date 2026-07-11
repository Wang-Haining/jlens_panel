import importlib
import json
import math
import sys

import pytest

from jlens_panel.readouts import (
    CandidateSet,
    JLensReadout,
    LayerMismatchError,
    LogitLensReadout,
    NextTokenReadout,
    RawResidualProbeReadout,
    ReadoutError,
    ReadoutMethod,
    ReadoutPanel,
    ReadoutRecord,
    ReadoutRequest,
    TextOnlyReadout,
    aggregate_metrics,
    evaluate_record,
    multiclass_log_loss,
    restrict_candidate_scores,
    stable_rank,
    stable_softmax,
    validate_same_layer,
)


class FakeMulticlassEstimator:
    classes_ = [2, 5, 7]

    def decision_function(self, values: object) -> list[list[float]]:
        assert values == [[0.25, -0.5]]
        return [[-1.0, 3.0, 0.5]]


class FakeBinaryEstimator:
    classes_ = [2, 5]

    def decision_function(self, values: object) -> list[float]:
        return [4.0]


class FakeBatchEstimator:
    classes_ = [2, 5, 7]

    def decision_function(self, values: object) -> list[list[float]]:
        assert values == [[0.25, -0.5], [-0.25, 0.5]]
        return [[-1.0, 3.0, 0.5], [2.0, -2.0, 1.0]]


class FakeBinaryBatchEstimator:
    classes_ = [2, 5]

    def decision_function(self, values: object) -> list[float]:
        assert values == [[0.1, 0.2], [0.3, 0.4]]
        return [4.0, -2.0]


class FakeTensorVector:
    shape = (2,)

    def detach(self) -> "FakeTensorVector":
        return self

    def cpu(self) -> "FakeTensorVector":
        return self

    def tolist(self) -> list[float]:
        return [0.25, -0.5]


@pytest.fixture
def candidates() -> CandidateSet:
    return CandidateSet.from_pairs([(2, "Mars"), (5, "Venus"), (7, "Jupiter")])


def test_import_does_not_eagerly_load_heavy_optional_dependencies() -> None:
    before = set(sys.modules)
    module = importlib.import_module("jlens_panel.readouts")
    imported = set(sys.modules) - before

    assert module.ReadoutMethod.JLENS.value == "jlens"
    assert "torch" not in imported
    assert "sklearn" not in imported


def test_candidate_restriction_accepts_vectors_and_keyed_scores(
    candidates: CandidateSet,
) -> None:
    assert restrict_candidate_scores(
        [0.0, 0.1, 2.0, 0.3, 0.4, 5.0, 0.6, 7.0], candidates
    ) == (2.0, 5.0, 7.0)
    assert restrict_candidate_scores({"Mars": 1.0, 5: 2.0, "7": 3.0}, candidates) == (
        1.0,
        2.0,
        3.0,
    )

    with pytest.raises(ReadoutError, match="has no score"):
        restrict_candidate_scores({2: 1.0, 5: 2.0}, candidates)


def test_candidate_contract_rejects_duplicates() -> None:
    with pytest.raises(ReadoutError, match="token_ids must be unique"):
        CandidateSet.from_pairs([(2, "Mars"), (2, "Venus")])
    with pytest.raises(ReadoutError, match="texts must be unique"):
        CandidateSet.from_pairs([(2, "Mars"), (5, "Mars")])


def test_metrics_are_stable_for_extreme_logits_and_ties() -> None:
    probabilities = stable_softmax([10_000.0, 9_999.0, -10_000.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert math.isfinite(multiclass_log_loss([10_000.0, -10_000.0], 1))
    assert multiclass_log_loss([10_000.0, -10_000.0], 1) == pytest.approx(20_000.0)
    assert stable_rank([3.0, 3.0, 2.0], 0) == 1
    assert stable_rank([3.0, 3.0, 2.0], 1) == 2


def test_complete_panel_scores_same_candidates_and_serializes(
    candidates: CandidateSet,
) -> None:
    request = ReadoutRequest(
        example_id="bridge-001",
        candidates=candidates,
        layer=12,
        jlens_logits={2: 1.0, 5: 4.0, 7: -1.0},
        logit_lens_logits=[0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0],
        residual=[0.25, -0.5],
        next_token_logits={2: -2.0, 5: 0.0, 7: 5.0},
        text="The fourth planet is the bridge concept.",
        metadata={"split": "dev", "seed": 17},
    )
    panel = ReadoutPanel(
        [
            JLensReadout(),
            LogitLensReadout(),
            RawResidualProbeReadout(
                layer=12,
                estimator=FakeMulticlassEstimator(),
            ),
            NextTokenReadout(),
            TextOnlyReadout(
                lambda text, support: {
                    candidate.text: float(index)
                    for index, candidate in enumerate(support)
                }
            ),
        ]
    )

    result = panel.score(request)

    assert {record.method for record in result.records} == set(ReadoutMethod)
    assert validate_same_layer(result.records) == 12
    assert all(record.candidates == candidates for record in result.records)
    raw_probe = next(
        record for record in result.records if record.method is ReadoutMethod.RAW_PROBE
    )
    assert raw_probe.logits == (-1.0, 3.0, 0.5)
    assert json.loads(json.dumps(result.to_dict()))["example_id"] == "bridge-001"

    metrics = result.evaluate(target_token_id=5)
    assert len(metrics) == 5
    assert all(math.isfinite(row.log_loss) for row in metrics)


def test_same_layer_validation_rejects_mismatch(candidates: CandidateSet) -> None:
    records = (
        ReadoutRecord(
            example_id="example",
            method=ReadoutMethod.JLENS,
            candidates=candidates,
            logits=(1.0, 2.0, 3.0),
            layer=10,
        ),
        ReadoutRecord(
            example_id="example",
            method=ReadoutMethod.LOGIT_LENS,
            candidates=candidates,
            logits=(1.0, 2.0, 3.0),
            layer=11,
        ),
    )

    with pytest.raises(LayerMismatchError, match="must match"):
        validate_same_layer(records)


def test_raw_probe_enforces_configured_layer(candidates: CandidateSet) -> None:
    probe = RawResidualProbeReadout(
        layer=8,
        estimator=FakeMulticlassEstimator(),
    )
    request = ReadoutRequest(
        example_id="example",
        candidates=candidates,
        layer=7,
        residual=[0.25, -0.5],
    )

    with pytest.raises(ReadoutError, match="requires layer 8"):
        probe.score(request)


def test_raw_probe_converts_tensor_like_residual_before_sklearn(
    candidates: CandidateSet,
) -> None:
    probe = RawResidualProbeReadout(
        layer=8,
        estimator=FakeMulticlassEstimator(),
    )

    record = probe.score(
        ReadoutRequest(
            example_id="tensor-like",
            candidates=candidates,
            layer=8,
            residual=FakeTensorVector(),
        )
    )

    assert record.logits == (-1.0, 3.0, 0.5)


def test_binary_probe_margin_becomes_two_candidate_logits() -> None:
    candidates = CandidateSet.from_pairs([(2, "Mars"), (5, "Venus")])
    probe = RawResidualProbeReadout(layer=3, estimator=FakeBinaryEstimator())
    record = probe.score(
        ReadoutRequest(
            example_id="binary",
            candidates=candidates,
            layer=3,
            residual=[0.1, 0.2],
        )
    )

    assert record.logits == (-2.0, 2.0)
    assert record.probabilities[1] > 0.98


def test_probe_batch_scoring_calls_estimator_once(candidates: CandidateSet) -> None:
    probe = RawResidualProbeReadout(layer=8, estimator=FakeBatchEstimator())

    logits = probe.score_batch_logits(
        [[0.25, -0.5], [-0.25, 0.5]],
        candidates,
    )

    assert logits == ((-1.0, 3.0, 0.5), (2.0, -2.0, 1.0))


def test_binary_probe_batch_normalizes_each_margin() -> None:
    candidates = CandidateSet.from_pairs([(2, "Mars"), (5, "Venus")])
    probe = RawResidualProbeReadout(layer=3, estimator=FakeBinaryBatchEstimator())

    logits = probe.score_batch_logits([[0.1, 0.2], [0.3, 0.4]], candidates)

    assert logits == ((-2.0, 2.0), (1.0, -1.0))


def test_metric_records_aggregate(candidates: CandidateSet) -> None:
    first = ReadoutRecord(
        example_id="one",
        method=ReadoutMethod.JLENS,
        candidates=candidates,
        logits=(4.0, 2.0, 0.0),
        layer=4,
    )
    second = ReadoutRecord(
        example_id="two",
        method=ReadoutMethod.JLENS,
        candidates=candidates,
        logits=(0.0, 2.0, 4.0),
        layer=4,
    )

    summary = aggregate_metrics([evaluate_record(first, 2), evaluate_record(second, 2)])

    assert summary["top1_accuracy"] == 0.5
    assert summary["mean_reciprocal_rank"] == pytest.approx(2.0 / 3.0)
    assert summary["log_loss"] > 0.0


def test_panel_requires_all_five_methods() -> None:
    with pytest.raises(ReadoutError, match="missing"):
        ReadoutPanel([JLensReadout()], require_all=True)
