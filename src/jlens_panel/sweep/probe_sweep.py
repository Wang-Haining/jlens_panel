"""Train-only probes, calibrated lens metrics, and frozen sprint gates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jlens_panel.calibration import NullCalibration, calibrate
from jlens_panel.readouts import (
    CandidateSet,
    RawResidualProbeReadout,
    ReadoutMethod,
    ReadoutRecord,
    aggregate_metrics,
    evaluate_record,
)
from jlens_panel.sweep.positions import ALL_POSITION_NAMES

SWEEP_METHODS = (
    "probe",
    "jlens_raw",
    "jlens_calibrated",
    "logitlens_raw",
    "logitlens_calibrated",
)
RESULT_COLUMNS = (
    "position",
    "layer",
    "method",
    "C",
    "dev_top1",
    "dev_mrr",
    "dev_logloss",
    "n_train",
    "n_dev",
)


class ProbeSweepError(ValueError):
    """Raised when sweep arrays, calibration cells, or gates are inconsistent."""


@dataclass(frozen=True, slots=True)
class SweepResultRow:
    """One tidy output row with the exact preregistered CSV columns."""

    position: str
    layer: int
    method: str
    c: float | None
    dev_top1: float
    dev_mrr: float
    dev_logloss: float
    n_train: int
    n_dev: int

    def __post_init__(self) -> None:
        if self.position not in ALL_POSITION_NAMES:
            raise ProbeSweepError(f"unknown sweep position: {self.position!r}")
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ProbeSweepError("sweep row layer must be non-negative")
        if self.method not in SWEEP_METHODS:
            raise ProbeSweepError(f"unknown sweep method: {self.method!r}")
        if (self.method == "probe") != (self.c is not None):
            raise ProbeSweepError("only probe rows may carry C")
        metrics = (self.dev_top1, self.dev_mrr, self.dev_logloss)
        if not all(math.isfinite(value) for value in metrics):
            raise ProbeSweepError("sweep row metrics must be finite")
        if not 0.0 <= self.dev_top1 <= 1.0 or not 0.0 <= self.dev_mrr <= 1.0:
            raise ProbeSweepError("sweep accuracy and MRR must be in [0, 1]")
        if self.n_train < 1 or self.n_dev < 1:
            raise ProbeSweepError("sweep row counts must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-ready mapping with the exact frozen field names."""

        return {
            "position": self.position,
            "layer": self.layer,
            "method": self.method,
            "C": "" if self.c is None else self.c,
            "dev_top1": self.dev_top1,
            "dev_mrr": self.dev_mrr,
            "dev_logloss": self.dev_logloss,
            "n_train": self.n_train,
            "n_dev": self.n_dev,
        }


def _rows(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not (
        hasattr(value, "__len__") and hasattr(value, "__getitem__")
    ):
        raise ProbeSweepError(f"{name} must be a row sequence")
    return value  # type: ignore[return-value]


def _score_row(value: object, *, candidate_count: int, name: str) -> tuple[float, ...]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProbeSweepError(f"{name} must be a score sequence")
    if len(value) != candidate_count:
        raise ProbeSweepError(f"{name} candidate count changed")
    try:
        scores = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ProbeSweepError(f"{name} scores must be numeric") from error
    if not all(math.isfinite(score) for score in scores):
        raise ProbeSweepError(f"{name} scores must be finite")
    return scores


def _aggregate_logits(
    logits: Sequence[Sequence[float]],
    *,
    example_ids: Sequence[str],
    target_token_ids: Sequence[int],
    candidates: CandidateSet,
    method: ReadoutMethod,
    layer: int,
) -> dict[str, float]:
    if not (len(logits) == len(example_ids) == len(target_token_ids)):
        raise ProbeSweepError("dev logits, IDs, and targets have different lengths")
    metrics = (
        evaluate_record(
            ReadoutRecord(
                example_id=example_id,
                method=method,
                candidates=candidates,
                logits=tuple(row),
                layer=layer,
            ),
            target_token_id,
        )
        for row, example_id, target_token_id in zip(
            logits,
            example_ids,
            target_token_ids,
            strict=True,
        )
    )
    return aggregate_metrics(metrics)


def _result_row(
    *,
    position: str,
    layer: int,
    method: str,
    c: float | None,
    metrics: Mapping[str, float],
    n_train: int,
    n_dev: int,
) -> SweepResultRow:
    return SweepResultRow(
        position=position,
        layer=layer,
        method=method,
        c=c,
        dev_top1=float(metrics["top1_accuracy"]),
        dev_mrr=float(metrics["mean_reciprocal_rank"]),
        dev_logloss=float(metrics["log_loss"]),
        n_train=n_train,
        n_dev=n_dev,
    )


def _gate_status(value: float, threshold: float) -> str:
    return "pass" if value >= threshold else "fail"


def run_probe_sweep(
    *,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    layers: Sequence[int],
    train_residuals: Mapping[tuple[str, int], object],
    dev_residuals: Mapping[tuple[str, int], object],
    dev_scores: Mapping[tuple[str, str, int], object],
    train_target_token_ids: Sequence[int],
    dev_target_token_ids: Sequence[int],
    dev_example_ids: Sequence[str],
    calibrations: Mapping[tuple[str, str, int], NullCalibration],
    c_values: Sequence[float],
    gates: Mapping[str, float],
) -> tuple[tuple[SweepResultRow, ...], dict[str, object], dict[str, object]]:
    """Evaluate all cells and return tidy rows, heatmap, and gate summary."""

    inventory = tuple(sorted(candidates))
    normalized_layers = tuple(sorted(set(layers)))
    if len(inventory) != 16 or set(candidate_token_ids) != set(inventory):
        raise ProbeSweepError("sprint sweep requires the frozen 16-way support")
    if not normalized_layers:
        raise ProbeSweepError("sweep source layers cannot be empty")
    if tuple(c_values) != (1.0, 0.01, 0.1):
        raise ProbeSweepError("probe C grid changed")
    expected_gates = {
        "g_info_min_probe_top1": 0.25,
        "g_const_max_label_share": 0.5,
        "g_const_min_entropy_bits": 1.5,
        "g_lens_min_probe_ratio": 0.5,
    }
    if dict(gates) != expected_gates:
        raise ProbeSweepError("sweep gates changed")
    if not train_target_token_ids or not dev_target_token_ids:
        raise ProbeSweepError("sweep requires non-empty train and dev targets")
    if len(dev_target_token_ids) != len(dev_example_ids):
        raise ProbeSweepError("dev IDs and targets have different lengths")
    if len(set(dev_example_ids)) != len(dev_example_ids):
        raise ProbeSweepError("dev example IDs must be unique")
    candidate_set = CandidateSet.from_pairs(
        (candidate_token_ids[candidate], candidate) for candidate in inventory
    )
    if not set(train_target_token_ids).issubset(candidate_set.token_ids) or not set(
        dev_target_token_ids
    ).issubset(candidate_set.token_ids):
        raise ProbeSweepError("sweep targets include a non-candidate token")

    expected_cells = {
        (position, layer)
        for position in ALL_POSITION_NAMES
        for layer in normalized_layers
    }
    if set(train_residuals) != expected_cells or set(dev_residuals) != expected_cells:
        raise ProbeSweepError("residual cell inventory changed")
    expected_score_cells = {
        (method, position, layer)
        for method in ("jlens", "logit_lens")
        for position, layer in expected_cells
    }
    if set(dev_scores) != expected_score_cells or set(calibrations) != (
        expected_score_cells
    ):
        raise ProbeSweepError("lens score or calibration cell inventory changed")
    if any(calibration.n_null_prompts != 200 for calibration in calibrations.values()):
        raise ProbeSweepError("every calibration cell must use 200 null prompts")

    results: list[SweepResultRow] = []
    selected_probe: dict[tuple[str, int], tuple[float, dict[str, float]]] = {}
    lens_metrics: dict[tuple[str, int, str], dict[str, float]] = {}
    calibrated_jlens_logits: dict[tuple[str, int], tuple[tuple[float, ...], ...]] = {}
    n_train = len(train_target_token_ids)
    n_dev = len(dev_target_token_ids)

    for position in ALL_POSITION_NAMES:
        for layer in normalized_layers:
            train_rows = _rows(
                train_residuals[(position, layer)],
                name=f"train residuals {position}/{layer}",
            )
            dev_rows = _rows(
                dev_residuals[(position, layer)],
                name=f"dev residuals {position}/{layer}",
            )
            if len(train_rows) != n_train or len(dev_rows) != n_dev:
                raise ProbeSweepError("residual row counts disagree with targets")
            probe_candidates: list[tuple[float, dict[str, float]]] = []
            for c in c_values:
                probe = RawResidualProbeReadout(layer=layer, c=float(c))
                probe.fit(train_rows, train_target_token_ids, layer=layer)
                probe_metrics = _aggregate_logits(
                    probe.score_batch_logits(dev_rows, candidate_set),
                    example_ids=dev_example_ids,
                    target_token_ids=dev_target_token_ids,
                    candidates=candidate_set,
                    method=ReadoutMethod.RAW_PROBE,
                    layer=layer,
                )
                probe_candidates.append((float(c), probe_metrics))
                results.append(
                    _result_row(
                        position=position,
                        layer=layer,
                        method="probe",
                        c=float(c),
                        metrics=probe_metrics,
                        n_train=n_train,
                        n_dev=n_dev,
                    )
                )
            selected_probe[(position, layer)] = max(
                probe_candidates,
                key=lambda item: (
                    item[1]["top1_accuracy"],
                    -tuple(c_values).index(item[0]),
                ),
            )

            for raw_method, raw_output_name, calibrated_output_name, enum_method in (
                (
                    "jlens",
                    "jlens_raw",
                    "jlens_calibrated",
                    ReadoutMethod.JLENS,
                ),
                (
                    "logit_lens",
                    "logitlens_raw",
                    "logitlens_calibrated",
                    ReadoutMethod.LOGIT_LENS,
                ),
            ):
                raw_rows = _rows(
                    dev_scores[(raw_method, position, layer)],
                    name=f"{raw_method} scores {position}/{layer}",
                )
                if len(raw_rows) != n_dev:
                    raise ProbeSweepError("lens score row count disagrees with dev")
                normalized_raw = tuple(
                    _score_row(
                        row,
                        candidate_count=len(inventory),
                        name=f"{raw_method} scores",
                    )
                    for row in raw_rows
                )
                calibration = calibrations[(raw_method, position, layer)]
                normalized_calibrated = tuple(
                    tuple(
                        calibrate(
                            dict(zip(inventory, row, strict=True)),
                            calibration,
                            mode="center",
                        )[candidate]
                        for candidate in inventory
                    )
                    for row in normalized_raw
                )
                raw_metrics = _aggregate_logits(
                    normalized_raw,
                    example_ids=dev_example_ids,
                    target_token_ids=dev_target_token_ids,
                    candidates=candidate_set,
                    method=enum_method,
                    layer=layer,
                )
                calibrated_metrics = _aggregate_logits(
                    normalized_calibrated,
                    example_ids=dev_example_ids,
                    target_token_ids=dev_target_token_ids,
                    candidates=candidate_set,
                    method=enum_method,
                    layer=layer,
                )
                lens_metrics[(position, layer, raw_output_name)] = raw_metrics
                lens_metrics[(position, layer, calibrated_output_name)] = (
                    calibrated_metrics
                )
                if raw_method == "jlens":
                    calibrated_jlens_logits[(position, layer)] = normalized_calibrated
                results.extend(
                    (
                        _result_row(
                            position=position,
                            layer=layer,
                            method=raw_output_name,
                            c=None,
                            metrics=raw_metrics,
                            n_train=n_train,
                            n_dev=n_dev,
                        ),
                        _result_row(
                            position=position,
                            layer=layer,
                            method=calibrated_output_name,
                            c=None,
                            metrics=calibrated_metrics,
                            n_train=n_train,
                            n_dev=n_dev,
                        ),
                    )
                )

    best_position, best_layer = ALL_POSITION_NAMES[0], normalized_layers[0]
    best_c, best_probe_metrics = selected_probe[(best_position, best_layer)]
    for position in ALL_POSITION_NAMES:
        for layer in normalized_layers:
            candidate_c, candidate_metrics = selected_probe[(position, layer)]
            if candidate_metrics["top1_accuracy"] > best_probe_metrics["top1_accuracy"]:
                best_position = position
                best_layer = layer
                best_c = candidate_c
                best_probe_metrics = candidate_metrics

    best_key = (best_position, best_layer)
    predictions = tuple(
        inventory[max(range(len(inventory)), key=lambda index: (row[index], -index))]
        for row in calibrated_jlens_logits[best_key]
    )
    prediction_counts = Counter(predictions)
    proportions = {
        candidate: prediction_counts[candidate] / n_dev for candidate in inventory
    }
    max_label = max(
        inventory,
        key=lambda candidate: (
            proportions[candidate],
            -inventory.index(candidate),
        ),
    )
    max_share = proportions[max_label]
    entropy_bits = -math.fsum(
        probability * math.log2(probability)
        for probability in proportions.values()
        if probability > 0.0
    )

    g_info_value = float(best_probe_metrics["top1_accuracy"])
    g_info_status = _gate_status(g_info_value, expected_gates["g_info_min_probe_top1"])
    if (
        max_share <= expected_gates["g_const_max_label_share"]
        and entropy_bits > expected_gates["g_const_min_entropy_bits"]
    ):
        g_const_status = "pass"
    else:
        g_const_status = "fail"

    jlens_top1 = lens_metrics[(best_position, best_layer, "jlens_calibrated")][
        "top1_accuracy"
    ]
    logit_lens_top1 = lens_metrics[(best_position, best_layer, "logitlens_calibrated")][
        "top1_accuracy"
    ]
    ratio_threshold = expected_gates["g_lens_min_probe_ratio"] * g_info_value
    if g_info_status != "pass":
        g_lens_status = "not_reached"
    elif jlens_top1 >= ratio_threshold and jlens_top1 > logit_lens_top1:
        g_lens_status = "pass"
    else:
        g_lens_status = "fail"

    gates_summary: dict[str, object] = {
        "selection_protocol": {
            "selection_split": "dev",
            "evaluation_split": "dev",
            "same_split_selection_and_reporting": True,
            "probe_C_tie_order": [1.0, 0.01, 0.1],
            "best_cell_tie_order": {
                "positions": list(ALL_POSITION_NAMES),
                "layers": list(normalized_layers),
            },
            "interpretation": "selected diagnostic maximum, not an unbiased estimate",
        },
        "best_probe_cell": {
            "position": best_position,
            "layer": best_layer,
            "selected_C": best_c,
            "dev_top1": g_info_value,
        },
        "g_info": {
            "status": g_info_status,
            "value": g_info_value,
            "threshold": expected_gates["g_info_min_probe_top1"],
        },
        "g_const": {
            "status": g_const_status,
            "evaluated_at": "best_probe_cell",
            "position": best_position,
            "layer": best_layer,
            "candidate_argmax_tie_order": list(inventory),
            "max_label": max_label,
            "max_label_share": max_share,
            "max_share_threshold": expected_gates["g_const_max_label_share"],
            "entropy_bits": entropy_bits,
            "entropy_threshold": expected_gates["g_const_min_entropy_bits"],
            "prediction_counts": dict(prediction_counts),
        },
        "g_lens": {
            "status": g_lens_status,
            "evaluated_at": "best_probe_cell",
            "position": best_position,
            "layer": best_layer,
            "jlens_calibrated_top1": jlens_top1,
            "probe_top1": g_info_value,
            "probe_ratio": jlens_top1 / g_info_value if g_info_value else None,
            "minimum_ratio": expected_gates["g_lens_min_probe_ratio"],
            "logitlens_calibrated_top1": logit_lens_top1,
        },
    }
    heatmap: dict[str, object] = {
        "schema_version": "jlens-panel-sweep-heatmap-v1",
        "selection": "maximum dev_top1; C tie order 1.0, 0.01, 0.1",
        "best_cell_tie_order": {
            "positions": list(ALL_POSITION_NAMES),
            "layers": list(normalized_layers),
        },
        "same_split_selection_and_reporting": True,
        "probe_dev_top1": {
            position: {
                str(layer): selected_probe[(position, layer)][1]["top1_accuracy"]
                for layer in normalized_layers
            }
            for position in ALL_POSITION_NAMES
        },
        "selected_C": {
            position: {
                str(layer): selected_probe[(position, layer)][0]
                for layer in normalized_layers
            }
            for position in ALL_POSITION_NAMES
        },
        "gates": gates_summary,
    }
    return tuple(results), heatmap, gates_summary
