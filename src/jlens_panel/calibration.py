"""Validated null-score calibration with atomic, self-checking artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from jlens_panel.provenance import sha256_file, write_json_atomic

CALIBRATION_SCHEMA = "jlens-panel-null-calibration-v1"
CALIBRATION_METHODS = frozenset({"jlens", "logit_lens"})
CALIBRATION_POSITION_TYPES = frozenset(
    {
        "template_tail",
        "content_last",
        "clue_last",
        "meanpool_content8",
        "decode_1",
        "decode_2",
        "decode_4",
        "decode_8",
    }
)
CALIBRATION_MODES = frozenset({"center", "zscore"})
ZSCORE_STD_FLOOR = 1e-6
POSITION_RESOLVER_SCHEMA = "jlens-panel-positions-v1"
NULL_RENDERING_POLICY = MappingProxyType(
    {
        "template_tail": "chat_wrapped_tail",
        "content_last": "raw_text_tail",
        "clue_last": "raw_text_tail",
        "meanpool_content8": "raw_text_last8_mean",
        "decode_1": "chat_wrapped_generated_token",
        "decode_2": "chat_wrapped_generated_token",
        "decode_4": "chat_wrapped_generated_token",
        "decode_8": "chat_wrapped_generated_token",
    }
)
REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "model_name",
        "model_revision",
        "config_sha256",
        "git_revision",
        "upstream_commit",
        "jlens_source_sha256",
        "transformers_version",
        "jlens_version",
        "torch_version",
        "cuda_runtime",
        "cuda_driver_version",
        "gpu_name",
        "gpu_compute_capability",
        "deterministic_algorithms",
        "allow_tf32",
        "cublas_workspace_config",
        "chat_template_sha256",
        "eos_policy",
        "lens_sha256",
        "corpus_sha256",
        "sample_sha256",
        "sample_seed",
        "null_prompt_count",
        "candidate_token_ids",
        "source_layers",
        "max_seq_len",
        "decode_steps",
        "ddof",
        "resolver_schema",
        "rendering_policy",
    }
)


class CalibrationError(ValueError):
    """Raised when calibration inputs violate the frozen contract."""


class CalibrationArtifactError(CalibrationError):
    """Raised when a serialized calibration is missing, corrupt, or conflicting."""


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CalibrationError("JSON mapping keys must be strings")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationError("JSON numeric values must be finite")
        return value
    raise CalibrationError("calibration values must be JSON-compatible")


def _deep_freeze_json(value: object) -> object:
    ready = _json_ready(value)
    if isinstance(ready, dict):
        return MappingProxyType(
            {key: _deep_freeze_json(item) for key, item in ready.items()}
        )
    if isinstance(ready, list):
        return tuple(_deep_freeze_json(item) for item in ready)
    return ready


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CalibrationError("calibration values must be JSON-compatible") from error


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validated_scores(
    value: Mapping[str, object],
    *,
    name: str,
    nonnegative: bool,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or len(value) < 2:
        raise CalibrationError(f"{name} must contain at least two candidates")
    validated: dict[str, float] = {}
    for candidate, raw_score in value.items():
        if (
            not isinstance(candidate, str)
            or not candidate
            or candidate.strip() != candidate
        ):
            raise CalibrationError(f"{name} candidate keys must be stripped strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise CalibrationError(f"{name} values must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise CalibrationError(f"{name} values must be finite")
        if nonnegative and score < 0.0:
            raise CalibrationError(f"{name} values must be non-negative")
        validated[candidate] = score
    return dict(sorted(validated.items()))


def _validate_provenance(
    provenance: Mapping[str, object],
    *,
    candidates: Sequence[str],
    layer: int,
    n_null_prompts: int,
) -> Mapping[str, object]:
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != REQUIRED_PROVENANCE_FIELDS
    ):
        raise CalibrationError("calibration provenance fields do not match the schema")
    value = _json_ready(provenance)
    assert isinstance(value, dict)
    if not isinstance(value["model_name"], str) or not value["model_name"]:
        raise CalibrationError("provenance model_name must be non-empty")
    if (
        not isinstance(value["model_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["model_revision"]) is None
    ):
        raise CalibrationError("provenance model_revision must be a pinned commit")
    for name in (
        "config_sha256",
        "lens_sha256",
        "corpus_sha256",
        "sample_sha256",
    ):
        if not _is_sha256(value[name]):
            raise CalibrationError(f"provenance {name} must be a SHA-256 digest")
    if (
        not isinstance(value["git_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["git_revision"]) is None
    ):
        raise CalibrationError("provenance git_revision must be a pinned commit")
    if (
        not isinstance(value["upstream_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["upstream_commit"]) is None
    ):
        raise CalibrationError("provenance upstream_commit must be pinned")
    for name in ("jlens_source_sha256", "chat_template_sha256"):
        if not _is_sha256(value[name]):
            raise CalibrationError(f"provenance {name} must be a SHA-256 digest")
    for name in (
        "transformers_version",
        "jlens_version",
        "torch_version",
        "cuda_runtime",
        "cuda_driver_version",
        "gpu_name",
    ):
        if not isinstance(value[name], str) or not value[name]:
            raise CalibrationError(f"provenance {name} must be non-empty")
    if "H100" not in value["gpu_name"]:
        raise CalibrationError("provenance GPU must be an H100")
    capability = value["gpu_compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in capability
        )
    ):
        raise CalibrationError("provenance GPU compute capability is invalid")
    if value["deterministic_algorithms"] is not True:
        raise CalibrationError("provenance deterministic algorithms must be enabled")
    if value["allow_tf32"] is not False:
        raise CalibrationError("provenance TF32 must be disabled")
    if value["cublas_workspace_config"] != ":4096:8":
        raise CalibrationError("provenance CUBLAS workspace config changed")
    if value["eos_policy"] != "mask_eos_for_tokens_1_through_7":
        raise CalibrationError("provenance EOS policy changed")
    if isinstance(value["sample_seed"], bool) or not isinstance(
        value["sample_seed"], int
    ):
        raise CalibrationError("provenance sample_seed must be an integer")
    if (
        isinstance(value["null_prompt_count"], bool)
        or not isinstance(value["null_prompt_count"], int)
        or value["null_prompt_count"] != n_null_prompts
    ):
        raise CalibrationError("provenance null_prompt_count disagrees with artifact")
    if (
        isinstance(value["max_seq_len"], bool)
        or not isinstance(value["max_seq_len"], int)
        or value["max_seq_len"] < 1
    ):
        raise CalibrationError("provenance max_seq_len must be positive")
    decode_steps = value["decode_steps"]
    if (
        not isinstance(decode_steps, list)
        or any(
            isinstance(step, bool) or not isinstance(step, int) for step in decode_steps
        )
        or decode_steps != [1, 2, 4, 8]
    ):
        raise CalibrationError("provenance decode_steps must be [1, 2, 4, 8]")
    if isinstance(value["ddof"], bool) or not isinstance(value["ddof"], int):
        raise CalibrationError("provenance ddof must be integer zero")
    if value["ddof"] != 0:
        raise CalibrationError("provenance ddof must be zero for population moments")
    if value["resolver_schema"] != POSITION_RESOLVER_SCHEMA:
        raise CalibrationError("provenance resolver schema changed")
    if value["rendering_policy"] != NULL_RENDERING_POLICY:
        raise CalibrationError("provenance null rendering policy changed")

    token_ids = value["candidate_token_ids"]
    if not isinstance(token_ids, dict) or set(token_ids) != set(candidates):
        raise CalibrationError("provenance candidate token inventory changed")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids.values()
    ) or len(set(token_ids.values())) != len(token_ids):
        raise CalibrationError("provenance candidate token IDs must be unique integers")

    source_layers = value["source_layers"]
    if (
        not isinstance(source_layers, list)
        or not source_layers
        or any(
            isinstance(source_layer, bool)
            or not isinstance(source_layer, int)
            or source_layer < 0
            for source_layer in source_layers
        )
        or source_layers != sorted(set(source_layers))
        or layer not in source_layers
    ):
        raise CalibrationError("provenance source layers are invalid for this cell")
    return _deep_freeze_json(value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NullCalibration:
    """Candidate-wise null moments for one method, position, and layer."""

    method: str
    position_type: str
    layer: int
    candidate_means: dict[str, float]
    candidate_stds: dict[str, float]
    n_null_prompts: int
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise CalibrationError(f"unsupported calibration method: {self.method!r}")
        if self.position_type not in CALIBRATION_POSITION_TYPES:
            raise CalibrationError(
                f"unsupported calibration position: {self.position_type!r}"
            )
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise CalibrationError("calibration layer must be an integer")
        if self.layer < 0:
            raise CalibrationError("calibration layer must be non-negative")
        if (
            isinstance(self.n_null_prompts, bool)
            or not isinstance(self.n_null_prompts, int)
            or self.n_null_prompts < 1
        ):
            raise CalibrationError("n_null_prompts must be a positive integer")

        means = _validated_scores(
            self.candidate_means,
            name="candidate_means",
            nonnegative=False,
        )
        stds = _validated_scores(
            self.candidate_stds,
            name="candidate_stds",
            nonnegative=True,
        )
        if means.keys() != stds.keys():
            raise CalibrationError("candidate means and standard deviations disagree")
        provenance = _validate_provenance(
            self.provenance,
            candidates=tuple(means),
            layer=self.layer,
            n_null_prompts=self.n_null_prompts,
        )

        object.__setattr__(self, "candidate_means", MappingProxyType(means))
        object.__setattr__(self, "candidate_stds", MappingProxyType(stds))
        object.__setattr__(self, "provenance", provenance)

    @property
    def candidates(self) -> tuple[str, ...]:
        """Return candidate labels in the artifact's stable insertion order."""

        return tuple(self.candidate_means)

    def to_payload(self) -> dict[str, object]:
        """Return the exact JSON payload protected by the embedded checksum."""

        return {
            "method": self.method,
            "position_type": self.position_type,
            "layer": self.layer,
            "candidate_means": dict(self.candidate_means),
            "candidate_stds": dict(self.candidate_stds),
            "n_null_prompts": self.n_null_prompts,
            "provenance": _json_ready(self.provenance),
        }

    def save(self, path: str | Path) -> str:
        """Atomically save a new self-checking artifact and return its file SHA-256."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=output.parent,
                prefix=f".{output.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            write_json_atomic(
                temporary,
                {
                    "schema_version": CALIBRATION_SCHEMA,
                    "payload_sha256": _payload_sha256(payload),
                    "calibration": payload,
                },
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise CalibrationArtifactError(
                    f"refusing to overwrite calibration artifact: {output}"
                ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return sha256_file(output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> NullCalibration:
        """Load and validate schema, payload checksum, and optional file checksum."""

        source = Path(path)
        if not source.is_file():
            raise CalibrationArtifactError(f"missing calibration artifact: {source}")
        try:
            raw_bytes = source.read_bytes()
        except OSError as error:
            raise CalibrationArtifactError(
                f"cannot read calibration artifact: {source}"
            ) from error
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if expected_sha256 is not None:
            if not _is_sha256(expected_sha256):
                raise CalibrationArtifactError(
                    "expected calibration SHA-256 is malformed"
                )
            if file_sha256 != expected_sha256:
                raise CalibrationArtifactError("calibration file SHA-256 mismatch")
        try:
            artifact = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CalibrationArtifactError(
                f"cannot decode calibration artifact: {source}"
            ) from error
        if not isinstance(artifact, dict) or set(artifact) != {
            "schema_version",
            "payload_sha256",
            "calibration",
        }:
            raise CalibrationArtifactError("invalid calibration artifact envelope")
        if artifact["schema_version"] != CALIBRATION_SCHEMA:
            raise CalibrationArtifactError("unsupported calibration schema")
        payload = artifact["calibration"]
        if not isinstance(payload, dict) or set(payload) != {
            "method",
            "position_type",
            "layer",
            "candidate_means",
            "candidate_stds",
            "n_null_prompts",
            "provenance",
        }:
            raise CalibrationArtifactError("invalid calibration payload fields")
        if artifact["payload_sha256"] != _payload_sha256(payload):
            raise CalibrationArtifactError("calibration payload SHA-256 mismatch")
        try:
            return cls(**payload)
        except (CalibrationError, TypeError) as error:
            raise CalibrationArtifactError(
                "invalid calibration payload values"
            ) from error


@dataclass(slots=True)
class NullScoreAccumulator:
    """Online population moments over candidate-only score mappings."""

    candidates: tuple[str, ...]
    count: int = 0
    _means: dict[str, float] = field(init=False, repr=False)
    _m2: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates))
        if len(candidates) < 2 or len(candidates) != len(set(candidates)):
            raise CalibrationError("accumulator candidates must be unique")
        if any(
            not isinstance(candidate, str)
            or not candidate
            or candidate.strip() != candidate
            for candidate in candidates
        ):
            raise CalibrationError("accumulator candidates must be stripped strings")
        if self.count != 0:
            raise CalibrationError("new accumulator count must be zero")
        self.candidates = candidates
        self._means = dict.fromkeys(candidates, 0.0)
        self._m2 = dict.fromkeys(candidates, 0.0)

    def update(self, scores: Mapping[str, object]) -> None:
        """Consume one complete candidate-restricted score mapping."""

        values = _validated_scores(scores, name="null scores", nonnegative=False)
        if set(values) != set(self.candidates):
            raise CalibrationError("null score candidate support changed")
        self.count += 1
        for candidate, value in values.items():
            delta = value - self._means[candidate]
            self._means[candidate] += delta / self.count
            delta_after = value - self._means[candidate]
            self._m2[candidate] += delta * delta_after

    def build(
        self,
        *,
        method: str,
        position_type: str,
        layer: int,
        provenance: Mapping[str, object],
    ) -> NullCalibration:
        """Freeze population moments into a validated calibration artifact."""

        if self.count < 1:
            raise CalibrationError("cannot build calibration without null scores")
        stds = {
            candidate: math.sqrt(max(0.0, self._m2[candidate] / self.count))
            for candidate in self.candidates
        }
        return NullCalibration(
            method=method,
            position_type=position_type,
            layer=layer,
            candidate_means=dict(self._means),
            candidate_stds=stds,
            n_null_prompts=self.count,
            provenance=provenance,
        )

    def to_state(self) -> dict[str, object]:
        """Return a JSON-compatible resumable sufficient-statistics state."""

        return {
            "candidates": list(self.candidates),
            "count": self.count,
            "means": dict(self._means),
            "m2": dict(self._m2),
        }

    @classmethod
    def from_state(cls, value: Mapping[str, object]) -> NullScoreAccumulator:
        """Restore validated Welford statistics without replaying prompt scores."""

        if not isinstance(value, Mapping) or set(value) != {
            "candidates",
            "count",
            "means",
            "m2",
        }:
            raise CalibrationError("null accumulator state fields changed")
        raw_candidates = value["candidates"]
        if isinstance(raw_candidates, (str, bytes)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise CalibrationError("null accumulator state candidates are invalid")
        accumulator = cls(tuple(raw_candidates))  # type: ignore[arg-type]
        count = value["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CalibrationError("null accumulator state count is invalid")
        means = _validated_scores(
            value["means"],  # type: ignore[arg-type]
            name="null accumulator means",
            nonnegative=False,
        )
        m2 = _validated_scores(
            value["m2"],  # type: ignore[arg-type]
            name="null accumulator m2",
            nonnegative=True,
        )
        if tuple(means) != accumulator.candidates or tuple(m2) != (
            accumulator.candidates
        ):
            raise CalibrationError("null accumulator state support changed")
        if count == 0 and (any(means.values()) or any(m2.values())):
            raise CalibrationError("empty null accumulator state has nonzero moments")
        if count == 1 and any(m2.values()):
            raise CalibrationError("single-item null accumulator has nonzero m2")
        accumulator.count = count
        accumulator._means = means
        accumulator._m2 = m2
        return accumulator


def calibrate(
    scores: Mapping[str, float],
    cal: NullCalibration,
    *,
    mode: str = "center",
) -> dict[str, float]:
    """Apply candidate-wise centering or z-scoring with a frozen std floor."""

    if mode not in CALIBRATION_MODES:
        raise CalibrationError(f"unsupported calibration mode: {mode!r}")
    values = _validated_scores(scores, name="scores", nonnegative=False)
    if set(values) != set(cal.candidates):
        raise CalibrationError("score candidate support disagrees with calibration")
    centered = {
        candidate: values[candidate] - cal.candidate_means[candidate]
        for candidate in cal.candidates
    }
    if mode == "center":
        return centered
    return {
        candidate: centered[candidate]
        / max(cal.candidate_stds[candidate], ZSCORE_STD_FLOOR)
        for candidate in cal.candidates
    }


def load_calibrations(
    paths: Sequence[str | Path],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[tuple[str, str, int], NullCalibration]:
    """Load a unique method-position-layer calibration lookup."""

    lookup: dict[tuple[str, str, int], NullCalibration] = {}
    normalized_paths = tuple(Path(raw_path).resolve() for raw_path in paths)
    if expected_sha256 is None:
        expected: dict[str, str] = {}
    else:
        expected = {}
        for raw_path, digest in expected_sha256.items():
            normalized = str(Path(raw_path).resolve())
            if normalized in expected:
                raise CalibrationArtifactError(
                    "expected SHA-256 mapping contains path aliases"
                )
            if not _is_sha256(digest):
                raise CalibrationArtifactError(
                    "expected SHA-256 mapping contains a malformed digest"
                )
            expected[normalized] = digest
        actual_keys = {str(path) for path in normalized_paths}
        if set(expected) != actual_keys:
            raise CalibrationArtifactError(
                "expected SHA-256 mapping must exactly cover calibration paths"
            )
    reference_provenance: str | None = None
    reference_candidates: tuple[str, ...] | None = None
    for path in normalized_paths:
        calibration = NullCalibration.load(
            path,
            expected_sha256=(
                expected[str(path)] if expected_sha256 is not None else None
            ),
        )
        serialized_provenance = _canonical_json(calibration.provenance)
        if reference_provenance is None:
            reference_provenance = serialized_provenance
            reference_candidates = calibration.candidates
        elif (
            serialized_provenance != reference_provenance
            or calibration.candidates != reference_candidates
        ):
            raise CalibrationArtifactError(
                "calibration bundle mixes incompatible provenance or candidates"
            )
        key = (
            calibration.method,
            calibration.position_type,
            calibration.layer,
        )
        if key in lookup:
            raise CalibrationArtifactError(f"duplicate calibration cell: {key}")
        lookup[key] = calibration
    if not lookup:
        raise CalibrationArtifactError("no calibration artifacts were provided")
    return lookup
