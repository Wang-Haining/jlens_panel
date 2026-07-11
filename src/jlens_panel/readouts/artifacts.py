"""Compact, validated, and atomic persistence for readout extraction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from jlens_panel.provenance import sha256_file, write_json_atomic
from jlens_panel.storage import directory_size

ARTIFACT_SCHEMA = "jlens-panel-readout-artifact-v1"
EXTRACTION_MANIFEST_SCHEMA = "jlens-panel-readout-extraction-manifest-v1"
EXTRACTION_MANIFEST_NAME = "extraction_manifest.json"
SCORE_SCHEMA = "jlens-panel-readout-score-v1"
SPLITS = ("train", "dev", "test")
STORED_SCORE_METHODS = ("jlens", "logit_lens", "next_token", "text_only")
_SAFE_ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


class ArtifactError(ValueError):
    """Raised when an extraction input or artifact violates the frozen schema."""


class ArtifactConflictError(ArtifactError):
    """Raised when resume finds an artifact from a different extraction."""


class RunHardLimitError(RuntimeError):
    """Raised before an atomic artifact write would exceed the run hard limit."""


@dataclass(slots=True)
class RunByteBudget:
    """Incremental byte accounting that avoids rescanning the run per example."""

    root: Path
    hard_limit_bytes: int
    current_bytes: int

    @classmethod
    def inspect(cls, root: str | Path, hard_limit_bytes: int) -> RunByteBudget:
        """Scan the run once and initialize an incremental hard-limit budget."""

        if hard_limit_bytes <= 0:
            raise RunHardLimitError("run hard limit must be positive")
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        current = directory_size(path)
        if current > hard_limit_bytes:
            raise RunHardLimitError(
                f"run already exceeds hard limit: {current} > {hard_limit_bytes} bytes"
            )
        return cls(path.resolve(), int(hard_limit_bytes), current)

    def check_addition(self, artifact_bytes: int) -> None:
        """Fail before committing an artifact that would exceed the limit."""

        prospective = self.current_bytes + artifact_bytes
        if prospective > self.hard_limit_bytes:
            raise RunHardLimitError(
                f"artifact write would exceed run hard limit: "
                f"{prospective} > {self.hard_limit_bytes} bytes"
            )

    def commit(self, artifact_bytes: int) -> None:
        """Account for one successfully committed artifact."""

        self.current_bytes += artifact_bytes


@dataclass(frozen=True, slots=True)
class ExtractionExample:
    """Minimal extraction view over one synthetic bridge example."""

    example_id: str
    split: str
    candidates: tuple[str, ...]
    gold_bridge: str
    agent_a_prompt: str
    agent_a_probe_prompt: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExtractionExample:
        """Parse the stable synthetic schema, accepting two documented aliases."""

        example_id = _aliased_string(value, "example_id", "item_id")
        split = _required_string(value, "split")
        if split not in SPLITS:
            raise ArtifactError(f"unknown split {split!r} for {example_id!r}")

        raw_candidates = value.get("candidate_bridges", value.get("candidates"))
        if isinstance(raw_candidates, (str, bytes)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise ArtifactError(
                f"{example_id!r} must provide candidate_bridges as a sequence"
            )
        if not all(isinstance(candidate, str) for candidate in raw_candidates):
            raise ArtifactError(f"{example_id!r} candidates must all be strings")
        candidates = canonical_candidate_inventory(raw_candidates)
        gold_bridge = _required_string(value, "gold_bridge")
        if gold_bridge not in candidates:
            raise ArtifactError(
                f"gold_bridge {gold_bridge!r} is absent from {example_id!r} candidates"
            )

        return cls(
            example_id=example_id,
            split=split,
            candidates=candidates,
            gold_bridge=gold_bridge,
            agent_a_prompt=_required_string(value, "agent_a_prompt"),
            agent_a_probe_prompt=_required_string(value, "agent_a_probe_prompt"),
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable hash over every extraction-relevant input field."""

        return stable_fingerprint(
            {
                "example_id": self.example_id,
                "split": self.split,
                "candidates": list(self.candidates),
                "gold_bridge": self.gold_bridge,
                "agent_a_prompt": self.agent_a_prompt,
                "agent_a_probe_prompt": self.agent_a_probe_prompt,
            }
        )


def canonical_candidate_inventory(candidates: Iterable[str]) -> tuple[str, ...]:
    """Return a deterministic shared candidate inventory or fail closed."""

    values = tuple(candidates)
    if len(values) < 2:
        raise ArtifactError("at least two candidates are required")
    if not all(isinstance(candidate, str) for candidate in values):
        raise ArtifactError("candidates must all be strings")
    if any(not candidate or candidate.strip() != candidate for candidate in values):
        raise ArtifactError("candidates must be non-empty stripped strings")
    if len(values) != len(set(values)):
        raise ArtifactError("candidate strings must be unique")
    return tuple(sorted(values))


def validate_shared_inventory(
    examples: Sequence[ExtractionExample],
) -> tuple[str, ...]:
    """Require every example to use the exact same normalized inventory."""

    if not examples:
        raise ArtifactError("no extraction examples were provided")
    inventory = examples[0].candidates
    mismatched = [
        example.example_id for example in examples if example.candidates != inventory
    ]
    if mismatched:
        preview = ", ".join(mismatched[:3])
        raise ArtifactError(f"candidate inventory mismatch in: {preview}")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ArtifactError("example_id values must be unique across extraction inputs")
    return inventory


def build_text_only_prompt(visible_text: str, candidates: Sequence[str]) -> str:
    """Build the plain behavioral-classifier prompt ending exactly in Answer:."""

    if not isinstance(visible_text, str) or not visible_text.strip():
        raise ArtifactError("visible text must be non-empty")
    inventory = canonical_candidate_inventory(candidates)
    choices = "\n".join(f"- {candidate}" for candidate in inventory)
    return (
        f"{visible_text.strip()}\n\n"
        "Choose exactly one bridge concept from these candidates:\n"
        f"{choices}\n\n"
        "Return only the candidate text.\n"
        "Answer:"
    )


def artifact_path(root: str | Path, split: str, example_id: str) -> Path:
    """Return the deterministic per-example path while rejecting path traversal."""

    if split not in SPLITS:
        raise ArtifactError(f"unknown split: {split!r}")
    if not _SAFE_ITEM_ID.fullmatch(example_id):
        raise ArtifactError(f"unsafe example_id for artifact path: {example_id!r}")
    return Path(root) / split / f"{example_id}.pt"


def stable_fingerprint(value: object) -> str:
    """Hash a JSON-compatible object using canonical encoding."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactError("fingerprint input must be JSON-compatible") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def path_fingerprint(path: str | Path) -> str:
    """Hash one file or a directory tree deterministically."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise ArtifactError(f"cannot fingerprint missing path: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ArtifactError(f"cannot fingerprint empty directory: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_artifact(
    *,
    example: ExtractionExample,
    candidate_token_ids: Mapping[str, int],
    layer: int,
    residual: object,
    scores: Mapping[str, Mapping[str, float]],
    extraction_fingerprint: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate one compact artifact payload."""

    payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "example_id": example.example_id,
        "split": example.split,
        "candidates": list(example.candidates),
        "candidate_token_ids": dict(candidate_token_ids),
        "gold_bridge": example.gold_bridge,
        "gold_token_id": int(candidate_token_ids[example.gold_bridge]),
        "layer": layer,
        "residual": residual,
        "residual_dtype": "float16",
        "residual_device": "cpu",
        "scores": {
            method: dict(method_scores) for method, method_scores in scores.items()
        },
        "fingerprints": {
            "example": example.fingerprint,
            "candidate_inventory": stable_fingerprint(list(example.candidates)),
            "extraction": extraction_fingerprint,
        },
        "provenance": dict(provenance),
    }
    validate_artifact(payload)
    return payload


def build_extraction_manifest(
    *,
    examples: Sequence[ExtractionExample],
    candidate_token_ids: Mapping[str, int],
    layer: int,
    extraction_fingerprint: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build an immutable run-level contract for extraction and later scoring."""

    inventory = validate_shared_inventory(examples)
    counts = {split: 0 for split in SPLITS}
    for example in examples:
        counts[example.split] += 1
    manifest: dict[str, object] = {
        "schema_version": EXTRACTION_MANIFEST_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "extraction_sha256": extraction_fingerprint,
        "canonical_layer": layer,
        "candidates": list(inventory),
        "candidate_token_ids": dict(candidate_token_ids),
        "expected_counts": counts,
        "expected_total": len(examples),
        "provenance": dict(provenance),
    }
    validate_extraction_manifest(manifest)
    return manifest


def validate_extraction_manifest(value: Mapping[str, object]) -> None:
    """Validate the immutable top-level extraction contract."""

    required = {
        "schema_version",
        "artifact_schema",
        "extraction_sha256",
        "canonical_layer",
        "candidates",
        "candidate_token_ids",
        "expected_counts",
        "expected_total",
        "provenance",
    }
    missing = required - set(value)
    if missing:
        raise ArtifactError(
            "extraction manifest is missing fields: " + ", ".join(sorted(missing))
        )
    if value["schema_version"] != EXTRACTION_MANIFEST_SCHEMA:
        raise ArtifactError("unsupported extraction manifest schema")
    if value["artifact_schema"] != ARTIFACT_SCHEMA:
        raise ArtifactError("extraction manifest names an unsupported artifact schema")
    if not _is_sha256(value["extraction_sha256"]):
        raise ArtifactError("extraction manifest fingerprint must be SHA-256")
    layer = value["canonical_layer"]
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ArtifactError("extraction manifest layer must be non-negative")
    raw_candidates = value["candidates"]
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise ArtifactError("extraction manifest candidates must be a sequence")
    candidates = canonical_candidate_inventory(raw_candidates)  # type: ignore[arg-type]
    if tuple(raw_candidates) != candidates:
        raise ArtifactError("extraction manifest candidates are not canonical")
    token_ids = value["candidate_token_ids"]
    if not isinstance(token_ids, Mapping) or set(token_ids) != set(candidates):
        raise ArtifactError("extraction manifest token ids do not match candidates")
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in token_ids.values()
    ):
        raise ArtifactError("extraction manifest token ids must be non-negative")
    if len(set(token_ids.values())) != len(token_ids):
        raise ArtifactError("extraction manifest token ids must be unique")
    counts = value["expected_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(SPLITS):
        raise ArtifactError("extraction manifest counts must cover train/dev/test")
    if not all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in counts.values()
    ):
        raise ArtifactError("extraction manifest counts must be non-negative integers")
    if value["expected_total"] != sum(counts.values()):
        raise ArtifactError("extraction manifest total does not match split counts")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise ArtifactError("extraction manifest provenance must be a mapping")
    try:
        json.dumps(dict(provenance), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            "extraction manifest provenance must be JSON-compatible"
        ) from error


def ensure_extraction_manifest(
    root: str | Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Create the run manifest once or validate a compatible resumed run."""

    validate_extraction_manifest(manifest)
    path = Path(root) / EXTRACTION_MANIFEST_NAME
    if path.exists():
        existing = load_extraction_manifest(root)
        identity_fields = (
            "artifact_schema",
            "extraction_sha256",
            "canonical_layer",
            "candidates",
            "candidate_token_ids",
            "expected_counts",
            "expected_total",
        )
        if any(existing[field] != manifest[field] for field in identity_fields):
            raise ArtifactConflictError(
                f"existing extraction manifest is incompatible: {path}"
            )
        return existing
    write_json_atomic(path, dict(manifest))
    return dict(manifest)


def load_extraction_manifest(root: str | Path) -> dict[str, object]:
    """Load and validate the run-level extraction manifest."""

    path = Path(root) / EXTRACTION_MANIFEST_NAME
    if not path.is_file():
        raise ArtifactError(f"missing extraction manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"invalid extraction manifest JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ArtifactError("extraction manifest must be a JSON object")
    manifest = dict(value)
    validate_extraction_manifest(manifest)
    return manifest


def validate_artifact(value: Mapping[str, object]) -> None:
    """Validate all compact artifact invariants without importing Torch."""

    required = {
        "schema_version",
        "example_id",
        "split",
        "candidates",
        "candidate_token_ids",
        "gold_bridge",
        "gold_token_id",
        "layer",
        "residual",
        "residual_dtype",
        "residual_device",
        "scores",
        "fingerprints",
        "provenance",
    }
    missing = required - set(value)
    if missing:
        raise ArtifactError(f"artifact is missing fields: {', '.join(sorted(missing))}")
    if value["schema_version"] != ARTIFACT_SCHEMA:
        raise ArtifactError(f"unsupported artifact schema: {value['schema_version']!r}")

    example_id = value["example_id"]
    split = value["split"]
    if not isinstance(example_id, str) or not _SAFE_ITEM_ID.fullmatch(example_id):
        raise ArtifactError("artifact example_id is invalid")
    if split not in SPLITS:
        raise ArtifactError(f"artifact split is invalid: {split!r}")

    raw_candidates = value["candidates"]
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise ArtifactError("artifact candidates must be a sequence")
    candidates = canonical_candidate_inventory(raw_candidates)  # type: ignore[arg-type]
    if tuple(raw_candidates) != candidates:
        raise ArtifactError("artifact candidates must use canonical order")

    token_ids = value["candidate_token_ids"]
    if not isinstance(token_ids, Mapping) or set(token_ids) != set(candidates):
        raise ArtifactError("candidate_token_ids must match candidates exactly")
    normalized_ids: list[int] = []
    for candidate in candidates:
        token_id = token_ids[candidate]
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ArtifactError("candidate token ids must be non-negative integers")
        normalized_ids.append(token_id)
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ArtifactError("candidate token ids must be unique")

    gold = value["gold_bridge"]
    if gold not in candidates:
        raise ArtifactError("artifact gold_bridge is not a candidate")
    if value["gold_token_id"] != token_ids[gold]:
        raise ArtifactError("artifact gold_token_id does not match gold_bridge")
    layer = value["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ArtifactError("artifact layer must be a non-negative integer")
    if value["residual_dtype"] != "float16" or value["residual_device"] != "cpu":
        raise ArtifactError("artifact residual must be declared float16 on CPU")
    _validate_residual_if_introspectable(value["residual"])

    raw_scores = value["scores"]
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(
        STORED_SCORE_METHODS
    ):
        raise ArtifactError(
            "artifact scores must contain exactly: " + ", ".join(STORED_SCORE_METHODS)
        )
    for method in STORED_SCORE_METHODS:
        method_scores = raw_scores[method]
        if not isinstance(method_scores, Mapping) or set(method_scores) != set(
            candidates
        ):
            raise ArtifactError(f"{method} scores must match candidates exactly")
        for score in method_scores.values():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ArtifactError(f"{method} scores must be numeric")
            if not math.isfinite(float(score)):
                raise ArtifactError(f"{method} scores must be finite")

    fingerprints = value["fingerprints"]
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {
        "example",
        "candidate_inventory",
        "extraction",
    }:
        raise ArtifactError("artifact fingerprints are incomplete")
    if not all(_is_sha256(item) for item in fingerprints.values()):
        raise ArtifactError("artifact fingerprints must be SHA-256 hex digests")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise ArtifactError("artifact provenance must be a mapping")
    try:
        json.dumps(dict(provenance), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError("artifact provenance must be JSON-compatible") from error


SaveFunction: TypeAlias = Callable[[Mapping[str, object], Path], None]
LoadFunction: TypeAlias = Callable[[Path], Mapping[str, object]]


def save_artifact_atomic(
    path: str | Path,
    value: Mapping[str, object],
    *,
    run_root: str | Path,
    hard_limit_bytes: int,
    save_fn: SaveFunction | None = None,
    byte_budget: RunByteBudget | None = None,
) -> Path:
    """Atomically save one artifact without allowing a run-limit overshoot."""

    validate_artifact(value)
    if hard_limit_bytes <= 0:
        raise RunHardLimitError("run hard limit must be positive")
    output = Path(path)
    root = Path(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    if not output.resolve().is_relative_to(resolved_root):
        raise ArtifactError("artifact output must be inside run_root")
    if output.exists():
        raise ArtifactConflictError(f"refusing to overwrite artifact: {output}")
    writer = save_fn or _torch_save
    budget = byte_budget or RunByteBudget.inspect(root, hard_limit_bytes)
    if budget.root != resolved_root or budget.hard_limit_bytes != int(hard_limit_bytes):
        raise RunHardLimitError("byte budget does not match run_root and hard limit")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
        writer(value, temporary)
        artifact_bytes = temporary.stat().st_size
        budget.check_addition(artifact_bytes)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output)
        budget.commit(artifact_bytes)
        return output
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_artifact(
    path: str | Path,
    *,
    load_fn: LoadFunction | None = None,
) -> dict[str, object]:
    """Load one trusted local artifact onto CPU and validate its schema."""

    loader = load_fn or _torch_load
    value = loader(Path(path))
    if not isinstance(value, Mapping):
        raise ArtifactError(f"artifact is not a mapping: {path}")
    payload = dict(value)
    validate_artifact(payload)
    return payload


def resume_matches(
    artifact: Mapping[str, object],
    *,
    example: ExtractionExample,
    extraction_fingerprint: str,
) -> bool:
    """Return whether an existing artifact exactly matches this extraction."""

    validate_artifact(artifact)
    fingerprints = artifact["fingerprints"]
    assert isinstance(fingerprints, Mapping)
    return (
        artifact["example_id"] == example.example_id
        and artifact["split"] == example.split
        and fingerprints["example"] == example.fingerprint
        and fingerprints["extraction"] == extraction_fingerprint
    )


def discover_split_artifacts(root: str | Path, split: str) -> tuple[Path, ...]:
    """Return deterministic artifact paths for one required split."""

    if split not in SPLITS:
        raise ArtifactError(f"unknown split: {split!r}")
    folder = Path(root) / split
    paths = tuple(sorted(folder.glob("*.pt"))) if folder.exists() else ()
    if not paths:
        raise ArtifactError(f"no {split} artifacts found below {folder}")
    return paths


def residual_row(value: object) -> list[float]:
    """Convert a stored CPU residual tensor-like object to one sklearn row."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    float_method = getattr(value, "float", None)
    if callable(float_method):
        value = float_method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError("residual must be a one-dimensional tensor-like value")
    if value and isinstance(value[0], (list, tuple)):
        raise ArtifactError("stored residual must not be batched")
    row = [float(item) for item in value]
    if not row or not all(math.isfinite(item) for item in row):
        raise ArtifactError("stored residual must contain finite values")
    return row


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[Mapping[str, object]],
) -> Path:
    """Atomically write compact JSONL records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        return output
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def pickle_save(value: Mapping[str, object], path: Path) -> None:
    """Dependency-free serializer intended for tests and tiny local fixtures."""

    with path.open("wb") as handle:
        pickle.dump(dict(value), handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path: Path) -> Mapping[str, object]:
    """Dependency-free loader paired with :func:`pickle_save`."""

    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise ArtifactError("pickle fixture is not a mapping")
    return value


def _torch_save(value: Mapping[str, object], path: Path) -> None:
    import torch

    torch.save(dict(value), path)


def _torch_load(path: Path) -> Mapping[str, object]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_residual_if_introspectable(value: object) -> None:
    ndim = getattr(value, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ArtifactError("artifact residual must be one-dimensional")
    dtype = getattr(value, "dtype", None)
    if dtype is not None and not str(dtype).endswith("float16"):
        raise ArtifactError(f"artifact residual dtype is not float16: {dtype}")
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise ArtifactError(f"artifact residual device is not CPU: {device}")


def _required_string(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ArtifactError(f"{field} must be a non-empty string")
    return item


def _aliased_string(
    value: Mapping[str, object],
    primary: str,
    alias: str,
) -> str:
    first = value.get(primary)
    second = value.get(alias)
    if first is not None and second is not None and first != second:
        raise ArtifactError(f"conflicting {primary} and {alias} values")
    item = first if first is not None else second
    if not isinstance(item, str) or not item.strip():
        raise ArtifactError(f"{primary} or {alias} must be a non-empty string")
    return item


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
