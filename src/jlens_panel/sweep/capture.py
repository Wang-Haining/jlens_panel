"""Bounded multi-position capture with candidate-only lens scoring."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from jlens_panel.calibration import NullCalibration, NullScoreAccumulator
from jlens_panel.corpus import SampledPrompt
from jlens_panel.modeling import render_chat
from jlens_panel.provenance import sha256_file
from jlens_panel.readouts.artifacts import RunByteBudget, RunHardLimitError
from jlens_panel.sweep.positions import (
    ALL_POSITION_NAMES,
    DECODE_STEPS,
    resolve_static_positions,
)

CAPTURE_SCHEMA = "jlens-panel-sweep-capture-v1"
CAPTURE_SPLITS = ("train", "dev")
CAPTURE_METHODS = ("jlens", "logit_lens")
NULL_ACCUMULATOR_STATE_SCHEMA = "jlens-panel-null-accumulators-v1"
_SAFE_EXAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


class SweepCaptureError(ValueError):
    """Raised when capture inputs or artifacts violate the sprint contract."""


class SweepCaptureConflictError(SweepCaptureError):
    """Raised when a capture would overwrite incompatible state."""


SaveFunction: TypeAlias = Callable[[Mapping[str, object], Path], None]
LoadFunction: TypeAlias = Callable[[bytes], Mapping[str, object]]
DiskCheck: TypeAlias = Callable[[], None]
CheckpointWriter: TypeAlias = Callable[[Mapping[str, object]], None]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _flat_ids(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SweepCaptureError(f"{name} must be a flat sequence")
    ids = tuple(value)
    if not ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in ids
    ):
        raise SweepCaptureError(f"{name} must contain non-negative integers")
    return ids


def _shape(value: object, *, name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise SweepCaptureError(f"{name} must be a tensor-like value")
    try:
        shape = tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as error:
        raise SweepCaptureError(f"{name} has an invalid shape") from error
    if any(dimension < 1 for dimension in shape):
        raise SweepCaptureError(f"{name} dimensions must be positive")
    return shape


def _tensor_is_finite(value: object) -> bool:
    isfinite = getattr(value, "isfinite", None)
    if not callable(isfinite):
        return True
    finite = isfinite()
    all_method = getattr(finite, "all", None)
    if not callable(all_method):
        return True
    scalar = all_method()
    item = getattr(scalar, "item", None)
    return bool(item() if callable(item) else scalar)


def _validate_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype_name: str,
) -> None:
    if _shape(value, name=name) != shape:
        raise SweepCaptureError(f"{name} has the wrong shape")
    dtype = getattr(value, "dtype", None)
    if dtype is None or str(dtype) not in {dtype_name, f"torch.{dtype_name}"}:
        raise SweepCaptureError(f"{name} must have dtype {dtype_name}")
    device = getattr(value, "device", None)
    if device is None or str(device) != "cpu":
        raise SweepCaptureError(f"{name} must reside on CPU")
    if not _tensor_is_finite(value):
        raise SweepCaptureError(f"{name} contains non-finite values")


def capture_artifact_path(
    root: str | Path,
    split: str,
    example_id: str,
) -> Path:
    """Return a train/dev-only artifact path while rejecting traversal."""

    if split not in CAPTURE_SPLITS:
        raise SweepCaptureError(f"capture split must be train or dev, got {split!r}")
    if not _SAFE_EXAMPLE_ID.fullmatch(example_id):
        raise SweepCaptureError(f"unsafe example ID: {example_id!r}")
    return Path(root) / split / f"{example_id}.pt"


def build_capture_artifact(
    *,
    example_id: str,
    split: str,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    gold_bridge: str,
    layers: Sequence[int],
    hidden_size: int,
    residuals: Mapping[str, object],
    scores: Mapping[str, Mapping[str, object]],
    example_fingerprint: str,
    dataset_fingerprint: str,
    capture_fingerprint: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate one compact train/dev sweep artifact."""

    inventory = tuple(sorted(candidates))
    if (
        len(inventory) < 2
        or len(inventory) != len(set(inventory))
        or any(
            not isinstance(candidate, str)
            or not candidate
            or candidate.strip() != candidate
            for candidate in inventory
        )
    ):
        raise SweepCaptureError("capture candidates must be unique strings")
    if set(candidate_token_ids) != set(inventory):
        raise SweepCaptureError("candidate token IDs do not match capture candidates")
    if gold_bridge not in inventory:
        raise SweepCaptureError("capture gold bridge is not a candidate")
    payload: dict[str, object] = {
        "schema_version": CAPTURE_SCHEMA,
        "example_id": example_id,
        "split": split,
        "candidates": list(inventory),
        "candidate_token_ids": {
            candidate: candidate_token_ids[candidate] for candidate in inventory
        },
        "gold_bridge": gold_bridge,
        "gold_token_id": candidate_token_ids[gold_bridge],
        "positions": list(ALL_POSITION_NAMES),
        "layers": list(layers),
        "hidden_size": hidden_size,
        "residual_dtype": "float16",
        "residual_device": "cpu",
        "residuals": dict(residuals),
        "scores": {
            method: dict(method_scores) for method, method_scores in scores.items()
        },
        "fingerprints": {
            "example": example_fingerprint,
            "dataset": dataset_fingerprint,
            "capture": capture_fingerprint,
        },
        "provenance": dict(provenance),
    }
    validate_capture_artifact(payload)
    return payload


def validate_capture_artifact(value: Mapping[str, object]) -> None:
    """Fail closed on schema, support, tensor, or provenance drift."""

    required = {
        "schema_version",
        "example_id",
        "split",
        "candidates",
        "candidate_token_ids",
        "gold_bridge",
        "gold_token_id",
        "positions",
        "layers",
        "hidden_size",
        "residual_dtype",
        "residual_device",
        "residuals",
        "scores",
        "fingerprints",
        "provenance",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SweepCaptureError("capture artifact fields do not match the schema")
    if value["schema_version"] != CAPTURE_SCHEMA:
        raise SweepCaptureError("unsupported capture artifact schema")
    example_id = value["example_id"]
    split = value["split"]
    if not isinstance(example_id, str):
        raise SweepCaptureError("capture example ID must be a string")
    capture_artifact_path(".", str(split), example_id)

    candidates = value["candidates"]
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or len(candidates) < 2
        or any(
            not isinstance(candidate, str)
            or not candidate
            or candidate.strip() != candidate
            for candidate in candidates
        )
        or list(candidates) != sorted(set(candidates))
    ):
        raise SweepCaptureError("capture candidates must be unique and sorted")
    inventory = tuple(candidates)
    token_ids = value["candidate_token_ids"]
    if not isinstance(token_ids, Mapping) or tuple(token_ids) != inventory:
        raise SweepCaptureError("capture candidate token map changed")
    normalized_ids = tuple(token_ids[candidate] for candidate in inventory)
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in normalized_ids
    ) or len(normalized_ids) != len(set(normalized_ids)):
        raise SweepCaptureError("capture candidate token IDs must be unique integers")
    gold = value["gold_bridge"]
    if gold not in inventory or value["gold_token_id"] != token_ids[gold]:
        raise SweepCaptureError("capture gold label or token ID changed")

    if value["positions"] != list(ALL_POSITION_NAMES):
        raise SweepCaptureError("capture position inventory changed")
    layers = value["layers"]
    if (
        isinstance(layers, (str, bytes))
        or not isinstance(layers, Sequence)
        or not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers
        )
        or list(layers) != sorted(set(layers))
    ):
        raise SweepCaptureError("capture layers must be unique and sorted")
    hidden_size = value["hidden_size"]
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 1
    ):
        raise SweepCaptureError("capture hidden size must be positive")
    if value["residual_dtype"] != "float16" or value["residual_device"] != "cpu":
        raise SweepCaptureError("capture residual declaration must be float16 CPU")

    residuals = value["residuals"]
    if not isinstance(residuals, Mapping) or tuple(residuals) != ALL_POSITION_NAMES:
        raise SweepCaptureError("capture residual position inventory changed")
    for position in ALL_POSITION_NAMES:
        _validate_tensor(
            residuals[position],
            name=f"{position} residuals",
            shape=(len(layers), hidden_size),
            dtype_name="float16",
        )

    scores = value["scores"]
    if not isinstance(scores, Mapping) or tuple(scores) != CAPTURE_METHODS:
        raise SweepCaptureError("capture score methods changed")
    for method in CAPTURE_METHODS:
        method_scores = scores[method]
        if not isinstance(method_scores, Mapping) or tuple(method_scores) != (
            ALL_POSITION_NAMES
        ):
            raise SweepCaptureError(f"{method} score positions changed")
        for position in ALL_POSITION_NAMES:
            _validate_tensor(
                method_scores[position],
                name=f"{method}/{position} scores",
                shape=(len(layers), len(inventory)),
                dtype_name="float32",
            )

    fingerprints = value["fingerprints"]
    if (
        not isinstance(fingerprints, Mapping)
        or set(fingerprints)
        != {
            "example",
            "dataset",
            "capture",
        }
        or not all(_is_sha256(digest) for digest in fingerprints.values())
    ):
        raise SweepCaptureError("capture fingerprints are incomplete")
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise SweepCaptureError("capture provenance must be a non-empty mapping")
    try:
        json.dumps(dict(provenance), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SweepCaptureError("capture provenance must be JSON-compatible") from error


def _torch_save(value: Mapping[str, object], path: Path) -> None:
    import torch

    torch.save(dict(value), path)


def _torch_load(raw_bytes: bytes) -> Mapping[str, object]:
    import torch

    return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)


def save_capture_artifact_atomic(
    path: str | Path,
    value: Mapping[str, object],
    *,
    run_root: str | Path,
    byte_budget: RunByteBudget,
    save_fn: SaveFunction | None = None,
) -> tuple[Path, str]:
    """Publish one complete artifact without overwrite or budget overshoot."""

    validate_capture_artifact(value)
    output = Path(path)
    root = Path(run_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.resolve().is_relative_to(root):
        raise SweepCaptureError("capture output must be inside the run root")
    if byte_budget.root != root:
        raise RunHardLimitError("capture byte budget does not match the run root")
    writer = save_fn or _torch_save
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        writer(value, temporary)
        artifact_bytes = temporary.stat().st_size
        byte_budget.check_addition(artifact_bytes)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise SweepCaptureConflictError(
                f"refusing to overwrite capture artifact: {output}"
            ) from error
        byte_budget.commit(artifact_bytes)
        return output, sha256_file(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_capture_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    load_fn: LoadFunction | None = None,
) -> dict[str, object]:
    """Hash and load the same immutable bytes, then validate the artifact."""

    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
    except OSError as error:
        raise SweepCaptureError(f"cannot read capture artifact: {source}") from error
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise SweepCaptureError("expected capture SHA-256 is malformed")
        if digest != expected_sha256:
            raise SweepCaptureError("capture artifact SHA-256 mismatch")
    loader = load_fn or _torch_load
    try:
        value = loader(raw_bytes)
    except Exception as error:
        raise SweepCaptureError(f"cannot decode capture artifact: {source}") from error
    if not isinstance(value, Mapping):
        raise SweepCaptureError("capture artifact is not a mapping")
    payload = dict(value)
    validate_capture_artifact(payload)
    return payload


def candidate_only_logits(
    lens_model: object,
    residual: object,
    candidate_token_ids: Sequence[int],
) -> object:
    """Apply final norm and only the requested unembedding rows."""

    import torch
    import torch.nn.functional as functional

    ids = _flat_ids(candidate_token_ids, name="candidate token IDs")
    try:
        head = lens_model._lm_head  # type: ignore[attr-defined]
        final_norm = lens_model._final_norm  # type: ignore[attr-defined]
        softcap = lens_model._logit_softcap  # type: ignore[attr-defined]
    except AttributeError as error:
        raise SweepCaptureError(
            "unsupported J-lens model unembedding adapter"
        ) from error
    with torch.inference_mode():
        id_tensor = torch.as_tensor(ids, dtype=torch.long, device=head.weight.device)
        normalized = final_norm(
            residual.to(dtype=head.weight.dtype, device=head.weight.device)
        )
        weight = head.weight.index_select(0, id_tensor)
        bias = None if head.bias is None else head.bias.index_select(0, id_tensor)
        logits = functional.linear(normalized, weight, bias)
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
    if logits.shape[-1] != len(ids):
        raise SweepCaptureError("candidate-only unembedding returned the wrong shape")
    return logits


def _exact_input_ids(
    tokenizer: object, text: str, *, max_seq_len: int
) -> tuple[int, ...]:
    try:
        encoded = tokenizer(  # type: ignore[operator]
            text,
            add_special_tokens=False,
            truncation=False,
        )
    except (TypeError, ValueError) as error:
        raise SweepCaptureError("tokenizer cannot encode exact input text") from error
    if not isinstance(encoded, Mapping):
        raise SweepCaptureError("tokenizer output must be a mapping")
    input_ids = _flat_ids(encoded.get("input_ids"), name="input IDs")
    if len(input_ids) > max_seq_len:
        raise SweepCaptureError(
            f"input has {len(input_ids)} tokens above max_seq_len={max_seq_len}"
        )
    return input_ids


def _forward_activations(
    bundle: object,
    input_ids: Sequence[int],
    layers: Sequence[int],
) -> dict[int, object]:
    import torch
    from jlens.hooks import ActivationRecorder

    model = bundle.lens_model  # type: ignore[attr-defined]
    ids = torch.as_tensor(
        [list(input_ids)],
        dtype=torch.long,
        device=model.input_device,
    )
    requested = tuple(sorted(set(layers)))
    with (
        torch.inference_mode(),
        ActivationRecorder(
            model.layers,
            at=requested,
        ) as recorder,
    ):
        model.forward(ids)
    activations: dict[int, object] = {}
    for layer in requested:
        activation = recorder.activations[layer]
        if tuple(activation.shape[:2]) != (1, len(input_ids)):
            raise SweepCaptureError(
                "recorded activation shape disagrees with input IDs"
            )
        activations[layer] = activation
    return activations


def _pin_jacobians(
    bundle: object,
    activations: Mapping[int, object],
    layers: Sequence[int],
) -> None:
    import torch

    lens = bundle.lens  # type: ignore[attr-defined]
    for layer in layers:
        destination = activations[layer].device
        jacobian = lens.jacobians[layer]
        if jacobian.device != destination or jacobian.dtype != torch.float32:
            moved = jacobian.to(device=destination, dtype=torch.float32)
            try:
                lens.jacobians[layer] = moved
            except (TypeError, RuntimeError) as error:
                raise SweepCaptureError(
                    f"cannot pin J-lens Jacobian for layer {layer}"
                ) from error


def _greedy_token(
    bundle: object,
    final_residual: object,
    *,
    forbidden_token_ids: Sequence[int] = (),
) -> int:
    """Select one token from ephemeral full logits without persisting them."""

    import torch

    with torch.inference_mode():
        logits = bundle.lens_model.unembed(final_residual)  # type: ignore[attr-defined]
        if getattr(logits, "ndim", None) != 1:
            raise SweepCaptureError("greedy next-token logits must be one-dimensional")
        for token_id in forbidden_token_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < len(logits)
            ):
                raise SweepCaptureError("forbidden greedy token ID is out of range")
            logits[token_id] = float("-inf")
        return int(logits.argmax(dim=-1).item())


def _decode_residuals(
    bundle: object,
    *,
    initial_ids: Sequence[int],
    initial_activations: Mapping[int, object],
    layers: Sequence[int],
    final_layer: int,
    max_seq_len: int,
) -> dict[str, dict[int, object]]:
    """Capture states located on generated tokens 1, 2, 4, and 8."""

    ids = list(initial_ids)
    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)  # type: ignore[attr-defined]
    forbidden_eos = () if eos_token_id is None else (int(eos_token_id),)
    token_id = _greedy_token(
        bundle,
        initial_activations[final_layer][0, -1],
        forbidden_token_ids=forbidden_eos,
    )
    captured: dict[str, dict[int, object]] = {}
    for generated_step in range(1, max(DECODE_STEPS) + 1):
        if len(ids) >= max_seq_len:
            raise SweepCaptureError("decode step would exceed max_seq_len")
        ids.append(token_id)
        activations = _forward_activations(
            bundle,
            ids,
            (*layers, final_layer),
        )
        if generated_step in DECODE_STEPS:
            captured[f"decode_{generated_step}"] = {
                layer: activations[layer][0, -1].float() for layer in layers
            }
        if (
            generated_step < max(DECODE_STEPS)
            and eos_token_id is not None
            and token_id == int(eos_token_id)
        ):
            raise SweepCaptureError("EOS masking failed before decode_8")
        if generated_step < max(DECODE_STEPS):
            next_generated_step = generated_step + 1
            token_id = _greedy_token(
                bundle,
                activations[final_layer][0, -1],
                forbidden_token_ids=(
                    forbidden_eos if next_generated_step < max(DECODE_STEPS) else ()
                ),
            )
    if tuple(captured) != tuple(f"decode_{step}" for step in DECODE_STEPS):
        raise SweepCaptureError("decode capture did not produce all frozen steps")
    return captured


def _score_residuals(
    bundle: object,
    residuals: Mapping[str, Mapping[int, object]],
    *,
    layers: Sequence[int],
    candidate_token_ids: Mapping[str, int],
) -> dict[str, dict[str, object]]:
    import torch

    candidates = tuple(sorted(candidate_token_ids))
    ids = tuple(candidate_token_ids[candidate] for candidate in candidates)
    output: dict[str, dict[str, list[object]]] = {
        method: {position: [] for position in ALL_POSITION_NAMES}
        for method in CAPTURE_METHODS
    }
    with torch.inference_mode():
        for layer in layers:
            matrix = torch.stack(
                [residuals[position][layer].float() for position in ALL_POSITION_NAMES],
                dim=0,
            )
            transported = bundle.lens.transport(matrix, layer)  # type: ignore[attr-defined]
            method_logits = {
                "jlens": candidate_only_logits(bundle.lens_model, transported, ids),  # type: ignore[attr-defined]
                "logit_lens": candidate_only_logits(bundle.lens_model, matrix, ids),  # type: ignore[attr-defined]
            }
            for method, logits in method_logits.items():
                logits_cpu = logits.detach().float().cpu()
                for position_index, position in enumerate(ALL_POSITION_NAMES):
                    output[method][position].append(logits_cpu[position_index])
    return {
        method: {
            position: torch.stack(rows, dim=0)
            for position, rows in method_positions.items()
        }
        for method, method_positions in output.items()
    }


def capture_task_example(
    bundle: object,
    example: object,
    *,
    candidate_token_ids: Mapping[str, int],
    max_seq_len: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]], tuple[int, ...]]:
    """Capture all eight positions and every fitted source layer for one item."""

    split = getattr(example, "split", None)
    if split not in CAPTURE_SPLITS:
        raise SweepCaptureError("task capture accepts train/dev examples only")
    import torch

    if bundle.lens is None:  # type: ignore[attr-defined]
        raise SweepCaptureError("task capture requires a fitted lens")
    layers = tuple(sorted(set(int(layer) for layer in bundle.lens.source_layers)))  # type: ignore[attr-defined]
    if not layers:
        raise SweepCaptureError("fitted lens has no source layers")
    final_layer = int(bundle.lens_model.n_layers) - 1  # type: ignore[attr-defined]
    rendered = render_chat(
        bundle.tokenizer,  # type: ignore[attr-defined]
        [{"role": "user", "content": example.agent_a_prompt}],
    )
    resolved = resolve_static_positions(
        bundle.tokenizer,  # type: ignore[attr-defined]
        example,
        rendered,
        max_seq_len=max_seq_len,
    )
    initial = _forward_activations(
        bundle,
        resolved.input_ids,
        (*layers, final_layer),
    )
    _pin_jacobians(bundle, initial, layers)
    residuals: dict[str, dict[int, object]] = {}
    for selection in resolved.selections:
        per_layer: dict[int, object] = {}
        for layer in layers:
            activation = initial[layer][0]
            if selection.reduction == "mean":
                indices = torch.as_tensor(
                    selection.token_indices,
                    dtype=torch.long,
                    device=activation.device,
                )
                residual = activation.index_select(0, indices).float().mean(dim=0)
            else:
                residual = activation[selection.index].float()
            per_layer[layer] = residual
        residuals[selection.name] = per_layer
    residuals.update(
        _decode_residuals(
            bundle,
            initial_ids=resolved.input_ids,
            initial_activations=initial,
            layers=layers,
            final_layer=final_layer,
            max_seq_len=max_seq_len,
        )
    )
    if tuple(residuals) != ALL_POSITION_NAMES:
        raise SweepCaptureError("task residual position inventory changed")
    scores = _score_residuals(
        bundle,
        residuals,
        layers=layers,
        candidate_token_ids=candidate_token_ids,
    )
    compact_residuals = {
        position: torch.stack(
            [residuals[position][layer] for layer in layers],
            dim=0,
        )
        .to(device="cpu", dtype=torch.float16)
        .contiguous()
        for position in ALL_POSITION_NAMES
    }
    return compact_residuals, scores, layers


def _capture_null_prompt_residuals(
    bundle: object,
    prompt: str,
    *,
    layers: Sequence[int],
    max_seq_len: int,
) -> dict[str, dict[int, object]]:
    final_layer = int(bundle.lens_model.n_layers) - 1  # type: ignore[attr-defined]
    raw_ids = _exact_input_ids(bundle.tokenizer, prompt, max_seq_len=max_seq_len)  # type: ignore[attr-defined]
    if len(raw_ids) < 8:
        raise SweepCaptureError("null prompt has fewer than eight raw-text tokens")
    raw = _forward_activations(bundle, raw_ids, layers)
    _pin_jacobians(bundle, raw, layers)
    residuals: dict[str, dict[int, object]] = {
        "content_last": {layer: raw[layer][0, -1].float() for layer in layers},
        "clue_last": {layer: raw[layer][0, -1].float() for layer in layers},
        "meanpool_content8": {
            layer: raw[layer][0, -8:].float().mean(dim=0) for layer in layers
        },
    }

    rendered = render_chat(
        bundle.tokenizer,  # type: ignore[attr-defined]
        [{"role": "user", "content": prompt}],
    )
    chat_ids = _exact_input_ids(
        bundle.tokenizer,  # type: ignore[attr-defined]
        rendered,
        max_seq_len=max_seq_len,
    )
    chat = _forward_activations(bundle, chat_ids, (*layers, final_layer))
    residuals["template_tail"] = {layer: chat[layer][0, -1].float() for layer in layers}
    residuals.update(
        _decode_residuals(
            bundle,
            initial_ids=chat_ids,
            initial_activations=chat,
            layers=layers,
            final_layer=final_layer,
            max_seq_len=max_seq_len,
        )
    )
    return {position: residuals[position] for position in ALL_POSITION_NAMES}


def serialize_null_accumulators(
    accumulators: Mapping[tuple[str, str, int], NullScoreAccumulator],
    *,
    layers: Sequence[int],
    completed_prompts: int,
) -> dict[str, object]:
    """Serialize every cell's sufficient statistics for atomic checkpointing."""

    normalized_layers = tuple(sorted(set(layers)))
    expected_keys = {
        (method, position, layer)
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in normalized_layers
    }
    if set(accumulators) != expected_keys:
        raise SweepCaptureError("null accumulator cell inventory changed")
    if (
        isinstance(completed_prompts, bool)
        or not isinstance(completed_prompts, int)
        or completed_prompts < 0
        or any(
            accumulator.count != completed_prompts
            for accumulator in accumulators.values()
        )
    ):
        raise SweepCaptureError("null accumulator completion count changed")
    return {
        "schema_version": NULL_ACCUMULATOR_STATE_SCHEMA,
        "completed_prompts": completed_prompts,
        "cells": [
            {
                "method": method,
                "position": position,
                "layer": layer,
                "accumulator": accumulators[(method, position, layer)].to_state(),
            }
            for method in CAPTURE_METHODS
            for position in ALL_POSITION_NAMES
            for layer in normalized_layers
        ],
    }


def restore_null_accumulators(
    value: Mapping[str, object],
    *,
    candidates: Sequence[str],
    layers: Sequence[int],
) -> tuple[int, dict[tuple[str, str, int], NullScoreAccumulator]]:
    """Restore a complete, compatible null-accumulator checkpoint."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "completed_prompts",
        "cells",
    }:
        raise SweepCaptureError("null checkpoint state fields changed")
    if value["schema_version"] != NULL_ACCUMULATOR_STATE_SCHEMA:
        raise SweepCaptureError("null checkpoint state schema changed")
    completed = value["completed_prompts"]
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise SweepCaptureError("null checkpoint completion count is invalid")
    raw_cells = value["cells"]
    if isinstance(raw_cells, (str, bytes)) or not isinstance(raw_cells, Sequence):
        raise SweepCaptureError("null checkpoint cells must be a sequence")
    normalized_candidates = tuple(sorted(candidates))
    normalized_layers = tuple(sorted(set(layers)))
    restored: dict[tuple[str, str, int], NullScoreAccumulator] = {}
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping) or set(raw_cell) != {
            "method",
            "position",
            "layer",
            "accumulator",
        }:
            raise SweepCaptureError("null checkpoint cell fields changed")
        key = (raw_cell["method"], raw_cell["position"], raw_cell["layer"])
        if (
            key[0] not in CAPTURE_METHODS
            or key[1] not in ALL_POSITION_NAMES
            or isinstance(key[2], bool)
            or not isinstance(key[2], int)
            or key[2] not in normalized_layers
        ):
            raise SweepCaptureError("null checkpoint cell identity is invalid")
        typed_key = (str(key[0]), str(key[1]), int(key[2]))
        if typed_key in restored:
            raise SweepCaptureError("null checkpoint contains a duplicate cell")
        try:
            accumulator = NullScoreAccumulator.from_state(
                raw_cell["accumulator"]  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            raise SweepCaptureError("null checkpoint accumulator is invalid") from error
        if accumulator.candidates != normalized_candidates or accumulator.count != (
            completed
        ):
            raise SweepCaptureError("null checkpoint accumulator identity changed")
        restored[typed_key] = accumulator
    expected_keys = {
        (method, position, layer)
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in normalized_layers
    }
    if set(restored) != expected_keys:
        raise SweepCaptureError("null checkpoint cell inventory is incomplete")
    return completed, restored


def capture_null_calibrations(
    bundle: object,
    sampled_prompts: Sequence[SampledPrompt],
    *,
    candidate_token_ids: Mapping[str, int],
    max_seq_len: int,
    provenance: Mapping[str, object],
    disk_check_every: int,
    disk_check: DiskCheck,
    resume_state: Mapping[str, object] | None = None,
    checkpoint_every: int | None = None,
    checkpoint_writer: CheckpointWriter | None = None,
) -> tuple[NullCalibration, ...]:
    """Capture candidate-only null moments without storing prompts or trajectories."""

    if bundle.lens is None:  # type: ignore[attr-defined]
        raise SweepCaptureError("null capture requires a fitted lens")
    if len(sampled_prompts) != 200:
        raise SweepCaptureError("sprint null capture requires exactly 200 prompts")
    if tuple(prompt.sample_index for prompt in sampled_prompts) != tuple(range(200)):
        raise SweepCaptureError("null prompt sample indices changed")
    corpus_indices = tuple(prompt.corpus_index for prompt in sampled_prompts)
    if len(corpus_indices) != len(set(corpus_indices)):
        raise SweepCaptureError("null prompts must be sampled without replacement")
    if disk_check_every < 1:
        raise SweepCaptureError("disk check interval must be positive")
    layers = tuple(sorted(set(int(layer) for layer in bundle.lens.source_layers)))  # type: ignore[attr-defined]
    candidates = tuple(sorted(candidate_token_ids))
    if (checkpoint_every is None) != (checkpoint_writer is None):
        raise SweepCaptureError(
            "checkpoint interval and writer must be configured together"
        )
    if checkpoint_every is not None and checkpoint_every < 1:
        raise SweepCaptureError("checkpoint interval must be positive")
    if resume_state is None:
        completed_prompts = 0
        accumulators = {
            (method, position, layer): NullScoreAccumulator(candidates)
            for method in CAPTURE_METHODS
            for position in ALL_POSITION_NAMES
            for layer in layers
        }
    else:
        completed_prompts, accumulators = restore_null_accumulators(
            resume_state,
            candidates=candidates,
            layers=layers,
        )
        if completed_prompts > len(sampled_prompts):
            raise SweepCaptureError("null checkpoint exceeds the prompt sample")
    disk_check()
    for prompt_index in range(completed_prompts, len(sampled_prompts)):
        sampled = sampled_prompts[prompt_index]
        if prompt_index and prompt_index % disk_check_every == 0:
            disk_check()
        residuals = _capture_null_prompt_residuals(
            bundle,
            sampled.text,
            layers=layers,
            max_seq_len=max_seq_len,
        )
        scores = _score_residuals(
            bundle,
            residuals,
            layers=layers,
            candidate_token_ids=candidate_token_ids,
        )
        for method in CAPTURE_METHODS:
            for position in ALL_POSITION_NAMES:
                for layer_index, layer in enumerate(layers):
                    row = scores[method][position][layer_index]
                    accumulators[(method, position, layer)].update(
                        {
                            candidate: float(row[candidate_index].item())
                            for candidate_index, candidate in enumerate(candidates)
                        }
                    )
        completed_prompts = prompt_index + 1
        if (
            checkpoint_writer is not None
            and checkpoint_every is not None
            and (
                completed_prompts % checkpoint_every == 0
                or completed_prompts == len(sampled_prompts)
            )
        ):
            checkpoint_writer(
                serialize_null_accumulators(
                    accumulators,
                    layers=layers,
                    completed_prompts=completed_prompts,
                )
            )
    disk_check()
    return tuple(
        accumulators[(method, position, layer)].build(
            method=method,
            position_type=position,
            layer=layer,
            provenance=provenance,
        )
        for method in CAPTURE_METHODS
        for position in ALL_POSITION_NAMES
        for layer in layers
    )
