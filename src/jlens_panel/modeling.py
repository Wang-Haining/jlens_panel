"""Lazy GPU adapters for fitting, reading, and generating with J-lens models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ModelBundle:
    """Loaded HuggingFace model, tokenizer, lens adapter, and optional lens."""

    hf_model: Any
    tokenizer: Any
    lens_model: Any
    lens: Any | None = None


@dataclass(frozen=True)
class PreSpeechSnapshot:
    """One deterministic pre-speech state and its unsliced readouts."""

    layer: int
    residual: Any
    jlens_logits: Any
    logit_lens_logits: Any
    next_token_logits: Any
    input_ids: Any


def _torch_dtype(torch: Any, name: str) -> Any:
    aliases = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
    try:
        return getattr(torch, aliases[name])
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_model_bundle(
    *,
    model_name: str,
    revision: str,
    dtype: str,
    device_map: str,
    lens_path: str | None = None,
) -> ModelBundle:
    """Load GPU dependencies only when an execution script requests them."""

    import jlens
    import torch
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=_torch_dtype(torch, dtype),
        device_map=device_map,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
    )
    resolved_revision = getattr(hf_model.config, "_commit_hash", None)
    if resolved_revision is not None and resolved_revision != revision:
        raise RuntimeError(
            "loaded model revision does not match the pinned revision: "
            f"{resolved_revision} != {revision}"
        )
    lens_model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(lens_path) if lens_path else None
    return ModelBundle(
        hf_model=hf_model,
        tokenizer=tokenizer,
        lens_model=lens_model,
        lens=lens,
    )


def render_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Render messages with the pinned tokenizer's native chat template."""

    if not messages:
        raise ValueError("messages cannot be empty")
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def resolve_candidate_token_ids(
    tokenizer: Any,
    candidates: Iterable[str],
) -> dict[str, int]:
    """Resolve candidates to unique single token IDs or fail closed.

    A leading-space variant is preferred because concepts are read in ordinary
    English continuations. The unprefixed form is accepted as a fallback.
    """

    resolved: dict[str, int] = {}
    seen_ids: set[int] = set()
    for candidate in candidates:
        if not candidate or candidate.strip() != candidate:
            raise ValueError(f"Candidate must be non-empty and stripped: {candidate!r}")
        variants = (f" {candidate}", candidate)
        token_id: int | None = None
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_id = int(ids[0])
                break
        if token_id is None:
            raise ValueError(f"Candidate is not a single token: {candidate!r}")
        if token_id in seen_ids:
            raise ValueError(f"Candidates map to a duplicate token ID: {candidate!r}")
        resolved[candidate] = token_id
        seen_ids.add(token_id)
    if len(resolved) < 2:
        raise ValueError("At least two candidate tokens are required")
    return resolved


def canonical_layer(source_layers: Sequence[int], strategy: str) -> int:
    """Choose a preregistered canonical layer from fitted source layers."""

    layers = sorted(set(int(layer) for layer in source_layers))
    if not layers:
        raise ValueError("source_layers cannot be empty")
    if strategy != "middle_of_middle_third":
        raise ValueError(f"Unknown canonical layer strategy: {strategy}")
    middle_third = layers[len(layers) // 3 : 2 * len(layers) // 3]
    if not middle_third:
        middle_third = layers
    return middle_third[len(middle_third) // 2]


def capture_pre_speech(
    bundle: ModelBundle,
    prompt: str,
    *,
    layer: int,
    max_seq_len: int = 512,
) -> PreSpeechSnapshot:
    """Run one forward pass and compute all activation-based readouts."""

    if bundle.lens is None:
        raise ValueError("capture_pre_speech requires a fitted lens")
    if layer not in bundle.lens.source_layers:
        raise ValueError(f"Layer {layer} was not fitted by the lens")

    from jlens.hooks import ActivationRecorder

    model = bundle.lens_model
    final_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=[layer, final_layer]) as recorder:
        model.forward(input_ids)

    residual = recorder.activations[layer][0, -1].detach().float()
    final_residual = recorder.activations[final_layer][0, -1].detach().float()
    transported = bundle.lens.transport(residual, layer)
    jlens_logits = model.unembed(transported).detach().float().cpu()
    logit_lens_logits = model.unembed(residual).detach().float().cpu()
    next_token_logits = model.unembed(final_residual).detach().float().cpu()
    return PreSpeechSnapshot(
        layer=layer,
        residual=residual.cpu(),
        jlens_logits=jlens_logits,
        logit_lens_logits=logit_lens_logits,
        next_token_logits=next_token_logits,
        input_ids=input_ids.detach().cpu(),
    )


def capture_next_token_logits(
    bundle: ModelBundle,
    prompt: str,
    *,
    max_seq_len: int = 512,
) -> Any:
    """Return deterministic final-layer logits at the last prompt position."""

    from jlens.hooks import ActivationRecorder

    model = bundle.lens_model
    final_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
    residual = recorder.activations[final_layer][0, -1].detach().float()
    return model.unembed(residual).detach().float().cpu()


def candidate_scores(logits: Any, token_ids: Mapping[str, int]) -> dict[str, float]:
    """Immediately restrict a vocabulary-sized tensor to candidate scores."""

    if getattr(logits, "ndim", None) != 1:
        raise ValueError("logits must be a one-dimensional vocabulary tensor")
    return {
        candidate: float(logits[token_id].item())
        for candidate, token_id in token_ids.items()
    }


def generate_completion(
    bundle: ModelBundle,
    messages: Sequence[Mapping[str, str]],
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> str:
    """Generate only the assistant continuation under an explicit seed."""

    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt = render_chat(bundle.tokenizer, messages)
    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    encoded = {
        key: value.to(bundle.lens_model.input_device) for key, value in encoded.items()
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    do_sample = temperature > 0
    with torch.no_grad():
        output = bundle.hf_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    continuation = output[0, encoded["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(continuation, skip_special_tokens=True).strip()
