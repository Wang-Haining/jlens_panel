"""Dependency-light loading and deterministic sampling for text corpora."""

from __future__ import annotations

import json
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class CorpusError(ValueError):
    """Raised when a text corpus violates the shared loader contract."""


@dataclass(frozen=True, slots=True)
class SampledPrompt:
    """One reproducibly selected prompt and its original corpus index."""

    sample_index: int
    corpus_index: int
    text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise CorpusError("sample_index must be a non-negative integer")
        if (
            isinstance(self.corpus_index, bool)
            or not isinstance(self.corpus_index, int)
            or self.corpus_index < 0
        ):
            raise CorpusError("corpus_index must be a non-negative integer")
        if not isinstance(self.text, str) or not self.text.strip():
            raise CorpusError("sampled prompt text must be non-empty")

    @property
    def text_sha256(self) -> str:
        """Return a content hash without exposing prompt text in manifests."""

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_prompts(path: str | Path) -> list[str]:
    """Load a JSON list or JSONL records with a text/prompt field."""

    source = Path(path)
    try:
        if source.suffix == ".jsonl":
            values: Any = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            values = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot load corpus: {source}") from error
    if not isinstance(values, list):
        raise CorpusError("Fit corpus must contain a JSON list")

    prompts: list[str] = []
    for value in values:
        if isinstance(value, str):
            prompt = value
        elif isinstance(value, dict):
            prompt = value.get("text") or value.get("prompt")
        else:
            prompt = None
        if not isinstance(prompt, str) or not prompt.strip():
            raise CorpusError("Every fit-corpus record must provide non-empty text")
        prompts.append(prompt)
    return prompts


def sample_prompts(
    prompts: Sequence[str],
    *,
    count: int,
    seed: int,
) -> tuple[SampledPrompt, ...]:
    """Sample corpus rows without replacement under an explicit seed."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CorpusError("sample count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CorpusError("sample seed must be an integer")
    if len(prompts) < count:
        raise CorpusError(
            f"corpus has {len(prompts)} prompts but sample requires {count}"
        )
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise CorpusError("sample input must contain only non-empty prompt strings")

    indices = random.Random(seed).sample(range(len(prompts)), count)
    return tuple(
        SampledPrompt(
            sample_index=sample_index,
            corpus_index=corpus_index,
            text=prompts[corpus_index],
        )
        for sample_index, corpus_index in enumerate(indices)
    )


def sampled_prompt_manifest(
    sampled_prompts: Sequence[SampledPrompt],
) -> tuple[dict[str, int | str], ...]:
    """Return the ordered, text-free identity records for a prompt sample."""

    sampled = tuple(sampled_prompts)
    if tuple(prompt.sample_index for prompt in sampled) != tuple(range(len(sampled))):
        raise CorpusError("sample indices must be contiguous and ordered")
    corpus_indices = tuple(prompt.corpus_index for prompt in sampled)
    if len(corpus_indices) != len(set(corpus_indices)):
        raise CorpusError("sampled corpus indices must be unique")
    return tuple(
        {
            "sample_index": prompt.sample_index,
            "corpus_index": prompt.corpus_index,
            "text_sha256": prompt.text_sha256,
        }
        for prompt in sampled
    )


def sampled_prompt_fingerprint(sampled_prompts: Sequence[SampledPrompt]) -> str:
    """Hash ordered prompt indices and content hashes using canonical JSON."""

    payload = json.dumps(
        sampled_prompt_manifest(sampled_prompts),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
