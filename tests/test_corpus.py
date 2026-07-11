import importlib.util
import json
from pathlib import Path

import pytest

from jlens_panel.corpus import (
    CorpusError,
    load_prompts,
    sample_prompts,
    sampled_prompt_fingerprint,
    sampled_prompt_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def load_fit_lens_module():
    spec = importlib.util.spec_from_file_location(
        "fit_lens_for_test",
        ROOT / "scripts" / "fit_lens.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_lens_uses_the_shared_corpus_loader() -> None:
    assert load_fit_lens_module().load_prompts is load_prompts


def test_load_prompts_accepts_shared_json_and_jsonl_contract(tmp_path: Path) -> None:
    json_path = tmp_path / "prompts.json"
    json_path.write_text(
        json.dumps(["one", {"text": "two"}, {"prompt": "three"}]),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "prompts.jsonl"
    jsonl_path.write_text('"one"\n{"text": "two"}\n', encoding="utf-8")

    assert load_prompts(json_path) == ["one", "two", "three"]
    assert load_prompts(jsonl_path) == ["one", "two"]

    json_path.write_text(json.dumps([{"missing": "text"}]), encoding="utf-8")
    with pytest.raises(CorpusError, match="non-empty text"):
        load_prompts(json_path)


def test_sample_prompts_is_seeded_without_replacement() -> None:
    prompts = [f"prompt-{index}" for index in range(6)]

    first = sample_prompts(prompts, count=4, seed=17)
    second = sample_prompts(prompts, count=4, seed=17)

    assert first == second
    assert len({prompt.corpus_index for prompt in first}) == 4
    assert [prompt.sample_index for prompt in first] == list(range(4))
    assert all(prompt.text == prompts[prompt.corpus_index] for prompt in first)
    assert sampled_prompt_manifest(first) == sampled_prompt_manifest(second)
    assert sampled_prompt_fingerprint(first) == sampled_prompt_fingerprint(second)
    assert all("text" not in item for item in sampled_prompt_manifest(first))


def test_sample_prompts_fails_closed() -> None:
    with pytest.raises(CorpusError, match="requires 3"):
        sample_prompts(["one", "two"], count=3, seed=1)
    with pytest.raises(CorpusError, match="positive integer"):
        sample_prompts(["one"], count=True, seed=1)
    with pytest.raises(CorpusError, match="seed must be an integer"):
        sample_prompts(["one"], count=1, seed=True)
