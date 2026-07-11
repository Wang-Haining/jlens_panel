#!/usr/bin/env python3
"""Fail-fast validation of the pinned GPU runtime and candidate vocabulary."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

from jlens_panel.config import load_config
from jlens_panel.data.synthetic_bridge import DEFAULT_BRIDGE_CANDIDATES, read_jsonl
from jlens_panel.modeling import resolve_candidate_token_ids
from jlens_panel.provenance import (
    git_is_dirty,
    git_revision,
    installed_versions,
    write_json_atomic,
)


def jlens_source_commit() -> str | None:
    """Read the VCS commit recorded by pip for the installed jlens package."""

    distribution = metadata.distribution("jlens")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    value = json.loads(direct_url)
    return value.get("vcs_info", {}).get("commit_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--data-jsonl")
    parser.add_argument("--output", default="results/runtime_check.json")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit a tokenizer download instead of requiring the Tempest cache.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repository = Path.cwd()
    revision = git_revision(repository)
    if revision is None:
        raise RuntimeError("runtime check must run from a Git checkout")
    if git_is_dirty(repository):
        raise RuntimeError("runtime Git checkout is dirty; sync through GitHub first")
    expected_commit = str(config["lens"]["upstream_commit"])
    installed_commit = jlens_source_commit()
    if installed_commit != expected_commit:
        raise RuntimeError(
            f"jlens commit mismatch: expected {expected_commit}, got {installed_commit}"
        )

    if args.data_jsonl:
        candidates = read_jsonl(args.data_jsonl)[0].candidate_bridges
    else:
        candidates = DEFAULT_BRIDGE_CANDIDATES

    import torch
    import transformers

    model_config = config["model"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        local_files_only=not args.allow_download,
    )
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if (
        tokenizer_revision is not None
        and tokenizer_revision != model_config["revision"]
    ):
        raise RuntimeError(
            "tokenizer revision does not match the pinned model revision: "
            f"{tokenizer_revision} != {model_config['revision']}"
        )
    candidate_token_ids = resolve_candidate_token_ids(tokenizer, candidates)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the runtime-check job")

    result = {
        "candidate_token_ids": candidate_token_ids,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "jlens_commit": installed_commit,
        "git_revision": revision,
        "model": model_config["name"],
        "model_revision": model_config["revision"],
        "tokenizer_revision": tokenizer_revision,
        "packages": installed_versions(),
    }
    write_json_atomic(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
