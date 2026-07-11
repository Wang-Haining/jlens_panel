#!/usr/bin/env python3
"""Extract compact candidate-only readouts from Agent A pre-speech states."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jlens_panel.config import load_config
from jlens_panel.io import read_jsonl
from jlens_panel.modeling import (
    candidate_scores,
    canonical_layer,
    capture_next_token_logits,
    capture_pre_speech,
    load_model_bundle,
    render_chat,
    resolve_candidate_token_ids,
)
from jlens_panel.provenance import build_manifest, git_revision, sha256_file
from jlens_panel.readouts.artifacts import (
    ArtifactConflictError,
    ExtractionExample,
    LoadFunction,
    RunByteBudget,
    SaveFunction,
    artifact_path,
    build_artifact,
    build_extraction_manifest,
    build_text_only_prompt,
    ensure_extraction_manifest,
    load_artifact,
    path_fingerprint,
    resume_matches,
    save_artifact_atomic,
    stable_fingerprint,
    validate_shared_inventory,
)
from jlens_panel.storage import build_disk_guard

CapturePreSpeech = Callable[..., Any]
CaptureNextToken = Callable[..., Any]
RenderChat = Callable[[Any, Sequence[Mapping[str, str]]], str]
ResidualConverter = Callable[[object], object]
DiskCheck = Callable[[], None]


def load_extraction_examples(paths: Sequence[str | Path]) -> list[ExtractionExample]:
    """Load one or more JSONL files under the stable synthetic schema."""

    if not paths:
        raise ValueError("at least one input JSONL path is required")
    examples = [
        ExtractionExample.from_mapping(record)
        for path in paths
        for record in read_jsonl(path)
    ]
    validate_shared_inventory(examples)
    return examples


def to_float16_cpu(residual: object) -> object:
    """Detach and compact one residual, importing Torch only in an extraction run."""

    import torch

    detach = getattr(residual, "detach", None)
    value = detach() if callable(detach) else residual
    to = getattr(value, "to", None)
    if not callable(to):
        raise ValueError("captured residual does not implement tensor.to")
    return to(device="cpu", dtype=torch.float16)


def extraction_identity(
    *,
    config_sha256: str,
    lens_sha256: str,
    model_name: str,
    model_revision: str,
    layer: int,
    max_seq_len: int,
    candidates: Sequence[str],
    candidate_token_ids: Mapping[str, int],
    input_sha256: Mapping[str, str],
    git_revision_value: str | None,
) -> str:
    """Return the static identity used to reject incompatible resume files."""

    return stable_fingerprint(
        {
            "config_sha256": config_sha256,
            "lens_sha256": lens_sha256,
            "model_name": model_name,
            "model_revision": model_revision,
            "layer": layer,
            "max_seq_len": max_seq_len,
            "candidates": list(candidates),
            "candidate_token_ids": dict(candidate_token_ids),
            "input_sha256": dict(input_sha256),
            "git_revision": git_revision_value,
        }
    )


def extract_examples(
    examples: Sequence[ExtractionExample],
    *,
    output_dir: str | Path,
    bundle: Any,
    layer: int,
    candidate_token_ids: Mapping[str, int],
    extraction_fingerprint: str,
    provenance: Mapping[str, object],
    max_seq_len: int,
    hard_limit_bytes: int,
    limit: int | None = None,
    disk_check_every: int = 25,
    disk_check: DiskCheck | None = None,
    capture_pre_speech_fn: CapturePreSpeech = capture_pre_speech,
    capture_next_token_logits_fn: CaptureNextToken = capture_next_token_logits,
    render_chat_fn: RenderChat = render_chat,
    residual_converter: ResidualConverter = to_float16_cpu,
    save_fn: SaveFunction | None = None,
    load_fn: LoadFunction | None = None,
) -> dict[str, int]:
    """Extract one artifact per example with strict resume and disk semantics."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if disk_check_every < 1:
        raise ValueError("disk_check_every must be positive")
    inventory = validate_shared_inventory(examples)
    if tuple(candidate_token_ids) != inventory:
        raise ValueError("candidate_token_ids must follow canonical inventory order")
    if len(set(candidate_token_ids.values())) != len(candidate_token_ids):
        raise ValueError("candidate_token_ids must be unique")

    selected = _limit_per_split(examples, limit)
    output_root = Path(output_dir)
    stats = {"requested": len(selected), "written": 0, "skipped": 0}
    if disk_check is not None:
        disk_check()
    manifest = build_extraction_manifest(
        examples=selected,
        candidate_token_ids=candidate_token_ids,
        layer=layer,
        extraction_fingerprint=extraction_fingerprint,
        provenance=provenance,
    )
    stored_manifest = ensure_extraction_manifest(output_root, manifest)
    run_provenance = stored_manifest["provenance"]
    assert isinstance(run_provenance, Mapping)
    byte_budget = RunByteBudget.inspect(output_root, hard_limit_bytes)
    for index, example in enumerate(selected):
        if disk_check is not None and index and index % disk_check_every == 0:
            disk_check()
        destination = artifact_path(output_root, example.split, example.example_id)
        if destination.exists():
            existing = load_artifact(destination, load_fn=load_fn)
            if not resume_matches(
                existing,
                example=example,
                extraction_fingerprint=extraction_fingerprint,
            ):
                raise ArtifactConflictError(
                    f"existing artifact is incompatible with this run: {destination}"
                )
            stats["skipped"] += 1
            continue

        natural_prompt = render_chat_fn(
            bundle.tokenizer,
            [{"role": "user", "content": example.agent_a_prompt}],
        )
        # Exactly one pre-speech capture produces all three model-side readouts.
        snapshot = capture_pre_speech_fn(
            bundle,
            natural_prompt,
            layer=layer,
            max_seq_len=max_seq_len,
        )
        if int(snapshot.layer) != layer:
            raise ValueError(
                f"capture returned layer {snapshot.layer}, expected canonical layer {layer}"
            )

        # Restrict vocabulary tensors immediately; never retain them in artifacts.
        stored_scores: dict[str, dict[str, float]] = {
            "jlens": candidate_scores(snapshot.jlens_logits, candidate_token_ids),
            "logit_lens": candidate_scores(
                snapshot.logit_lens_logits, candidate_token_ids
            ),
            "next_token": candidate_scores(
                snapshot.next_token_logits, candidate_token_ids
            ),
        }
        compact_residual = residual_converter(snapshot.residual)

        text_prompt = build_text_only_prompt(
            example.agent_a_probe_prompt,
            inventory,
        )
        text_vocab_logits = capture_next_token_logits_fn(
            bundle,
            text_prompt,
            max_seq_len=max_seq_len,
        )
        stored_scores["text_only"] = candidate_scores(
            text_vocab_logits,
            candidate_token_ids,
        )

        artifact = build_artifact(
            example=example,
            candidate_token_ids=candidate_token_ids,
            layer=layer,
            residual=compact_residual,
            scores=stored_scores,
            extraction_fingerprint=extraction_fingerprint,
            provenance=run_provenance,
        )
        save_artifact_atomic(
            destination,
            artifact,
            run_root=output_root,
            hard_limit_bytes=hard_limit_bytes,
            save_fn=save_fn,
            byte_budget=byte_budget,
        )
        stats["written"] += 1

    if disk_check is not None:
        disk_check()
    return stats


def _limit_per_split(
    examples: Sequence[ExtractionExample],
    limit: int | None,
) -> tuple[ExtractionExample, ...]:
    if limit is None:
        return tuple(examples)
    counts = {"train": 0, "dev": 0, "test": 0}
    selected: list[ExtractionExample] = []
    for example in examples:
        if counts[example.split] < limit:
            selected.append(example)
            counts[example.split] += 1
    return tuple(selected)


def build_disk_check(
    *,
    config: Mapping[str, object],
    project_root: str | Path,
    environment: str,
) -> DiskCheck:
    """Build the periodic filesystem guard; RunByteBudget tracks run size."""

    return build_disk_guard(
        config,
        project_root=project_root,
        environment=environment,
    )


def move_canonical_jacobian_to_input_device(bundle: Any, layer: int) -> None:
    """Pin only the canonical Jacobian on the model input device once per run."""

    jacobians = bundle.lens.jacobians
    destination = bundle.lens_model.input_device
    moved = jacobians[layer].to(device=destination)
    try:
        jacobians[layer] = moved
    except (TypeError, RuntimeError) as error:
        raise RuntimeError(
            "lens.jacobians does not support pinning the canonical layer"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract candidate-only J-lens panel artifacts."
    )
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Synthetic JSONL; repeat for train/dev/test files.",
    )
    parser.add_argument("--lens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", choices=("local", "tempest"), default="local")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--disk-check-every", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug limit applied independently to each split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")

    config = load_config(args.config)
    disk_check_every = (
        args.disk_check_every
        if args.disk_check_every is not None
        else int(config["storage"].get("disk_check_interval", 25))
    )
    if disk_check_every < 1:
        raise SystemExit("disk check interval must be positive")
    examples = load_extraction_examples(args.input)
    inventory = validate_shared_inventory(examples)
    disk_check = build_disk_check(
        config=config,
        project_root=args.project_root,
        environment=args.environment,
    )
    # Check disk before importing any GPU dependency or loading model weights.
    disk_check()

    model_config = config["model"]
    lens_config = config["lens"]
    bundle = load_model_bundle(
        model_name=model_config["name"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        device_map=model_config["device_map"],
        lens_path=args.lens,
    )
    candidate_token_ids = resolve_candidate_token_ids(bundle.tokenizer, inventory)
    layer = canonical_layer(
        bundle.lens.source_layers,
        lens_config["canonical_layer_strategy"],
    )
    move_canonical_jacobian_to_input_device(bundle, layer)
    config_hash = sha256_file(args.config)
    lens_hash = path_fingerprint(args.lens)
    input_hashes = {str(Path(path).resolve()): sha256_file(path) for path in args.input}
    revision = git_revision(args.project_root)
    identity = extraction_identity(
        config_sha256=config_hash,
        lens_sha256=lens_hash,
        model_name=model_config["name"],
        model_revision=model_config["revision"],
        layer=layer,
        max_seq_len=args.max_seq_len,
        candidates=inventory,
        candidate_token_ids=candidate_token_ids,
        input_sha256=input_hashes,
        git_revision_value=revision,
    )
    provenance = build_manifest(
        config_path=args.config,
        project_root=args.project_root,
        extra={
            "inputs": input_hashes,
            "lens": str(Path(args.lens).resolve()),
            "lens_sha256": lens_hash,
            "model_name": model_config["name"],
            "model_revision_requested": model_config["revision"],
            "model_revision_resolved": getattr(
                bundle.hf_model.config, "_commit_hash", None
            ),
            "upstream_commit": lens_config["upstream_commit"],
            "canonical_layer": layer,
            "candidate_inventory_sha256": stable_fingerprint(list(inventory)),
            "extraction_sha256": identity,
            "git_revision": revision,
        },
    )
    hard_limit = int(float(config["storage"]["run_hard_limit_gb"]) * 10**9)
    stats = extract_examples(
        examples,
        output_dir=args.output_dir,
        bundle=bundle,
        layer=layer,
        candidate_token_ids=candidate_token_ids,
        extraction_fingerprint=identity,
        provenance=provenance,
        max_seq_len=args.max_seq_len,
        hard_limit_bytes=hard_limit,
        limit=args.limit,
        disk_check_every=disk_check_every,
        disk_check=disk_check,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
