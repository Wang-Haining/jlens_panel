#!/usr/bin/env python3
"""Fit a reproducible Jacobian Lens with resumable upstream checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from jlens_panel.config import load_config
from jlens_panel.corpus import load_prompts
from jlens_panel.modeling import load_model_bundle
from jlens_panel.provenance import (
    build_manifest,
    git_revision,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.storage import enforce_disk_guard, inspect_disk


def ensure_fit_state_manifest(
    path: Path,
    *,
    fit_spec: dict[str, Any],
    config_path: str | Path,
    project_root: str | Path,
) -> None:
    """Create or validate the immutable identity of a resumable fit state."""

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_spec = existing["extra"]["fit_spec"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid fit-state manifest: {path}") from error
        if existing_spec != fit_spec:
            raise ValueError(
                "fit inputs changed since the checkpoint was created; "
                "use a new output path"
            )
        return
    write_json_atomic(
        path,
        build_manifest(
            config_path=config_path,
            project_root=project_root,
            extra={"fit_spec": fit_spec},
        ),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", choices=("local", "tempest"), default="local")
    args = parser.parse_args()

    config = load_config(args.config)
    storage = config["storage"]
    status = inspect_disk(args.project_root)
    if args.environment == "tempest":
        minimum_free_bytes = int(float(storage["tempest_minimum_free_tb"]) * 10**12)
    else:
        minimum_free_bytes = int(float(storage["local_minimum_free_gb"]) * 10**9)
    enforce_disk_guard(
        status,
        minimum_free_bytes=minimum_free_bytes,
        maximum_used_fraction=float(storage["filesystem_warning_fraction"]),
        maximum_project_bytes=int(float(storage["project_warning_gb"]) * 10**9),
    )

    model_config = config["model"]
    lens_config = config["lens"]
    prompts = load_prompts(args.prompts)[: int(lens_config["fit_prompts"])]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint or output.with_suffix(".fit-state.pt"))
    fit_spec = {
        "config_sha256": sha256_file(args.config),
        "fit_corpus": str(Path(args.prompts).resolve()),
        "fit_corpus_sha256": sha256_file(args.prompts),
        "fit_prompts_requested": len(prompts),
        "model_name": model_config["name"],
        "model_revision": model_config["revision"],
        "sequence_length": int(lens_config["sequence_length"]),
        "skip_first_positions": int(lens_config["skip_first_positions"]),
        "checkpoint_every": int(lens_config["checkpoint_every"]),
        "upstream_commit": lens_config["upstream_commit"],
        "git_revision": git_revision(args.project_root),
    }
    fit_state_manifest = checkpoint.with_suffix(".manifest.json")
    if checkpoint.exists() and not fit_state_manifest.exists():
        raise ValueError(
            "fit checkpoint exists without its identity manifest; "
            "use a new output path"
        )
    ensure_fit_state_manifest(
        fit_state_manifest,
        fit_spec=fit_spec,
        config_path=args.config,
        project_root=args.project_root,
    )
    bundle = load_model_bundle(
        model_name=model_config["name"],
        revision=model_config["revision"],
        dtype=model_config["dtype"],
        device_map=model_config["device_map"],
    )

    import jlens

    lens = jlens.fit(
        bundle.lens_model,
        prompts=prompts,
        max_seq_len=int(lens_config["sequence_length"]),
        skip_first=int(lens_config["skip_first_positions"]),
        checkpoint_path=str(checkpoint),
        checkpoint_every=int(lens_config["checkpoint_every"]),
        resume=True,
    )
    lens.save(str(output))
    manifest = build_manifest(
        config_path=args.config,
        project_root=args.project_root,
        extra={
            "fit_corpus": str(Path(args.prompts).resolve()),
            "fit_corpus_sha256": sha256_file(args.prompts),
            "fit_prompts_requested": len(prompts),
            "fit_prompts_used": lens.n_prompts,
            "lens_path": str(output.resolve()),
            "lens_sha256": sha256_file(output),
            "model_name": model_config["name"],
            "model_revision_requested": model_config["revision"],
            "model_revision_resolved": getattr(
                bundle.hf_model.config, "_commit_hash", None
            ),
            "source_layers": lens.source_layers,
            "upstream_commit": lens_config["upstream_commit"],
        },
    )
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)


if __name__ == "__main__":
    main()
