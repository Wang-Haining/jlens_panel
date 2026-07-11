#!/usr/bin/env python3
"""Generate deterministic JSONL data for the J-Lens Panel dirty run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jlens_panel.data.synthetic_bridge import (  # noqa: E402
    DEFAULT_BRIDGE_CANDIDATES,
    SCHEMA_VERSION,
    BridgeDataError,
    dataset_fingerprint,
    generate_dataset,
    gold_bridge_counts,
    validate_candidates,
    write_dataset_jsonl,
)
from jlens_panel.provenance import build_manifest, write_json_atomic  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate controlled two-agent bridge-concept JSONL splits."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/dirty_run.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--min-distractors", type=int, default=6)
    parser.add_argument("--max-distractors", type=int, default=10)
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Bridge concept; repeat exactly 16 times.",
    )
    candidates.add_argument(
        "--candidates-file",
        type=Path,
        help="JSON array or UTF-8 file with one bridge concept per line.",
    )
    return parser


def _load_candidates(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        value = json.loads(text)
        if not isinstance(value, list):
            raise BridgeDataError("candidate JSON must contain a list")
        return value
    return [line for line in (item.strip() for item in text.splitlines()) if line]


def _resolve_candidates(args: argparse.Namespace) -> tuple[str, ...]:
    if args.candidates_file is not None:
        return validate_candidates(_load_candidates(args.candidates_file))
    if args.candidates is not None:
        return validate_candidates(args.candidates)
    return DEFAULT_BRIDGE_CANDIDATES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        candidates = _resolve_candidates(args)
        dataset = generate_dataset(
            candidates=candidates,
            seed=args.seed,
            train_size=args.train_size,
            dev_size=args.dev_size,
            test_size=args.test_size,
            min_distractors=args.min_distractors,
            max_distractors=args.max_distractors,
        )
        paths = write_dataset_jsonl(args.output_dir, dataset)
    except (BridgeDataError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))

    summary = {
        "schema": SCHEMA_VERSION,
        "seed": args.seed,
        "distractor_range": [args.min_distractors, args.max_distractors],
        "candidate_bridges": list(candidates),
        "dataset_sha256": dataset_fingerprint(dataset),
        "splits": {
            split: {
                "examples": len(dataset[split]),
                "gold_bridge_counts": gold_bridge_counts(dataset[split], candidates),
                "path": str(paths[split].resolve()),
                "sha256": _sha256(paths[split]),
            }
            for split in ("train", "dev", "test")
        },
    }
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    manifest = build_manifest(
        config_path=args.config,
        project_root=args.project_root,
        extra={"dataset": summary},
    )
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {**summary, "manifest": str(manifest_path.resolve())},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
