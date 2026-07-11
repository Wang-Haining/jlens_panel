#!/usr/bin/env python3
"""Run or resume the model-backed paired clarification dirty experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jlens_panel.config import load_config  # noqa: E402
from jlens_panel.data import read_jsonl as read_bridge_jsonl  # noqa: E402
from jlens_panel.experiment import (  # noqa: E402
    CONDITIONS,
    ClarificationExperiment,
    ExperimentItem,
    JsonlResultStore,
)
from jlens_panel.experiment.adapters import (  # noqa: E402
    FrozenReadoutSelector,
    GenerationSettings,
    HFReceiverAdapter,
    HFSenderAdapter,
)
from jlens_panel.provenance import (  # noqa: E402
    build_manifest,
    git_revision,
    sha256_file,
    write_json_atomic,
)
from jlens_panel.storage import (  # noqa: E402
    DiskStatus,
    directory_size,
    enforce_disk_guard,
    inspect_disk,
)

BundleLoader = Callable[..., Any]
Completion = Callable[..., str]
DiskInspector = Callable[[str | Path], DiskStatus]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without loading model dependencies."""

    parser = argparse.ArgumentParser(
        description="Run the four-branch synthetic clarification dirty experiment."
    )
    parser.add_argument("--config", type=Path, default=Path("config/dirty_run.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--best-non-j", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        action="append",
        dest="seeds",
        help="Generation seed; repeat for paired stochastic replications.",
    )
    parser.add_argument("--initial-max-tokens", type=_positive_int)
    parser.add_argument("--clarification-max-tokens", type=_positive_int)
    parser.add_argument("--receiver-max-tokens", type=_positive_int)
    parser.add_argument("--sender-temperature", type=float)
    parser.add_argument("--receiver-temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--tempest",
        action="store_true",
        help="Apply the Tempest free-space threshold instead of the local one.",
    )
    return parser


def load_experiment_items(path: str | Path) -> list[ExperimentItem]:
    """Map the stable synthetic schema onto the experiment protocol."""

    examples = read_bridge_jsonl(path)
    return [
        ExperimentItem(
            item_id=example.example_id,
            context=example.agent_a_prompt,
            question=example.agent_b_prompt_template,
            gold_bridge=example.gold_bridge,
            gold_answer=example.final_answer,
        )
        for example in examples
    ]


def run(
    args: argparse.Namespace,
    *,
    bundle_loader: BundleLoader | None = None,
    completion: Completion | None = None,
    disk_inspector: DiskInspector = inspect_disk,
) -> dict[str, object]:
    """Execute a resumable dirty run with injectable model dependencies."""

    config = load_config(args.config)
    experiment_config = config["experiment"]
    settings = GenerationSettings(
        initial_max_tokens=(
            args.initial_max_tokens
            if args.initial_max_tokens is not None
            else int(experiment_config["initial_message_max_tokens"])
        ),
        clarification_max_tokens=(
            args.clarification_max_tokens
            if args.clarification_max_tokens is not None
            else int(experiment_config["clarification_max_tokens"])
        ),
        receiver_max_tokens=(
            args.receiver_max_tokens
            if args.receiver_max_tokens is not None
            else int(experiment_config["receiver_max_tokens"])
        ),
        sender_temperature=(
            args.sender_temperature
            if args.sender_temperature is not None
            else float(experiment_config["sender_temperature"])
        ),
        receiver_temperature=(
            args.receiver_temperature
            if args.receiver_temperature is not None
            else float(experiment_config["receiver_temperature"])
        ),
        top_p=(
            args.top_p if args.top_p is not None else float(experiment_config["top_p"])
        ),
    )
    seeds = _resolve_seeds(args.seeds, config)
    items = load_experiment_items(args.data)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("no synthetic examples selected")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    if args.output.exists() and not manifest_path.exists():
        raise ValueError(
            "result JSONL exists without its immutable manifest; refusing to resume"
        )
    run_spec = _run_spec(args, config, settings, seeds, items)
    _write_or_validate_manifest(
        manifest_path,
        run_spec=run_spec,
        config_path=args.config,
        project_root=args.project_root,
    )
    _enforce_runtime_disk_guard(
        args.output,
        config=config,
        tempest=args.tempest,
        project_root=args.project_root,
        disk_inspector=disk_inspector,
    )

    store = JsonlResultStore(args.output)
    if _is_complete(store, items, seeds):
        return _completion_summary(
            store,
            items=items,
            seeds=seeds,
            output=args.output,
            manifest=manifest_path,
        )

    selector = FrozenReadoutSelector.from_files(
        score_jsonl=args.scores,
        best_non_j_json=args.best_non_j,
    )
    if bundle_loader is None:
        from jlens_panel.modeling import load_model_bundle

        bundle_loader = load_model_bundle
    model = config["model"]
    bundle = bundle_loader(
        model_name=model["name"],
        revision=model["revision"],
        dtype=model["dtype"],
        device_map=model["device_map"],
        lens_path=None,
    )
    if completion is None:
        from jlens_panel.modeling import generate_completion

        completion = generate_completion

    sender = HFSenderAdapter(
        bundle=bundle,
        settings=settings,
        completion=completion,
    )
    receiver = HFReceiverAdapter(
        bundle=bundle,
        settings=settings,
        completion=completion,
    )
    experiment = ClarificationExperiment(
        sender=sender,
        selector=selector,
        receiver=receiver,
        store=store,
    )

    disk_check_interval = int(config["storage"]["disk_check_interval"])
    item_seed_count = 0
    for item in items:
        for seed in seeds:
            experiment.run_item(item, seed=seed)
            item_seed_count += 1
            if item_seed_count % disk_check_interval == 0:
                _enforce_runtime_disk_guard(
                    args.output,
                    config=config,
                    tempest=args.tempest,
                    project_root=args.project_root,
                    disk_inspector=disk_inspector,
                )

    _enforce_runtime_disk_guard(
        args.output,
        config=config,
        tempest=args.tempest,
        project_root=args.project_root,
        disk_inspector=disk_inspector,
    )

    return _completion_summary(
        store,
        items=items,
        seeds=seeds,
        output=args.output,
        manifest=manifest_path,
    )


def _is_complete(
    store: JsonlResultStore,
    items: Sequence[ExperimentItem],
    seeds: Sequence[int],
) -> bool:
    completed = {
        (outcome.item_id, outcome.seed, outcome.condition)
        for outcome in store.all_outcomes()
    }
    return all(
        store.initial_message_for(item.item_id, seed) is not None
        and all(
            (item.item_id, seed, condition) in completed for condition in CONDITIONS
        )
        for item in items
        for seed in seeds
    )


def _completion_summary(
    store: JsonlResultStore,
    *,
    items: Sequence[ExperimentItem],
    seeds: Sequence[int],
    output: Path,
    manifest: Path,
) -> dict[str, object]:
    selected_ids = {item.item_id for item in items}
    selected_seeds = set(seeds)
    completed = [
        outcome
        for outcome in store.all_outcomes()
        if outcome.item_id in selected_ids and outcome.seed in selected_seeds
    ]
    return {
        "output": str(output.resolve()),
        "manifest": str(manifest.resolve()),
        "items": len(items),
        "seeds": list(seeds),
        "completed_outcomes": len(completed),
        "expected_outcomes": len(items) * len(seeds) * len(CONDITIONS),
    }


def _resolve_seeds(
    explicit: list[int] | None, config: dict[str, Any]
) -> tuple[int, ...]:
    values = (
        explicit if explicit is not None else config["experiment"]["generation_seeds"]
    )
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise ValueError("at least one generation seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("generation seeds must be non-negative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("generation seeds must be unique")
    return seeds


def _run_spec(
    args: argparse.Namespace,
    config: dict[str, Any],
    settings: GenerationSettings,
    seeds: tuple[int, ...],
    items: Sequence[ExperimentItem],
) -> dict[str, object]:
    return {
        "schema_version": "dirty-clarification-run-v1",
        "model": dict(config["model"]),
        "data": _file_spec(args.data),
        "scores": _file_spec(args.scores),
        "best_non_j": _file_spec(args.best_non_j),
        "output": str(args.output.resolve()),
        "item_ids": [item.item_id for item in items],
        "seeds": list(seeds),
        "generation": asdict(settings),
    }


def _file_spec(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_or_validate_manifest(
    path: Path,
    *,
    run_spec: dict[str, object],
    config_path: Path,
    project_root: Path,
) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_spec = existing["extra"]["run_spec"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid existing run manifest: {path}") from error
        if existing_spec != run_spec:
            raise ValueError(
                "existing manifest does not match this invocation; use a new output"
            )
        if existing.get("config_sha256") != sha256_file(config_path):
            raise ValueError("configuration changed since this run was created")
        current_revision = git_revision(project_root)
        if existing.get("git_revision") != current_revision:
            raise ValueError("Git revision changed since this run was created")
        return
    manifest = build_manifest(
        config_path=config_path,
        project_root=project_root,
        extra={"run_spec": run_spec},
    )
    write_json_atomic(path, manifest)


def _enforce_runtime_disk_guard(
    output: Path,
    *,
    config: dict[str, Any],
    tempest: bool,
    project_root: Path,
    disk_inspector: DiskInspector,
) -> None:
    storage = config["storage"]
    output.parent.mkdir(parents=True, exist_ok=True)
    status = disk_inspector(project_root)
    if tempest:
        minimum_free = int(float(storage["tempest_minimum_free_tb"]) * 10**12)
    else:
        minimum_free = int(float(storage["local_minimum_free_gb"]) * 10**9)
    maximum_project = int(float(storage["project_warning_gb"]) * 10**9)
    maximum_run = int(float(storage["run_hard_limit_gb"]) * 10**9)
    enforce_disk_guard(
        status,
        minimum_free_bytes=minimum_free,
        maximum_used_fraction=float(storage["filesystem_warning_fraction"]),
        maximum_project_bytes=maximum_project,
    )
    run_size = directory_size(output.parent)
    if run_size >= maximum_run:
        raise ValueError(
            f"run directory reached the hard limit: {run_size} >= {maximum_run} bytes"
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
