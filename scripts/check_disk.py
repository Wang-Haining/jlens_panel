#!/usr/bin/env python3
"""Check configured disk thresholds before or during a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jlens_panel.config import load_config
from jlens_panel.storage import enforce_disk_guard, inspect_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dirty_run.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--environment", choices=("local", "tempest"), default="local")
    args = parser.parse_args()

    config = load_config(args.config)
    storage = config["storage"]
    status = inspect_disk(Path(args.project_root))
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
    print(
        json.dumps(
            {
                "free_bytes": status.free_bytes,
                "project_bytes": status.project_bytes,
                "used_fraction": status.used_fraction,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
