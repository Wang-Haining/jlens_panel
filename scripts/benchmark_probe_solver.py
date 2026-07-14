#!/usr/bin/env python3
"""Benchmark bounded sklearn probe fits on one authorized train/dev cell."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_positions import _load_capture_arrays, _load_train_dev


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/generated/dirty_v3/train.jsonl")
    parser.add_argument("--dev", default="data/generated/dirty_v3/dev.jsonl")
    parser.add_argument("--artifacts", default="artifacts/sweep_v3")
    parser.add_argument("--position", default="template_tail")
    parser.add_argument("--layer", type=int, default=0)
    return parser


def _fit_once(
    *,
    solver: str,
    max_iter: int,
    tol: float,
    train_rows: object,
    train_targets: tuple[int, ...],
    dev_rows: object,
    dev_targets: tuple[int, ...],
    threads: int,
    dual: bool = False,
) -> None:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    estimator = LogisticRegression(
        C=1.0,
        dual=dual,
        max_iter=max_iter,
        random_state=0,
        solver=solver,
        tol=tol,
    )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=threads):
            estimator.fit(train_rows, train_targets)
            predictions = estimator.predict(dev_rows)
    elapsed = time.monotonic() - started
    accuracy = float(np.mean(predictions == np.asarray(dev_targets)))
    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    print(
        f"solver={solver} dual={dual} threads={threads} max_iter={max_iter} "
        f"tol={tol:g} elapsed={elapsed:.3f}s converged={converged} "
        f"n_iter={estimator.n_iter_.tolist()} dev_top1={accuracy:.6f}",
        flush=True,
    )


def main() -> None:
    args = _parser().parse_args()
    load_started = time.monotonic()
    train, dev = _load_train_dev(
        Path(args.train),
        Path(args.dev),
        expected_train=1000,
        expected_dev=200,
    )
    (
        train_cells,
        dev_cells,
        _scores,
        train_targets,
        dev_targets,
        _dev_ids,
        _candidates,
        _candidate_token_ids,
        _layers,
        _capture_manifest_sha256,
        _capture_index_sha256,
    ) = _load_capture_arrays(Path(args.artifacts), train=train, dev=dev)
    cell = (args.position, args.layer)
    train_rows = train_cells[cell]
    dev_rows = dev_cells[cell]
    print(
        f"loaded={time.monotonic() - load_started:.3f}s "
        f"train_shape={train_rows.shape} dev_shape={dev_rows.shape}",
        flush=True,
    )
    for threads in (1, 4, 12, 32):
        _fit_once(
            solver="lbfgs",
            max_iter=10,
            tol=1e-4,
            train_rows=train_rows,
            train_targets=train_targets,
            dev_rows=dev_rows,
            dev_targets=dev_targets,
            threads=threads,
        )
    for threads in (1, 4, 12):
        _fit_once(
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4,
            train_rows=train_rows,
            train_targets=train_targets,
            dev_rows=dev_rows,
            dev_targets=dev_targets,
            threads=threads,
        )


if __name__ == "__main__":
    main()
