import importlib.util
from pathlib import Path

import pytest

from jlens_panel.data import DEFAULT_BRIDGE_CANDIDATES, generate_split, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sweep_positions_for_test",
    ROOT / "scripts" / "sweep_positions.py",
)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def test_train_dev_loader_never_accepts_a_mislabeled_split(tmp_path: Path) -> None:
    train = generate_split(
        "train",
        size=16,
        candidates=DEFAULT_BRIDGE_CANDIDATES,
        seed=17,
    )
    dev = generate_split(
        "dev",
        size=16,
        candidates=DEFAULT_BRIDGE_CANDIDATES,
        seed=17,
    )
    train_path = write_jsonl(tmp_path / "train.jsonl", train)
    dev_path = write_jsonl(tmp_path / "dev.jsonl", dev)

    loaded_train, loaded_dev = SWEEP._load_train_dev(
        train_path,
        dev_path,
        expected_train=16,
        expected_dev=16,
    )

    assert len(loaded_train) == len(loaded_dev) == 16
    wrong_dev_path = write_jsonl(tmp_path / "wrong_dev.jsonl", train)
    with pytest.raises(SWEEP.SweepRunError, match="exactly train/dev"):
        SWEEP._load_train_dev(
            train_path,
            wrong_dev_path,
            expected_train=16,
            expected_dev=16,
        )


def test_capture_manifest_is_immutable_and_reusable(tmp_path: Path) -> None:
    path = tmp_path / "capture_manifest.json"
    identity = {"capture": "fixed", "test_or_smoke_read": False}

    first = SWEEP._ensure_capture_manifest(
        path,
        identity=identity,
        expected_counts={"train": 2, "dev": 1},
        config_path=ROOT / "config" / "sprint.yaml",
        project_root=ROOT,
    )
    second = SWEEP._ensure_capture_manifest(
        path,
        identity=identity,
        expected_counts={"train": 2, "dev": 1},
        config_path=ROOT / "config" / "sprint.yaml",
        project_root=ROOT,
    )

    assert first == second
    with pytest.raises(SWEEP.SweepRunError, match="identity changed"):
        SWEEP._ensure_capture_manifest(
            path,
            identity={"capture": "changed", "test_or_smoke_read": False},
            expected_counts={"train": 2, "dev": 1},
            config_path=ROOT / "config" / "sprint.yaml",
            project_root=ROOT,
        )


def test_csv_writer_uses_exact_frozen_columns(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    row = {
        "position": "template_tail",
        "layer": 0,
        "method": "probe",
        "C": 1.0,
        "dev_top1": 0.25,
        "dev_mrr": 0.4,
        "dev_logloss": 2.0,
        "n_train": 1000,
        "n_dev": 200,
    }

    SWEEP._write_csv_atomic(path, [row])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(SWEEP.RESULT_COLUMNS)
    assert len(lines) == 2


def test_artifact_preflight_estimate_stays_below_run_budget() -> None:
    per_artifact = SWEEP._estimate_artifact_bytes(
        layers=27,
        hidden_size=3584,
        candidates=16,
    )

    assert per_artifact * 1200 < 20_000_000_000 - SWEEP._INDEX_RESERVE_BYTES


def test_sbatch_is_one_h100_train_dev_only_and_calibrates_first() -> None:
    text = (ROOT / "runs" / "05_position_sweep.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --account=group-jasonclark" in text
    assert "#SBATCH --gres=gpu:h100:1" in text
    assert "#SBATCH --time=0-04:00:00" in text
    assert "--phase capture" in text
    assert "--phase analyze" in text
    assert "export JLENS_PROBE_BLAS_THREADS=4" in text
    assert "train.jsonl" in text and "dev.jsonl" in text
    assert "test.jsonl" not in text and "smoke" not in text.casefold()
    assert text.index("scripts/calibrate_null.py") < text.index(
        "scripts/sweep_positions.py"
    )
