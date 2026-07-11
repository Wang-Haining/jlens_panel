from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "recalibrate_v3",
        ROOT / "scripts" / "recalibrate_v3.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECALIBRATE = _load_script()
CANDIDATES = ("amber", "bamboo", "copper", "delta")
TOKEN_IDS = {candidate: index + 10 for index, candidate in enumerate(CANDIDATES)}


def _artifact(
    example_id: str,
    split: str,
    gold: str,
    scores: dict[str, float],
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": split,
        "candidates": list(CANDIDATES),
        "candidate_token_ids": TOKEN_IDS,
        "gold_bridge": gold,
        "gold_token_id": TOKEN_IDS[gold],
        "layer": 13,
        "scores": {
            "jlens": dict(scores),
            "logit_lens": dict(scores),
            "next_token": dict(scores),
        },
        "fingerprints": {"extraction": "e" * 64},
    }


def _manifest(train_count: int, dev_count: int) -> dict[str, object]:
    return {
        "candidates": list(CANDIDATES),
        "candidate_token_ids": TOKEN_IDS,
        "canonical_layer": 13,
        "extraction_sha256": "e" * 64,
        "expected_counts": {
            "train": train_count,
            "dev": dev_count,
            "test": 999,
        },
    }


def _frozen_manifest(config_sha256: str = "c" * 64) -> dict[str, object]:
    manifest = _manifest(1000, 200)
    manifest.update(
        {
            "candidates": sorted(RECALIBRATE.DEFAULT_BRIDGE_CANDIDATES),
            "candidate_token_ids": {
                candidate: index
                for index, candidate in enumerate(
                    sorted(RECALIBRATE.DEFAULT_BRIDGE_CANDIDATES)
                )
            },
            "canonical_layer": RECALIBRATE.EXPECTED_LAYER,
            "extraction_sha256": RECALIBRATE.EXPECTED_EXTRACTION_SHA256,
            "provenance": {"config_sha256": config_sha256},
        }
    )
    return manifest


def test_centering_uses_train_only_and_removes_constant_argmax() -> None:
    prior = {"amber": 10.0, "bamboo": 0.0, "copper": 0.0, "delta": 0.0}
    train = [
        _artifact(f"train-{index}", "train", candidate, prior)
        for index, candidate in enumerate(CANDIDATES)
    ]
    dev = []
    for index, candidate in enumerate(CANDIDATES):
        scores = dict(prior)
        scores[candidate] += 1.0
        dev.append(_artifact(f"dev-{index}", "dev", candidate, scores))

    summary = RECALIBRATE.build_summary(
        {"train": train, "dev": dev},
        _manifest(len(train), len(dev)),
        source={"fixture": True},
    )

    assert summary["candidate_means"]["jlens"] == prior
    assert summary["methods"]["jlens"]["raw_dev"]["metrics"][
        "top1_accuracy"
    ] == pytest.approx(0.25)
    assert summary["methods"]["jlens"]["centered_dev"]["metrics"][
        "top1_accuracy"
    ] == pytest.approx(1.0)
    distribution = summary["methods"]["jlens"]["centered_dev"][
        "prediction_distribution"
    ]
    assert distribution["max_prediction_share"] == pytest.approx(0.25)
    assert distribution["prediction_entropy_bits"] == pytest.approx(2.0)
    assert summary["gates"]["g_const"]["pass"] is True


def test_dev_values_cannot_change_train_means() -> None:
    train = [
        _artifact(
            "train-0",
            "train",
            "amber",
            {"amber": 4.0, "bamboo": 3.0, "copper": 2.0, "delta": 1.0},
        )
    ]
    first = RECALIBRATE.compute_train_means(train, CANDIDATES)
    dev = _artifact(
        "dev-0",
        "dev",
        "delta",
        {candidate: 1_000_000.0 for candidate in CANDIDATES},
    )
    RECALIBRATE.evaluate_dev(
        [dev],
        candidates=CANDIDATES,
        candidate_token_ids=TOKEN_IDS,
        layer=13,
        candidate_means=first,
    )
    second = RECALIBRATE.compute_train_means(train, CANDIDATES)
    assert second == first


def test_loader_never_discovers_test_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _artifact(
        "train-0",
        "train",
        "amber",
        {candidate: float(index) for index, candidate in enumerate(CANDIDATES)},
    )
    dev = _artifact(
        "dev-0",
        "dev",
        "bamboo",
        {candidate: float(index) for index, candidate in enumerate(CANDIDATES)},
    )
    payloads = {"train.pt": train, "dev.pt": dev}
    discovered: list[str] = []
    loader_candidates = tuple(f"candidate-{index}" for index in range(16))
    loader_manifest = _manifest(1, 1)
    loader_manifest["candidates"] = list(loader_candidates)
    loader_manifest["candidate_token_ids"] = {
        candidate: index for index, candidate in enumerate(loader_candidates)
    }

    monkeypatch.setattr(
        RECALIBRATE,
        "load_extraction_manifest",
        lambda _root: loader_manifest,
    )

    def fake_discover(_root: Path, split: str) -> tuple[Path, ...]:
        assert split != "test"
        discovered.append(split)
        return (Path(f"{split}.pt"),)

    monkeypatch.setattr(RECALIBRATE, "discover_split_artifacts", fake_discover)
    monkeypatch.setattr(
        RECALIBRATE,
        "load_artifact",
        lambda path, load_fn=None: payloads[path.name],
    )
    monkeypatch.setattr(
        RECALIBRATE,
        "_validate_artifact_contract",
        lambda *args, **kwargs: None,
    )

    artifacts, _ = RECALIBRATE.load_train_dev_artifacts("artifacts/dirty_v3")

    assert discovered == ["train", "dev"]
    assert set(artifacts) == {"train", "dev"}


def test_fail_closed_on_test_or_nonfinite_scores() -> None:
    test_artifact = _artifact(
        "test-0",
        "test",
        "amber",
        {candidate: 0.0 for candidate in CANDIDATES},
    )
    with pytest.raises(RECALIBRATE.RecalibrationError, match="train artifacts only"):
        RECALIBRATE.compute_train_means([test_artifact], CANDIDATES)

    invalid = _artifact(
        "train-0",
        "train",
        "amber",
        {"amber": float("nan"), "bamboo": 0.0, "copper": 0.0, "delta": 0.0},
    )
    with pytest.raises(RECALIBRATE.RecalibrationError, match="finite"):
        RECALIBRATE.compute_train_means([invalid], CANDIDATES)


def test_frozen_source_contract_rejects_wrong_config_and_counts() -> None:
    manifest = _frozen_manifest()
    RECALIBRATE.validate_v3_source_contract(manifest, config_sha256="c" * 64)

    with pytest.raises(RECALIBRATE.RecalibrationError, match="config hash"):
        RECALIBRATE.validate_v3_source_contract(
            manifest,
            config_sha256="d" * 64,
        )
    wrong_count = _frozen_manifest()
    wrong_count["expected_counts"]["dev"] = 199
    with pytest.raises(RECALIBRATE.RecalibrationError, match="dev count"):
        RECALIBRATE.validate_v3_source_contract(
            wrong_count,
            config_sha256="c" * 64,
        )


def test_output_paths_are_distinct_and_immutable(tmp_path: Path) -> None:
    shared = tmp_path / "shared.json"
    with pytest.raises(RECALIBRATE.RecalibrationError, match="distinct"):
        RECALIBRATE.validate_output_paths(shared, shared)

    output = tmp_path / "summary.json"
    manifest = tmp_path / "summary.manifest.json"
    RECALIBRATE.validate_output_paths(output, manifest)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(RECALIBRATE.RecalibrationError, match="overwrite"):
        RECALIBRATE.validate_output_paths(output, manifest)


def test_exact_checkpoint_equality_is_ambiguous() -> None:
    gate = RECALIBRATE.s0_checkpoint(0.25)
    assert gate["status"] == "ambiguous_equal_threshold"
    assert gate["short_circuit"] is None
    assert RECALIBRATE.s0_checkpoint(0.255)["short_circuit"] is True
    assert RECALIBRATE.s0_checkpoint(0.245)["short_circuit"] is False
