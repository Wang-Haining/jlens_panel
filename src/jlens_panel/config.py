"""Configuration loading with lightweight structural validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a dirty-run configuration is incomplete or invalid."""


_REQUIRED_SECTIONS = {
    "project",
    "model",
    "lens",
    "data",
    "readouts",
    "experiment",
    "gates",
    "storage",
}
_READOUT_METHODS = {
    "jlens",
    "logit_lens",
    "raw_probe",
    "next_token",
    "text_only",
}
_CONDITIONS = {"generic", "jlens_targeted", "best_non_j", "oracle"}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be numeric") from error
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be between zero and one")
    return number


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its top-level contract."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")

    missing = sorted(_REQUIRED_SECTIONS - value.keys())
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    model = _mapping(value["model"], "model")
    for key in ("name", "revision", "dtype", "device_map"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ConfigError(f"model.{key} must be a non-empty string")
    if re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is None:
        raise ConfigError("model.revision must be a pinned 40-character commit")

    lens = _mapping(value["lens"], "lens")
    if re.fullmatch(r"[0-9a-f]{40}", str(lens.get("upstream_commit", ""))) is None:
        raise ConfigError("lens.upstream_commit must be a pinned Git commit")
    _positive_integer(lens.get("fit_prompts"), "lens.fit_prompts")
    _positive_integer(lens.get("checkpoint_every"), "lens.checkpoint_every")

    data = _mapping(value["data"], "data")
    if _positive_integer(data.get("candidate_count"), "data.candidate_count") != 16:
        raise ConfigError("data.candidate_count must be exactly 16 for the dirty run")
    for key in ("train_size", "dev_size", "test_size", "smoke_size"):
        _positive_integer(data.get(key), f"data.{key}")

    readouts = _mapping(value["readouts"], "readouts")
    methods = readouts.get("methods")
    if not isinstance(methods, list) or set(methods) != _READOUT_METHODS:
        raise ConfigError("readouts.methods must contain the five fixed readouts")
    if len(methods) != len(set(methods)):
        raise ConfigError("readouts.methods must not contain duplicates")

    experiment = _mapping(value["experiment"], "experiment")
    conditions = experiment.get("conditions")
    if not isinstance(conditions, list) or set(conditions) != _CONDITIONS:
        raise ConfigError("experiment.conditions must contain the four fixed branches")
    if len(conditions) != len(set(conditions)):
        raise ConfigError("experiment.conditions must not contain duplicates")
    seeds = experiment.get("generation_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ConfigError("experiment.generation_seeds must be unique integers")
    _unit_interval(experiment.get("sender_temperature"), "sender_temperature")
    _unit_interval(experiment.get("receiver_temperature"), "receiver_temperature")
    _unit_interval(experiment.get("top_p"), "top_p")

    gates = _mapping(value["gates"], "gates")
    for key in (
        "min_sender_bridge_accuracy",
        "min_oracle_accuracy_lift",
        "min_targeted_accuracy_lift",
    ):
        _unit_interval(gates.get(key), f"gates.{key}")
    _positive_integer(gates.get("min_omitted_items"), "gates.min_omitted_items")

    storage = _mapping(value["storage"], "storage")
    _positive_integer(storage.get("disk_check_interval"), "storage.disk_check_interval")
    _unit_interval(
        storage.get("filesystem_warning_fraction"),
        "storage.filesystem_warning_fraction",
    )

    return value
