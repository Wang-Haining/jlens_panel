from pathlib import Path

import pytest

from jlens_panel.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_loads() -> None:
    config = load_config(ROOT / "config" / "dirty_run.yaml")

    assert config["data"]["candidate_count"] == 16
    assert config["experiment"]["conditions"] == [
        "generic",
        "jlens_targeted",
        "best_non_j",
        "oracle",
    ]


def test_sprint_config_loads_with_frozen_positions_and_gates() -> None:
    config = load_config(ROOT / "config" / "sprint.yaml")

    assert config["sweep"]["positions"] == [
        "template_tail",
        "content_last",
        "clue_last",
        "meanpool_content8",
        "decode_1",
        "decode_2",
        "decode_4",
        "decode_8",
    ]
    assert config["sweep"]["gates"] == {
        "g_info_min_probe_top1": 0.25,
        "g_const_max_label_share": 0.5,
        "g_const_min_entropy_bits": 1.5,
        "g_lens_min_probe_ratio": 0.5,
    }


def test_missing_sections_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("project: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing configuration sections"):
        load_config(path)


def test_unpinned_model_revision_fails(tmp_path: Path) -> None:
    source = (ROOT / "config" / "dirty_run.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        source.replace(
            "a09a35458c702b33eeacc393d103063234e8bc28",
            "main",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="model.revision"):
        load_config(path)
