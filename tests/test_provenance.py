import json
from pathlib import Path

from jlens_panel.provenance import (
    installed_versions,
    package_source_fingerprint,
    sha256_file,
    write_json_atomic,
)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("silent committee\n", encoding="utf-8")

    assert sha256_file(path) == (
        "4da3c3efa98d09c40aa49e72e4409af83750df99a33d02762151ce23af772a5b"
    )


def test_write_json_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_json_atomic(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_installed_versions_has_fixed_keys() -> None:
    versions = installed_versions()

    assert set(versions) == {
        "jlens",
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    }


def test_package_source_fingerprint_is_stable_without_importing_package() -> None:
    first = package_source_fingerprint("jlens_panel")
    second = package_source_fingerprint("jlens_panel")

    assert first == second
    assert len(first) == 64
