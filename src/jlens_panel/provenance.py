"""Immutable run manifests and file provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(cwd: str | Path) -> str | None:
    """Return the current Git revision, or None in a pre-commit repository."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(cwd: str | Path) -> bool | None:
    """Return whether tracked/untracked files differ, or None outside Git."""

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout) if result.returncode == 0 else None


def installed_versions() -> dict[str, str | None]:
    """Record relevant package versions without importing heavyweight modules."""

    versions: dict[str, str | None] = {}
    for distribution in (
        "jlens",
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def package_source_fingerprint(package: str) -> str:
    """Hash a package's Python sources without importing the package."""

    if not package or not package.replace("_", "a").isalnum():
        raise ValueError("package name must be a simple Python identifier")
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        raise ValueError(f"cannot locate package sources: {package}")
    origin = Path(spec.origin).resolve()
    root = origin.parent if origin.name == "__init__.py" else origin
    files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    if not files:
        raise ValueError(f"package has no Python sources: {package}")
    digest = hashlib.sha256()
    for source in files:
        relative = source.relative_to(root).as_posix() if root.is_dir() else source.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    *,
    config_path: str | Path,
    project_root: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for one immutable run."""

    config = Path(config_path)
    root = Path(project_root)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "git_revision": git_revision(root),
        "git_is_dirty": git_is_dirty(root),
        "hostname": platform.node(),
        "packages": installed_versions(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Atomically write JSON so interrupted jobs never leave partial state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(output)
