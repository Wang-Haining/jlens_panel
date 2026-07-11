"""Disk-space guardrails for local and Tempest runs."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


class DiskGuardError(RuntimeError):
    """Raised before a run would violate a storage safety threshold."""


@dataclass(frozen=True)
class DiskStatus:
    """A point-in-time disk and project-size measurement."""

    total_bytes: int
    used_bytes: int
    free_bytes: int
    project_bytes: int

    @property
    def used_fraction(self) -> float:
        """Return filesystem utilization as a fraction in [0, 1]."""

        return self.used_bytes / self.total_bytes if self.total_bytes else 1.0


def directory_size(path: str | Path) -> int:
    """Return the total size of regular files below path."""

    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def inspect_disk(project_root: str | Path) -> DiskStatus:
    """Measure the filesystem containing project_root and project usage."""

    root = Path(project_root).resolve()
    usage = shutil.disk_usage(root)
    return DiskStatus(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        project_bytes=directory_size(root),
    )


def enforce_disk_guard(
    status: DiskStatus,
    *,
    minimum_free_bytes: int,
    maximum_used_fraction: float,
    maximum_project_bytes: int,
) -> None:
    """Fail closed when any configured disk threshold is crossed."""

    violations: list[str] = []
    if status.free_bytes < minimum_free_bytes:
        violations.append(
            f"free bytes {status.free_bytes} below minimum {minimum_free_bytes}"
        )
    if status.used_fraction >= maximum_used_fraction:
        violations.append(
            f"used fraction {status.used_fraction:.3f} at/above "
            f"{maximum_used_fraction:.3f}"
        )
    if status.project_bytes >= maximum_project_bytes:
        violations.append(
            f"project bytes {status.project_bytes} at/above {maximum_project_bytes}"
        )
    if violations:
        raise DiskGuardError("; ".join(violations))


def build_disk_guard(
    config: Mapping[str, object],
    *,
    project_root: str | Path,
    environment: str,
) -> Callable[[], None]:
    """Build the shared local/Tempest periodic disk check."""

    if environment not in ("local", "tempest"):
        raise DiskGuardError(f"unknown disk-guard environment: {environment!r}")
    storage = config.get("storage")
    if not isinstance(storage, Mapping):
        raise DiskGuardError("configuration storage section must be a mapping")
    try:
        if environment == "tempest":
            minimum_free = int(float(storage["tempest_minimum_free_tb"]) * 10**12)
        else:
            minimum_free = int(float(storage["local_minimum_free_gb"]) * 10**9)
        maximum_used_fraction = float(storage["filesystem_warning_fraction"])
        maximum_project_bytes = int(float(storage["project_warning_gb"]) * 10**9)
    except (KeyError, TypeError, ValueError) as error:
        raise DiskGuardError("configuration storage thresholds are invalid") from error

    def check() -> None:
        enforce_disk_guard(
            inspect_disk(project_root),
            minimum_free_bytes=minimum_free,
            maximum_used_fraction=maximum_used_fraction,
            maximum_project_bytes=maximum_project_bytes,
        )

    return check
