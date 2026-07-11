import pytest

from jlens_panel.storage import (
    DiskGuardError,
    DiskStatus,
    build_disk_guard,
    enforce_disk_guard,
)


def test_disk_guard_accepts_safe_status() -> None:
    enforce_disk_guard(
        DiskStatus(
            total_bytes=100,
            used_bytes=50,
            free_bytes=50,
            project_bytes=5,
        ),
        minimum_free_bytes=20,
        maximum_used_fraction=0.8,
        maximum_project_bytes=10,
    )


def test_disk_guard_reports_all_violations() -> None:
    with pytest.raises(DiskGuardError) as error:
        enforce_disk_guard(
            DiskStatus(
                total_bytes=100,
                used_bytes=90,
                free_bytes=10,
                project_bytes=20,
            ),
            minimum_free_bytes=20,
            maximum_used_fraction=0.8,
            maximum_project_bytes=10,
        )

    message = str(error.value)
    assert "free bytes" in message
    assert "used fraction" in message
    assert "project bytes" in message


def test_build_disk_guard_rejects_unknown_environment() -> None:
    with pytest.raises(DiskGuardError, match="unknown.*environment"):
        build_disk_guard({}, project_root=".", environment="cluster")
