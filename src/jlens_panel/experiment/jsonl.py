"""Append-only JSONL persistence and resume support for experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from .core import Condition, InitialMessage, OutcomeRecord, outcome_resume_key

try:  # pragma: no cover - Windows fallback; Tempest and macOS provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class JsonlStoreError(RuntimeError):
    """Raised when an append-only result file violates its schema."""


class JsonlResultStore:
    """A duplicate-safe append-only store keyed by explicit resume keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def resume_keys(self) -> set[str]:
        """Return all completed common-message and outcome keys."""

        return {self._resume_key(value, line) for line, value in self._records()}

    def initial_message_for(self, item_id: str, seed: int) -> InitialMessage | None:
        """Load the stored common message for an item/seed pair, if present."""

        expected = json.dumps(
            ["initial_message", item_id, int(seed)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        matches = [
            InitialMessage.from_dict(value)
            for _, value in self._records()
            if value.get("resume_key") == expected
        ]
        return self._one_or_none(matches, expected)

    def get_or_append_initial(self, message: InitialMessage) -> InitialMessage:
        """Persist a common message once and return the winning stored value."""

        self._append_if_missing(message.to_dict())
        stored = self.initial_message_for(message.item_id, message.seed)
        if stored is None:  # pragma: no cover - defensive disk invariant
            raise JsonlStoreError("initial message disappeared after append")
        return stored

    def append_outcome(self, outcome: OutcomeRecord) -> bool:
        """Append an outcome unless its branch resume key already exists."""

        return self._append_if_missing(outcome.to_dict())

    def outcome_for(
        self,
        item_id: str,
        seed: int,
        condition: Condition | str,
    ) -> OutcomeRecord | None:
        """Load one stored branch outcome, if present."""

        expected = outcome_resume_key(item_id, seed, condition)
        matches = [
            OutcomeRecord.from_dict(value)
            for _, value in self._records()
            if value.get("resume_key") == expected
        ]
        return self._one_or_none(matches, expected)

    def outcomes_for(self, item_id: str, seed: int) -> list[OutcomeRecord]:
        """Load all stored branch outcomes for one item/seed pair."""

        outcomes = [
            OutcomeRecord.from_dict(value)
            for _, value in self._records()
            if value.get("record_type") == "outcome"
            and value.get("item_id") == item_id
            and value.get("seed") == seed
        ]
        keys = [outcome.resume_key for outcome in outcomes]
        if len(keys) != len(set(keys)):
            raise JsonlStoreError(
                f"duplicate outcome key for item {item_id!r}, seed {seed}"
            )
        return outcomes

    def all_outcomes(self) -> list[OutcomeRecord]:
        """Load every outcome, ignoring common-message records."""

        outcomes = [
            OutcomeRecord.from_dict(value)
            for _, value in self._records()
            if value.get("record_type") == "outcome"
        ]
        keys = [outcome.resume_key for outcome in outcomes]
        if len(keys) != len(set(keys)):
            raise JsonlStoreError("JSONL file contains duplicate outcome resume keys")
        return outcomes

    def _append_if_missing(self, value: dict[str, object]) -> bool:
        key = value.get("resume_key")
        if not isinstance(key, str) or not key:
            raise JsonlStoreError("record must have a non-empty string resume_key")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            _lock(handle, exclusive=True)
            try:
                handle.seek(0)
                existing = {
                    self._resume_key(record, line)
                    for line, record in _records_from_handle(handle, self.path)
                }
                if key in existing:
                    return False
                handle.seek(0, os.SEEK_END)
                handle.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                return True
            finally:
                _unlock(handle)

    def _records(self) -> Iterator[tuple[int, dict[str, Any]]]:
        if not self.path.exists():
            return iter(())

        def iterator() -> Iterator[tuple[int, dict[str, Any]]]:
            with self.path.open(encoding="utf-8") as handle:
                _lock(handle, exclusive=False)
                try:
                    yield from _records_from_handle(handle, self.path)
                finally:
                    _unlock(handle)

        return iterator()

    @staticmethod
    def _resume_key(value: dict[str, Any], line: int) -> str:
        key = value.get("resume_key")
        if not isinstance(key, str) or not key:
            raise JsonlStoreError(f"line {line} has no valid resume_key")
        return key

    @staticmethod
    def _one_or_none(values: list[Any], key: str) -> Any:
        if len(values) > 1:
            raise JsonlStoreError(f"duplicate JSONL resume key: {key}")
        return values[0] if values else None


def _records_from_handle(
    handle: Any, path: Path
) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise JsonlStoreError(
                f"invalid JSON at {path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise JsonlStoreError(
                f"JSONL record at {path}:{line_number} is not an object"
            )
        yield line_number, value


def _lock(handle: Any, *, exclusive: bool) -> None:
    if fcntl is not None:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), operation)


def _unlock(handle: Any) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
