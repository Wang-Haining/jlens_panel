from pathlib import Path

from jlens_panel.io import append_jsonl, completed_keys, read_jsonl


def test_append_and_resume_keys(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    append_jsonl(path, {"item_id": "a", "seed": 1, "value": 2})
    append_jsonl(path, {"item_id": "b", "seed": 1, "value": 3})

    assert list(read_jsonl(path))[1]["value"] == 3
    assert completed_keys(path, ["item_id", "seed"]) == {("a", 1), ("b", 1)}
