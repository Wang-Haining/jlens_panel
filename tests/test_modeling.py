import pytest

from jlens_panel.modeling import canonical_layer, resolve_candidate_token_ids


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.mapping.get(text, [99, 100])


def test_candidate_tokens_prefer_leading_space() -> None:
    tokenizer = FakeTokenizer({" Alpha": [1], "Alpha": [2], " Beta": [3]})

    assert resolve_candidate_token_ids(tokenizer, ["Alpha", "Beta"]) == {
        "Alpha": 1,
        "Beta": 3,
    }


def test_candidate_tokens_reject_duplicates() -> None:
    tokenizer = FakeTokenizer({" Alpha": [1], " Beta": [1]})

    with pytest.raises(ValueError, match="duplicate"):
        resolve_candidate_token_ids(tokenizer, ["Alpha", "Beta"])


def test_canonical_layer_uses_middle_of_middle_third() -> None:
    assert canonical_layer(list(range(30)), "middle_of_middle_third") == 15
