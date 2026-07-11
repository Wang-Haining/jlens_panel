from jlens_panel.upstream_eval import pass_at


def test_pass_at_is_item_macro_average_and_keeps_missing_tokens() -> None:
    records = [
        {"item": "one", "jlens_rank": 1},
        {"item": "one", "jlens_rank": None},
        {"item": "two", "jlens_rank": 2},
    ]

    assert pass_at(records, "jlens_rank", 1) == 0.25
    assert pass_at(records, "jlens_rank", 2) == 0.75
