from dataclasses import dataclass

import pytest

from jlens_panel.data.synthetic_bridge import (
    DEFAULT_BRIDGE_CANDIDATES,
    generate_dataset,
)
from jlens_panel.sweep.positions import (
    PositionResolutionError,
    PositionSelection,
    resolve_static_positions,
)

QWEN_CONTENT = "Notes:\n- Cedar clue ends here.\nQuestion: respond."
QWEN_GOLD_FACT = "Cedar clue ends here."
QWEN_RENDERED = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    f"{QWEN_CONTENT}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN_INPUT_IDS = (
    151644,
    8948,
    198,
    2610,
    525,
    1207,
    16948,
    11,
    3465,
    553,
    54364,
    14817,
    13,
    1446,
    525,
    264,
    10950,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
    21667,
    510,
    12,
    56648,
    29989,
    10335,
    1588,
    624,
    14582,
    25,
    5889,
    13,
    151645,
    198,
    151644,
    77091,
    198,
)
QWEN_OFFSETS = (
    (0, 12),
    (12, 18),
    (18, 19),
    (19, 22),
    (22, 26),
    (26, 28),
    (28, 31),
    (31, 32),
    (32, 40),
    (40, 43),
    (43, 51),
    (51, 57),
    (57, 58),
    (58, 62),
    (62, 66),
    (66, 68),
    (68, 76),
    (76, 86),
    (86, 87),
    (87, 97),
    (97, 98),
    (98, 110),
    (110, 114),
    (114, 115),
    (115, 120),
    (120, 122),
    (122, 123),
    (123, 129),
    (129, 134),
    (134, 139),
    (139, 144),
    (144, 146),
    (146, 154),
    (154, 155),
    (155, 163),
    (163, 164),
    (164, 174),
    (174, 175),
    (175, 187),
    (187, 196),
    (196, 197),
)


@dataclass(frozen=True)
class ExampleFixture:
    agent_a_prompt: str
    gold_agent_a_fact: str
    agent_a_facts: tuple[str, ...]


class FrozenTokenizer:
    def __init__(
        self,
        *,
        rendered: str,
        input_ids: tuple[int, ...],
        offsets: tuple[tuple[int, int], ...],
    ) -> None:
        self.rendered = rendered
        self.input_ids = input_ids
        self.offsets = offsets

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "add_special_tokens": False,
            "return_offsets_mapping": True,
            "truncation": False,
        }
        assert text == self.rendered
        return {"input_ids": self.input_ids, "offset_mapping": self.offsets}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text == "<|im_end|>"
        return [151645]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert messages == [{"role": "user", "content": QWEN_CONTENT}]
        assert not tokenize and add_generation_prompt
        return self.rendered


class CharacterTokenizer:
    marker_id = 999

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["return_offsets_mapping"] is True
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            if text.startswith("<|im_end|>", index):
                input_ids.append(self.marker_id)
                offsets.append((index, index + len("<|im_end|>")))
                index += len("<|im_end|>")
            else:
                input_ids.append(ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text == "<|im_end|>"
        return [self.marker_id]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        content = messages[0]["content"]
        return f"SYSTEM<|im_end|>\nUSER\n{content}<|im_end|>\nASSISTANT\n"


def character_fixture(
    *,
    content: str = "zero one two clue three four five six",
    fact: str = "clue",
    between_content_and_marker: str = "",
) -> tuple[CharacterTokenizer, ExampleFixture, str]:
    rendered = (
        "SYSTEM<|im_end|>\nUSER\n"
        f"{content}{between_content_and_marker}<|im_end|>\nASSISTANT\n"
    )
    example = ExampleFixture(
        agent_a_prompt=content,
        gold_agent_a_fact=fact,
        agent_a_facts=(fact, "distractor"),
    )
    return CharacterTokenizer(), example, rendered


def test_hand_tokenized_fixture_resolves_exact_indices_and_pool() -> None:
    tokenizer, example, rendered = character_fixture()

    resolved = resolve_static_positions(
        tokenizer,
        example,
        rendered,
        max_seq_len=512,
    )

    content_start = rendered.index(example.agent_a_prompt)
    content_end = content_start + len(example.agent_a_prompt)
    clue_end = rendered.index(example.gold_agent_a_fact) + len(
        example.gold_agent_a_fact
    )
    assert resolved.selection("template_tail").index == len(resolved.input_ids) - 1
    # One ten-character marker before the content is represented by one token.
    assert resolved.selection("content_last").index == content_end - 1 - 9
    assert resolved.selection("clue_last").index == clue_end - 1 - 9
    assert resolved.selection("meanpool_content8").token_indices == tuple(
        range(content_end - 8 - 9, content_end - 9)
    )
    assert content_start == 22


def test_pinned_qwen_chat_template_regression_fixture() -> None:
    tokenizer = FrozenTokenizer(
        rendered=QWEN_RENDERED,
        input_ids=QWEN_INPUT_IDS,
        offsets=QWEN_OFFSETS,
    )
    example = ExampleFixture(
        agent_a_prompt=QWEN_CONTENT,
        gold_agent_a_fact=QWEN_GOLD_FACT,
        agent_a_facts=(QWEN_GOLD_FACT, "a different fact"),
    )

    resolved = resolve_static_positions(
        tokenizer,
        example,
        QWEN_RENDERED,
        max_seq_len=512,
    )

    assert QWEN_RENDERED.endswith("<|im_start|>assistant\n")
    assert resolved.input_ids == QWEN_INPUT_IDS
    assert resolved.selection("template_tail").index == 40
    assert resolved.selection("content_last").index == 35
    assert resolved.selection("clue_last").index == 31
    assert resolved.selection("meanpool_content8").token_indices == tuple(range(28, 36))


@pytest.mark.parametrize(
    ("content", "fact", "between", "message"),
    [
        ("short", "short", "", "fewer than eight"),
        ("one clue two clue three four", "clue", "", "exactly once"),
        ("one two three four five six", "missing", "", "exactly once"),
    ],
)
def test_resolver_fails_closed_on_semantic_ambiguity(
    content: str,
    fact: str,
    between: str,
    message: str,
) -> None:
    tokenizer, example, rendered = character_fixture(
        content=content,
        fact=fact,
        between_content_and_marker=between,
    )

    with pytest.raises(PositionResolutionError, match=message):
        resolve_static_positions(
            tokenizer,
            example,
            rendered,
            max_seq_len=512,
        )


def test_resolver_rejects_render_that_is_not_native_chat_template() -> None:
    tokenizer, example, rendered = character_fixture()
    rendered = rendered.replace("<|im_end|>\nASSISTANT", "\n<|im_end|>\nASSISTANT")

    with pytest.raises(PositionResolutionError, match="native user chat template"):
        resolve_static_positions(
            tokenizer,
            example,
            rendered,
            max_seq_len=512,
        )


def test_resolver_rejects_malformed_offsets_and_length_limit() -> None:
    example = ExampleFixture(
        agent_a_prompt=QWEN_CONTENT,
        gold_agent_a_fact=QWEN_GOLD_FACT,
        agent_a_facts=(QWEN_GOLD_FACT,),
    )
    malformed = FrozenTokenizer(
        rendered=QWEN_RENDERED,
        input_ids=QWEN_INPUT_IDS,
        offsets=QWEN_OFFSETS[:-1],
    )
    with pytest.raises(PositionResolutionError, match="different lengths"):
        resolve_static_positions(
            malformed,
            example,
            QWEN_RENDERED,
            max_seq_len=512,
        )

    valid = FrozenTokenizer(
        rendered=QWEN_RENDERED,
        input_ids=QWEN_INPUT_IDS,
        offsets=QWEN_OFFSETS,
    )
    with pytest.raises(PositionResolutionError, match="above max_seq_len"):
        resolve_static_positions(
            valid,
            example,
            QWEN_RENDERED,
            max_seq_len=40,
        )


def test_pooled_selection_has_no_single_index() -> None:
    selection = PositionSelection(
        "meanpool_content8",
        tuple(range(8)),
        "mean",
    )
    with pytest.raises(PositionResolutionError, match="pooled position"):
        _ = selection.index


def test_decode_positions_share_the_typed_selection_contract() -> None:
    assert PositionSelection("decode_4", (7,), "select").index == 7


def test_real_synthetic_example_derives_and_resolves_gold_fact() -> None:
    example = generate_dataset(
        candidates=DEFAULT_BRIDGE_CANDIDATES,
        seed=17,
        train_size=16,
        dev_size=16,
        test_size=16,
    )["train"][0]
    tokenizer = CharacterTokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": example.agent_a_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    resolved = resolve_static_positions(
        tokenizer,
        example,
        rendered,
        max_seq_len=10_000,
    )

    assert example.gold_agent_a_fact in example.agent_a_facts
    clue_index = resolved.selection("clue_last").index
    assert rendered[
        resolved.offsets[clue_index][0] : resolved.offsets[clue_index][1]
    ] == (example.gold_agent_a_fact[-1])
