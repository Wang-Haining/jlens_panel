"""Pure, fail-closed token-position resolution for the v3 sprint sweep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

STATIC_POSITION_NAMES = (
    "template_tail",
    "content_last",
    "clue_last",
    "meanpool_content8",
)
DECODE_STEPS = (1, 2, 4, 8)
DECODE_POSITION_NAMES = tuple(f"decode_{step}" for step in DECODE_STEPS)
ALL_POSITION_NAMES = STATIC_POSITION_NAMES + DECODE_POSITION_NAMES
USER_END_MARKER = "<|im_end|>"
Reduction = Literal["select", "mean"]


class PositionResolutionError(ValueError):
    """Raised when a requested semantic position cannot be resolved uniquely."""


class PositionExample(Protocol):
    """Structured fields needed to resolve task positions without model imports."""

    agent_a_prompt: str
    agent_a_facts: Sequence[str]

    @property
    def gold_agent_a_fact(self) -> str:
        """Return the exact gold fact embedded in ``agent_a_prompt``."""


@dataclass(frozen=True, slots=True)
class PositionSelection:
    """One singleton or mean-pooled selection over encoded prompt positions."""

    name: str
    token_indices: tuple[int, ...]
    reduction: Reduction

    def __post_init__(self) -> None:
        indices = tuple(self.token_indices)
        if self.name not in ALL_POSITION_NAMES:
            raise PositionResolutionError(f"unknown position: {self.name!r}")
        if self.reduction not in ("select", "mean"):
            raise PositionResolutionError(f"unknown position reduction: {self.reduction!r}")
        if not indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise PositionResolutionError("token indices must be non-negative integers")
        if len(indices) != len(set(indices)) or tuple(sorted(indices)) != indices:
            raise PositionResolutionError("token indices must be unique and increasing")
        if self.reduction == "select" and len(indices) != 1:
            raise PositionResolutionError("select positions require exactly one index")
        if self.name == "meanpool_content8":
            if self.reduction != "mean" or len(indices) != 8:
                raise PositionResolutionError(
                    "meanpool_content8 requires exactly eight mean-pooled indices"
                )
        elif self.reduction != "select":
            raise PositionResolutionError(
                f"singleton position {self.name!r} must use select reduction"
            )
        object.__setattr__(self, "token_indices", indices)

    @property
    def index(self) -> int:
        """Return a singleton index and reject pooled selections."""

        if self.reduction != "select":
            raise PositionResolutionError(f"{self.name} is a pooled position")
        return self.token_indices[0]


@dataclass(frozen=True, slots=True)
class ResolvedPositions:
    """Encoded IDs, offsets, and every static position from the same tokenization."""

    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    selections: tuple[PositionSelection, ...]

    def __post_init__(self) -> None:
        input_ids = tuple(self.input_ids)
        offsets = tuple(tuple(offset) for offset in self.offsets)
        selections = tuple(self.selections)
        if not input_ids or len(input_ids) != len(offsets):
            raise PositionResolutionError("input IDs and offsets must be non-empty peers")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in input_ids
        ):
            raise PositionResolutionError("input IDs must be non-negative integers")
        if tuple(selection.name for selection in selections) != STATIC_POSITION_NAMES:
            raise PositionResolutionError("static position inventory or order changed")
        if any(
            index >= len(input_ids)
            for selection in selections
            for index in selection.token_indices
        ):
            raise PositionResolutionError("resolved token index is out of range")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "selections", selections)

    def selection(self, name: str) -> PositionSelection:
        """Return one known static selection or fail closed."""

        for selection in self.selections:
            if selection.name == name:
                return selection
        raise PositionResolutionError(f"unknown static position: {name!r}")


def _flat_integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PositionResolutionError(f"tokenizer {name} must be a flat sequence")
    flattened = tuple(value)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in flattened
    ):
        raise PositionResolutionError(
            f"tokenizer {name} must contain non-negative integers"
        )
    return flattened


def _validated_offsets(value: object, *, text_length: int) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PositionResolutionError("tokenizer offsets must be a flat sequence")
    offsets: list[tuple[int, int]] = []
    last_nonempty_end = 0
    for raw_offset in value:
        if (
            isinstance(raw_offset, (str, bytes))
            or not isinstance(raw_offset, Sequence)
            or len(raw_offset) != 2
        ):
            raise PositionResolutionError("each tokenizer offset must be a pair")
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > text_length
        ):
            raise PositionResolutionError("tokenizer offsets are out of bounds")
        if end > start:
            if start < last_nonempty_end:
                raise PositionResolutionError("tokenizer offsets are not monotone")
            last_nonempty_end = end
        offsets.append((start, end))
    return tuple(offsets)


def _unique_span(text: str, needle: str, *, name: str) -> tuple[int, int]:
    if not isinstance(needle, str) or not needle:
        raise PositionResolutionError(f"{name} must be a non-empty string")
    starts: list[int] = []
    search_from = 0
    while True:
        start = text.find(needle, search_from)
        if start < 0:
            break
        starts.append(start)
        search_from = start + 1
    if len(starts) != 1:
        raise PositionResolutionError(
            f"{name} must occur exactly once in the rendered prompt; found {len(starts)}"
        )
    return starts[0], starts[0] + len(needle)


def _overlapping_token_indices(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
    *,
    name: str,
) -> tuple[int, ...]:
    span_start, span_end = span
    indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span_end and end > span_start
    )
    if not indices:
        raise PositionResolutionError(f"{name} span has no encoded token")
    return indices


def _tokenizer_mapping(tokenizer: object, rendered_prompt: str) -> Mapping[str, object]:
    try:
        encoded = tokenizer(  # type: ignore[operator]
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        raise PositionResolutionError(
            "tokenizer must provide a fast offset mapping"
        ) from error
    if not isinstance(encoded, Mapping):
        raise PositionResolutionError("tokenizer output must be a mapping")
    return encoded


def resolve_static_positions(
    tokenizer: object,
    example: PositionExample,
    rendered_prompt: str,
    *,
    max_seq_len: int,
) -> ResolvedPositions:
    """Resolve all static cells from one exact, untruncated tokenization."""

    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise PositionResolutionError("rendered prompt must be non-empty")
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len < 1
    ):
        raise PositionResolutionError("max_seq_len must be a positive integer")
    content = example.agent_a_prompt
    try:
        gold_fact = example.gold_agent_a_fact
    except (AttributeError, ValueError) as error:
        raise PositionResolutionError("cannot derive the gold Agent A fact") from error
    if tuple(example.agent_a_facts).count(gold_fact) != 1:
        raise PositionResolutionError("gold Agent A fact is not unique in structured facts")

    try:
        expected_render = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise PositionResolutionError(
            "tokenizer cannot verify the native user chat template"
        ) from error
    if not isinstance(expected_render, str) or rendered_prompt != expected_render:
        raise PositionResolutionError(
            "rendered prompt does not match the native user chat template"
        )

    encoded = _tokenizer_mapping(tokenizer, rendered_prompt)
    input_ids = _flat_integer_sequence(encoded.get("input_ids"), name="input_ids")
    if not input_ids:
        raise PositionResolutionError("rendered prompt encoded to no tokens")
    if len(input_ids) > max_seq_len:
        raise PositionResolutionError(
            f"rendered prompt has {len(input_ids)} tokens above max_seq_len={max_seq_len}"
        )
    offsets = _validated_offsets(
        encoded.get("offset_mapping"),
        text_length=len(rendered_prompt),
    )
    if len(offsets) != len(input_ids):
        raise PositionResolutionError("input IDs and offsets have different lengths")

    content_span = _unique_span(rendered_prompt, content, name="raw user content")
    clue_span = _unique_span(rendered_prompt, gold_fact, name="gold Agent A fact")
    if not (
        content_span[0] <= clue_span[0]
        and clue_span[1] <= content_span[1]
    ):
        raise PositionResolutionError("gold Agent A fact lies outside user content")
    marker_start = rendered_prompt.find(USER_END_MARKER, content_span[1])
    if marker_start < 0:
        raise PositionResolutionError("user-closing <|im_end|> is missing")
    if marker_start != content_span[1]:
        raise PositionResolutionError(
            "unexpected text occurs between user content and closing <|im_end|>"
        )

    content_indices = _overlapping_token_indices(
        offsets,
        content_span,
        name="raw user content",
    )
    clue_indices = _overlapping_token_indices(
        offsets,
        clue_span,
        name="gold Agent A fact",
    )
    if len(content_indices) < 8:
        raise PositionResolutionError(
            "raw user content has fewer than eight encoded tokens"
        )

    try:
        marker_ids = tokenizer.encode(  # type: ignore[attr-defined]
            USER_END_MARKER,
            add_special_tokens=False,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise PositionResolutionError("tokenizer cannot encode the user-close marker") from error
    marker_token_ids = _flat_integer_sequence(marker_ids, name="marker input_ids")
    if len(marker_token_ids) != 1:
        raise PositionResolutionError("<|im_end|> must encode to exactly one token")
    content_last = content_indices[-1]
    marker_index = content_last + 1
    if marker_index >= len(input_ids) or input_ids[marker_index] != marker_token_ids[0]:
        raise PositionResolutionError(
            "content_last is not immediately before the user-close marker"
        )
    marker_end = marker_start + len(USER_END_MARKER)
    if offsets[marker_index] != (marker_start, marker_end):
        raise PositionResolutionError("user-close marker offset is not exact")
    content_offset = offsets[content_last]
    if not content_offset[0] <= content_span[1] - 1 < content_offset[1]:
        raise PositionResolutionError(
            "content_last token does not contain the content's final character"
        )
    clue_last = clue_indices[-1]
    clue_offset = offsets[clue_last]
    if not clue_offset[0] <= clue_span[1] - 1 < clue_offset[1]:
        raise PositionResolutionError(
            "clue_last token does not contain the fact's final character"
        )

    selections = (
        PositionSelection("template_tail", (len(input_ids) - 1,), "select"),
        PositionSelection("content_last", (content_last,), "select"),
        PositionSelection("clue_last", (clue_last,), "select"),
        PositionSelection(
            "meanpool_content8",
            content_indices[-8:],
            "mean",
        ),
    )
    return ResolvedPositions(
        input_ids=input_ids,
        offsets=offsets,
        selections=selections,
    )
