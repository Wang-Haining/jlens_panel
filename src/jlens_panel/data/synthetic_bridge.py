"""Deterministic two-agent synthetic bridge-concept data.

Each example contains a derived relational chain deliberately split across two
agents::

    source entity --Agent A clue/reasoning--> bridge concept
    bridge concept --Agent B relation--> final answer

No candidate bridge string appears in Agent A's visible text: A must derive the
concept from a split-specific clue. Agent B sees a separate literal relation
table and must use Agent A's message to recover the final answer. Six to ten
structurally matched distractor chains are included in both agents' contexts.

The train, development, and test splits use disjoint surface templates and
non-label entity families. The 16 bridge labels are intentionally shared across
splits because they define the fixed classification space. This module uses only
the Python standard library so importing data schemas never loads ML runtimes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = ("train", "dev", "test")
CANDIDATE_COUNT = 16
MIN_DISTRACTORS = 6
MAX_DISTRACTORS = 10
SCHEMA_VERSION = "synthetic-bridge-v2"
AGENT_A_MESSAGE_PLACEHOLDER = "{agent_a_message}"

DEFAULT_BRIDGE_CANDIDATES: tuple[str, ...] = (
    "amber",
    "bamboo",
    "copper",
    "delta",
    "desert",
    "eagle",
    "elm",
    "frost",
    "harbor",
    "iris",
    "lemon",
    "maple",
    "marble",
    "ocean",
    "tiger",
    "violin",
)

BRIDGE_CLUES: dict[str, dict[Split, str]] = {
    "amber": {
        "train": "fossilized tree resin often used in jewelry",
        "dev": "the golden organic gemstone formed from ancient sap",
        "test": "hardened prehistoric sap that can preserve trapped insects",
    },
    "bamboo": {
        "train": "the giant woody grass eaten by pandas",
        "dev": "the fast-growing grass with hollow jointed stalks",
        "test": "the tall segmented grass used for scaffolding and panda food",
    },
    "copper": {
        "train": "the chemical element with atomic number 29",
        "dev": "the reddish conductive metal represented by the symbol Cu",
        "test": "the metal commonly used in electrical wiring and bronze",
    },
    "delta": {
        "train": "the triangular sediment deposit where a river meets the sea",
        "dev": "the branching river-mouth landform built from silt",
        "test": "the low alluvial landform at a river outlet",
    },
    "elm": {
        "train": "the shade tree devastated by a disease named for the Dutch",
        "dev": "the tall deciduous tree with serrated leaves and winged seeds",
        "test": "a street-tree hardwood genus devastated by a fungal epidemic",
    },
    "frost": {
        "train": "ice crystals formed from water vapor on a cold surface",
        "dev": "the thin frozen coating seen on grass before sunrise",
        "test": "a surface deposit created when airborne moisture freezes",
    },
    "desert": {
        "train": "an arid landscape that receives very little rainfall",
        "dev": "the dry biome known for sparse vegetation and dunes",
        "test": "the extremely dry environment exemplified by the Sahara",
    },
    "eagle": {
        "train": "the large bird of prey used as a United States emblem",
        "dev": "a soaring raptor with powerful talons and keen eyesight",
        "test": "the broad-winged predatory bird that builds a large eyrie",
    },
    "harbor": {
        "train": "a sheltered coastal place where ships can anchor",
        "dev": "protected water beside a coast used by vessels",
        "test": "a safe anchorage for boats near shore",
    },
    "iris": {
        "train": "the colored ring surrounding the pupil of an eye",
        "dev": "the eye structure that controls how much light enters",
        "test": "the pigmented circular membrane around the pupil",
    },
    "lemon": {
        "train": "the sour yellow citrus fruit rich in acid",
        "dev": "the bright yellow fruit whose juice has a sharp tart taste",
        "test": "the citrus fruit commonly paired with sugar in a tart drink",
    },
    "maple": {
        "train": "the tree whose leaf appears on the Canadian flag",
        "dev": "the tree tapped for sap that is boiled into sweet syrup",
        "test": "the deciduous tree known for winged seeds and sugary sap",
    },
    "marble": {
        "train": "the metamorphic rock formed when limestone recrystallizes",
        "dev": "the polished stone used in classical statues and columns",
        "test": "a veined metamorphic stone common in sculpture",
    },
    "ocean": {
        "train": "a vast body of salt water covering most of Earth",
        "dev": "one of Earth's five major divisions of salt water",
        "test": "the great saltwater expanse separating continents",
    },
    "tiger": {
        "train": "the orange big cat marked with black stripes",
        "dev": "the largest living feline, recognizable by vertical stripes",
        "test": "the striped Asian predator in the big-cat family",
    },
    "violin": {
        "train": "the small bowed string instrument held under the chin",
        "dev": "the four-string orchestral instrument played with a bow",
        "test": "the high-pitched member of the bowed string family",
    },
}


class BridgeDataError(ValueError):
    """Raised when synthetic bridge data violates its experimental contract."""


@dataclass(frozen=True)
class TemplateFamily:
    """Split-specific surface realization for the two relational hops."""

    name: str
    agent_a_fact: str
    agent_b_fact: str
    public_question: str
    agent_a_probe_question: str
    agent_b_question: str

    def render_agent_a_fact(self, source: str, bridge: str) -> str:
        return self.agent_a_fact.format(source=source, bridge=bridge)

    def render_agent_b_fact(self, bridge: str, answer: str) -> str:
        return self.agent_b_fact.format(bridge=bridge, answer=answer)

    def render_public_question(self, source: str) -> str:
        return self.public_question.format(source=source)

    def render_agent_a_probe_question(self, source: str) -> str:
        return self.agent_a_probe_question.format(source=source)


@dataclass(frozen=True)
class EntityFamily:
    """Naming vocabulary reserved for a single split."""

    name: str
    source_prefix: str
    answer_prefix: str


@dataclass(frozen=True)
class SplitResources:
    """Template and entity resources that must never cross split boundaries."""

    templates: tuple[TemplateFamily, ...]
    entities: tuple[EntityFamily, ...]


@dataclass(frozen=True)
class BridgeChain:
    """One source-to-bridge-to-answer relational chain."""

    source_entity: str
    bridge_concept: str
    final_answer: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_entity": self.source_entity,
            "bridge_concept": self.bridge_concept,
            "final_answer": self.final_answer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BridgeChain":
        required = {"source_entity", "bridge_concept", "final_answer"}
        if set(value) != required:
            raise BridgeDataError(
                "bridge chain keys must be exactly " + ", ".join(sorted(required))
            )
        fields = {key: value[key] for key in required}
        if not all(isinstance(item, str) and item for item in fields.values()):
            raise BridgeDataError("bridge chain values must be non-empty strings")
        return cls(
            source_entity=fields["source_entity"],
            bridge_concept=fields["bridge_concept"],
            final_answer=fields["final_answer"],
        )


@dataclass(frozen=True)
class SyntheticBridgeExample:
    """A fully rendered, serializable two-agent bridge example."""

    example_id: str
    split: Split
    generation_seed: int
    template_family: str
    entity_family: str
    candidate_bridges: tuple[str, ...]
    source_entity: str
    gold_bridge: str
    final_answer: str
    distractor_chains: tuple[BridgeChain, ...]
    agent_a_facts: tuple[str, ...]
    agent_b_facts: tuple[str, ...]
    agent_a_prompt: str
    agent_a_probe_prompt: str
    agent_b_prompt_template: str
    schema_version: str = SCHEMA_VERSION

    @property
    def gold_chain(self) -> BridgeChain:
        """Return the explicit gold relational chain."""

        return BridgeChain(
            source_entity=self.source_entity,
            bridge_concept=self.gold_bridge,
            final_answer=self.final_answer,
        )

    @property
    def distractor_count(self) -> int:
        return len(self.distractor_chains)

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "split": self.split,
            "generation_seed": self.generation_seed,
            "template_family": self.template_family,
            "entity_family": self.entity_family,
            "candidate_bridges": list(self.candidate_bridges),
            "source_entity": self.source_entity,
            "gold_bridge": self.gold_bridge,
            "final_answer": self.final_answer,
            "gold_chain": self.gold_chain.to_dict(),
            "distractor_count": self.distractor_count,
            "distractor_chains": [chain.to_dict() for chain in self.distractor_chains],
            "agent_a_facts": list(self.agent_a_facts),
            "agent_b_facts": list(self.agent_b_facts),
            "agent_a_prompt": self.agent_a_prompt,
            "agent_a_probe_prompt": self.agent_a_probe_prompt,
            "agent_b_prompt_template": self.agent_b_prompt_template,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SyntheticBridgeExample":
        """Parse and validate an example from its JSON representation."""

        required = {
            "schema_version",
            "example_id",
            "split",
            "generation_seed",
            "template_family",
            "entity_family",
            "candidate_bridges",
            "source_entity",
            "gold_bridge",
            "final_answer",
            "gold_chain",
            "distractor_count",
            "distractor_chains",
            "agent_a_facts",
            "agent_b_facts",
            "agent_a_prompt",
            "agent_a_probe_prompt",
            "agent_b_prompt_template",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise BridgeDataError(
                f"invalid example keys; missing={missing or 'none'} extra={extra or 'none'}"
            )

        split = value["split"]
        if split not in SPLITS:
            raise BridgeDataError(f"unknown split: {split!r}")
        if isinstance(value["generation_seed"], bool) or not isinstance(
            value["generation_seed"], int
        ):
            raise BridgeDataError("generation_seed must be an integer")

        list_fields = (
            "candidate_bridges",
            "distractor_chains",
            "agent_a_facts",
            "agent_b_facts",
        )
        for key in list_fields:
            if not isinstance(value[key], list):
                raise BridgeDataError(f"{key} must be a JSON list")

        scalar_fields = (
            "schema_version",
            "example_id",
            "template_family",
            "entity_family",
            "source_entity",
            "gold_bridge",
            "final_answer",
            "agent_a_prompt",
            "agent_a_probe_prompt",
            "agent_b_prompt_template",
        )
        for key in scalar_fields:
            if not isinstance(value[key], str) or not value[key]:
                raise BridgeDataError(f"{key} must be a non-empty string")

        gold_chain = BridgeChain.from_dict(_require_mapping(value["gold_chain"]))
        distractors = tuple(
            BridgeChain.from_dict(_require_mapping(item))
            for item in value["distractor_chains"]
        )
        if value["distractor_count"] != len(distractors):
            raise BridgeDataError("distractor_count does not match distractor_chains")

        example = cls(
            schema_version=value["schema_version"],
            example_id=value["example_id"],
            split=split,
            generation_seed=value["generation_seed"],
            template_family=value["template_family"],
            entity_family=value["entity_family"],
            candidate_bridges=tuple(value["candidate_bridges"]),
            source_entity=value["source_entity"],
            gold_bridge=value["gold_bridge"],
            final_answer=value["final_answer"],
            distractor_chains=distractors,
            agent_a_facts=tuple(value["agent_a_facts"]),
            agent_b_facts=tuple(value["agent_b_facts"]),
            agent_a_prompt=value["agent_a_prompt"],
            agent_a_probe_prompt=value["agent_a_probe_prompt"],
            agent_b_prompt_template=value["agent_b_prompt_template"],
        )
        if example.gold_chain != gold_chain:
            raise BridgeDataError("explicit gold fields disagree with gold_chain")
        validate_example(example)
        return example


_SPLIT_RESOURCES: dict[Split, SplitResources] = {
    "train": SplitResources(
        templates=(
            TemplateFamily(
                name="archive_assignment",
                agent_a_fact=("Archive index {source} has this bridge clue: {bridge}."),
                agent_b_fact="Bridge code {bridge} unlocks destination {answer}.",
                public_question=(
                    "Which destination is ultimately reached from archive index "
                    "{source}?"
                ),
                agent_a_probe_question=(
                    "Which bridge concept is implied by the clue for archive index "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Using Agent A's bridge code, which destination is unlocked? "
                    "Reply with only the destination."
                ),
            ),
            TemplateFamily(
                name="observatory_signal",
                agent_a_fact=(
                    "Signal record {source} encodes this marker clue: {bridge}."
                ),
                agent_b_fact="Marker {bridge} resolves to station {answer}.",
                public_question=(
                    "Which station ultimately receives signal record {source}?"
                ),
                agent_a_probe_question=(
                    "Which marker concept is implied by the clue for signal record "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Using Agent A's marker, identify the station. "
                    "Reply with only the station."
                ),
            ),
            TemplateFamily(
                name="guild_courier",
                agent_a_fact="Dispatch {source} carries this token clue: {bridge}.",
                agent_b_fact="Token {bridge} delivers to chamber {answer}.",
                public_question=(
                    "Which chamber ultimately receives dispatch {source}?"
                ),
                agent_a_probe_question=(
                    "Which token concept is implied by the clue for dispatch "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Using Agent A's token, name the delivery chamber. "
                    "Reply with only the chamber."
                ),
            ),
        ),
        entities=(
            EntityFamily("cedar_folio", "cedar", "cedar"),
            EntityFamily("bronze_signal", "bronze", "bronze"),
            EntityFamily("orbit_dispatch", "orbit", "orbit"),
        ),
    ),
    "dev": SplitResources(
        templates=(
            TemplateFamily(
                name="harbor_manifest",
                agent_a_fact=(
                    "Manifest {source} contains this transfer-pennant clue: {bridge}."
                ),
                agent_b_fact="A {bridge} pennant routes cargo to berth {answer}.",
                public_question=(
                    "At which berth should cargo from manifest {source} arrive?"
                ),
                agent_a_probe_question=(
                    "Which pennant concept is implied by the clue on manifest "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Route the cargo using Agent A's pennant. "
                    "Reply with only the berth."
                ),
            ),
            TemplateFamily(
                name="garden_pollinator",
                agent_a_fact=(
                    "Plot {source} contains this pollinator-sign clue: {bridge}."
                ),
                agent_b_fact="Pollinator sign {bridge} corresponds to greenhouse {answer}.",
                public_question=(
                    "Which greenhouse is ultimately associated with plot {source}?"
                ),
                agent_a_probe_question=(
                    "Which pollinator concept is implied by the clue for plot "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Map Agent A's sign to its greenhouse. "
                    "Reply with only the greenhouse."
                ),
            ),
            TemplateFamily(
                name="museum_catalog",
                agent_a_fact=(
                    "Catalog card {source} contains this curator-seal clue: {bridge}."
                ),
                agent_b_fact="Curator seal {bridge} indexes gallery {answer}.",
                public_question=(
                    "Which gallery is ultimately indexed by catalog card {source}?"
                ),
                agent_a_probe_question=(
                    "Which seal concept is implied by the clue on catalog card "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Use Agent A's seal to find the gallery. "
                    "Reply with only the gallery."
                ),
            ),
        ),
        entities=(
            EntityFamily("quartz_manifest", "quartz", "quartz"),
            EntityFamily("lilac_plot", "lilac", "lilac"),
            EntityFamily("tide_catalog", "tide", "tide"),
        ),
    ),
    "test": SplitResources(
        templates=(
            TemplateFamily(
                name="transit_transfer",
                agent_a_fact=(
                    "Stop {source} carries this transfer-symbol clue: {bridge}."
                ),
                agent_b_fact="The onward terminus for symbol {bridge} is {answer}.",
                public_question=(
                    "What is the onward terminus for a traveler at stop {source}?"
                ),
                agent_a_probe_question=(
                    "Which transfer concept is implied by the clue at stop {source}? "
                    "Reply with only the concept."
                ),
                agent_b_question=(
                    "Given Agent A's symbol, determine the onward terminus. "
                    "Reply with only the terminus."
                ),
            ),
            TemplateFamily(
                name="festival_stage",
                agent_a_fact=(
                    "Performance slip {source} lists this access-motif clue: {bridge}."
                ),
                agent_b_fact="Access motif {bridge} admits entry to stage {answer}.",
                public_question=(
                    "Which stage can be entered with performance slip {source}?"
                ),
                agent_a_probe_question=(
                    "Which motif concept is implied by the clue on performance slip "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Use Agent A's motif to identify the stage. "
                    "Reply with only the stage."
                ),
            ),
            TemplateFamily(
                name="laboratory_sample",
                agent_a_fact=(
                    "Sample label {source} carries this reagent-tag clue: {bridge}."
                ),
                agent_b_fact="Reagent tag {bridge} is processed in bay {answer}.",
                public_question=(
                    "In which bay should sample label {source} ultimately be processed?"
                ),
                agent_a_probe_question=(
                    "Which tag concept is implied by the clue on sample label "
                    "{source}? Reply with only the concept."
                ),
                agent_b_question=(
                    "Using Agent A's tag, locate the processing bay. "
                    "Reply with only the bay."
                ),
            ),
        ),
        entities=(
            EntityFamily("ember_stop", "ember", "ember"),
            EntityFamily("ivory_slip", "ivory", "ivory"),
            EntityFamily("zenith_sample", "zenith", "zenith"),
        ),
    ),
}


def validate_candidates(candidates: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze the fixed 16-concept label space."""

    if isinstance(candidates, (str, bytes)):
        raise BridgeDataError("candidates must be a sequence of 16 strings")
    values = tuple(candidates)
    if len(values) != CANDIDATE_COUNT:
        raise BridgeDataError(
            f"expected exactly {CANDIDATE_COUNT} candidates, got {len(values)}"
        )

    normalized: list[str] = []
    for index, candidate in enumerate(values):
        if not isinstance(candidate, str):
            raise BridgeDataError(f"candidate {index} is not a string")
        if not candidate or candidate != candidate.strip():
            raise BridgeDataError(
                f"candidate {index} must be non-empty without surrounding whitespace"
            )
        if any(character in candidate for character in ("\n", "\r", "\t")):
            raise BridgeDataError(f"candidate {index} contains control whitespace")
        if AGENT_A_MESSAGE_PLACEHOLDER in candidate:
            raise BridgeDataError(f"candidate {index} contains a reserved placeholder")
        normalized.append(candidate.casefold())
    if len(set(normalized)) != len(normalized):
        raise BridgeDataError("candidate concepts must be unique ignoring case")
    return values


def contains_candidate_word(text: str, candidate: str) -> bool:
    """Return whether a candidate occurs as a Unicode-aware lexical word."""

    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    normalized_candidate = unicodedata.normalize("NFKC", candidate).casefold()
    pattern = rf"(?<!\w){re.escape(normalized_candidate)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def bridge_clue(candidate: str, split: Split) -> str:
    """Return the frozen split-specific clue for a bridge concept."""

    try:
        clue = BRIDGE_CLUES[candidate][split]
    except KeyError as error:
        raise BridgeDataError(
            f"no frozen {split} clue for bridge candidate {candidate!r}"
        ) from error
    if any(contains_candidate_word(clue, label) for label in DEFAULT_BRIDGE_CANDIDATES):
        raise BridgeDataError(f"bridge clue leaks a candidate word: {clue!r}")
    return clue


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeDataError("expected a JSON object")
    return value


def _derived_seed(seed: int, split: Split) -> int:
    payload = f"{SCHEMA_VERSION}|{seed}|{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _balanced_schedule(size: int, width: int, rng: random.Random) -> list[int]:
    schedule = [index % width for index in range(size)]
    rng.shuffle(schedule)
    return schedule


def _identifier(
    *, seed: int, split: Split, index: int, template: str, entity: str, gold: str
) -> str:
    payload = f"{seed}|{split}|{index}|{template}|{entity}|{gold}".encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:12]
    return f"bridge-{split}-{index:06d}-{suffix}"


def _entity_name(prefix: str, index: int, slot: int) -> str:
    """Build a split-safe entity name without exposing gold/distractor status."""

    return f"{prefix} {index:06d} unit {slot:02d}"


def _build_agent_a_prompt(
    *,
    facts: Sequence[str],
    public_question: str,
) -> str:
    fact_lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "You are Agent A in a two-agent relay. The relations below are private "
        "to you.\n\n"
        "PRIVATE AGENT A RELATIONS:\n"
        f"{fact_lines}\n\n"
        f"PUBLIC END-TO-END QUESTION: {public_question}\n\n"
        "Agent B has a separate private lookup table needed to determine the "
        "final answer. Send Agent B one concise, helpful handoff sentence. Do "
        "not guess or fabricate the final answer."
    )


def _build_agent_a_probe_prompt(*, facts: Sequence[str], question: str) -> str:
    fact_lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "PRIVATE AGENT A RELATIONS:\n"
        f"{fact_lines}\n\n"
        f"BEHAVIORAL COMPETENCE PROBE: {question}"
    )


def _build_agent_b_prompt_template(*, facts: Sequence[str], question: str) -> str:
    fact_lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "AGENT B RELATIONS:\n"
        f"{fact_lines}\n\n"
        "AGENT A MESSAGE:\n"
        f"{AGENT_A_MESSAGE_PLACEHOLDER}\n\n"
        f"QUESTION: {question}"
    )


def _find_template(split: Split, name: str) -> TemplateFamily:
    for template in _SPLIT_RESOURCES[split].templates:
        if template.name == name:
            return template
    raise BridgeDataError(f"template family {name!r} does not belong to {split}")


def _find_entity_family(split: Split, name: str) -> EntityFamily:
    for family in _SPLIT_RESOURCES[split].entities:
        if family.name == name:
            return family
    raise BridgeDataError(f"entity family {name!r} does not belong to {split}")


def generate_split(
    split: Split,
    *,
    size: int,
    candidates: Sequence[str],
    seed: int,
    min_distractors: int = MIN_DISTRACTORS,
    max_distractors: int = MAX_DISTRACTORS,
) -> list[SyntheticBridgeExample]:
    """Generate one deterministic split with balanced gold bridge labels."""

    if split not in SPLITS:
        raise BridgeDataError(f"unknown split: {split!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BridgeDataError("split size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BridgeDataError("seed must be an integer")
    if not (MIN_DISTRACTORS <= min_distractors <= max_distractors <= MAX_DISTRACTORS):
        raise BridgeDataError(
            f"distractor range must lie within {MIN_DISTRACTORS}-{MAX_DISTRACTORS}"
        )

    labels = validate_candidates(candidates)
    if set(labels) != set(BRIDGE_CLUES):
        raise BridgeDataError(
            "derived bridge generation requires the frozen 16-candidate clue set"
        )
    resources = _SPLIT_RESOURCES[split]
    if len(resources.templates) != len(resources.entities):
        raise BridgeDataError(
            "each template needs one semantically aligned entity family"
        )
    rng = random.Random(_derived_seed(seed, split))
    gold_schedule = _balanced_schedule(size, len(labels), rng)
    template_schedule = _balanced_schedule(size, len(resources.templates), rng)
    entity_schedule = template_schedule

    examples: list[SyntheticBridgeExample] = []
    for index in range(size):
        template = resources.templates[template_schedule[index]]
        entity_family = resources.entities[entity_schedule[index]]
        gold_bridge = labels[gold_schedule[index]]
        distractor_count = rng.randint(min_distractors, max_distractors)
        distractor_bridges = rng.sample(
            [candidate for candidate in labels if candidate != gold_bridge],
            distractor_count,
        )

        source_slots = rng.sample(range(100), distractor_count + 1)
        answer_slots = rng.sample(range(100, 200), distractor_count + 1)
        source = _entity_name(entity_family.source_prefix, index, source_slots[0])
        answer = _entity_name(entity_family.answer_prefix, index, answer_slots[0])
        distractor_chains = tuple(
            BridgeChain(
                source_entity=_entity_name(
                    entity_family.source_prefix,
                    index,
                    source_slots[distractor_index],
                ),
                bridge_concept=bridge,
                final_answer=_entity_name(
                    entity_family.answer_prefix,
                    index,
                    answer_slots[distractor_index],
                ),
            )
            for distractor_index, bridge in enumerate(distractor_bridges, start=1)
        )
        gold_chain = BridgeChain(source, gold_bridge, answer)
        all_chains = [gold_chain, *distractor_chains]

        agent_a_facts = [
            template.render_agent_a_fact(
                chain.source_entity,
                bridge_clue(chain.bridge_concept, split),
            )
            for chain in all_chains
        ]
        agent_b_facts = [
            template.render_agent_b_fact(chain.bridge_concept, chain.final_answer)
            for chain in all_chains
        ]
        rng.shuffle(agent_a_facts)
        rng.shuffle(agent_b_facts)

        candidate_order = list(labels)
        rng.shuffle(candidate_order)
        public_question = template.render_public_question(source)
        agent_a_probe_question = template.render_agent_a_probe_question(source)
        example = SyntheticBridgeExample(
            example_id=_identifier(
                seed=seed,
                split=split,
                index=index,
                template=template.name,
                entity=entity_family.name,
                gold=gold_bridge,
            ),
            split=split,
            generation_seed=seed,
            template_family=template.name,
            entity_family=entity_family.name,
            candidate_bridges=tuple(candidate_order),
            source_entity=source,
            gold_bridge=gold_bridge,
            final_answer=answer,
            distractor_chains=distractor_chains,
            agent_a_facts=tuple(agent_a_facts),
            agent_b_facts=tuple(agent_b_facts),
            agent_a_prompt=_build_agent_a_prompt(
                facts=agent_a_facts,
                public_question=public_question,
            ),
            agent_a_probe_prompt=_build_agent_a_probe_prompt(
                facts=agent_a_facts,
                question=agent_a_probe_question,
            ),
            agent_b_prompt_template=_build_agent_b_prompt_template(
                facts=agent_b_facts,
                question=template.agent_b_question,
            ),
        )
        validate_example(example)
        examples.append(example)

    counts = gold_bridge_counts(examples, labels)
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("internal error: gold bridge schedule is not balanced")
    return examples


def generate_dataset(
    *,
    candidates: Sequence[str],
    seed: int,
    train_size: int,
    dev_size: int,
    test_size: int,
    min_distractors: int = MIN_DISTRACTORS,
    max_distractors: int = MAX_DISTRACTORS,
) -> dict[Split, list[SyntheticBridgeExample]]:
    """Generate and cross-validate all three deterministic splits."""

    labels = validate_candidates(candidates)
    sizes: dict[Split, int] = {
        "train": train_size,
        "dev": dev_size,
        "test": test_size,
    }
    dataset = {
        split: generate_split(
            split,
            size=sizes[split],
            candidates=labels,
            seed=seed,
            min_distractors=min_distractors,
            max_distractors=max_distractors,
        )
        for split in SPLITS
    }
    validate_dataset(dataset, candidates=labels)
    return dataset


def validate_example(example: SyntheticBridgeExample) -> None:
    """Fail closed on ambiguity, leakage, or serialization drift in one example."""

    if example.schema_version != SCHEMA_VERSION:
        raise BridgeDataError(f"unsupported schema version: {example.schema_version!r}")
    if example.split not in SPLITS:
        raise BridgeDataError(f"unknown split: {example.split!r}")
    if not example.example_id.startswith(f"bridge-{example.split}-"):
        raise BridgeDataError("example_id does not encode its split")
    if isinstance(example.generation_seed, bool) or not isinstance(
        example.generation_seed, int
    ):
        raise BridgeDataError("generation_seed must be an integer")

    candidates = validate_candidates(example.candidate_bridges)
    template = _find_template(example.split, example.template_family)
    _find_entity_family(example.split, example.entity_family)

    scalar_values = {
        "source_entity": example.source_entity,
        "gold_bridge": example.gold_bridge,
        "final_answer": example.final_answer,
        "agent_a_prompt": example.agent_a_prompt,
        "agent_a_probe_prompt": example.agent_a_probe_prompt,
        "agent_b_prompt_template": example.agent_b_prompt_template,
    }
    for name, value in scalar_values.items():
        if not isinstance(value, str) or not value:
            raise BridgeDataError(f"{name} must be a non-empty string")

    if example.gold_bridge not in candidates:
        raise BridgeDataError("gold_bridge is not in candidate_bridges")
    if not MIN_DISTRACTORS <= example.distractor_count <= MAX_DISTRACTORS:
        raise BridgeDataError(
            f"distractor_count must be {MIN_DISTRACTORS}-{MAX_DISTRACTORS}"
        )
    if len(example.agent_a_facts) != example.distractor_count + 1:
        raise BridgeDataError("agent_a_facts must contain gold plus all distractors")
    if len(example.agent_b_facts) != example.distractor_count + 1:
        raise BridgeDataError("agent_b_facts must contain gold plus all distractors")
    if not all(isinstance(fact, str) and fact for fact in example.agent_a_facts):
        raise BridgeDataError("agent_a_facts must be non-empty strings")
    if not all(isinstance(fact, str) and fact for fact in example.agent_b_facts):
        raise BridgeDataError("agent_b_facts must be non-empty strings")

    distractor_bridges = [chain.bridge_concept for chain in example.distractor_chains]
    if len(set(distractor_bridges)) != len(distractor_bridges):
        raise BridgeDataError("distractor bridge concepts must be unique")
    if example.gold_bridge in distractor_bridges:
        raise BridgeDataError("gold bridge cannot also be a distractor")
    if not set(distractor_bridges).issubset(candidates):
        raise BridgeDataError("every distractor bridge must be a candidate")

    chains = [example.gold_chain, *example.distractor_chains]
    sources = [chain.source_entity.casefold() for chain in chains]
    answers = [chain.final_answer.casefold() for chain in chains]
    if len(set(sources)) != len(sources):
        raise BridgeDataError("source entities must be unique within an example")
    if len(set(answers)) != len(answers):
        raise BridgeDataError("final answers must be unique within an example")
    forbidden_answers = {candidate.casefold() for candidate in candidates}
    if any(answer in forbidden_answers for answer in answers):
        raise BridgeDataError("final answers must not collide with bridge candidates")

    expected_a_facts = sorted(
        template.render_agent_a_fact(
            chain.source_entity,
            bridge_clue(chain.bridge_concept, example.split),
        )
        for chain in chains
    )
    expected_b_facts = sorted(
        template.render_agent_b_fact(chain.bridge_concept, chain.final_answer)
        for chain in chains
    )
    if sorted(example.agent_a_facts) != expected_a_facts:
        raise BridgeDataError("agent_a_facts do not encode the declared chains")
    if sorted(example.agent_b_facts) != expected_b_facts:
        raise BridgeDataError("agent_b_facts do not encode the declared chains")

    expected_a_prompt = _build_agent_a_prompt(
        facts=example.agent_a_facts,
        public_question=template.render_public_question(example.source_entity),
    )
    if example.agent_a_prompt != expected_a_prompt:
        raise BridgeDataError("agent_a_prompt does not match its structured fields")
    expected_probe_prompt = _build_agent_a_probe_prompt(
        facts=example.agent_a_facts,
        question=template.render_agent_a_probe_question(example.source_entity),
    )
    if example.agent_a_probe_prompt != expected_probe_prompt:
        raise BridgeDataError(
            "agent_a_probe_prompt does not match its structured fields"
        )
    expected_b_prompt = _build_agent_b_prompt_template(
        facts=example.agent_b_facts,
        question=template.agent_b_question,
    )
    if example.agent_b_prompt_template != expected_b_prompt:
        raise BridgeDataError(
            "agent_b_prompt_template does not match its structured fields"
        )

    if example.agent_b_prompt_template.count(AGENT_A_MESSAGE_PLACEHOLDER) != 1:
        raise BridgeDataError("agent_b_prompt_template needs one message placeholder")
    if "CANDIDATE BRIDGE CONCEPTS" in example.agent_a_prompt:
        raise BridgeDataError(
            "candidate inventory leaked into Agent A's handoff prompt"
        )
    agent_a_visible = example.agent_a_prompt + "\n" + example.agent_a_probe_prompt
    leaked_candidates = [
        candidate
        for candidate in candidates
        if contains_candidate_word(agent_a_visible, candidate)
    ]
    if leaked_candidates:
        raise BridgeDataError(
            "candidate bridge words leaked into Agent A-visible text: "
            + ", ".join(leaked_candidates)
        )
    for chain in chains:
        if contains_candidate_word(agent_a_visible, chain.final_answer):
            raise BridgeDataError("a final answer leaked into Agent A-visible text")
        if contains_candidate_word(
            example.agent_b_prompt_template, chain.source_entity
        ):
            raise BridgeDataError("a source entity leaked into Agent B-visible text")
    source_units = {_unit_identifier(chain.source_entity) for chain in chains}
    answer_units = {_unit_identifier(chain.final_answer) for chain in chains}
    if not source_units.isdisjoint(answer_units):
        raise BridgeDataError("source and answer unit identifiers must be disjoint")
    all_prompts = (
        example.agent_a_prompt
        + example.agent_a_probe_prompt
        + example.agent_b_prompt_template
    ).casefold()
    if "[gold]" in all_prompts:
        raise BridgeDataError("gold markers must never appear in model prompts")


def gold_bridge_counts(
    examples: Sequence[SyntheticBridgeExample], candidates: Sequence[str]
) -> dict[str, int]:
    """Count gold labels while preserving the caller's canonical label order."""

    labels = validate_candidates(candidates)
    observed = Counter(example.gold_bridge for example in examples)
    unknown = set(observed) - set(labels)
    if unknown:
        raise BridgeDataError(f"unknown gold bridges: {sorted(unknown)}")
    return {candidate: observed[candidate] for candidate in labels}


def _all_entities(example: SyntheticBridgeExample) -> set[str]:
    chains = [example.gold_chain, *example.distractor_chains]
    return {
        value.casefold()
        for chain in chains
        for value in (chain.source_entity, chain.final_answer)
    }


def _unit_identifier(entity: str) -> str:
    try:
        return entity.rsplit(" unit ", 1)[1]
    except IndexError as error:  # pragma: no cover - generator invariant
        raise BridgeDataError(f"entity has no unit identifier: {entity!r}") from error


def validate_dataset(
    dataset: Mapping[str, Sequence[SyntheticBridgeExample]],
    *,
    candidates: Sequence[str] | None = None,
) -> None:
    """Validate split disjointness, balance, identifiers, and common label space."""

    if set(dataset) != set(SPLITS):
        raise BridgeDataError(f"dataset must contain exactly these splits: {SPLITS}")
    if any(not dataset[split] for split in SPLITS):
        raise BridgeDataError("train, dev, and test splits must all be non-empty")

    first = dataset["train"][0]
    labels = validate_candidates(candidates or first.candidate_bridges)
    canonical_labels = {candidate.casefold() for candidate in labels}
    example_ids: set[str] = set()
    generation_seeds: set[int] = set()
    template_sets: dict[Split, set[str]] = {}
    entity_family_sets: dict[Split, set[str]] = {}
    entity_sets: dict[Split, set[str]] = {}

    for split in SPLITS:
        templates: set[str] = set()
        entity_families: set[str] = set()
        entities: set[str] = set()
        for example in dataset[split]:
            validate_example(example)
            if example.split != split:
                raise BridgeDataError(
                    f"example {example.example_id} is stored in the wrong split"
                )
            if example.example_id in example_ids:
                raise BridgeDataError(f"duplicate example_id: {example.example_id}")
            example_ids.add(example.example_id)
            generation_seeds.add(example.generation_seed)
            if {
                candidate.casefold() for candidate in example.candidate_bridges
            } != canonical_labels:
                raise BridgeDataError("candidate label space changed across examples")
            templates.add(example.template_family)
            entity_families.add(example.entity_family)
            entities.update(_all_entities(example))

        counts = gold_bridge_counts(dataset[split], labels)
        if max(counts.values()) - min(counts.values()) > 1:
            raise BridgeDataError(f"gold bridge labels are imbalanced in {split}")
        template_sets[split] = templates
        entity_family_sets[split] = entity_families
        entity_sets[split] = entities

    if len(generation_seeds) != 1:
        raise BridgeDataError("all splits must come from one generation seed")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if not template_sets[left].isdisjoint(template_sets[right]):
                raise BridgeDataError(f"template families overlap: {left} and {right}")
            if not entity_family_sets[left].isdisjoint(entity_family_sets[right]):
                raise BridgeDataError(f"entity families overlap: {left} and {right}")
            if not entity_sets[left].isdisjoint(entity_sets[right]):
                raise BridgeDataError(f"concrete entities overlap: {left} and {right}")


def render_agent_b_prompt(example: SyntheticBridgeExample, agent_a_message: str) -> str:
    """Insert an observed Agent A message without interpreting its contents."""

    validate_example(example)
    if not isinstance(agent_a_message, str) or not agent_a_message.strip():
        raise BridgeDataError("agent_a_message must be a non-empty string")
    return example.agent_b_prompt_template.replace(
        AGENT_A_MESSAGE_PLACEHOLDER, agent_a_message.strip(), 1
    )


def _canonical_line(example: SyntheticBridgeExample) -> str:
    return json.dumps(
        example.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def write_jsonl(path: str | Path, examples: Sequence[SyntheticBridgeExample]) -> Path:
    """Atomically write validated examples as canonical, reproducible JSONL."""

    values = list(examples)
    if not values:
        raise BridgeDataError("refusing to write an empty JSONL split")
    for example in values:
        validate_example(example)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for example in values:
                handle.write(_canonical_line(example))
                handle.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def read_jsonl(path: str | Path) -> list[SyntheticBridgeExample]:
    """Read JSONL with line-numbered schema and semantic validation errors."""

    source = Path(path)
    examples: list[SyntheticBridgeExample] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BridgeDataError(
                    f"{source}:{line_number}: blank lines are invalid"
                )
            try:
                value = json.loads(line)
                example = SyntheticBridgeExample.from_dict(_require_mapping(value))
            except (json.JSONDecodeError, BridgeDataError) as error:
                raise BridgeDataError(f"{source}:{line_number}: {error}") from error
            examples.append(example)
    if not examples:
        raise BridgeDataError(f"{source}: JSONL file is empty")
    return examples


def write_dataset_jsonl(
    output_dir: str | Path,
    dataset: Mapping[str, Sequence[SyntheticBridgeExample]],
) -> dict[Split, Path]:
    """Validate a dataset jointly, then write train/dev/test JSONL files."""

    validate_dataset(dataset)
    root = Path(output_dir)
    return {
        split: write_jsonl(root / f"{split}.jsonl", dataset[split]) for split in SPLITS
    }


def dataset_fingerprint(
    dataset: Mapping[str, Sequence[SyntheticBridgeExample]],
) -> str:
    """Return a stable SHA-256 over split order and canonical example JSON."""

    validate_dataset(dataset)
    digest = hashlib.sha256()
    for split in SPLITS:
        digest.update(f"[{split}]\n".encode("utf-8"))
        for example in dataset[split]:
            digest.update(_canonical_line(example).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def expected_label_count_bounds(size: int) -> tuple[int, int]:
    """Return the floor/ceiling counts implied by 16-way balance."""

    if size < 0:
        raise BridgeDataError("size must be non-negative")
    return size // CANDIDATE_COUNT, math.ceil(size / CANDIDATE_COUNT)
