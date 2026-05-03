#!/usr/bin/env python3
"""
Build a semantic graph from a word problem and write it to Neo4j.

The graph focuses on meaningful math/story relations such as:
- HAS_QUANTITY
- GAINS_QUANTITY
- LOSES_QUANTITY
- TRANSFER_TO
- RATE_MULTIPLIER
- ASKS_FOR
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from llm_openrouter import (
        OpenRouterError,
        extract_constructed_text,
        extract_structured_problem,
    )
except ImportError:  # pragma: no cover - depends on runtime path
    OpenRouterError = RuntimeError
    extract_constructed_text = None
    extract_structured_problem = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - depends on local environment
    GraphDatabase = None


LETTER_PATTERN = r"[^\W\d_]"
WORD_PATTERN = rf"{LETTER_PATTERN}+(?:[-']{LETTER_PATTERN}+)*"
NUMBER_PATTERN = (
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\d+(?:,\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)

CAP_WORD_PATTERN = rf"(?-i:[A-Z]{LETTER_PATTERN}*)"
CAP_NAME_PATTERN = rf"{CAP_WORD_PATTERN}(?:\s+{CAP_WORD_PATTERN})*"
KINSHIP_PATTERN = r"(?:his|her|their)\s+(?:sister|brother|mother|father|friend|cousin)"
PRONOUN_PATTERN = r"(?:he|she|they|him|her|them)"

PERSON_CAPTURE_PATTERN = (
    rf"(?:{CAP_NAME_PATTERN}|{KINSHIP_PATTERN}|{PRONOUN_PATTERN})"
)
ITEM_CAPTURE_PATTERN = rf"{WORD_PATTERN}(?:\s+{WORD_PATTERN})?"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
PERSON_WORD_BLACKLIST = {
    "if",
    "how",
    "what",
    "when",
    "where",
    "why",
    "who",
    "then",
    "and",
    "or",
    "the",
    "a",
    "an",
}
ITEM_WORD_BLACKLIST = {
    "a",
    "an",
    "the",
    "on",
    "in",
    "at",
    "to",
    "for",
    "of",
    "per",
    "week",
    "day",
    "month",
    "year",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "get",
    "gets",
    "got",
    "have",
    "has",
    "had",
    "now",
    "long",
}
PRONOUNS = {"he", "she", "they", "him", "her", "them"}
ITEM_PRONOUNS = {"it", "them", "this", "that"}
KINSHIP_PREFIX_RE = re.compile(
    r"^(?:his|her|their)\s+(?:sister|brother|mother|father|friend|cousin)$",
    flags=re.IGNORECASE,
)

TRANSFER_PATTERNS = [
    re.compile(
        rf"(?P<source>{PERSON_CAPTURE_PATTERN})\s+"
        r"(?:gives?|gave|sends?|sent|hands?|handed|offers?|offered|donates?|donated)\s+"
        rf"(?P<target>{PERSON_CAPTURE_PATTERN})\s+"
        rf"(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_CAPTURE_PATTERN})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<source>{PERSON_CAPTURE_PATTERN})\s+"
        r"(?:gives?|gave|sends?|sent|hands?|handed|offers?|offered|donates?|donated)\s+"
        rf"(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_CAPTURE_PATTERN})\s+to\s+"
        rf"(?P<target>{PERSON_CAPTURE_PATTERN})",
        flags=re.IGNORECASE,
    ),
]

HAS_PATTERN = re.compile(
    rf"(?P<subject>{PERSON_CAPTURE_PATTERN})\s+"
    r"(?:has|have|had|owns?|keeps?|holds?|watched?|watches)\s+"
    r"(?:a\s+total\s+of\s+)?"
    rf"(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_CAPTURE_PATTERN})",
    flags=re.IGNORECASE,
)

GAIN_PATTERN = re.compile(
    rf"(?P<subject>{PERSON_CAPTURE_PATTERN})\s+"
    r"(?:buys?|bought|gets?|got|earns?|earned|receives?|received|finds?|found|collects?|collected)\s+"
    rf"(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_CAPTURE_PATTERN})",
    flags=re.IGNORECASE,
)

LOSS_PATTERN = re.compile(
    rf"(?P<subject>{PERSON_CAPTURE_PATTERN})\s+"
    r"(?:loses?|lost|spends?|spent|uses?|used|eats?|ate|sells?|sold)\s+"
    rf"(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_CAPTURE_PATTERN})",
    flags=re.IGNORECASE,
)

RATE_PATTERN = re.compile(
    rf"(?P<subject>{PERSON_CAPTURE_PATTERN}).{{0,80}}?"
    rf"(?P<multiplier>{NUMBER_PATTERN})\s+times\s+as\s+(?:often|many|much)\s+as\s+"
    rf"(?P<other>{PERSON_CAPTURE_PATTERN})",
    flags=re.IGNORECASE,
)

DEPRECIATION_PATTERNS = [
    re.compile(
        rf"(?P<subject>{CAP_NAME_PATTERN})['’]s\s+(?P<item>{ITEM_CAPTURE_PATTERN})\s+"
        r"(?:goes?|went|drops?|dropped|decreases?|decreased|falls?|fell)\s+down\s+in\s+value\s+by\s+\$?(?P<value>"
        rf"{NUMBER_PATTERN})\s+(?:a|per)\s+(?P<period>{WORD_PATTERN})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<subject>{CAP_NAME_PATTERN})['’]s\s+(?P<item>{ITEM_CAPTURE_PATTERN})\s+"
        r"(?:depreciates?|depreciated)\s+by\s+\$?(?P<value>"
        rf"{NUMBER_PATTERN})\s+(?:a|per)\s+(?P<period>{WORD_PATTERN})",
        flags=re.IGNORECASE,
    ),
]

PURCHASE_VALUE_PATTERN = re.compile(
    rf"(?P<subject>{PERSON_CAPTURE_PATTERN})\s+"
    r"(?:bought|purchased|paid\s+for)\s+"
    rf"(?P<item_ref>it|them|this|that|{ITEM_CAPTURE_PATTERN})\s+for\s+\$?(?P<value>{NUMBER_PATTERN})",
    flags=re.IGNORECASE,
)

WORTH_AFTER_PATTERN = re.compile(
    rf"how\s+much\s+is\s+(?P<item_ref>it|this|that|{ITEM_CAPTURE_PATTERN})\s+worth\s+after\s+"
    rf"(?P<value>{NUMBER_PATTERN})\s+(?P<period>{WORD_PATTERN})",
    flags=re.IGNORECASE,
)

EACH_UNIT_RATE_PATTERN = re.compile(
    rf"each\s+(?P<base_item>{ITEM_CAPTURE_PATTERN})\s+"
    r"(?:is|are|costs?|lasts?|equals?)\s+"
    rf"(?P<value>{NUMBER_PATTERN})\s+(?P<target_item>{ITEM_CAPTURE_PATTERN})(?:\s+long)?",
    flags=re.IGNORECASE,
)

QUESTION_PATTERNS = [
    re.compile(
        rf"how\s+(?:many|much)\s+(?P<item>{ITEM_CAPTURE_PATTERN}).{{0,80}}?"
        rf"(?:does|do|did)\s+(?P<subject>{PERSON_CAPTURE_PATTERN})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<subject>{PERSON_CAPTURE_PATTERN})\s+c(?:ó|o)\s+bao\s+nhi(?:ê|e)u\s+"
        rf"(?P<item>{ITEM_CAPTURE_PATTERN})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"bao\s+nhi(?:ê|e)u\s+(?P<item>{ITEM_CAPTURE_PATTERN})",
        flags=re.IGNORECASE,
    ),
]

SAFE_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SAFE_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")


@dataclass
class Node:
    key: str
    label: str
    name: str


@dataclass
class Edge:
    source_key: str
    target_key: str
    rel_type: str
    properties: Dict[str, object]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def parse_number(value: str) -> Optional[float]:
    word_to_number = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    value_norm = normalize_text(value).lower()
    if not value_norm:
        return None
    if value_norm in word_to_number:
        return float(word_to_number[value_norm])
    raw = value_norm.replace("$", "")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d+", raw):
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_period_name(value: str) -> Optional[str]:
    period = normalize_text(value).lower().strip(" ,.;:!?")
    if not period:
        return None
    if period.endswith("s") and len(period) > 3 and not period.endswith("ss"):
        period = period[:-1]
    return period


def normalize_person_name(value: str, last_person: Optional[str]) -> Optional[str]:
    text = normalize_text(value).strip(" ,.;:!?")
    if not text:
        return None

    lower = text.lower()
    if lower in PRONOUNS:
        return last_person

    if KINSHIP_PREFIX_RE.match(text):
        return " ".join(part.capitalize() for part in lower.split())

    tokens = text.split()
    while tokens and tokens[0].lower() in PERSON_WORD_BLACKLIST:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in PERSON_WORD_BLACKLIST:
        tokens = tokens[:-1]

    if not tokens:
        return None

    candidate = " ".join(tokens)
    if candidate.lower() in PERSON_WORD_BLACKLIST:
        return None
    return candidate


def normalize_item_name(value: str) -> Optional[str]:
    text = normalize_text(value).strip(" ,.;:!?")
    if not text:
        return None

    tokens = text.lower().split()
    while tokens and tokens[0] in ITEM_WORD_BLACKLIST:
        tokens = tokens[1:]
    while tokens and tokens[-1] in ITEM_WORD_BLACKLIST:
        tokens = tokens[:-1]

    if not tokens:
        return None

    if len(tokens) > 2:
        tokens = tokens[:2]

    item = " ".join(tokens)
    if item.endswith("s") and len(item) > 3 and not item.endswith("ss"):
        item = item[:-1]
    return item


def node_key(label: str, name: str) -> str:
    return f"{label.lower()}:{normalize_text(name).lower()}"


def dedupe_edges(edges: List[Edge]) -> List[Edge]:
    seen = set()
    deduped: List[Edge] = []
    for edge in edges:
        signature = (
            edge.source_key,
            edge.target_key,
            edge.rel_type,
            edge.properties.get("item"),
            edge.properties.get("value"),
            edge.properties.get("sentence_idx"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(edge)
    return deduped


CANONICAL_LABELS = {
    "person": "Person",
    "item": "Item",
    "period": "Period",
    "problem": "Problem",
}

ALLOWED_RELATIONS = {
    "HAS_QUANTITY",
    "GAINS_QUANTITY",
    "LOSES_QUANTITY",
    "TRANSFER_TO",
    "RATE_MULTIPLIER",
    "UNIT_RATE",
    "OWNS_ITEM",
    "HAS_VALUE",
    "VALUE_DECREASE_PER_PERIOD",
    "ASKS_FOR",
    "ASKS_WORTH_AFTER",
    "LESS_THAN_BY",
    "MORE_THAN_BY",
    "TIMES_OF",
}


def canonical_label(raw: object) -> Optional[str]:
    if raw is None:
        return None
    label = normalize_text(str(raw))
    if not label:
        return None
    lowered = label.lower()
    if lowered in CANONICAL_LABELS:
        return CANONICAL_LABELS[lowered]
    for allowed in CANONICAL_LABELS.values():
        if label == allowed:
            return allowed
    return None


def normalize_relation_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    rel = normalize_text(str(raw)).upper()
    if not rel:
        return None
    rel = re.sub(r"[^A-Z0-9]+", "_", rel).strip("_")
    if rel in ALLOWED_RELATIONS:
        return rel
    return None


def _structured_ref_to_node_key(
    ref: object,
    *,
    nodes: Dict[str, Node],
    default_label: Optional[str] = None,
) -> Optional[str]:
    label: Optional[str] = None
    name: Optional[str] = None

    if isinstance(ref, dict):
        label = canonical_label(ref.get("type") or ref.get("label"))
        raw_name = ref.get("name")
        if raw_name is not None:
            name = normalize_text(str(raw_name))
    elif isinstance(ref, str):
        text = normalize_text(ref)
        if ":" in text:
            prefix, suffix = text.split(":", 1)
            label = canonical_label(prefix)
            name = normalize_text(suffix)
        else:
            label = canonical_label(default_label)
            name = text

    if not label and default_label:
        label = canonical_label(default_label)
    if not label or not name:
        return None

    key = node_key(label, name)
    if key not in nodes:
        nodes[key] = Node(key=key, label=label, name=name)
    return key


def extract_graph_from_structured(
    structured: Dict[str, Any],
    *,
    problem_id: str,
    problem_text: str,
) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    sentences: List[str] = []

    problem_key = node_key("Problem", problem_id)
    nodes[problem_key] = Node(key=problem_key, label="Problem", name=problem_id)

    for entity in structured.get("entities", []) if isinstance(structured.get("entities"), list) else []:
        if not isinstance(entity, dict):
            continue
        label = canonical_label(entity.get("type") or entity.get("label"))
        name = normalize_text(str(entity.get("name", "")))
        if not label or not name:
            continue
        key = node_key(label, name)
        if key not in nodes:
            nodes[key] = Node(key=key, label=label, name=name)

    raw_facts: List[Dict[str, Any]] = []
    if isinstance(structured.get("facts"), list):
        raw_facts.extend([fact for fact in structured["facts"] if isinstance(fact, dict)])
    if isinstance(structured.get("question"), dict):
        raw_facts.append(structured["question"])

    for idx, fact in enumerate(raw_facts, start=1):
        rel_type = normalize_relation_type(fact.get("relation") or fact.get("type"))
        if not rel_type:
            continue
        source_key = _structured_ref_to_node_key(
            fact.get("source"),
            nodes=nodes,
            default_label="Problem",
        )
        target_key = _structured_ref_to_node_key(
            fact.get("target"),
            nodes=nodes,
            default_label="Item",
        )
        if not source_key or not target_key:
            continue

        props: Dict[str, object] = {"sentence_idx": idx}

        sentence_raw = fact.get("sentence")
        if sentence_raw is not None:
            sentence = normalize_text(str(sentence_raw))
            if sentence:
                props["sentence"] = sentence
                if sentence not in sentences:
                    sentences.append(sentence)

        if rel_type == "RATE_MULTIPLIER":
            value_raw = fact.get("multiplier", fact.get("value"))
            if value_raw is not None:
                if isinstance(value_raw, (int, float)):
                    props["multiplier"] = float(value_raw)
                else:
                    parsed = parse_number(str(value_raw))
                    if parsed is not None:
                        props["multiplier"] = parsed
        else:
            value_raw = fact.get("value")
            if value_raw is not None:
                if isinstance(value_raw, (int, float)):
                    props["value"] = float(value_raw)
                else:
                    parsed = parse_number(str(value_raw))
                    if parsed is not None:
                        props["value"] = parsed

        unit_raw = fact.get("unit")
        if unit_raw is not None:
            unit = normalize_text(str(unit_raw)).lower()
            if unit:
                props["item"] = unit

        period_raw = fact.get("period")
        if period_raw is not None:
            period = normalize_text(str(period_raw)).lower()
            if period:
                props["period"] = period
                if "item" not in props and rel_type in {"ASKS_WORTH_AFTER"}:
                    props["item"] = period

        edges.append(
            Edge(
                source_key=source_key,
                target_key=target_key,
                rel_type=rel_type,
                properties=props,
            )
        )

    if not sentences:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalize_text(problem_text)) if s.strip()]
    return nodes, dedupe_edges(edges), sentences


def _to_ref_dict(raw_ref: str) -> Optional[Dict[str, str]]:
    text = normalize_text(raw_ref)
    if not text or text == "-":
        return None
    if ":" not in text:
        return {"type": "Item", "name": text}
    raw_label, raw_name = text.split(":", 1)
    label = canonical_label(raw_label)
    name = normalize_text(raw_name)
    if not label or not name or name == "-":
        return None
    return {"type": label, "name": name}


def extract_graph_from_constructed_text(
    constructed_text: str,
    *,
    problem_id: str,
    problem_text: str,
) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
    structured: Dict[str, Any] = {"entities": [], "facts": []}

    lines = [
        normalize_text(line)
        for line in constructed_text.splitlines()
        if normalize_text(line) and not normalize_text(line).startswith("#")
    ]

    for line in lines:
        parts = [part.strip() for part in line.split("|")]
        if not parts:
            continue
        head = parts[0].upper()

        if head == "ENTITY":
            if len(parts) < 3:
                continue
            structured["entities"].append({"type": parts[1], "name": parts[2]})
            continue

        if head not in {"FACT", "QUESTION"}:
            continue
        if len(parts) < 4:
            continue

        relation = parts[1]
        source_ref = _to_ref_dict(parts[2])
        target_ref = _to_ref_dict(parts[3])
        value_raw = parts[4] if len(parts) > 4 else "-"
        unit_raw = parts[5] if len(parts) > 5 else "-"
        period_raw = parts[6] if len(parts) > 6 else "-"

        fact: Dict[str, Any] = {
            "relation": relation,
            "source": source_ref or {"type": "Problem", "name": problem_id},
            "target": target_ref or {"type": "Problem", "name": problem_id},
            "value": None,
            "unit": None,
            "period": None,
            "sentence": line,
        }

        if value_raw and value_raw != "-":
            parsed = parse_number(value_raw)
            fact["value"] = parsed if parsed is not None else None
        if unit_raw and unit_raw != "-":
            fact["unit"] = normalize_text(unit_raw)
        if period_raw and period_raw != "-":
            fact["period"] = normalize_text(period_raw)

        if head == "QUESTION":
            structured["question"] = fact
        else:
            structured["facts"].append(fact)

    return extract_graph_from_structured(
        structured,
        problem_id=problem_id,
        problem_text=problem_text,
    )


def build_quantity_node_from_edge(edge: Edge) -> Optional[Node]:
    value = edge.properties.get("value")
    multiplier = edge.properties.get("multiplier")
    item = edge.properties.get("item")
    sentence_idx = edge.properties.get("sentence_idx", 0)

    amount: Optional[float] = None
    unit: Optional[str] = None
    kind: Optional[str] = None
    if isinstance(value, (int, float)):
        amount = float(value)
        unit = str(item) if item else None
        kind = "value"
    elif isinstance(multiplier, (int, float)):
        amount = float(multiplier)
        unit = "times"
        kind = "multiplier"
    else:
        return None

    key = (
        f"quantity:{edge.source_key}|{edge.rel_type}|{edge.target_key}|"
        f"{sentence_idx}|{amount}|{unit or '-'}|{kind}"
    )
    if unit:
        name = f"{amount:g} {unit}"
    else:
        name = f"{amount:g}"
    return Node(key=key, label="Quantity", name=name)


class SemanticExtractor:
    def __init__(self) -> None:
        self.nlp = None
        self.nlp_model_name = "rule-only"
        self.last_person: Optional[str] = None
        self.last_item: Optional[str] = None
        self._load_spacy_if_available()

    def _load_spacy_if_available(self) -> None:
        try:
            import spacy
        except ImportError:
            return

        for model_name in ("en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"):
            try:
                self.nlp = spacy.load(model_name)
                self.nlp_model_name = model_name
                return
            except OSError:
                continue

        self.nlp = spacy.blank("en")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        self.nlp_model_name = "spacy.blank(en)"

    def split_sentences(self, text: str) -> List[str]:
        clean_text = normalize_text(text)
        if not clean_text:
            return []

        if self.nlp is not None:
            doc = self.nlp(clean_text)
            return [normalize_text(sent.text) for sent in doc.sents if normalize_text(sent.text)]

        return [s.strip() for s in SENTENCE_SPLIT_RE.split(clean_text) if s.strip()]

    def _ensure_node(self, nodes: Dict[str, Node], label: str, name: str) -> str:
        key = node_key(label, name)
        if key not in nodes:
            nodes[key] = Node(key=key, label=label, name=name)
        if label == "Item":
            self.last_item = name
        return key

    def _resolve_person(self, raw: str) -> Optional[str]:
        person = normalize_person_name(raw, self.last_person)
        if not person:
            return None
        if person.lower() not in PRONOUNS:
            self.last_person = person
        return person

    def _resolve_item(self, raw: str) -> Optional[str]:
        raw_norm = normalize_text(raw).lower()
        if raw_norm in ITEM_PRONOUNS:
            return self.last_item
        item = normalize_item_name(raw)
        if item:
            self.last_item = item
        return item

    def _add_quantity_edge(
        self,
        edges: List[Edge],
        source_key: str,
        target_key: str,
        rel_type: str,
        sentence: str,
        sentence_idx: int,
        value: Optional[float],
        item: Optional[str],
    ) -> None:
        props: Dict[str, object] = {
            "sentence": sentence,
            "sentence_idx": sentence_idx,
        }
        if value is not None:
            props["value"] = value
        if item:
            props["item"] = item
        edges.append(
            Edge(
                source_key=source_key,
                target_key=target_key,
                rel_type=rel_type,
                properties=props,
            )
        )

    def _extract_transfer(
        self,
        sentence: str,
        sentence_idx: int,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        for pattern in TRANSFER_PATTERNS:
            for match in pattern.finditer(sentence):
                source_name = self._resolve_person(match.group("source"))
                target_name = self._resolve_person(match.group("target"))
                item_name = self._resolve_item(match.group("item"))
                value = parse_number(match.group("value"))

                if not source_name or not target_name or not item_name:
                    continue

                source_key = self._ensure_node(nodes, "Person", source_name)
                target_key = self._ensure_node(nodes, "Person", target_name)
                item_key = self._ensure_node(nodes, "Item", item_name)

                self._add_quantity_edge(
                    edges,
                    source_key=source_key,
                    target_key=target_key,
                    rel_type="TRANSFER_TO",
                    sentence=sentence,
                    sentence_idx=sentence_idx,
                    value=value,
                    item=item_name,
                )
                self._add_quantity_edge(
                    edges,
                    source_key=source_key,
                    target_key=item_key,
                    rel_type="LOSES_QUANTITY",
                    sentence=sentence,
                    sentence_idx=sentence_idx,
                    value=value,
                    item=item_name,
                )
                self._add_quantity_edge(
                    edges,
                    source_key=target_key,
                    target_key=item_key,
                    rel_type="GAINS_QUANTITY",
                    sentence=sentence,
                    sentence_idx=sentence_idx,
                    value=value,
                    item=item_name,
                )

    def _extract_person_item_quantity(
        self,
        sentence: str,
        sentence_idx: int,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        patterns = [
            (HAS_PATTERN, "HAS_QUANTITY"),
            (GAIN_PATTERN, "GAINS_QUANTITY"),
            (LOSS_PATTERN, "LOSES_QUANTITY"),
        ]
        for pattern, rel_type in patterns:
            for match in pattern.finditer(sentence):
                subject_name = self._resolve_person(match.group("subject"))
                item_name = self._resolve_item(match.group("item"))
                value = parse_number(match.group("value"))
                if not subject_name or not item_name:
                    continue

                subject_key = self._ensure_node(nodes, "Person", subject_name)
                item_key = self._ensure_node(nodes, "Item", item_name)
                self._add_quantity_edge(
                    edges,
                    source_key=subject_key,
                    target_key=item_key,
                    rel_type=rel_type,
                    sentence=sentence,
                    sentence_idx=sentence_idx,
                    value=value,
                    item=item_name,
                )

    def _extract_rate_relation(
        self,
        sentence: str,
        sentence_idx: int,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        for match in RATE_PATTERN.finditer(sentence):
            subject_name = self._resolve_person(match.group("subject"))
            other_name = self._resolve_person(match.group("other"))
            multiplier = parse_number(match.group("multiplier"))
            if not subject_name or not other_name or multiplier is None:
                continue

            subject_key = self._ensure_node(nodes, "Person", subject_name)
            other_key = self._ensure_node(nodes, "Person", other_name)
            edges.append(
                Edge(
                    source_key=subject_key,
                    target_key=other_key,
                    rel_type="RATE_MULTIPLIER",
                    properties={
                        "multiplier": multiplier,
                        "sentence": sentence,
                        "sentence_idx": sentence_idx,
                    },
                )
            )

    def _extract_unit_rate_relation(
        self,
        sentence: str,
        sentence_idx: int,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        for match in EACH_UNIT_RATE_PATTERN.finditer(sentence):
            base_item = self._resolve_item(match.group("base_item"))
            target_item = self._resolve_item(match.group("target_item"))
            value = parse_number(match.group("value"))
            if not base_item or not target_item or value is None:
                continue

            base_key = self._ensure_node(nodes, "Item", base_item)
            target_key = self._ensure_node(nodes, "Item", target_item)
            edges.append(
                Edge(
                    source_key=base_key,
                    target_key=target_key,
                    rel_type="UNIT_RATE",
                    properties={
                        "value": value,
                        "item": target_item,
                        "per_item": base_item,
                        "sentence": sentence,
                        "sentence_idx": sentence_idx,
                    },
                )
            )

    def _extract_value_problem_relations(
        self,
        sentence: str,
        sentence_idx: int,
        problem_key: str,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        for pattern in DEPRECIATION_PATTERNS:
            for match in pattern.finditer(sentence):
                subject_name = self._resolve_person(match.group("subject"))
                item_name = self._resolve_item(match.group("item"))
                value = parse_number(match.group("value"))
                period = normalize_period_name(match.group("period"))
                if not subject_name or not item_name or value is None or not period:
                    continue

                subject_key = self._ensure_node(nodes, "Person", subject_name)
                item_key = self._ensure_node(nodes, "Item", item_name)
                period_key = self._ensure_node(nodes, "Period", period)
                edges.append(
                    Edge(
                        source_key=subject_key,
                        target_key=item_key,
                        rel_type="OWNS_ITEM",
                        properties={"sentence": sentence, "sentence_idx": sentence_idx},
                    )
                )
                edges.append(
                    Edge(
                        source_key=item_key,
                        target_key=period_key,
                        rel_type="VALUE_DECREASE_PER_PERIOD",
                        properties={
                            "value": value,
                            "item": "usd",
                            "period": period,
                            "sentence": sentence,
                            "sentence_idx": sentence_idx,
                        },
                    )
                )

        for match in PURCHASE_VALUE_PATTERN.finditer(sentence):
            subject_name = self._resolve_person(match.group("subject"))
            item_name = self._resolve_item(match.group("item_ref"))
            value = parse_number(match.group("value"))
            if not subject_name or not item_name or value is None:
                continue

            subject_key = self._ensure_node(nodes, "Person", subject_name)
            item_key = self._ensure_node(nodes, "Item", item_name)
            edges.append(
                Edge(
                    source_key=subject_key,
                    target_key=item_key,
                    rel_type="HAS_VALUE",
                    properties={
                        "value": value,
                        "item": "usd",
                        "sentence": sentence,
                        "sentence_idx": sentence_idx,
                    },
                )
            )

        for match in WORTH_AFTER_PATTERN.finditer(sentence):
            item_name = self._resolve_item(match.group("item_ref"))
            years = parse_number(match.group("value"))
            period = normalize_period_name(match.group("period"))
            if not item_name or years is None or not period:
                continue

            item_key = self._ensure_node(nodes, "Item", item_name)
            asker_name = self.last_person
            source_key = problem_key
            if asker_name:
                source_key = self._ensure_node(nodes, "Person", asker_name)
            edges.append(
                Edge(
                    source_key=source_key,
                    target_key=item_key,
                    rel_type="ASKS_WORTH_AFTER",
                    properties={
                        "value": years,
                        "item": period,
                        "period": period,
                        "sentence": sentence,
                        "sentence_idx": sentence_idx,
                    },
                )
            )

    def _extract_question(
        self,
        sentence: str,
        sentence_idx: int,
        problem_key: str,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> None:
        for idx, pattern in enumerate(QUESTION_PATTERNS):
            for match in pattern.finditer(sentence):
                item_name = self._resolve_item(match.group("item"))
                if not item_name:
                    continue
                item_key = self._ensure_node(nodes, "Item", item_name)

                subject_name = None
                if "subject" in match.groupdict():
                    raw_subject = match.groupdict().get("subject")
                    if raw_subject:
                        subject_name = self._resolve_person(raw_subject)
                if not subject_name and idx == 2:
                    subject_name = self.last_person

                if subject_name:
                    subject_key = self._ensure_node(nodes, "Person", subject_name)
                    edges.append(
                        Edge(
                            source_key=subject_key,
                            target_key=item_key,
                            rel_type="ASKS_FOR",
                            properties={
                                "sentence": sentence,
                                "sentence_idx": sentence_idx,
                            },
                        )
                    )
                else:
                    edges.append(
                        Edge(
                            source_key=problem_key,
                            target_key=item_key,
                            rel_type="ASKS_FOR",
                            properties={
                                "sentence": sentence,
                                "sentence_idx": sentence_idx,
                            },
                        )
                    )

    def extract(
        self, text: str, problem_id: str
    ) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []
        sentences = self.split_sentences(text)

        problem_key = node_key("Problem", problem_id)
        nodes[problem_key] = Node(key=problem_key, label="Problem", name=problem_id)

        for idx, sentence in enumerate(sentences, start=1):
            self._extract_transfer(sentence, idx, nodes, edges)
            self._extract_person_item_quantity(sentence, idx, nodes, edges)
            self._extract_rate_relation(sentence, idx, nodes, edges)
            self._extract_unit_rate_relation(sentence, idx, nodes, edges)
            self._extract_value_problem_relations(sentence, idx, problem_key, nodes, edges)
            self._extract_question(sentence, idx, problem_key, nodes, edges)

        return nodes, dedupe_edges(edges), sentences


def push_to_neo4j(
    uri: str,
    user: str,
    password: str,
    database: str,
    nodes: Dict[str, Node],
    edges: List[Edge],
    clear_db: bool,
) -> None:
    if GraphDatabase is None:
        raise RuntimeError("Missing dependency 'neo4j'. Install with: pip install neo4j")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            if clear_db:
                session.run("MATCH (n) DETACH DELETE n").consume()

            for node in nodes.values():
                if not SAFE_LABEL_RE.match(node.label):
                    raise ValueError(f"Unsafe Neo4j label: {node.label}")
                session.run(
                    f"""
                    MERGE (n:{node.label} {{key: $key}})
                    SET n.name = $name
                    """,
                    key=node.key,
                    name=node.name,
                ).consume()

            for edge in edges:
                if not SAFE_REL_TYPE_RE.match(edge.rel_type):
                    raise ValueError(f"Unsafe Neo4j relationship type: {edge.rel_type}")
                session.run(
                    f"""
                    MATCH (a {{key: $source_key}})
                    MATCH (b {{key: $target_key}})
                    CREATE (a)-[r:{edge.rel_type}]->(b)
                    SET r += $props
                    """,
                    source_key=edge.source_key,
                    target_key=edge.target_key,
                    props=edge.properties,
                ).consume()

                quantity_node = build_quantity_node_from_edge(edge)
                if quantity_node is None:
                    continue
                amount = edge.properties.get("value", edge.properties.get("multiplier"))
                if amount is None:
                    continue
                unit = edge.properties.get("item")
                kind = "value" if "value" in edge.properties else "multiplier"

                session.run(
                    """
                    MERGE (q:Quantity {key: $q_key})
                    SET q.name = $q_name,
                        q.amount = $amount,
                        q.unit = $unit,
                        q.kind = $kind,
                        q.rel_type = $rel_type,
                        q.sentence_idx = $sentence_idx
                    """,
                    q_key=quantity_node.key,
                    q_name=quantity_node.name,
                    amount=float(amount),
                    unit=unit,
                    kind=kind,
                    rel_type=edge.rel_type,
                    sentence_idx=edge.properties.get("sentence_idx"),
                ).consume()
                session.run(
                    """
                    MATCH (a {key: $source_key})
                    MATCH (b {key: $target_key})
                    MATCH (q:Quantity {key: $q_key})
                    MERGE (a)-[:HAS_QUANTITY_VALUE]->(q)
                    MERGE (q)-[:DESCRIBES]->(b)
                    """,
                    source_key=edge.source_key,
                    target_key=edge.target_key,
                    q_key=quantity_node.key,
                ).consume()


def print_preview(model_name: str, sentences: List[str], nodes: Dict[str, Node], edges: List[Edge]) -> None:
    label_counts: Dict[str, int] = {}
    rel_counts: Dict[str, int] = {}

    for node in nodes.values():
        label_counts[node.label] = label_counts.get(node.label, 0) + 1
    for edge in edges:
        rel_counts[edge.rel_type] = rel_counts.get(edge.rel_type, 0) + 1

    print(f"[info] NLP backend: {model_name}")
    print(f"[info] Sentences: {len(sentences)}")
    print(f"[info] Nodes: {len(nodes)}")
    print(f"[info] Semantic edges: {len(edges)}")

    if label_counts:
        print("[info] Node labels:")
        for label in sorted(label_counts):
            print(f"  - {label}: {label_counts[label]}")
    if rel_counts:
        print("[info] Relation types:")
        for rel_type in sorted(rel_counts):
            print(f"  - {rel_type}: {rel_counts[rel_type]}")

    preview_edges = edges[:12]
    if preview_edges:
        print("[preview] First edges:")
        for edge in preview_edges:
            value = edge.properties.get("value")
            item = edge.properties.get("item")
            extras = []
            if value is not None:
                extras.append(f"value={value}")
            if item:
                extras.append(f"item={item}")
            extras_text = f" ({', '.join(extras)})" if extras else ""
            print(f"  - {edge.rel_type}: {edge.source_key} -> {edge.target_key}{extras_text}")


def parse_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return data


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        default=Path("Formalizer/local_config.json"),
        help="Path to local JSON config file.",
    )
    base_args, _ = base_parser.parse_known_args()
    config = parse_config(base_args.config)

    parser = argparse.ArgumentParser(
        description="Extract semantic relations from Input/problem.txt and build Neo4j graph."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_args.config,
        help="Path to local JSON config file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(str(config.get("input", "Input/problem.txt"))),
        help="Path to text file that contains the problem(s).",
    )
    parser.add_argument(
        "--uri",
        default=str(config.get("uri", os.getenv("NEO4J_URI", "neo4j://localhost:7687"))),
        help="Neo4j URI, e.g. neo4j://localhost:7687",
    )
    parser.add_argument(
        "--user",
        default=str(config.get("user", os.getenv("NEO4J_USER", "neo4j"))),
        help="Neo4j username.",
    )
    parser.add_argument(
        "--password",
        default=str(config.get("password", os.getenv("NEO4J_PASSWORD", ""))).strip() or None,
        help="Neo4j password. Can also be set via NEO4J_PASSWORD env.",
    )
    parser.add_argument(
        "--database",
        default=str(config.get("database", os.getenv("NEO4J_DATABASE", "neo4j"))),
        help="Neo4j database name.",
    )
    parser.add_argument(
        "--document-id",
        default=str(config.get("document_id", "problem_txt")),
        help="Problem node ID in Neo4j.",
    )
    parser.add_argument(
        "--parser",
        choices=("semantic", "llm", "hybrid", "llm_text", "hybrid_text"),
        default=str(config.get("parser", "semantic")),
        help=(
            "Extraction backend: "
            "semantic, llm(json), hybrid(llm json -> fallback), "
            "llm_text(constructed text), hybrid_text(llm text -> fallback)."
        ),
    )
    parser.add_argument(
        "--openrouter-model",
        default=str(config.get("openrouter_model", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))),
        help="OpenRouter model name, e.g. openai/gpt-4o-mini.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=str(config.get("openrouter_api_key", os.getenv("OPENROUTER_API_KEY", ""))).strip() or None,
        help="OpenRouter API key. Prefer env OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--openrouter-timeout",
        type=int,
        default=int(config.get("openrouter_timeout", os.getenv("OPENROUTER_TIMEOUT", "90"))),
        help="OpenRouter request timeout in seconds.",
    )
    parser.add_argument(
        "--openrouter-site-url",
        default=str(config.get("openrouter_site_url", os.getenv("OPENROUTER_SITE_URL", ""))).strip() or None,
        help="Optional site URL sent via HTTP-Referer.",
    )
    parser.add_argument(
        "--openrouter-app-name",
        default=str(config.get("openrouter_app_name", os.getenv("OPENROUTER_APP_NAME", ""))).strip() or None,
        help="Optional app name sent via X-Title.",
    )
    parser.add_argument(
        "--constructed-out",
        type=Path,
        default=Path(str(config.get("constructed_out", "Formalizer/constructed_summary.txt"))),
        help="Output path to save LLM constructed text when parser is llm_text/hybrid_text.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing nodes/relationships before import.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print extraction result, do not write to Neo4j.",
    )
    args = parser.parse_args()

    argv = set(os.sys.argv[1:])
    if "--clear" not in argv and bool(config.get("clear", False)):
        args.clear = True
    if "--dry-run" not in argv and bool(config.get("dry_run", False)):
        args.dry_run = True
    return args


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    content = args.input.read_text(encoding="utf-8")
    if not content.strip():
        print(f"[warn] Input file is empty: {args.input}")
        return

    backend_name = ""
    nodes: Dict[str, Node]
    edges: List[Edge]
    sentences: List[str]

    llm_error: Optional[Exception] = None
    if args.parser in {"llm", "hybrid", "llm_text", "hybrid_text"}:
        if args.parser in {"llm", "hybrid"} and extract_structured_problem is None:
            llm_error = OpenRouterError(
                "LLM module not available. Ensure Formalizer/llm_openrouter.py is present."
            )
        elif args.parser in {"llm_text", "hybrid_text"} and extract_constructed_text is None:
            llm_error = OpenRouterError(
                "LLM text module not available. Ensure Formalizer/llm_openrouter.py is present."
            )
        elif not args.openrouter_api_key:
            llm_error = OpenRouterError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY or --openrouter-api-key."
            )
        else:
            try:
                if args.parser in {"llm", "hybrid"}:
                    structured = extract_structured_problem(
                        content,
                        api_key=args.openrouter_api_key,
                        model=args.openrouter_model,
                        timeout_sec=args.openrouter_timeout,
                        site_url=args.openrouter_site_url,
                        app_name=args.openrouter_app_name,
                    )
                    nodes, edges, sentences = extract_graph_from_structured(
                        structured,
                        problem_id=args.document_id,
                        problem_text=content,
                    )
                    backend_name = f"openrouter-json:{args.openrouter_model}"
                else:
                    constructed_text = extract_constructed_text(
                        content,
                        api_key=args.openrouter_api_key,
                        model=args.openrouter_model,
                        timeout_sec=args.openrouter_timeout,
                        site_url=args.openrouter_site_url,
                        app_name=args.openrouter_app_name,
                    )
                    if args.constructed_out:
                        args.constructed_out.parent.mkdir(parents=True, exist_ok=True)
                        args.constructed_out.write_text(constructed_text + "\n", encoding="utf-8")
                        print(f"[info] Constructed summary saved: {args.constructed_out}")
                    nodes, edges, sentences = extract_graph_from_constructed_text(
                        constructed_text,
                        problem_id=args.document_id,
                        problem_text=content,
                    )
                    backend_name = f"openrouter-text:{args.openrouter_model}"
            except Exception as exc:
                llm_error = exc

    if not backend_name:
        if args.parser in {"llm", "llm_text"}:
            raise RuntimeError(f"LLM extraction failed: {llm_error}")
        if llm_error is not None and args.parser in {"hybrid", "hybrid_text"}:
            print(f"[warn] LLM extraction failed, fallback to semantic parser: {llm_error}")
        extractor = SemanticExtractor()
        nodes, edges, sentences = extractor.extract(content, problem_id=args.document_id)
        backend_name = extractor.nlp_model_name

    print_preview(backend_name, sentences, nodes, edges)

    if args.dry_run:
        print("[info] Dry run enabled, skip Neo4j write.")
        return

    if not args.password:
        args.password = getpass("Neo4j password: ").strip()
    if not args.password:
        raise ValueError("Missing Neo4j password. Pass --password or set it in config.")

    push_to_neo4j(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        nodes=nodes,
        edges=edges,
        clear_db=args.clear,
    )
    print("[done] Semantic graph has been written to Neo4j.")
    print("[hint] In Neo4j Browser, run: MATCH (n)-[r]->(m) RETURN n,r,m")


if __name__ == "__main__":
    main()
