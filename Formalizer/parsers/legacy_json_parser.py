from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from graph_model import Edge, Node, dedupe_edges, node_key, normalize_text, parse_number


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
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
    label = normalize_text(raw)
    return CANONICAL_LABELS.get(label.lower())


def normalize_relation_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    rel_type = re.sub(r"[^A-Z0-9]+", "_", normalize_text(raw).upper()).strip("_")
    if rel_type in ALLOWED_RELATIONS:
        return rel_type
    return None


def ref_to_node_key(
    ref: object,
    *,
    nodes: Dict[str, Node],
    default_label: str = "Item",
) -> Optional[str]:
    label: Optional[str] = None
    name: Optional[str] = None

    if isinstance(ref, dict):
        label = canonical_label(ref.get("type") or ref.get("label"))
        if ref.get("name") is not None:
            name = normalize_text(ref["name"])
    elif isinstance(ref, str):
        text = normalize_text(ref)
        if ":" in text:
            raw_label, raw_name = text.split(":", 1)
            label = canonical_label(raw_label)
            name = normalize_text(raw_name)
        else:
            label = canonical_label(default_label)
            name = text

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

    raw_entities = structured.get("entities")
    for entity in raw_entities if isinstance(raw_entities, list) else []:
        if not isinstance(entity, dict):
            continue
        label = canonical_label(entity.get("type") or entity.get("label"))
        name = normalize_text(entity.get("name", ""))
        if not label or not name:
            continue
        key = node_key(label, name)
        nodes.setdefault(key, Node(key=key, label=label, name=name))

    raw_facts: List[Dict[str, Any]] = []
    if isinstance(structured.get("facts"), list):
        raw_facts.extend([fact for fact in structured["facts"] if isinstance(fact, dict)])
    if isinstance(structured.get("question"), dict):
        raw_facts.append(structured["question"])

    for idx, fact in enumerate(raw_facts, start=1):
        rel_type = normalize_relation_type(fact.get("relation") or fact.get("type"))
        if not rel_type:
            continue
        source_key = ref_to_node_key(fact.get("source"), nodes=nodes, default_label="Problem")
        target_key = ref_to_node_key(fact.get("target"), nodes=nodes, default_label="Item")
        if not source_key or not target_key:
            continue

        props: Dict[str, object] = {"sentence_idx": idx}
        if fact.get("sentence"):
            sentence = normalize_text(fact["sentence"])
            props["sentence"] = sentence
            if sentence not in sentences:
                sentences.append(sentence)

        value_raw = fact.get("multiplier", fact.get("value")) if rel_type == "RATE_MULTIPLIER" else fact.get("value")
        if value_raw is not None:
            if isinstance(value_raw, (int, float)):
                props["multiplier" if rel_type == "RATE_MULTIPLIER" else "value"] = float(value_raw)
            else:
                parsed = parse_number(value_raw)
                if parsed is not None:
                    props["multiplier" if rel_type == "RATE_MULTIPLIER" else "value"] = parsed

        if fact.get("unit"):
            props["item"] = normalize_text(fact["unit"]).lower()
        if fact.get("period"):
            props["period"] = normalize_text(fact["period"]).lower()

        edges.append(Edge(source_key, target_key, rel_type, props))

    if not sentences:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalize_text(problem_text)) if s.strip()]
    return nodes, dedupe_edges(edges), sentences
