from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - depends on local environment
    GraphDatabase = None


SAFE_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
SAFE_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class Node:
    key: str
    label: str
    name: str
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    source_key: str
    target_key: str
    rel_type: str
    properties: Dict[str, object] = field(default_factory=dict)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def parse_number(value: object) -> Optional[float]:
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
    value_norm = normalize_text(value).lower().replace("$", "")
    if not value_norm:
        return None
    if value_norm in word_to_number:
        return float(word_to_number[value_norm])
    raw = value_norm
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif re.fullmatch(r"\d+,\d+", raw):
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def node_key(label: str, name: str) -> str:
    return f"{label.lower()}:{normalize_text(name).lower()}"


def safe_neo4j_label(raw: object, default: str = "Thing") -> str:
    text = normalize_text(raw or default)
    parts = re.findall(r"[A-Za-z0-9]+", text)
    label = "".join(part[:1].upper() + part[1:] for part in parts) or default
    if not label[0].isalpha():
        label = f"{default}{label}"
    return label


def safe_neo4j_rel_type(raw: object, default: str = "RELATED_TO") -> str:
    text = normalize_text(raw or default).upper()
    rel_type = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    if not rel_type or not rel_type[0].isalpha():
        return default
    return rel_type


def safe_property_key(raw: object, default: str = "property") -> str:
    text = normalize_text(raw or default).lower()
    key = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if not key:
        key = default
    if key[0].isdigit():
        key = f"p_{key}"
    return key


def _neo4j_property_value(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = normalize_text(value)
        return text if text else None
    if isinstance(value, list):
        converted = [_neo4j_property_value(item) for item in value]
        converted = [item for item in converted if item is not None]
        if all(isinstance(item, (bool, int, float, str)) for item in converted):
            return converted
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_text(value)


def to_neo4j_properties(raw: object) -> Dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    props: Dict[str, object] = {}
    for key, value in raw.items():
        prop_key = safe_property_key(key)
        prop_value = _neo4j_property_value(value)
        if prop_value is not None:
            props[prop_key] = prop_value
    return props


def dedupe_edges(edges: List[Edge]) -> List[Edge]:
    seen = set()
    deduped: List[Edge] = []
    for edge in edges:
        signature = (
            edge.source_key,
            edge.target_key,
            edge.rel_type,
            json.dumps(to_neo4j_properties(edge.properties), sort_keys=True),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(edge)
    return deduped


def build_quantity_node_from_edge(edge: Edge) -> Optional[Node]:
    value = edge.properties.get("value")
    multiplier = edge.properties.get("multiplier")
    unit = edge.properties.get("item") or edge.properties.get("unit")

    if isinstance(value, (int, float)):
        amount = float(value)
        kind = "value"
    elif isinstance(multiplier, (int, float)):
        amount = float(multiplier)
        kind = "multiplier"
        unit = unit or "times"
    else:
        return None

    unit_text = str(unit) if unit else None
    name = f"{amount:g} {unit_text}" if unit_text else f"{amount:g}"
    key = (
        f"quantity:{edge.source_key}|{edge.rel_type}|{edge.target_key}|"
        f"{amount:g}|{unit_text or '-'}|{kind}"
    )
    return Node(key=key, label="Quantity", name=name)


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
                props = to_neo4j_properties(node.properties)
                props.pop("key", None)
                props.pop("name", None)
                session.run(
                    f"""
                    MERGE (n:{node.label} {{key: $key}})
                    SET n.name = $name
                    SET n += $props
                    """,
                    key=node.key,
                    name=node.name,
                    props=props,
                ).consume()

            for edge in edges:
                if not SAFE_REL_TYPE_RE.match(edge.rel_type):
                    raise ValueError(f"Unsafe Neo4j relationship type: {edge.rel_type}")
                props = to_neo4j_properties(edge.properties)
                session.run(
                    f"""
                    MATCH (a {{key: $source_key}})
                    MATCH (b {{key: $target_key}})
                    CREATE (a)-[r:{edge.rel_type}]->(b)
                    SET r += $props
                    """,
                    source_key=edge.source_key,
                    target_key=edge.target_key,
                    props=props,
                ).consume()

                quantity_node = build_quantity_node_from_edge(edge)
                if quantity_node is None:
                    continue
                amount = edge.properties.get("value", edge.properties.get("multiplier"))
                unit = edge.properties.get("item", edge.properties.get("unit"))
                session.run(
                    """
                    MERGE (q:Quantity {key: $q_key})
                    SET q.name = $q_name,
                        q.amount = $amount,
                        q.unit = $unit,
                        q.rel_type = $rel_type
                    """,
                    q_key=quantity_node.key,
                    q_name=quantity_node.name,
                    amount=float(amount),
                    unit=unit,
                    rel_type=edge.rel_type,
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


def print_preview(
    backend_name: str,
    sentences: List[str],
    nodes: Dict[str, Node],
    edges: List[Edge],
) -> None:
    label_counts: Dict[str, int] = {}
    rel_counts: Dict[str, int] = {}

    for node in nodes.values():
        label_counts[node.label] = label_counts.get(node.label, 0) + 1
    for edge in edges:
        rel_counts[edge.rel_type] = rel_counts.get(edge.rel_type, 0) + 1

    print(f"[info] Parser backend: {backend_name}")
    print(f"[info] Sentences: {len(sentences)}")
    print(f"[info] Nodes: {len(nodes)}")
    print(f"[info] Edges: {len(edges)}")

    if label_counts:
        print("[info] Node labels:")
        for label in sorted(label_counts):
            print(f"  - {label}: {label_counts[label]}")
    if rel_counts:
        print("[info] Relation types:")
        for rel_type in sorted(rel_counts):
            print(f"  - {rel_type}: {rel_counts[rel_type]}")

    if edges:
        print("[preview] First edges:")
        for edge in edges[:12]:
            props = to_neo4j_properties(edge.properties)
            extras = []
            for key in ("value", "amount", "unit", "item"):
                if key in props:
                    extras.append(f"{key}={props[key]}")
            extras_text = f" ({', '.join(extras)})" if extras else ""
            print(f"  - {edge.rel_type}: {edge.source_key} -> {edge.target_key}{extras_text}")


def build_debug_payload(
    *,
    backend_name: str,
    problem_id: str,
    problem_text: str,
    sentences: List[str],
    nodes: Dict[str, Node],
    edges: List[Edge],
    llm_error: Optional[Exception] = None,
) -> Dict[str, object]:
    return {
        "backend": backend_name,
        "problem_id": problem_id,
        "problem_text": normalize_text(problem_text),
        "llm_error": str(llm_error) if llm_error is not None else None,
        "sentences": sentences,
        "classes": [
            {
                "key": node.key,
                "name": node.name,
                "properties": to_neo4j_properties(node.properties),
            }
            for node in nodes.values()
            if node.label == "Class"
        ],
        "objects": [
            {
                "key": node.key,
                "label": node.label,
                "name": node.name,
                "properties": to_neo4j_properties(node.properties),
            }
            for node in nodes.values()
            if node.label != "Class"
        ],
        "relationships": [
            {
                "source_key": edge.source_key,
                "source_name": nodes[edge.source_key].name if edge.source_key in nodes else edge.source_key,
                "type": edge.rel_type,
                "target_key": edge.target_key,
                "target_name": nodes[edge.target_key].name if edge.target_key in nodes else edge.target_key,
                "properties": to_neo4j_properties(edge.properties),
            }
            for edge in edges
        ],
    }


def write_debug_json(path: Optional[Path], payload: Dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[info] Debug graph saved: {path}")
