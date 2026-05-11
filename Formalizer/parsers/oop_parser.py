from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from graph_model import (
    Edge,
    Node,
    dedupe_edges,
    node_key,
    normalize_text,
    parse_number,
    safe_neo4j_label,
    safe_neo4j_rel_type,
    to_neo4j_properties,
)
from parsers.compact_parser import extract_graph_from_compact


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SOLUTION_CLASSES = {"Answer", "Solution", "ComputationStep", "CalculationStep", "Derivation"}
SOLUTION_RELATIONS = {"INPUT_TO", "OUTPUTS", "SOLVES", "COMPUTES", "CALCULATES", "DERIVES"}


def normalize_object_id(raw: object, fallback: str) -> str:
    text = normalize_text(raw or fallback)
    object_id = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return object_id or fallback


def _first_number_from_props(props: Dict[str, object]) -> Optional[float]:
    for key in ("value", "amount", "quantity", "price", "cost", "duration", "multiplier"):
        value = props.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parsed = parse_number(value)
            if parsed is not None:
                return parsed
    return None


def _ensure_class_node(nodes: Dict[str, Node], class_name: str) -> str:
    key = f"class:{class_name.lower()}"
    if key not in nodes:
        nodes[key] = Node(
            key=key,
            label="Class",
            name=class_name,
            properties={"class_name": class_name},
        )
    return key


def extract_graph_from_oop_model(
    model_data: Dict[str, Any],
    *,
    problem_id: str,
    problem_text: str,
) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
    compact = model_data.get("compact")
    if isinstance(compact, str) and compact.strip():
        try:
            return extract_graph_from_compact(
                compact.strip(),
                problem_id=problem_id,
                problem_text=problem_text,
            )
        except ValueError:
            pass

    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    sentences: List[str] = []
    object_keys: Dict[str, str] = {}

    problem_key = node_key("Problem", problem_id)
    nodes[problem_key] = Node(
        key=problem_key,
        label="Problem",
        name=problem_id,
        properties={"text": normalize_text(problem_text)},
    )

    def collect_sentence(raw: object) -> None:
        if raw is None:
            return
        sentence = normalize_text(raw)
        if sentence and sentence not in sentences:
            sentences.append(sentence)

    def ensure_object(raw_obj: object, fallback_prefix: str = "object") -> Optional[str]:
        if isinstance(raw_obj, str):
            object_id = normalize_object_id(raw_obj, fallback_prefix)
            if object_id in object_keys:
                return object_keys[object_id]
            class_name = "Thing"
            name = normalize_text(raw_obj)
            attrs: Dict[str, object] = {}
        elif isinstance(raw_obj, dict):
            object_id = normalize_object_id(
                raw_obj.get("id") or raw_obj.get("object_id") or raw_obj.get("name"),
                f"{fallback_prefix}_{len(object_keys) + 1}",
            )
            if object_id in object_keys:
                return object_keys[object_id]
            class_name = safe_neo4j_label(
                raw_obj.get("class") or raw_obj.get("type") or raw_obj.get("label"),
                default="Thing",
            )
            if class_name in SOLUTION_CLASSES:
                return None
            name = normalize_text(raw_obj.get("name") or object_id)
            attrs = raw_obj.get("attributes") if isinstance(raw_obj.get("attributes"), dict) else {}
        else:
            return None

        class_key = _ensure_class_node(nodes, class_name)
        key = f"object:{object_id}"
        props = to_neo4j_properties(attrs)
        props.update({"object_id": object_id, "class_name": class_name})
        nodes[key] = Node(key=key, label=class_name, name=name, properties=props)
        object_keys[object_id] = key

        edges.append(Edge(key, class_key, "INSTANCE_OF", {}))
        edges.append(Edge(problem_key, key, "CONTAINS_OBJECT", {}))
        return key

    def resolve_ref(raw_ref: object, fallback_prefix: str) -> Optional[str]:
        if isinstance(raw_ref, dict):
            return ensure_object(raw_ref, fallback_prefix=fallback_prefix)
        ref_id = normalize_object_id(raw_ref, fallback_prefix)
        if ref_id in object_keys:
            return object_keys[ref_id]
        if raw_ref is None or normalize_text(raw_ref).lower() in {"", "none", "null"}:
            return None
        return ensure_object(str(raw_ref), fallback_prefix=fallback_prefix)

    raw_classes = model_data.get("classes")
    for class_def in raw_classes if isinstance(raw_classes, list) else []:
        if not isinstance(class_def, dict):
            continue
        class_name = safe_neo4j_label(class_def.get("name"), default="Thing")
        if class_name in SOLUTION_CLASSES:
            continue
        class_key = _ensure_class_node(nodes, class_name)
        props: Dict[str, object] = {}
        if class_def.get("description"):
            props["description"] = normalize_text(class_def["description"])
        if class_def.get("attributes") is not None:
            props["attributes_json"] = json.dumps(class_def.get("attributes"), ensure_ascii=False, sort_keys=True)
        nodes[class_key].properties.update(to_neo4j_properties(props))

    raw_objects = model_data.get("objects")
    for idx, raw_obj in enumerate(raw_objects if isinstance(raw_objects, list) else [], start=1):
        ensure_object(raw_obj, fallback_prefix=f"object_{idx}")

    question = model_data.get("question")
    question_key: Optional[str] = None
    if isinstance(question, dict):
        question_obj = dict(question)
        question_obj.setdefault("class", "Question")
        question_obj.setdefault("name", question.get("asks") or "question")
        question_obj.setdefault(
            "attributes",
            question.get("attributes") if isinstance(question.get("attributes"), dict) else {},
        )
        question_key = ensure_object(question_obj, fallback_prefix="question")
        collect_sentence(question.get("sentence") or question.get("asks"))
        if question_key:
            edges.append(
                Edge(
                    source_key=problem_key,
                    target_key=question_key,
                    rel_type="HAS_QUESTION",
                    properties=to_neo4j_properties({"asks": question.get("asks")}),
                )
            )
            target_key = resolve_ref(question.get("target"), fallback_prefix="question_target")
            if target_key:
                edges.append(
                    Edge(
                        source_key=question_key,
                        target_key=target_key,
                        rel_type="ASKS_FOR",
                        properties=to_neo4j_properties({"asks": question.get("asks")}),
                    )
                )

    raw_relationships = model_data.get("relationships")
    for idx, rel in enumerate(raw_relationships if isinstance(raw_relationships, list) else [], start=1):
        if not isinstance(rel, dict):
            continue
        rel_type = safe_neo4j_rel_type(rel.get("type") or rel.get("relation"))
        if rel_type in SOLUTION_RELATIONS:
            continue

        source_key = resolve_ref(rel.get("source") or rel.get("from"), fallback_prefix=f"source_{idx}")
        target_key = resolve_ref(rel.get("target") or rel.get("to"), fallback_prefix=f"target_{idx}")
        if not source_key or not target_key:
            continue

        raw_attrs: Dict[str, object] = {}
        if isinstance(rel.get("attributes"), dict):
            raw_attrs.update(rel["attributes"])
        for key, value in rel.items():
            if key in {"type", "relation", "source", "from", "target", "to", "attributes"}:
                continue
            raw_attrs[key] = value

        props = to_neo4j_properties(raw_attrs)
        props.setdefault("sentence_idx", idx)
        if "sentence" in props:
            collect_sentence(props["sentence"])
        number = _first_number_from_props(props)
        if number is not None and "value" not in props and "multiplier" not in props:
            props["value"] = number
        if "item" not in props and "unit" in props:
            props["item"] = props["unit"]

        edges.append(Edge(source_key, target_key, rel_type, props))

    if not sentences:
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalize_text(problem_text)) if s.strip()]
    return nodes, dedupe_edges(edges), sentences
