from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from graph_model import Edge, Node, normalize_text, parse_number
from parsers.legacy_json_parser import canonical_label, extract_graph_from_structured


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
) -> Tuple[Dict[str, Node], list[Edge], list[str]]:
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

        if head == "ENTITY" and len(parts) >= 3:
            structured["entities"].append({"type": parts[1], "name": parts[2]})
            continue

        if head not in {"FACT", "QUESTION"} or len(parts) < 4:
            continue

        value_raw = parts[4] if len(parts) > 4 else "-"
        unit_raw = parts[5] if len(parts) > 5 else "-"
        period_raw = parts[6] if len(parts) > 6 else "-"
        parsed_value = parse_number(value_raw) if value_raw and value_raw != "-" else None

        fact: Dict[str, Any] = {
            "relation": parts[1],
            "source": _to_ref_dict(parts[2]) or {"type": "Problem", "name": problem_id},
            "target": _to_ref_dict(parts[3]) or {"type": "Problem", "name": problem_id},
            "value": parsed_value,
            "unit": normalize_text(unit_raw) if unit_raw and unit_raw != "-" else None,
            "period": normalize_text(period_raw) if period_raw and period_raw != "-" else None,
            "sentence": line,
        }

        if head == "QUESTION":
            structured["question"] = fact
        else:
            structured["facts"].append(fact)

    return extract_graph_from_structured(
        structured,
        problem_id=problem_id,
        problem_text=problem_text,
    )
