from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graph_model import Edge, Node, dedupe_edges, node_key, normalize_text, parse_number


@dataclass
class CompactTree:
    name: str
    children: List["CompactTree"]


def _split_duration(raw: str) -> Tuple[Optional[float], Optional[str]]:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-zA-Z]+)", raw.strip())
    if not match:
        return None, None
    return float(match.group(1)), match.group(2).lower()


def _humanize(raw: str) -> str:
    return raw.strip().replace("_", " ")


class CompactExpressionParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> CompactTree:
        if not self.text:
            raise ValueError("Compact expression is empty.")
        node = self._parse_node()
        self._skip_spaces()
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected compact expression suffix: {self.text[self.pos:]}")
        return node

    def _skip_spaces(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _parse_node(self) -> CompactTree:
        self._skip_spaces()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in "(),":
            self.pos += 1
        name = self.text[start:self.pos].strip()
        if not name:
            raise ValueError("Empty compact node.")

        children: List[CompactTree] = []
        self._skip_spaces()
        if self.pos < len(self.text) and self.text[self.pos] == "(":
            self.pos += 1
            while True:
                self._skip_spaces()
                if self.pos >= len(self.text):
                    raise ValueError("Unclosed compact node.")
                if self.text[self.pos] == ")":
                    self.pos += 1
                    break
                children.append(self._parse_node())
                self._skip_spaces()
                if self.pos < len(self.text) and self.text[self.pos] == ",":
                    self.pos += 1
                    continue
                if self.pos < len(self.text) and self.text[self.pos] == ")":
                    self.pos += 1
                    break
                raise ValueError(f"Expected ',' or ')' near: {self.text[self.pos:]}")
        return CompactTree(name=name, children=children)


def _money_value(raw: str) -> Optional[float]:
    text = raw.strip()
    if not text.startswith("$"):
        return None
    return parse_number(text[1:])


def _numeric_child(tree: CompactTree) -> Optional[float]:
    if len(tree.children) != 1 or tree.children[0].children:
        return None
    return parse_number(tree.children[0].name)


def _money_child(tree: CompactTree) -> Optional[float]:
    if len(tree.children) != 1 or tree.children[0].children:
        return None
    return _money_value(tree.children[0].name)


def _numeric_or_word_amount(raw: str) -> Optional[float]:
    return parse_number(raw.replace("_", " "))


def _classify(tree: CompactTree) -> Tuple[str, str, Dict[str, object], bool]:
    raw = tree.name.strip()
    lower = raw.lower()

    if lower.startswith("target:"):
        target_raw = raw.split(":", 1)[1]
        amount, unit = _split_duration(target_raw)
        props: Dict[str, object] = {"role": "target"}
        name = f"target: {_humanize(target_raw)}"
        if amount is not None and unit:
            props.update({"amount": amount, "unit": unit})
            name = f"target: {amount:g} {unit}"
        return "TargetPeriod", name, props, False

    amount, unit = _split_duration(raw)
    if amount is not None and unit:
        label = "DailyRoutine" if amount == 1 and unit.startswith("day") else "Period"
        return label, f"{amount:g} {unit}", {"amount": amount, "unit": unit}, False

    duration_with_child = re.fullmatch(r"([a-zA-Z]+)_([a-zA-Z]+)", lower)
    child_amount = _numeric_child(tree)
    if duration_with_child and child_amount is not None:
        raw_unit = duration_with_child.group(2)
        if raw_unit.rstrip("s") in {"day", "week", "month", "year"}:
            unit = raw_unit.rstrip("s")
            return "Period", f"{child_amount:g} {unit}", {"amount": child_amount, "unit": unit}, True

    word_duration = re.fullmatch(r"([a-zA-Z]+)_([a-zA-Z]+)", lower)
    if word_duration:
        word_amount = _numeric_or_word_amount(word_duration.group(1))
        raw_unit = word_duration.group(2)
        if word_amount is not None and raw_unit.rstrip("s") in {"day", "week", "month", "year"}:
            unit = raw_unit.rstrip("s")
            return "Period", f"{word_amount:g} {unit}", {"amount": word_amount, "unit": unit}, False

    price = _money_value(raw)
    if price is not None:
        return "MoneyAmount", f"${price:g}", {"amount": price, "unit": "usd"}, False

    price = _money_child(tree)
    if price is not None:
        item = _humanize(raw)
        return "ItemPrice", f"{item} (${price:g})", {"item": item, "price": price, "unit": "usd"}, True

    amount = _numeric_child(tree)
    per_match = re.fullmatch(r"(.+)_per_([a-zA-Z]+)", lower)
    if amount is not None and per_match:
        item = _humanize(per_match.group(1))
        period = per_match.group(2)
        return "Rate", f"{amount:g} {item}/{period}", {"amount": amount, "item": item, "period": period}, True

    per_match = re.fullmatch(r"(\d+(?:\.\d+)?)_(.+)_per_([a-zA-Z]+)", lower)
    if per_match:
        amount = float(per_match.group(1))
        item = _humanize(per_match.group(2))
        period = per_match.group(3)
        return "Rate", f"{amount:g} {item}/{period}", {"amount": amount, "item": item, "period": period}, False

    amount = parse_number(raw)
    if amount is not None:
        return "Quantity", f"{amount:g}", {"amount": amount}, False

    if raw[:1].isupper():
        return "Person", _humanize(raw), {}, False

    return "Thing", _humanize(raw), {}, False


def _edge_type(parent: Node, child: Node) -> str:
    if parent.label == "TargetPeriod":
        return "HAS_BASE"
    if parent.label == "DailyRoutine":
        return "HAS_ACTOR" if child.label == "Person" else "INCLUDES"
    if parent.label == "Person" and child.label == "ItemPrice":
        return "BUYS"
    if parent.label == "Person" and child.label == "Rate":
        return "HAS_RATE"
    return "CONTAINS"


def extract_graph_from_compact(
    compact_text: str,
    *,
    problem_id: str,
    problem_text: str,
) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
    tree = CompactExpressionParser(compact_text).parse()
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []

    problem_key = node_key("Problem", problem_id)
    nodes[problem_key] = Node(
        key=problem_key,
        label="Problem",
        name=problem_id,
        properties={"text": normalize_text(problem_text), "compact": compact_text},
    )

    def add_tree(current: CompactTree, parent_key: Optional[str], index_path: str) -> str:
        label, name, props, collapse_single_value_child = _classify(current)
        key = f"compact:{index_path}:{label.lower()}:{normalize_text(name).lower()}"
        props = {**props, "compact_token": current.name}
        nodes[key] = Node(key=key, label=label, name=name, properties=props)

        if parent_key is None:
            edges.append(Edge(problem_key, key, "HAS_COMPACT_ROOT", {}))
        else:
            parent_node = nodes[parent_key]
            edges.append(Edge(parent_key, key, _edge_type(parent_node, nodes[key]), {}))

        if collapse_single_value_child:
            return key

        for child_idx, child in enumerate(current.children, start=1):
            add_tree(child, key, f"{index_path}_{child_idx}")
        return key

    add_tree(tree, None, "1")
    return nodes, dedupe_edges(edges), [compact_text]
