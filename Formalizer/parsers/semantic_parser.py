from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from graph_model import Edge, Node, dedupe_edges, node_key, normalize_text, parse_number


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER_PATTERN = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
NAME_PATTERN = r"(?:[A-Z][a-zA-Z]*|he|she|they|him|her|them)"
ITEM_PATTERN = r"[a-zA-Z]+(?:\s+[a-zA-Z]+)?"

HAS_PATTERN = re.compile(
    rf"(?P<subject>{NAME_PATTERN})\s+(?:has|have|had|owns?)\s+(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_PATTERN})",
    flags=re.IGNORECASE,
)
GAIN_PATTERN = re.compile(
    rf"(?P<subject>{NAME_PATTERN})\s+(?:buys?|bought|gets?|got|earns?|earned|receives?|received|grabs?)\s+(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_PATTERN})",
    flags=re.IGNORECASE,
)
LOSS_PATTERN = re.compile(
    rf"(?P<subject>{NAME_PATTERN})\s+(?:loses?|lost|spends?|spent|uses?|used|eats?|ate|sells?|sold)\s+(?P<value>{NUMBER_PATTERN})\s+(?P<item>{ITEM_PATTERN})",
    flags=re.IGNORECASE,
)
QUESTION_PATTERN = re.compile(
    rf"how\s+(?:many|much)\s+(?P<item>{ITEM_PATTERN}).{{0,100}}?(?:does|do|did|has|have)\s+(?P<subject>{NAME_PATTERN})",
    flags=re.IGNORECASE,
)

ITEM_STOPWORDS = {
    "a",
    "an",
    "the",
    "per",
    "day",
    "week",
    "month",
    "year",
    "morning",
    "afternoon",
    "every",
    "after",
    "on",
    "in",
    "for",
}
PRONOUNS = {"he", "she", "they", "him", "her", "them"}


def _clean_person(raw: str, last_person: Optional[str]) -> Optional[str]:
    text = normalize_text(raw).strip(" ,.;:!?")
    if not text:
        return None
    if text.lower() in PRONOUNS:
        return last_person
    return text[:1].upper() + text[1:]


def _clean_item(raw: str) -> Optional[str]:
    tokens = normalize_text(raw).lower().strip(" ,.;:!?").split()
    while tokens and tokens[0] in ITEM_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in ITEM_STOPWORDS:
        tokens = tokens[:-1]
    if not tokens:
        return None
    if len(tokens) > 2:
        tokens = tokens[:2]
    item = " ".join(tokens)
    if item.endswith("s") and len(item) > 3 and not item.endswith("ss"):
        item = item[:-1]
    return item


class SemanticParser:
    name = "semantic-rules"

    def __init__(self) -> None:
        self.last_person: Optional[str] = None

    def _ensure_node(self, nodes: Dict[str, Node], label: str, name: str) -> str:
        key = node_key(label, name)
        if key not in nodes:
            nodes[key] = Node(key=key, label=label, name=name)
        return key

    def _add_quantity_edge(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
        *,
        subject: str,
        item: str,
        value: Optional[float],
        rel_type: str,
        sentence: str,
        sentence_idx: int,
    ) -> None:
        subject_key = self._ensure_node(nodes, "Person", subject)
        item_key = self._ensure_node(nodes, "Item", item)
        props = {"sentence": sentence, "sentence_idx": sentence_idx, "item": item}
        if value is not None:
            props["value"] = value
        edges.append(Edge(subject_key, item_key, rel_type, props))

    def extract(self, text: str, problem_id: str) -> Tuple[Dict[str, Node], List[Edge], List[str]]:
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalize_text(text)) if s.strip()]

        problem_key = node_key("Problem", problem_id)
        nodes[problem_key] = Node(key=problem_key, label="Problem", name=problem_id)

        for idx, sentence in enumerate(sentences, start=1):
            for pattern, rel_type in (
                (HAS_PATTERN, "HAS_QUANTITY"),
                (GAIN_PATTERN, "GAINS_QUANTITY"),
                (LOSS_PATTERN, "LOSES_QUANTITY"),
            ):
                for match in pattern.finditer(sentence):
                    subject = _clean_person(match.group("subject"), self.last_person)
                    item = _clean_item(match.group("item"))
                    value = parse_number(match.group("value"))
                    if not subject or not item:
                        continue
                    if subject.lower() not in PRONOUNS:
                        self.last_person = subject
                    self._add_quantity_edge(
                        nodes,
                        edges,
                        subject=subject,
                        item=item,
                        value=value,
                        rel_type=rel_type,
                        sentence=sentence,
                        sentence_idx=idx,
                    )

            for match in QUESTION_PATTERN.finditer(sentence):
                item = _clean_item(match.group("item"))
                subject = _clean_person(match.group("subject"), self.last_person)
                if not item:
                    continue
                item_key = self._ensure_node(nodes, "Item", item)
                source_key = problem_key
                if subject:
                    source_key = self._ensure_node(nodes, "Person", subject)
                edges.append(Edge(source_key, item_key, "ASKS_FOR", {"sentence": sentence, "sentence_idx": idx}))

        return nodes, dedupe_edges(edges), sentences
