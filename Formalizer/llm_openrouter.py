#!/usr/bin/env python3
"""
OpenRouter client for structured extraction of math word problems.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OpenRouterError(RuntimeError):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()

    # Prefer fenced JSON blocks when present.
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    # Fallback: find first plausible JSON object.
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise OpenRouterError("LLM response did not contain a JSON object.")
        candidate = candidate[start : end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"Failed to decode JSON from LLM response: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenRouterError("Structured output must be a JSON object.")
    return data


def _build_prompt(problem_text: str) -> str:
    return (
        "Extract the math problem into STRICT JSON only.\n"
        "No markdown, no explanations, only one JSON object.\n\n"
        "Schema:\n"
        "{\n"
        '  "entities": [\n'
        '    {"type":"Person|Item|Period|Problem","name":"string"}\n'
        "  ],\n"
        '  "facts": [\n'
        "    {\n"
        '      "relation":"HAS_QUANTITY|GAINS_QUANTITY|LOSES_QUANTITY|TRANSFER_TO|RATE_MULTIPLIER|UNIT_RATE|OWNS_ITEM|HAS_VALUE|VALUE_DECREASE_PER_PERIOD|ASKS_FOR|ASKS_WORTH_AFTER",\n'
        '      "source":{"type":"Person|Item|Period|Problem","name":"string"},\n'
        '      "target":{"type":"Person|Item|Period|Problem","name":"string"},\n'
        '      "value": number or null,\n'
        '      "unit":"string or null",\n'
        '      "period":"string or null",\n'
        '      "sentence":"string or null"\n'
        "    }\n"
        "  ],\n"
        '  "question": {\n'
        '    "relation":"ASKS_FOR|ASKS_WORTH_AFTER",\n'
        '    "source":{"type":"Person|Problem","name":"string"},\n'
        '    "target":{"type":"Item|Person|Period","name":"string"},\n'
        '    "value": number or null,\n'
        '    "unit":"string or null",\n'
        '    "period":"string or null",\n'
        '    "sentence":"string or null"\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Keep names short and canonical.\n"
        "- Put numeric facts in value whenever possible.\n"
        "- If unknown, use null.\n"
        "- Include question as the question object.\n\n"
        f"Problem:\n{problem_text.strip()}\n"
    )


def _build_constructed_text_prompt(problem_text: str) -> str:
    return (
        "Convert the math word problem into a strict line-based DSL.\n"
        "Return plain text only. No markdown fences.\n\n"
        "Line formats:\n"
        "ENTITY|<Label>|<Name>\n"
        "FACT|<Relation>|<SourceLabel>:<SourceName>|<TargetLabel>:<TargetName>|<Value>|<Unit>|<Period>\n"
        "QUESTION|<Relation>|<SourceLabel>:<SourceName>|<TargetLabel>:<TargetName>|<Value>|<Unit>|<Period>\n\n"
        "Allowed Label: Person, Item, Period, Problem\n"
        "Allowed Relation: HAS_QUANTITY, GAINS_QUANTITY, LOSES_QUANTITY, TRANSFER_TO, RATE_MULTIPLIER, UNIT_RATE, OWNS_ITEM, HAS_VALUE, VALUE_DECREASE_PER_PERIOD, ASKS_FOR, ASKS_WORTH_AFTER, LESS_THAN_BY, MORE_THAN_BY, TIMES_OF\n"
        "Use '-' for unknown Value/Unit/Period.\n"
        "Keep names short canonical.\n"
        "Always include at least one QUESTION line.\n\n"
        "Example:\n"
        "ENTITY|Person|An\n"
        "ENTITY|Person|Binh\n"
        "ENTITY|Item|Keo\n"
        "FACT|HAS_QUANTITY|Person:An|Item:Keo|50|keo|-\n"
        "FACT|LESS_THAN_BY|Person:Binh|Person:An|3|keo|-\n"
        "QUESTION|ASKS_FOR|Person:Binh|Item:Keo|-|-|-\n\n"
        f"Problem:\n{problem_text.strip()}\n"
    )


def _call_openrouter_text(
    *,
    prompt: str,
    api_key: str,
    model: str,
    timeout_sec: int = 90,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
    as_json_object: bool = False,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You convert math word problems into deterministic structured formats."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if as_json_object:
        payload["response_format"] = {"type": "json_object"}

    request_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **({"HTTP-Referer": site_url} if site_url else {}),
            **({"X-Title": app_name} if app_name else {}),
        },
        data=request_data,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = "<unavailable>"
        raise OpenRouterError(
            f"OpenRouter HTTP error {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenRouterError(f"OpenRouter network error: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"Invalid JSON from OpenRouter: {exc}") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict):
                text += str(part.get("text", ""))
            else:
                text += str(part)
    else:
        text = str(content or "")
    if not text.strip():
        raise OpenRouterError("OpenRouter response had empty content.")
    return text.strip()


def extract_structured_problem(
    problem_text: str,
    *,
    api_key: str,
    model: str,
    timeout_sec: int = 90,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not api_key:
        raise OpenRouterError("Missing OpenRouter API key.")
    if not model:
        raise OpenRouterError("Missing OpenRouter model name.")

    text = _call_openrouter_text(
        prompt=_build_prompt(problem_text),
        api_key=api_key,
        model=model,
        timeout_sec=timeout_sec,
        site_url=site_url,
        app_name=app_name,
        as_json_object=True,
    )

    structured = _extract_json_object(text)
    if "facts" not in structured:
        structured["facts"] = []
    if "entities" not in structured:
        structured["entities"] = []
    return structured


def extract_constructed_text(
    problem_text: str,
    *,
    api_key: str,
    model: str,
    timeout_sec: int = 90,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> str:
    if not api_key:
        raise OpenRouterError("Missing OpenRouter API key.")
    if not model:
        raise OpenRouterError("Missing OpenRouter model name.")

    text = _call_openrouter_text(
        prompt=_build_constructed_text_prompt(problem_text),
        api_key=api_key,
        model=model,
        timeout_sec=timeout_sec,
        site_url=site_url,
        app_name=app_name,
        as_json_object=False,
    )
    # Strip accidental markdown fencing if model still adds it.
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if not text:
        raise OpenRouterError("OpenRouter constructed text output is empty.")
    return text
