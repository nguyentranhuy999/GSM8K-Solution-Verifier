from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class FormalizerConfig:
    input_path: Path = Path("Input/Problem.txt")
    output_path: Path = Path("Output/ProblemEntities.yaml")
    plan_path: Path = Path("Output/PlanEntities.yaml")
    model: str = "openai/gpt-4o-mini"
    timeout_seconds: int = 60
    max_tokens: int = 1200
    max_retries: int = 2


class ProblemFormalizer:
    def __init__(self, config: FormalizerConfig | None = None) -> None:
        load_dotenv()
        self.config = config or FormalizerConfig()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", self.config.model)

        if not self.api_key:
            raise EnvironmentError("Missing OPENROUTER_API_KEY in .env.")

    def read_problem(self) -> str:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Cannot find input file: {self.config.input_path}")

        problem = self.config.input_path.read_text(encoding="utf-8").strip()
        if not problem:
            raise ValueError(f"Input file is empty: {self.config.input_path}")

        return problem

    def build_prompt(self, problem: str, previous_error: str | None = None) -> str:
        retry_note = ""
        if previous_error:
            retry_note = f"""

Your previous output was rejected for this reason:
{previous_error}

Return a corrected YAML now. Remove all computed/intermediate entities.
""".rstrip()

        return f"""
You are a strict Problem Formalizer for grade-school math word problems.

Task:
Extract ONLY entities whose numeric values are explicitly stated in the problem,
plus exactly one target entity for the answer being asked.

Output format:
- Return ONLY valid YAML.
- Do not use markdown fences.
- Do not explain.
- The top-level YAML object maps snake_case entity names to exactly 3 fields:
  value, unit, location.

Allowed fields:
  value:
    - numeric value for explicit inputs.
    - empty string "" for the target.
    - convert number words to numbers, e.g. "six" -> 6.
    - convert fractions/percentages/scalars to numbers, e.g. 1/4 -> 0.25,
      10% -> 0.10, seven times -> 7.
  unit:
    - original unit category, e.g. dollars, days, tabs, coffees, inches.
    - use "" for pure scalars/fractions/multipliers.
  location:
    - "input" for explicit values from the problem.
    - "target" for the final asked quantity.

Strict rules:
- Do NOT solve the problem.
- Do NOT compute intermediate values.
- Do NOT include values that require arithmetic.
- Do NOT create variables like after_first, remaining_after_second,
  tabs_closed_first, total_cost_per_day, total_spent, etc.
- For phrases like "closed 1/4 of the remaining tabs", include only the explicit
  fraction 0.25 as an input scalar. Do not compute how many tabs were closed.
- For phrases like "closed half of the remaining tabs", include only the explicit
  fraction 0.5 as an input scalar.
- Fractions, percentages, ratios, and multipliers written in the problem are
  allowed as scalar input entities.
- The target entity must have value: "".
- There must be exactly one target entity.
- The target must represent the exact quantity asked in the final question.
- If the question asks "how many ... end up with", "how many ... remain", or
  "how many ... left", the target is the final remaining quantity, not a
  purchased/received/spent/intermediate quantity.

Example problem:
Nancy buys 2 coffees a day. She grabs a double espresso for $3.00 every morning.
In the afternoon, she grabs an iced coffee for $2.50. After 20 days, how much
money has she spent on coffee?

Example output:
coffees_per_day:
  value: 2
  unit: coffees
  location: input
morning_coffee_price:
  value: 3.00
  unit: dollars
  location: input
afternoon_coffee_price:
  value: 2.50
  unit: dollars
  location: input
days:
  value: 20
  unit: days
  location: input
total_cost:
  value: ""
  unit: dollars
  location: target

Problem:
{problem}
{retry_note}
""".strip()

    def call_openrouter(self, prompt: str) -> str:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Problem Formalizer",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract explicit math problem entities into "
                            "strict YAML. Never solve the problem."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": self.config.max_tokens,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected OpenRouter response: {data}") from exc

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"OpenRouter returned empty content: {data}")

        return content

    def clean_yaml_text(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:yaml|yml)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        return text

    def parse_entities(
        self,
        raw_output: str,
        problem: str,
    ) -> dict[str, dict[str, Any]]:
        yaml_text = self.clean_yaml_text(raw_output)
        parsed = yaml.safe_load(yaml_text)
        self.normalize_target(parsed, problem)
        self.validate_entities(parsed, problem)
        return parsed

    def normalize_target(self, entities: Any, problem: str) -> None:
        if not isinstance(entities, dict):
            return

        inferred_target = self.infer_target_from_question(problem)
        if not inferred_target:
            return

        target_name, target_unit = inferred_target
        existing_targets = [
            name
            for name, fields in entities.items()
            if isinstance(fields, dict) and fields.get("location") == "target"
        ]

        for name in existing_targets:
            if name != target_name and name in entities:
                del entities[name]

        entities[target_name] = {
            "value": "",
            "unit": target_unit,
            "location": "target",
        }

    def infer_target_from_question(self, problem: str) -> tuple[str, str] | None:
        question = self.extract_question(problem)
        normalized_question = question.lower()

        if "end up with" in normalized_question and "token" in normalized_question:
            return ("tokens_remaining", "tokens")

        if (
            any(phrase in normalized_question for phrase in ("remain", "remaining", "left"))
            and "token" in normalized_question
        ):
            return ("tokens_remaining", "tokens")

        return None

    def extract_question(self, problem: str) -> str:
        parts = [part.strip() for part in problem.split("?") if part.strip()]
        if not parts:
            return problem.strip()

        return parts[-1]

    def validate_entities(self, entities: Any, problem: str) -> None:
        if not isinstance(entities, dict):
            raise ValueError("Output YAML must be a dictionary.")

        target_count = 0
        required_fields = {"value", "unit", "location"}

        target_name = ""

        for name, fields in entities.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid entity name: {name!r}")

            if not isinstance(fields, dict):
                raise ValueError(f"Entity {name!r} must be a dictionary.")

            if set(fields.keys()) != required_fields:
                raise ValueError(
                    f"Entity {name!r} must have exactly fields: "
                    "value, unit, location."
                )

            location = fields["location"]
            value = fields["value"]
            unit = fields["unit"]

            if location not in {"input", "target"}:
                raise ValueError(
                    f"location of {name!r} must be either 'input' or 'target'."
                )

            if not isinstance(unit, str):
                raise ValueError(f"unit of {name!r} must be a string.")

            if location == "target":
                target_count += 1
                target_name = name
                if value not in ("", None):
                    raise ValueError(f"Target entity {name!r} must have empty value.")
            else:
                if self.looks_computed_name(name):
                    raise ValueError(
                        f"Input entity {name!r} looks computed. ProblemFormalizer "
                        "must only extract explicit inputs and target."
                    )

                if not isinstance(value, (int, float)):
                    raise ValueError(f"value of input entity {name!r} must be numeric.")

        if target_count != 1:
            raise ValueError(f"Expected exactly one target entity, got {target_count}.")

        self.validate_target_matches_question(target_name, problem)

    def validate_target_matches_question(self, target_name: str, problem: str) -> None:
        normalized_problem = problem.lower()
        final_quantity_phrases = (
            "end up with",
            "ended up with",
            "remain",
            "remaining",
            "left",
            "have left",
        )
        misleading_target_markers = (
            "received",
            "bought",
            "purchased",
            "spent",
            "wasted",
            "used",
            "lost",
        )
        final_target_markers = (
            "final",
            "remaining",
            "left",
            "end",
            "ending",
            "result",
        )

        asks_for_final_quantity = any(
            phrase in normalized_problem for phrase in final_quantity_phrases
        )
        if not asks_for_final_quantity:
            return

        if any(marker in target_name for marker in misleading_target_markers):
            raise ValueError(
                f"Target {target_name!r} is an intermediate action quantity, but the "
                "question asks for the final/remaining quantity."
            )

        if not any(marker in target_name for marker in final_target_markers):
            raise ValueError(
                f"Target {target_name!r} does not look like a final/remaining "
                "quantity requested by the question."
            )

    def looks_computed_name(self, name: str) -> bool:
        computed_markers = (
            "after_",
            "remaining_after",
            "total_",
            "_total",
            "spent_total",
            "closed_first",
            "closed_second",
            "closed_third",
            "after_first",
            "after_second",
            "after_third",
        )
        return any(marker in name for marker in computed_markers)

    def save_yaml(self, entities: dict[str, dict[str, Any]]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                entities,
                file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    def copy_to_plan_entities(self) -> None:
        self.config.plan_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.config.output_path, self.config.plan_path)

    def run(self) -> dict[str, dict[str, Any]]:
        problem = self.read_problem()
        previous_error: str | None = None
        last_raw_output = ""

        for attempt in range(1, self.config.max_retries + 1):
            prompt = self.build_prompt(problem, previous_error)
            raw_output = self.call_openrouter(prompt)
            last_raw_output = raw_output

            try:
                entities = self.parse_entities(raw_output, problem)
                break
            except ValueError as exc:
                previous_error = str(exc)
                if attempt == self.config.max_retries:
                    raise ValueError(
                        f"Could not produce valid problem entities after "
                        f"{attempt} attempts. Last raw output:\n{last_raw_output}"
                    ) from exc

        self.save_yaml(entities)
        self.copy_to_plan_entities()
        return entities


def main() -> None:
    formalizer = ProblemFormalizer()
    formalizer.run()
    print(f"Saved problem entities to: {formalizer.config.output_path}")
    print(f"Copied problem entities to: {formalizer.config.plan_path}")


if __name__ == "__main__":
    main()
