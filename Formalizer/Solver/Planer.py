import os
import re
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import requests

from dotenv import load_dotenv


load_dotenv()


ROOT = Path(__file__).resolve().parents[2]

PROBLEM_PATH = ROOT / "Input" / "Problem.txt"
ENTITIES_PATH = ROOT / "Output" / "PlanEntities.yaml"
PLAN_PATH = ROOT / "Output" / "Plan.yaml"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


class Planner:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("Missing OPENROUTER_API_KEY")

    def read_problem(self) -> str:
        return PROBLEM_PATH.read_text(encoding="utf-8").strip()

    def read_entities(self) -> Dict[str, Any]:
        if not ENTITIES_PATH.exists():
            raise FileNotFoundError(f"Cannot find entities file: {ENTITIES_PATH}")

        with ENTITIES_PATH.open(encoding="utf-8") as file:
            entities = yaml.safe_load(file)

        if not isinstance(entities, dict):
            raise ValueError("PlanEntities.yaml must contain a dictionary")

        return entities

    def build_prompt(self, problem: str, entities: Dict[str, Any]) -> str:
        entities_yaml = yaml.safe_dump(
            entities,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()

        return f"""
You are a math word problem planner.

Given a word problem and extracted entities, create a calculation plan.

Problem:
{problem}

Extracted entities:
{entities_yaml}

Return ONLY valid JSON.

Format:
{{
  "step1": {{
    "expr": "entity_a + entity_b",
    "result": "new_entity",
    "result_unit": "unit"
  }},
  "step2": {{
    "expr": "new_entity * entity_c",
    "result": "target_entity",
    "result_unit": "unit"
  }}
}}

Rules:
- Step names must be step1, step2, step3, ...
- expr must use entity names from Extracted entities or previous step results.
- result is the variable created by that step.
- result_unit is the unit of result.
- Do not calculate numeric values.
- The final step result must answer the question.
- The final step result must be the target entity from Extracted entities.
- The final step result_unit must equal the target entity unit.
- Use meaningful variable names based on the problem.
- If a step computes one specific object type, use that object's unit.
- Do not blindly normalize intermediate units to the target unit. For example,
  if a step computes total pens, result_unit should be "pens"; only the final
  aggregate answer should use "items" when the target unit is "items".
- For unit conversion, write expressions like:
  minutes / 60
  hours * 60
  dollars * 100

Example:

Problem:
Nancy buys 2 coffees a day. She grabs a double espresso for $3.00 every morning.
In the afternoon, she grabs an iced coffee for $2.50.
After 20 days, how much money has she spent on coffee?

Output:
{{
  "step1": {{
    "expr": "morning_coffee_price + afternoon_coffee_price",
    "result": "daily_cost",
    "result_unit": "dollars"
  }},
  "step2": {{
    "expr": "daily_cost * days",
    "result": "total_cost",
    "result_unit": "dollars"
  }}
}}
""".strip()

    def call_llm(self, prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You output strict JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0
            },
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter error {response.status_code}: {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]

    def parse_json_output(self, raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()

        if text.startswith("```json"):
            text = text[len("```json"):].strip()

        if text.startswith("```"):
            text = text[len("```"):].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON:\n{text}") from e

        if not isinstance(data, dict):
            raise ValueError("Plan must be a JSON object")

        return data

    def validate_plan(self, plan: Dict[str, Any]) -> None:
        index = 1

        for step_name, step in plan.items():
            expected_name = f"step{index}"

            if step_name != expected_name:
                raise ValueError(
                    f"Invalid step name. Expected {expected_name}, got {step_name}"
                )

            if not isinstance(step, dict):
                raise ValueError(f"{step_name} must be a dictionary")

            for field in ["expr", "result", "result_unit"]:
                if field not in step:
                    raise ValueError(f"{step_name} is missing field: {field}")

                if not isinstance(step[field], str):
                    raise ValueError(f"{step_name}.{field} must be a string")

            index += 1

    def target_entity(self, entities: Dict[str, Any]) -> tuple[str, str]:
        targets = [
            (name, fields)
            for name, fields in entities.items()
            if isinstance(fields, dict) and fields.get("location") == "target"
        ]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one target entity, got {len(targets)}")

        target_name, target_fields = targets[0]
        target_unit = target_fields.get("unit", "")
        if not isinstance(target_unit, str):
            raise ValueError(f"Target unit for {target_name} must be a string")

        return target_name, target_unit

    def entity_unit(self, name: str, entities: Dict[str, Any], plan: Dict[str, Any]) -> str:
        if name in entities and isinstance(entities[name], dict):
            unit = entities[name].get("unit", "")
            return unit if isinstance(unit, str) else ""

        for step in plan.values():
            if isinstance(step, dict) and step.get("result") == name:
                unit = step.get("result_unit", "")
                return unit if isinstance(unit, str) else ""

        return ""

    def normalize_plan_units(self, plan: Dict[str, Any], entities: Dict[str, Any]) -> None:
        target_name, target_unit = self.target_entity(entities)

        for step in plan.values():
            if step["result"] == target_name:
                step["result_unit"] = target_unit

    def validate_plan_against_entities(
        self,
        plan: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> None:
        target_name, target_unit = self.target_entity(entities)
        if not plan:
            raise ValueError("Plan must have at least one step")

        last_step = next(reversed(plan.values()))
        if last_step["result"] != target_name:
            raise ValueError(
                f"Final step result must be target {target_name}, "
                f"got {last_step['result']}"
            )

        if last_step["result_unit"] != target_unit:
            raise ValueError(
                f"Final step unit must be {target_unit}, "
                f"got {last_step['result_unit']}"
            )

    def write_yaml(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False
            )

    def run(self) -> None:
        problem = self.read_problem()
        entities = self.read_entities()

        prompt = self.build_prompt(problem, entities)
        raw_output = self.call_llm(prompt)

        plan = self.parse_json_output(raw_output)
        self.normalize_plan_units(plan, entities)
        self.validate_plan(plan)
        self.validate_plan_against_entities(plan, entities)

        self.write_yaml(PLAN_PATH, plan)

        print(f"Saved plan to {PLAN_PATH}")


if __name__ == "__main__":
    planner = Planner()
    planner.run()
