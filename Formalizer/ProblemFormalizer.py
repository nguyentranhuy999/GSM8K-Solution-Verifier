"""
ProblemFormalizer.py

Nhiệm vụ:
- Đọc đề bài từ Input/problem.txt
- Gửi đề bài qua LLM bằng OpenRouter API
- Yêu cầu LLM trích xuất các thực thể trong đề bài thành cấu trúc có 4 trường:
    value: giá trị được cho trong đề bài, hoặc tên biến target nếu là mục tiêu
    type: "units" hoặc "scaler"
    unit: đơn vị của thực thể, ví dụ: dollars, days, apples, people, ...
    location: "input", "computed", hoặc "target"
- Lưu kết quả ra Output/formalized_problem.yaml

Lưu ý về type:
- "units": thực thể có cùng loại đơn vị với target hoặc là đại lượng chính cần tính.
- "scaler": thực thể dùng để nhân/chia/tính toán theo đề bài, thường là số lần, số ngày, số người, tỉ lệ, phần trăm, v.v.

Cách dùng:
1. Cài package:
   pip install requests pyyaml python-dotenv

2. Tạo file .env:
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_MODEL=openai/gpt-4o-mini

3. Chạy:
   python ProblemFormalizer.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv


@dataclass
class ProblemFormalizerConfig:
    input_path: Path = Path("Input/problem.txt")
    output_path: Path = Path("Output/formalized_problem.yaml")
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: int = 30


class ProblemFormalizer:
    def __init__(self, config: Optional[ProblemFormalizerConfig] = None) -> None:
        load_dotenv()
        self.config = config or ProblemFormalizerConfig()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", self.config.model)

        if not self.api_key:
            raise EnvironmentError(
                "Missing OPENROUTER_API_KEY. Please set it in your .env file or environment variables."
            )

    def read_problem(self) -> str:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Cannot find input file: {self.config.input_path}")

        problem_text = self.config.input_path.read_text(encoding="utf-8").strip()
        if not problem_text:
            raise ValueError(f"Input file is empty: {self.config.input_path}")

        return problem_text

    def build_prompt(self, problem_text: str) -> str:
        return f"""
You are a math word-problem formalizer.

Your task is to read the problem and extract entities into a clean YAML-compatible JSON object.

Each entity must have exactly these fields:
- value: the numeric value explicitly given in the problem. For the target, use the variable name being asked for.
- type: either "units" or "scaler".
- unit: the unit of the entity, such as dollars, days, weeks, apples, people, puppies, percent, ratio, etc.
- location: one of "input", "computed", or "target".

Type rules:
- Use "units" for entities that have the same unit category as the target answer or represent the main quantity being accumulated, shared, paid, counted, bought, etc.
- Use "scaler" for entities used to scale, multiply, divide, repeat, split, or compare quantities, such as days, weeks, months, people, roommates, times, fractions, percentages, or rates.
- Even if a scaler has a real-world unit like days or people, it should still be type "scaler" when its role is to scale another value.

Location rules:
- "input": values explicitly given in the problem.
- "computed": intermediate variables that are useful for solving the problem, but do not appear directly as the final answer.
- "target": the final quantity being asked for.

Naming rules:
- Use clear snake_case variable names.
- Do not use spaces in variable names.
- Avoid vague names like number_1 or value_2.
- Include a target entity.
- Include useful computed entities only if they are clearly needed.

Output rules:
- Return ONLY valid JSON.
- Do not include markdown fences.
- Do not include explanation.
- The top-level object should map variable names to their entity fields.

Example output format:
{{
  "morning_coffee_price": {{
    "value": 3.00,
    "type": "units",
    "unit": "dollars",
    "location": "input"
  }},
  "days": {{
    "value": 20,
    "type": "scaler",
    "unit": "days",
    "location": "input"
  }},
  "total_cost": {{
    "value": "total_cost",
    "type": "units",
    "unit": "dollars",
    "location": "target"
  }}
}}

Problem:
{problem_text}
""".strip()

    def call_openrouter(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Problem Formalizer",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You extract structured entities from math word problems. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "OpenRouter request failed. Check internet access, API key, and model name."
            ) from exc

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected OpenRouter response format: {data}") from exc

        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            )

        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason")
            message_keys = sorted(message.keys())
            raise ValueError(
                "OpenRouter returned an empty assistant message. "
                f"model={self.model!r}, finish_reason={finish_reason!r}, "
                f"message_keys={message_keys}. "
                "Try a chat model that returns text content, for example openai/gpt-4o-mini."
            )

        return content

    def extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON even if the model accidentally wraps it in markdown fences."""
        text = raw_text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"Model did not return valid JSON:\n{text}")
            parsed = json.loads(match.group(0))

        self.validate_entities(parsed)
        return parsed

    def validate_entities(self, entities: Dict[str, Any]) -> None:
        if not isinstance(entities, dict):
            raise ValueError("Formalized output must be a JSON object.")

        required_fields = {"value", "type", "unit", "location"}
        valid_types = {"units", "scaler"}
        valid_locations = {"input", "computed", "target"}

        target_count = 0

        for name, entity in entities.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid entity name: {name}")

            if not isinstance(entity, dict):
                raise ValueError(f"Entity '{name}' must be an object.")

            fields = set(entity.keys())
            if fields != required_fields:
                raise ValueError(
                    f"Entity '{name}' must have exactly fields {required_fields}, got {fields}"
                )

            if entity["type"] not in valid_types:
                raise ValueError(
                    f"Entity '{name}' has invalid type '{entity['type']}'. Expected one of {valid_types}."
                )

            if entity["location"] not in valid_locations:
                raise ValueError(
                    f"Entity '{name}' has invalid location '{entity['location']}'. Expected one of {valid_locations}."
                )

            if entity["location"] == "target":
                target_count += 1

        if target_count != 1:
            raise ValueError(f"Expected exactly one target entity, got {target_count}.")

    def save_yaml(self, entities: Dict[str, Any]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                entities,
                file,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    def formalize(self) -> Dict[str, Any]:
        print(f"Reading problem from {self.config.input_path}...", flush=True)
        problem_text = self.read_problem()
        prompt = self.build_prompt(problem_text)
        print(f"Calling OpenRouter model {self.model}...", flush=True)
        raw_response = self.call_openrouter(prompt)
        print("Parsing model response...", flush=True)
        entities = self.extract_json(raw_response)
        print(f"Saving YAML to {self.config.output_path}...", flush=True)
        self.save_yaml(entities)
        return entities


def main() -> None:
    formalizer = ProblemFormalizer()
    entities = formalizer.formalize()

    print("Formalized problem saved successfully.")
    print(f"Output path: {formalizer.config.output_path}")
    print(yaml.safe_dump(entities, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
