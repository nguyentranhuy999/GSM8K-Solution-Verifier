"""
Formalizer/ProblemFomalizer.py

Nhiệm vụ:
- Đọc đề bài từ Input/Problem.txt
- Gọi LLM qua OpenRouter để trích xuất các thực thể có giá trị số được nêu trực tiếp trong đề bài
- Không tính toán các thực thể trung gian chưa có số trực tiếp trong đề
- Tạo thêm đúng 1 thực thể target cần tìm, value rỗng
- Ghi kết quả vào Output/ProblemEntities.yaml
- Nếu thành công/thất bại, ghi trạng thái vào Output/Log.yaml
- Nếu thành công, copy Output/ProblemEntities.yaml sang:
  - Output/PlanEntities.yaml
  - Output/StudentAnswerEntities.yaml

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "Input" / "Problem.txt"
OUTPUT_DIR = ROOT_DIR / "Output"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_RETRIES = 3
NUMBER_TOLERANCE = 1e-9

NUMBER_WORDS = {
    "zero": 0,
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
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

SCALE_WORDS = {
    "hundred": 100,
    "thousand": 1000,
}

FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "fourth": 4,
    "fourths": 4,
    "quarter": 4,
    "quarters": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
}

MULTIPLIER_WORDS = {
    "twice": 2,
    "double": 2,
    "doubled": 2,
    "thrice": 3,
    "triple": 3,
    "tripled": 3,
}

UNIT_BASE_ALIASES = {
    "inch": "inch",
    "inches": "inch",
    "in": "inch",
    "foot": "foot",
    "feet": "foot",
    "ft": "foot",
    "yard": "yard",
    "yards": "yard",
    "mile": "mile",
    "miles": "mile",
    "millimeter": "millimeter",
    "millimeters": "millimeter",
    "mm": "millimeter",
    "centimeter": "centimeter",
    "centimeters": "centimeter",
    "cm": "centimeter",
    "meter": "meter",
    "meters": "meter",
    "m": "meter",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "km": "kilometer",
    "second": "second",
    "seconds": "second",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "cent": "cent",
    "cents": "cent",
    "dollar": "dollar",
    "dollars": "dollar",
    "usd": "dollar",
    "ounce": "ounce",
    "ounces": "ounce",
    "oz": "ounce",
    "pound": "pound",
    "pounds": "pound",
    "lb": "pound",
    "lbs": "pound",
    "ton": "ton",
    "tons": "ton",
    "item": "item",
    "items": "item",
    "piece": "item",
    "pieces": "item",
    "dozen": "dozen",
    "dozens": "dozen",
}

STANDARD_CONVERSION_FACTORS = [
    ("inch", "foot", "unit_conversion_inches_per_foot", 12),
    ("foot", "yard", "unit_conversion_feet_per_yard", 3),
    ("foot", "mile", "unit_conversion_feet_per_mile", 5280),
    ("yard", "mile", "unit_conversion_yards_per_mile", 1760),
    ("millimeter", "centimeter", "unit_conversion_millimeters_per_centimeter", 10),
    ("centimeter", "meter", "unit_conversion_centimeters_per_meter", 100),
    ("meter", "kilometer", "unit_conversion_meters_per_kilometer", 1000),
    ("second", "minute", "unit_conversion_seconds_per_minute", 60),
    ("minute", "hour", "unit_conversion_minutes_per_hour", 60),
    ("hour", "day", "unit_conversion_hours_per_day", 24),
    ("day", "week", "unit_conversion_days_per_week", 7),
    ("month", "year", "unit_conversion_months_per_year", 12),
    ("cent", "dollar", "unit_conversion_cents_per_dollar", 100),
    ("ounce", "pound", "unit_conversion_ounces_per_pound", 16),
    ("pound", "ton", "unit_conversion_pounds_per_ton", 2000),
    ("item", "dozen", "unit_conversion_items_per_dozen", 12),
]


class ProblemFormalizerError(Exception):
    """Lỗi riêng cho ProblemFormalizer."""


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    payload = {
        "ProblemFormalizer": status,
    }
    if message:
        payload["message"] = message

    with LOG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def read_problem() -> str:
    if not INPUT_PATH.exists():
        raise ProblemFormalizerError(f"Không tìm thấy file: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ProblemFormalizerError("Input/Problem.txt đang rỗng.")

    return text


def build_system_prompt() -> str:
    return """
Bạn là một bộ formalize đề toán thành YAML.

Nhiệm vụ của bạn:
1. Chỉ trích xuất các thực thể có giá trị số được nói trực tiếp trong đề bài.
2. Nếu số được viết bằng chữ, hãy chuyển thành số. Ví dụ: "two" -> 2, "twenty" -> 20, "half" -> 0.5.
3. Không tính toán bất kỳ đại lượng nào không được cho trực tiếp trong đề bài.
4. Không tạo thực thể trung gian phải suy ra bằng phép tính.
5. Luôn tạo thêm đúng một thực thể là đáp số cần tìm, có location là target và value rỗng.
6. Các thực thể trong đề bài trực tiếp có location là input.
7. Trong file này, location chỉ được là input hoặc target.
8. Phân số, phần trăm, tỉ lệ, hệ số nhân được nói trực tiếp trong đề cũng là input scalar.
   Ví dụ: "a third" -> value 0.3333333333333333, unit rỗng; "a fourth" -> 0.25; "seven times" -> 7.
9. Với cụm như "20% heavier", "20% more", "20% increase", chỉ trích xuất phần trăm trực tiếp là 0.2.
   Không tự tạo multiplier 1.2 vì đó là số phải suy ra từ 1 + 0.2.
10. Với cụm như "20% lighter", "20% less", chỉ trích xuất phần trăm trực tiếp là 0.2.
    Không tự tạo multiplier 0.8 vì đó là số phải suy ra từ 1 - 0.2.
11. Với rate như "2 books a month", "5 pages per day", giữ chu kỳ trong tên entity.
    Ví dụ: books_per_month, pages_per_day. Không đổi value thành tổng theo năm/ngày khác.
12. Với quan hệ so sánh như "X has 2 fewer than Y", "X has 5 more than Y":
    số trực tiếp là độ chênh lệch, không phải số lượng thật của X.
    Tên entity phải thể hiện đây là độ chênh, ví dụ books_borrowed_fewer_than_books_bought.
13. Không được bỏ sót số trực tiếp đi kèm đơn vị trong đề bài.
    Ví dụ "in 7 days" phải có một input entity value 7, unit days, không được bỏ vì đã có "5 days" trước đó.
14. Target phải là đúng đại lượng được hỏi, không phải đại lượng trung gian.
    Ví dụ câu hỏi "How many old books ... reread" thì target phải chứa reread, như old_books_to_reread.
    Không đặt target là total_books_needed nếu đề hỏi số sách cũ phải đọc lại.
    Ví dụ câu hỏi "How many friends can she invite?" thì target phải là số friends, như total_friends.
    Không đặt target là total_cost nếu đề hỏi số bạn được mời.

Mỗi thực thể phải có đúng 4 trường:
- value: giá trị số được cho trực tiếp trong đề bài. Với target, để rỗng bằng null.
- unit: đơn vị trực tiếp của thực thể, viết bằng tiếng Anh dạng số nhiều nếu phù hợp. Ví dụ: dollars, days, books, pens. Với scalar như phần trăm, phân số, nhân tử, có thể để rỗng.
- location: input hoặc target.
- grand_unit: đơn vị đối chiếu theo target.

Quy tắc grand_unit:
- Nếu thực thể có cùng loại đơn vị với target hoặc là thành phần có thể cộng/tổng hợp vào target, grand_unit là unit của target.
  Ví dụ target là tổng items, sách có unit books và bút có unit pens thì grand_unit của sách/bút là items.
- Nếu thực thể không liên quan trực tiếp đến đơn vị target, giữ grand_unit là unit của chính nó hoặc để rỗng nếu là scalar.
  Ví dụ target là tổng pens, books không liên quan thì grand_unit của books là books.
- Với scalar như phần trăm, phân số, hệ số nhân, grand_unit có thể để rỗng.
- Với target, grand_unit thường bằng unit của target.

Quy tắc loại bỏ:
- Không đưa vào các số chỉ là nhãn, tên riêng, số thứ tự không tham gia bài toán.
- Không đưa vào đại lượng được suy ra. Ví dụ đề nói có 2 cà phê mỗi ngày gồm cà phê sáng $3 và cà phê chiều $2.50, không tự tính chi phí/ngày.
- Không đưa vào kết quả của phép tính dù phép tính rất đơn giản. Ví dụ đề nói "7 tokens" và "seven times as many" thì giữ scalar 7, không tạo entity giá trị 49.
- Không biến phần trăm tăng/giảm thành hệ số cuối. Ví dụ "20% heavier" giữ 0.2, không tạo 1.2.
- Nếu một số trực tiếp trong đề bị thừa và không liên quan đến lời hỏi, có thể vẫn giữ nếu nó là dữ kiện số trong đề; nhưng không được tự tính thêm.

Tên biến:
- Dùng snake_case.
- Tên phải rõ nghĩa theo vai trò trong đề bài.
- Target nên đặt theo đại lượng cần tìm, ví dụ total_cost, remaining_tabs, total_items.

Định dạng output bắt buộc:
- Chỉ trả về YAML thuần.
- Không dùng Markdown.
- Không bọc trong ```.
- Không giải thích.
- Không thêm trường ngoài 4 trường đã yêu cầu.

Ví dụ:
morning_coffee_price:
  value: 3.00
  unit: dollars
  location: input
  grand_unit: dollars

afternoon_coffee_price:
  value: 2.50
  unit: dollars
  location: input
  grand_unit: dollars

days:
  value: 20
  unit: days
  location: input
  grand_unit:

fraction_wasted_pac_man:
  value: 0.3333333333333333
  unit:
  location: input
  grand_unit:

parent_token_multiplier:
  value: 7
  unit:
  location: input
  grand_unit:

total_cost:
  value:
  unit: dollars
  location: target
  grand_unit: dollars
""".strip()


def build_user_prompt(problem: str, previous_error: Optional[str] = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sinh lại YAML, chỉ dùng số được nêu trực tiếp trong đề. Không tạo số suy ra.
""".rstrip()

    return f"""
Hãy formalize đề bài sau thành YAML theo đúng quy tắc.

Đề bài:
{problem}
{retry_note}
""".strip()


def call_openrouter(problem: str, previous_error: Optional[str] = None) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ProblemFormalizerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "GSM8K-Solution-Verifier"),
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(problem, previous_error=previous_error)},
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise ProblemFormalizerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise ProblemFormalizerError(
            f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}"
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ProblemFormalizerError(f"Response OpenRouter không đúng định dạng: {response.text[:1000]}") from exc


def strip_markdown_fence(text: str) -> str:
    text = text.strip()

    fenced = re.match(r"^```(?:yaml|yml)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return text


def parse_yaml(text: str) -> Dict[str, Any]:
    clean_text = strip_markdown_fence(text)

    try:
        parsed = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise ProblemFormalizerError(f"LLM trả về YAML không hợp lệ: {exc}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise ProblemFormalizerError("YAML output phải là dictionary không rỗng.")

    return parsed


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def normalize_unit_text(unit: Any) -> Optional[str]:
    unit = normalize_empty(unit)
    if unit is None:
        return None
    return str(unit).strip().lower().replace("-", "_").replace(" ", "_")


def unit_base_keys(unit: Any) -> set[str]:
    text = normalize_unit_text(unit)
    if not text:
        return set()

    candidates = {text}
    for prefix in ("cubic_", "cube_", "square_", "sq_"):
        if text.startswith(prefix):
            candidates.add(text[len(prefix):])

    for suffix in ("_cubed", "_squared"):
        if text.endswith(suffix):
            candidates.add(text[: -len(suffix)])

    if text.endswith("3"):
        candidates.add(text[:-1])
    if text.endswith("2"):
        candidates.add(text[:-1])

    return {
        UNIT_BASE_ALIASES[candidate]
        for candidate in candidates
        if candidate in UNIT_BASE_ALIASES
    }


def collect_unit_bases(entities: Dict[str, Dict[str, Any]]) -> set[str]:
    bases: set[str] = set()
    for entity in entities.values():
        bases.update(unit_base_keys(entity.get("unit")))
        bases.update(unit_base_keys(entity.get("grand_unit")))
    return bases


AMBIGUOUS_TEXT_UNIT_ALIASES = {"in", "m", "g", "l"}


def collect_unit_bases_from_problem(problem: str) -> set[str]:
    text = problem.lower()
    bases: set[str] = set()

    for alias, base in UNIT_BASE_ALIASES.items():
        if alias in AMBIGUOUS_TEXT_UNIT_ALIASES:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", text):
            bases.add(base)

    return bases


def add_standard_conversion_entities(
    entities: Dict[str, Dict[str, Any]],
    problem: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Bổ sung hằng chuyển đổi đơn vị chuẩn dưới dạng input entity.

    Các hằng này không phải kết quả trung gian của đề bài; chúng là kiến thức
    đơn vị cần có để Planner giữ expr chỉ gồm biến.
    """
    unit_bases = collect_unit_bases(entities)
    if problem:
        unit_bases.update(collect_unit_bases_from_problem(problem))

    enriched = dict(entities)

    for smaller_unit, larger_unit, entity_name, value in STANDARD_CONVERSION_FACTORS:
        if smaller_unit not in unit_bases or larger_unit not in unit_bases:
            continue
        if entity_name in enriched:
            continue

        enriched[entity_name] = {
            "value": value,
            "unit": None,
            "location": "input",
            "grand_unit": None,
        }

    return enriched


def problem_asks_invited_friends(problem: str) -> bool:
    text = problem.lower()
    if "invite" not in text:
        return False
    return bool(re.search(r"\bhow many friends\b", text))


def looks_like_invited_friends_problem(problem: str, entities: Dict[str, Dict[str, Any]]) -> bool:
    if not problem_asks_invited_friends(problem):
        return False

    target_names = [
        name
        for name, entity in entities.items()
        if entity.get("location") == "target"
    ]
    target_text = " ".join(
        f"{name} {normalize_empty(entities[name].get('unit')) or ''}"
        for name in target_names
    ).lower()

    if "friend" not in target_text:
        return False

    text = problem.lower()

    return bool(
        re.search(
            r"\b(?:she|he|they|we|i|[a-z]+)\s+and\s+"
            r"(?:her|his|their|our|my)\s+friends\b",
            text,
        )
    )


def add_default_context_entities(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    enriched = dict(entities)

    if problem_asks_invited_friends(problem) and "host_count" not in enriched:
        enriched["host_count"] = {
            "value": 1,
            "unit": "people",
            "location": "input",
            "grand_unit": "people",
        }

    return enriched


NON_COUNT_COMPONENT_BASES = {
    "inch",
    "foot",
    "yard",
    "mile",
    "millimeter",
    "centimeter",
    "meter",
    "kilometer",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "cent",
    "dollar",
    "ounce",
    "pound",
    "ton",
}


def normalize_item_target_grand_units(
    entities: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    target_units = [
        normalize_unit_text(entity.get("unit"))
        for entity in entities.values()
        if entity.get("location") == "target"
    ]
    if not any(unit in {"item", "items"} for unit in target_units):
        return entities

    target_unit = next((unit for unit in target_units if unit in {"item", "items"}), "items")
    normalized = dict(entities)

    for name, entity in normalized.items():
        if entity.get("location") != "input":
            continue
        if normalize_empty(entity.get("unit")) is None:
            continue

        unit_bases = unit_base_keys(entity.get("unit"))
        if unit_bases & NON_COUNT_COMPONENT_BASES:
            continue

        entity = dict(entity)
        entity["grand_unit"] = target_unit
        normalized[name] = entity

    return normalized


def max_retries() -> int:
    raw = os.getenv("PROBLEM_FORMALIZER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProblemFormalizerError("PROBLEM_FORMALIZER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise ProblemFormalizerError("PROBLEM_FORMALIZER_MAX_RETRIES phải >= 1.")
    return value


def coerce_number(value: Any, entity_name: str) -> Optional[float | int]:
    value = normalize_empty(value)

    if value is None:
        return None

    if isinstance(value, bool):
        raise ProblemFormalizerError(f"{entity_name}.value không được là boolean.")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            number = float(cleaned)
        except ValueError as exc:
            raise ProblemFormalizerError(f"{entity_name}.value phải là số, hiện là: {value!r}") from exc

        if number.is_integer():
            return int(number)
        return number

    raise ProblemFormalizerError(f"{entity_name}.value phải là số hoặc rỗng cho target.")


def parse_number_word_phrase(text: str) -> Optional[int]:
    tokens = re.split(r"[\s-]+", text.strip().lower())
    total = 0
    current = 0
    found = False

    for token in tokens:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            found = True
            continue

        if token in SCALE_WORDS:
            scale = SCALE_WORDS[token]
            current = max(current, 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            found = True
            continue

        return None

    if not found:
        return None

    return total + current


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def span_is_covered(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(spans_overlap(span, existing) for existing in spans)


def extract_direct_numeric_values(problem: str) -> list[float]:
    text = problem.lower()
    values: list[float] = []
    covered_spans: list[tuple[int, int]] = []

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", text):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator == 0:
            continue
        values.append(numerator / denominator)
        covered_spans.append(match.span())

    fraction_pattern = re.compile(
        r"\b(?:(a|an)|((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        r"(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine))?))\s+"
        r"(half|halves|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|"
        r"eighths?|ninths?|tenths?)\b"
    )
    for match in fraction_pattern.finditer(text):
        if span_is_covered(match.span(), covered_spans):
            continue

        numerator = 1 if match.group(1) else parse_number_word_phrase(match.group(2) or "")
        denominator = FRACTION_DENOMINATORS[match.group(3)]
        if numerator is None:
            continue

        values.append(numerator / denominator)
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(half|quarter)\b", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        denominator = FRACTION_DENOMINATORS[match.group(1)]
        values.append(1 / denominator)
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(twice|double|doubled|thrice|triple|tripled)\b", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        values.append(float(MULTIPLIER_WORDS[match.group(1)]))
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*%", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        values.append(float(match.group(1).replace(",", "")) / 100)
        covered_spans.append(match.span())

    for match in re.finditer(r"(?<![\w/])\d+(?:,\d{3})*(?:\.\d+)?(?!\s*/|\w|%)", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        raw_value = match.group(0).replace(",", "")
        values.append(float(raw_value))
        covered_spans.append(match.span())

    number_word_pattern = re.compile(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand))*\b"
    )
    for match in number_word_pattern.finditer(text):
        if span_is_covered(match.span(), covered_spans):
            continue
        value = parse_number_word_phrase(match.group(0))
        if value is None:
            continue
        values.append(float(value))
        covered_spans.append(match.span())

    return values


def extract_required_unit_numeric_values(problem: str) -> list[float]:
    """
    Lấy các số dạng digit đi kèm một unit word ngay sau đó.

    Đây là lớp bắt lỗi "miss số có đơn vị" như "in 7 days" mà không ép LLM
    phải giữ mọi số có thể là nhãn/noise.
    """
    text = problem.lower()
    values: list[float] = []

    pattern = re.compile(
        r"(?<![\w/])"
        r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?!\s*/|\w|%)"
        r"\s+([a-z][a-z-]*)\b"
    )
    ignored_following_words = {
        "am",
        "pm",
        "a",
        "an",
        "the",
    }

    for match in pattern.finditer(text):
        following_word = match.group(2)
        if following_word in ignored_following_words:
            continue
        values.append(float(match.group(1).replace(",", "")))

    return values


def numbers_equal(left: float | int, right: float | int) -> bool:
    return abs(float(left) - float(right)) <= NUMBER_TOLERANCE


def format_number_for_error(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def validate_values_are_direct(problem: str, entities: Dict[str, Dict[str, Any]]) -> None:
    expected_values = extract_direct_numeric_values(problem)
    unused_expected = expected_values.copy()
    input_values = [
        entity["value"]
        for entity in entities.values()
        if entity["location"] == "input"
    ]

    for entity_name, entity in entities.items():
        if entity["location"] != "input":
            continue

        value = entity["value"]
        match_index = next(
            (
                index
                for index, expected_value in enumerate(unused_expected)
                if numbers_equal(value, expected_value)
            ),
            None,
        )
        if match_index is None:
            raise ProblemFormalizerError(
                f"{entity_name}.value={format_number_for_error(value)} không phải số được nêu trực tiếp trong đề."
            )
        del unused_expected[match_index]

    require_all_direct_values = os.getenv("PROBLEM_FORMALIZER_REQUIRE_ALL_DIRECT_VALUES") == "1"
    if require_all_direct_values and unused_expected:
        missing = ", ".join(format_number_for_error(value) for value in unused_expected)
        raise ProblemFormalizerError(
            f"Thiếu thực thể input cho các số/scalar trực tiếp trong đề: {missing}."
        )

    unused_input_values = input_values.copy()
    missing_required_values: list[float] = []
    for required_value in extract_required_unit_numeric_values(problem):
        match_index = next(
            (
                index
                for index, input_value in enumerate(unused_input_values)
                if numbers_equal(input_value, required_value)
            ),
            None,
        )
        if match_index is None:
            missing_required_values.append(required_value)
            continue
        del unused_input_values[match_index]

    if missing_required_values:
        missing = ", ".join(format_number_for_error(value) for value in missing_required_values)
        raise ProblemFormalizerError(
            f"Thiếu thực thể input cho số trực tiếp có đơn vị trong đề: {missing}."
        )


def target_entity_name(entities: Dict[str, Dict[str, Any]]) -> str:
    targets = [name for name, entity in entities.items() if entity.get("location") == "target"]
    if len(targets) != 1:
        raise ProblemFormalizerError(f"Phải có đúng 1 entity target, hiện có {len(targets)}.")
    return targets[0]


def validate_target_name_matches_question(problem: str, entities: Dict[str, Dict[str, Any]]) -> None:
    text = problem.lower()
    target_name = target_entity_name(entities)

    required_terms: list[str] = []
    if re.search(r"\breread\b", text):
        required_terms.append("reread")
    if problem_asks_invited_friends(problem):
        required_terms.append("friend")

    missing_terms = [term for term in required_terms if term not in target_name]
    if missing_terms:
        raise ProblemFormalizerError(
            f"Target {target_name!r} chưa khớp đại lượng được hỏi; "
            f"tên target cần chứa {missing_terms}."
        )


def validate_and_normalize(entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    required_fields = {"value", "unit", "location", "grand_unit"}
    normalized: Dict[str, Dict[str, Any]] = {}
    target_count = 0

    for entity_name, entity in entities.items():
        if not isinstance(entity_name, str) or not entity_name.strip():
            raise ProblemFormalizerError("Tên entity phải là string không rỗng.")

        if not re.match(r"^[a-z][a-z0-9_]*$", entity_name):
            raise ProblemFormalizerError(
                f"Tên entity phải là snake_case hợp lệ, hiện là: {entity_name!r}"
            )

        if not isinstance(entity, dict):
            raise ProblemFormalizerError(f"Entity {entity_name} phải là dictionary.")

        fields = set(entity.keys())
        missing = required_fields - fields
        extra = fields - required_fields
        if missing:
            raise ProblemFormalizerError(f"Entity {entity_name} thiếu trường: {sorted(missing)}")
        if extra:
            raise ProblemFormalizerError(f"Entity {entity_name} có trường thừa: {sorted(extra)}")

        location = str(entity["location"]).strip()
        if location not in {"input", "target"}:
            raise ProblemFormalizerError(
                f"{entity_name}.location chỉ được là input hoặc target, hiện là: {location!r}"
            )

        value = coerce_number(entity["value"], entity_name)

        if location == "input" and value is None:
            raise ProblemFormalizerError(f"{entity_name} có location input thì value bắt buộc là số.")

        if location == "target":
            target_count += 1
            if value is not None:
                raise ProblemFormalizerError(f"{entity_name} là target nên value phải rỗng/null.")

        unit = normalize_empty(entity["unit"])
        grand_unit = normalize_empty(entity["grand_unit"])

        if unit is not None:
            unit = str(unit).strip()
        if grand_unit is not None:
            grand_unit = str(grand_unit).strip()

        normalized[entity_name] = {
            "value": value,
            "unit": unit,
            "location": location,
            "grand_unit": grand_unit,
        }

    if target_count != 1:
        raise ProblemFormalizerError(f"Phải có đúng 1 entity target, hiện có {target_count}.")

    return normalized


def dump_entities(entities: Dict[str, Dict[str, Any]]) -> None:
    ensure_dirs()
    with PROBLEM_ENTITIES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            entities,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def copy_entities_to_downstream_files() -> None:
    shutil.copyfile(PROBLEM_ENTITIES_PATH, PLAN_ENTITIES_PATH)
    shutil.copyfile(PROBLEM_ENTITIES_PATH, STUDENT_ANSWER_ENTITIES_PATH)


def run() -> None:
    try:
        ensure_dirs()
        problem = read_problem()
        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        entities: Optional[Dict[str, Dict[str, Any]]] = None

        for _ in range(max_retries()):
            raw_response = call_openrouter(problem, previous_error=previous_error)

            try:
                parsed_entities = parse_yaml(raw_response)
                candidate_entities = validate_and_normalize(parsed_entities)
                validate_values_are_direct(problem, candidate_entities)
                validate_target_name_matches_question(problem, candidate_entities)
            except ProblemFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            entities = candidate_entities
            break

        if entities is None:
            raise ProblemFormalizerError(str(last_validation_error))

        entities = normalize_item_target_grand_units(entities)
        entities = add_standard_conversion_entities(entities, problem=problem)
        entities = add_default_context_entities(problem, entities)
        dump_entities(entities)
        copy_entities_to_downstream_files()
        write_log("Pass ProblemFormalizer")
        print("Pass ProblemFormalizer")
    except Exception as exc:
        write_log("Fail ProblemFormalizer", str(exc))
        print("Fail ProblemFormalizer")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
