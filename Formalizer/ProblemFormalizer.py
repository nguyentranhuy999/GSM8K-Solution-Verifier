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
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 5
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
    "twins": 2,
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
    "pair": "pair",
    "pairs": "pair",
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
    ("day", "year", "unit_conversion_days_per_year", 365),
    ("week", "year", "unit_conversion_weeks_per_year", 52),
    ("month", "year", "unit_conversion_months_per_year", 12),
    ("cent", "dollar", "unit_conversion_cents_per_dollar", 100),
    ("ounce", "pound", "unit_conversion_ounces_per_pound", 16),
    ("pound", "ton", "unit_conversion_pounds_per_ton", 2000),
    ("item", "pair", "unit_conversion_items_per_pair", 2),
    ("item", "dozen", "unit_conversion_items_per_dozen", 12),
]

STANDARD_CONVERSION_ALIASES = {
    entity_name.replace("unit_conversion_", ""): entity_name
    for _, _, entity_name, _ in STANDARD_CONVERSION_FACTORS
}

CONVERSION_BY_UNITS = {
    (smaller_unit, larger_unit): value
    for smaller_unit, larger_unit, _, value in STANDARD_CONVERSION_FACTORS
}


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
    Chỉ dùng tên dạng *_per_* khi đề thực sự nói rate trên một đơn vị, như "per", "each",
    "a/an", hoặc mẫu tương đương với mẫu số là 1.
    Ví dụ "1/4 inch represents 8 miles" KHÔNG phải miles_per_inch value 8.
    Hãy giữ hai số trực tiếp thành hai entity riêng có cùng source clause.
12. Với quan hệ so sánh như "X has 2 fewer than Y", "X has 5 more than Y":
    số trực tiếp là độ chênh lệch, không phải số lượng thật của X.
    Tên entity phải thể hiện đây là độ chênh, ví dụ books_borrowed_fewer_than_books_bought.
    Chỉ dùng more_than/fewer_than/less_than cho quan hệ cộng/trừ.
    Với "twice as many as", "three times as many as", "half as many as",
    đây là quan hệ nhân/chia, phải tạo scalar multiplier/fraction, không đặt tên là more_than.
    Ví dụ "Kenley has twice as many as McKenna" -> kenley_to_mckenna_multiplier: 2.
    Ví dụ "Joshua made half as many as Miles" -> joshua_to_miles_fraction: 0.5.
    Ví dụ "twins" nghĩa là 2, hãy tạo scalar/value 2 nếu cần tính số con.
13. Không được bỏ sót số trực tiếp đi kèm đơn vị trong đề bài.
    Ví dụ "in 7 days" phải có một input entity value 7, unit days, không được bỏ vì đã có "5 days" trước đó.
    Ví dụ "$.50 each" -> 0.5, "40A each" -> 40, "$500/month" -> 500.
    Ví dụ "three iPhones", "four grades", "five weeks" cũng phải có input entity value tương ứng.
    Nếu cùng một số xuất hiện ở hai dữ kiện khác nhau, không được dùng một entity để đại diện cho cả hai nếu vai trò khác nhau.
    Ví dụ "5 llamas got pregnant with twins" và "traded 8 calves for 2 new adult llamas" cần có cả twins/count 2 và new_adult_llamas 2 nếu cả hai tham gia tính.
    Ví dụ "finished half of a 500 piece puzzle" không được tạo 250; phải giữ 500 và fraction 0.5.
    Không đổi đơn vị ngay trong ProblemEntities. Ví dụ "12000 meters" phải giữ value 12000, unit meters; không đổi thành 12 kilometers.
    Không tạo complement fraction. Ví dụ có "2/5 are boys" thì giữ 0.4 cho boys_fraction; không tự tạo girls_fraction 0.6.
14. Target phải là đúng đại lượng được hỏi, không phải đại lượng trung gian.
    Ví dụ câu hỏi "How many old books ... reread" thì target phải chứa reread, như old_books_to_reread.
    Không đặt target là total_books_needed nếu đề hỏi số sách cũ phải đọc lại.
    Ví dụ câu hỏi "How many friends can she invite?" thì target phải là số friends, như total_friends.
    Không đặt target là total_cost nếu đề hỏi số bạn được mời.
15. Mỗi entity phải có source là phrase/clause gốc trong đề bài làm bằng chứng cho value/target.
    - Với input, source phải là đoạn ngắn trong đề có chứa số hoặc chữ số tương ứng.
    - Với target, source là câu hỏi hoặc phrase hỏi đại lượng cần tìm.
    - Nếu nhiều số nằm trong cùng một quan hệ, dùng cùng source clause để giữ quan hệ đó.
      Ví dụ "1/4 inch represents 8 miles of actual road distance" thì entity 0.25 inches
      và entity 8 miles đều dùng source này.
    - Planner sẽ dùng source để hiểu vai trò của số, nên không nhét toàn bộ quan hệ vào tên biến.

Mỗi thực thể phải có đúng 5 trường:
- value: giá trị số được cho trực tiếp trong đề bài. Với target, để rỗng bằng null.
- unit: đơn vị trực tiếp của thực thể, viết bằng tiếng Anh dạng số nhiều nếu phù hợp. Ví dụ: dollars, days, books, pens. Với scalar như phần trăm, phân số, nhân tử, có thể để rỗng.
- location: input hoặc target.
- grand_unit: đơn vị đối chiếu theo target.
- source: chuỗi ngắn trích từ đề bài, luôn đặt trong quotes. Với input, source là phrase/clause chứa số. Với target, source là phrase/câu hỏi.

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
- Không thêm trường ngoài 5 trường đã yêu cầu.

Ví dụ:
morning_coffee_price:
  value: 3.00
  unit: dollars
  location: input
  grand_unit: dollars
  source: "morning coffee $3"

afternoon_coffee_price:
  value: 2.50
  unit: dollars
  location: input
  grand_unit: dollars
  source: "afternoon coffee $2.50"

days:
  value: 20
  unit: days
  location: input
  grand_unit:
  source: "20 days"

fraction_wasted_pac_man:
  value: 0.3333333333333333
  unit:
  location: input
  grand_unit:
  source: "a third of them were Pac-Man"

parent_token_multiplier:
  value: 7
  unit:
  location: input
  grand_unit:
  source: "seven times as many tokens as a parent"

total_cost:
  value:
  unit: dollars
  location: target
  grand_unit: dollars
  source: "How much did she spend?"
""".strip()


def build_user_prompt(problem: str, previous_error: Optional[str] = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sinh lại YAML, chỉ dùng số được nêu trực tiếp trong đề. Không tạo số suy ra.
Nếu lỗi liên quan tên *_per_* với source dạng "A represents/corresponds to/is equivalent to B"
mà A không phải 1 đơn vị, tuyệt đối không dùng bất kỳ entity name nào chứa "_per_" cho
hai số trong relation đó. Hãy tạo hai entity input riêng có cùng source clause, ví dụ
scale_source_quantity và scale_target_quantity hoặc map_scale_map_distance và
map_scale_actual_distance.
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
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
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
        package_conversion = entity_name in {
            "unit_conversion_items_per_pair",
            "unit_conversion_items_per_dozen",
        }
        if larger_unit not in unit_bases:
            continue
        if not package_conversion and smaller_unit not in unit_bases:
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


def normalize_standard_conversion_aliases(
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    normalized = dict(entities)

    for alias_name, canonical_name in STANDARD_CONVERSION_ALIASES.items():
        if alias_name not in normalized:
            continue
        if canonical_name in normalized:
            normalized.pop(alias_name)
            continue

        entity = dict(normalized.pop(alias_name))
        entity["location"] = "input"
        entity["unit"] = None
        entity["grand_unit"] = None
        normalized[canonical_name] = entity

    return normalized


def extract_direct_unit_mentions(problem: str) -> list[tuple[float, str, str]]:
    text = problem.lower()
    mentions: list[tuple[float, str, str]] = []
    pattern = re.compile(
        r"(?<![\w/])"
        r"(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
        r"(?!\s*/\s*\d|%)"
        r"\s*(?:/\s*)?([a-z][a-z-]*)\b"
    )
    ignored_following_words = {"am", "pm", "percent", "percentage", "a", "an", "the"}

    for match in pattern.finditer(text):
        unit_word = match.group(2)
        if unit_word in ignored_following_words:
            continue
        base = UNIT_BASE_ALIASES.get(unit_word)
        if not base:
            continue
        mentions.append((float(match.group(1).replace(",", "")), unit_word, base))

    return mentions


def repair_converted_input_values(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    mentions = extract_direct_unit_mentions(problem)
    if not mentions:
        return entities

    repaired = dict(entities)
    for name, entity in list(repaired.items()):
        if entity.get("location") != "input":
            continue
        value = normalize_empty(entity.get("value"))
        if value is None:
            continue
        current_bases = unit_base_keys(entity.get("unit"))
        if not current_bases:
            continue

        numeric_value = float(value)
        for original_value, original_unit_word, original_base in mentions:
            for current_base in current_bases:
                factor = CONVERSION_BY_UNITS.get((original_base, current_base))
                if not factor:
                    continue
                if not numbers_equal(original_value / factor, numeric_value):
                    continue

                entity = dict(entity)
                entity["value"] = int(original_value) if original_value.is_integer() else original_value
                entity["unit"] = original_unit_word
                repaired[name] = entity
                break
            else:
                continue
            break

    return repaired


def drop_computed_complement_scalars(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Loại các scalar complement mà LLM hay tự suy ra.

    Ví dụ đề chỉ nói "2/5 are boys" thì 3/5 girls là phép tính
    identity - boys_fraction, không phải input trực tiếp.
    """
    direct_values = extract_direct_numeric_values(problem)
    direct_unit_values = [
        entity.get("value")
        for entity in entities.values()
        if entity.get("location") == "input"
        and normalize_empty(entity.get("unit")) is not None
    ]
    normalized = dict(entities)

    for name, entity in list(normalized.items()):
        if entity.get("location") != "input":
            continue
        if normalize_empty(entity.get("unit")) is not None:
            continue

        terms = set(name.split("_"))
        if not terms & {"fraction", "percent", "percentage", "ratio", "share", "portion"}:
            continue

        value = normalize_empty(entity.get("value"))
        if value is None:
            continue
        if any(numbers_equal(value, direct_value) for direct_value in direct_values):
            continue
        if any(numbers_equal(value, unit_value) for unit_value in direct_unit_values if unit_value is not None):
            continue
        if any(
            0 < float(direct_value) < 1
            and numbers_equal(float(value), 1 - float(direct_value))
            for direct_value in direct_values
        ):
            normalized.pop(name)

    return normalized


def drop_unknown_zero_placeholders(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Một lỗi thường gặp của LLM là bịa age/count = 0 cho thông tin chỉ nói bằng
    quan hệ như "younger brother". Nếu đề không có số 0 trực tiếp, bỏ entity đó
    để Planner dùng quan hệ/ngữ cảnh thay vì một số giả.
    """
    direct_values = extract_direct_numeric_values(problem)
    if any(numbers_equal(value, 0) for value in direct_values):
        return entities

    normalized = dict(entities)
    placeholder_terms = {"age", "count", "number"}
    for name, entity in list(normalized.items()):
        if entity.get("location") != "input":
            continue
        value = normalize_empty(entity.get("value"))
        if value is None or not numbers_equal(value, 0):
            continue
        if not (placeholder_terms & set(name.split("_"))):
            continue
        normalized.pop(name)

    return normalized


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
    text = problem.lower()

    if problem_asks_invited_friends(problem) and "host_count" not in enriched:
        enriched["host_count"] = {
            "value": 1,
            "unit": "people",
            "location": "input",
            "grand_unit": "people",
        }

    target_percentage = any(
        entity.get("location") == "target"
        and (
            "percent" in str(name).lower().split("_")
            or "percentage" in str(name).lower().split("_")
            or "percent" in str(name).lower()
            or "percentage" in str(name).lower()
            or normalize_unit_text(entity.get("unit")) in {"percent", "percentage"}
            or normalize_unit_text(entity.get("grand_unit")) in {"percent", "percentage"}
        )
        for name, entity in enriched.items()
    )
    if target_percentage and "percentage_scale" not in enriched:
        enriched["percentage_scale"] = {
            "value": 100,
            "unit": None,
            "location": "input",
            "grand_unit": None,
        }

    needs_identity = bool(
        re.search(
            r"\bas many as\b|\bremainder\b|\bremaining\b|\bthe rest\b|\bleft\b|\bdiscount\b|\bnot\s+(?:in|on|at)\b"
            r"|\bgive\b.*\beach\b|\bsame amount\b|\btwice\b|\bdouble\b|\btriple\b|\bthrice\b",
            text,
        )
    )
    if needs_identity and "identity_multiplier" not in enriched:
        enriched["identity_multiplier"] = {
            "value": 1,
            "unit": None,
            "location": "input",
            "grand_unit": None,
        }

    if re.search(r"\bsplit\s+(?:the\s+)?remaining\b", problem.lower()) and "split_count" not in enriched:
        enriched["split_count"] = {
            "value": 2,
            "unit": None,
            "location": "input",
            "grand_unit": None,
        }

    if (
        re.search(r"\broommates?\b", problem.lower())
        and re.search(r"\bdivide\b|\bequally\b|\bshare\b", problem.lower())
        and "host_count" not in enriched
    ):
        enriched["host_count"] = {
            "value": 1,
            "unit": "people",
            "location": "input",
            "grand_unit": "people",
        }

    family_match = re.search(
        r"\bfamily\s+(?:consists\s+of|includes)\s+(.*?)(?:\.|\?|$)",
        text,
    )
    if family_match:
        family_members = family_match.group(1)

        def add_people_count(name: str, value: int) -> None:
            if name in enriched:
                return
            enriched[name] = {
                "value": value,
                "unit": "people",
                "location": "input",
                "grand_unit": "people",
            }

        if re.search(r"\b(?:her|him|herself|himself|self|me)\b", family_members):
            add_people_count("self_count", 1)
        if re.search(r"\b(?:younger|older|little|big)?\s*(?:brother|sister|sibling)\b", family_members):
            add_people_count("sibling_count", 1)
        if re.search(r"\bparents\b", family_members):
            add_people_count("parents_count", 2)
        else:
            parent_count = 0
            parent_count += len(re.findall(r"\bmother\b", family_members))
            parent_count += len(re.findall(r"\bfather\b", family_members))
            if parent_count:
                add_people_count("parents_count", parent_count)
        if re.search(r"\bgrandparents\b", family_members):
            add_people_count("grandparents_count", 2)
        else:
            grandparent_count = 0
            grandparent_count += len(re.findall(r"\bgrandfather\b", family_members))
            grandparent_count += len(re.findall(r"\bgrandmother\b", family_members))
            if grandparent_count:
                add_people_count("grandparents_count", grandparent_count)

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


PAIR_TARGET_UNITS = {
    "earring",
    "earrings",
    "shoe",
    "shoes",
    "sock",
    "socks",
    "glove",
    "gloves",
}


def normalize_pair_target_units(
    entities: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    normalized = dict(entities)

    has_pair_input = any(
        entity.get("location") == "input" and "pair" in unit_base_keys(entity.get("unit"))
        for entity in normalized.values()
    )
    if not has_pair_input:
        return normalized

    for name, entity in list(normalized.items()):
        if entity.get("location") != "target":
            continue
        if "pair" not in unit_base_keys(entity.get("unit")):
            continue

        name_terms = {term for term in name.split("_") if term}
        target_terms = name_terms & PAIR_TARGET_UNITS
        if not target_terms:
            continue

        target_unit = sorted(target_terms, key=len, reverse=True)[0]
        entity = dict(entity)
        entity["unit"] = target_unit
        entity["grand_unit"] = target_unit
        normalized[name] = entity

    return normalized


def normalize_relative_delta_units(
    entities: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    targets = [
        entity
        for entity in entities.values()
        if entity.get("location") == "target"
    ]
    if len(targets) != 1:
        return entities

    target_unit = normalize_empty(targets[0].get("unit"))
    target_grand_unit = normalize_empty(targets[0].get("grand_unit")) or target_unit
    if target_unit is None:
        return entities

    normalized = dict(entities)
    for name, entity in list(normalized.items()):
        if not any(marker in name for marker in ("_more_than", "_less_than", "_fewer_than")):
            continue
        if normalize_empty(entity.get("unit")) is not None:
            continue

        entity = dict(entity)
        entity["unit"] = target_unit
        entity["grand_unit"] = target_grand_unit
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

    for match in re.finditer(r"\b(\d+(?:,\d{3})*)\s+(\d+)\s*/\s*(\d+)\b", text):
        whole = float(match.group(1).replace(",", ""))
        numerator = float(match.group(2))
        denominator = float(match.group(3))
        if denominator == 0:
            continue
        values.append(whole + numerator / denominator)
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", text):
        if span_is_covered(match.span(), covered_spans):
            continue
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
        r"(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine))?))[\s-]+"
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

    for match in re.finditer(r"\b(twice|double|doubled|thrice|triple|tripled|twins)\b", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        values.append(float(MULTIPLIER_WORDS[match.group(1)]))
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*%", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        values.append(float(match.group(1).replace(",", "")) / 100)
        covered_spans.append(match.span())

    for match in re.finditer(r"\b(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s+(?:percent|percentage)\b", text):
        if span_is_covered(match.span(), covered_spans):
            continue
        values.append(float(match.group(1).replace(",", "")) / 100)
        covered_spans.append(match.span())

    percent_word_pattern = re.compile(
        r"\b((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand))*)\s+(?:percent|percentage)\b"
    )
    for match in percent_word_pattern.finditer(text):
        if span_is_covered(match.span(), covered_spans):
            continue
        value = parse_number_word_phrase(match.group(1))
        if value is None:
            continue
        values.append(float(value) / 100)
        covered_spans.append(match.span())

    for match in re.finditer(
        r"(?<![\w/])(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?!\s*/\s*\d|%)",
        text,
    ):
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


def extract_required_scalar_values(problem: str) -> list[float]:
    text = problem.lower()
    values: list[float] = []
    covered_spans: list[tuple[int, int]] = []

    for match in re.finditer(r"\b(\d+(?:,\d{3})*)\s+(\d+)\s*/\s*(\d+)\b", text):
        whole = float(match.group(1).replace(",", ""))
        numerator = float(match.group(2))
        denominator = float(match.group(3))
        if denominator == 0:
            continue
        values.append(whole + numerator / denominator)
        covered_spans.append(match.span())

    scalar_patterns = [
        r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b",
        r"\b(?:(?:a|an)|(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        r"(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine))?))[\s-]+"
        r"(?:half|halves|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|"
        r"eighths?|ninths?|tenths?)\b",
        r"\b(?:half|quarter)\b",
        r"\b(?:twice|double|doubled|thrice|triple|tripled)\b",
        r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*%",
        r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s+(?:percent|percentage)\b",
        r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand)(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
        r"seventy|eighty|ninety|hundred|thousand))*)\s+(?:percent|percentage)\b",
        r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s+times\b",
        r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand)(?:[\s-]+(?:zero|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
        r"seventy|eighty|ninety|hundred|thousand))*)\s+times\b",
    ]

    for pattern in scalar_patterns:
        for match in re.finditer(pattern, text):
            if span_is_covered(match.span(), covered_spans):
                continue
            snippet = match.group(0)
            extracted = extract_direct_numeric_values(snippet)
            if not extracted:
                continue
            values.append(extracted[0])
            covered_spans.append(match.span())

    return values


def extract_required_word_unit_numeric_values(problem: str) -> list[float]:
    text = problem.lower()
    values: list[float] = []

    number_word = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred|thousand)(?:[\s-]+(?:zero|one|two|three|four|five|six|"
        r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
        r"sixty|seventy|eighty|ninety|hundred|thousand))*"
    )
    pattern = re.compile(rf"\b({number_word})\s+([a-z][a-z-]*)\b")
    ignored_following_words = {
        "percent",
        "percentage",
        "times",
        "more",
        "less",
        "fewer",
        "as",
        "than",
        "people",
        "person",
        "girls",
        "girl",
        "boys",
        "boy",
        "men",
        "man",
        "women",
        "woman",
    }

    for match in pattern.finditer(text):
        following_word = match.group(2)
        if following_word in ignored_following_words:
            continue
        value = parse_number_word_phrase(match.group(1))
        if value is None:
            continue
        if value == 1:
            continue
        values.append(float(value))

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
        r"(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
        r"(?!\s*/\s*\d|%)"
        r"\s*(?:/\s*)?([a-z][a-z-]*)\b"
    )
    ignored_following_words = {
        "am",
        "pm",
        "percent",
        "percentage",
        "a",
        "an",
        "the",
    }

    for match in pattern.finditer(text):
        following_word = match.group(2)
        if following_word in ignored_following_words:
            continue
        prefix = text[max(0, match.start() - 12): match.start()]
        if re.search(r"\bat\s+least\s+$", prefix):
            continue
        value = float(match.group(1).replace(",", ""))
        if numbers_equal(value, 1):
            continue
        values.append(value)

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
        if entity_name.startswith("unit_conversion_"):
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

    missing_required_values: list[float] = []
    required_values = (
        extract_required_unit_numeric_values(problem)
        + extract_required_word_unit_numeric_values(problem)
        + extract_required_scalar_values(problem)
    )
    for required_value in required_values:
        if not any(numbers_equal(input_value, required_value) for input_value in input_values):
            missing_required_values.append(required_value)

    if missing_required_values:
        unique_missing: list[float] = []
        for value in missing_required_values:
            if not any(numbers_equal(value, existing) for existing in unique_missing):
                unique_missing.append(value)
        missing = ", ".join(format_number_for_error(value) for value in unique_missing)
        raise ProblemFormalizerError(
            f"Thiếu thực thể input cho số/scalar trực tiếp bắt buộc trong đề: {missing}."
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
    if re.search(r"\bnot\s+in\b|\bnot\s+(?:on|at)\b", text):
        required_terms.append("not")

    missing_terms = [term for term in required_terms if term not in target_name]
    if missing_terms:
        raise ProblemFormalizerError(
            f"Target {target_name!r} chưa khớp đại lượng được hỏi; "
            f"tên target cần chứa {missing_terms}."
        )

    if re.search(r"\bhow much\b.*\bgive\b.*\beach\b", text) and not any(
        term in target_name for term in ("give", "each", "per")
    ):
        raise ProblemFormalizerError(
            f"Target {target_name!r} chưa khớp đại lượng được hỏi; "
            "câu hỏi hỏi số tiền phải give to each người, target cần thể hiện amount_each/per_person."
        )


def validate_multiplicative_relationship_names(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
) -> None:
    text = problem.lower()
    if not re.search(r"\b(?:twice|double|thrice|triple|[a-z]+|\d+)\s+as\s+many\s+as\b", text):
        return

    required_scalar_values = extract_required_scalar_values(problem)
    for entity_name, entity in entities.items():
        if "_more_than_" not in entity_name and "_less_than_" not in entity_name and "_fewer_than_" not in entity_name:
            continue

        value = entity.get("value")
        if value is None:
            continue
        if not any(numbers_equal(value, scalar_value) for scalar_value in required_scalar_values):
            continue

        raise ProblemFormalizerError(
            f"{entity_name!r} đang dùng tên quan hệ cộng/trừ cho một multiplier/fraction. "
            "Với 'twice/three times/half as many as', hãy tạo entity dạng *_multiplier hoặc *_fraction, "
            "không dùng more_than/fewer_than/less_than."
        )


def validate_per_entity_names_match_source(
    entities: Dict[str, Dict[str, Any]],
) -> None:
    """
    Không cho entity name dạng *_per_* encode sai một relation chưa chuẩn hóa.

    Ví dụ "1/4 inch represents 8 miles" không được đặt là miles_per_inch: 8,
    vì "per inch" ngụ ý mẫu số là 1 inch trong khi source nói 0.25 inch.
    """
    relation_words = r"\b(?:represents?|corresponds?\s+to|equivalent\s+to|equals?)\b"

    for entity_name, entity in entities.items():
        if "_per_" not in entity_name:
            continue
        if entity_name.startswith("unit_conversion_"):
            continue

        source = normalize_empty(entity.get("source"))
        if source is None:
            continue

        source_text = str(source).lower()
        if not re.search(relation_words, source_text):
            continue

        source_values = extract_direct_numeric_values(source_text)
        if len(source_values) < 2:
            continue

        first_value = source_values[0]
        if numbers_equal(first_value, 1):
            continue

        raise ProblemFormalizerError(
            f"{entity_name!r} dùng tên dạng *_per_* nhưng source {source!r} có vế trái "
            f"không phải 1 đơn vị. Hãy tạo hai entity riêng có cùng source clause thay vì "
            "chuẩn hóa rate trong tên biến."
        )


def validate_and_normalize(entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    required_fields = {"value", "unit", "location", "grand_unit", "source"}
    normalized: Dict[str, Dict[str, Any]] = {}
    target_count = 0

    for raw_entity_name, entity in entities.items():
        if not isinstance(raw_entity_name, str) or not raw_entity_name.strip():
            raise ProblemFormalizerError("Tên entity phải là string không rỗng.")

        entity_name = raw_entity_name.strip().lower()

        if not re.match(r"^[a-z][a-z0-9_]*$", entity_name):
            raise ProblemFormalizerError(
                f"Tên entity phải là snake_case hợp lệ, hiện là: {raw_entity_name!r}"
            )
        if entity_name in normalized:
            raise ProblemFormalizerError(f"Tên entity bị trùng sau normalize lowercase: {entity_name!r}")

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
        source = normalize_empty(entity.get("source"))

        if unit is not None:
            unit = str(unit).strip()
        if grand_unit is not None:
            grand_unit = str(grand_unit).strip()
        if source is None or not str(source).strip():
            raise ProblemFormalizerError(
                f"{entity_name}.source phải là phrase/clause gốc trong đề bài, không được rỗng."
            )
        source = str(source).strip()

        normalized[entity_name] = {
            "value": value,
            "unit": unit,
            "location": location,
            "grand_unit": grand_unit,
            "source": source,
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
                candidate_entities = normalize_standard_conversion_aliases(candidate_entities)
                candidate_entities = repair_converted_input_values(problem, candidate_entities)
                candidate_entities = drop_computed_complement_scalars(problem, candidate_entities)
                candidate_entities = drop_unknown_zero_placeholders(problem, candidate_entities)
                validate_values_are_direct(problem, candidate_entities)
                validate_target_name_matches_question(problem, candidate_entities)
                validate_multiplicative_relationship_names(problem, candidate_entities)
                validate_per_entity_names_match_source(candidate_entities)
            except ProblemFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            entities = candidate_entities
            break

        if entities is None:
            raise ProblemFormalizerError(str(last_validation_error))

        entities = normalize_item_target_grand_units(entities)
        entities = normalize_pair_target_units(entities)
        entities = normalize_relative_delta_units(entities)
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
