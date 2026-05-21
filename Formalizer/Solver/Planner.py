"""
Formalizer/Solver/Planner.py

Nhiệm vụ:
- Đọc đề bài từ Input/Problem.txt
- Đọc các thực thể đề bài từ Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để sinh kế hoạch giải bài toán
- Ghi kế hoạch vào Output/Plan.yaml
- Sau khi có plan, dùng code Python để thêm các thực thể result vào Output/PlanEntities.yaml
- Ghi trạng thái Pass Planner / Fail Planner vào Output/Log.yaml

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional

Ghi chú:
- LLM chỉ sinh kế hoạch symbolic, không phải nơi thực thi số học chính.
- Các bước trong plan được đặt step1, step2, step3, ...
- Mỗi step gồm:
  - expr
  - result
  - result_unit
  - result_grand_unit
- Code có hỗ trợ đọc alias grand_result_unit từ output LLM, nhưng khi lưu sẽ chuẩn hóa thành result_grand_unit.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "Input" / "Problem.txt"
OUTPUT_DIR = ROOT_DIR / "Output"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_RETRIES = 3


class PlannerError(Exception):
    """Lỗi riêng cho Planner."""


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise PlannerError(f"Không tìm thấy file: {path}")
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlannerError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise PlannerError(f"File YAML phải có dạng dictionary: {path}")

    return data


def write_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    ensure_dirs()
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    log_data["Planner"] = status
    if message:
        log_data["Planner_message"] = message
    elif "Planner_message" in log_data:
        del log_data["Planner_message"]
    write_yaml_file(LOG_PATH, log_data)


def read_problem() -> str:
    if not INPUT_PATH.exists():
        raise PlannerError(f"Không tìm thấy file: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise PlannerError("Input/Problem.txt đang rỗng.")

    return text


def validate_problem_entities(entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    required_fields = {"value", "unit", "location", "grand_unit"}
    normalized: Dict[str, Dict[str, Any]] = {}
    target_count = 0

    if not entities:
        raise PlannerError("Output/ProblemEntities.yaml đang rỗng.")

    for name, entity in entities.items():
        if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise PlannerError(f"Tên entity không hợp lệ: {name!r}")

        if not isinstance(entity, dict):
            raise PlannerError(f"Entity {name} phải là dictionary.")

        missing = required_fields - set(entity.keys())
        if missing:
            raise PlannerError(f"Entity {name} thiếu trường: {sorted(missing)}")

        location = entity.get("location")
        if location not in {"input", "target"}:
            raise PlannerError(f"Entity {name} có location không hợp lệ: {location!r}")

        if location == "target":
            target_count += 1

        normalized[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": location,
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }

    if target_count != 1:
        raise PlannerError(f"ProblemEntities phải có đúng 1 target, hiện có {target_count}.")

    return normalized


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def build_system_prompt() -> str:
    return """
Bạn là một Planner sinh kế hoạch giải toán symbolic từ đề bài và danh sách entity đã formalize.

Nhiệm vụ:
- Sinh các bước tính toán cần thiết để đi từ entity input tới entity target.
- Không tự thêm input entity mới không có trong ProblemEntities.
- Được tạo entity trung gian bằng trường result.
- Bước cuối cùng phải tạo ra đúng entity target đã có trong ProblemEntities.
- Không thực hiện giải thích, chỉ trả YAML thuần.

Mỗi bước có đúng 4 trường:
- expr: biểu thức tính toán symbolic, dùng tên entity. Ví dụ: morning_coffee_price + afternoon_coffee_price
- result: tên entity được tạo ra bởi bước đó.
- result_unit: đơn vị của result. Với scalar có thể để rỗng/null.
- result_grand_unit: grand unit của result theo target. Với scalar có thể để rỗng/null.

Quy tắc step:
- Tên bước phải là step1, step2, step3, ... liên tục, không bỏ số.
- step sau được dùng result của step trước.
- result phải là snake_case.
- expr chỉ nên dùng tên entity, số hằng cần thiết, và toán tử đơn giản: +, -, *, /, parentheses.
- Nếu cần đổi đơn vị, hãy tạo một step riêng. Ví dụ: hours * 60 -> minutes.
- Với bước đổi đơn vị, expr vẫn phải thể hiện phép đổi đơn vị rõ ràng.

Quy tắc không ảo giác:
- Chỉ dùng entity trong ProblemEntities hoặc result của step trước.
- Không tạo bước không cần thiết.
- Không đưa value vào Plan.yaml.
- Không tính toán ra số trong Plan.yaml.
- Không nhân đôi dữ kiện đếm nếu các thành phần đã được liệt kê đầy đủ.
  Ví dụ: "Nancy buys 2 coffees a day", rồi đề cho giá morning coffee và afternoon coffee.
  Khi đó daily cost là morning_coffee_price + afternoon_coffee_price, KHÔNG nhân thêm coffees_per_day.
  Chỉ dùng entity đếm như coffees_per_day nếu đề cho giá của một item đơn lẻ mà chưa liệt kê từng item.
- Với chuỗi tăng/giảm theo hệ số như "each new X has r times as many as the last":
  nếu biết tổng nhiều kỳ và cần tìm kỳ đầu, dùng tổng cấp số nhân.
  Ví dụ có n kỳ, hệ số r, kỳ đầu là first thì total = first * (1 + r + ... + r ** (n - 1)).
  Không được lấy total / n rồi nhân/chia với r; đó là trung bình, không phải kỳ đầu.

Quy tắc đơn vị:
- result_unit là đơn vị trực tiếp của result.
- result_grand_unit là đơn vị đối chiếu theo target.
- Nếu result là target thì result_unit và result_grand_unit phải khớp với unit và grand_unit của target.

Định dạng output bắt buộc:
step1:
  expr: entity_a + entity_b
  result: intermediate_entity
  result_unit: dollars
  result_grand_unit: dollars

step2:
  expr: intermediate_entity * days
  result: target_entity
  result_unit: dollars
  result_grand_unit: dollars

Chỉ trả YAML thuần, không Markdown, không ``` và không giải thích.
""".strip()


def build_user_prompt(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    entities_yaml = yaml.safe_dump(
        entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sinh lại plan, chỉ sửa lỗi đó và giữ schema.
""".rstrip()

    return f"""
Hãy sinh kế hoạch giải toán cho đề bài sau.

Đề bài:
{problem}

ProblemEntities.yaml:
{entities_yaml}
{retry_note}
""".strip()


def call_openrouter(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise PlannerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

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
            {
                "role": "user",
                "content": build_user_prompt(problem, entities, previous_error=previous_error),
            },
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise PlannerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise PlannerError(f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise PlannerError(f"Response OpenRouter không đúng định dạng: {response.text[:1000]}") from exc


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:yaml|yml)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_plan(text: str) -> Dict[str, Any]:
    clean_text = strip_markdown_fence(text)

    try:
        parsed = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise PlannerError(f"LLM trả về YAML không hợp lệ: {exc}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise PlannerError("Plan output phải là dictionary không rỗng.")

    return parsed


def extract_expr_tokens(expr: str) -> List[str]:
    """Lấy các token giống tên biến trong expr."""
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def is_money_unit(unit: Any) -> bool:
    unit_text = str(normalize_empty(unit) or "").lower()
    return unit_text in {"dollar", "dollars", "usd", "$", "cent", "cents"}


def is_count_unit(unit: Any) -> bool:
    unit_text = str(normalize_empty(unit) or "").lower()
    if not unit_text:
        return False
    if is_money_unit(unit_text):
        return False
    if unit_text in {"day", "days", "hour", "hours", "minute", "minutes", "week", "weeks"}:
        return False
    return True


def entity_unit(name: str, problem_entities: Dict[str, Dict[str, Any]], plan: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if name in problem_entities:
        return normalize_empty(problem_entities[name].get("unit"))

    for step in plan.values():
        if step.get("result") == name:
            return normalize_empty(step.get("result_unit"))

    return None


def result_expr(name: str, plan: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for step in plan.values():
        if step.get("result") == name:
            return normalize_empty(step.get("expr"))
    return None


def validate_no_obvious_double_count(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    """
    Bắt lỗi phổ biến: đã cộng giá của từng item trong ngày rồi lại nhân với
    count "2 items per day". Ví dụ coffee: (morning + afternoon) * coffees_per_day.
    """
    for step_name, step in plan.items():
        if not is_money_unit(step.get("result_unit")):
            continue

        expr = step["expr"]
        if "*" not in expr:
            continue

        tokens = extract_expr_tokens(expr)
        count_tokens = [
            token
            for token in tokens
            if token in problem_entities and is_count_unit(problem_entities[token].get("unit"))
        ]
        if not count_tokens:
            continue

        for token in tokens:
            previous_expr = result_expr(token, plan)
            if not previous_expr:
                continue

            previous_tokens = extract_expr_tokens(previous_expr)
            money_inputs = [
                previous_token
                for previous_token in previous_tokens
                if previous_token in problem_entities and is_money_unit(problem_entities[previous_token].get("unit"))
            ]
            if len(set(money_inputs)) >= 2:
                raise PlannerError(
                    f"{step_name}.expr có vẻ nhân đôi dữ kiện đếm {count_tokens}. "
                    f"Step trước {token!r} đã cộng các giá tiền thành phần {sorted(set(money_inputs))}."
                )


def validate_important_scalar_inputs_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    important_markers = ("fraction", "percent", "percentage", "ratio", "multiplier", "factor", "rate")
    unused = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        if normalize_empty(entity.get("unit")) is not None:
            continue
        if not any(marker in name for marker in important_markers):
            continue
        if name not in plan_tokens:
            unused.append(name)

    if unused:
        raise PlannerError(
            f"Plan chưa dùng các scalar input quan trọng: {unused}. "
            "Nếu entity này biểu diễn fraction/ratio/multiplier/rate từ đề, expr phải tham chiếu trực tiếp entity đó."
        )


def target_entity_name(problem_entities: Dict[str, Dict[str, Any]]) -> str:
    targets = [name for name, entity in problem_entities.items() if entity.get("location") == "target"]
    if len(targets) != 1:
        raise PlannerError(f"Phải có đúng 1 target, hiện có {len(targets)}.")
    return targets[0]


def normalize_step_fields(step_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise PlannerError(f"{step_name} phải là dictionary.")

    # Hỗ trợ alias do prompt trước đó từng dùng nhầm tên.
    if "grand_result_unit" in step and "result_grand_unit" not in step:
        step["result_grand_unit"] = step.pop("grand_result_unit")

    required_fields = {"expr", "result", "result_unit", "result_grand_unit"}
    fields = set(step.keys())
    missing = required_fields - fields
    extra = fields - required_fields

    if missing:
        raise PlannerError(f"{step_name} thiếu trường: {sorted(missing)}")
    if extra:
        raise PlannerError(f"{step_name} có trường thừa: {sorted(extra)}")

    expr = step["expr"]
    result = step["result"]

    if not isinstance(expr, str) or not expr.strip():
        raise PlannerError(f"{step_name}.expr phải là string không rỗng.")
    if not isinstance(result, str) or not re.match(r"^[a-z][a-z0-9_]*$", result):
        raise PlannerError(f"{step_name}.result phải là snake_case hợp lệ.")

    result_unit = normalize_empty(step.get("result_unit"))
    result_grand_unit = normalize_empty(step.get("result_grand_unit"))

    if result_unit is not None:
        result_unit = str(result_unit).strip()
    if result_grand_unit is not None:
        result_grand_unit = str(result_grand_unit).strip()

    return {
        "expr": expr.strip(),
        "result": result.strip(),
        "result_unit": result_unit,
        "result_grand_unit": result_grand_unit,
    }


def validate_and_normalize_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    target_name = target_entity_name(problem_entities)
    available_entities = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
    }
    normalized_plan: Dict[str, Dict[str, Any]] = {}

    expected_step_names = [f"step{i}" for i in range(1, len(raw_plan) + 1)]
    actual_step_names = list(raw_plan.keys())

    if actual_step_names != expected_step_names:
        raise PlannerError(
            f"Step phải liên tục theo thứ tự {expected_step_names}, hiện là {actual_step_names}."
        )

    for step_index, step_name in enumerate(expected_step_names, start=1):
        step = normalize_step_fields(step_name, raw_plan[step_name])

        if "=" in step["expr"]:
            raise PlannerError(f"{step_name}.expr không được chứa dấu '='; expr phải là biểu thức tính toán.")

        expr_tokens = extract_expr_tokens(step["expr"])
        unknown_tokens = [token for token in expr_tokens if token not in available_entities]
        if unknown_tokens:
            raise PlannerError(
                f"{step_name}.expr dùng entity chưa tồn tại hoặc chưa được tạo: {unknown_tokens}"
            )

        result = step["result"]
        if result in available_entities and result != target_name:
            raise PlannerError(
                f"{step_name}.result {result!r} đã tồn tại trong input entity, không nên ghi đè."
            )

        if step_index < len(expected_step_names) and result == target_name:
            raise PlannerError("Target chỉ nên được tạo ở bước cuối cùng.")

        available_entities.add(result)
        normalized_plan[step_name] = step

    last_step = normalized_plan[expected_step_names[-1]]
    if last_step["result"] != target_name:
        raise PlannerError(
            f"Bước cuối phải tạo target {target_name!r}, hiện tạo {last_step['result']!r}."
        )

    target = problem_entities[target_name]
    target_unit = normalize_empty(target.get("unit"))
    target_grand_unit = normalize_empty(target.get("grand_unit"))

    if normalize_empty(last_step.get("result_unit")) != target_unit:
        raise PlannerError(
            f"result_unit của bước cuối phải khớp target.unit: {target_unit!r}."
        )

    if normalize_empty(last_step.get("result_grand_unit")) != target_grand_unit:
        raise PlannerError(
            f"result_grand_unit của bước cuối phải khớp target.grand_unit: {target_grand_unit!r}."
        )

    validate_no_obvious_double_count(normalized_plan, problem_entities)
    validate_important_scalar_inputs_used(normalized_plan, problem_entities)

    return normalized_plan


def max_retries() -> int:
    raw = os.getenv("PLANNER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise PlannerError("PLANNER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise PlannerError("PLANNER_MAX_RETRIES phải >= 1.")
    return value


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    return False


def parse_numeric(value: Any) -> float | int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    number = float(str(value).replace(",", ""))
    if number.is_integer():
        return int(number)
    return number


def safe_eval_conversion_expr(expr: str, entities: Dict[str, Dict[str, Any]]) -> Optional[float | int]:
    """
    Thử tính value cho các bước đổi đơn vị đơn giản.

    Hàm này không bắt buộc mọi step đều tính được value.
    Nó chỉ tính khi:
    - expr chỉ gồm số, toán tử đơn giản, ngoặc, và entity đã có value số.
    - không dùng function/call/import/attribute.

    Nếu không đủ điều kiện, trả None.
    """
    allowed_chars = set("0123456789.+-*/() _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if any(ch not in allowed_chars for ch in expr):
        return None

    tokens = extract_expr_tokens(expr)
    local_values: Dict[str, float | int] = {}

    for token in tokens:
        entity = entities.get(token)
        if not entity:
            return None
        value = normalize_empty(entity.get("value"))
        if not is_number(value):
            return None
        local_values[token] = parse_numeric(value)

    try:
        result = eval(expr, {"__builtins__": {}}, local_values)  # noqa: S307 - sandboxed simple arithmetic
    except Exception:
        return None

    if not is_number(result):
        return None

    return parse_numeric(result)


def looks_like_unit_conversion(step: Dict[str, Any], existing_entities: Dict[str, Dict[str, Any]]) -> bool:
    """
    Heuristic để nhận diện bước đổi đơn vị.

    Vì Planner không phải executor chính, hàm này chỉ dùng để thêm value cho biến mới
    khi step trông giống đổi đơn vị và có thể tính an toàn.
    """
    expr = step["expr"]
    result_unit = normalize_empty(step.get("result_unit"))
    result_grand_unit = normalize_empty(step.get("result_grand_unit"))

    if result_unit is None:
        return False

    expr_tokens = extract_expr_tokens(expr)
    if len(expr_tokens) != 1:
        return False

    source_name = expr_tokens[0]
    source = existing_entities.get(source_name)
    if not source:
        return False

    source_unit = normalize_empty(source.get("unit"))
    if source_unit is None:
        return False

    if source_unit == result_unit:
        return False

    # Nếu grand unit đổi theo cùng target hoặc unit trực tiếp đổi khác nhau, coi là đổi đơn vị.
    return True or result_grand_unit is not None


def merge_plan_results_into_plan_entities(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Thêm result của từng step vào PlanEntities.yaml bằng Python.

    Quy tắc:
    - Nếu result chưa có entity: thêm mới.
    - Nếu result đã có entity target: giữ nguyên location target.
    - Nếu là bước đổi đơn vị và có thể tính toán an toàn: thêm value cho biến result.
    - Với step bình thường: value để rỗng/null vì việc thực thi số học nên thuộc module executor/solver khác.
    """
    plan_entities: Dict[str, Dict[str, Any]] = {
        name: {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": entity.get("location"),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }
        for name, entity in problem_entities.items()
    }

    for step_name, step in plan.items():
        result = step["result"]
        result_unit = normalize_empty(step.get("result_unit"))
        result_grand_unit = normalize_empty(step.get("result_grand_unit"))

        computed_value: Optional[float | int] = None
        if looks_like_unit_conversion(step, plan_entities):
            computed_value = safe_eval_conversion_expr(step["expr"], plan_entities)

        if result not in plan_entities:
            plan_entities[result] = {
                "value": computed_value,
                "unit": result_unit,
                "location": step_name,
                "grand_unit": result_grand_unit,
            }
        else:
            old_location = plan_entities[result].get("location")
            plan_entities[result]["unit"] = result_unit
            if old_location != "target":
                plan_entities[result]["location"] = step_name
            plan_entities[result]["grand_unit"] = result_grand_unit

            # Không tự tính target value ở Planner, trừ khi chính nó là bước đổi đơn vị đơn giản.
            if computed_value is not None:
                plan_entities[result]["value"] = computed_value
            else:
                plan_entities[result]["value"] = normalize_empty(plan_entities[result].get("value"))

    return plan_entities


def run() -> None:
    try:
        ensure_dirs()
        problem = read_problem()
        raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=True)
        problem_entities = validate_problem_entities(raw_problem_entities)

        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        plan: Optional[Dict[str, Dict[str, Any]]] = None

        for _ in range(max_retries()):
            raw_response = call_openrouter(
                problem,
                problem_entities,
                previous_error=previous_error,
            )

            try:
                raw_plan = parse_plan(raw_response)
                candidate_plan = validate_and_normalize_plan(raw_plan, problem_entities)
            except PlannerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            plan = candidate_plan
            break

        if plan is None:
            raise PlannerError(str(last_validation_error))

        write_yaml_file(PLAN_PATH, plan)

        plan_entities = merge_plan_results_into_plan_entities(plan, problem_entities)
        write_yaml_file(PLAN_ENTITIES_PATH, plan_entities)

        write_log("Pass Planner")
        print("Pass Planner")
    except Exception as exc:
        write_log("Fail Planner", str(exc))
        print("Fail Planner")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
