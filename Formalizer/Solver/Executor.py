"""
Formalizer/Solver/Executor.py

Nhiệm vụ:
1. Đọc:
   - Output/PlanEntities.yaml
   - Output/Plan.yaml
2. Thực thi tuần tự các bước trong Plan.yaml bằng Python.
3. Gắn reported_expr vào từng step trong Plan.yaml.
4. Gắn value cho các entity result chưa có value trong PlanEntities.yaml.
5. Thêm expr và formalized_expr cho từng entity trong PlanEntities.yaml.
6. Gọi Verify/InsideChecker.py.
7. Nếu Output/Error.yaml có lỗi, gọi LLM qua OpenRouter để sửa Plan.yaml / PlanEntities.yaml,
   sau đó lặp lại cho tới khi InsideChecker không còn lỗi.

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional
EXECUTOR_MAX_REPAIR_ITERATIONS=5  # optional, dùng để tránh vòng lặp vô hạn

Ghi chú:
- Executor chỉ cho phép biểu thức số học an toàn: +, -, *, /, **, %, ngoặc, unary +/-.
- Không cho phép function call, attribute, import, indexing, statement, assignment.
- LLM chỉ được dùng ở pha repair khi InsideChecker báo lỗi.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
import yaml
from dotenv import load_dotenv


getcontext().prec = 28

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "Output"
INPUT_PATH = ROOT_DIR / "Input" / "Problem.txt"
PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
ERROR_PATH = OUTPUT_DIR / "Error.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"
INSIDE_CHECKER_PATH = ROOT_DIR / "Verifier" / "InsideChecker.py"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_REPAIR_ITERATIONS = 5


class ExecutorError(Exception):
    """Lỗi riêng cho Executor."""


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def read_text(path: Path, *, required: bool = True) -> str:
    if not path.exists():
        if required:
            raise ExecutorError(f"Không tìm thấy file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise ExecutorError(f"Không tìm thấy file: {path}")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ExecutorError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ExecutorError(f"File YAML phải là dictionary: {path}")
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
    log_data["Executor"] = status
    if message:
        log_data["Executor_message"] = message
    elif "Executor_message" in log_data:
        del log_data["Executor_message"]
    write_yaml_file(LOG_PATH, log_data)


# -----------------------------------------------------------------------------
# Decimal / formatting helpers
# -----------------------------------------------------------------------------

def to_decimal(value: Any, *, name: str = "value") -> Decimal:
    value = normalize_empty(value)
    if value is None:
        raise ExecutorError(f"{name} đang rỗng, không thể dùng để tính toán.")
    if isinstance(value, bool):
        raise ExecutorError(f"{name} không được là boolean.")

    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ExecutorError(f"{name} phải là số, hiện là {value!r}.") from exc


def decimal_to_yaml_number(value: Decimal) -> int | float:
    value = normalize_decimal(value)
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def normalize_decimal(value: Decimal) -> Decimal:
    if value == value.to_integral_value():
        return value.quantize(Decimal("1"))
    return value.normalize()


def format_decimal(value: Decimal, *, min_decimal_places: int = 0) -> str:
    """Format Decimal dễ đọc, ví dụ 5.50 được giữ nếu min_decimal_places=2."""
    value = normalize_decimal(value)

    if value == value.to_integral_value():
        if min_decimal_places > 0:
            return f"{value:.{min_decimal_places}f}"
        return str(int(value))

    text = format(value, "f")
    integer_part, _, frac_part = text.partition(".")
    frac_part = frac_part.rstrip("0")
    if len(frac_part) < min_decimal_places:
        frac_part = frac_part + ("0" * (min_decimal_places - len(frac_part)))
    return f"{integer_part}.{frac_part}"


def infer_decimal_places_from_raw(value: Any) -> int:
    value = normalize_empty(value)
    if value is None:
        return 0
    text = str(value)
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1].rstrip())


def result_min_decimal_places(expr_tokens: Iterable[str], entities: Dict[str, Dict[str, Any]]) -> int:
    max_places = 0
    for token in expr_tokens:
        if token in entities:
            max_places = max(max_places, infer_decimal_places_from_raw(entities[token].get("value")))
    return max_places


# -----------------------------------------------------------------------------
# Safe expression evaluator
# -----------------------------------------------------------------------------

ALLOWED_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def expr_tokens(expr: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def validate_entity_name(name: str) -> None:
    if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise ExecutorError(f"Tên entity không hợp lệ: {name!r}")


def eval_ast_node(node: ast.AST, values: Dict[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return eval_ast_node(node.body, values)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ExecutorError("Boolean không được phép trong biểu thức.")
        if isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        raise ExecutorError(f"Hằng không hợp lệ trong biểu thức: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ExecutorError(f"Biến {node.id!r} chưa có value để tính toán.")
        return values[node.id]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise ExecutorError(f"Toán tử không được hỗ trợ: {op_type.__name__}")
        left = eval_ast_node(node.left, values)
        right = eval_ast_node(node.right, values)
        if op_type is ast.Div and right == 0:
            raise ExecutorError("Không thể chia cho 0.")
        return ALLOWED_BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ExecutorError(f"Unary operator không được hỗ trợ: {op_type.__name__}")
        operand = eval_ast_node(node.operand, values)
        return ALLOWED_UNARY_OPS[op_type](operand)

    raise ExecutorError(f"Biểu thức chứa thành phần không an toàn: {type(node).__name__}")


def safe_eval_expr(expr: str, values: Dict[str, Decimal]) -> Decimal:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExecutorError(f"Biểu thức không hợp lệ: {expr!r}") from exc

    return normalize_decimal(eval_ast_node(tree, values))


def replace_names_with_values(expr: str, entities: Dict[str, Dict[str, Any]]) -> str:
    tokens = expr_tokens(expr)
    replacements: Dict[str, str] = {}
    min_places = result_min_decimal_places(tokens, entities)

    for token in sorted(set(tokens), key=len, reverse=True):
        if token not in entities:
            raise ExecutorError(f"Expr dùng entity chưa tồn tại: {token}")
        value = to_decimal(entities[token].get("value"), name=f"{token}.value")
        places = max(infer_decimal_places_from_raw(entities[token].get("value")), min_places)
        replacements[token] = format_decimal(value, min_decimal_places=places)

    def repl(match: re.Match[str]) -> str:
        name = match.group(0)
        return replacements.get(name, name)

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, expr)


def values_for_expr(expr: str, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Decimal]:
    values: Dict[str, Decimal] = {}
    for token in sorted(set(expr_tokens(expr))):
        if token not in entities:
            raise ExecutorError(f"Expr dùng entity chưa tồn tại: {token}")
        values[token] = to_decimal(entities[token].get("value"), name=f"{token}.value")
    return values


# -----------------------------------------------------------------------------
# Validation / normalization
# -----------------------------------------------------------------------------

def normalize_plan_step_fields(step_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise ExecutorError(f"{step_name} phải là dictionary.")

    # Hỗ trợ alias do các prompt trước có dùng nhầm tên.
    if "grand_result_unit" in step and "result_grand_unit" not in step:
        step["result_grand_unit"] = step.pop("grand_result_unit")

    required = {"expr", "result", "result_unit", "result_grand_unit"}
    optional = {"reported_expr"}
    allowed = required | optional

    missing = required - set(step.keys())
    extra = set(step.keys()) - allowed
    if missing:
        raise ExecutorError(f"{step_name} thiếu trường: {sorted(missing)}")
    if extra:
        raise ExecutorError(f"{step_name} có trường thừa: {sorted(extra)}")

    expr = step["expr"]
    result = step["result"]
    if not isinstance(expr, str) or not expr.strip():
        raise ExecutorError(f"{step_name}.expr phải là string không rỗng.")
    validate_entity_name(result)

    normalized = {
        "expr": expr.strip(),
        "result": result,
        "result_unit": normalize_empty(step.get("result_unit")),
        "result_grand_unit": normalize_empty(step.get("result_grand_unit")),
    }
    if "reported_expr" in step:
        normalized["reported_expr"] = step["reported_expr"]
    return normalized


def validate_and_normalize_plan(raw_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not raw_plan:
        raise ExecutorError("Output/Plan.yaml đang rỗng.")

    expected = [f"step{i}" for i in range(1, len(raw_plan) + 1)]
    actual = list(raw_plan.keys())
    if actual != expected:
        raise ExecutorError(f"Plan step phải liên tục {expected}, hiện là {actual}.")

    return {step_name: normalize_plan_step_fields(step_name, raw_plan[step_name]) for step_name in expected}


def validate_and_normalize_entities(raw_entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not raw_entities:
        raise ExecutorError("Output/PlanEntities.yaml đang rỗng.")

    normalized: Dict[str, Dict[str, Any]] = {}
    base_fields = {"value", "unit", "location", "grand_unit"}
    optional_fields = {"expr", "formalized_expr"}

    for name, entity in raw_entities.items():
        validate_entity_name(name)
        if not isinstance(entity, dict):
            raise ExecutorError(f"Entity {name} phải là dictionary.")

        missing = base_fields - set(entity.keys())
        if missing:
            raise ExecutorError(f"Entity {name} thiếu trường: {sorted(missing)}")

        extra = set(entity.keys()) - base_fields - optional_fields
        if extra:
            raise ExecutorError(f"Entity {name} có trường thừa: {sorted(extra)}")

        normalized[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
            "expr": normalize_empty(entity.get("expr")),
            "formalized_expr": normalize_empty(entity.get("formalized_expr")),
        }

    return normalized


# -----------------------------------------------------------------------------
# Formalized expression builder
# -----------------------------------------------------------------------------

def parenthesize_if_needed(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr
    if expr.startswith("(") and expr.endswith(")"):
        return expr
    if re.search(r"\s[+\-]\s", expr):
        return f"({expr})"
    return expr


def formalize_expr(expr: str, formalized_by_entity: Dict[str, Optional[str]]) -> str:
    """
    Chuẩn hóa expr về phép tính chỉ dựa trên các input entity.

    Ví dụ:
    - daily_cost = morning_coffee_price + afternoon_coffee_price
    - total_cost = daily_cost * days
    => formalized total_cost = (morning_coffee_price + afternoon_coffee_price) * days
    """
    result = expr
    for token in sorted(set(expr_tokens(expr)), key=len, reverse=True):
        replacement = formalized_by_entity.get(token)
        if replacement:
            replacement = parenthesize_if_needed(replacement)
            result = re.sub(rf"\b{re.escape(token)}\b", replacement, result)
    return result


# -----------------------------------------------------------------------------
# Execution core
# -----------------------------------------------------------------------------

def execute_plan_once(
    plan: Dict[str, Dict[str, Any]],
    entities: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Thực thi Plan.yaml một lần và cập nhật PlanEntities.yaml.
    """
    updated_plan = deepcopy(plan)
    updated_entities = deepcopy(entities)

    result_expr_by_entity: Dict[str, str] = {}
    formalized_by_entity: Dict[str, Optional[str]] = {}

    # Input entity có expr/formalized_expr rỗng.
    for name, entity in updated_entities.items():
        if entity.get("location") == "input":
            entity["expr"] = None
            entity["formalized_expr"] = None
            formalized_by_entity[name] = None

    for step_name, step in updated_plan.items():
        expr = step["expr"]
        result_name = step["result"]
        validate_entity_name(result_name)

        # Kiểm tra toàn bộ biến trong expr đã có value.
        values = values_for_expr(expr, updated_entities)
        result_value = safe_eval_expr(expr, values)

        numeric_expr = replace_names_with_values(expr, updated_entities)
        min_places = result_min_decimal_places(expr_tokens(expr), updated_entities)
        formatted_result = format_decimal(result_value, min_decimal_places=min_places)
        step["reported_expr"] = f"{numeric_expr} = {formatted_result}"

        if result_name not in updated_entities:
            updated_entities[result_name] = {
                "value": None,
                "unit": normalize_empty(step.get("result_unit")),
                "location": step_name,
                "grand_unit": normalize_empty(step.get("result_grand_unit")),
                "expr": None,
                "formalized_expr": None,
            }

        updated_entities[result_name]["value"] = decimal_to_yaml_number(result_value)
        updated_entities[result_name]["unit"] = normalize_empty(step.get("result_unit"))
        updated_entities[result_name]["grand_unit"] = normalize_empty(step.get("result_grand_unit"))

        # Nếu result đang là target thì giữ location target theo one-shot của user.
        # Nếu entity chưa phải target thì location là step tính ra nó.
        old_location = updated_entities[result_name].get("location")
        if old_location != "target":
            updated_entities[result_name]["location"] = step_name

        updated_entities[result_name]["expr"] = expr
        result_expr_by_entity[result_name] = expr

        f_expr = formalize_expr(expr, formalized_by_entity)
        updated_entities[result_name]["formalized_expr"] = f_expr
        formalized_by_entity[result_name] = f_expr

    # Với entity không xuất hiện trong step result và không phải input, cố gắng bổ sung expr/formalized_expr nếu có location step.
    for name, entity in updated_entities.items():
        if entity.get("location") == "input":
            entity["expr"] = None
            entity["formalized_expr"] = None
            continue

        if entity.get("expr") is None and name in result_expr_by_entity:
            entity["expr"] = result_expr_by_entity[name]
        if entity.get("formalized_expr") is None and name in formalized_by_entity:
            entity["formalized_expr"] = formalized_by_entity[name]

    return updated_plan, updated_entities


# -----------------------------------------------------------------------------
# InsideChecker integration
# -----------------------------------------------------------------------------

def run_inside_checker() -> Tuple[bool, str]:
    """
    Chạy Verify/InsideChecker.py.

    Return:
    - (True, output) nếu không có lỗi trong Output/Error.yaml
    - (False, output) nếu có lỗi hoặc checker chạy lỗi
    """
    if not INSIDE_CHECKER_PATH.exists():
        raise ExecutorError(f"Không tìm thấy file checker: {INSIDE_CHECKER_PATH}")

    # Xóa Error.yaml cũ để tránh đọc nhầm lỗi từ lần trước.
    if ERROR_PATH.exists():
        ERROR_PATH.unlink()

    completed = subprocess.run(
        [sys.executable, str(INSIDE_CHECKER_PATH)],
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )

    output = ""
    if completed.stdout:
        output += completed.stdout
    if completed.stderr:
        output += completed.stderr

    if completed.returncode != 0:
        return False, output.strip()

    if not ERROR_PATH.exists():
        return True, output.strip()

    error_text = ERROR_PATH.read_text(encoding="utf-8").strip()
    if not error_text:
        return True, output.strip()

    try:
        error_data = yaml.safe_load(error_text)
    except yaml.YAMLError:
        return False, output.strip()

    if error_data in (None, {}, [], ""):
        return True, output.strip()

    return False, output.strip()


# -----------------------------------------------------------------------------
# LLM repair
# -----------------------------------------------------------------------------

def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:yaml|yml|json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def build_repair_system_prompt() -> str:
    return """
Bạn là một hệ thống sửa file YAML cho pipeline giải toán.

Bạn nhận vào:
- Input/Problem.txt
- Output/PlanEntities.yaml
- Output/Plan.yaml
- Output/Error.yaml

Nhiệm vụ:
- Dựa vào lỗi trong Error.yaml để sửa Plan.yaml và/hoặc PlanEntities.yaml.
- Chỉ sửa những gì cần thiết.
- Giữ đúng schema.
- Không giải thích.
- Không dùng Markdown.

Schema Plan.yaml:
step1:
  expr: entity_a + entity_b
  result: result_entity
  result_unit: dollars
  result_grand_unit: dollars
  reported_expr: 1 + 2 = 3   # optional, Executor có thể ghi lại

Schema PlanEntities.yaml:
entity_name:
  value: 123                 # hoặc null nếu chưa tính
  unit: dollars              # hoặc null
  location: input|target|step1|step2|...
  grand_unit: dollars        # hoặc null
  expr: entity_a + entity_b  # input thì null
  formalized_expr: ...       # input thì null

Quy tắc quan trọng:
- Plan step phải liên tục: step1, step2, ...
- Expr chỉ dùng entity đã có trong PlanEntities hoặc result từ step trước.
- Bước cuối phải tạo ra target nếu target đang tồn tại trong PlanEntities.
- Không tự thêm dữ kiện input mới không có trong đề.
- Có thể sửa result_unit/result_grand_unit nếu sai.
- Nếu đổi tên result trong Plan.yaml thì phải đồng bộ PlanEntities.yaml.
- Với input entity, expr và formalized_expr phải là null.

Output bắt buộc là một YAML object có đúng 2 key:
Plan.yaml:
  ... nội dung Plan.yaml đã sửa ...
PlanEntities.yaml:
  ... nội dung PlanEntities.yaml đã sửa ...
""".strip()


def build_repair_user_prompt() -> str:
    problem = read_text(INPUT_PATH, required=True).strip()
    plan_entities = read_text(PLAN_ENTITIES_PATH, required=True).strip()
    plan = read_text(PLAN_PATH, required=True).strip()
    error = read_text(ERROR_PATH, required=False).strip()

    return f"""
Hãy sửa các file YAML dựa trên lỗi.

Input/Problem.txt:
{problem}

Output/PlanEntities.yaml:
{plan_entities}

Output/Plan.yaml:
{plan}

Output/Error.yaml:
{error}
""".strip()


def call_openrouter_for_repair() -> Dict[str, Any]:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ExecutorError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

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
            {"role": "system", "content": build_repair_system_prompt()},
            {"role": "user", "content": build_repair_user_prompt()},
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise ExecutorError(f"Không gọi được OpenRouter repair: {exc}") from exc

    if response.status_code >= 400:
        raise ExecutorError(f"OpenRouter repair trả lỗi {response.status_code}: {response.text[:1000]}")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ExecutorError(f"Response OpenRouter repair không đúng định dạng: {response.text[:1000]}") from exc

    content = strip_markdown_fence(content)
    try:
        repaired = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ExecutorError(f"LLM repair trả YAML không hợp lệ: {exc}") from exc

    if not isinstance(repaired, dict):
        raise ExecutorError("LLM repair output phải là dictionary.")

    if "Plan.yaml" not in repaired or "PlanEntities.yaml" not in repaired:
        raise ExecutorError("LLM repair output phải có đúng key Plan.yaml và PlanEntities.yaml.")

    if not isinstance(repaired["Plan.yaml"], dict):
        raise ExecutorError("Plan.yaml sau repair phải là dictionary.")
    if not isinstance(repaired["PlanEntities.yaml"], dict):
        raise ExecutorError("PlanEntities.yaml sau repair phải là dictionary.")

    return repaired


def apply_repair(repaired: Dict[str, Any]) -> None:
    write_yaml_file(PLAN_PATH, repaired["Plan.yaml"])
    write_yaml_file(PLAN_ENTITIES_PATH, repaired["PlanEntities.yaml"])


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def load_and_execute_current_files() -> None:
    raw_plan = read_yaml_file(PLAN_PATH, required=True)
    raw_entities = read_yaml_file(PLAN_ENTITIES_PATH, required=True)
    raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=False)

    target_entities = [
        (name, fields)
        for name, fields in raw_problem_entities.items()
        if isinstance(fields, dict) and fields.get("location") == "target"
    ]
    if len(target_entities) == 1:
        target_name, target_fields = target_entities[0]
        raw_entities.setdefault(target_name, dict(target_fields))
        raw_entities[target_name]["location"] = "target"

    plan = validate_and_normalize_plan(raw_plan)
    entities = validate_and_normalize_entities(raw_entities)

    updated_plan, updated_entities = execute_plan_once(plan, entities)

    write_yaml_file(PLAN_PATH, updated_plan)
    write_yaml_file(PLAN_ENTITIES_PATH, updated_entities)


def max_repair_iterations() -> int:
    raw = os.getenv("EXECUTOR_MAX_REPAIR_ITERATIONS")
    if raw is None:
        return DEFAULT_MAX_REPAIR_ITERATIONS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ExecutorError("EXECUTOR_MAX_REPAIR_ITERATIONS phải là số nguyên.") from exc
    if value < 0:
        raise ExecutorError("EXECUTOR_MAX_REPAIR_ITERATIONS không được âm.")
    return value


def run() -> None:
    try:
        ensure_dirs()
        load_dotenv(ROOT_DIR / ".env")
        load_dotenv()

        max_iterations = max_repair_iterations()

        for iteration in range(max_iterations + 1):
            load_and_execute_current_files()

            checker_ok, checker_output = run_inside_checker()
            if checker_ok:
                write_log("Pass Executor")
                print("Pass Executor")
                return

            if iteration >= max_iterations:
                message = (
                    "InsideChecker vẫn còn lỗi sau "
                    f"{max_iterations} lần repair. Checker output: {checker_output}"
                )
                write_log("Fail Executor", message)
                print("Fail Executor")
                print(f"Reason: {message}", file=sys.stderr)
                raise SystemExit(1)

            repaired = call_openrouter_for_repair()
            apply_repair(repaired)

        raise ExecutorError("Vòng lặp Executor kết thúc bất thường.")

    except SystemExit:
        raise
    except Exception as exc:
        write_log("Fail Executor", str(exc))
        print("Fail Executor")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
