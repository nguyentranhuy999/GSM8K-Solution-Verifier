"""
Verify/CompareChecker.py

Nhiệm vụ:
- So sánh lời giải reference với lời giải học sinh sau khi đã formalize, execute và map.
- Reference mặc định là Solver/Plan:
  - Output/Plan.yaml
  - Output/PlanEntities.yaml
- Nếu chạy với --reference teacher, reference là:
  - Output/TeacherPlan.yaml
  - Output/TeacherAnswerEntities.yaml
- Luôn đọc:
  - Output/StudentPlan.yaml
  - Output/StudentAnswerEntities.yaml
- Ghi output vào:
  - Output/Diagnosis.yaml
  - Output/Wrong.yaml

Các lỗi/nhãn:
- combine step
- step separation
- extra step
- reverse steps
- all right
- wrong relationship
- different calculation

Quy tắc Wrong.yaml:
- Nếu có wrong relationship: ghi Yes
- Nếu chỉ có các nhãn khác: ghi No
- Nếu không có lỗi/nhãn: ghi No

Ghi chú:
- File này không dùng LLM.
- Giả định Mapper.py đã thêm trường map vào PlanEntities.yaml và StudentAnswerEntities.yaml.
- Với StudentAnswerEntities.yaml: entity["map"] là tên entity tương ứng trong PlanEntities.yaml.
- Với PlanEntities.yaml: entity["map"] là tên entity tương ứng trong StudentAnswerEntities.yaml.
"""

from __future__ import annotations

import argparse
import ast
import operator
import random
import re
import sys
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml


getcontext().prec = 28

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Output"

PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
TEACHER_PLAN_PATH = OUTPUT_DIR / "TeacherPlan.yaml"
TEACHER_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "TeacherAnswerEntities.yaml"
STUDENT_PLAN_PATH = OUTPUT_DIR / "StudentPlan.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"

DIAGNOSIS_PATH = OUTPUT_DIR / "Diagnosis.yaml"
WRONG_PATH = OUTPUT_DIR / "Wrong.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"


class CompareCheckerError(Exception):
    """Lỗi riêng cho CompareChecker."""


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise CompareCheckerError(f"Không tìm thấy file: {path}")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CompareCheckerError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CompareCheckerError(f"File YAML phải là dictionary: {path}")
    return data


def write_yaml_file(path: Path, data: Any) -> None:
    ensure_dirs()
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def read_yaml_any(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise CompareCheckerError(f"Không tìm thấy file: {path}")
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CompareCheckerError(f"File YAML không hợp lệ: {path} - {exc}") from exc


def normalize_diagnosis_item(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        label = item.strip()
        step = None
        entity = None
    elif isinstance(item, dict):
        label = str(item.get("diagnosis", "")).strip()
        step = item.get("step")
        entity = item.get("entity")
    else:
        return None

    if not label:
        return None

    return {
        "diagnosis": label,
        "step": step if step not in {"", "null", "None"} else None,
        "entity": entity if entity not in {"", "null", "None"} else None,
    }


def read_diagnosis_file() -> List[Dict[str, Any]]:
    raw_data = read_yaml_any(DIAGNOSIS_PATH, required=False)
    if isinstance(raw_data, dict):
        raw_items = raw_data.get("diagnosis", [])
    elif isinstance(raw_data, list):
        raw_items = raw_data
    else:
        raw_items = []

    items: List[Dict[str, Any]] = []
    for item in raw_items:
        normalized = normalize_diagnosis_item(item)
        if normalized is not None:
            items.append(normalized)
    return items


def merge_diagnosis_items(new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for item in read_diagnosis_file() + new_items:
        normalized = normalize_diagnosis_item(item)
        if normalized is None:
            continue
        key = (
            normalized.get("diagnosis"),
            normalized.get("step"),
            normalized.get("entity"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)

    has_real_error = any(item.get("diagnosis") != "all right" for item in merged)
    if has_real_error:
        merged = [item for item in merged if item.get("diagnosis") != "all right"]

    return merged


def append_diagnosis_file(diagnosis: List[Dict[str, Any]]) -> None:
    write_yaml_file(DIAGNOSIS_PATH, merge_diagnosis_items(diagnosis))


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    log_data["CompareChecker"] = status
    if message:
        log_data["CompareChecker_message"] = message
    elif "CompareChecker_message" in log_data:
        del log_data["CompareChecker_message"]
    write_yaml_file(LOG_PATH, log_data)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def add_diagnosis(items: List[Dict[str, Any]], diagnosis: str, step: Optional[str] = None, entity: Optional[str] = None) -> None:
    item = {
        "diagnosis": diagnosis,
        "step": step,
        "entity": entity,
    }
    if item not in items:
        items.append(item)


CORE_DIAGNOSIS_LABELS = {
    "misreading",
    "wrong calculation",
    "logic error",
    "wrong target",
    "wrong relationship",
    "different calculation",
    "do not convert units",
    "unit missing",
    "wrong units conversion",
    "wrong unit conversion",
    "double count",
    "missing step",
    "only final answer",
}


COMPARE_CORE_LABELS = {"wrong relationship", "different calculation", "missing step"}


def has_core_diagnosis(items: List[Dict[str, Any]]) -> bool:
    return any(item.get("diagnosis") in CORE_DIAGNOSIS_LABELS for item in items)


def has_compare_core_diagnosis(items: List[Dict[str, Any]]) -> bool:
    return any(item.get("diagnosis") in COMPARE_CORE_LABELS for item in items)


def reclassify_answer_literal_misreading_as_missing_step(
    items: List[Dict[str, Any]],
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Sau Mapper, answer-literal có thể map sang intermediate của reference.

    Nếu học sinh dùng trực tiếp số intermediate mà không trình bày bước tạo ra
    nó, lỗi đúng là missing step. Trước Mapper, InsideChecker chỉ thấy đó là số
    không nằm trực tiếp trong input nên có thể ghi misreading; normalize lại ở
    đây để tránh biến thiếu bước thành đọc sai dữ kiện.
    """
    changed = False
    normalized_items: List[Dict[str, Any]] = []

    for item in items:
        current = dict(item)
        entity_name = normalize_empty(current.get("entity"))
        student_entity = student_entities.get(str(entity_name)) if entity_name else None
        mapped_name = normalize_empty(student_entity.get("map")) if student_entity else None
        plan_entity = plan_entities.get(str(mapped_name)) if mapped_name else None

        if (
            current.get("diagnosis") == "misreading"
            and entity_name
            and student_entity
            and plan_entity
            and is_answer_literal_entity(str(entity_name), student_entity)
            and re.fullmatch(r"step\d+", str(normalize_empty(plan_entity.get("location")) or ""))
            and decimal_equal(student_entity.get("value"), plan_entity.get("value"))
        ):
            current["diagnosis"] = "missing step"
            changed = True

        normalized_items.append(current)

    return normalized_items, changed


def plan_input_value_exists(
    value: Any,
    plan_entities: Dict[str, Dict[str, Any]],
) -> bool:
    for entity in plan_entities.values():
        if entity.get("location") != "input":
            continue
        if decimal_equal(value, entity.get("value")):
            return True
    return False


def find_student_step_using_token(student_plan: Dict[str, Any], token: str) -> Optional[str]:
    for step_name in step_names(student_plan):
        step = student_plan.get(step_name)
        if not isinstance(step, dict):
            continue
        expr = normalize_empty(step.get("expr"))
        if token in expr_tokens(expr):
            return step_name
    return None


def check_unmapped_answer_literal_misreading(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_plan: Dict[str, Any],
    diagnosis: List[Dict[str, Any]],
) -> None:
    """
    Gắn misreading sau Mapper, khi đã biết literal đó không phải input/reference
    intermediate hợp lệ và kết quả cuối thật sự khác reference.

    Rule này thay thế cách cũ trong InsideChecker vốn quá sớm: trước Mapper, một
    số như 1800 có thể trông như số bịa, nhưng sau Mapper mới biết nó là
    intermediate bị thiếu bước.
    """
    plan_target = first_target_entity(plan_entities)
    student_target = first_target_entity(student_entities)
    if not plan_target or not student_target:
        return

    if decimal_equal(plan_entities[plan_target].get("value"), student_entities[student_target].get("value")):
        return

    student_target_expr = (
        normalize_empty(student_entities[student_target].get("formalized_expr"))
        or normalize_empty(student_entities[student_target].get("expr"))
    )
    if not student_target_expr:
        return

    for token in expr_tokens(str(student_target_expr)):
        entity = student_entities.get(token)
        if not is_answer_literal_entity(token, entity):
            continue
        if normalize_empty(entity.get("map")):
            continue
        if plan_input_value_exists(entity.get("value"), plan_entities):
            continue
        add_diagnosis(
            diagnosis,
            "misreading",
            step=find_student_step_using_token(student_plan, token),
            entity=token,
        )


def to_decimal(value: Any) -> Optional[Decimal]:
    value = normalize_empty(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def decimal_equal(a: Any, b: Any, tolerance: Decimal = Decimal("0.000001")) -> bool:
    da = to_decimal(a)
    db = to_decimal(b)
    if da is None or db is None:
        return normalize_empty(a) == normalize_empty(b)
    return abs(da - db) <= tolerance


def decimal_expr_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def expr_tokens(expr: Optional[str]) -> List[str]:
    if not expr:
        return []
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", str(expr))


def step_names(plan: Dict[str, Any]) -> List[str]:
    def key_fn(name: str) -> int:
        match = re.fullmatch(r"step(\d+)", str(name))
        return int(match.group(1)) if match else 10**9

    return sorted([key for key in plan.keys() if re.fullmatch(r"step\d+", str(key))], key=key_fn)


def count_steps(plan: Dict[str, Any]) -> int:
    return len(step_names(plan))


def step_result_order(plan: Dict[str, Any]) -> List[str]:
    results: List[str] = []
    for step_name in step_names(plan):
        step = plan.get(step_name)
        if isinstance(step, dict) and step.get("result"):
            results.append(str(step["result"]))
    return results


def mapped_student_step_result_order(
    student_plan: Dict[str, Any],
    student_entities: Dict[str, Dict[str, Any]],
) -> List[str]:
    results: List[str] = []
    for step_name in step_names(student_plan):
        step = student_plan.get(step_name)
        if not isinstance(step, dict) or not step.get("result"):
            continue
        result = str(step["result"])
        mapped = normalize_empty(student_entities.get(result, {}).get("map"))
        results.append(str(mapped or result))
    return results


def step_order_matches(
    plan: Dict[str, Any],
    student_plan: Dict[str, Any],
    student_entities: Dict[str, Dict[str, Any]],
) -> bool:
    return step_result_order(plan) == mapped_student_step_result_order(student_plan, student_entities)


def target_entities(entities: Dict[str, Dict[str, Any]]) -> List[str]:
    return [name for name, entity in entities.items() if entity.get("location") == "target"]


def first_target_entity(entities: Dict[str, Dict[str, Any]]) -> Optional[str]:
    targets = target_entities(entities)
    return targets[0] if targets else None


def normalize_unit(unit: Any) -> Optional[str]:
    unit = normalize_empty(unit)
    if unit is None:
        return None
    return str(unit).strip().lower().replace(" ", "_")


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def normalize_plan(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(raw_plan)
    for sname in step_names(plan):
        step = plan[sname]
        if not isinstance(step, dict):
            continue
        if "grand_result_unit" in step and "result_grand_unit" not in step:
            step["result_grand_unit"] = step.pop("grand_result_unit")
    return plan


def normalize_entities(raw_entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entities: Dict[str, Dict[str, Any]] = {}
    for name, entity in raw_entities.items():
        if not isinstance(entity, dict):
            continue
        entities[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
            "expr": normalize_empty(entity.get("expr")),
            "formalized_expr": normalize_empty(entity.get("formalized_expr")),
            "source_type": normalize_empty(entity.get("source_type")),
            "map": normalize_empty(entity.get("map")),
        }
    return entities


# -----------------------------------------------------------------------------
# Unit conversion detection
# -----------------------------------------------------------------------------

CONVERTIBLE_UNIT_GROUPS = [
    # length
    {"mm", "millimeter", "millimeters", "cm", "centimeter", "centimeters", "m", "meter", "meters", "km", "kilometer", "kilometers", "inch", "inches", "ft", "foot", "feet", "yard", "yards", "mile", "miles"},
    # mass
    {"mg", "milligram", "milligrams", "g", "gram", "grams", "kg", "kilogram", "kilograms", "ton", "tons", "lb", "lbs", "pound", "pounds", "ounce", "ounces"},
    # volume
    {"ml", "milliliter", "milliliters", "l", "liter", "liters", "gallon", "gallons", "quart", "quarts", "pint", "pints", "cup", "cups"},
    # area
    {"mm2", "cm2", "m2", "km2", "square_meters", "square_meter", "square_centimeters", "square_centimeter", "square_feet", "square_foot", "hectare", "hectares", "acre", "acres"},
    # time
    {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"},
    # dozen/count package
    {"dozen", "dozens", "item", "items", "piece", "pieces"},
]


def is_convertible_unit(unit: Any) -> bool:
    unit = normalize_unit(unit)
    if not unit:
        return False
    return any(unit in group for group in CONVERTIBLE_UNIT_GROUPS)


def is_convertible_metadata(entity: Dict[str, Any]) -> bool:
    return is_convertible_unit(entity.get("unit")) or is_convertible_unit(entity.get("grand_unit"))


# -----------------------------------------------------------------------------
# Safe expression evaluation with random substitution
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


def eval_ast_node(node: ast.AST, values: Dict[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return eval_ast_node(node.body, values)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise CompareCheckerError("Boolean không được phép trong biểu thức.")
        if isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        raise CompareCheckerError(f"Hằng không hợp lệ: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id not in values:
            raise CompareCheckerError(f"Thiếu value cho biến {node.id!r}.")
        return values[node.id]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise CompareCheckerError(f"Toán tử không được hỗ trợ: {op_type.__name__}")
        left = eval_ast_node(node.left, values)
        right = eval_ast_node(node.right, values)
        if op_type is ast.Div and right == 0:
            raise CompareCheckerError("Chia cho 0.")
        return Decimal(str(ALLOWED_BIN_OPS[op_type](left, right)))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise CompareCheckerError(f"Unary operator không được hỗ trợ: {op_type.__name__}")
        return Decimal(str(ALLOWED_UNARY_OPS[op_type](eval_ast_node(node.operand, values))))

    raise CompareCheckerError(f"Biểu thức chứa thành phần không an toàn: {type(node).__name__}")


def safe_eval_expr(expr: str, values: Dict[str, Decimal]) -> Decimal:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CompareCheckerError(f"Biểu thức không hợp lệ: {expr!r}") from exc
    return eval_ast_node(tree, values)


def parenthesize_expr(expr: str) -> str:
    expr = str(expr).strip()
    if expr.startswith("(") and expr.endswith(")"):
        return expr
    return f"({expr})"


def replace_student_tokens_with_plan_tokens(
    expr: str,
    student_entities: Dict[str, Dict[str, Any]],
    plan_entities: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Đưa formalized_expr của student về namespace của plan bằng trường map."""
    if not expr:
        return expr

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = student_entities.get(token, {}).get("map")
        if mapped:
            if plan_entities:
                mapped_formalized = normalize_empty(plan_entities.get(mapped, {}).get("formalized_expr"))
                if mapped_formalized:
                    return parenthesize_expr(str(mapped_formalized))
            return mapped
        literal_value = answer_literal_expr_value(token, student_entities)
        if literal_value is not None:
            return literal_value
        return token

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, expr)


def is_answer_literal_entity(name: str, entity: Optional[Dict[str, Any]]) -> bool:
    if not entity:
        return False
    if entity.get("source_type") == "answer_literal":
        return True
    return name.startswith(("student_answer_number_", "teacher_answer_number_"))


def answer_literal_expr_value(token: str, entities: Dict[str, Dict[str, Any]]) -> Optional[str]:
    entity = entities.get(token)
    if not is_answer_literal_entity(token, entity):
        return None
    value = to_decimal(entity.get("value") if entity else None)
    if value is None:
        return None
    return decimal_expr_text(value)


def replace_answer_literals_with_values(expr: Optional[str], entities: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not expr:
        return expr

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        literal_value = answer_literal_expr_value(token, entities)
        return literal_value if literal_value is not None else token

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, str(expr))


def expression_uses_answer_literal(expr: Optional[str], entities: Dict[str, Dict[str, Any]]) -> bool:
    if not expr:
        return False
    return any(is_answer_literal_entity(token, entities.get(token)) for token in expr_tokens(expr))


def comparable_plan_expr(expr: Optional[str], plan_entities: Dict[str, Dict[str, Any]]) -> Optional[str]:
    return replace_answer_literals_with_values(expr, plan_entities)


def comparable_student_expr(
    expr: Optional[str],
    student_entities: Dict[str, Dict[str, Any]],
    plan_entities: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    if not expr:
        return expr
    mapped = replace_student_tokens_with_plan_tokens(str(expr), student_entities, plan_entities)
    return replace_answer_literals_with_values(mapped, plan_entities)


def expression_token_set(expr: Optional[str]) -> Set[str]:
    return set(expr_tokens(expr))


def extra_step_candidates_with_irrelevant_inputs(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Tìm step dư kiểu học sinh tính thêm một đại lượng rồi triệt tiêu nó về sau.

    Dấu hiệu tổng quát: entity trung gian của student không map được sang reference,
    và formalized_expr của nó dùng một input không xuất hiện trong công thức target
    reference. Ví dụ tính tổng 5 ngày rồi chia lại cho 5 trước khi nhân 7 ngày.
    """
    plan_target = first_target_entity(plan_entities)
    if not plan_target:
        return []

    plan_target_expr = comparable_plan_expr(plan_entities[plan_target].get("formalized_expr"), plan_entities)
    plan_target_tokens = expression_token_set(plan_target_expr)
    if not plan_target_tokens:
        return []

    candidates: List[str] = []
    for student_name, student_entity in student_entities.items():
        location = normalize_empty(student_entity.get("location"))
        if not isinstance(location, str) or not re.fullmatch(r"step\d+", location):
            continue
        if normalize_empty(student_entity.get("map")):
            continue
        if is_answer_literal_entity(student_name, student_entity):
            continue
        if has_equivalent_unmapped_plan_step_entity(student_entity, plan_entities, student_entities):
            continue

        expr = normalize_empty(student_entity.get("formalized_expr")) or normalize_empty(student_entity.get("expr"))
        comparable_expr = comparable_student_expr(expr, student_entities, plan_entities)
        candidate_tokens = expression_token_set(comparable_expr)
        irrelevant_inputs = [
            token
            for token in candidate_tokens - plan_target_tokens
            if token in plan_entities and plan_entities[token].get("location") == "input"
        ]
        if irrelevant_inputs:
            candidates.append(student_name)

    return candidates


def same_unit_and_value(student_entity: Dict[str, Any], plan_entity: Dict[str, Any]) -> bool:
    if normalize_empty(student_entity.get("unit")) != normalize_empty(plan_entity.get("unit")):
        return False
    if normalize_empty(student_entity.get("grand_unit")) != normalize_empty(plan_entity.get("grand_unit")):
        return False
    return decimal_equal(student_entity.get("value"), plan_entity.get("value"))


def has_equivalent_unmapped_plan_step_entity(
    student_entity: Dict[str, Any],
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> bool:
    student_expr = normalize_empty(student_entity.get("formalized_expr")) or normalize_empty(student_entity.get("expr"))
    comparable_student = comparable_student_expr(student_expr, student_entities, plan_entities)
    if not comparable_student:
        return False

    for plan_name, plan_entity in plan_entities.items():
        location = normalize_empty(plan_entity.get("location"))
        if not isinstance(location, str) or not re.fullmatch(r"step\d+", location):
            continue
        if normalize_empty(plan_entity.get("map")):
            continue
        if is_answer_literal_entity(plan_name, plan_entity):
            continue
        if not same_unit_and_value(student_entity, plan_entity):
            continue

        plan_expr = normalize_empty(plan_entity.get("formalized_expr")) or normalize_empty(plan_entity.get("expr"))
        comparable_plan = comparable_plan_expr(plan_expr, plan_entities)
        if equivalent_by_random_substitution(
            comparable_plan,
            comparable_student,
            trials=3,
            fixed_values=fixed_context_values(plan_entities),
        ):
            return True

    return False


def unmapped_student_step_entities(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> List[str]:
    candidates: List[str] = []
    for student_name, student_entity in student_entities.items():
        location = normalize_empty(student_entity.get("location"))
        if not isinstance(location, str) or not re.fullmatch(r"step\d+", location):
            continue
        if normalize_empty(student_entity.get("map")):
            continue
        if is_answer_literal_entity(student_name, student_entity):
            continue
        if has_equivalent_unmapped_plan_step_entity(student_entity, plan_entities, student_entities):
            continue
        candidates.append(student_name)
    return candidates


def check_missing_step_from_answer_literals(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_plan: Dict[str, Any],
    diagnosis: List[Dict[str, Any]],
) -> None:
    """
    Nếu học sinh dùng trực tiếp một số trung gian và Mapper map số đó tới step
    của reference, nhưng không có step học sinh nào tạo intermediate đó, thì đó
    là missing step.

    Ví dụ student viết `24000 * 3 = 72000`; teacher có step tạo
    `beats_in_two_hours = 24000`. Literal 24000 không phải cách tính khác, mà
    là kết quả trung gian chưa được chứng minh trong lời giải học sinh.
    """
    student_step_maps = {
        normalize_empty(entity.get("map"))
        for entity in student_entities.values()
        if isinstance(normalize_empty(entity.get("location")), str)
        and re.fullmatch(r"step\d+", str(normalize_empty(entity.get("location"))))
        and not is_answer_literal_entity("", entity)
    }

    for student_name, student_entity in student_entities.items():
        if not is_answer_literal_entity(student_name, student_entity):
            continue
        mapped_plan_name = normalize_empty(student_entity.get("map"))
        if not mapped_plan_name or mapped_plan_name not in plan_entities:
            continue

        plan_entity = plan_entities[mapped_plan_name]
        plan_location = normalize_empty(plan_entity.get("location"))
        if not isinstance(plan_location, str) or not re.fullmatch(r"step\d+", plan_location):
            continue
        if normalize_empty(plan_entity.get("expr")) is None:
            continue
        if mapped_plan_name in student_step_maps:
            continue

        step = None
        for step_name in step_names(student_plan):
            step_data = student_plan[step_name]
            if not isinstance(step_data, dict):
                continue
            if student_name in expr_tokens(normalize_empty(step_data.get("expr"))):
                step = step_name
                break
        add_diagnosis(diagnosis, "missing step", step=step, entity=student_name)


def simplify_identity_multiplier(expr: str) -> str:
    """
    identity_multiplier là context entity do ProblemFormalizer thêm với value 1.
    Nó không làm thay đổi quan hệ toán học, nên phải được rút gọn trước khi
    random substitution. Nếu random hóa entity này như biến thường, các biểu
    thức tương đương như `x * fraction` và `x / 1 * fraction` sẽ bị coi là khác.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr

    class IdentityMultiplierSimplifier(ast.NodeTransformer):
        @staticmethod
        def is_identity(node: ast.AST) -> bool:
            return isinstance(node, ast.Name) and node.id == "identity_multiplier"

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
            node = self.generic_visit(node)
            if not isinstance(node, ast.BinOp):
                return node
            if isinstance(node.op, ast.Mult):
                if self.is_identity(node.left):
                    return node.right
                if self.is_identity(node.right):
                    return node.left
            if isinstance(node.op, ast.Div) and self.is_identity(node.right):
                return node.left
            return node

    simplified = IdentityMultiplierSimplifier().visit(tree)
    ast.fix_missing_locations(simplified)
    return ast.unparse(simplified)


FIXED_CONTEXT_TOKENS = {"identity_multiplier", "host_count", "percentage_scale", "split_count"}
FIXED_INPUT_HINTS = {
    "double",
    "fraction",
    "half",
    "multiplier",
    "per",
    "percent",
    "percentage",
    "quarter",
    "rate",
    "ratio",
    "scale",
    "third",
    "times",
}


def decimal_is_integer(value: Decimal) -> bool:
    return value == value.to_integral_value()


def is_fixed_input_constant(name: str, entity: Dict[str, Any]) -> bool:
    """
    Một số input trong đề là hệ số cố định, không phải đại lượng cần giữ symbolic.

    Ví dụ cùng một quan hệ có thể được viết là `x / 3` hoặc `x * one_third`.
    Nếu random-substitute `one_third` như biến tự do, CompareChecker sẽ tưởng
    hai biểu thức khác quan hệ. Ta chỉ fix các input không có unit và có dấu hiệu
    là scale/rate/ratio/multiplier để tránh che lỗi dùng nhầm đại lượng có unit.
    """
    if normalize_empty(entity.get("location")) != "input":
        return False
    if normalize_empty(entity.get("unit")) is not None or normalize_empty(entity.get("grand_unit")) is not None:
        return False

    value = to_decimal(entity.get("value"))
    if value is None:
        return False
    if not decimal_is_integer(value):
        return True

    text = f"{name} {entity.get('source') or ''}".lower()
    return any(hint in text for hint in FIXED_INPUT_HINTS)


def fixed_context_values(entities: Dict[str, Dict[str, Any]]) -> Dict[str, Decimal]:
    values: Dict[str, Decimal] = {}
    for name in FIXED_CONTEXT_TOKENS:
        if name not in entities:
            continue
        value = to_decimal(entities[name].get("value"))
        if value is not None:
            values[name] = value
    for name, entity in entities.items():
        if name in values:
            continue
        if is_fixed_input_constant(name, entity):
            value = to_decimal(entity.get("value"))
            if value is not None:
                values[name] = value
    return values


def equivalent_by_random_substitution(
    expr_a: Optional[str],
    expr_b: Optional[str],
    *,
    trials: int = 3,
    fixed_values: Optional[Dict[str, Decimal]] = None,
) -> bool:
    if not expr_a or not expr_b:
        return False

    fixed_values = fixed_values or {}
    expr_a = simplify_identity_multiplier(expr_a)
    expr_b = simplify_identity_multiplier(expr_b)
    tokens = sorted(set(expr_tokens(expr_a)) | set(expr_tokens(expr_b)))
    if not tokens:
        try:
            return decimal_equal(safe_eval_expr(expr_a, {}), safe_eval_expr(expr_b, {}))
        except Exception:
            return False

    rng = random.Random(20260521)
    for _ in range(trials):
        values = {
            token: fixed_values.get(token, Decimal(rng.randint(2, 20)))
            for token in tokens
        }
        try:
            value_a = safe_eval_expr(expr_a, values)
            value_b = safe_eval_expr(expr_b, values)
        except Exception:
            return False
        if not decimal_equal(value_a, value_b):
            return False

    return True


def equivalent_under_entity_values(
    expr_a: Optional[str],
    expr_b: Optional[str],
    entities: Dict[str, Dict[str, Any]],
) -> bool:
    if not expr_a or not expr_b:
        return False

    tokens = sorted(set(expr_tokens(expr_a)) | set(expr_tokens(expr_b)))
    values: Dict[str, Decimal] = {}
    for token in tokens:
        entity = entities.get(token)
        if not entity:
            return False
        value = to_decimal(entity.get("value"))
        if value is None:
            return False
        values[token] = value

    try:
        return decimal_equal(safe_eval_expr(expr_a, values), safe_eval_expr(expr_b, values))
    except Exception:
        return False


def expressions_textually_same(expr_a: Any, expr_b: Any) -> bool:
    a = normalize_empty(expr_a)
    b = normalize_empty(expr_b)
    if a is None or b is None:
        return a == b
    return re.sub(r"\s+", "", str(a)) == re.sub(r"\s+", "", str(b))


# -----------------------------------------------------------------------------
# Step lookup
# -----------------------------------------------------------------------------

def find_student_step_by_result(student_plan: Dict[str, Any], entity_name: str) -> Optional[str]:
    for sname in step_names(student_plan):
        step = student_plan[sname]
        if isinstance(step, dict) and step.get("result") == entity_name:
            return sname
    return None


def find_student_step_by_expr(student_plan: Dict[str, Any], expr: Optional[str]) -> Optional[str]:
    if not expr:
        return None
    for sname in step_names(student_plan):
        step = student_plan[sname]
        if isinstance(step, dict) and expressions_textually_same(step.get("expr"), expr):
            return sname
    return None


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_step_structure_change(
    plan: Dict[str, Any],
    student_plan: Dict[str, Any],
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    diagnosis: List[Dict[str, Any]],
) -> None:
    plan_target = first_target_entity(plan_entities)
    student_target = first_target_entity(student_entities)
    if not plan_target or not student_target:
        return

    plan_expr = normalize_empty(plan_entities[plan_target].get("formalized_expr"))
    student_expr = normalize_empty(student_entities[student_target].get("formalized_expr"))
    if not plan_expr or not student_expr:
        return

    plan_expr_for_compare = replace_answer_literals_with_values(plan_expr, plan_entities)
    student_expr_mapped = replace_student_tokens_with_plan_tokens(student_expr, student_entities, plan_entities)
    student_expr_mapped = replace_answer_literals_with_values(student_expr_mapped, plan_entities)
    if not equivalent_by_random_substitution(
        plan_expr_for_compare,
        student_expr_mapped,
        trials=3,
        fixed_values=fixed_context_values(plan_entities),
    ):
        return

    plan_step_count = count_steps(plan)
    student_step_count = count_steps(student_plan)
    extra_candidates = extra_step_candidates_with_irrelevant_inputs(plan_entities, student_entities)
    if not extra_candidates and student_step_count == plan_step_count:
        extra_candidates = unmapped_student_step_entities(plan_entities, student_entities)
    if extra_candidates:
        for entity_name in extra_candidates:
            step = find_student_step_by_result(student_plan, entity_name)
            add_diagnosis(diagnosis, "extra step", step=step, entity=entity_name)
        return

    if student_step_count < plan_step_count:
        if expression_uses_answer_literal(student_expr, student_entities):
            add_diagnosis(diagnosis, "different calculation", entity=student_target)
        else:
            add_diagnosis(diagnosis, "combine step")
    elif student_step_count > plan_step_count:
        add_diagnosis(diagnosis, "step separation")
    elif student_step_count == plan_step_count:
        # Theo yêu cầu: nếu số bước bằng nhau và formalized_expr tương đương nhau thì reverse steps.
        # Để tránh ghi reverse steps khi mọi thứ đúng hoàn toàn, check_all_right sẽ được chạy riêng;
        # ở main sẽ ưu tiên all right nếu không có diagnosis khác.
        add_diagnosis(diagnosis, "reverse steps")


def check_all_right(
    plan: Dict[str, Any],
    student_plan: Dict[str, Any],
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    diagnosis: List[Dict[str, Any]],
) -> bool:
    """
    True nếu toàn bộ entity đã map có map/value/unit/expr khớp nhau và không có entity student chưa map.
    """
    if not step_order_matches(plan, student_plan, student_entities):
        return False

    for student_name, student_entity in student_entities.items():
        mapped_plan_name = normalize_empty(student_entity.get("map"))
        if not mapped_plan_name or mapped_plan_name not in plan_entities:
            return False

        plan_entity = plan_entities[mapped_plan_name]

        # map khớp 2 chiều nếu phía plan có map.
        reverse_map = normalize_empty(plan_entity.get("map"))
        if reverse_map is not None and reverse_map != student_name:
            return False

        if not decimal_equal(student_entity.get("value"), plan_entity.get("value")):
            return False
        if normalize_empty(student_entity.get("unit")) != normalize_empty(plan_entity.get("unit")):
            return False

        student_expr = normalize_empty(student_entity.get("expr"))
        plan_expr = normalize_empty(plan_entity.get("expr"))
        if expressions_textually_same(student_expr, plan_expr):
            continue

        student_formalized = normalize_empty(student_entity.get("formalized_expr"))
        plan_formalized = normalize_empty(plan_entity.get("formalized_expr"))
        if not student_formalized or not plan_formalized:
            return False

        plan_formalized_for_compare = replace_answer_literals_with_values(plan_formalized, plan_entities)
        student_formalized_mapped = replace_student_tokens_with_plan_tokens(
            student_formalized,
            student_entities,
            plan_entities,
        )
        student_formalized_mapped = replace_answer_literals_with_values(student_formalized_mapped, plan_entities)
        if not equivalent_by_random_substitution(
            plan_formalized_for_compare,
            student_formalized_mapped,
            trials=3,
            fixed_values=fixed_context_values(plan_entities),
        ):
            return False

    add_diagnosis(diagnosis, "all right")
    return True


def check_wrong_relationship(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_plan: Dict[str, Any],
    diagnosis: List[Dict[str, Any]],
) -> None:
    for student_name, student_entity in student_entities.items():
        mapped_plan_name = normalize_empty(student_entity.get("map"))
        if not mapped_plan_name or mapped_plan_name not in plan_entities:
            continue

        plan_entity = plan_entities[mapped_plan_name]
        s_expr = normalize_empty(student_entity.get("expr"))
        p_expr = normalize_empty(plan_entity.get("expr"))
        s_formalized = normalize_empty(student_entity.get("formalized_expr"))
        p_formalized = normalize_empty(plan_entity.get("formalized_expr"))

        # Không có quan hệ từ phía học sinh là missing step, không phải quan hệ sai.
        # InsideChecker chịu trách nhiệm ghi nhãn thiếu bước.
        if s_expr is None:
            continue
        if expressions_textually_same(s_expr, p_expr):
            continue
        if s_formalized and p_formalized:
            p_formalized_for_compare = replace_answer_literals_with_values(p_formalized, plan_entities)
            s_formalized_mapped = replace_student_tokens_with_plan_tokens(s_formalized, student_entities, plan_entities)
            s_formalized_mapped = replace_answer_literals_with_values(s_formalized_mapped, plan_entities)
            if equivalent_by_random_substitution(
                p_formalized_for_compare,
                s_formalized_mapped,
                trials=3,
                fixed_values=fixed_context_values(plan_entities),
            ):
                continue
            if same_unit_and_value(student_entity, plan_entity) and equivalent_under_entity_values(
                p_formalized_for_compare,
                s_formalized_mapped,
                plan_entities,
            ):
                continue

        step = find_student_step_by_expr(student_plan, s_expr) or find_student_step_by_result(student_plan, student_name)
        add_diagnosis(diagnosis, "wrong relationship", step=step, entity=student_name)


def check_different_calculation(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    diagnosis: List[Dict[str, Any]],
) -> None:
    plan_target = first_target_entity(plan_entities)
    student_target = first_target_entity(student_entities)
    if not plan_target or not student_target:
        return

    plan_target_entity = plan_entities[plan_target]
    student_target_entity = student_entities[student_target]

    plan_expr = normalize_empty(plan_target_entity.get("formalized_expr"))
    student_expr = normalize_empty(student_target_entity.get("formalized_expr"))
    if not plan_expr or not student_expr:
        return

    plan_expr_for_compare = replace_answer_literals_with_values(plan_expr, plan_entities)
    student_expr_mapped = replace_student_tokens_with_plan_tokens(student_expr, student_entities, plan_entities)
    student_expr_mapped = replace_answer_literals_with_values(student_expr_mapped, plan_entities)

    if expressions_textually_same(plan_expr_for_compare, student_expr_mapped):
        return

    # "đáp án lại đúng": so value target trước.
    if not decimal_equal(plan_target_entity.get("value"), student_target_entity.get("value")):
        return

    expressions_equivalent = equivalent_by_random_substitution(
        plan_expr_for_compare,
        student_expr_mapped,
        trials=3,
        fixed_values=fixed_context_values(plan_entities),
    )
    if expressions_equivalent:
        return

    add_diagnosis(diagnosis, "different calculation", entity=student_target)


# -----------------------------------------------------------------------------
# Diagnosis policy
# -----------------------------------------------------------------------------

def remove_structural_label_if_all_right(diagnosis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Nếu all right tồn tại, bỏ reverse steps/combine/step separation/extra step để tránh mâu thuẫn.
    Vì yêu cầu all right là trường hợp hoàn toàn khớp.
    """
    has_all_right = any(item.get("diagnosis") == "all right" for item in diagnosis)
    if not has_all_right:
        return diagnosis
    return [
        item for item in diagnosis
        if item.get("diagnosis") not in {"reverse steps", "combine step", "step separation", "extra step"}
    ]


def current_wrong_is_yes() -> bool:
    if not WRONG_PATH.exists():
        return False
    return WRONG_PATH.read_text(encoding="utf-8").strip().lower() == "yes"


def write_wrong_value(value: str) -> None:
    if value != "Yes" and current_wrong_is_yes():
        return
    WRONG_PATH.write_text(f"{value}\n", encoding="utf-8")


WRONG_CAUSING_LABELS = {
    "do not convert units",
    "logic error",
    "misreading",
    "missing step",
    "only final answer",
    "unit missing",
    "wrong calculation",
    "wrong relationship",
    "wrong target",
    "wrong unit conversion",
    "wrong units conversion",
}


def update_wrong_file(diagnosis: List[Dict[str, Any]]) -> None:
    all_items = read_diagnosis_file() + diagnosis
    has_wrong = any(item.get("diagnosis") in WRONG_CAUSING_LABELS for item in all_items)
    write_wrong_value("Yes" if has_wrong else "No")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a student plan against a reference plan.")
    parser.add_argument(
        "--reference",
        choices=["plan", "teacher"],
        default="plan",
        help=(
            "plan: so sánh với Plan.yaml/PlanEntities.yaml; "
            "teacher: so sánh với TeacherPlan.yaml/TeacherAnswerEntities.yaml."
        ),
    )
    return parser.parse_args()


def reference_paths(reference: str) -> Tuple[Path, Path]:
    if reference == "teacher":
        return TEACHER_PLAN_PATH, TEACHER_ANSWER_ENTITIES_PATH
    return PLAN_PATH, PLAN_ENTITIES_PATH


def run() -> None:
    try:
        args = parse_args()
        ensure_dirs()
        reference_plan_path, reference_entities_path = reference_paths(args.reference)

        plan = normalize_plan(read_yaml_file(reference_plan_path, required=True))
        plan_entities = normalize_entities(read_yaml_file(reference_entities_path, required=True))
        student_plan = normalize_plan(read_yaml_file(STUDENT_PLAN_PATH, required=True))
        student_entities = normalize_entities(read_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, required=True))

        existing_diagnosis = read_diagnosis_file()
        existing_diagnosis, diagnosis_reclassified = reclassify_answer_literal_misreading_as_missing_step(
            existing_diagnosis,
            plan_entities,
            student_entities,
        )
        if diagnosis_reclassified:
            write_yaml_file(DIAGNOSIS_PATH, existing_diagnosis)

        diagnosis: List[Dict[str, Any]] = []
        check_unmapped_answer_literal_misreading(
            plan_entities,
            student_entities,
            student_plan,
            diagnosis,
        )

        existing_has_core_error = has_core_diagnosis(existing_diagnosis + diagnosis)

        # Nếu InsideChecker đã bắt lỗi gốc, không diff graph nữa. Graph sau lỗi
        # gốc thường chỉ phản ánh propagation, không phải lỗi học sinh mới.
        if not existing_has_core_error:
            check_missing_step_from_answer_literals(plan_entities, student_entities, student_plan, diagnosis)
            if not has_compare_core_diagnosis(diagnosis):
                check_wrong_relationship(plan_entities, student_entities, student_plan, diagnosis)
                check_different_calculation(plan_entities, student_entities, diagnosis)

        # Nếu InsideChecker đã bắt lỗi semantic như misreading/wrong calculation,
        # không thêm nhãn structural như reverse steps dựa trên biểu thức symbolic.
        has_core_error = existing_has_core_error or any(
            item.get("diagnosis") in COMPARE_CORE_LABELS for item in diagnosis
        )

        if not has_core_error:
            all_right = check_all_right(plan, student_plan, plan_entities, student_entities, diagnosis)
            if not all_right:
                check_step_structure_change(plan, student_plan, plan_entities, student_entities, diagnosis)

        diagnosis = remove_structural_label_if_all_right(diagnosis)

        append_diagnosis_file(diagnosis)
        update_wrong_file(diagnosis)

        write_log("Pass CompareChecker")
        print("Pass CompareChecker")
    except Exception as exc:
        write_log("Fail CompareChecker", str(exc))
        print("Fail CompareChecker")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
