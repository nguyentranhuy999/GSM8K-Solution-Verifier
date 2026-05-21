"""
Verify/InsideChecker.py

Nhiệm vụ:
- Kiểm tra lỗi sai nội tại của lời giải qua plan và entities.
- Có 2 chế độ:
  1. LLM/reference mode:
     - Input:
       - Output/Plan.yaml
       - Output/PlanEntities.yaml
     - Output:
       - Output/Error.yaml
  2. Student mode:
     - Input:
       - Output/StudentPlan.yaml
       - Output/StudentAnswerEntities.yaml
     - Output:
       - Output/Diagnosis.yaml

Cách chạy:
- Mặc định check LLM/reference:
  python3 Verify/InsideChecker.py

- Check LLM/reference rõ ràng:
  python3 Verify/InsideChecker.py --mode llm

- Check lời giải học sinh:
  python3 Verify/InsideChecker.py --mode student

Các lỗi check:
- wrong target
- wrong calculation
- unit missing       # student mode only
- only final answer
- wrong relationship
- do not convert units
- missing step
- misreading
- logic error
- extra step

Quy tắc Wrong.yaml:
- Nếu có lỗi khác extra step: ghi Yes
- Nếu chỉ có extra step: ghi No
- Nếu không có lỗi: ghi rỗng hoặc không đổi tùy pipeline; file này ghi No để biểu diễn không sai.
"""

from __future__ import annotations

import argparse
import ast
import operator
import re
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml


getcontext().prec = 28

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Output"

PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
STUDENT_PLAN_PATH = OUTPUT_DIR / "StudentPlan.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"

ERROR_PATH = OUTPUT_DIR / "Error.yaml"
DIAGNOSIS_PATH = OUTPUT_DIR / "Diagnosis.yaml"
WRONG_PATH = OUTPUT_DIR / "Wrong.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"


class InsideCheckerError(Exception):
    """Lỗi riêng cho InsideChecker."""


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
            raise InsideCheckerError(f"Không tìm thấy file: {path}")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InsideCheckerError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InsideCheckerError(f"File YAML phải là dictionary: {path}")
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


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    log_data["InsideChecker"] = status
    if message:
        log_data["InsideChecker_message"] = message
    elif "InsideChecker_message" in log_data:
        del log_data["InsideChecker_message"]
    write_yaml_file(LOG_PATH, log_data)


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------

def to_decimal(value: Any, *, context: str = "value") -> Decimal:
    value = normalize_empty(value)
    if value is None:
        raise InsideCheckerError(f"{context} đang rỗng.")
    if isinstance(value, bool):
        raise InsideCheckerError(f"{context} không được là boolean.")
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise InsideCheckerError(f"{context} không phải số: {value!r}") from exc


def decimal_equal(a: Decimal, b: Decimal, tolerance: Decimal = Decimal("0.000001")) -> bool:
    return abs(a - b) <= tolerance


def parse_numbers_from_text(text: str) -> List[Decimal]:
    if text is None:
        return []
    # Hỗ trợ $3.00, -3.5, 1800, 1,000.25. Không xử lý phân số chữ ở checker này.
    raw_numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    numbers: List[Decimal] = []
    for raw in raw_numbers:
        try:
            numbers.append(Decimal(raw.replace(",", "")))
        except InvalidOperation:
            continue
    return numbers


def split_reported_expr(reported_expr: str) -> Tuple[str, str]:
    if not isinstance(reported_expr, str) or "=" not in reported_expr:
        raise InsideCheckerError(f"reported_expr phải có dấu '=': {reported_expr!r}")
    lhs, rhs = reported_expr.rsplit("=", 1)
    return lhs.strip(), rhs.strip().rstrip(".。;, ")


def parse_reported_rhs_value(reported_expr: str) -> Decimal:
    _, rhs = split_reported_expr(reported_expr)
    numbers = parse_numbers_from_text(rhs)
    if not numbers:
        raise InsideCheckerError(f"Không tìm thấy giá trị sau dấu '=' trong reported_expr: {reported_expr!r}")
    return numbers[-1]


# -----------------------------------------------------------------------------
# Safe arithmetic evaluator
# -----------------------------------------------------------------------------

ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def eval_ast_node(node: ast.AST, values: Optional[Dict[str, Decimal]] = None) -> Decimal:
    values = values or {}

    if isinstance(node, ast.Expression):
        return eval_ast_node(node.body, values)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise InsideCheckerError("Boolean không được phép trong biểu thức.")
        if isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        raise InsideCheckerError(f"Hằng không hợp lệ: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id not in values:
            raise InsideCheckerError(f"Biến {node.id!r} chưa có value.")
        return values[node.id]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise InsideCheckerError(f"Toán tử không được hỗ trợ: {op_type.__name__}")
        left = eval_ast_node(node.left, values)
        right = eval_ast_node(node.right, values)
        if op_type is ast.Div and right == 0:
            raise InsideCheckerError("Chia cho 0.")
        return Decimal(str(ALLOWED_BIN_OPS[op_type](left, right)))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise InsideCheckerError(f"Unary operator không được hỗ trợ: {op_type.__name__}")
        return Decimal(str(ALLOWED_UNARY_OPS[op_type](eval_ast_node(node.operand, values))))

    raise InsideCheckerError(f"Biểu thức chứa thành phần không an toàn: {type(node).__name__}")


def safe_eval_arithmetic(expr: str, values: Optional[Dict[str, Decimal]] = None) -> Decimal:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise InsideCheckerError(f"Biểu thức không hợp lệ: {expr!r}") from exc
    return eval_ast_node(tree, values or {})


# -----------------------------------------------------------------------------
# Expression helpers
# -----------------------------------------------------------------------------

def expr_tokens(expr: Optional[str]) -> List[str]:
    if not expr:
        return []
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", str(expr))


def step_names(plan: Dict[str, Any]) -> List[str]:
    def key_fn(name: str) -> int:
        match = re.fullmatch(r"step(\d+)", name)
        return int(match.group(1)) if match else 10**9

    return sorted([key for key in plan.keys() if re.fullmatch(r"step\d+", str(key))], key=key_fn)


def step_number(step_name: Optional[str]) -> Optional[int]:
    if not isinstance(step_name, str):
        return None
    match = re.fullmatch(r"step(\d+)", step_name)
    if not match:
        return None
    return int(match.group(1))


def values_for_symbolic_expr(expr: str, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Decimal]:
    values: Dict[str, Decimal] = {}
    for token in sorted(set(expr_tokens(expr))):
        if token not in entities:
            raise InsideCheckerError(f"Entity {token!r} không tồn tại.")
        values[token] = to_decimal(entities[token].get("value"), context=f"{token}.value")
    return values


def add_error(errors: List[Dict[str, Any]], diagnosis: str, step: Optional[str] = None, entity: Optional[str] = None) -> None:
    item = {
        "diagnosis": diagnosis,
        "step": step,
        "entity": entity,
    }
    if item not in errors:
        errors.append(item)


# -----------------------------------------------------------------------------
# Normalize inputs
# -----------------------------------------------------------------------------

def normalize_plan(raw_plan: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    plan = deepcopy(raw_plan)
    if not plan:
        return plan

    for name in step_names(plan):
        step = plan[name]
        if not isinstance(step, dict):
            raise InsideCheckerError(f"{name} phải là dictionary.")
        if "grand_result_unit" in step and "result_grand_unit" not in step:
            step["result_grand_unit"] = step.pop("grand_result_unit")
        step.setdefault("expr", None)
        step.setdefault("result", None)
        step.setdefault("result_unit", None)
        step.setdefault("result_grand_unit", None)
        step.setdefault("reported_expr", None)

    if mode == "student" and "target" not in plan:
        # Không raise ngay để check_wrong_target ghi lỗi được.
        plan["target"] = None

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
            **({"map": normalize_empty(entity.get("map"))} if "map" in entity else {}),
        }
    return entities


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_wrong_target(plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]], *, mode: str) -> None:
    target_entities = [name for name, ent in entities.items() if ent.get("location") == "target"]

    if not target_entities:
        add_error(errors, "wrong target", entity=None)
        return

    for target_name in target_entities:
        ent = entities[target_name]
        value = normalize_empty(ent.get("value"))
        expr = normalize_empty(ent.get("expr"))
        if value is None and expr is None:
            add_error(errors, "wrong target", entity=target_name)

    if mode == "student":
        chosen_target = normalize_empty(plan.get("target"))
        if chosen_target is None:
            add_error(errors, "wrong target", entity=None)
        elif chosen_target not in entities or entities[chosen_target].get("location") != "target":
            add_error(errors, "wrong target", entity=chosen_target)


def check_negative_count_target(entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    for target_name, entity in entities.items():
        if entity.get("location") != "target":
            continue
        if not is_generic_count_unit(entity.get("unit")):
            continue

        value = normalize_empty(entity.get("value"))
        if value is None:
            continue

        try:
            numeric_value = to_decimal(value, context=f"{target_name}.value")
        except InsideCheckerError:
            continue

        if numeric_value < 0:
            add_error(errors, "logic error", step=entity.get("location"), entity=target_name)


def check_wrong_calculation(plan: Dict[str, Any], errors: List[Dict[str, Any]]) -> None:
    for sname in step_names(plan):
        step = plan[sname]
        reported_expr = normalize_empty(step.get("reported_expr"))
        if not reported_expr:
            continue

        try:
            lhs, _ = split_reported_expr(reported_expr)
            expected = safe_eval_arithmetic(lhs)
            actual = parse_reported_rhs_value(reported_expr)
        except Exception:
            add_error(errors, "wrong calculation", step=sname, entity=step.get("result"))
            continue

        if not decimal_equal(expected, actual):
            add_error(errors, "wrong calculation", step=sname, entity=step.get("result"))


def check_unit_missing(entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]], *, mode: str) -> None:
    if mode != "student":
        return
    for name, entity in entities.items():
        if normalize_empty(entity.get("unit")) == "missing":
            add_error(errors, "unit missing", step=entity.get("location"), entity=name)


def check_only_final_answer(plan: Dict[str, Any], errors: List[Dict[str, Any]]) -> None:
    steps = step_names(plan)
    if len(steps) != 1:
        return
    step = plan[steps[0]]
    if normalize_empty(step.get("expr")) is None:
        add_error(errors, "only final answer", step=steps[0], entity=step.get("result"))


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

NON_COUNT_UNITS = {
    "dollar", "dollars", "usd", "$", "cent", "cents",
    "mm", "millimeter", "millimeters", "cm", "centimeter", "centimeters",
    "m", "meter", "meters", "km", "kilometer", "kilometers",
    "inch", "inches", "ft", "foot", "feet", "yard", "yards", "mile", "miles",
    "mg", "milligram", "milligrams", "g", "gram", "grams",
    "kg", "kilogram", "kilograms", "ton", "tons",
    "lb", "lbs", "pound", "pounds", "ounce", "ounces",
    "ml", "milliliter", "milliliters", "l", "liter", "liters",
    "gallon", "gallons", "quart", "quarts", "pint", "pints", "cup", "cups",
    "mm2", "cm2", "m2", "km2",
    "square_meters", "square_meter", "square_centimeters", "square_centimeter",
    "square_feet", "square_foot", "hectare", "hectares", "acre", "acres",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
}


def normalized_unit(unit: Any) -> Optional[str]:
    unit = normalize_empty(unit)
    if unit is None:
        return None
    return str(unit).strip().lower().replace(" ", "_")


def is_generic_count_unit(unit: Any) -> bool:
    normalized = normalized_unit(unit)
    if not normalized:
        return False
    return normalized not in NON_COUNT_UNITS


def same_convertible_family(unit_a: Any, unit_b: Any) -> bool:
    a = normalized_unit(unit_a)
    b = normalized_unit(unit_b)
    if not a or not b:
        return False
    for group in CONVERTIBLE_UNIT_GROUPS:
        if a in group and b in group:
            return True
    return False


def collect_add_sub_pairs(expr: str) -> List[Tuple[str, str]]:
    """Lấy các cặp operand trực tiếp trong phép + hoặc -."""
    pairs: List[Tuple[str, str]] = []
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return pairs

    def first_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
        return names[0] if names else None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = first_name(node.left)
            right = first_name(node.right)
            if left and right:
                pairs.append((left, right))
    return pairs


def check_wrong_relationship(plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    for sname in step_names(plan):
        step = plan[sname]
        expr = normalize_empty(step.get("expr"))
        if not expr:
            continue

        for left_name, right_name in collect_add_sub_pairs(expr):
            if left_name not in entities or right_name not in entities:
                continue

            left = entities[left_name]
            right = entities[right_name]
            left_unit = normalize_empty(left.get("unit"))
            right_unit = normalize_empty(right.get("unit"))
            left_grand = normalize_empty(left.get("grand_unit"))
            right_grand = normalize_empty(right.get("grand_unit"))

            unit_diff = left_unit != right_unit
            grand_diff = left_grand != right_grand

            if unit_diff and grand_diff:
                if is_generic_count_unit(left_unit) and is_generic_count_unit(right_unit):
                    continue
                if same_convertible_family(left_unit, right_unit) or same_convertible_family(left_grand, right_grand):
                    add_error(errors, "do not convert units", step=sname, entity=step.get("result"))
                else:
                    add_error(errors, "wrong relationship", step=sname, entity=step.get("result"))


def produced_step_by_entity(plan: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for sname in step_names(plan):
        result = normalize_empty(plan[sname].get("result"))
        if result:
            mapping[result] = sname
    return mapping


def check_missing_step(plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    produced_by = produced_step_by_entity(plan)

    for name, entity in entities.items():
        location = normalize_empty(entity.get("location"))
        expr = normalize_empty(entity.get("expr"))

        if location == "input":
            continue
        if not expr:
            # target không có expr/value đã được wrong target bắt; entity trung gian không expr là missing step.
            if location != "target":
                add_error(errors, "missing step", step=location, entity=name)
            continue

        current_step_no = step_number(location)
        if location == "target":
            # Target có thể location target, nên dùng step tạo ra nó nếu có.
            current_step_no = step_number(produced_by.get(name)) or 10**9

        for token in expr_tokens(expr):
            if token not in entities:
                add_error(errors, "missing step", step=location if location != "target" else produced_by.get(name), entity=token)
                continue

            token_entity = entities[token]
            token_value = normalize_empty(token_entity.get("value"))
            if token_value is None:
                add_error(errors, "missing step", step=location if location != "target" else produced_by.get(name), entity=token)
                continue

            token_location = normalize_empty(token_entity.get("location"))
            token_step_no = step_number(token_location)
            if token_step_no is not None and current_step_no is not None and token_step_no >= current_step_no:
                add_error(errors, "missing step", step=location if location != "target" else produced_by.get(name), entity=token)


def numbers_match_sequence(expected_values: List[Decimal], reported_numbers: List[Decimal]) -> List[Tuple[int, Decimal, Optional[Decimal]]]:
    """
    So sánh theo thứ tự nhẹ giữa value entity trong expr và các số ở vế trái reported_expr.
    Trả về các mismatch theo index.
    """
    mismatches: List[Tuple[int, Decimal, Optional[Decimal]]] = []
    for idx, expected in enumerate(expected_values):
        actual = reported_numbers[idx] if idx < len(reported_numbers) else None
        if actual is None or not decimal_equal(expected, actual):
            mismatches.append((idx, expected, actual))
    return mismatches


def check_misreading_and_logic_error(plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    for sname in step_names(plan):
        step = plan[sname]
        expr = normalize_empty(step.get("expr"))
        reported_expr = normalize_empty(step.get("reported_expr"))
        if not expr or not reported_expr or "=" not in str(reported_expr):
            continue

        try:
            lhs, _ = split_reported_expr(str(reported_expr))
        except Exception:
            continue

        expected_values: List[Decimal] = []
        token_names: List[Optional[str]] = []

        expr_items = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b|-?\d+(?:\.\d+)?", str(expr))
        for item in expr_items:
            if item in entities:
                value = normalize_empty(entities[item].get("value"))
                if value is None:
                    continue
                try:
                    expected_values.append(to_decimal(value, context=f"{item}.value"))
                    token_names.append(item)
                except Exception:
                    continue
            else:
                try:
                    expected_values.append(Decimal(item))
                    token_names.append(None)
                except InvalidOperation:
                    continue

        reported_numbers = parse_numbers_from_text(lhs)
        mismatches = numbers_match_sequence(expected_values, reported_numbers)

        for idx, _, _ in mismatches:
            if idx >= len(token_names):
                continue
            token = token_names[idx]
            if token is None:
                continue
            token_location = entities[token].get("location")
            if token_location == "input":
                add_error(errors, "misreading", step=sname, entity=token)
            else:
                add_error(errors, "logic error", step=sname, entity=token)


def remove_extra_step_once(plan: Dict[str, Any], target_entity: Optional[str]) -> Tuple[Optional[Tuple[str, str]], Dict[str, Any]]:
    """
    Tìm một extra step, trả về ((step, entity), new_plan_without_step).
    Extra step: result không được dùng trong step sau và không phải target.
    """
    steps = step_names(plan)
    used_later: Dict[str, Set[str]] = {s: set() for s in steps}

    for i, sname in enumerate(steps):
        later_steps = steps[i + 1 :]
        used: Set[str] = set()
        for later in later_steps:
            used.update(expr_tokens(normalize_empty(plan[later].get("expr"))))
        used_later[sname] = used

    for sname in steps:
        result = normalize_empty(plan[sname].get("result"))
        if not result:
            continue
        if result == target_entity:
            continue
        if result not in used_later[sname]:
            new_plan = deepcopy(plan)
            new_plan.pop(sname, None)
            # Reindex step để tiếp tục scan chính xác.
            remaining_steps = step_names(new_plan)
            reindexed: Dict[str, Any] = {}
            for idx, old_step in enumerate(remaining_steps, start=1):
                reindexed[f"step{idx}"] = new_plan[old_step]
            if "target" in new_plan:
                reindexed["target"] = new_plan["target"]
            return (sname, result), reindexed

    return None, plan


def check_extra_step(plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    target_entities = [name for name, ent in entities.items() if ent.get("location") == "target"]
    target_entity = normalize_empty(plan.get("target")) or (target_entities[0] if target_entities else None)

    plan_copy = deepcopy(plan)
    seen: Set[Tuple[str, str]] = set()

    while True:
        extra, plan_copy = remove_extra_step_once(plan_copy, target_entity)
        if extra is None:
            break
        sname, entity = extra
        if (sname, entity) not in seen:
            add_error(errors, "extra step", step=sname, entity=entity)
            seen.add((sname, entity))


# -----------------------------------------------------------------------------
# Wrong.yaml policy
# -----------------------------------------------------------------------------

def update_wrong_file(errors: List[Dict[str, Any]]) -> None:
    if not errors:
        WRONG_PATH.write_text("No\n", encoding="utf-8")
        return

    has_non_extra = any(err.get("diagnosis") != "extra step" for err in errors)
    WRONG_PATH.write_text(("Yes" if has_non_extra else "No") + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def load_inputs(mode: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Path]:
    if mode == "llm":
        raw_plan = read_yaml_file(PLAN_PATH, required=True)
        raw_entities = read_yaml_file(PLAN_ENTITIES_PATH, required=True)
        output_path = ERROR_PATH
    elif mode == "student":
        raw_plan = read_yaml_file(STUDENT_PLAN_PATH, required=True)
        raw_entities = read_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, required=True)
        output_path = DIAGNOSIS_PATH
    else:
        raise InsideCheckerError(f"Mode không hợp lệ: {mode}")

    plan = normalize_plan(raw_plan, mode=mode)
    entities = normalize_entities(raw_entities)
    return plan, entities, output_path


def run_checks(mode: str) -> List[Dict[str, Any]]:
    plan, entities, _ = load_inputs(mode)
    errors: List[Dict[str, Any]] = []

    check_wrong_target(plan, entities, errors, mode=mode)
    check_negative_count_target(entities, errors)
    check_wrong_calculation(plan, errors)
    check_unit_missing(entities, errors, mode=mode)
    check_only_final_answer(plan, errors)
    check_wrong_relationship(plan, entities, errors)
    check_missing_step(plan, entities, errors)
    check_misreading_and_logic_error(plan, entities, errors)
    check_extra_step(plan, entities, errors)

    return errors


def write_outputs(mode: str, errors: List[Dict[str, Any]]) -> None:
    if mode == "llm":
        write_yaml_file(ERROR_PATH, errors)
    else:
        write_yaml_file(DIAGNOSIS_PATH, errors)

    update_wrong_file(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check internal consistency of plan/entities.")
    parser.add_argument(
        "--mode",
        choices=["llm", "student"],
        default="llm",
        help="llm: check Output/Plan.yaml + Output/PlanEntities.yaml; student: check Output/StudentPlan.yaml + Output/StudentAnswerEntities.yaml",
    )
    return parser.parse_args()


def run() -> None:
    try:
        ensure_dirs()
        args = parse_args()
        errors = run_checks(args.mode)
        write_outputs(args.mode, errors)
        write_log("Pass InsideChecker")
        print("Pass InsideChecker")
    except Exception as exc:
        write_log("Fail InsideChecker", str(exc))
        print("Fail InsideChecker")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
