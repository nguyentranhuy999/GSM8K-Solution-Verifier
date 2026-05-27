"""
Verify/CompareChecker.py

Nhiệm vụ:
- So sánh lời giải chuẩn/LLM với lời giải học sinh sau khi đã formalize, execute và map.
- Đọc:
  - Output/Plan.yaml
  - Output/PlanEntities.yaml
  - Output/StudentPlan.yaml
  - Output/StudentAnswerEntities.yaml
- Ghi output vào:
  - Output/Diagnosis.yaml
  - Output/Wrong.yaml

Các lỗi/nhãn:
- wrong units conversion
- combine step
- step separation
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


def replace_student_tokens_with_plan_tokens(expr: str, student_entities: Dict[str, Dict[str, Any]]) -> str:
    """Đưa formalized_expr của student về namespace của plan bằng trường map."""
    if not expr:
        return expr

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = student_entities.get(token, {}).get("map")
        return mapped or token

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, expr)


def equivalent_by_random_substitution(expr_a: Optional[str], expr_b: Optional[str], *, trials: int = 3) -> bool:
    if not expr_a or not expr_b:
        return False

    tokens = sorted(set(expr_tokens(expr_a)) | set(expr_tokens(expr_b)))
    if not tokens:
        try:
            return decimal_equal(safe_eval_expr(expr_a, {}), safe_eval_expr(expr_b, {}))
        except Exception:
            return False

    rng = random.Random(20260521)
    for _ in range(trials):
        values = {token: Decimal(rng.randint(2, 20)) for token in tokens}
        try:
            value_a = safe_eval_expr(expr_a, values)
            value_b = safe_eval_expr(expr_b, values)
        except Exception:
            return False
        if not decimal_equal(value_a, value_b):
            return False

    return True


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

def check_wrong_units_conversion(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    diagnosis: List[Dict[str, Any]],
) -> None:
    for student_name, student_entity in student_entities.items():
        mapped_plan_name = normalize_empty(student_entity.get("map"))
        if not mapped_plan_name or mapped_plan_name not in plan_entities:
            continue

        plan_entity = plan_entities[mapped_plan_name]
        if decimal_equal(student_entity.get("value"), plan_entity.get("value")):
            continue

        # Chỉ coi là wrong units conversion nếu entity thuộc nhóm đơn vị cần đổi.
        if is_convertible_metadata(student_entity) or is_convertible_metadata(plan_entity):
            add_diagnosis(diagnosis, "wrong units conversion", step=student_entity.get("location"), entity=student_name)


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

    student_expr_mapped = replace_student_tokens_with_plan_tokens(student_expr, student_entities)
    if not equivalent_by_random_substitution(plan_expr, student_expr_mapped, trials=3):
        return

    plan_step_count = count_steps(plan)
    student_step_count = count_steps(student_plan)

    if student_step_count < plan_step_count:
        add_diagnosis(diagnosis, "combine step")
    elif student_step_count > plan_step_count:
        add_diagnosis(diagnosis, "step separation")
    elif student_step_count == plan_step_count:
        # Theo yêu cầu: nếu số bước bằng nhau và formalized_expr tương đương nhau thì reverse steps.
        # Để tránh ghi reverse steps khi mọi thứ đúng hoàn toàn, check_all_right sẽ được chạy riêng;
        # ở main sẽ ưu tiên all right nếu không có diagnosis khác.
        add_diagnosis(diagnosis, "reverse steps")


def check_all_right(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    diagnosis: List[Dict[str, Any]],
) -> bool:
    """
    True nếu toàn bộ entity đã map có map/value/unit/expr khớp nhau và không có entity student chưa map.
    """
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
        if not expressions_textually_same(student_entity.get("expr"), plan_entity.get("expr")):
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

        if s_expr is None and p_expr is None:
            continue
        if expressions_textually_same(s_expr, p_expr):
            continue
        if s_formalized and p_formalized:
            s_formalized_mapped = replace_student_tokens_with_plan_tokens(s_formalized, student_entities)
            if equivalent_by_random_substitution(p_formalized, s_formalized_mapped, trials=3):
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

    student_expr_mapped = replace_student_tokens_with_plan_tokens(student_expr, student_entities)

    if expressions_textually_same(plan_expr, student_expr_mapped):
        return

    # "đáp án lại đúng": so value target trước.
    if not decimal_equal(plan_target_entity.get("value"), student_target_entity.get("value")):
        return

    if not equivalent_by_random_substitution(plan_expr, student_expr_mapped, trials=3):
        add_diagnosis(diagnosis, "different calculation", entity=student_target)


# -----------------------------------------------------------------------------
# Diagnosis policy
# -----------------------------------------------------------------------------

def remove_structural_label_if_all_right(diagnosis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Nếu all right tồn tại, bỏ reverse steps/combine/step separation để tránh mâu thuẫn.
    Vì yêu cầu all right là trường hợp hoàn toàn khớp.
    """
    has_all_right = any(item.get("diagnosis") == "all right" for item in diagnosis)
    if not has_all_right:
        return diagnosis
    return [
        item for item in diagnosis
        if item.get("diagnosis") not in {"reverse steps", "combine step", "step separation"}
    ]


def update_wrong_file(diagnosis: List[Dict[str, Any]]) -> None:
    has_wrong_relationship = any(item.get("diagnosis") == "wrong relationship" for item in diagnosis)
    WRONG_PATH.write_text(("Yes" if has_wrong_relationship else "No") + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run() -> None:
    try:
        ensure_dirs()

        plan = normalize_plan(read_yaml_file(PLAN_PATH, required=True))
        plan_entities = normalize_entities(read_yaml_file(PLAN_ENTITIES_PATH, required=True))
        student_plan = normalize_plan(read_yaml_file(STUDENT_PLAN_PATH, required=True))
        student_entities = normalize_entities(read_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, required=True))

        diagnosis: List[Dict[str, Any]] = []

        # Các lỗi thực sự trước.
        check_wrong_units_conversion(plan_entities, student_entities, diagnosis)
        check_wrong_relationship(plan_entities, student_entities, student_plan, diagnosis)
        check_different_calculation(plan_entities, student_entities, diagnosis)

        # Nếu không có lỗi quan hệ/sai đổi đơn vị/cách khác, mới xét all right hoặc cấu trúc bước.
        has_core_error = any(
            item.get("diagnosis") in {"wrong units conversion", "wrong relationship", "different calculation"}
            for item in diagnosis
        )

        if not has_core_error:
            all_right = check_all_right(plan_entities, student_entities, diagnosis)
            if not all_right:
                check_step_structure_change(plan, student_plan, plan_entities, student_entities, diagnosis)
        else:
            # Vẫn có thể ghi combine/step separation/reverse nếu biểu thức target tương đương,
            # nhưng không ghi nếu đã wrong relationship vì quan hệ sai thường quan trọng hơn.
            if not any(item.get("diagnosis") == "wrong relationship" for item in diagnosis):
                check_step_structure_change(plan, student_plan, plan_entities, student_entities, diagnosis)

        diagnosis = remove_structural_label_if_all_right(diagnosis)

        write_yaml_file(DIAGNOSIS_PATH, diagnosis)
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
