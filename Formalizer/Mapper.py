"""
Formalizer/Mapper.py

Nhiệm vụ:
- Đọc:
  - Output/PlanEntities.yaml
  - Output/StudentAnswerEntities.yaml
- Map các entity trong StudentAnswerEntities.yaml với PlanEntities.yaml.
- Thêm trường map vào toàn bộ entity của cả 2 file.
- Không dùng LLM.

Ý tưởng map:
1. Các entity chung ban đầu từ ProblemEntities.yaml, tức các entity từ đầu đến entity có location: target,
   được auto-map theo tên với nhau.
2. Các entity sau target được map dựa trên quan hệ trong expr/formalized_expr.
3. Việc map không phụ thuộc từng step cứng nhắc. Mapper sẽ lặp nhiều vòng:
   - Dựa vào các entity đã map, chuẩn hóa expr của student về phía plan.
   - So sánh expression signature giữa student entity và plan entity.
   - Map được từ cả hai phía:
     + từ các entity step sớm như daily_cost
     + từ expr của target như total_cost để suy ra daily_cost
4. Nếu không map được thì map để rỗng/null.

Output:
- Update Output/PlanEntities.yaml
- Update Output/StudentAnswerEntities.yaml
- Ghi Pass Mapper / Fail Mapper vào Output/Log.yaml

Ghi chú:
- Trong mô tả user có một chỗ ghi Output/ProblemEntities.yaml, nhưng one-shot và logic pipeline
  đều là update Output/PlanEntities.yaml và Output/StudentAnswerEntities.yaml. File này làm theo one-shot.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Output"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"


class MapperError(Exception):
    """Lỗi riêng cho Mapper."""


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
            raise MapperError(f"Không tìm thấy file: {path}")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MapperError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise MapperError(f"File YAML phải là dictionary: {path}")
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
    log_data["Mapper"] = status
    if message:
        log_data["Mapper_message"] = message
    elif "Mapper_message" in log_data:
        del log_data["Mapper_message"]
    write_yaml_file(LOG_PATH, log_data)


# -----------------------------------------------------------------------------
# Validation / utilities
# -----------------------------------------------------------------------------

def validate_entity_name(name: str) -> None:
    if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise MapperError(f"Tên entity không hợp lệ: {name!r}")


def normalize_entities(raw_entities: Dict[str, Any], *, file_label: str) -> Dict[str, Dict[str, Any]]:
    if not raw_entities:
        raise MapperError(f"{file_label} đang rỗng.")

    required = {"value", "unit", "location", "grand_unit"}
    optional = {"expr", "formalized_expr", "map"}
    normalized: Dict[str, Dict[str, Any]] = {}

    for name, entity in raw_entities.items():
        validate_entity_name(name)
        if not isinstance(entity, dict):
            raise MapperError(f"Entity {name} trong {file_label} phải là dictionary.")

        missing = required - set(entity.keys())
        if missing:
            raise MapperError(f"Entity {name} trong {file_label} thiếu trường: {sorted(missing)}")

        extra = set(entity.keys()) - required - optional
        if extra:
            raise MapperError(f"Entity {name} trong {file_label} có trường thừa: {sorted(extra)}")

        normalized[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
            "expr": normalize_empty(entity.get("expr")),
            "formalized_expr": normalize_empty(entity.get("formalized_expr")),
            "map": normalize_empty(entity.get("map")),
        }

    return normalized


def expr_tokens(expr: Optional[str]) -> List[str]:
    if not expr:
        return []
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def first_target_index(entities: Dict[str, Dict[str, Any]]) -> Optional[int]:
    for idx, (_, entity) in enumerate(entities.items()):
        if entity.get("location") == "target":
            return idx
    return None


def entity_names_until_target(entities: Dict[str, Dict[str, Any]]) -> List[str]:
    names = list(entities.keys())
    idx = first_target_index(entities)
    if idx is None:
        return []
    return names[: idx + 1]


def location_step_number(location: Any) -> Optional[int]:
    if not isinstance(location, str):
        return None
    match = re.fullmatch(r"step(\d+)", location)
    if not match:
        return None
    return int(match.group(1))


def is_input_or_target(entity: Dict[str, Any]) -> bool:
    return entity.get("location") in {"input", "target"}


def compatible_metadata(student_entity: Dict[str, Any], plan_entity: Dict[str, Any]) -> bool:
    """
    Kiểm tra nhẹ metadata để tránh map nhầm.
    Không so value vì học sinh có thể tính sai value.
    """
    s_unit = normalize_empty(student_entity.get("unit"))
    p_unit = normalize_empty(plan_entity.get("unit"))
    s_grand = normalize_empty(student_entity.get("grand_unit"))
    p_grand = normalize_empty(plan_entity.get("grand_unit"))

    # result_unit có thể là missing do học sinh quên đơn vị, vẫn cho map.
    if s_unit not in {None, "missing"} and p_unit not in {None, "missing"} and s_unit != p_unit:
        return False

    if s_grand not in {None, "missing"} and p_grand not in {None, "missing"} and s_grand != p_grand:
        return False

    return True


# -----------------------------------------------------------------------------
# Expression canonicalization
# -----------------------------------------------------------------------------

COMMUTATIVE_OPS = {"Add", "Mult"}


def replace_names_by_map(expr: str, student_to_plan: Dict[str, str]) -> str:
    """Thay tên entity student trong expr bằng tên entity plan nếu đã map."""
    if not expr:
        return expr

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return student_to_plan.get(token, token)

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, expr)


def canonical_ast(node: ast.AST) -> Any:
    """
    Tạo chữ ký AST để so sánh expression.
    - Add và Mult được coi là giao hoán nên sort operands.
    - Sub/Div giữ thứ tự.
    - Name giữ tên biến.
    - Constant giữ giá trị text mức đơn giản.
    """
    if isinstance(node, ast.Expression):
        return canonical_ast(node.body)

    if isinstance(node, ast.Name):
        return ("Name", node.id)

    if isinstance(node, ast.Constant):
        return ("Const", str(node.value))

    if isinstance(node, ast.Num):  # pragma: no cover
        return ("Const", str(node.n))

    if isinstance(node, ast.UnaryOp):
        return ("Unary", type(node.op).__name__, canonical_ast(node.operand))

    if isinstance(node, ast.BinOp):
        op_name = type(node.op).__name__
        left = canonical_ast(node.left)
        right = canonical_ast(node.right)

        if op_name in COMMUTATIVE_OPS:
            flattened = flatten_commutative(op_name, [left, right])
            return ("BinOp", op_name, tuple(sorted(flattened, key=repr)))

        return ("BinOp", op_name, left, right)

    # Nếu expression có thành phần lạ, dùng dump để vẫn có signature nhưng khó match hơn.
    return ("Other", ast.dump(node, include_attributes=False))


def flatten_commutative(op_name: str, parts: List[Any]) -> List[Any]:
    flattened: List[Any] = []
    for part in parts:
        if isinstance(part, tuple) and len(part) == 3 and part[0] == "BinOp" and part[1] == op_name:
            flattened.extend(list(part[2]))
        else:
            flattened.append(part)
    return flattened


def canonical_expr(expr: Optional[str]) -> Optional[Any]:
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        # fallback string normalize nếu expr không parse được.
        normalized = re.sub(r"\s+", "", expr)
        return ("Raw", normalized)
    return canonical_ast(tree)


def expression_signature(
    expr: Optional[str],
    *,
    student_to_plan: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    if not expr:
        return None
    if student_to_plan:
        expr = replace_names_by_map(expr, student_to_plan)
    return canonical_expr(expr)


def entity_signature_candidates(
    name: str,
    entity: Dict[str, Any],
    *,
    student_to_plan: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """Sinh các signature có thể dùng để map entity."""
    candidates: List[Any] = []

    for key in ("formalized_expr", "expr"):
        sig = expression_signature(entity.get(key), student_to_plan=student_to_plan)
        if sig is not None and sig not in candidates:
            candidates.append(sig)

    return candidates


# -----------------------------------------------------------------------------
# Mapping core
# -----------------------------------------------------------------------------

def auto_map_common_prefix_until_target(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Auto-map các entity chung từ đầu đến entity location target.

    Theo giả định pipeline: 2 file có chung phần entity từ ProblemEntities.yaml.
    Vì vậy map theo tên cho các entity xuất hiện ở cả 2 phía trong prefix đến target.
    """
    plan_prefix = entity_names_until_target(plan_entities)
    student_prefix = entity_names_until_target(student_entities)
    common_prefix_names = [name for name in student_prefix if name in plan_prefix]

    student_to_plan: Dict[str, str] = {}
    plan_to_student: Dict[str, str] = {}

    for name in common_prefix_names:
        student_to_plan[name] = name
        plan_to_student[name] = name

    return student_to_plan, plan_to_student


def build_plan_signature_index(
    plan_entities: Dict[str, Dict[str, Any]],
    mapped_plan_names: Set[str],
) -> Dict[Any, List[str]]:
    index: Dict[Any, List[str]] = {}
    for plan_name, plan_entity in plan_entities.items():
        if plan_name in mapped_plan_names:
            continue
        if is_input_or_target(plan_entity):
            # Input/target đã được xử lý ở auto-map prefix.
            continue
        for sig in entity_signature_candidates(plan_name, plan_entity):
            index.setdefault(sig, []).append(plan_name)
    return index


def map_by_same_result_name(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_to_plan: Dict[str, str],
    plan_to_student: Dict[str, str],
) -> bool:
    """Map bổ sung theo cùng tên result nếu metadata tương thích."""
    changed = False
    for student_name, student_entity in student_entities.items():
        if student_name in student_to_plan:
            continue
        if student_name not in plan_entities:
            continue
        if student_name in plan_to_student:
            continue
        plan_entity = plan_entities[student_name]
        if not compatible_metadata(student_entity, plan_entity):
            continue

        student_to_plan[student_name] = student_name
        plan_to_student[student_name] = student_name
        changed = True
    return changed


def map_by_expression_signature(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_to_plan: Dict[str, str],
    plan_to_student: Dict[str, str],
) -> bool:
    """
    Map entity chưa map bằng signature của expr/formalized_expr.
    Vì student expr được thay token theo student_to_plan hiện có, việc map được lan truyền dần.
    """
    changed = False
    mapped_plan_names = set(plan_to_student.keys())
    plan_index = build_plan_signature_index(plan_entities, mapped_plan_names)

    for student_name, student_entity in student_entities.items():
        if student_name in student_to_plan:
            continue
        if is_input_or_target(student_entity):
            continue

        s_candidates = entity_signature_candidates(
            student_name,
            student_entity,
            student_to_plan=student_to_plan,
        )

        possible_plan_names: List[str] = []
        for sig in s_candidates:
            possible_plan_names.extend(plan_index.get(sig, []))

        counts = Counter(possible_plan_names)
        if not counts:
            continue

        # Ưu tiên candidate xuất hiện nhiều signature match hơn, sau đó ưu tiên tên giống nhau.
        ranked = sorted(
            counts.keys(),
            key=lambda plan_name: (
                counts[plan_name],
                plan_name == student_name,
                -abs((location_step_number(plan_entities[plan_name].get("location")) or 0) - (location_step_number(student_entity.get("location")) or 0)),
            ),
            reverse=True,
        )

        compatible = [
            plan_name
            for plan_name in ranked
            if plan_name not in plan_to_student
            and compatible_metadata(student_entity, plan_entities[plan_name])
        ]

        if len(compatible) == 1:
            plan_name = compatible[0]
            student_to_plan[student_name] = plan_name
            plan_to_student[plan_name] = student_name
            changed = True
        elif len(compatible) > 1:
            # Nếu có nhiều ứng viên, chỉ map khi có ứng viên cùng tên để tránh map sai.
            if student_name in compatible:
                student_to_plan[student_name] = student_name
                plan_to_student[student_name] = student_name
                changed = True

    return changed


def infer_missing_internal_maps_from_target_expr(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_to_plan: Dict[str, str],
    plan_to_student: Dict[str, str],
) -> bool:
    """
    Map lan truyền từ expr của target.

    Ví dụ target cả hai phía đều là:
      daily_cost * days
    days đã map, daily_cost chưa map.
    Khi so sánh cấu trúc target expr, có thể suy ra student daily_cost -> plan daily_cost.
    """
    changed = False

    target_student_names = [name for name, ent in student_entities.items() if ent.get("location") == "target"]
    for s_target in target_student_names:
        p_target = student_to_plan.get(s_target)
        if not p_target or p_target not in plan_entities:
            continue

        s_expr = student_entities[s_target].get("expr")
        p_expr = plan_entities[p_target].get("expr")
        if not s_expr or not p_expr:
            continue

        s_tokens = [tok for tok in expr_tokens(s_expr) if tok not in student_to_plan]
        p_tokens = [tok for tok in expr_tokens(p_expr) if tok not in plan_to_student]

        if len(set(s_tokens)) == 1 and len(set(p_tokens)) == 1:
            s_unknown = list(set(s_tokens))[0]
            p_unknown = list(set(p_tokens))[0]

            # Thử thay giả định rồi so signature target.
            trial_map = dict(student_to_plan)
            trial_map[s_unknown] = p_unknown
            s_sig = expression_signature(s_expr, student_to_plan=trial_map)
            p_sig = expression_signature(p_expr)

            if s_sig == p_sig and compatible_metadata(student_entities[s_unknown], plan_entities[p_unknown]):
                student_to_plan[s_unknown] = p_unknown
                plan_to_student[p_unknown] = s_unknown
                changed = True

    return changed


def compute_mapping(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    student_to_plan, plan_to_student = auto_map_common_prefix_until_target(plan_entities, student_entities)

    # Lặp để map lan truyền. Giới hạn theo số entity để tránh vòng lặp vô hạn.
    max_rounds = max(len(plan_entities), len(student_entities)) + 5
    for _ in range(max_rounds):
        changed = False
        changed |= map_by_same_result_name(plan_entities, student_entities, student_to_plan, plan_to_student)
        changed |= infer_missing_internal_maps_from_target_expr(plan_entities, student_entities, student_to_plan, plan_to_student)
        changed |= map_by_expression_signature(plan_entities, student_entities, student_to_plan, plan_to_student)
        changed |= infer_missing_internal_maps_from_target_expr(plan_entities, student_entities, student_to_plan, plan_to_student)
        if not changed:
            break

    return student_to_plan, plan_to_student


def apply_map_fields(
    plan_entities: Dict[str, Dict[str, Any]],
    student_entities: Dict[str, Dict[str, Any]],
    student_to_plan: Dict[str, str],
    plan_to_student: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    updated_plan = deepcopy(plan_entities)
    updated_student = deepcopy(student_entities)

    for plan_name, entity in updated_plan.items():
        entity["map"] = plan_to_student.get(plan_name)

    for student_name, entity in updated_student.items():
        entity["map"] = student_to_plan.get(student_name)

    return updated_plan, updated_student


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run() -> None:
    try:
        ensure_dirs()
        raw_plan_entities = read_yaml_file(PLAN_ENTITIES_PATH, required=True)
        raw_student_entities = read_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, required=True)

        plan_entities = normalize_entities(raw_plan_entities, file_label="Output/PlanEntities.yaml")
        student_entities = normalize_entities(raw_student_entities, file_label="Output/StudentAnswerEntities.yaml")

        student_to_plan, plan_to_student = compute_mapping(plan_entities, student_entities)
        updated_plan, updated_student = apply_map_fields(
            plan_entities,
            student_entities,
            student_to_plan,
            plan_to_student,
        )

        write_yaml_file(PLAN_ENTITIES_PATH, updated_plan)
        write_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, updated_student)

        write_log("Pass Mapper")
        print("Pass Mapper")
    except Exception as exc:
        write_log("Fail Mapper", str(exc))
        print("Fail Mapper")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
