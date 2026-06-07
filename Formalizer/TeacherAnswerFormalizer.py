"""
Formalizer/TeacherAnswerFormalizer.py

Nhiệm vụ:
- Đọc:
  - Input/Problem.txt
  - Input/TeacherAnswer.txt
  - Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để formalize lời giải chuẩn của giáo viên.
- Ghi output:
  - Output/TeacherPlan.yaml
  - Output/TeacherAnswerEntities.yaml

Luồng này dùng khi muốn so sánh bài học sinh với lời giải giáo viên thay vì
lời giải do Solver tự lập.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Formalizer import StudentAnswerFormalizer as student_formalizer
from Verifier import InsideChecker as inside_checker


INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

PROBLEM_PATH = INPUT_DIR / "Problem.txt"
TEACHER_ANSWER_PATH = INPUT_DIR / "TeacherAnswer.txt"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"

TEACHER_PLAN_PATH = OUTPUT_DIR / "TeacherPlan.yaml"
TEACHER_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "TeacherAnswerEntities.yaml"
TEACHER_TRACE_PATH = OUTPUT_DIR / "TeacherTrace.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3


class TeacherAnswerFormalizerError(Exception):
    """Lỗi riêng cho TeacherAnswerFormalizer."""


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path, *, required: bool = True) -> str:
    return student_formalizer.read_text_file(path, required=required)


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    return student_formalizer.read_yaml_file(path, required=required)


def write_yaml_file(path: Path, data: Any) -> None:
    student_formalizer.write_yaml_file(path, data)


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    log_data["TeacherAnswerFormalizer"] = status
    if message:
        log_data["TeacherAnswerFormalizer_message"] = message
    elif "TeacherAnswerFormalizer_message" in log_data:
        del log_data["TeacherAnswerFormalizer_message"]
    write_yaml_file(LOG_PATH, log_data)


def build_system_prompt() -> str:
    return """
Bạn là một bộ formalize lời giải chuẩn của giáo viên thành kế hoạch tính toán symbolic.

Bạn nhận vào:
- Đề bài gốc
- ProblemEntities.yaml đã formalize từ đề bài
- Lời giải chuẩn của giáo viên

Nhiệm vụ chính:
1. Tạo TeacherPlan.yaml mô tả chính xác các bước giáo viên đã làm.
2. Lời giải giáo viên là reference để so sánh với bài học sinh, nên không tự tạo lời giải khác.
3. Không sửa thứ tự các phép tính giáo viên đã viết.
4. reported_expr là biểu thức số học giáo viên thực sự viết hoặc ngụ ý trong lời giải, có dấu '='.
5. expr là quan hệ symbolic giữa các entity.
6. Khi cần một số từ đề bài, dùng đúng entity trong ProblemEntities, không viết số literal trong expr.
7. Nếu giáo viên viết phép tính bằng số, reported_expr phải giữ phép tính số đó.
8. Được tạo entity trung gian làm result.
9. Step cuối phải tạo target của đề bài.

Schema output bắt buộc là YAML thuần có đúng 1 key cấp cao:
TeacherPlan.yaml:
  step1:
    expr: entity_a + entity_b
    result: intermediate_entity
    result_unit: dollars
    result_grand_unit: dollars
    reported_expr: 3.00 + 2.50 = 5.50
  step2:
    expr: intermediate_entity * days
    result: total_cost
    result_unit: dollars
    result_grand_unit: dollars
    reported_expr: 5.50 * 20 = 110.00
  target: total_cost

Mỗi step trong TeacherPlan.yaml có đúng 5 trường:
- expr
- result: entity được tạo ra, phải là một tên entity snake_case, không bao giờ là biểu thức.
- result_unit
- result_grand_unit
- reported_expr

Quy tắc step:
- Tên step phải là step1, step2, step3, ... liên tục.
- target nằm cuối TeacherPlan.yaml, cùng cấp với step1/step2.
- target phải là target của đề bài, không bao giờ là biểu thức.
- Không gộp nhiều phép tính giáo viên viết thành một step.
- Mỗi dòng/phần có dấu "=" trong lời giải giáo viên phải tạo đúng một step riêng, theo đúng thứ tự xuất hiện.
- reported_expr phải giữ đúng phép tính giáo viên viết ở dòng đó.
- expr phải tương ứng với chính reported_expr của step đó.
- Nếu một step sau dùng lại kết quả của phép tính trước, expr của step sau phải dùng entity result đã tạo,
  không được tự bung lại phép tính trước trong expr của step sau.
  Ví dụ nếu giáo viên đã viết `7 * 7 = 49` rồi sau đó viết `36 - 28 + 49 = 57`,
  phải có một step riêng tạo entity cho 49; step sau dùng entity đó, không viết lại `7 * 7`.
- Không viết biểu thức vào result. Ví dụ sai: result: remaining_after_second - closed_third.
  Đúng: expr: remaining_after_second - closed_third, result: remaining_tabs.
- Không được dùng cùng một result cho nhiều step.
- Không dùng target entity cho bước trung gian. Chỉ dùng target làm result cho bước thật sự tạo đáp án/kết luận.
- expr phải bao phủ toàn bộ vế trái của reported_expr. Ví dụ reported_expr là 30 * 2 + 50 = 110
  thì expr phải có entity tương ứng cho cả 30, 2 và 50, không được bỏ bớt toán hạng.
- Khi thay value của entity vào expr, giá trị expr phải đúng bằng vế trái của reported_expr.
  Nếu reported_expr là `7 * 100 = 700` thì expr phải biểu diễn đúng `7 * 100`, không được thêm chia/nhân
  khiến giá trị symbolic khác phép tính giáo viên viết.
- Không tạo bước chỉ copy một entity sang entity khác.
- Không tạo reported_expr dạng tautology như 110 = 110. Nếu step đã tính ra đáp án cuối,
  đặt result của step đó là target.

Quy trình tự kiểm tra nội bộ trước khi trả output, không được ghi phần này ra YAML:
1. Đọc Input/TeacherAnswer.txt theo từng dòng từ trên xuống dưới.
2. Tự lập danh sách các phép tính giáo viên viết hoặc ngụ ý rõ ràng, đặc biệt các dòng có dấu "=".
3. Kiểm tra mỗi phép tính trong danh sách đó có đúng một step tương ứng, cùng thứ tự, cùng reported_expr.
4. Kiểm tra không có phép tính nào bị gộp vào expr của step khác.
5. Kiểm tra result và target chỉ là tên entity snake_case, không phải biểu thức.
6. Kiểm tra step cuối tạo đúng target của đề bài.

Quy tắc output:
- Chỉ trả YAML thuần.
- Không Markdown.
- Không ```.
- Không giải thích.
""".strip()


def build_user_prompt(
    problem: str,
    problem_entities: Dict[str, Any],
    teacher_answer: str,
    calculation_trace: list[Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    problem_entities_yaml = yaml.safe_dump(
        problem_entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    trace_yaml = yaml.safe_dump(
        calculation_trace,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sửa lỗi trên bằng cách đọc lại trực tiếp Input/TeacherAnswer.txt. Không tạo
step ngoài lời giải, không bỏ step có phép tính rõ ràng, và không gộp phép tính
đã được viết riêng vào expr của step khác.
""".rstrip()

    return f"""
Hãy formalize lời giải chuẩn của giáo viên sau.

Input/Problem.txt:
{problem}

Output/ProblemEntities.yaml:
{problem_entities_yaml}

Input/TeacherAnswer.txt:
{teacher_answer}

CalculationTrace.yaml:
{trace_yaml}

Bắt buộc:
- Mỗi item trong CalculationTrace.yaml phải tạo đúng một step tương ứng trong TeacherPlan.yaml.
- Thứ tự step phải theo đúng thứ tự CalculationTrace.yaml.
- step.reported_expr phải khớp reported_expr trong trace về mặt phép tính.
- Không bỏ trace item dù giá trị đó được dùng lại ở step sau.
- Nếu cần thêm step ngụ ý để tạo target của đề bài, chỉ thêm khi lời giải thật sự kết luận đại lượng đó.
{retry_note}
""".strip()


def call_openrouter(
    problem: str,
    problem_entities: Dict[str, Any],
    teacher_answer: str,
    calculation_trace: list[Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise TeacherAnswerFormalizerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

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
                "content": build_user_prompt(
                    problem,
                    problem_entities,
                    teacher_answer,
                    calculation_trace,
                    previous_error=previous_error,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise TeacherAnswerFormalizerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise TeacherAnswerFormalizerError(
            f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}"
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise TeacherAnswerFormalizerError(
            f"Response OpenRouter không đúng định dạng: {response.text[:1000]}"
        ) from exc


def parse_llm_output(text: str) -> Dict[str, Any]:
    clean_text = student_formalizer.strip_markdown_fence(text)
    try:
        data = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise TeacherAnswerFormalizerError(f"LLM trả YAML không hợp lệ: {exc}") from exc

    if not isinstance(data, dict):
        raise TeacherAnswerFormalizerError("LLM output phải là dictionary.")

    raw_plan = data.get("TeacherPlan.yaml")
    if raw_plan is None:
        raw_plan = data.get("Plan.yaml")
    if raw_plan is None:
        raise TeacherAnswerFormalizerError("LLM output thiếu key TeacherPlan.yaml.")
    if not isinstance(raw_plan, dict):
        raise TeacherAnswerFormalizerError("TeacherPlan.yaml phải là dictionary.")

    return raw_plan


def max_retries() -> int:
    raw = os.getenv("TEACHER_FORMALIZER_MAX_RETRIES")
    if raw is None:
        raw = os.getenv("STUDENT_FORMALIZER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise TeacherAnswerFormalizerError("TEACHER_FORMALIZER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise TeacherAnswerFormalizerError("TEACHER_FORMALIZER_MAX_RETRIES phải >= 1.")
    return value


def target_entity_name(problem_entities: Dict[str, Dict[str, Any]]) -> str:
    targets = [name for name, entity in problem_entities.items() if entity.get("location") == "target"]
    if len(targets) != 1:
        raise TeacherAnswerFormalizerError(f"ProblemEntities phải có đúng 1 target, hiện có {len(targets)}.")
    return targets[0]


def unique_entity_for_decimal_value(
    value: Any,
    entities: Dict[str, Dict[str, Any]],
    *,
    locations: Optional[set[str]] = None,
) -> Optional[str]:
    matches = []
    for name, entity in entities.items():
        if locations is not None and entity.get("location") not in locations:
            continue
        try:
            entity_value = inside_checker.to_decimal(entity.get("value"), context=f"{name}.value")
        except inside_checker.InsideCheckerError:
            continue
        if inside_checker.decimal_equal(entity_value, value):
            matches.append(name)
    return matches[0] if len(matches) == 1 else None


def numeric_ast_value(node: ast.AST) -> Optional[Any]:
    try:
        return inside_checker.safe_eval_arithmetic(ast.unparse(node))
    except Exception:
        return None


def build_expr_from_reported_lhs(
    lhs: str,
    entities: Dict[str, Dict[str, Any]],
    prior_results_by_value: Dict[str, str],
) -> Optional[str]:
    try:
        tree = ast.parse(lhs.replace("×", "*").replace("÷", "/"), mode="eval")
    except SyntaxError:
        return None

    def entity_for_value(value: Any) -> Optional[str]:
        value_key = str(value.normalize()) if hasattr(value, "normalize") else str(value)
        if value_key in prior_results_by_value:
            return prior_results_by_value[value_key]

        return (
            unique_entity_for_decimal_value(value, entities, locations={"input"})
            or unique_entity_for_decimal_value(value, entities, locations={"target", "step1", "step2", "step3", "step4", "step5"})
        )

    class ReportedNumberMapper(ast.NodeTransformer):
        unresolved_numeric = False

        def replace_numeric_node(self, node: ast.AST) -> ast.AST:
            value = numeric_ast_value(node)
            if value is None:
                return self.generic_visit(node)
            entity_name = entity_for_value(value)
            if not entity_name:
                self.unresolved_numeric = True
                return node
            return ast.copy_location(ast.Name(id=entity_name, ctx=ast.Load()), node)

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
            if isinstance(node.op, ast.Div):
                value = numeric_ast_value(node)
                entity_name = entity_for_value(value) if value is not None else None
                if entity_name:
                    return ast.copy_location(ast.Name(id=entity_name, ctx=ast.Load()), node)
            return self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:  # noqa: N802
            if numeric_ast_value(node) is not None:
                return self.replace_numeric_node(node)
            return self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return node
            return self.replace_numeric_node(node)

    mapper = ReportedNumberMapper()
    transformed = mapper.visit(tree.body)
    if mapper.unresolved_numeric:
        return None
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def expr_matches_reported_lhs(
    expr: str,
    lhs: str,
    entities: Dict[str, Dict[str, Any]],
) -> bool:
    try:
        values = inside_checker.values_for_symbolic_expr(expr, entities)
        symbolic_value = inside_checker.safe_eval_arithmetic(expr, values)
        reported_lhs_value = inside_checker.safe_eval_arithmetic(lhs)
    except inside_checker.InsideCheckerError:
        return False
    return inside_checker.decimal_equal(symbolic_value, reported_lhs_value)


def repair_teacher_exprs_from_reported_lhs(
    plan: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    known_entities = {name: dict(entity) for name, entity in entities.items()}
    prior_results_by_value: Dict[str, str] = {}

    for step_name in student_formalizer.plan_step_names(plan):
        step = plan[step_name]
        if not isinstance(step, dict):
            continue

        reported_expr = student_formalizer.normalize_empty(step.get("reported_expr"))
        if not reported_expr:
            continue

        try:
            lhs, _ = inside_checker.split_reported_expr(str(reported_expr))
        except inside_checker.InsideCheckerError:
            continue

        current_expr = student_formalizer.normalize_empty(step.get("expr"))
        if not current_expr or not expr_matches_reported_lhs(str(current_expr), lhs, known_entities):
            rebuilt_expr = build_expr_from_reported_lhs(lhs, known_entities, prior_results_by_value)
            if rebuilt_expr and expr_matches_reported_lhs(rebuilt_expr, lhs, known_entities):
                step["expr"] = rebuilt_expr

        result = student_formalizer.normalize_empty(step.get("result"))
        if result:
            try:
                result_value = inside_checker.parse_reported_rhs_value(str(reported_expr))
            except inside_checker.InsideCheckerError:
                continue
            value_key = str(result_value.normalize())
            prior_results_by_value.setdefault(value_key, str(result))
            known_entities[str(result)] = {
                "value": int(result_value) if result_value == result_value.to_integral_value() else float(result_value),
                "unit": step.get("result_unit"),
                "location": step_name,
                "grand_unit": step.get("result_grand_unit"),
                "expr": step.get("expr"),
                "formalized_expr": step.get("expr"),
            }

    return plan


def validate_teacher_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    teacher_answer: str,
    calculation_trace: list[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], list[Dict[str, Any]]]:
    try:
        raw_plan = student_formalizer.prune_copy_steps_from_raw_plan(raw_plan)
        raw_plan = student_formalizer.align_raw_plan_to_calculation_trace(raw_plan, calculation_trace)
        plan = student_formalizer.validate_and_normalize_student_plan(
            raw_plan,
            problem_entities,
            student_answer=teacher_answer,
            plan_label="TeacherPlan",
        )
        plan, answer_literal_entities = student_formalizer.materialize_numeric_literals_in_plan(
            plan,
            problem_entities,
            prefix="teacher_answer",
            source_label="TeacherAnswer.txt",
        )
        plan, answer_literal_entities = student_formalizer.repair_plan_exprs_from_reported_lhs(
            plan,
            problem_entities,
            answer_literal_entities,
            prefix="teacher_answer",
            source_label="TeacherAnswer.txt",
        )
        plan = student_formalizer.validate_and_normalize_student_plan(
            plan,
            {**problem_entities, **answer_literal_entities},
            student_answer=teacher_answer,
            plan_label="TeacherPlan",
        )
        plan = repair_teacher_exprs_from_reported_lhs(
            plan,
            {**problem_entities, **answer_literal_entities},
        )
        plan, additional_literal_entities = student_formalizer.materialize_numeric_literals_in_plan(
            plan,
            {**problem_entities, **answer_literal_entities},
            prefix="teacher_answer",
            source_label="TeacherAnswer.txt",
        )
        answer_literal_entities.update(additional_literal_entities)
        plan = student_formalizer.validate_and_normalize_student_plan(
            plan,
            {**problem_entities, **answer_literal_entities},
            student_answer=teacher_answer,
            plan_label="TeacherPlan",
        )
        grounding_warnings = student_formalizer.validate_reported_expr_grounded_in_student_answer(
            plan,
            teacher_answer,
            answer_label="lời giải giáo viên",
        )
        student_formalizer.validate_plan_covers_calculation_trace(
            plan,
            calculation_trace,
            plan_label="TeacherPlan",
        )
        teacher_entities_for_validation = student_formalizer.merge_student_plan_into_entities(
            plan,
            problem_entities,
            extra_entities=answer_literal_entities,
        )
        validate_teacher_expr_matches_reported_expr(plan, teacher_entities_for_validation)
    except student_formalizer.StudentAnswerFormalizerError as exc:
        raise TeacherAnswerFormalizerError(str(exc)) from exc

    target_name = target_entity_name(problem_entities)
    if plan["target"] != target_name:
        raise TeacherAnswerFormalizerError(
            f"TeacherPlan target phải là target của đề bài {target_name!r}, hiện là {plan['target']!r}."
        )

    step_keys = [key for key in plan.keys() if key.startswith("step")]
    if not step_keys:
        raise TeacherAnswerFormalizerError("TeacherPlan.yaml phải có ít nhất một step.")
    if plan[step_keys[-1]]["result"] != target_name:
        raise TeacherAnswerFormalizerError(
            f"Step cuối của TeacherPlan phải tạo target {target_name!r}, "
            f"hiện tạo {plan[step_keys[-1]]['result']!r}."
        )

    return plan, answer_literal_entities, grounding_warnings


def validate_teacher_expr_matches_reported_expr(
    plan: Dict[str, Any],
    teacher_entities: Dict[str, Dict[str, Any]],
) -> None:
    for step_name in student_formalizer.plan_step_names(plan):
        step = plan[step_name]
        if not isinstance(step, dict):
            continue

        expr = student_formalizer.normalize_empty(step.get("expr"))
        reported_expr = student_formalizer.normalize_empty(step.get("reported_expr"))
        if not expr or not reported_expr:
            continue

        try:
            lhs, _ = inside_checker.split_reported_expr(str(reported_expr))
            entity_values = inside_checker.values_for_symbolic_expr(str(expr), teacher_entities)
            symbolic_value = inside_checker.safe_eval_arithmetic(str(expr), entity_values)
            reported_lhs_value = inside_checker.safe_eval_arithmetic(lhs)
            reported_rhs_value = inside_checker.parse_reported_rhs_value(str(reported_expr))
        except inside_checker.InsideCheckerError as exc:
            raise TeacherAnswerFormalizerError(
                f"{step_name}.expr hoặc reported_expr không kiểm tra được: {exc}"
            ) from exc

        if not inside_checker.decimal_equal(reported_lhs_value, reported_rhs_value):
            raise TeacherAnswerFormalizerError(
                f"{step_name}.reported_expr tính sai trong lời giải giáo viên: {reported_expr!r}."
            )

        if not inside_checker.decimal_equal(symbolic_value, reported_lhs_value):
            raise TeacherAnswerFormalizerError(
                f"{step_name}.expr không khớp reported_expr. "
                f"expr={expr!r} cho value {symbolic_value}, nhưng vế trái reported_expr "
                f"{lhs!r} cho value {reported_lhs_value}."
            )


def run() -> None:
    try:
        ensure_dirs()
        problem = read_text_file(PROBLEM_PATH, required=True)
        teacher_answer = read_text_file(TEACHER_ANSWER_PATH, required=True)
        if not teacher_answer:
            raise TeacherAnswerFormalizerError("Input/TeacherAnswer.txt đang rỗng.")

        raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=True)
        problem_entities = student_formalizer.validate_problem_entities(raw_problem_entities)
        calculation_trace = student_formalizer.extract_calculation_trace(
            problem,
            teacher_answer,
            answer_label="lời giải giáo viên",
            allowed_grounding_values=student_formalizer.trace_allowed_grounding_values(problem_entities),
        )
        write_yaml_file(TEACHER_TRACE_PATH, {"CalculationTrace.yaml": calculation_trace})

        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        teacher_plan: Optional[Dict[str, Any]] = None
        answer_literal_entities: Dict[str, Dict[str, Any]] = {}
        grounding_warnings: list[Dict[str, Any]] = []

        for _ in range(max_retries()):
            raw_response = call_openrouter(
                problem,
                problem_entities,
                teacher_answer,
                calculation_trace,
                previous_error=previous_error,
            )

            try:
                raw_teacher_plan = parse_llm_output(raw_response)
                candidate_plan, candidate_literal_entities, candidate_grounding_warnings = validate_teacher_plan(
                    raw_teacher_plan,
                    problem_entities,
                    teacher_answer,
                    calculation_trace,
                )
            except TeacherAnswerFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            teacher_plan = candidate_plan
            answer_literal_entities = candidate_literal_entities
            grounding_warnings = candidate_grounding_warnings
            break

        if teacher_plan is None:
            try:
                raw_trace_plan = student_formalizer.build_trace_derived_student_plan(
                    calculation_trace,
                    problem_entities,
                )
                candidate_plan, candidate_literal_entities, candidate_grounding_warnings = validate_teacher_plan(
                    raw_trace_plan,
                    problem_entities,
                    teacher_answer,
                    calculation_trace,
                )
            except TeacherAnswerFormalizerError as trace_exc:
                raise TeacherAnswerFormalizerError(
                    f"{last_validation_error}; trace-derived fallback cũng fail: {trace_exc}"
                ) from trace_exc

            teacher_plan = candidate_plan
            answer_literal_entities = candidate_literal_entities
            grounding_warnings = candidate_grounding_warnings
            student_formalizer.write_log_items(
                "TeacherAnswerFormalizer_trace_derived_plan",
                [
                    {
                        "reason": (
                            "LLM TeacherPlan retries failed; built TeacherPlan "
                            "from grounded CalculationTrace."
                        ),
                        "previous_error": str(last_validation_error) if last_validation_error else None,
                    }
                ],
            )

        teacher_entities = student_formalizer.merge_student_plan_into_entities(
            teacher_plan,
            problem_entities,
            extra_entities=answer_literal_entities,
        )

        write_yaml_file(TEACHER_PLAN_PATH, teacher_plan)
        write_yaml_file(TEACHER_ANSWER_ENTITIES_PATH, teacher_entities)

        write_log("Pass TeacherAnswerFormalizer")
        student_formalizer.write_log_items("TeacherAnswerFormalizer_grounding_warnings", grounding_warnings)
        print("Pass TeacherAnswerFormalizer")
    except Exception as exc:
        write_log("Fail TeacherAnswerFormalizer", str(exc))
        print("Fail TeacherAnswerFormalizer")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
