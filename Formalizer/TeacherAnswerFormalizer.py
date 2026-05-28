"""
Formalizer/TeacherAnswerFormalizer.py

Nhiệm vụ:
- Đọc:
  - Input/Problem.txt
  - Input/TeacherAnswer.txt
  - Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để formalize lời giải chuẩn của giáo viên.
- Ghi output debug:
  - Output/TeacherPlan.yaml
  - Output/TeacherAnswerEntities.yaml
- Đồng thời ghi reference side cho pipeline so sánh hiện có:
  - Output/Plan.yaml
  - Output/PlanEntities.yaml

Luồng này dùng khi muốn so sánh bài học sinh với lời giải giáo viên thay vì
lời giải do Solver tự lập.
"""

from __future__ import annotations

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


INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

PROBLEM_PATH = INPUT_DIR / "Problem.txt"
TEACHER_ANSWER_PATH = INPUT_DIR / "TeacherAnswer.txt"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"

PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
TEACHER_PLAN_PATH = OUTPUT_DIR / "TeacherPlan.yaml"
TEACHER_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "TeacherAnswerEntities.yaml"
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
- result
- result_unit
- result_grand_unit
- reported_expr

Quy tắc step:
- Tên step phải là step1, step2, step3, ... liên tục.
- target nằm cuối TeacherPlan.yaml, cùng cấp với step1/step2.
- target phải là target của đề bài.
- Không gộp nhiều phép tính giáo viên viết thành một step.
- Mỗi dòng/phần có dấu "=" trong lời giải giáo viên phải tạo đúng một step riêng, theo đúng thứ tự xuất hiện.
- reported_expr phải giữ đúng phép tính giáo viên viết ở dòng đó.
- expr phải tương ứng với chính reported_expr của step đó.
- Không tạo bước chỉ copy một entity sang entity khác. Nếu step đã tính ra đáp án cuối, đặt result của step đó là target.

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
    previous_error: Optional[str] = None,
) -> str:
    problem_entities_yaml = yaml.safe_dump(
        problem_entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    required_equations = student_formalizer.extract_equations_from_student_answer(teacher_answer)
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sửa bằng cách tạo đủ step cho TẤT CẢ required reported_expr dưới đây, không bỏ dòng nào.
""".rstrip()

    return f"""
Hãy formalize lời giải chuẩn của giáo viên sau.

Input/Problem.txt:
{problem}

Output/ProblemEntities.yaml:
{problem_entities_yaml}

Input/TeacherAnswer.txt:
{teacher_answer}

Required reported_expr theo đúng thứ tự, mỗi dòng là một step riêng:
{yaml.safe_dump(required_equations, allow_unicode=True, sort_keys=False).strip()}
{retry_note}
""".strip()


def call_openrouter(
    problem: str,
    problem_entities: Dict[str, Any],
    teacher_answer: str,
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


def validate_teacher_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    teacher_answer: str,
) -> Dict[str, Any]:
    try:
        plan = student_formalizer.validate_and_normalize_student_plan(
            raw_plan,
            problem_entities,
            student_answer=teacher_answer,
        )
        student_formalizer.validate_reported_expr_grounded_in_student_answer(plan, teacher_answer)
        student_formalizer.validate_reported_exprs_follow_student_answer(plan, teacher_answer)
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

    return plan


def reference_plan_from_teacher_plan(teacher_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in teacher_plan.items()
        if key.startswith("step")
    }


def run() -> None:
    try:
        ensure_dirs()
        problem = read_text_file(PROBLEM_PATH, required=True)
        teacher_answer = read_text_file(TEACHER_ANSWER_PATH, required=True)
        if not teacher_answer:
            raise TeacherAnswerFormalizerError("Input/TeacherAnswer.txt đang rỗng.")

        raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=True)
        problem_entities = student_formalizer.validate_problem_entities(raw_problem_entities)

        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        teacher_plan: Optional[Dict[str, Any]] = None

        for _ in range(max_retries()):
            raw_response = call_openrouter(
                problem,
                problem_entities,
                teacher_answer,
                previous_error=previous_error,
            )

            try:
                raw_teacher_plan = parse_llm_output(raw_response)
                candidate_plan = validate_teacher_plan(raw_teacher_plan, problem_entities, teacher_answer)
            except TeacherAnswerFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            teacher_plan = candidate_plan
            break

        if teacher_plan is None:
            raise TeacherAnswerFormalizerError(str(last_validation_error))

        teacher_entities = student_formalizer.merge_student_plan_into_entities(teacher_plan, problem_entities)
        reference_plan = reference_plan_from_teacher_plan(teacher_plan)

        write_yaml_file(TEACHER_PLAN_PATH, teacher_plan)
        write_yaml_file(TEACHER_ANSWER_ENTITIES_PATH, teacher_entities)
        write_yaml_file(PLAN_PATH, reference_plan)
        write_yaml_file(PLAN_ENTITIES_PATH, teacher_entities)

        write_log("Pass TeacherAnswerFormalizer")
        print("Pass TeacherAnswerFormalizer")
    except Exception as exc:
        write_log("Fail TeacherAnswerFormalizer", str(exc))
        print("Fail TeacherAnswerFormalizer")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
