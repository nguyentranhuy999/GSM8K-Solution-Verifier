"""
Formalizer/StudentAnswerFormalizer.py

Nhiệm vụ:
- Đọc:
  - Input/Problem.txt
  - Input/StudentAnswer.txt
  - Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để formalize bài làm học sinh thành Output/StudentPlan.yaml.
- StudentPlan.yaml gồm các step theo thứ tự và dòng target ở cuối.
- Mỗi step gồm:
  - expr
  - result
  - result_unit
  - result_grand_unit
  - reported_expr
- Nếu LLM phát hiện lỗi chính tả / bài trả lời tối nghĩa / trả lời bằng chữ,
  ghi nhãn vào Output/Diagnosis.yaml và ghi No vào Output/Wrong.yaml.
- Sau khi có StudentPlan.yaml, dùng Python để update Output/StudentAnswerEntities.yaml:
  - thêm result entity từ các step
  - value lấy từ giá trị sau dấu '=' trong reported_expr
  - unit lấy từ result_unit
  - location là step tạo ra entity, trừ target thì giữ location target nếu đã có
  - grand_unit lấy từ result_grand_unit
  - expr lấy từ step.expr
  - formalized_expr là expr đã bung về các input entity
- Ghi Pass StudentAnswerFormalizer / Fail StudentAnswerFormalizer vào Output/Log.yaml.

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional

Ghi chú:
- File này không tự sửa phép tính của học sinh. Nếu học sinh viết sai 2.50 thành 2.00
  thì reported_expr vẫn giữ 2.00 như lời giải học sinh.
- expr là quan hệ symbolic giữa entity, còn reported_expr là phép tính học sinh thực sự báo cáo.
"""

from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

PROBLEM_PATH = INPUT_DIR / "Problem.txt"
STUDENT_ANSWER_PATH = INPUT_DIR / "StudentAnswer.txt"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
STUDENT_PLAN_PATH = OUTPUT_DIR / "StudentPlan.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"
DIAGNOSIS_PATH = OUTPUT_DIR / "Diagnosis.yaml"
# Một số code cũ có thể đang dùng nhầm Diagonosis.yaml; file chính ở đây là Diagnosis.yaml.
LEGACY_DIAGONOSIS_PATH = OUTPUT_DIR / "Diagonosis.yaml"
WRONG_PATH = OUTPUT_DIR / "Wrong.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class StudentAnswerFormalizerError(Exception):
    """Lỗi riêng cho StudentAnswerFormalizer."""


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def read_text_file(path: Path, *, required: bool = True) -> str:
    if not path.exists():
        if required:
            raise StudentAnswerFormalizerError(f"Không tìm thấy file: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise StudentAnswerFormalizerError(f"Không tìm thấy file: {path}")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StudentAnswerFormalizerError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise StudentAnswerFormalizerError(f"File YAML phải là dictionary: {path}")
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
    log_data["StudentAnswerFormalizer"] = status
    if message:
        log_data["StudentAnswerFormalizer_message"] = message
    elif "StudentAnswerFormalizer_message" in log_data:
        del log_data["StudentAnswerFormalizer_message"]
    write_yaml_file(LOG_PATH, log_data)


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

def build_system_prompt() -> str:
    return """
Bạn là một bộ formalize bài làm học sinh thành kế hoạch tính toán symbolic.

Bạn nhận vào:
- Đề bài gốc
- ProblemEntities.yaml đã formalize từ đề bài
- Bài làm học sinh

Nhiệm vụ chính:
1. Tạo StudentPlan.yaml mô tả chính xác các bước học sinh đã làm.
2. Không sửa lỗi tính toán của học sinh.
3. Không sửa giá trị học sinh dùng sai trong reported_expr.
4. expr là quan hệ symbolic giữa các entity.
5. reported_expr là biểu thức số học học sinh thực sự viết hoặc ngụ ý trong lời giải, có dấu '='.
6. Nếu học sinh dùng sai số từ đề bài, vẫn map expr theo entity đúng trong đề, nhưng reported_expr phải giữ số học sinh dùng.
   Ví dụ đề có afternoon_coffee_price = 2.50, học sinh viết 2.00:
   expr: morning_coffee_price + afternoon_coffee_price
   reported_expr: 3.00 + 2.00 = 5.00
7. Không tự thêm input entity mới nếu chỉ là học sinh chép sai số từ entity đã có.
8. Được tạo entity trung gian làm result.
9. Step cuối thường tạo target của đề bài, nhưng nếu học sinh kết luận bằng thực thể khác thì vẫn ghi target là thực thể học sinh chọn.

Schema output bắt buộc là YAML thuần có đúng 2 key cấp cao:
StudentPlan.yaml:
  step1:
    expr: entity_a + entity_b
    result: intermediate_entity
    result_unit: dollars
    result_grand_unit: dollars
    reported_expr: 3.00 + 2.00 = 5.00
  step2:
    expr: intermediate_entity * days
    result: total_cost
    result_unit: dollars
    result_grand_unit: dollars
    reported_expr: 5.00 * 20 = 100.00
  target: total_cost
Diagnosis.yaml:
  []

Mỗi step trong StudentPlan.yaml có đúng 5 trường:
- expr: biểu thức symbolic bằng tên entity.
- result: entity được tạo ra.
- result_unit: đơn vị result. Nếu học sinh quên đơn vị thì ghi missing. Nếu scalar thì có thể null.
- result_grand_unit: grand unit theo target. Scalar có thể null.
- reported_expr: phép tính số học học sinh báo cáo, phải có dấu '='. Giá trị sau '=' là value của result học sinh tính ra.

Quy tắc step:
- Tên step phải là step1, step2, step3, ... liên tục.
- target nằm cuối StudentPlan.yaml, cùng cấp với step1/step2.
- target là tên thực thể học sinh chọn ở phần kết luận cuối.
- Nếu học sinh kết luận final answer là 25.0 và step2 result là total_cost, target: total_cost.
- Nếu học sinh tính đúng target của đề nhưng kết luận nhầm sang entity khác, target phải là entity học sinh kết luận.
- Không gộp nhiều phép tính học sinh viết thành một step.
- Mỗi dòng/phần có dấu "=" trong bài làm học sinh phải tạo đúng một step riêng, theo đúng thứ tự xuất hiện.
- reported_expr phải giữ đúng phép tính học sinh viết ở dòng đó, không thay bằng phép tính tương đương hay phép tính tổng hợp.
- expr phải tương ứng với chính reported_expr của step đó, không được dùng expr của bước khác.

Quy tắc Diagnosis.yaml:
- Nếu bài làm có lỗi chính tả đáng kể, thêm diagnosis: spelling errors.
- Nếu bài trả lời tối nghĩa, thiếu dẫn giải, hoặc gần như chỉ ghi đáp án không rõ phép tính, thêm diagnosis: word problem.
- Nếu bài trả lời bằng chữ thay vì số, thêm diagnosis: answer by word.
- Mỗi diagnosis chỉ cần trường diagnosis. Không thêm step/entity; code Python sẽ tự thêm step và entity rỗng.
- Nếu không có nhãn nào, trả về [].

Ví dụ Diagnosis.yaml:
Diagnosis.yaml:
  - diagnosis: spelling errors
  - diagnosis: answer by word

Quy tắc đơn vị:
- Nếu học sinh viết đơn vị rõ ràng, dùng đơn vị đó.
- Nếu học sinh quên đơn vị, result_unit: missing.
- result_grand_unit vẫn cố gắng suy ra theo target nếu có thể.

Quy tắc output:
- Chỉ trả YAML thuần.
- Không Markdown.
- Không ```.
- Không giải thích.
""".strip()


def build_user_prompt(problem: str, problem_entities: Dict[str, Any], student_answer: str) -> str:
    problem_entities_yaml = yaml.safe_dump(
        problem_entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return f"""
Hãy formalize bài làm học sinh sau.

Input/Problem.txt:
{problem}

Output/ProblemEntities.yaml:
{problem_entities_yaml}

Input/StudentAnswer.txt:
{student_answer}
""".strip()


# -----------------------------------------------------------------------------
# OpenRouter
# -----------------------------------------------------------------------------

def call_openrouter(problem: str, problem_entities: Dict[str, Any], student_answer: str) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise StudentAnswerFormalizerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

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
            {"role": "user", "content": build_user_prompt(problem, problem_entities, student_answer)},
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise StudentAnswerFormalizerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise StudentAnswerFormalizerError(
            f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}"
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise StudentAnswerFormalizerError(
            f"Response OpenRouter không đúng định dạng: {response.text[:1000]}"
        ) from exc


# -----------------------------------------------------------------------------
# Parsing / validation
# -----------------------------------------------------------------------------

def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:yaml|yml|json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_llm_output(text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    clean_text = strip_markdown_fence(text)
    try:
        data = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise StudentAnswerFormalizerError(f"LLM trả YAML không hợp lệ: {exc}") from exc

    if not isinstance(data, dict):
        raise StudentAnswerFormalizerError("LLM output phải là dictionary.")

    if "StudentPlan.yaml" not in data:
        raise StudentAnswerFormalizerError("LLM output thiếu key StudentPlan.yaml.")

    raw_plan = data["StudentPlan.yaml"]
    if not isinstance(raw_plan, dict):
        raise StudentAnswerFormalizerError("StudentPlan.yaml phải là dictionary.")

    raw_diagnosis = data.get("Diagnosis.yaml", [])
    if raw_diagnosis is None:
        raw_diagnosis = []
    if not isinstance(raw_diagnosis, list):
        raise StudentAnswerFormalizerError("Diagnosis.yaml phải là list.")

    return raw_plan, raw_diagnosis


def validate_entity_name(name: str) -> None:
    if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise StudentAnswerFormalizerError(f"Tên entity không hợp lệ: {name!r}")


def expr_tokens(expr: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def normalize_step_fields(step_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise StudentAnswerFormalizerError(f"{step_name} phải là dictionary.")

    if "grand_result_unit" in step and "result_grand_unit" not in step:
        step["result_grand_unit"] = step.pop("grand_result_unit")

    required = {"expr", "result", "result_unit", "result_grand_unit", "reported_expr"}
    extra = set(step.keys()) - required
    missing = required - set(step.keys())

    if missing:
        raise StudentAnswerFormalizerError(f"{step_name} thiếu trường: {sorted(missing)}")
    if extra:
        raise StudentAnswerFormalizerError(f"{step_name} có trường thừa: {sorted(extra)}")

    expr = step["expr"]
    result = step["result"]
    reported_expr = step["reported_expr"]

    if not isinstance(expr, str) or not expr.strip():
        raise StudentAnswerFormalizerError(f"{step_name}.expr phải là string không rỗng.")
    if not isinstance(reported_expr, str) or "=" not in reported_expr:
        raise StudentAnswerFormalizerError(f"{step_name}.reported_expr phải là string có dấu '='.")

    validate_entity_name(result)

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
        "reported_expr": reported_expr.strip(),
    }


def validate_and_normalize_student_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Any],
) -> Dict[str, Any]:
    if not raw_plan:
        raise StudentAnswerFormalizerError("StudentPlan.yaml đang rỗng.")

    if "target" not in raw_plan:
        raise StudentAnswerFormalizerError("StudentPlan.yaml phải có dòng target ở cuối.")

    keys = list(raw_plan.keys())
    if keys[-1] != "target":
        raise StudentAnswerFormalizerError("Dòng target phải nằm cuối StudentPlan.yaml.")

    step_keys = keys[:-1]
    expected = [f"step{i}" for i in range(1, len(step_keys) + 1)]
    if step_keys != expected:
        raise StudentAnswerFormalizerError(f"Step phải liên tục {expected}, hiện là {step_keys}.")

    available_entities = set(problem_entities.keys())
    normalized: Dict[str, Any] = {}

    for step_name in step_keys:
        step = normalize_step_fields(step_name, raw_plan[step_name])
        unknown_tokens = [token for token in expr_tokens(step["expr"]) if token not in available_entities]
        if unknown_tokens:
            raise StudentAnswerFormalizerError(
                f"{step_name}.expr dùng entity chưa tồn tại hoặc chưa được tạo: {unknown_tokens}"
            )
        available_entities.add(step["result"])
        normalized[step_name] = step

    target = raw_plan["target"]
    if not isinstance(target, str) or not target.strip():
        raise StudentAnswerFormalizerError("target phải là tên entity string không rỗng.")
    validate_entity_name(target.strip())

    # target có thể là result vừa tạo hoặc entity đã có trong đề.
    if target.strip() not in available_entities:
        raise StudentAnswerFormalizerError(
            f"target {target!r} không tồn tại trong ProblemEntities hoặc result của StudentPlan."
        )

    normalized["target"] = target.strip()
    return normalized


def decimal_values_from_text(text: str) -> List[Decimal]:
    values: List[Decimal] = []
    for match in re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text):
        try:
            values.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    return values


def decimal_equal(left: Decimal, right: Decimal) -> bool:
    return left == right


def validate_reported_expr_grounded_in_student_answer(
    student_plan: Dict[str, Any],
    student_answer: str,
) -> None:
    """
    Không cho LLM tự dựng lời giải khác với bài làm học sinh.

    reported_expr là phép tính học sinh thực sự báo cáo, nên các số xuất hiện
    trong đó phải có dấu vết trong Input/StudentAnswer.txt. Guard này bắt các
    trường hợp LLM bỏ qua bài làm học sinh và tự giải lại đề bài.
    """
    answer_values = decimal_values_from_text(student_answer)
    if not answer_values:
        return

    for step_name, step in student_plan.items():
        if not step_name.startswith("step"):
            continue

        reported_expr = str(step.get("reported_expr", ""))
        reported_values = decimal_values_from_text(reported_expr)
        ungrounded_values = [
            value
            for value in reported_values
            if not any(decimal_equal(value, answer_value) for answer_value in answer_values)
        ]
        if ungrounded_values:
            formatted = ", ".join(str(value.normalize()) for value in ungrounded_values)
            raise StudentAnswerFormalizerError(
                f"{step_name}.reported_expr chứa số không có trong bài làm học sinh: {formatted}."
            )


def normalize_arithmetic_text(text: str) -> str:
    text = text.lower()
    text = re.sub(
        r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s*of\s*(\d+(?:\.\d+)?)",
        r"\1*\2",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


def arithmetic_fingerprint(text: str) -> str:
    text = normalize_arithmetic_text(text)
    if "=" not in text:
        return text

    lhs, rhs = text.split("=", 1)
    factors = lhs.split("*")
    if len(factors) > 1 and all(factors):
        lhs = "*".join(sorted(factors))
    return f"{lhs}={rhs}"


def extract_equations_from_student_answer(student_answer: str) -> List[str]:
    equations: List[str] = []

    for line in student_answer.splitlines():
        if "=" not in line:
            continue

        line = re.sub(
            r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
            r"\1 * \2",
            line,
            flags=re.IGNORECASE,
        )

        # Giữ phần phép tính quanh dấu "=" và bỏ chữ/đơn vị ở hai bên.
        allowed = r"0-9.,+\-*/×÷()\s"
        match = re.search(rf"([{allowed}]+=[{allowed}]+)", line)
        if not match:
            continue

        equation = match.group(1).strip()
        equation = equation.rstrip(".。;, ")
        equations.append(arithmetic_fingerprint(equation))

    return equations


def validate_reported_exprs_follow_student_answer(
    student_plan: Dict[str, Any],
    student_answer: str,
) -> None:
    expected_equations = extract_equations_from_student_answer(student_answer)
    if not expected_equations:
        return

    reported_equations = [
        arithmetic_fingerprint(str(step.get("reported_expr", "")))
        for name, step in student_plan.items()
        if name.startswith("step")
    ]

    if reported_equations != expected_equations:
        raise StudentAnswerFormalizerError(
            "StudentPlan không khớp các phép tính học sinh viết. "
            f"Expected reported_expr theo thứ tự: {expected_equations}; "
            f"hiện là: {reported_equations}."
        )


def validate_problem_entities(raw_entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    required = {"value", "unit", "location", "grand_unit"}
    if not raw_entities:
        raise StudentAnswerFormalizerError("Output/ProblemEntities.yaml đang rỗng.")

    normalized: Dict[str, Dict[str, Any]] = {}
    for name, entity in raw_entities.items():
        validate_entity_name(name)
        if not isinstance(entity, dict):
            raise StudentAnswerFormalizerError(f"Entity {name} phải là dictionary.")
        missing = required - set(entity.keys())
        if missing:
            raise StudentAnswerFormalizerError(f"Entity {name} thiếu trường: {sorted(missing)}")
        normalized[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }
    return normalized


# -----------------------------------------------------------------------------
# Diagnosis / Wrong
# -----------------------------------------------------------------------------

ALLOWED_DIAGNOSIS = {"spelling errors", "word problem", "answer by word"}


def normalize_diagnosis(raw_diagnosis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()

    for item in raw_diagnosis:
        if isinstance(item, str):
            label = item.strip()
        elif isinstance(item, dict):
            label = str(item.get("diagnosis", "")).strip()
        else:
            continue

        if label not in ALLOWED_DIAGNOSIS:
            continue
        if label in seen:
            continue
        seen.add(label)

        # User yêu cầu step và entity rỗng do Python gán.
        normalized.append({
            "diagnosis": label,
            "step": None,
            "entity": None,
        })

    return normalized


def write_diagnosis_and_wrong(diagnosis: List[Dict[str, Any]]) -> None:
    write_yaml_file(DIAGNOSIS_PATH, diagnosis)
    # Ghi thêm file legacy do user từng viết Diagonosis.yaml, để tránh module khác tìm tên cũ bị lỗi.
    write_yaml_file(LEGACY_DIAGONOSIS_PATH, diagnosis)

    if diagnosis:
        WRONG_PATH.write_text("No\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# reported_expr value parser
# -----------------------------------------------------------------------------

def parse_decimal_from_text(text: str) -> Decimal:
    cleaned = text.strip()
    cleaned = cleaned.rstrip(".。;, ")
    cleaned = cleaned.replace(",", "")

    # Lấy số cuối cùng sau dấu '='. Hỗ trợ $25.00, 25.0 dollars, etc.
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not matches:
        raise StudentAnswerFormalizerError(f"Không tìm thấy số trong phần sau dấu '=': {text!r}")

    try:
        return Decimal(matches[-1])
    except InvalidOperation as exc:
        raise StudentAnswerFormalizerError(f"Không parse được số: {matches[-1]!r}") from exc


def value_from_reported_expr(reported_expr: str) -> int | float:
    if "=" not in reported_expr:
        raise StudentAnswerFormalizerError(f"reported_expr thiếu dấu '=': {reported_expr!r}")
    rhs = reported_expr.split("=")[-1]
    value = parse_decimal_from_text(rhs)
    if value == value.to_integral_value():
        return int(value)
    return float(value)


# -----------------------------------------------------------------------------
# formalized_expr builder
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
    result = expr
    for token in sorted(set(expr_tokens(expr)), key=len, reverse=True):
        replacement = formalized_by_entity.get(token)
        if replacement:
            replacement = parenthesize_if_needed(replacement)
            result = re.sub(rf"\b{re.escape(token)}\b", replacement, result)
    return result


# -----------------------------------------------------------------------------
# StudentAnswerEntities merge
# -----------------------------------------------------------------------------

def initial_student_entities(problem_entities: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    existing = read_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, required=False)
    if existing:
        source = existing
    else:
        source = problem_entities

    entities: Dict[str, Dict[str, Any]] = {}
    for name, entity in source.items():
        validate_entity_name(name)
        if not isinstance(entity, dict):
            raise StudentAnswerFormalizerError(f"Entity {name} trong StudentAnswerEntities phải là dictionary.")

        entities[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
            "expr": normalize_empty(entity.get("expr")),
            "formalized_expr": normalize_empty(entity.get("formalized_expr")),
        }

    return entities


def merge_student_plan_into_entities(
    student_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    entities = initial_student_entities(problem_entities)

    formalized_by_entity: Dict[str, Optional[str]] = {}
    for name, entity in entities.items():
        if entity.get("location") == "input":
            entity["expr"] = None
            entity["formalized_expr"] = None
            formalized_by_entity[name] = None

    step_names = [key for key in student_plan.keys() if key.startswith("step")]
    for step_name in step_names:
        step = student_plan[step_name]
        result = step["result"]
        result_value = value_from_reported_expr(step["reported_expr"])
        result_unit = normalize_empty(step.get("result_unit"))
        result_grand_unit = normalize_empty(step.get("result_grand_unit"))
        expr = step["expr"]
        f_expr = formalize_expr(expr, formalized_by_entity)

        if result not in entities:
            entities[result] = {
                "value": result_value,
                "unit": result_unit,
                "location": step_name,
                "grand_unit": result_grand_unit,
                "expr": expr,
                "formalized_expr": f_expr,
            }
        else:
            entities[result]["value"] = result_value
            entities[result]["unit"] = result_unit
            entities[result]["grand_unit"] = result_grand_unit
            entities[result]["expr"] = expr
            entities[result]["formalized_expr"] = f_expr

            # Nếu entity là target từ đề bài thì giữ location: target theo one-shot.
            if entities[result].get("location") != "target":
                entities[result]["location"] = step_name

        formalized_by_entity[result] = f_expr

    # Đảm bảo mọi entity đều có expr/formalized_expr.
    for name, entity in entities.items():
        if entity.get("location") == "input":
            entity["expr"] = None
            entity["formalized_expr"] = None
        else:
            entity.setdefault("expr", None)
            entity.setdefault("formalized_expr", None)

    return entities


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run() -> None:
    try:
        ensure_dirs()
        problem = read_text_file(PROBLEM_PATH, required=True)
        student_answer = read_text_file(STUDENT_ANSWER_PATH, required=True)
        if not student_answer:
            raise StudentAnswerFormalizerError("Input/StudentAnswer.txt đang rỗng.")

        raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=True)
        problem_entities = validate_problem_entities(raw_problem_entities)

        raw_response = call_openrouter(problem, problem_entities, student_answer)
        raw_student_plan, raw_diagnosis = parse_llm_output(raw_response)

        student_plan = validate_and_normalize_student_plan(raw_student_plan, problem_entities)
        validate_reported_expr_grounded_in_student_answer(student_plan, student_answer)
        validate_reported_exprs_follow_student_answer(student_plan, student_answer)
        diagnosis = normalize_diagnosis(raw_diagnosis)

        write_yaml_file(STUDENT_PLAN_PATH, student_plan)
        write_diagnosis_and_wrong(diagnosis)

        student_entities = merge_student_plan_into_entities(student_plan, problem_entities)
        write_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, student_entities)

        write_log("Pass StudentAnswerFormalizer")
        print("Pass StudentAnswerFormalizer")
    except Exception as exc:
        write_log("Fail StudentAnswerFormalizer", str(exc))
        print("Fail StudentAnswerFormalizer")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
