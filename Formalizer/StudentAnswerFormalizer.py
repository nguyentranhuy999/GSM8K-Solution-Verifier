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

import ast
import json
import os
import re
import sys
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
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
WRONG_PATH = OUTPUT_DIR / "Wrong.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3


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


def read_yaml_any(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise StudentAnswerFormalizerError(f"Không tìm thấy file: {path}")
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StudentAnswerFormalizerError(f"File YAML không hợp lệ: {path} - {exc}") from exc


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
    log_data["StudentAnswerFormalizer"] = status
    if message:
        log_data["StudentAnswerFormalizer_message"] = message
    elif "StudentAnswerFormalizer_message" in log_data:
        del log_data["StudentAnswerFormalizer_message"]
    write_yaml_file(LOG_PATH, log_data)


def write_log_items(key: str, items: List[Dict[str, Any]]) -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    if items:
        log_data[key] = items
    else:
        log_data.pop(key, None)
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

Mỗi step trong StudentPlan.yaml có 5 trường bắt buộc:
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
- expr phải bao phủ toàn bộ vế trái của reported_expr. Ví dụ reported_expr là 30 * 2 + 50 = 110
  thì expr phải có entity tương ứng cho cả 30, 2 và 50, không được bỏ bớt toán hạng.
- Không tạo bước chỉ copy một entity sang entity khác.
- Không tạo reported_expr dạng tautology như 110 = 110. Nếu step trước đã tính ra đáp án cuối,
  đặt result của step đó là target thay vì thêm bước copy.

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


def build_user_prompt(
    problem: str,
    problem_entities: Dict[str, Any],
    student_answer: str,
    previous_error: Optional[str] = None,
) -> str:
    problem_entities_yaml = yaml.safe_dump(
        problem_entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    equation_hints = extract_equations_from_student_answer(student_answer)
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sửa lỗi trên. Những equation rõ ràng được liệt kê bên dưới sẽ được code
validator kiểm tra, nên không được bỏ hoặc gộp vào step khác.
""".rstrip()

    return f"""
Hãy formalize bài làm học sinh sau.

Input/Problem.txt:
{problem}

Output/ProblemEntities.yaml:
{problem_entities_yaml}

Input/StudentAnswer.txt:
{student_answer}

Detected explicit equations từ regex:
{yaml.safe_dump(equation_hints, allow_unicode=True, sort_keys=False).strip()}
{retry_note}
""".strip()


# -----------------------------------------------------------------------------
# OpenRouter
# -----------------------------------------------------------------------------

def call_openrouter(
    problem: str,
    problem_entities: Dict[str, Any],
    student_answer: str,
    previous_error: Optional[str] = None,
) -> str:
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
            {
                "role": "user",
                "content": build_user_prompt(
                    problem,
                    problem_entities,
                    student_answer,
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


def normalize_expr_text(expr: Any, result: str) -> Optional[str]:
    expr = normalize_empty(expr)
    if expr is None:
        return None
    if not isinstance(expr, str) or not expr.strip():
        raise StudentAnswerFormalizerError("expr phải là string không rỗng hoặc null.")

    expr_text = expr.strip()
    if "=" not in expr_text:
        return expr_text

    left, right = expr_text.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not right:
        raise StudentAnswerFormalizerError(f"expr assignment thiếu vế phải: {expr_text!r}")

    # LLM đôi khi trả `result = expression` trong field expr. Plan schema đã có
    # field result riêng, nên giữ RHS để downstream parse được.
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", left):
        return right

    raise StudentAnswerFormalizerError(f"expr không được chứa phương trình nhiều vế: {expr_text!r}")


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

    result = step["result"]
    reported_expr = step["reported_expr"]

    if not isinstance(result, str):
        raise StudentAnswerFormalizerError(f"{step_name}.result phải là string.")
    expr = normalize_expr_text(step["expr"], result)

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
        "expr": expr,
        "result": result.strip(),
        "result_unit": result_unit,
        "result_grand_unit": result_grand_unit,
        "reported_expr": reported_expr.strip(),
    }


SIMPLE_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def simple_reported_number_value(text: str) -> Optional[Decimal]:
    cleaned = normalize_arithmetic_text(text)
    cleaned = re.sub(r"[a-zA-Z_]+", "", cleaned)
    if not SIMPLE_NUMBER_RE.fullmatch(cleaned):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def reported_expr_is_tautology(reported_expr: str) -> bool:
    parts = [part.strip() for part in reported_expr.split("=")]
    if len(parts) < 2:
        return False

    for left, right in zip(parts, parts[1:]):
        left_value = simple_reported_number_value(left)
        right_value = simple_reported_number_value(right)
        if left_value is not None and right_value is not None and left_value == right_value:
            return True
    return False


def expr_is_single_entity_reference(expr: Optional[str]) -> bool:
    if expr is None:
        return False
    return bool(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", expr.strip()))


def reported_lhs_has_arithmetic_operator(reported_expr: str) -> bool:
    lhs = reported_expr.rsplit("=", 1)[0]
    normalized_lhs = normalize_arithmetic_text(lhs)
    if "*" in normalized_lhs or "/" in normalized_lhs or "+" in normalized_lhs:
        return True
    return bool(re.search(r"(?<!^)-", normalized_lhs))


def validate_step_is_not_copy_step(step_name: str, step: Dict[str, Any]) -> None:
    if reported_expr_is_tautology(step["reported_expr"]):
        raise StudentAnswerFormalizerError(
            f"{step_name}.reported_expr là tautology/copy step ({step['reported_expr']!r}). "
            "Không tạo bước kiểu 110 = 110; nếu step trước đã ra đáp án cuối thì đặt result của step đó là target."
        )

    if (
        expr_is_single_entity_reference(step["expr"])
        and step["expr"] != step["result"]
        and not reported_lhs_has_arithmetic_operator(step["reported_expr"])
    ):
        raise StudentAnswerFormalizerError(
            f"{step_name} chỉ copy entity {step['expr']!r} sang {step['result']!r}. "
            "Không tạo bước đổi tên/copy; hãy đặt result của phép tính thật là target nếu đó là đáp án cuối."
        )


def validate_and_normalize_student_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Any],
    student_answer: Optional[str] = None,
    *,
    plan_label: str = "StudentPlan",
) -> Dict[str, Any]:
    if not raw_plan:
        raise StudentAnswerFormalizerError(f"{plan_label}.yaml đang rỗng.")

    if "target" not in raw_plan:
        raise StudentAnswerFormalizerError(f"{plan_label}.yaml phải có dòng target ở cuối.")

    keys = list(raw_plan.keys())
    if keys[-1] != "target":
        raise StudentAnswerFormalizerError(f"Dòng target phải nằm cuối {plan_label}.yaml.")

    step_keys = keys[:-1]
    expected = [f"step{i}" for i in range(1, len(step_keys) + 1)]
    if step_keys != expected:
        raise StudentAnswerFormalizerError(f"Step phải liên tục {expected}, hiện là {step_keys}.")

    available_entities = set(problem_entities.keys())
    normalized: Dict[str, Any] = {}

    for step_name in step_keys:
        step = normalize_step_fields(step_name, raw_plan[step_name])
        unknown_tokens = [
            token
            for token in expr_tokens(step["expr"])
            if token not in available_entities
        ] if step["expr"] else []
        if unknown_tokens:
            raise StudentAnswerFormalizerError(
                f"{step_name}.expr dùng entity chưa tồn tại hoặc chưa được tạo: {unknown_tokens}"
            )
        validate_step_is_not_copy_step(step_name, step)
        available_entities.add(step["result"])
        normalized[step_name] = step

    target = raw_plan["target"]
    if not isinstance(target, str) or not target.strip():
        raise StudentAnswerFormalizerError("target phải là tên entity string không rỗng.")
    validate_entity_name(target.strip())

    # target có thể là result vừa tạo hoặc entity đã có trong đề.
    if target.strip() not in available_entities:
        raise StudentAnswerFormalizerError(
            f"target {target!r} không tồn tại trong ProblemEntities hoặc result của {plan_label}."
        )

    normalized["target"] = target.strip()
    if student_answer:
        apply_reported_expr_unit_hints(normalized, student_answer)
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


def reported_expr_grounding_warnings(
    student_plan: Dict[str, Any],
    student_answer: str,
    *,
    answer_label: str = "bài làm học sinh",
) -> List[Dict[str, Any]]:
    """
    Cảnh báo khi LLM đưa số không xuất hiện nguyên văn trong lời giải.

    Đây chỉ là soft guard. Nhiều lời giải dùng số bằng chữ, phần trăm, phân số
    hoặc bỏ qua số trung gian, nên không được dùng check này để dừng pipeline.
    """
    answer_values = decimal_values_from_text(student_answer)
    if not answer_values:
        return []

    warnings: List[Dict[str, Any]] = []

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
            warnings.append(
                {
                    "step": step_name,
                    "reported_expr": reported_expr,
                    "values": [str(value.normalize()) for value in ungrounded_values],
                    "source": answer_label,
                }
            )

    return warnings


def validate_reported_expr_grounded_in_student_answer(
    student_plan: Dict[str, Any],
    student_answer: str,
    *,
    answer_label: str = "bài làm học sinh",
) -> List[Dict[str, Any]]:
    return reported_expr_grounding_warnings(
        student_plan,
        student_answer,
        answer_label=answer_label,
    )


CURRENCY_SYMBOL_RE = r"[$€£¥]"


def normalize_arithmetic_text(text: str) -> str:
    text = text.lower()
    text = re.sub(CURRENCY_SYMBOL_RE, "", text)
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
    if len(factors) > 1 and all(factors) and not re.search(r"[+\-()]", lhs):
        lhs = "*".join(sorted(factors))
    return f"{lhs}={rhs}"


def equation_strings_from_line(line: str) -> List[str]:
    if "=" not in line:
        return []

    line = re.sub(CURRENCY_SYMBOL_RE, "", line)
    line = line.replace(",", "")
    line = re.sub(
        r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
        r"\1 * \2",
        line,
        flags=re.IGNORECASE,
    )
    line = line.replace("×", "*").replace("÷", "/")

    # Bỏ từ mô tả hoặc đơn vị xen giữa số, nhưng giữ cấu trúc số học.
    # Ví dụ: "Total = 30 notebooks + 80 pens = 110 items" -> "30 + 80 = 110".
    line = re.sub(r"[A-Za-z_]+", " ", line)
    line = re.sub(r"[^0-9.+\-*/()=\s]", " ", line)
    line = re.sub(r"\s+", " ", line)

    equations: List[str] = []
    pattern = re.compile(
        r"(?<![0-9.])"
        r"([-+]?\d(?:[0-9.\s()+\-*/]*?))"
        r"\s*=\s*"
        r"([-+]?\d+(?:\.\d+)?)"
        r"(?![0-9])"
    )
    for match in pattern.finditer(line):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()
        equation = f"{lhs} = {rhs}"
        equations.append(equation)
    return equations


def extract_equations_from_student_answer(student_answer: str) -> List[str]:
    equations: List[str] = []

    for line in student_answer.splitlines():
        for equation in equation_strings_from_line(line):
            equations.append(arithmetic_fingerprint(equation))

    return equations


def reported_expr_fingerprints(plan: Dict[str, Any]) -> List[str]:
    return [
        arithmetic_fingerprint(str(step.get("reported_expr", "")))
        for name, step in plan.items()
        if name.startswith("step") and isinstance(step, dict)
    ]


def validate_reported_exprs_include_explicit_equations(
    plan: Dict[str, Any],
    answer_text: str,
    *,
    answer_label: str = "bài làm học sinh",
) -> None:
    """
    Nếu lời giải có phép tính explicit dạng `a + b = c`, plan không được bỏ.

    Đây mềm hơn validator exact cũ: plan có thể có thêm step implied, nhưng mọi
    equation rõ ràng code extract được phải xuất hiện trong reported_expr theo
    đúng thứ tự. Nhờ vậy không bỏ sót step như `7 * 7 = 49`.
    """
    expected = extract_equations_from_student_answer(answer_text)
    if not expected:
        return

    reported = reported_expr_fingerprints(plan)
    reported_index = 0
    missing: List[str] = []

    for expected_expr in expected:
        found = False
        for candidate_index in range(reported_index, len(reported)):
            if reported[candidate_index] == expected_expr:
                found = True
                reported_index = candidate_index + 1
                break
        if not found:
            missing.append(expected_expr)

    if missing:
        raise StudentAnswerFormalizerError(
            f"StudentPlan bỏ hoặc gộp equation rõ ràng trong {answer_label}. "
            f"Expected explicit equations theo thứ tự: {expected}; "
            f"reported_expr hiện có: {reported}; thiếu: {missing}."
        )


def reported_expr_unit_hints_from_student_answer(student_answer: str) -> Dict[str, Tuple[str, str]]:
    hints: Dict[str, Tuple[str, str]] = {}
    for line in student_answer.splitlines():
        if not re.search(CURRENCY_SYMBOL_RE, line):
            continue
        for equation in equation_strings_from_line(line):
            hints[arithmetic_fingerprint(equation)] = ("dollars", "dollars")
    return hints


def apply_reported_expr_unit_hints(student_plan: Dict[str, Any], student_answer: str) -> None:
    hints = reported_expr_unit_hints_from_student_answer(student_answer)
    if not hints:
        return

    for step_name, step in student_plan.items():
        if not step_name.startswith("step"):
            continue
        hint = hints.get(arithmetic_fingerprint(str(step.get("reported_expr", ""))))
        if not hint:
            continue
        result_unit, result_grand_unit = hint
        step["result_unit"] = result_unit
        step["result_grand_unit"] = result_grand_unit


def max_retries() -> int:
    raw = os.getenv("STUDENT_FORMALIZER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise StudentAnswerFormalizerError("STUDENT_FORMALIZER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise StudentAnswerFormalizerError("STUDENT_FORMALIZER_MAX_RETRIES phải >= 1.")
    return value


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
            **(
                {"source": str(source).strip()}
                if (source := normalize_empty(entity.get("source"))) is not None
                else {}
            ),
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
    append_diagnosis_file(diagnosis)

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
# Answer-local numeric literals
# -----------------------------------------------------------------------------

def fraction_from_entity_value(value: Any) -> Optional[Fraction]:
    value = normalize_empty(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return Fraction(str(value).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def fraction_from_numeric_ast(node: ast.AST) -> Optional[Fraction]:
    if isinstance(node, ast.Expression):
        return fraction_from_numeric_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        try:
            return Fraction(str(node.value))
        except ValueError:
            return None

    if isinstance(node, ast.UnaryOp):
        operand = fraction_from_numeric_ast(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None

    if isinstance(node, ast.BinOp):
        left = fraction_from_numeric_ast(node.left)
        right = fraction_from_numeric_ast(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return None
            return left / right
        return None

    return None


def fraction_slug(value: Fraction) -> str:
    sign = "neg_" if value < 0 else ""
    value = abs(value)
    if value.denominator == 1:
        return f"{sign}{value.numerator}"
    return f"{sign}{value.numerator}_over_{value.denominator}"


def fraction_yaml_value(value: Fraction) -> int | str:
    if value.denominator == 1:
        return int(value.numerator)
    decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal_value.normalize(), "f")


def make_unique_entity_name(base: str, reserved: set[str]) -> str:
    name = base
    suffix = 2
    while name in reserved:
        name = f"{base}_{suffix}"
        suffix += 1
    reserved.add(name)
    return name


def unique_input_entity_for_value(
    value: Fraction,
    problem_entities: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    matches: List[str] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        entity_value = fraction_from_entity_value(entity.get("value"))
        if entity_value == value:
            matches.append(name)
    return matches[0] if len(matches) == 1 else None


def materialize_numeric_literals_in_plan(
    plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    *,
    prefix: str = "student_answer",
    source_label: str = "student answer",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Đổi numeric literal trong expr thành entity cục bộ của lời giải.

    ProblemEntities chỉ nên chứa số được cho trong đề. Nếu học sinh/giáo viên
    viết một hằng số phát sinh trong lời giải như `* 2`, hằng số đó thuộc
    answer-local context, nên được biểu diễn bằng entity riêng trong
    StudentAnswerEntities/TeacherAnswerEntities thay vì để literal trong expr.
    """
    updated_plan = deepcopy(plan)
    extra_entities: Dict[str, Dict[str, Any]] = {}
    reserved = set(problem_entities.keys())
    for step_name, step in updated_plan.items():
        if step_name.startswith("step") and isinstance(step, dict):
            result = normalize_empty(step.get("result"))
            if result:
                reserved.add(str(result))

    literal_names_by_value: Dict[Fraction, str] = {}

    def entity_for_literal(value: Fraction, raw_text: str) -> str:
        existing = unique_input_entity_for_value(value, problem_entities)
        if existing:
            return existing

        if value in literal_names_by_value:
            return literal_names_by_value[value]

        base = f"{prefix}_number_{fraction_slug(value)}"
        name = make_unique_entity_name(base, reserved)
        literal_names_by_value[value] = name
        extra_entities[name] = {
            "value": fraction_yaml_value(value),
            "unit": None,
            "location": "input",
            "grand_unit": None,
            "source": f"numeric literal {raw_text} from {source_label}",
            "source_type": "answer_literal",
            "expr": None,
            "formalized_expr": None,
        }
        return name

    class NumericLiteralTransformer(ast.NodeTransformer):
        def replace_if_numeric(self, node: ast.AST) -> ast.AST:
            value = fraction_from_numeric_ast(node)
            if value is None:
                return self.generic_visit(node)
            raw_text = ast.unparse(node)
            return ast.copy_location(ast.Name(id=entity_for_literal(value, raw_text), ctx=ast.Load()), node)

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
            return self.replace_if_numeric(node)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:  # noqa: N802
            return self.replace_if_numeric(node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            return self.replace_if_numeric(node)

    for step_name, step in updated_plan.items():
        if not step_name.startswith("step") or not isinstance(step, dict):
            continue
        expr = normalize_empty(step.get("expr"))
        if expr is None:
            continue
        normalized_expr = str(expr).replace("×", "*").replace("÷", "/")
        try:
            tree = ast.parse(normalized_expr, mode="eval")
        except SyntaxError as exc:
            raise StudentAnswerFormalizerError(f"{step_name}.expr không parse được: {expr!r}") from exc

        transformed_body = NumericLiteralTransformer().visit(tree.body)
        ast.fix_missing_locations(transformed_body)
        step["expr"] = ast.unparse(transformed_body)

    return updated_plan, extra_entities


# -----------------------------------------------------------------------------
# StudentAnswerEntities merge
# -----------------------------------------------------------------------------

def initial_student_entities(
    problem_entities: Dict[str, Dict[str, Any]],
    extra_entities: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    entities: Dict[str, Dict[str, Any]] = {}
    for name, entity in {**problem_entities, **(extra_entities or {})}.items():
        validate_entity_name(name)
        if not isinstance(entity, dict):
            raise StudentAnswerFormalizerError(f"Entity {name} trong StudentAnswerEntities phải là dictionary.")

        entities[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": normalize_empty(entity.get("location")),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
            **(
                {"source": str(source).strip()}
                if (source := normalize_empty(entity.get("source"))) is not None
                else {}
            ),
            **(
                {"source_type": str(source_type).strip()}
                if (source_type := normalize_empty(entity.get("source_type"))) is not None
                else {}
            ),
            "expr": normalize_empty(entity.get("expr")),
            "formalized_expr": normalize_empty(entity.get("formalized_expr")),
        }

    return entities


def merge_student_plan_into_entities(
    student_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    extra_entities: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    entities = initial_student_entities(problem_entities, extra_entities=extra_entities)

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
        f_expr = formalize_expr(expr, formalized_by_entity) if expr else None

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

        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        student_plan: Optional[Dict[str, Any]] = None
        answer_literal_entities: Dict[str, Dict[str, Any]] = {}
        grounding_warnings: List[Dict[str, Any]] = []
        diagnosis: List[Dict[str, Any]] = []

        for _ in range(max_retries()):
            raw_response = call_openrouter(
                problem,
                problem_entities,
                student_answer,
                previous_error=previous_error,
            )

            try:
                raw_student_plan, raw_diagnosis = parse_llm_output(raw_response)
                candidate_plan = validate_and_normalize_student_plan(
                    raw_student_plan,
                    problem_entities,
                    student_answer=student_answer,
                )
                candidate_plan, candidate_literal_entities = materialize_numeric_literals_in_plan(
                    candidate_plan,
                    problem_entities,
                    prefix="student_answer",
                    source_label="StudentAnswer.txt",
                )
                candidate_plan = validate_and_normalize_student_plan(
                    candidate_plan,
                    {**problem_entities, **candidate_literal_entities},
                    student_answer=student_answer,
                )
                candidate_grounding_warnings = validate_reported_expr_grounded_in_student_answer(
                    candidate_plan,
                    student_answer,
                )
                validate_reported_exprs_include_explicit_equations(
                    candidate_plan,
                    student_answer,
                )
            except StudentAnswerFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            student_plan = candidate_plan
            answer_literal_entities = candidate_literal_entities
            grounding_warnings = candidate_grounding_warnings
            diagnosis = normalize_diagnosis(raw_diagnosis)
            break

        if student_plan is None:
            raise StudentAnswerFormalizerError(str(last_validation_error))

        write_yaml_file(STUDENT_PLAN_PATH, student_plan)
        write_diagnosis_and_wrong(diagnosis)

        student_entities = merge_student_plan_into_entities(
            student_plan,
            problem_entities,
            extra_entities=answer_literal_entities,
        )
        write_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, student_entities)

        write_log("Pass StudentAnswerFormalizer")
        write_log_items("StudentAnswerFormalizer_grounding_warnings", grounding_warnings)
        print("Pass StudentAnswerFormalizer")
    except Exception as exc:
        write_log("Fail StudentAnswerFormalizer", str(exc))
        print("Fail StudentAnswerFormalizer")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
