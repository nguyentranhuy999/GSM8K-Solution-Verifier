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
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

PROBLEM_PATH = INPUT_DIR / "Problem.txt"
STUDENT_ANSWER_PATH = INPUT_DIR / "StudentAnswer.txt"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
STUDENT_PLAN_PATH = OUTPUT_DIR / "StudentPlan.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"
STUDENT_TRACE_PATH = OUTPUT_DIR / "StudentTrace.yaml"
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
- result: entity được tạo ra, phải là một tên entity snake_case, không bao giờ là biểu thức.
- result_unit: đơn vị result. Nếu học sinh quên đơn vị thì ghi missing. Nếu scalar thì có thể null.
- result_grand_unit: grand unit theo target. Scalar có thể null.
- reported_expr: phép tính số học học sinh báo cáo, phải có dấu '='. Giá trị sau '=' là value của result học sinh tính ra.

Quy tắc step:
- Tên step phải là step1, step2, step3, ... liên tục.
- target nằm cuối StudentPlan.yaml, cùng cấp với step1/step2.
- target là tên thực thể học sinh chọn ở phần kết luận cuối, không bao giờ là biểu thức.
- Nếu học sinh kết luận final answer là 25.0 và step2 result là total_cost, target: total_cost.
- Nếu học sinh tính đúng target của đề nhưng kết luận nhầm sang entity khác, target phải là entity học sinh kết luận.
- Nếu lời kết luận chỉ dùng cách diễn đạt đồng nghĩa với target trong ProblemEntities, dùng đúng tên target có sẵn.
  Ví dụ đề hỏi `money_left` và học sinh viết "Kekai keeps ten dollars", step cuối phải result: money_left,
  target: money_left; không tạo alias money_kept.
- Không gộp nhiều phép tính học sinh viết thành một step.
- Mỗi dòng/phần có dấu "=" trong bài làm học sinh phải tạo đúng một step riêng, theo đúng thứ tự xuất hiện.
- reported_expr phải giữ đúng phép tính học sinh viết ở dòng đó, không thay bằng phép tính tương đương hay phép tính tổng hợp.
- expr phải tương ứng với chính reported_expr của step đó, không được dùng expr của bước khác.
- expr chỉ dùng tên entity trần, không truy cập field `.value`.
  Ví dụ đúng: `shirts * shirt_price`. Sai: `shirts.value * shirt_price.value`.
- Nếu một step sau dùng lại kết quả của phép tính trước, expr của step sau phải dùng entity result đã tạo,
  không được tự bung lại phép tính trước trong expr của step sau.
  Ví dụ nếu học sinh đã viết `7 * 7 = 49` rồi sau đó viết `36 - 28 + 49 = 57`,
  phải có một step riêng tạo entity cho 49; step sau dùng entity đó, không viết lại `7 * 7`.
- Không viết biểu thức vào result. Ví dụ sai: result: remaining_after_second - closed_third.
  Đúng: expr: remaining_after_second - closed_third, result: remaining_tabs.
- Không được dùng cùng một result cho nhiều step.
- Không dùng target entity cho bước trung gian. Chỉ dùng target làm result cho bước thật sự tạo đáp án/kết luận.
- expr phải bao phủ toàn bộ vế trái của reported_expr. Ví dụ reported_expr là 30 * 2 + 50 = 110
  thì expr phải có entity tương ứng cho cả 30, 2 và 50, không được bỏ bớt toán hạng.
- Không tạo bước chỉ copy một entity sang entity khác.
- Không tạo reported_expr dạng tautology như 110 = 110. Nếu step trước đã tính ra đáp án cuối,
  đặt result của step đó là target thay vì thêm bước copy.

Quy trình tự kiểm tra nội bộ trước khi trả output, không được ghi phần này ra YAML:
1. Đọc Input/StudentAnswer.txt theo từng dòng từ trên xuống dưới.
2. Tự lập danh sách các phép tính học sinh viết hoặc ngụ ý rõ ràng, đặc biệt các dòng có dấu "=".
3. Kiểm tra mỗi phép tính trong danh sách đó có đúng một step tương ứng, cùng thứ tự, cùng reported_expr.
4. Kiểm tra không có phép tính nào bị gộp vào expr của step khác.
5. Kiểm tra result và target chỉ là tên entity snake_case, không phải biểu thức.
6. Kiểm tra step cuối hoặc target phản ánh đúng phần học sinh kết luận cuối cùng.

Quy tắc Diagnosis.yaml:
- Nếu bài làm học sinh có lỗi chính tả đáng kể, thêm diagnosis: spelling errors.
- Không gán spelling errors vì lỗi chính tả trong đề bài.
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


def build_trace_system_prompt(answer_label: str) -> str:
    return f"""
Bạn là parser trace phép tính cho {answer_label}.

Nhiệm vụ:
- Chỉ đọc lời giải được cung cấp.
- Trích các phép tính theo đúng thứ tự xuất hiện.
- Không giải lại bài toán.
- Không sửa phép tính sai.
- Không map entity.
- Không gộp nhiều phép tính thành một item.
- Không bỏ phép tính có dấu "=".

Schema output bắt buộc là YAML thuần:
CalculationTrace.yaml:
  - source_text: "dòng hoặc cụm từ trong lời giải"
    reported_expr: "phép tính số học có dấu ="
    result_text: "mô tả ngắn đại lượng được tính"

Quy tắc:
- Mỗi dòng/cụm có phép tính rõ ràng tạo đúng một item.
- Nếu phép tính nằm trong ngoặc, dùng phép tính trong ngoặc làm reported_expr.
- Nếu lời giải viết "a of b = c", bắt buộc chuẩn hóa reported_expr thành "a * b = c".
- Nếu lời giải viết số bằng chữ, bắt buộc đổi number word sang chữ số trong reported_expr.
  Ví dụ: `five shirts ... one dollar ... five dollar` phải thành `"5 * 1 = 5"`;
  `Half of twenty dollar is ten dollar` phải thành `"20 / 2 = 10"`.
- reported_expr chỉ chứa phép tính số học dạng `vế trái = kết quả`, không ghi lại câu văn.
  Sai: `"five shirts will sell for five dollar"`.
  Đúng: `"5 * 1 = 5"`.
- Nếu lời giải chỉ nêu một dữ kiện đơn lẻ, ví dụ "He spent 7 tokens", không tạo item.
- Nếu lời giải có câu kết luận final answer nhưng không có phép tính mới, không tạo item riêng.
- Nếu lời giải ngụ ý một phép tính cuối rất rõ từ các giá trị vừa nêu, có thể tạo item cho phép tính đó.
- Không tạo hai item liền nhau cho cùng một phép tính khi câu sau chỉ viết rõ phép tính đã được câu trước mô tả.
  Ví dụ "he loses half of 50" rồi "Half of 50 is 25" chỉ tạo một item `"50 / 2 = 25"`.
- reported_expr phải giữ giá trị học sinh/giáo viên dùng, kể cả nếu sai.
- Nếu lời giải mô tả quan hệ rồi ghi một kết quả sai, reported_expr phải dùng kết quả sai đó.
  Ví dụ lời giải viết "A is 72. B is 2 more, so B is 70" thì reported_expr là "72 + 2 = 70",
  KHÔNG được sửa thành "72 + 2 = 74".
- Nếu step sau dùng lại kết quả sai trước đó, phải dùng chính kết quả sai mà lời giải đã báo.
  Ví dụ sau "B is 70", câu "C is 5 less than B, so C is 65" phải là "70 - 5 = 65",
  KHÔNG được dùng kết quả đúng tự tính như "74 - 5 = 69".
- Luôn quote string bằng dấu nháy kép, đặc biệt source_text có dấu ":".
- Nếu không có phép tính nào, trả về:
  CalculationTrace.yaml: []

Quy tắc output:
- Chỉ trả YAML thuần.
- Không Markdown.
- Không ```.
- Không giải thích.
""".strip()


def build_trace_user_prompt(problem: str, answer_text: str, answer_label: str, previous_error: Optional[str] = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trace trước bị reject vì lỗi:
{previous_error}

Hãy đọc lại lời giải và trả đúng schema CalculationTrace.yaml.
""".rstrip()

    return f"""
Hãy trích calculation trace từ {answer_label} sau.

Input/Problem.txt:
{problem}

Answer text:
{answer_text}
{retry_note}
""".strip()


def build_user_prompt(
    problem: str,
    problem_entities: Dict[str, Any],
    student_answer: str,
    calculation_trace: List[Dict[str, Any]],
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

Hãy sửa lỗi trên bằng cách đọc lại trực tiếp Input/StudentAnswer.txt. Không tạo
step ngoài lời giải, không bỏ step có phép tính rõ ràng, và không gộp phép tính
đã được viết riêng vào expr của step khác.
""".rstrip()

    return f"""
Hãy formalize bài làm học sinh sau.

Input/Problem.txt:
{problem}

Output/ProblemEntities.yaml:
{problem_entities_yaml}

Input/StudentAnswer.txt:
{student_answer}

CalculationTrace.yaml:
{trace_yaml}

Bắt buộc:
- Mỗi item trong CalculationTrace.yaml phải tạo đúng một step tương ứng trong StudentPlan.yaml.
- Thứ tự step phải theo đúng thứ tự CalculationTrace.yaml.
- step.reported_expr phải khớp reported_expr trong trace về mặt phép tính.
- Không bỏ trace item dù giá trị đó được dùng lại ở step sau.
- Nếu cần thêm step ngụ ý để biểu diễn kết luận cuối của học sinh, chỉ thêm khi kết luận đó thật sự có trong lời giải.
{retry_note}
""".strip()


# -----------------------------------------------------------------------------
# OpenRouter
# -----------------------------------------------------------------------------

def call_openrouter(
    problem: str,
    problem_entities: Dict[str, Any],
    student_answer: str,
    calculation_trace: List[Dict[str, Any]],
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


def call_trace_openrouter(
    problem: str,
    answer_text: str,
    *,
    answer_label: str,
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
            {"role": "system", "content": build_trace_system_prompt(answer_label)},
            {
                "role": "user",
                "content": build_trace_user_prompt(
                    problem,
                    answer_text,
                    answer_label,
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
        raise StudentAnswerFormalizerError(f"Không gọi được OpenRouter trace parser: {exc}") from exc

    if response.status_code >= 400:
        raise StudentAnswerFormalizerError(
            f"OpenRouter trace parser trả lỗi {response.status_code}: {response.text[:1000]}"
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise StudentAnswerFormalizerError(
            f"Response OpenRouter trace parser không đúng định dạng: {response.text[:1000]}"
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


def parse_trace_output(text: str) -> List[Dict[str, Any]]:
    clean_text = strip_markdown_fence(text)
    try:
        data = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        repaired_text = repair_unquoted_trace_yaml_scalars(clean_text)
        try:
            data = yaml.safe_load(repaired_text)
        except yaml.YAMLError:
            raise StudentAnswerFormalizerError(f"Trace parser trả YAML không hợp lệ: {exc}") from exc

    if isinstance(data, dict):
        raw_trace = data.get("CalculationTrace.yaml")
    else:
        raw_trace = data

    if raw_trace is None:
        raw_trace = []
    if not isinstance(raw_trace, list):
        raise StudentAnswerFormalizerError("CalculationTrace.yaml phải là list.")

    trace: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_trace, start=1):
        if not isinstance(item, dict):
            raise StudentAnswerFormalizerError(f"Trace item {idx} phải là dictionary.")
        reported_expr = item.get("reported_expr")
        if not isinstance(reported_expr, str) or "=" not in reported_expr:
            raise StudentAnswerFormalizerError(
                f"Trace item {idx}.reported_expr phải là phép tính số học có dấu '='; "
                "không trả câu văn. Nếu lời giải dùng number word, hãy đổi sang chữ số và toán tử."
            )
        reported_expr = normalize_reported_expr_text(reported_expr)
        if reported_expr_is_tautology(reported_expr):
            continue
        if not trace_reported_expr_is_numeric_arithmetic(reported_expr):
            raise StudentAnswerFormalizerError(
                f"Trace item {idx}.reported_expr chưa phải phép tính số học chuẩn: {reported_expr!r}. "
                "Hãy đổi number word sang chữ số và chỉ dùng toán tử +, -, *, / cùng dấu '='."
            )
        source_text = item.get("source_text")
        result_text = item.get("result_text")
        trace_item = {
            "source_text": str(source_text).strip() if source_text is not None else None,
            "reported_expr": reported_expr.strip(),
            "result_text": str(result_text).strip() if result_text is not None else None,
        }
        repair_trace_rhs_from_trace_text(trace_item)
        reported_expr = str(trace_item["reported_expr"])
        if trace and arithmetic_fingerprint(trace[-1]["reported_expr"]) == arithmetic_fingerprint(reported_expr):
            # LLM đôi khi tạo một item cho câu mô tả phép tính và thêm một item
            # giống hệt cho câu ngay sau đó viết phép tính rõ ràng. Giữ item sau.
            trace[-1] = trace_item
        else:
            trace.append(trace_item)

    return trace


def decimal_text_for_trace(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def repair_trace_unit_conversion_sentence(item: Dict[str, Any]) -> bool:
    source_text = str(normalize_empty(item.get("source_text")) or "").lower()
    source_text = source_text.replace(",", "")

    patterns = [
        (r"\b(\d+(?:\.\d+)?)\s+hours?\b.*\b(?:equals?|is|are|becomes?)\b.*\b(\d+(?:\.\d+)?)\s+minutes?\b", Decimal("60")),
        (r"\b(\d+(?:\.\d+)?)\s+feet\b.*\b(?:equals?|is|are|becomes?)\b.*\b(\d+(?:\.\d+)?)\s+inches?\b", Decimal("12")),
        (r"\b(\d+(?:\.\d+)?)\s+dozens?\b.*\b(?:equals?|is|are|becomes?)\b.*\b(\d+(?:\.\d+)?)\s+(?:items?|pieces?|oranges?)\b", Decimal("12")),
    ]
    for pattern, factor in patterns:
        match = re.search(pattern, source_text)
        if not match:
            continue
        left = Decimal(match.group(1))
        right = Decimal(match.group(2))
        item["reported_expr"] = normalize_reported_expr_text(
            f"{decimal_text_for_trace(left)} * {decimal_text_for_trace(factor)} = {decimal_text_for_trace(right)}"
        )
        return True

    return False


def rhs_value_from_reported_expr(reported_expr: str) -> Optional[Decimal]:
    if "=" not in reported_expr:
        return None
    _, rhs = reported_expr.rsplit("=", 1)
    values = decimal_values_from_text(rhs)
    return values[-1] if values else None


def decimal_value_mentioned_in_text(value: Decimal, text: str) -> bool:
    values = semantic_decimal_values_from_text(
        text,
        include_derived_group_values=False,
    )
    return decimal_value_present(value, values)


def decimal_value_mentioned_in_trace_text(value: Decimal, item: Dict[str, Any]) -> bool:
    trace_text = " ".join(
        str(normalize_empty(item.get(field)) or "")
        for field in ("source_text", "result_text")
    )
    return decimal_value_mentioned_in_text(value, trace_text)


def safe_eval_trace_numeric_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return safe_eval_trace_numeric_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        value = safe_eval_trace_numeric_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = safe_eval_trace_numeric_node(node.left)
        right = safe_eval_trace_numeric_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise StudentAnswerFormalizerError("Trace expression chia cho 0.")
            return left / right
    raise StudentAnswerFormalizerError(f"Trace expression không hỗ trợ node {type(node).__name__}.")


def safe_eval_trace_arithmetic(expr: str) -> Decimal:
    normalized = normalize_reported_expr_text(expr)
    tree = ast.parse(normalized, mode="eval")
    return safe_eval_trace_numeric_node(tree)


def repair_trace_rhs_from_trace_text(
    item: Dict[str, Any],
    *,
    answer_text: Optional[str] = None,
) -> bool:
    """
    Trace parser đôi khi lấy nhầm số mô tả như "half"/"one year"/"two years"
    làm kết quả sau dấu '='. Nếu vế trái tự tính được và kết quả đúng đó đã
    xuất hiện trong chính source/result text, sửa RHS về số được lời giải nêu.
    """
    reported_expr = normalize_empty(item.get("reported_expr"))
    if not isinstance(reported_expr, str) or "=" not in reported_expr:
        return False

    lhs, _rhs = reported_expr.rsplit("=", 1)
    try:
        lhs_value = safe_eval_trace_arithmetic(lhs)
    except Exception:
        return False

    rhs_value = rhs_value_from_reported_expr(reported_expr)
    if rhs_value is not None and decimal_value_present(lhs_value, [rhs_value]):
        return False
    lhs_value_in_trace = decimal_value_mentioned_in_trace_text(lhs_value, item)
    lhs_value_in_answer = bool(answer_text) and decimal_value_mentioned_in_text(lhs_value, str(answer_text))
    if not lhs_value_in_trace and not lhs_value_in_answer:
        return False

    item["reported_expr"] = normalize_reported_expr_text(
        f"{lhs.strip()} = {decimal_text_for_trace(lhs_value)}"
    )
    return True


def replace_first_ungrounded_lhs_value(
    lhs: str,
    replacement: Decimal,
    grounded_values: List[Decimal],
    allowed_values: List[Decimal],
) -> str:
    replacement_text = decimal_text_for_trace(replacement)
    replaced = False

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced:
            return match.group(0)
        try:
            value = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return match.group(0)
        if decimal_value_present(value, grounded_values) or decimal_value_present(value, allowed_values):
            return match.group(0)
        replaced = True
        return replacement_text

    return re.sub(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", repl, lhs, count=1)


def repair_trace_ungrounded_rhs(
    trace: List[Dict[str, Any]],
    *,
    answer_text: Optional[str] = None,
    allowed_values: Optional[List[Decimal]] = None,
) -> List[Dict[str, Any]]:
    repaired_trace = deepcopy(trace)
    allowed_values = allowed_values or []
    answer_values = (
        semantic_decimal_values_from_text(answer_text, include_derived_group_values=False)
        if answer_text
        else []
    )
    previous_rhs: Optional[Decimal] = None

    for item in repaired_trace:
        if repair_trace_unit_conversion_sentence(item):
            previous_rhs = rhs_value_from_reported_expr(str(item.get("reported_expr") or ""))
            continue

        repaired_rhs_from_text = repair_trace_rhs_from_trace_text(item, answer_text=answer_text)

        reported_expr = normalize_empty(item.get("reported_expr"))
        if not isinstance(reported_expr, str) or "=" not in reported_expr:
            continue

        lhs, rhs = reported_expr.rsplit("=", 1)

        if previous_rhs is not None and answer_values:
            repaired_lhs = replace_first_ungrounded_lhs_value(
                lhs,
                previous_rhs,
                answer_values,
                allowed_values,
            )
            if repaired_lhs != lhs:
                lhs = repaired_lhs
                item["reported_expr"] = normalize_reported_expr_text(f"{lhs.strip()} = {rhs.strip()}")

        rhs_values = decimal_values_from_text(rhs)
        if not rhs_values:
            previous_rhs = rhs_value_from_reported_expr(str(item.get("reported_expr") or ""))
            continue
        rhs_value = rhs_values[-1]
        if repaired_rhs_from_text:
            previous_rhs = rhs_value
            continue

        source_text = str(normalize_empty(item.get("source_text")) or "")
        source_values = semantic_decimal_values_from_text(
            source_text,
            include_derived_group_values=False,
        )
        if not source_values or decimal_value_present(rhs_value, source_values):
            continue

        lhs_values = semantic_decimal_values_from_text(
            lhs,
            include_derived_group_values=False,
        )
        candidates: List[Decimal] = []
        for value in source_values:
            if decimal_value_present(value, lhs_values):
                continue
            if decimal_value_present(value, candidates):
                continue
            candidates.append(value)

        if len(candidates) == 1:
            item["reported_expr"] = normalize_reported_expr_text(
                f"{lhs.strip()} = {decimal_text_for_trace(candidates[0])}"
            )
            rhs_value = candidates[0]

        previous_rhs = rhs_value

    return repaired_trace


def trace_reported_expr_is_numeric_arithmetic(reported_expr: str) -> bool:
    if "=" not in reported_expr:
        return False
    lhs, rhs = reported_expr.rsplit("=", 1)
    return (
        bool(re.search(r"\d", lhs))
        and bool(re.search(r"\d", rhs))
        and bool(re.search(r"[+\-*/]", lhs))
    )


def repair_unquoted_trace_yaml_scalars(text: str) -> str:
    """
    Trace parser đôi khi trả `source_text: Calculate X: 1 * 2 = 2`.
    YAML xem dấu `:` thứ hai là mapping mới. Với trace schema hẹp, có thể sửa
    an toàn bằng cách quote toàn bộ scalar của các field string.
    """
    repaired_lines: List[str] = []
    field_pattern = re.compile(r"^(\s*(?:-\s*)?(?:source_text|reported_expr|result_text):\s*)(.*)$")

    for line in text.splitlines():
        match = field_pattern.match(line)
        if not match:
            repaired_lines.append(line)
            continue

        prefix, value = match.groups()
        stripped = value.strip()
        if not stripped or stripped[0] in {'"', "'"} or stripped in {"null", "[]", "{}"}:
            repaired_lines.append(line)
            continue

        escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
        repaired_lines.append(f'{prefix}"{escaped}"')

    return "\n".join(repaired_lines)


def extract_calculation_trace(
    problem: str,
    answer_text: str,
    *,
    answer_label: str,
    allowed_grounding_values: Optional[List[Decimal]] = None,
) -> List[Dict[str, Any]]:
    previous_error: Optional[str] = None
    last_error: Optional[Exception] = None

    for _ in range(max_retries()):
        raw_response = call_trace_openrouter(
            problem,
            answer_text,
            answer_label=answer_label,
            previous_error=previous_error,
        )
        try:
            trace = parse_trace_output(raw_response)
            trace = repair_trace_ungrounded_rhs(
                trace,
                answer_text=answer_text,
                allowed_values=allowed_grounding_values,
            )
            validate_calculation_trace_grounded_in_answer(
                trace,
                answer_text,
                answer_label=answer_label,
                allowed_values=allowed_grounding_values,
            )
            return trace
        except StudentAnswerFormalizerError as exc:
            previous_error = str(exc)
            last_error = exc

    raise StudentAnswerFormalizerError(str(last_error))


def validate_entity_name(name: str) -> None:
    if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise StudentAnswerFormalizerError(f"Tên entity không hợp lệ: {name!r}")


def validate_result_entity_name(step_name: str, result: Any) -> None:
    if not isinstance(result, str):
        raise StudentAnswerFormalizerError(f"{step_name}.result phải là string.")
    if re.search(r"[+\-*/=()]", result) or re.search(r"\s", result.strip()):
        raise StudentAnswerFormalizerError(
            f"{step_name}.result đang là biểu thức hoặc chứa khoảng trắng: {result!r}. "
            "result phải là một tên entity snake_case. Đưa phép tính vào field expr, "
            "ví dụ expr: remaining_after_second - closed_third; result: remaining_tabs."
        )
    validate_entity_name(result)


def validate_named_entity_reference(field_label: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudentAnswerFormalizerError(f"{field_label} phải là tên entity string không rỗng.")
    name = value.strip()
    if re.search(r"[+\-*/=()]", name) or re.search(r"\s", name):
        raise StudentAnswerFormalizerError(
            f"{field_label} đang là biểu thức hoặc chứa khoảng trắng: {name!r}. "
            f"{field_label} phải là một tên entity snake_case, ví dụ remaining_tabs."
        )
    validate_entity_name(name)
    return name


def expr_tokens(expr: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def plan_step_names(plan: Dict[str, Any]) -> List[str]:
    def key_fn(name: str) -> int:
        match = re.fullmatch(r"step(\d+)", str(name))
        return int(match.group(1)) if match else 10**9

    return sorted(
        [key for key in plan.keys() if re.fullmatch(r"step\d+", str(key))],
        key=key_fn,
    )


def normalize_expr_text(expr: Any, result: str) -> Optional[str]:
    expr = normalize_empty(expr)
    if expr is None:
        return None
    if not isinstance(expr, str) or not expr.strip():
        raise StudentAnswerFormalizerError("expr phải là string không rỗng hoặc null.")

    expr_text = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.value\b", r"\1", expr.strip())
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


def normalize_reported_expr_text(reported_expr: str) -> str:
    text = reported_expr.strip()
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"[$€£¥]", "", text)
    text = text.replace(",", "")
    text = re.sub(r"(?<=\d)[.。;,]+(?=\s*$)", "", text)
    text = re.sub(
        r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
        r"\1 * \2",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


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

    validate_result_entity_name(step_name, result)
    expr = normalize_expr_text(step["expr"], result)

    if not isinstance(reported_expr, str) or "=" not in reported_expr:
        raise StudentAnswerFormalizerError(f"{step_name}.reported_expr phải là string có dấu '='.")

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
        "reported_expr": normalize_reported_expr_text(reported_expr),
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


def replace_expr_aliases(expr: Optional[str], aliases: Dict[str, str]) -> Optional[str]:
    if not expr or not aliases:
        return expr

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return aliases.get(token, token)

    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, expr)


def prune_copy_steps_from_raw_plan(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bỏ các step LLM tự tạo chỉ để copy/đổi tên entity, ví dụ `5 = 5`.

    Những step này không phải phép tính học sinh viết. Nếu result của copy step
    được dùng ở bước sau thì thay bằng entity nguồn để không làm đứt expr.
    """
    if not isinstance(raw_plan, dict) or "target" not in raw_plan:
        return raw_plan

    pruned: Dict[str, Any] = {}
    aliases: Dict[str, str] = {}
    kept_count = 0

    for step_name in [key for key in raw_plan if re.fullmatch(r"step\d+", str(key))]:
        raw_step = raw_plan.get(step_name)
        if not isinstance(raw_step, dict):
            kept_count += 1
            pruned[f"step{kept_count}"] = raw_step
            continue

        try:
            step = normalize_step_fields(step_name, dict(raw_step))
        except StudentAnswerFormalizerError:
            kept_count += 1
            pruned[f"step{kept_count}"] = raw_step
            continue

        step["expr"] = replace_expr_aliases(step.get("expr"), aliases)
        result = step["result"]
        expr = normalize_empty(step.get("expr"))
        is_copy = (
            expr_is_single_entity_reference(expr)
            and expr != result
            and (
                reported_expr_is_tautology(step["reported_expr"])
                or not reported_lhs_has_arithmetic_operator(step["reported_expr"])
            )
        )

        if is_copy:
            aliases[result] = str(expr)
            continue

        kept_count += 1
        pruned[f"step{kept_count}"] = step

    target = raw_plan.get("target")
    if isinstance(target, str):
        target = aliases.get(target, target)
    pruned["target"] = target
    return pruned


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
    created_results: set[str] = set()

    for step_name in step_keys:
        step = normalize_step_fields(step_name, raw_plan[step_name])
        result = step["result"]
        if result in created_results:
            raise StudentAnswerFormalizerError(
                f"{step_name}.result {result!r} đã được tạo ở step trước. "
                "Mỗi step phải tạo một result entity riêng; không ghi đè result cũ."
            )
        if result in problem_entities and problem_entities[result].get("location") == "input":
            raise StudentAnswerFormalizerError(
                f"{step_name}.result {result!r} là input entity từ ProblemEntities. "
                "Không được ghi đè input entity; hãy tạo result trung gian mới."
            )
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
        created_results.add(result)
        available_entities.add(step["result"])
        normalized[step_name] = step

    target = validate_named_entity_reference("target", raw_plan["target"])

    problem_targets = [
        name
        for name, entity in problem_entities.items()
        if isinstance(entity, dict) and entity.get("location") == "target"
    ]
    duplicate_problem_targets = [
        problem_target
        for problem_target in problem_targets
        if target != problem_target and (problem_target in target or target in problem_target)
    ]
    if duplicate_problem_targets:
        if len(duplicate_problem_targets) != 1:
            raise StudentAnswerFormalizerError(
                f"target {target!r} có vẻ là biến thể của nhiều target có sẵn {duplicate_problem_targets}; "
                "không thể tự canonicalize an toàn."
            )

        canonical_target = duplicate_problem_targets[0]
        steps = plan_step_names(normalized)
        final_step = normalized[steps[-1]] if steps else {}
        final_result = normalize_empty(final_step.get("result")) if isinstance(final_step, dict) else None

        if final_result == target:
            final_step["result"] = canonical_target
            created_results.discard(target)
            created_results.add(canonical_target)
            available_entities.add(canonical_target)
            target = canonical_target
        elif final_result == canonical_target:
            target = canonical_target
        else:
            raise StudentAnswerFormalizerError(
                f"target {target!r} có vẻ là biến thể của target có sẵn {duplicate_problem_targets}. "
                "Nếu lời giải kết luận đúng đại lượng được hỏi, dùng chính entity target trong ProblemEntities "
                "làm result/target của step cuối, không tạo target mới như *_value."
            )

    # Nếu lời giải chỉ có một target gốc và step cuối đang tạo một entity mới để
    # kết luận đáp án, canonicalize result đó về target gốc. Đây là lỗi đặt tên
    # của formalizer, không phải lỗi toán học của học sinh.
    if target not in problem_entities and len(problem_targets) == 1 and step_keys:
        canonical_target = problem_targets[0]
        canonical_target_entity = problem_entities.get(canonical_target, {})
        final_step_name = step_keys[-1]
        final_step = normalized.get(final_step_name)
        final_result = normalize_empty(final_step.get("result")) if isinstance(final_step, dict) else None
        target_is_final_result = target == final_result
        canonical_target_is_blank = (
            normalize_empty(canonical_target_entity.get("value")) is None
            and normalize_empty(canonical_target_entity.get("expr")) is None
        )
        if target_is_final_result and canonical_target_is_blank:
            final_step["result"] = canonical_target
            created_results.discard(target)
            created_results.add(canonical_target)
            available_entities.add(canonical_target)
            target = canonical_target

    # target có thể là result vừa tạo hoặc entity đã có trong đề.
    if target not in available_entities:
        raise StudentAnswerFormalizerError(
            f"target {target!r} không tồn tại trong ProblemEntities hoặc result của {plan_label}."
        )

    normalized["target"] = target
    validate_final_step_fills_existing_target(normalized, problem_entities, plan_label=plan_label)
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


NUMBER_WORD_VALUES = {
    "zero": Decimal("0"),
    "one": Decimal("1"),
    "two": Decimal("2"),
    "three": Decimal("3"),
    "four": Decimal("4"),
    "five": Decimal("5"),
    "six": Decimal("6"),
    "seven": Decimal("7"),
    "eight": Decimal("8"),
    "nine": Decimal("9"),
    "ten": Decimal("10"),
    "eleven": Decimal("11"),
    "twelve": Decimal("12"),
    "thirteen": Decimal("13"),
    "fourteen": Decimal("14"),
    "fifteen": Decimal("15"),
    "sixteen": Decimal("16"),
    "seventeen": Decimal("17"),
    "eighteen": Decimal("18"),
    "nineteen": Decimal("19"),
    "twenty": Decimal("20"),
}

FRACTION_WORD_VALUES = {
    "half": (Decimal("0.5"), Decimal("2")),
    "third": (Decimal("0.3333333333333333"), Decimal("3")),
    "quarter": (Decimal("0.25"), Decimal("4")),
    "fourth": (Decimal("0.25"), Decimal("4")),
}

MULTIPLIER_WORD_VALUES = {
    "twice": Decimal("2"),
    "double": Decimal("2"),
    "doubled": Decimal("2"),
    "thrice": Decimal("3"),
    "triple": Decimal("3"),
    "tripled": Decimal("3"),
}

GROUP_UNIT_WORD_VALUES = {
    "dozen": Decimal("12"),
    "dozens": Decimal("12"),
}


def semantic_decimal_values_from_text(
    text: str,
    *,
    include_derived_group_values: bool = True,
) -> List[Decimal]:
    values = decimal_values_from_text(text)
    lowered = str(text or "").lower()

    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?\s*%", lowered):
        try:
            values.append(Decimal(raw.replace("%", "").strip()) / Decimal("100"))
        except InvalidOperation:
            continue

    for word, value in NUMBER_WORD_VALUES.items():
        if re.search(rf"\b{word}\b", lowered):
            values.append(value)

    for word, word_values in FRACTION_WORD_VALUES.items():
        if re.search(rf"\b{word}s?\b", lowered):
            values.extend(word_values)

    for word, value in MULTIPLIER_WORD_VALUES.items():
        if re.search(rf"\b{word}\b", lowered):
            values.append(value)

    for word, value in GROUP_UNIT_WORD_VALUES.items():
        if re.search(rf"\b{word}\b", lowered):
            values.append(value)

    count_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORD_VALUES, key=len, reverse=True))
    for match in re.finditer(rf"\b({count_pattern})\s+(dozens?|pairs?)\b", lowered):
        raw_count = match.group(1)
        if raw_count in NUMBER_WORD_VALUES:
            count_value = NUMBER_WORD_VALUES[raw_count]
        else:
            try:
                count_value = Decimal(raw_count.replace(",", ""))
            except InvalidOperation:
                continue
        group_value = Decimal("12") if match.group(2).startswith("dozen") else Decimal("2")
        values.extend([count_value, group_value])
        if include_derived_group_values:
            values.append(count_value * group_value)

    return values


def decimal_value_present(value: Decimal, candidates: List[Decimal]) -> bool:
    for candidate in candidates:
        if abs(value - candidate) <= Decimal("0.000001"):
            return True
    return False


def trace_allowed_grounding_values(problem_entities: Dict[str, Dict[str, Any]]) -> List[Decimal]:
    values: List[Decimal] = []
    for name, entity in problem_entities.items():
        if not str(name).startswith("unit_conversion_"):
            continue
        value = normalize_empty(entity.get("value"))
        if value is None:
            continue
        try:
            values.append(Decimal(str(value).replace(",", "")))
        except (InvalidOperation, ValueError):
            continue
    return values


def validate_calculation_trace_grounded_in_answer(
    calculation_trace: List[Dict[str, Any]],
    answer_text: str,
    *,
    answer_label: str,
    allowed_values: Optional[List[Decimal]] = None,
) -> None:
    """
    Trace parser không được tự sinh phép tính dùng số không xuất hiện trong lời giải.

    Đây là guard chống lỗi missing-step bị che mất: nếu học sinh chỉ viết
    `24000 * 3 = 72000`, trace không được tự thêm `200 * 60 * 2 = 24000`
    chỉ vì đề bài có thể suy ra như vậy.
    """
    answer_values = semantic_decimal_values_from_text(
        answer_text,
        include_derived_group_values=False,
    )
    if not answer_values:
        return
    allowed_values = allowed_values or []

    errors: List[Dict[str, Any]] = []
    for idx, item in enumerate(calculation_trace, start=1):
        reported_expr = str(item.get("reported_expr") or "")
        reported_values = decimal_values_from_text(reported_expr)
        ungrounded = [
            value
            for value in reported_values
            if not decimal_value_present(value, answer_values)
            and not decimal_value_present(value, allowed_values)
        ]
        if ungrounded:
            errors.append(
                {
                    "trace_index": idx,
                    "reported_expr": reported_expr,
                    "ungrounded_values": [str(value.normalize()) for value in ungrounded],
                }
            )

    if errors:
        raise StudentAnswerFormalizerError(
            f"CalculationTrace.yaml có phép tính chứa số không xuất hiện trong {answer_label}: {errors}. "
            "Không tự suy bước còn thiếu từ đề bài; chỉ trace phép tính thật sự có trong lời giải."
        )


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
    text = re.sub(r"(?<=\d)\.(?=$|[^\d])", "", text)
    text = re.sub(
        r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s*of\s*(\d+(?:\.\d+)?)",
        r"\1*\2",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<![a-zA-Z_])[-+]?\d+(?:\.\d+)?", normalize_numeric_match_text, text)
    return text


def normalize_numeric_match_text(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    normalized = format(value.normalize(), "f")
    if normalized == "-0":
        return "0"
    return normalized


def arithmetic_fingerprint(text: str) -> str:
    text = normalize_arithmetic_text(text)
    if "=" not in text:
        return text

    lhs, rhs = text.split("=", 1)
    factors = lhs.split("*")
    if len(factors) > 1 and all(factors) and not re.search(r"[+\-()]", lhs):
        lhs = "*".join(sorted(factors))
    return f"{lhs}={rhs}"


def reported_expr_fingerprints(plan: Dict[str, Any]) -> List[str]:
    return [
        arithmetic_fingerprint(str(step.get("reported_expr", "")))
        for name, step in plan.items()
        if name.startswith("step") and isinstance(step, dict)
    ]


def align_raw_plan_to_calculation_trace(
    raw_plan: Dict[str, Any],
    calculation_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not calculation_trace or not isinstance(raw_plan, dict) or "target" not in raw_plan:
        return raw_plan

    expected = [
        arithmetic_fingerprint(str(item.get("reported_expr", "")))
        for item in calculation_trace
        if item.get("reported_expr")
    ]
    if not expected:
        return raw_plan

    step_keys = [key for key in raw_plan if re.fullmatch(r"step\d+", str(key))]
    step_fingerprints = [
        arithmetic_fingerprint(str(raw_plan[key].get("reported_expr", "")))
        if isinstance(raw_plan.get(key), dict)
        else ""
        for key in step_keys
    ]

    chosen_indexes: List[int] = []
    search_end = len(step_keys)
    for expected_expr in reversed(expected):
        found_index: Optional[int] = None
        for idx in range(search_end - 1, -1, -1):
            if idx in chosen_indexes:
                continue
            if step_fingerprints[idx] == expected_expr:
                found_index = idx
                break
        if found_index is None:
            return raw_plan
        chosen_indexes.append(found_index)
        search_end = found_index

    chosen_indexes = sorted(chosen_indexes)
    pruned_any = len(chosen_indexes) != len(step_keys)
    aligned: Dict[str, Any] = {}
    for new_idx, old_idx in enumerate(chosen_indexes, start=1):
        step = deepcopy(raw_plan[step_keys[old_idx]])
        if pruned_any and isinstance(step, dict):
            reported_expr = normalize_empty(step.get("reported_expr"))
            if isinstance(reported_expr, str) and "=" in reported_expr:
                lhs, _ = reported_expr.rsplit("=", 1)
                step["expr"] = normalize_reported_expr_text(lhs)
        aligned[f"step{new_idx}"] = step
    aligned["target"] = raw_plan.get("target")
    return aligned


def validate_plan_covers_calculation_trace(
    plan: Dict[str, Any],
    calculation_trace: List[Dict[str, Any]],
    *,
    plan_label: str = "StudentPlan",
    grounding_warnings: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not calculation_trace:
        return

    expected = [
        arithmetic_fingerprint(str(item.get("reported_expr", "")))
        for item in calculation_trace
        if item.get("reported_expr")
    ]
    reported = reported_expr_fingerprints(plan)

    if reported != expected:
        # Trace extraction is advisory. If the plan is otherwise valid and every
        # reported_expr is grounded in the answer text, do not fail the whole
        # formalizer just because trace grouped/split a step differently.
        if not grounding_warnings:
            return
        raise StudentAnswerFormalizerError(
            f"{plan_label} phải khớp CalculationTrace.yaml theo đúng thứ tự và đúng số step. "
            f"Trace expected: {expected}; reported_expr hiện có: {reported}. "
            "Không thêm step ngoài trace và không gộp/bỏ trace item."
        )

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
            f"{plan_label} bỏ hoặc gộp item từ CalculationTrace.yaml. "
            f"Trace expected theo thứ tự: {expected}; reported_expr hiện có: {reported}; thiếu: {missing}."
        )


def validate_final_step_fills_existing_target(
    plan: Dict[str, Any],
    problem_entities: Dict[str, Any],
    *,
    plan_label: str = "StudentPlan",
) -> None:
    target = normalize_empty(plan.get("target"))
    if not isinstance(target, str):
        return
    target_entity = problem_entities.get(target)
    if not isinstance(target_entity, dict) or target_entity.get("location") != "target":
        return

    steps = plan_step_names(plan)
    if not steps:
        return

    final_step_name = steps[-1]
    final_step = plan[final_step_name]
    if not isinstance(final_step, dict):
        return

    final_result = normalize_empty(final_step.get("result"))
    if final_result == target:
        return

    target_entity_current = problem_entities.get(target, {})
    if normalize_empty(target_entity_current.get("value")) is not None or normalize_empty(target_entity_current.get("expr")) is not None:
        return

    raise StudentAnswerFormalizerError(
        f"{plan_label} target {target!r} là entity target có sẵn nhưng step cuối "
        f"{final_step_name} tạo {final_result!r}, làm target bị rỗng. Nếu lời giải kết luận đại lượng được hỏi, "
        f"hãy đặt {final_step_name}.result = {target!r} và target: {target!r}; không tạo entity kết luận mới."
    )


def normalize_equation_fragment(fragment: str) -> str:
    fragment = re.sub(CURRENCY_SYMBOL_RE, "", fragment)
    fragment = fragment.replace(",", "")
    fragment = re.sub(
        r"(\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
        r"\1 * \2",
        fragment,
        flags=re.IGNORECASE,
    )
    fragment = fragment.replace("×", "*").replace("÷", "/")

    # Bỏ từ mô tả hoặc đơn vị xen giữa số, nhưng giữ cấu trúc số học.
    # Ví dụ: "Total = 30 notebooks + 80 pens = 110 items" -> "30 + 80 = 110".
    fragment = re.sub(r"[A-Za-z_]+", " ", fragment)
    fragment = re.sub(r"[^0-9.+\-*/()=\s]", " ", fragment)
    return re.sub(r"\s+", " ", fragment)


def equation_strings_from_fragment(fragment: str) -> List[str]:
    fragment = normalize_equation_fragment(fragment)
    equations: List[str] = []
    pattern = re.compile(
        r"(?<![0-9.])"
        r"([-+]?\d(?:[0-9.\s()+\-*/]*?))"
        r"\s*=\s*"
        r"([-+]?\d+(?:\.\d+)?)"
        r"(?![0-9])"
    )
    for match in pattern.finditer(fragment):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()
        equation = f"{lhs} = {rhs}"
        equations.append(equation)
    return equations


def equation_strings_from_line(line: str) -> List[str]:
    if "=" not in line:
        return []

    equations: List[str] = []

    # Nếu phép tính nằm trong ngoặc, lấy phần trong ngoặc trước để tránh số mô
    # tả ở ngoài ngoặc bị dính vào vế trái. Ví dụ:
    # "which is 100 tabs (400 * 1/5 = 80)" -> "400 * 1/5 = 80".
    for match in re.finditer(r"\(([^()]*)\)", line):
        content = match.group(1)
        if "=" in content:
            equations.extend(equation_strings_from_fragment(content))

    line_without_parenthesized_equations = re.sub(r"\([^()]*=[^()]*\)", " ", line)
    if "=" in line_without_parenthesized_equations:
        equations.extend(equation_strings_from_fragment(line_without_parenthesized_equations))

    return equations


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
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "half", "quarter",
}
FINAL_ANSWER_HINT_RE = re.compile(r"\b(?:final\s+answer|answer\s+is|therefore|thus|hence)\b", flags=re.IGNORECASE)
NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(sorted(NUMBER_WORDS)) + r")\b", flags=re.IGNORECASE)


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


def answer_concludes_with_number_word(student_answer: str) -> bool:
    lines = [line.strip() for line in student_answer.splitlines() if line.strip()]
    if not lines:
        return False

    conclusion = next(
        (line for line in reversed(lines) if FINAL_ANSWER_HINT_RE.search(line)),
        lines[-1],
    )
    return not re.search(r"\d", conclusion) and bool(NUMBER_WORD_RE.search(conclusion))


def append_answer_by_word_diagnosis(
    diagnosis: List[Dict[str, Any]],
    student_answer: str,
) -> List[Dict[str, Any]]:
    if not answer_concludes_with_number_word(student_answer):
        return diagnosis
    return normalize_diagnosis([*diagnosis, {"diagnosis": "answer by word"}])


def unique_entity_for_decimal_value(
    value: Decimal,
    entities: Dict[str, Dict[str, Any]],
    *,
    locations: Optional[set[str]] = None,
) -> Optional[str]:
    from Verifier import InsideChecker as inside_checker

    matches: List[str] = []
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


def numeric_ast_decimal_value(node: ast.AST) -> Optional[Decimal]:
    from Verifier import InsideChecker as inside_checker

    try:
        return inside_checker.safe_eval_arithmetic(ast.unparse(node))
    except Exception:
        return None


def build_answer_expr_from_reported_lhs(
    lhs: str,
    entities: Dict[str, Dict[str, Any]],
    prior_results_by_value: Dict[str, str],
    create_literal: Optional[Callable[[Decimal], str]] = None,
) -> Optional[str]:
    try:
        tree = ast.parse(lhs.replace("×", "*").replace("÷", "/"), mode="eval")
    except SyntaxError:
        return None

    def entity_for_value(value: Decimal) -> Optional[str]:
        value_key = str(value.normalize())
        if value_key in prior_results_by_value:
            return prior_results_by_value[value_key]
        return unique_entity_for_decimal_value(value, entities, locations={"input"}) or (
            create_literal(value) if create_literal else None
        )

    def scalar_input_entity_for_value(value: Decimal) -> Optional[str]:
        from Verifier import InsideChecker as inside_checker

        matches: List[str] = []
        for name, entity in entities.items():
            if entity.get("location") != "input":
                continue
            if normalize_empty(entity.get("unit")) is not None:
                continue
            if normalize_empty(entity.get("grand_unit")) is not None:
                continue
            entity_value = normalize_empty(entity.get("value"))
            if entity_value is None:
                continue
            try:
                if inside_checker.decimal_equal(
                    Decimal(str(entity_value).replace(",", "")),
                    value,
                ):
                    matches.append(name)
            except (InvalidOperation, ValueError):
                continue
        return matches[0] if len(matches) == 1 else None

    class ReportedNumberMapper(ast.NodeTransformer):
        unresolved_numeric = False

        def replace_numeric_node(self, node: ast.AST) -> ast.AST:
            value = numeric_ast_decimal_value(node)
            if value is None:
                return self.generic_visit(node)
            entity_name = entity_for_value(value)
            if not entity_name:
                self.unresolved_numeric = True
                return node
            return ast.copy_location(ast.Name(id=entity_name, ctx=ast.Load()), node)

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
            if isinstance(node.op, ast.Div):
                value = numeric_ast_decimal_value(node)
                if value is not None and is_fraction_literal_ast(node):
                    entity_name = entity_for_value(value)
                    if entity_name:
                        return ast.copy_location(ast.Name(id=entity_name, ctx=ast.Load()), node)

                denominator = numeric_ast_decimal_value(node.right) if is_simple_numeric_ast(node.right) else None
                if denominator not in {None, Decimal("0")}:
                    reciprocal_entity = scalar_input_entity_for_value(Decimal("1") / denominator)
                    if reciprocal_entity:
                        return ast.copy_location(
                            ast.BinOp(
                                left=self.visit(node.left),
                                op=ast.Mult(),
                                right=ast.Name(id=reciprocal_entity, ctx=ast.Load()),
                            ),
                            node,
                        )

                    entity_name = scalar_input_entity_for_value(denominator) or (
                        create_literal(denominator) if create_literal else None
                    )
                    if entity_name:
                        return ast.copy_location(
                            ast.BinOp(
                                left=self.visit(node.left),
                                op=ast.Div(),
                                right=ast.Name(id=entity_name, ctx=ast.Load()),
                            ),
                            node,
                        )

                entity_name = entity_for_value(value) if value is not None else None
                if entity_name and is_simple_numeric_ast(node):
                    return ast.copy_location(ast.Name(id=entity_name, ctx=ast.Load()), node)
            return self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:  # noqa: N802
            if numeric_ast_decimal_value(node) is not None:
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
    from Verifier import InsideChecker as inside_checker

    try:
        values = inside_checker.values_for_symbolic_expr(expr, entities)
        symbolic_value = inside_checker.safe_eval_arithmetic(expr, values)
        reported_lhs_value = inside_checker.safe_eval_arithmetic(lhs)
    except inside_checker.InsideCheckerError:
        return False
    return inside_checker.decimal_equal(symbolic_value, reported_lhs_value)


def known_entity_values(entities: Dict[str, Dict[str, Any]]) -> List[Decimal]:
    values: List[Decimal] = []
    for entity in entities.values():
        try:
            value = Decimal(str(normalize_empty(entity.get("value"))).replace(",", ""))
        except (InvalidOperation, TypeError):
            continue
        values.append(value)
    return values


def lhs_has_unresolved_number(lhs: str, entities: Dict[str, Dict[str, Any]]) -> bool:
    known_values = known_entity_values(entities)
    if not known_values:
        return False
    for value in decimal_values_from_text(lhs):
        if not decimal_value_present(value, known_values):
            return True
    return False


def repair_plan_exprs_from_reported_lhs(
    plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    answer_literal_entities: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    source_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Nếu LLM tự bung một số trung gian trong reported_expr thành phép tính từ đề,
    đổi lại thành answer-local literal.

    Ví dụ học sinh viết `24000 * 3 = 72000`. Nếu plan dùng
    `beats_per_minute * 60 * 2 * days`, checker sẽ tưởng học sinh đã làm bước
    đổi đơn vị. Thay vào đó expr phải là `student_answer_number_24000 * days`.
    """
    from Verifier import InsideChecker as inside_checker

    updated_plan = deepcopy(plan)
    extra_entities = dict(answer_literal_entities)
    reserved = set(problem_entities.keys()) | set(answer_literal_entities.keys())
    for step_name in plan_step_names(updated_plan):
        result = normalize_empty(updated_plan[step_name].get("result"))
        if result:
            reserved.add(str(result))

    literal_names_by_value: Dict[Decimal, str] = {}
    for name, entity in answer_literal_entities.items():
        try:
            value = Decimal(str(normalize_empty(entity.get("value"))).replace(",", ""))
        except (InvalidOperation, TypeError):
            continue
        literal_names_by_value[value] = name

    def create_literal(value: Decimal) -> str:
        for known_value, known_name in literal_names_by_value.items():
            if decimal_value_present(value, [known_value]):
                return known_name

        fraction_value = Fraction(str(value))
        base = f"{prefix}_number_{fraction_slug(fraction_value)}"
        name = make_unique_entity_name(base, reserved)
        literal_names_by_value[value] = name
        extra_entities[name] = {
            "value": int(value) if value == value.to_integral_value() else float(value),
            "unit": None,
            "location": "input",
            "grand_unit": None,
            "source": f"numeric literal {value} from {source_label}",
            "source_type": "answer_literal",
            "expr": None,
            "formalized_expr": None,
        }
        return name

    known_entities = {**problem_entities, **extra_entities}
    prior_results_by_value: Dict[str, str] = {}

    for step_name in plan_step_names(updated_plan):
        step = updated_plan[step_name]
        if not isinstance(step, dict):
            continue

        reported_expr = normalize_empty(step.get("reported_expr"))
        if not reported_expr:
            continue

        try:
            lhs, _ = inside_checker.split_reported_expr(str(reported_expr))
        except inside_checker.InsideCheckerError:
            continue

        current_expr = normalize_empty(step.get("expr"))
        should_rebuild = False
        if not current_expr:
            should_rebuild = True
        elif not expr_matches_reported_lhs(str(current_expr), lhs, known_entities):
            should_rebuild = True
        elif lhs_has_unresolved_number(lhs, known_entities):
            should_rebuild = True

        if should_rebuild:
            rebuilt_expr = build_answer_expr_from_reported_lhs(
                lhs,
                known_entities,
                prior_results_by_value,
                create_literal=create_literal,
            )
            if rebuilt_expr and expr_matches_reported_lhs(rebuilt_expr, lhs, {**known_entities, **extra_entities}):
                step["expr"] = rebuilt_expr
                known_entities = {**known_entities, **extra_entities}

        result = normalize_empty(step.get("result"))
        if result:
            try:
                result_value = inside_checker.parse_reported_rhs_value(str(reported_expr))
            except inside_checker.InsideCheckerError:
                continue
            prior_results_by_value.setdefault(str(result_value.normalize()), str(result))
            known_entities[str(result)] = {
                "value": int(result_value) if result_value == result_value.to_integral_value() else float(result_value),
                "unit": step.get("result_unit"),
                "location": step_name,
                "grand_unit": step.get("result_grand_unit"),
                "expr": step.get("expr"),
                "formalized_expr": step.get("expr"),
            }

    return updated_plan, extra_entities


def repair_answer_by_word_plan_exprs(
    plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    answer_literal_entities: Dict[str, Dict[str, Any]],
    student_answer: str,
) -> Dict[str, Any]:
    if not answer_concludes_with_number_word(student_answer):
        return plan

    from Verifier import InsideChecker as inside_checker

    known_entities = {
        name: dict(entity)
        for name, entity in {**problem_entities, **answer_literal_entities}.items()
    }
    prior_results_by_value: Dict[str, str] = {}

    for step_name in plan_step_names(plan):
        step = plan[step_name]
        if not isinstance(step, dict):
            continue

        reported_expr = normalize_empty(step.get("reported_expr"))
        if not reported_expr:
            continue

        try:
            lhs, _ = inside_checker.split_reported_expr(str(reported_expr))
        except inside_checker.InsideCheckerError:
            continue

        current_expr = normalize_empty(step.get("expr"))
        if not current_expr or not expr_matches_reported_lhs(str(current_expr), lhs, known_entities):
            rebuilt_expr = build_answer_expr_from_reported_lhs(lhs, known_entities, prior_results_by_value)
            if rebuilt_expr and expr_matches_reported_lhs(rebuilt_expr, lhs, known_entities):
                step["expr"] = rebuilt_expr

        result = normalize_empty(step.get("result"))
        if result:
            try:
                result_value = inside_checker.parse_reported_rhs_value(str(reported_expr))
            except inside_checker.InsideCheckerError:
                continue
            prior_results_by_value.setdefault(str(result_value.normalize()), str(result))
            known_entities[str(result)] = {
                "value": int(result_value) if result_value == result_value.to_integral_value() else float(result_value),
                "unit": step.get("result_unit"),
                "location": step_name,
                "grand_unit": step.get("result_grand_unit"),
                "expr": step.get("expr"),
                "formalized_expr": step.get("expr"),
            }

    return plan


def validate_answer_by_word_plan_exprs(
    plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    answer_literal_entities: Dict[str, Dict[str, Any]],
    student_answer: str,
) -> None:
    """
    Với đáp án viết số bằng chữ, trace parser đã chuẩn hóa number word về phép
    tính số. Guard này bắt lỗi formalizer map sai entity, nhưng không áp dụng
    cho bài số thông thường để InsideChecker vẫn có thể phát hiện misreading.
    """
    if not answer_concludes_with_number_word(student_answer):
        return

    from Verifier import InsideChecker as inside_checker

    entities = merge_student_plan_into_entities(
        plan,
        problem_entities,
        extra_entities=answer_literal_entities,
    )
    for step_name in plan_step_names(plan):
        step = plan[step_name]
        if not isinstance(step, dict):
            continue

        expr = normalize_empty(step.get("expr"))
        reported_expr = normalize_empty(step.get("reported_expr"))
        if not expr or not reported_expr:
            continue

        try:
            lhs, _ = inside_checker.split_reported_expr(str(reported_expr))
            values = inside_checker.values_for_symbolic_expr(str(expr), entities)
            symbolic_value = inside_checker.safe_eval_arithmetic(str(expr), values)
            reported_lhs_value = inside_checker.safe_eval_arithmetic(lhs)
        except inside_checker.InsideCheckerError as exc:
            raise StudentAnswerFormalizerError(
                f"{step_name}.expr hoặc reported_expr không kiểm tra được cho answer-by-word: {exc}"
            ) from exc

        if not inside_checker.decimal_equal(symbolic_value, reported_lhs_value):
            raise StudentAnswerFormalizerError(
                f"{step_name}.expr map sai number word. expr={expr!r} cho value {symbolic_value}, "
                f"nhưng vế trái reported_expr {lhs!r} cho value {reported_lhs_value}. "
                "Hãy đọc lại câu chữ và dùng entity đúng; không đổi phép tính học sinh viết."
            )


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


def is_only_final_answer(student_answer: str, calculation_trace: List[Dict[str, Any]]) -> bool:
    if calculation_trace:
        return False

    text = student_answer.strip()
    if not text:
        return False
    if re.search(r"[+*/=]", text):
        return False

    try:
        parse_decimal_from_text(text)
    except StudentAnswerFormalizerError:
        return False

    if FINAL_ANSWER_HINT_RE.search(text):
        return True

    return bool(
        re.fullmatch(
            r"\s*[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s+[a-zA-Z_]+)?\s*[.!]?\s*",
            text,
        )
    )


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def build_only_final_answer_plan(
    student_answer: str,
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    targets = [
        name
        for name, entity in problem_entities.items()
        if normalize_empty(entity.get("location")) == "target"
    ]
    if len(targets) != 1:
        raise StudentAnswerFormalizerError(
            "Không thể biểu diễn câu trả lời chỉ có đáp án cuối: "
            f"cần đúng 1 target entity, hiện có {targets}."
        )

    target = targets[0]
    answer_value = parse_decimal_from_text(student_answer)
    target_entity = problem_entities[target]
    return {
        "step1": {
            "expr": None,
            "result": target,
            "result_unit": normalize_empty(target_entity.get("unit")),
            "result_grand_unit": normalize_empty(target_entity.get("grand_unit")),
            "reported_expr": f"answer = {decimal_text(answer_value)}",
        },
        "target": target,
    }


def build_trace_derived_student_plan(
    calculation_trace: List[Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Dựng StudentPlan tối thiểu từ CalculationTrace khi LLM lập plan fail.

    Đây là fallback nội bộ code-base: trace đã được kiểm tra grounded trong lời
    giải, nên nếu plan YAML do LLM sai schema/target thì vẫn có thể tiếp tục
    symbolic pipeline bằng các entity trung gian generic.
    """
    targets = [
        name
        for name, entity in problem_entities.items()
        if normalize_empty(entity.get("location")) == "target"
    ]
    if len(targets) != 1:
        raise StudentAnswerFormalizerError(
            f"Không thể dựng StudentPlan từ trace: cần đúng 1 target entity, hiện có {targets}."
        )
    if not calculation_trace:
        raise StudentAnswerFormalizerError("Không thể dựng StudentPlan từ trace rỗng.")

    target = targets[0]
    target_entity = problem_entities[target]
    plan: Dict[str, Any] = {}
    trace_items = [
        item
        for item in calculation_trace
        if isinstance(item, dict) and normalize_empty(item.get("reported_expr"))
    ]
    if not trace_items:
        raise StudentAnswerFormalizerError("Không thể dựng StudentPlan từ trace không có reported_expr.")

    for idx, item in enumerate(trace_items, start=1):
        reported_expr = str(item["reported_expr"]).strip()
        if "=" not in reported_expr:
            raise StudentAnswerFormalizerError(
                f"Trace item {idx}.reported_expr thiếu dấu '=': {reported_expr!r}"
            )
        lhs, _ = reported_expr.rsplit("=", 1)
        is_final = idx == len(trace_items)
        result = target if is_final else f"trace_step_{idx}_result"
        plan[f"step{idx}"] = {
            "expr": normalize_reported_expr_text(lhs),
            "result": result,
            "result_unit": normalize_empty(target_entity.get("unit")) if is_final else None,
            "result_grand_unit": normalize_empty(target_entity.get("grand_unit")) if is_final else None,
            "reported_expr": normalize_reported_expr_text(reported_expr),
        }

    plan["target"] = target
    return plan


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


def is_simple_numeric_ast(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return is_simple_numeric_ast(node.operand)
    return False


def is_fraction_literal_ast(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    if not is_simple_numeric_ast(node.left) or not is_simple_numeric_ast(node.right):
        return False
    left = fraction_from_numeric_ast(node.left)
    right = fraction_from_numeric_ast(node.right)
    if left is None or right in {None, 0}:
        return False
    return abs(left) < abs(right)


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
    *,
    scalar_only: bool = False,
) -> Optional[str]:
    matches: List[str] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        if scalar_only and (
            normalize_empty(entity.get("unit")) is not None
            or normalize_empty(entity.get("grand_unit")) is not None
        ):
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
    prior_result_by_value: Dict[Fraction, str] = {}

    def entity_for_literal(value: Fraction, raw_text: str, *, scalar_only: bool = False) -> str:
        prior_result = prior_result_by_value.get(value)
        if prior_result:
            return prior_result

        existing = unique_input_entity_for_value(value, problem_entities, scalar_only=scalar_only)
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
            if isinstance(node.op, ast.Div):
                value = fraction_from_numeric_ast(node)
                if value is not None and is_fraction_literal_ast(node):
                    raw_text = ast.unparse(node)
                    return ast.copy_location(
                        ast.Name(id=entity_for_literal(value, raw_text, scalar_only=True), ctx=ast.Load()),
                        node,
                    )

                denominator = fraction_from_numeric_ast(node.right) if is_simple_numeric_ast(node.right) else None
                if denominator not in {None, 0}:
                    reciprocal_entity = unique_input_entity_for_value(
                        Fraction(1, 1) / denominator,
                        problem_entities,
                        scalar_only=True,
                    )
                    if reciprocal_entity:
                        return ast.copy_location(
                            ast.BinOp(
                                left=self.visit(node.left),
                                op=ast.Mult(),
                                right=ast.Name(id=reciprocal_entity, ctx=ast.Load()),
                            ),
                            node,
                        )

                    return ast.copy_location(
                        ast.BinOp(
                            left=self.visit(node.left),
                            op=ast.Div(),
                            right=ast.Name(
                                id=entity_for_literal(denominator, ast.unparse(node.right), scalar_only=True),
                                ctx=ast.Load(),
                            ),
                        ),
                        node,
                    )

            return self.generic_visit(node)

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

        result = normalize_empty(step.get("result"))
        reported_expr = normalize_empty(step.get("reported_expr"))
        if result and reported_expr:
            try:
                result_value = Fraction(str(value_from_reported_expr(str(reported_expr))))
            except (StudentAnswerFormalizerError, ValueError, ZeroDivisionError):
                continue
            prior_result_by_value.setdefault(result_value, str(result))

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
        calculation_trace = extract_calculation_trace(
            problem,
            student_answer,
            answer_label="bài làm học sinh",
            allowed_grounding_values=trace_allowed_grounding_values(problem_entities),
        )
        calculation_trace = repair_trace_ungrounded_rhs(
            calculation_trace,
            answer_text=student_answer,
            allowed_values=trace_allowed_grounding_values(problem_entities),
        )
        write_yaml_file(STUDENT_TRACE_PATH, {"CalculationTrace.yaml": calculation_trace})

        if is_only_final_answer(student_answer, calculation_trace):
            student_plan = build_only_final_answer_plan(student_answer, problem_entities)
            write_yaml_file(STUDENT_PLAN_PATH, student_plan)
            student_entities = merge_student_plan_into_entities(student_plan, problem_entities)
            write_yaml_file(STUDENT_ANSWER_ENTITIES_PATH, student_entities)
            write_log("Pass StudentAnswerFormalizer")
            print("Pass StudentAnswerFormalizer")
            return

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
                calculation_trace,
                previous_error=previous_error,
            )

            try:
                raw_student_plan, raw_diagnosis = parse_llm_output(raw_response)
                raw_student_plan = prune_copy_steps_from_raw_plan(raw_student_plan)
                raw_student_plan = align_raw_plan_to_calculation_trace(raw_student_plan, calculation_trace)
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
                candidate_plan, candidate_literal_entities = repair_plan_exprs_from_reported_lhs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    prefix="student_answer",
                    source_label="StudentAnswer.txt",
                )
                candidate_plan = repair_answer_by_word_plan_exprs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    student_answer,
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
                validate_plan_covers_calculation_trace(
                    candidate_plan,
                    calculation_trace,
                    grounding_warnings=candidate_grounding_warnings,
                )
                validate_answer_by_word_plan_exprs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    student_answer,
                )
            except StudentAnswerFormalizerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            student_plan = candidate_plan
            answer_literal_entities = candidate_literal_entities
            grounding_warnings = candidate_grounding_warnings
            diagnosis = append_answer_by_word_diagnosis(
                normalize_diagnosis(raw_diagnosis),
                student_answer,
            )
            break

        if student_plan is None:
            try:
                candidate_plan = build_trace_derived_student_plan(
                    calculation_trace,
                    problem_entities,
                )
                candidate_plan = validate_and_normalize_student_plan(
                    candidate_plan,
                    problem_entities,
                    student_answer=student_answer,
                )
                candidate_plan, candidate_literal_entities = materialize_numeric_literals_in_plan(
                    candidate_plan,
                    problem_entities,
                    prefix="student_answer",
                    source_label="StudentAnswer.txt",
                )
                candidate_plan, candidate_literal_entities = repair_plan_exprs_from_reported_lhs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    prefix="student_answer",
                    source_label="StudentAnswer.txt",
                )
                candidate_plan = repair_answer_by_word_plan_exprs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    student_answer,
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
                validate_plan_covers_calculation_trace(
                    candidate_plan,
                    calculation_trace,
                    grounding_warnings=candidate_grounding_warnings,
                )
                validate_answer_by_word_plan_exprs(
                    candidate_plan,
                    problem_entities,
                    candidate_literal_entities,
                    student_answer,
                )
            except StudentAnswerFormalizerError as trace_exc:
                raise StudentAnswerFormalizerError(
                    f"{last_validation_error}; trace-derived fallback cũng fail: {trace_exc}"
                ) from trace_exc

            student_plan = candidate_plan
            answer_literal_entities = candidate_literal_entities
            grounding_warnings = candidate_grounding_warnings
            diagnosis = append_answer_by_word_diagnosis([], student_answer)
            write_log_items(
                "StudentAnswerFormalizer_trace_derived_plan",
                [
                    {
                        "reason": (
                            "LLM StudentPlan retries failed; built StudentPlan "
                            "from grounded CalculationTrace."
                        ),
                        "previous_error": str(last_validation_error) if last_validation_error else None,
                    }
                ],
            )

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
