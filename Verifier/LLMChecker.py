"""
Verifier/LLMChecker.py

Fallback checker dùng LLM khi pipeline symbolic không formalize được, hoặc khi
CompareChecker sinh trường hợp mơ hồ:
  Diagnosis.yaml = different calculation
  Wrong.yaml = No

Input mặc định:
  - Input/Problem.txt
  - Input/StudentAnswer.txt
  - Input/TeacherAnswer.txt

Output:
  - Output/Diagnosis.yaml
  - Output/Wrong.yaml
  - Output/LLMChecker.yaml  # log/debug chi tiết từ LLMChecker

Cách chạy:
  python3 Verifier/LLMChecker.py --mode teacher
  python3 Verifier/LLMChecker.py --mode review
  python3 Verifier/LLMChecker.py --mode auto

Mode:
  - teacher: luôn so sánh trực tiếp lời giải học sinh với lời giải giáo viên.
  - review: chỉ gọi LLM nếu Diagnosis.yaml đang là different calculation và
    Wrong.yaml là No; nếu không đúng điều kiện thì bỏ qua.
  - auto: nếu đúng điều kiện review thì review, ngược lại chạy teacher fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

PROBLEM_PATH = INPUT_DIR / "Problem.txt"
STUDENT_ANSWER_PATH = INPUT_DIR / "StudentAnswer.txt"
TEACHER_ANSWER_PATH = INPUT_DIR / "TeacherAnswer.txt"

DIAGNOSIS_PATH = OUTPUT_DIR / "Diagnosis.yaml"
WRONG_PATH = OUTPUT_DIR / "Wrong.yaml"
LLM_CHECKER_PATH = OUTPUT_DIR / "LLMChecker.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
STUDENT_PLAN_PATH = OUTPUT_DIR / "StudentPlan.yaml"
STUDENT_ANSWER_ENTITIES_PATH = OUTPUT_DIR / "StudentAnswerEntities.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 2


class LLMCheckerError(Exception):
    """Lỗi riêng cho LLMChecker."""


ALLOWED_LABELS = {
    "all right",
    "answer by word",
    "combine step",
    "different calculation",
    "do not convert units",
    "extra step",
    "logic error",
    "misreading",
    "missing step",
    "only final answer",
    "reverse steps",
    "spelling errors",
    "step separation",
    "unit missing",
    "word problem",
    "wrong calculation",
    "wrong relationship",
    "wrong target",
    "wrong unit conversion",
}

LABEL_ALIASES = {
    "all right": "all right",
    "answer by word": "answer by word",
    "combine step": "combine step",
    "combine steps": "combine step",
    "different calculation": "different calculation",
    "diffirent caculation": "different calculation",
    "different caculation": "different calculation",
    "do not convert units": "do not convert units",
    "extra step": "extra step",
    "logic error": "logic error",
    "misreading": "misreading",
    "missing step": "missing step",
    "missing steps": "missing step",
    "only final answer": "only final answer",
    "reverse step": "reverse steps",
    "reverse steps": "reverse steps",
    "spelling error": "spelling errors",
    "spelling errors": "spelling errors",
    "step separation": "step separation",
    "unit missing": "unit missing",
    "unit(s) missing": "unit missing",
    "units missing": "unit missing",
    "word problem": "word problem",
    "wrong caculation": "wrong calculation",
    "wrong calculation": "wrong calculation",
    "wrong calculations": "wrong calculation",
    "wrong relationship": "wrong relationship",
    "wrong target": "wrong target",
    "wrong unit conversion": "wrong unit conversion",
    "wrong units conversion": "wrong unit conversion",
    "wrong units conversions": "wrong unit conversion",
}

SEVERE_LABELS_IF_WRONG_MISSING = {
    "do not convert units",
    "logic error",
    "misreading",
    "missing step",
    "wrong calculation",
    "wrong relationship",
    "wrong target",
    "wrong unit conversion",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Path, *, required: bool = True) -> str:
    if not path.exists():
        if required:
            raise LLMCheckerError(f"Không tìm thấy file: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_yaml_any(path: Path, *, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise LLMCheckerError(f"Không tìm thấy file: {path}")
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LLMCheckerError(f"File YAML không hợp lệ: {path} - {exc}") from exc


def write_yaml_file(path: Path, data: Any) -> None:
    ensure_dirs()
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    raw_log = read_yaml_any(LOG_PATH, required=False)
    log_data = raw_log if isinstance(raw_log, dict) else {}
    log_data["LLMChecker"] = status
    if message:
        log_data["LLMChecker_message"] = message
    elif "LLMChecker_message" in log_data:
        del log_data["LLMChecker_message"]
    write_yaml_file(LOG_PATH, log_data)


def strip_markdown_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:yaml|yml|json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return LABEL_ALIASES.get(text, text)


def normalize_wrong(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1", "wrong"}:
        return "Yes"
    if text in {"no", "n", "false", "0", "right", "correct"}:
        return "No"
    return ""


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "1", "valid", "right", "correct"}:
        return True
    if text in {"false", "no", "n", "0", "invalid", "wrong", "incorrect"}:
        return False
    return None


def parse_diagnosis_items(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, str):
        raw_items: List[Any] = [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    elif isinstance(raw, dict):
        maybe_items = raw.get("diagnosis", raw.get("errors", raw.get("labels", [])))
        if isinstance(maybe_items, list):
            raw_items = maybe_items
        else:
            raw_items = [maybe_items]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        return []

    items: List[Dict[str, Any]] = []
    seen = set()
    unknown_labels = []

    for raw_item in raw_items:
        if isinstance(raw_item, dict):
            label = normalize_label(raw_item.get("diagnosis") or raw_item.get("label") or raw_item.get("type"))
            step = raw_item.get("step")
            entity = raw_item.get("entity")
        else:
            label = normalize_label(raw_item)
            step = None
            entity = None

        if not label:
            continue
        if label not in ALLOWED_LABELS:
            unknown_labels.append(label)
            continue
        if label in seen:
            continue
        seen.add(label)
        items.append(
            {
                "diagnosis": label,
                "step": step if step not in {"", "null", "None"} else None,
                "entity": entity if entity not in {"", "null", "None"} else None,
            }
        )

    if unknown_labels:
        raise LLMCheckerError(f"LLM trả diagnosis không thuộc whitelist: {sorted(set(unknown_labels))}")

    if len(items) > 1 and any(item["diagnosis"] == "all right" for item in items):
        items = [item for item in items if item["diagnosis"] != "all right"]

    return items


def normalize_checker_result(raw_data: Any) -> Dict[str, Any]:
    if not isinstance(raw_data, dict):
        raise LLMCheckerError("LLM output phải là dictionary.")

    data = raw_data.get("LLMChecker.yaml", raw_data)
    if not isinstance(data, dict):
        raise LLMCheckerError("LLMChecker.yaml phải là dictionary.")

    diagnosis = parse_diagnosis_items(data.get("diagnosis", data.get("labels", [])))
    wrong = normalize_wrong(data.get("wrong"))

    if not wrong:
        wrong = "Yes" if any(item["diagnosis"] in SEVERE_LABELS_IF_WRONG_MISSING for item in diagnosis) else "No"

    if wrong == "No" and not diagnosis:
        diagnosis = [{"diagnosis": "all right", "step": None, "entity": None}]

    if wrong == "Yes":
        diagnosis = [item for item in diagnosis if item["diagnosis"] != "all right"]
        if not diagnosis:
            raise LLMCheckerError("LLM kết luận wrong=Yes nhưng không đưa diagnosis hợp lệ.")

    return {
        "wrong": wrong,
        "diagnosis": diagnosis,
        "relationship_valid": coerce_bool(data.get("relationship_valid")),
        "reason": str(data.get("reason", "")).strip(),
        "review_decision": str(data.get("review_decision", "")).strip(),
    }


def read_diagnosis_file() -> List[Dict[str, Any]]:
    return parse_diagnosis_items(read_yaml_any(DIAGNOSIS_PATH, required=False))


def merge_diagnosis_items(new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for item in read_diagnosis_file() + new_items:
        normalized_items = parse_diagnosis_items([item])
        for normalized in normalized_items:
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


def parse_llm_output(text: str) -> Dict[str, Any]:
    clean_text = strip_markdown_fence(text)
    try:
        raw_data = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise LLMCheckerError(f"LLM trả YAML không hợp lệ: {exc}") from exc
    return normalize_checker_result(raw_data)


def build_system_prompt() -> str:
    return """
Bạn là LLMChecker cho hệ thống chấm lời giải toán GSM8K.

Bạn nhận:
- Đề bài gốc
- Lời giải chuẩn của giáo viên
- Lời giải học sinh

Nhiệm vụ:
1. So sánh lời giải học sinh với lời giải giáo viên dựa trên đề bài gốc.
2. Không cần formalize thành entity. Hãy đọc lời giải như người chấm toán.
3. Tập trung vào quan hệ tính toán: học sinh có dùng đúng đại lượng, đúng phép toán, đúng target không.
4. Nếu học sinh ra cùng đáp án nhưng dùng quan hệ tính toán sai hoặc may mắn trùng số, phải ghi wrong relationship và wrong: Yes.
5. Nếu học sinh dùng cách khác giáo viên nhưng hợp lệ và đáp án đúng, ghi different calculation hoặc all right, wrong: No.
6. Nếu chỉ đảo thứ tự bước, gộp bước, tách bước, diễn đạt bằng chữ, sai chính tả, thiếu đơn vị mà không làm sai toán, wrong phải là No.
7. Nếu sai số học thuần túy trong một phép tính, ghi wrong calculation.
8. Nếu đọc sai đề, thiếu bước bắt buộc, sai logic, sai target, sai đổi đơn vị, wrong phải là Yes.

Label hợp lệ:
- all right
- answer by word
- combine step
- different calculation
- do not convert units
- extra step
- logic error
- misreading
- missing step
- only final answer
- reverse steps
- spelling errors
- step separation
- unit missing
- word problem
- wrong calculation
- wrong relationship
- wrong target
- wrong unit conversion

Output bắt buộc là YAML thuần, đúng schema:
LLMChecker.yaml:
  wrong: Yes
  diagnosis:
    - diagnosis: wrong relationship
      step: null
      entity: null
  relationship_valid: false
  review_decision: replace
  reason: giải thích ngắn bằng tiếng Việt

Quy tắc output:
- Chỉ trả YAML thuần.
- Không Markdown.
- Không ``` .
- diagnosis phải dùng đúng label whitelist.
- wrong chỉ được là Yes hoặc No.
- relationship_valid là true nếu quan hệ tính toán của học sinh hợp lệ, false nếu sai, null nếu không đủ dữ kiện.
""".strip()


def build_user_prompt(
    *,
    mode: str,
    problem: str,
    teacher_answer: str,
    student_answer: str,
    current_diagnosis: Optional[List[Dict[str, Any]]] = None,
    current_wrong: Optional[str] = None,
    snapshots: Optional[Dict[str, str]] = None,
    previous_error: Optional[str] = None,
) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sửa output, vẫn chỉ trả YAML thuần theo schema.
""".rstrip()

    review_note = ""
    if mode == "review":
        current_diagnosis_yaml = yaml.safe_dump(
            current_diagnosis or [],
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        snapshot_note = ""
        if snapshots:
            snapshot_parts = []
            for name, content in snapshots.items():
                if content.strip():
                    snapshot_parts.append(f"{name}:\n{content.strip()}")
            if snapshot_parts:
                snapshot_note = "\n\nFormalized output hiện có để tham khảo, nhưng text gốc là nguồn chính:\n" + "\n\n".join(snapshot_parts)

        review_note = f"""

Ngữ cảnh review:
- Checker deterministic hiện ghi Diagnosis.yaml là:
{current_diagnosis_yaml}
- Wrong.yaml hiện là: {current_wrong}

Nhiệm vụ riêng của review mode:
- Chỉ kiểm tra lại xem quan hệ tính toán của học sinh có đúng không.
- Nếu quan hệ đúng/hợp lệ, trả wrong: No, relationship_valid: true.
- Nếu quan hệ sai, trả wrong: Yes, diagnosis: wrong relationship, relationship_valid: false.
{snapshot_note}
""".rstrip()

    return f"""
Mode: {mode}

Input/Problem.txt:
{problem}

Input/TeacherAnswer.txt:
{teacher_answer}

Input/StudentAnswer.txt:
{student_answer}
{review_note}
{retry_note}
""".strip()


def max_retries() -> int:
    raw = os.getenv("LLM_CHECKER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMCheckerError("LLM_CHECKER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise LLMCheckerError("LLM_CHECKER_MAX_RETRIES phải >= 1.")
    return value


def call_openrouter(
    *,
    mode: str,
    problem: str,
    teacher_answer: str,
    student_answer: str,
    current_diagnosis: Optional[List[Dict[str, Any]]] = None,
    current_wrong: Optional[str] = None,
    snapshots: Optional[Dict[str, str]] = None,
    previous_error: Optional[str] = None,
) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMCheckerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

    model = os.getenv("LLM_CHECKER_MODEL", os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    max_tokens = int(os.getenv("LLM_CHECKER_MAX_TOKENS", os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    timeout = int(os.getenv("LLM_CHECKER_TIMEOUT", "120"))

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
                    mode=mode,
                    problem=problem,
                    teacher_answer=teacher_answer,
                    student_answer=student_answer,
                    current_diagnosis=current_diagnosis,
                    current_wrong=current_wrong,
                    snapshots=snapshots,
                    previous_error=previous_error,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMCheckerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise LLMCheckerError(f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMCheckerError(f"Response OpenRouter không đúng định dạng: {response.text[:1000]}") from exc


def read_inputs() -> Dict[str, str]:
    return {
        "problem": read_text_file(PROBLEM_PATH, required=True),
        "teacher_answer": read_text_file(TEACHER_ANSWER_PATH, required=True),
        "student_answer": read_text_file(STUDENT_ANSWER_PATH, required=True),
    }


def read_current_diagnosis() -> List[Dict[str, Any]]:
    raw = read_yaml_any(DIAGNOSIS_PATH, required=False)
    return parse_diagnosis_items(raw)


def read_current_wrong() -> str:
    if not WRONG_PATH.exists():
        return ""
    return normalize_wrong(WRONG_PATH.read_text(encoding="utf-8").strip())


def should_review_different_calculation() -> bool:
    diagnosis = read_current_diagnosis()
    wrong = read_current_wrong()
    has_different_calculation = any(item.get("diagnosis") == "different calculation" for item in diagnosis)
    return has_different_calculation and wrong == "No"


def read_snapshot(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_review_snapshots() -> Dict[str, str]:
    return {
        "Output/Plan.yaml": read_snapshot(PLAN_PATH),
        "Output/PlanEntities.yaml": read_snapshot(PLAN_ENTITIES_PATH),
        "Output/StudentPlan.yaml": read_snapshot(STUDENT_PLAN_PATH),
        "Output/StudentAnswerEntities.yaml": read_snapshot(STUDENT_ANSWER_ENTITIES_PATH),
    }


def call_llm_with_retries(
    *,
    mode: str,
    problem: str,
    teacher_answer: str,
    student_answer: str,
    current_diagnosis: Optional[List[Dict[str, Any]]] = None,
    current_wrong: Optional[str] = None,
    snapshots: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    previous_error: Optional[str] = None
    last_error: Optional[Exception] = None

    for _ in range(max_retries()):
        raw_response = call_openrouter(
            mode=mode,
            problem=problem,
            teacher_answer=teacher_answer,
            student_answer=student_answer,
            current_diagnosis=current_diagnosis,
            current_wrong=current_wrong,
            snapshots=snapshots,
            previous_error=previous_error,
        )
        try:
            result = parse_llm_output(raw_response)
            result["raw_response"] = strip_markdown_fence(raw_response)
            return result
        except LLMCheckerError as exc:
            previous_error = str(exc)
            last_error = exc

    raise LLMCheckerError(str(last_error))


def write_checker_outputs(*, mode: str, result: Dict[str, Any], overwrite_diagnosis: bool = True) -> None:
    debug_data = {
        "mode": mode,
        "wrong": result.get("wrong"),
        "diagnosis": result.get("diagnosis", []),
        "relationship_valid": result.get("relationship_valid"),
        "review_decision": result.get("review_decision", ""),
        "reason": result.get("reason", ""),
        "raw_response": result.get("raw_response", ""),
    }
    write_yaml_file(LLM_CHECKER_PATH, debug_data)

    if overwrite_diagnosis:
        append_diagnosis_file(result.get("diagnosis", []))
        WRONG_PATH.write_text(f"{result.get('wrong', 'No')}\n", encoding="utf-8")


def run_teacher_mode() -> None:
    inputs = read_inputs()
    result = call_llm_with_retries(mode="teacher", **inputs)
    write_checker_outputs(mode="teacher", result=result, overwrite_diagnosis=True)


def run_review_mode() -> None:
    current_diagnosis = read_current_diagnosis()
    current_wrong = read_current_wrong()

    if not should_review_different_calculation():
        write_checker_outputs(
            mode="review",
            result={
                "wrong": current_wrong or "No",
                "diagnosis": current_diagnosis,
                "relationship_valid": None,
                "review_decision": "skip",
                "reason": "Bỏ qua vì Diagnosis.yaml không phải different calculation hoặc Wrong.yaml không phải No.",
                "raw_response": "",
            },
            overwrite_diagnosis=False,
        )
        print("Skip LLMChecker")
        return

    inputs = read_inputs()
    result = call_llm_with_retries(
        mode="review",
        current_diagnosis=current_diagnosis,
        current_wrong=current_wrong,
        snapshots=read_review_snapshots(),
        **inputs,
    )

    relationship_valid = result.get("relationship_valid")
    has_wrong_relationship = any(item.get("diagnosis") == "wrong relationship" for item in result.get("diagnosis", []))

    if relationship_valid is False or (result.get("wrong") == "Yes" and has_wrong_relationship):
        replacement = {
            **result,
            "wrong": "Yes",
            "diagnosis": [
                {
                    "diagnosis": "wrong relationship",
                    "step": None,
                    "entity": None,
                }
            ],
            "review_decision": result.get("review_decision") or "replace",
        }
        write_checker_outputs(mode="review", result=replacement, overwrite_diagnosis=True)
        return

    kept = {
        **result,
        "wrong": current_wrong,
        "diagnosis": current_diagnosis,
        "review_decision": result.get("review_decision") or "keep",
    }
    write_checker_outputs(mode="review", result=kept, overwrite_diagnosis=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM fallback checker for solution verification.")
    parser.add_argument(
        "--mode",
        choices=["teacher", "review", "auto"],
        default="teacher",
        help=(
            "teacher: so sánh trực tiếp teacher/student; "
            "review: chỉ kiểm tra lại different calculation + Wrong=No; "
            "auto: review nếu đúng điều kiện, ngược lại teacher."
        ),
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    try:
        ensure_dirs()

        mode = args.mode
        if mode == "auto":
            mode = "review" if should_review_different_calculation() else "teacher"

        if mode == "teacher":
            run_teacher_mode()
        elif mode == "review":
            run_review_mode()
        else:
            raise LLMCheckerError(f"Mode không hợp lệ: {mode}")

        write_log("Pass LLMChecker")
        print("Pass LLMChecker")
    except Exception as exc:
        write_log("Fail LLMChecker", str(exc))
        print("Fail LLMChecker")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
