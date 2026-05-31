from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Benchmark" / "GSM8K Benchmark.csv"
DEFAULT_ERROR_DIR = ROOT / "ErrorBaseVerify"
DEFAULT_OUTPUT_NAME = "results.csv"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 1500


LABEL_ALIASES = {
    "all right": "all right",
    "correct": "all right",
    "right": "all right",
    "answer by word": "answer by word",
    "combine step": "combine step",
    "combine steps": "combine step",
    "different calculation": "different calculation",
    "different caculation": "different calculation",
    "diffirent caculation": "different calculation",
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

ERROR_CAUSING_LABELS = {
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
}


LABEL_RUBRIC = {
    "all right": "The student solution is mathematically correct and sufficiently matches the teacher solution. Use wrong: No.",
    "answer by word": "The student gives the answer mainly in words or phrasing instead of explicit numeric/formula steps, but the math is correct. Use wrong: No unless the math is also wrong.",
    "combine step": "The student combines multiple teacher calculation steps into one valid step. The math remains correct. Use wrong: No unless a combined relation becomes mathematically wrong.",
    "different calculation": "The student uses a different valid method or equivalent expression from the teacher and reaches the correct result. Use wrong: No.",
    "do not convert units": "The problem requires unit conversion, but the student keeps incompatible units or ignores the conversion. Use wrong: Yes.",
    "extra step": "The student adds a redundant or unnecessary step that does not change the final correct reasoning. Use wrong: No unless the extra step introduces a mathematical error.",
    "logic error": "The student applies an invalid reasoning pattern that is not just arithmetic, not just misreading, and not just unit conversion. Use wrong: Yes.",
    "misreading": "The student misunderstands a fact or condition in the problem text, such as using the wrong quantity, wrong person/item, wrong comparison, or wrong given value. Use wrong: Yes if it changes the solution.",
    "missing step": "The student omits a necessary calculation/reasoning step so the solution is incomplete or unsupported. Use wrong: Yes if the omission affects correctness; use wrong: No only if the final answer and reasoning are still clearly recoverable.",
    "only final answer": "The student provides only the final answer or nearly no supporting calculation. Usually a presentation/verification issue; wrong may be No if the final answer is correct, Yes if the answer is wrong or unsupported by required work.",
    "reverse steps": "The student performs valid steps in a different order from the teacher. Use wrong: No if the relationships and result are correct.",
    "spelling errors": "The student has spelling/wording mistakes that do not change the mathematics. Use wrong: No.",
    "step separation": "The student splits one teacher calculation into multiple smaller valid steps. Use wrong: No if the math remains correct.",
    "unit missing": "The student omits units in the written answer or intermediate values, but the numeric reasoning may still be correct. Use wrong: No for harmless omissions; Yes only when omission causes wrong interpretation.",
    "word problem": "The student expresses reasoning in natural language without matching the expected structured calculation style, but may still be mathematically correct. Use wrong: No unless the math is wrong.",
    "wrong calculation": "The student uses the correct relationship/formula but makes an arithmetic computation error, such as 5 * 6 = 35. Use wrong: Yes.",
    "wrong relationship": "The student uses the wrong mathematical relationship or operation between quantities, such as adding when multiplication is required, multiplying the wrong terms, double-counting, or using an invalid formula. Use wrong: Yes.",
    "wrong target": "The student solves for a different quantity than the question asks, even if that computed quantity is internally correct. Use wrong: Yes.",
    "wrong unit conversion": "The student attempts a unit conversion but uses the wrong conversion factor or converts in the wrong direction. Use wrong: Yes.",
}


class BaseVerifierError(Exception):
    """Lỗi riêng cho base verifier benchmark."""


def normalize_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        found = normalized.get(candidate)
        if found is not None:
            return found
    raise ValueError(f"Missing required column. Tried {candidates}. Found: {fieldnames}")


def read_benchmark_rows(input_path: Path, limit: Optional[int]) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        columns = {
            "question": find_column(fieldnames, ("question", "problem")),
            "teacher_answer": find_column(
                fieldnames,
                ("offical response", "official response", "teacher answer", "correct solution"),
            ),
            "student_answer": find_column(fieldnames, ("student answer", "student response")),
            "type": find_column(fieldnames, ("type", "label", "labels")),
            "wrong": find_column(fieldnames, ("wrong", "is wrong")),
        }
        rows = [
            {canonical: row.get(source, "") for canonical, source in columns.items()}
            for row in reader
        ]

    if limit is not None:
        rows = rows[:limit]
    return rows


def parse_indices(raw_indices: str) -> set[int]:
    selected: set[int] = set()
    for chunk in raw_indices.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid index range: {chunk}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(chunk))
    return selected


def normalize_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return LABEL_ALIASES.get(text, text)


def parse_label_set(raw: Any) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    labels = set()
    for part in re.split(r"[,;\n]+", text):
        label = normalize_label(part)
        if label:
            labels.add(label)
    return labels


def normalize_wrong(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"yes", "y", "true", "1", "wrong"}:
        return "yes"
    if text in {"no", "n", "false", "0", "right", "correct"}:
        return "no"
    return text


def labels_to_text(labels: set[str]) -> str:
    return "; ".join(sorted(labels))


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)


def score_labels(expected: set[str], predicted: set[str]) -> tuple[int, set[str], set[str], set[str]]:
    tp = expected & predicted
    fp = predicted - expected
    fn = expected - predicted
    return len(tp) - len(fp) - len(fn), tp, fp, fn


def strip_markdown_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:yaml|yml|json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def parse_predicted_labels(raw: Any) -> set[str]:
    if raw is None:
        return set()

    if isinstance(raw, str):
        raw_items: list[Any] = [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    elif isinstance(raw, dict):
        value = raw.get("diagnosis", raw.get("labels", raw.get("errors", [])))
        raw_items = value if isinstance(value, list) else [value]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        return set()

    labels: set[str] = set()
    unknown: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            label = normalize_label(item.get("diagnosis") or item.get("label") or item.get("type"))
        else:
            label = normalize_label(item)
        if not label:
            continue
        if label not in ALLOWED_LABELS:
            unknown.add(label)
            continue
        labels.add(label)

    if unknown:
        raise BaseVerifierError(f"LLM trả label ngoài whitelist: {sorted(unknown)}")

    if len(labels) > 1 and "all right" in labels:
        labels.remove("all right")

    return labels


def parse_model_output(text: str) -> dict[str, Any]:
    clean_text = strip_markdown_fence(text)
    try:
        raw_data = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise BaseVerifierError(f"LLM trả YAML không hợp lệ: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise BaseVerifierError("LLM output phải là dictionary.")

    data = raw_data.get("BaseVerifier.yaml", raw_data.get("LLMChecker.yaml", raw_data))
    if not isinstance(data, dict):
        raise BaseVerifierError("BaseVerifier.yaml phải là dictionary.")

    labels = parse_predicted_labels(data.get("diagnosis", data.get("labels", [])))
    wrong = normalize_wrong(data.get("wrong"))

    if not wrong:
        wrong = "yes" if labels & SEVERE_LABELS_IF_WRONG_MISSING else "no"
    if wrong not in {"yes", "no"}:
        raise BaseVerifierError(f"wrong phải là Yes/No, hiện là {wrong!r}.")

    if wrong == "no" and not labels:
        labels = {"all right"}
    if wrong == "yes" and "all right" in labels:
        labels.remove("all right")
    if wrong == "yes" and not labels:
        raise BaseVerifierError("LLM kết luận wrong=Yes nhưng không đưa diagnosis hợp lệ.")

    return {
        "wrong": wrong,
        "labels": labels,
        "reason": str(data.get("reason", "")).strip(),
        "raw_response": clean_text,
    }


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in sorted(ALLOWED_LABELS))
    rubric = "\n".join(
        f"- {label}: {LABEL_RUBRIC[label]}"
        for label in sorted(ALLOWED_LABELS)
    )
    return f"""
You are the base verifier for GSM8K student solutions.

You receive:
- the original word problem
- the official teacher solution
- the student's solution

Your task:
1. Decide whether the student solution is mathematically wrong.
2. Assign one or more error labels from the whitelist.
3. If the student uses a different but valid calculation and reaches the correct answer, use different calculation or all right with wrong: No.
4. If the student gets the same final answer by an invalid relationship, mark wrong relationship and wrong: Yes.
5. Step order changes, combining steps, splitting steps, answer by words, spelling errors, and harmless unit omissions usually have wrong: No.
6. Misreading, missing required logic, wrong relationship, wrong target, wrong calculation, and wrong unit conversion have wrong: Yes.

Allowed labels:
{labels}

Label rubric:
{rubric}

Important distinctions:
- wrong calculation vs wrong relationship:
  wrong calculation means the formula/relation is correct but arithmetic is wrong.
  wrong relationship means the formula/relation itself is wrong, even if arithmetic is computed correctly.
- different calculation vs all right:
  use different calculation when the student's valid method differs meaningfully from the teacher's method.
  use all right when there is no meaningful error or structural difference worth labeling.
- combine step, step separation, reverse steps, extra step, spelling errors, answer by word, word problem, and harmless unit missing are usually not mathematically wrong.
- If multiple labels apply, return all important labels, not only the most severe one.

Return YAML only, with this exact schema:
BaseVerifier.yaml:
  wrong: Yes
  diagnosis:
    - diagnosis: wrong relationship
      step: null
      entity: null
  reason: short explanation in Vietnamese

Rules:
- Do not return Markdown fences.
- wrong must be Yes or No.
- diagnosis labels must come from the whitelist exactly.
- If there is no error, return diagnosis: all right and wrong: No.
""".strip()


def build_user_prompt(row: dict[str, str], previous_error: Optional[str] = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Previous output was rejected:
{previous_error}

Fix the YAML output only.
""".rstrip()

    return f"""
Original problem:
{row.get('question', '').strip()}

Official teacher solution:
{row.get('teacher_answer', '').strip()}

Student solution:
{row.get('student_answer', '').strip()}
{retry_note}
""".strip()


def openrouter_settings(args: argparse.Namespace) -> tuple[str, str, dict[str, str], int, int]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise BaseVerifierError("Thiếu OPENROUTER_API_KEY trong .env hoặc biến môi trường.")

    model = args.model or os.getenv("BASE_VERIFIER_MODEL") or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    max_tokens = args.max_tokens or int(os.getenv("BASE_VERIFIER_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    timeout = int(os.getenv("BASE_VERIFIER_TIMEOUT", str(args.timeout)))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "GSM8K-Solution-Verifier"),
    }
    return model, base_url, headers, max_tokens, timeout


def call_openrouter(
    *,
    row: dict[str, str],
    args: argparse.Namespace,
    previous_error: Optional[str] = None,
) -> str:
    model, base_url, headers, max_tokens, timeout = openrouter_settings(args)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(row, previous_error=previous_error)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise BaseVerifierError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise BaseVerifierError(f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise BaseVerifierError(f"Response OpenRouter không đúng định dạng: {response.text[:1000]}") from exc


def call_model_with_retries(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    previous_error: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(max(args.retries, 1)):
        try:
            raw_response = call_openrouter(row=row, args=args, previous_error=previous_error)
            return parse_model_output(raw_response)
        except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
            previous_error = str(exc)
            last_error = exc
            if "OpenRouter trả lỗi 402" in previous_error:
                break
            if attempt + 1 < max(args.retries, 1) and args.retry_sleep > 0:
                time.sleep(args.retry_sleep)

    raise BaseVerifierError(str(last_error))


def result_fieldnames() -> list[str]:
    return [
        "index",
        "question",
        "expected_wrong",
        "predicted_wrong",
        "wrong_match",
        "expected_labels",
        "predicted_labels",
        "label_tp",
        "label_fp",
        "label_fn",
        "label_score",
        "label_exact_match",
        "exact_match",
        "error_stage",
        "error",
        "duration_seconds",
        "reason",
        "raw_response",
    ]


def write_results(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda item: int(item["index"]))
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in result_fieldnames()})


def derived_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def summary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def bad_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("error") or row.get("exact_match") != "yes"]


def write_bad_results(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    bad_path = derived_output_path(output_path, "_wrong")
    write_results(bad_path, bad_result_rows(rows))
    return bad_path


def remove_stale_csv_outputs(output_path: Path) -> None:
    for path in [output_path, derived_output_path(output_path, "_wrong")]:
        if path.exists():
            path.unlink()


def read_existing_results(output_path: Path) -> dict[int, dict[str, str]]:
    if not output_path.exists():
        return {}
    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        results: dict[int, dict[str, str]] = {}
        for row in reader:
            try:
                results[int(row["index"])] = row
            except (KeyError, ValueError):
                continue
        return results


def exact_reconstructed_row(index: int, row: dict[str, str]) -> dict[str, Any]:
    expected_wrong = normalize_wrong(row.get("wrong"))
    expected_labels = parse_label_set(row.get("type"))
    if not expected_labels and expected_wrong == "no":
        expected_labels = {"all right"}

    label_score, tp, fp, fn = score_labels(expected_labels, expected_labels)
    return {
        "index": index,
        "question": row.get("question", "").strip(),
        "expected_wrong": expected_wrong,
        "predicted_wrong": expected_wrong,
        "wrong_match": "yes",
        "expected_labels": labels_to_text(expected_labels),
        "predicted_labels": labels_to_text(expected_labels),
        "label_tp": labels_to_text(tp),
        "label_fp": labels_to_text(fp),
        "label_fn": labels_to_text(fn),
        "label_score": str(label_score),
        "label_exact_match": "yes",
        "exact_match": "yes",
        "error_stage": "",
        "error": "",
        "duration_seconds": "",
        "reason": "(reconstructed exact match)",
        "raw_response": "",
    }


def parse_case_summary(case_dir: Path, benchmark_row: dict[str, str]) -> dict[str, Any]:
    summary_text = (case_dir / "Summary.txt").read_text(encoding="utf-8")
    top_text, _, reason_and_raw = summary_text.partition("\nReason:\n")
    reason_text, _, raw_from_summary = reason_and_raw.partition("\n\nRaw response:\n")

    fields: dict[str, str] = {}
    for line in top_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    index = int(fields.get("id") or case_dir.name)
    expected_labels = parse_label_set(fields.get("expected labels") or benchmark_row.get("type"))
    expected_wrong = normalize_wrong(fields.get("expected wrong") or benchmark_row.get("wrong"))
    predicted_labels = parse_label_set(fields.get("predicted labels"))
    predicted_wrong = normalize_wrong(fields.get("predicted wrong"))
    label_score, tp, fp, fn = score_labels(expected_labels, predicted_labels)
    wrong_match = expected_wrong == predicted_wrong
    label_exact = expected_labels == predicted_labels
    error = fields.get("error", "")

    raw_response_path = case_dir / "RawResponse.yaml"
    raw_response = raw_response_path.read_text(encoding="utf-8").strip() if raw_response_path.exists() else raw_from_summary.strip()

    return {
        "index": index,
        "question": fields.get("question") or benchmark_row.get("question", "").strip(),
        "expected_wrong": expected_wrong,
        "predicted_wrong": predicted_wrong,
        "wrong_match": "yes" if wrong_match else "no",
        "expected_labels": labels_to_text(expected_labels),
        "predicted_labels": labels_to_text(predicted_labels),
        "label_tp": labels_to_text(tp),
        "label_fp": labels_to_text(fp),
        "label_fn": labels_to_text(fn),
        "label_score": str(label_score),
        "label_exact_match": "yes" if label_exact else "no",
        "exact_match": "yes" if wrong_match and label_exact and not error else "no",
        "error_stage": fields.get("error stage", ""),
        "error": error,
        "duration_seconds": "",
        "reason": reason_text.strip(),
        "raw_response": raw_response,
    }


def rebuild_results_from_error_report(input_path: Path, error_dir: Path, limit: Optional[int]) -> list[dict[str, Any]]:
    benchmark_rows = read_benchmark_rows(input_path, limit)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(benchmark_rows, start=1):
        case_dir = error_dir / str(index)
        if (case_dir / "Summary.txt").exists():
            results.append(parse_case_summary(case_dir, row))
        else:
            results.append(exact_reconstructed_row(index, row))
    results.sort(key=lambda item: int(item["index"]))
    return results


def compute_summary(rows: list[dict[str, Any]], total_limit: int, input_path: Path, output_path: Path) -> dict[str, Any]:
    completed = len(rows)
    error_rows = [row for row in rows if row.get("error")]
    attempted = [row for row in rows if not row.get("error")]
    attempted_rows = len(attempted)
    exact_matches = [row for row in attempted if row.get("exact_match") == "yes"]
    wrong_matches = [row for row in attempted if row.get("wrong_match") == "yes"]

    error_label_tp_count = 0
    error_label_fp_count = 0
    error_label_fn_count = 0
    error_label_expected_rows = 0
    error_label_partial_match_rows = 0
    wrong_yes_tp = 0
    wrong_yes_fp = 0
    wrong_yes_fn = 0
    wrong_yes_tn = 0

    for row in attempted:
        expected_labels = parse_label_set(row.get("expected_labels"))
        predicted_labels = parse_label_set(row.get("predicted_labels"))

        expected_error_labels = expected_labels & ERROR_CAUSING_LABELS
        predicted_error_labels = predicted_labels & ERROR_CAUSING_LABELS
        error_tp = expected_error_labels & predicted_error_labels
        error_fp = predicted_error_labels - expected_error_labels
        error_fn = expected_error_labels - predicted_error_labels

        error_label_tp_count += len(error_tp)
        error_label_fp_count += len(error_fp)
        error_label_fn_count += len(error_fn)
        if expected_error_labels:
            error_label_expected_rows += 1
            if error_tp:
                error_label_partial_match_rows += 1

        expected_wrong_yes = normalize_wrong(row.get("expected_wrong")) == "yes"
        predicted_wrong_yes = normalize_wrong(row.get("predicted_wrong")) == "yes"
        if expected_wrong_yes and predicted_wrong_yes:
            wrong_yes_tp += 1
        elif not expected_wrong_yes and predicted_wrong_yes:
            wrong_yes_fp += 1
        elif expected_wrong_yes and not predicted_wrong_yes:
            wrong_yes_fn += 1
        else:
            wrong_yes_tn += 1

    error_label_precision_micro = safe_divide(
        error_label_tp_count,
        error_label_tp_count + error_label_fp_count,
    )
    error_label_recall_micro = safe_divide(
        error_label_tp_count,
        error_label_tp_count + error_label_fn_count,
    )
    error_label_f1_micro = f1_score(error_label_precision_micro, error_label_recall_micro)
    wrong_yes_precision = safe_divide(wrong_yes_tp, wrong_yes_tp + wrong_yes_fp)
    wrong_yes_recall = safe_divide(wrong_yes_tp, wrong_yes_tp + wrong_yes_fn)
    wrong_yes_f1 = f1_score(wrong_yes_precision, wrong_yes_recall)
    wrong_accuracy = safe_divide(len(wrong_matches), attempted_rows)
    error_label_hit_rate = safe_divide(error_label_partial_match_rows, error_label_expected_rows)
    exact_match = safe_divide(len(exact_matches), attempted_rows)

    error_stage_counts: dict[str, int] = {}
    for row in error_rows:
        stage = str(row.get("error_stage") or "unknown")
        error_stage_counts[stage] = error_stage_counts.get(stage, 0) + 1

    return {
        "pipeline": "base verifier direct LLM",
        "model": os.getenv("BASE_VERIFIER_MODEL") or os.getenv("OPENROUTER_MODEL", ""),
        "input": str(input_path),
        "output": str(output_path),
        "total_limit": total_limit,
        "completed_rows": completed,
        "pending_rows": max(total_limit - completed, 0),
        "attempted_rows": attempted_rows,
        "error_rows": len(error_rows),
        "metrics": {
            "wrong_accuracy": wrong_accuracy,
            "wrong_f1": wrong_yes_f1,
            "error_label_hit_rate": error_label_hit_rate,
            "error_label_f1": error_label_f1_micro,
            "exact_match": exact_match,
        },
        "support": {
            "exact_match_rows": len(exact_matches),
            "wrong_match_rows": len(wrong_matches),
            "wrong_tp": wrong_yes_tp,
            "wrong_fp": wrong_yes_fp,
            "wrong_tn": wrong_yes_tn,
            "wrong_fn": wrong_yes_fn,
            "error_label_expected_rows": error_label_expected_rows,
            "error_label_partial_match_rows": error_label_partial_match_rows,
            "error_label_tp": error_label_tp_count,
            "error_label_fp": error_label_fp_count,
            "error_label_fn": error_label_fn_count,
        },
        "error_causing_labels": sorted(ERROR_CAUSING_LABELS),
        "error_stage_counts": error_stage_counts,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_summary(output_path: Path, rows: list[dict[str, Any]], total_limit: int, input_path: Path) -> Path:
    summary_path = summary_output_path(output_path)
    summary = compute_summary(rows, total_limit, input_path, output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def write_all_outputs(
    output_path: Path,
    rows: list[dict[str, Any]],
    total_limit: int,
    input_path: Path,
    *,
    write_csv: bool,
) -> tuple[Optional[Path], Path]:
    rows.sort(key=lambda item: int(item["index"]))
    bad_path: Optional[Path] = None
    if write_csv:
        write_results(output_path, rows)
        bad_path = write_bad_results(output_path, rows)
    summary_path = write_summary(output_path, rows, total_limit, input_path)
    return bad_path, summary_path


def run_one_row(index: int, total: int, row: dict[str, str], args: argparse.Namespace, *, emit_log: bool) -> dict[str, Any]:
    expected_wrong = normalize_wrong(row.get("wrong"))
    expected_labels = parse_label_set(row.get("type"))
    if not expected_labels and expected_wrong == "no":
        expected_labels = {"all right"}

    start_time = time.monotonic()
    result: dict[str, Any] = {
        "index": index,
        "question": row.get("question", "").strip(),
        "expected_wrong": expected_wrong,
        "predicted_wrong": "",
        "wrong_match": "no",
        "expected_labels": labels_to_text(expected_labels),
        "predicted_labels": "",
        "label_tp": "",
        "label_fp": "",
        "label_fn": "",
        "label_score": "",
        "label_exact_match": "no",
        "exact_match": "no",
        "error_stage": "",
        "error": "",
        "duration_seconds": "",
        "reason": "",
        "raw_response": "",
    }

    try:
        model_result = call_model_with_retries(row, args)
        predicted_labels = model_result["labels"]
        predicted_wrong = model_result["wrong"]
        label_score, tp, fp, fn = score_labels(expected_labels, predicted_labels)
        wrong_match = expected_wrong == predicted_wrong
        label_exact = expected_labels == predicted_labels

        result.update(
            {
                "predicted_wrong": predicted_wrong,
                "wrong_match": "yes" if wrong_match else "no",
                "predicted_labels": labels_to_text(predicted_labels),
                "label_tp": labels_to_text(tp),
                "label_fp": labels_to_text(fp),
                "label_fn": labels_to_text(fn),
                "label_score": str(label_score),
                "label_exact_match": "yes" if label_exact else "no",
                "exact_match": "yes" if wrong_match and label_exact else "no",
                "reason": model_result.get("reason", ""),
                "raw_response": model_result.get("raw_response", ""),
            }
        )

        status = "OK" if result["exact_match"] == "yes" else "MISMATCH"
        message = (
            f"[{index}/{total}] {status}: expected={labels_to_text(expected_labels)!r}/{expected_wrong}, "
            f"predicted={labels_to_text(predicted_labels)!r}/{predicted_wrong}"
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        result["error_stage"] = "BaseVerifier"
        result["error"] = f"{type(exc).__name__}: {exc}"
        message = f"[{index}/{total}] ERROR (BaseVerifier): {result['error']}"
    finally:
        result["duration_seconds"] = f"{time.monotonic() - start_time:.3f}"

    result["_progress_message"] = message
    if emit_log:
        print(message, flush=True)
    return result


def write_case_file(path: Path, content: Any, fallback: str = "# unavailable\n") -> None:
    text = str(content or "").strip()
    path.write_text((text + "\n") if text else fallback, encoding="utf-8")


def case_summary_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"ID: {row.get('index')}",
            f"Question: {row.get('question')}",
            f"Expected wrong: {row.get('expected_wrong')}",
            f"Predicted wrong: {row.get('predicted_wrong')}",
            f"Expected labels: {row.get('expected_labels')}",
            f"Predicted labels: {row.get('predicted_labels')}",
            f"TP: {row.get('label_tp')}",
            f"FP: {row.get('label_fp')}",
            f"FN: {row.get('label_fn')}",
            f"Label score: {row.get('label_score')}",
            f"Error stage: {row.get('error_stage')}",
            f"Error: {row.get('error')}",
            "",
            "Reason:",
            str(row.get("reason") or "").strip() or "(empty)",
            "",
            "Raw response:",
            str(row.get("raw_response") or "").strip() or "(empty)",
        ]
    ).rstrip() + "\n"


def dashboard_html(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    data_json = (
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Base Verify Benchmark Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f8; color: #1f2933; }}
    header {{ padding: 20px 28px; background: #fff; border-bottom: 1px solid #d9dee5; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .stat {{ background: #eef2f7; border: 1px solid #d9dee5; padding: 8px 10px; border-radius: 6px; min-width: 120px; }}
    .stat b {{ display: block; font-size: 18px; }}
    main {{ display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 98px); }}
    aside {{ background: #fff; border-right: 1px solid #d9dee5; overflow: auto; }}
    .filters {{ padding: 12px; display: grid; gap: 8px; border-bottom: 1px solid #d9dee5; }}
    input, select {{ width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #c7ced8; border-radius: 6px; background: #fff; }}
    .case {{ padding: 12px; border-bottom: 1px solid #edf0f4; cursor: pointer; }}
    .case:hover, .case.active {{ background: #edf5ff; }}
    .case .id {{ font-weight: 700; }}
    .case .meta {{ font-size: 12px; color: #64748b; margin-top: 3px; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; background: #fee2e2; color: #991b1b; font-size: 12px; }}
    .detail {{ padding: 18px 22px; overflow: auto; }}
    .panel {{ background: #fff; border: 1px solid #d9dee5; border-radius: 8px; margin-bottom: 14px; }}
    .panel h2 {{ font-size: 16px; margin: 0; padding: 12px 14px; border-bottom: 1px solid #edf0f4; }}
    .panel .body {{ padding: 12px 14px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 14px; border-bottom: 1px solid #edf0f4; }}
    .tab {{ padding: 7px 10px; border: 1px solid #c7ced8; background: #fff; border-radius: 6px; cursor: pointer; }}
    .tab.active {{ background: #1f6feb; color: #fff; border-color: #1f6feb; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 13px; line-height: 1.45; }}
    @media (max-width: 850px) {{ main {{ grid-template-columns: 1fr; }} aside {{ max-height: 45vh; border-right: 0; border-bottom: 1px solid #d9dee5; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Base Verify Benchmark Dashboard</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <input id="search" placeholder="Tìm id hoặc câu hỏi">
        <select id="stage"><option value="">Tất cả stage</option></select>
      </div>
      <div id="list"></div>
    </aside>
    <section class="detail" id="detail"></section>
  </main>
  <script id="dashboard-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const state = {{ selected: 0, tab: 'Summary.txt', query: '', stage: '' }};
    const stats = data.summary || {{}};
    const cases = data.cases || [];
    const statsEl = document.getElementById('stats');
    const listEl = document.getElementById('list');
    const detailEl = document.getElementById('detail');
    const searchEl = document.getElementById('search');
    const stageEl = document.getElementById('stage');
    function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    function renderStats() {{
      const metrics = stats.metrics || {{}};
      const items = [
        ['Wrong Accuracy', ((metrics.wrong_accuracy || 0) * 100).toFixed(2) + '%'],
        ['Wrong F1', ((metrics.wrong_f1 || 0) * 100).toFixed(2) + '%'],
        ['Error Label Hit Rate', ((metrics.error_label_hit_rate || 0) * 100).toFixed(2) + '%'],
        ['Error Label F1', ((metrics.error_label_f1 || 0) * 100).toFixed(2) + '%'],
        ['Exact Match', ((metrics.exact_match || 0) * 100).toFixed(2) + '%'],
      ];
      statsEl.innerHTML = items.map(([k,v]) => `<div class="stat"><b>${{esc(v)}}</b><span>${{esc(k)}}</span></div>`).join('');
    }}
    function filteredCases() {{
      return cases.filter(c => {{
        const q = state.query.toLowerCase();
        const okQuery = !q || String(c.id).includes(q) || String(c.question).toLowerCase().includes(q);
        const okStage = !state.stage || c.stage === state.stage;
        return okQuery && okStage;
      }});
    }}
    function renderList() {{
      const items = filteredCases();
      listEl.innerHTML = items.map((c, i) => `<div class="case ${{i === state.selected ? 'active' : ''}}" data-i="${{i}}">
        <div><span class="id">#${{esc(c.id)}}</span> <span class="badge">${{esc(c.kind)}}</span></div>
        <div class="meta">${{esc(c.stage)}} · expected ${{esc(c.expected)}} · predicted ${{esc(c.predicted)}}</div>
        <div class="meta">${{esc(c.question).slice(0, 130)}}</div>
      </div>`).join('') || '<div class="case">Không có case nào.</div>';
      [...listEl.querySelectorAll('.case[data-i]')].forEach(el => el.onclick = () => {{ state.selected = Number(el.dataset.i); state.tab = 'Summary.txt'; render(); }});
    }}
    function renderDetail() {{
      const items = filteredCases();
      const c = items[state.selected];
      if (!c) {{ detailEl.innerHTML = ''; return; }}
      const files = c.files || {{}};
      if (!files[state.tab]) state.tab = Object.keys(files)[0] || 'Summary.txt';
      detailEl.innerHTML = `
        <div class="panel"><h2>#${{esc(c.id)}} · ${{esc(c.kind)}}</h2><div class="body">
          <p><b>Question:</b> ${{esc(c.question)}}</p>
          <p><b>Expected:</b> ${{esc(c.expected)}} · <b>Predicted:</b> ${{esc(c.predicted)}} · <b>Score:</b> ${{esc(c.score)}}</p>
        </div></div>
        <div class="panel">
          <div class="tabs">${{Object.keys(files).map(name => `<button class="tab ${{name === state.tab ? 'active' : ''}}" data-tab="${{esc(name)}}">${{esc(name)}}</button>`).join('')}}</div>
          <div class="body"><pre>${{esc(files[state.tab] || '')}}</pre></div>
        </div>`;
      [...detailEl.querySelectorAll('.tab')].forEach(el => el.onclick = () => {{ state.tab = el.dataset.tab; renderDetail(); }});
    }}
    function renderStages() {{
      const stages = [...new Set(cases.map(c => c.stage).filter(Boolean))];
      stageEl.innerHTML = '<option value="">Tất cả stage</option>' + stages.map(s => `<option value="${{esc(s)}}">${{esc(s)}}</option>`).join('');
    }}
    function render() {{ renderStats(); renderList(); renderDetail(); }}
    searchEl.oninput = () => {{ state.query = searchEl.value; state.selected = 0; render(); }};
    stageEl.onchange = () => {{ state.stage = stageEl.value; state.selected = 0; render(); }};
    renderStages();
    render();
  </script>
</body>
</html>
"""


def write_error_report(error_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    error_dir.mkdir(parents=True, exist_ok=True)
    for stale_dir in error_dir.iterdir():
        if stale_dir.is_dir() and stale_dir.name.isdigit():
            shutil.rmtree(stale_dir)

    cases: list[dict[str, Any]] = []
    for row in bad_result_rows(rows):
        index = str(row.get("index"))
        case_dir = error_dir / index
        case_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "Summary.txt": case_summary_text(row),
            "RawResponse.yaml": row.get("raw_response", ""),
        }
        for name, content in files.items():
            fallback = "(empty)\n" if name.endswith(".txt") else "# unavailable\n"
            write_case_file(case_dir / name, content, fallback=fallback)

        kind = "base model error" if row.get("error") else "mismatch"
        cases.append(
            {
                "id": index,
                "kind": kind,
                "stage": row.get("error_stage") or ("exact mismatch" if row.get("exact_match") != "yes" else "ok"),
                "question": row.get("question", ""),
                "expected": f"{row.get('expected_labels')} / {row.get('expected_wrong')}",
                "predicted": f"{row.get('predicted_labels')} / {row.get('predicted_wrong')}",
                "score": row.get("label_score", ""),
                "files": files,
            }
        )

    metrics = summary.get("metrics", {})
    summary_text = "\n".join(
        [
            "# Base Verify Benchmark Summary",
            "",
            f"- Total: {summary.get('total_limit')}",
            f"- Completed: {summary.get('completed_rows')}",
            f"- Attempted: {summary.get('attempted_rows')}",
            f"- Errors: {summary.get('error_rows')}",
            f"- Wrong Accuracy: {metrics.get('wrong_accuracy'):.4f}",
            f"- Wrong F1: {metrics.get('wrong_f1'):.4f}",
            f"- Error Label Hit Rate: {metrics.get('error_label_hit_rate'):.4f}",
            f"- Error Label F1: {metrics.get('error_label_f1'):.4f}",
            f"- Exact Match: {metrics.get('exact_match'):.4f}",
            f"- Updated at: {summary.get('updated_at')}",
            "",
        ]
    )
    (error_dir / "Summary.md").write_text(summary_text, encoding="utf-8")
    (error_dir / "index.html").write_text(dashboard_html(cases, summary), encoding="utf-8")


def print_run_summary(
    summary: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    bad_path: Optional[Path],
    summary_path: Path,
    error_dir: Path,
) -> None:
    metrics = summary["metrics"]
    print(f"Completed rows: {summary['completed_rows']}")
    print(f"Attempted rows: {summary['attempted_rows']}")
    print(f"Error rows: {summary['error_rows']}")
    print(f"Wrong Accuracy: {metrics['wrong_accuracy']:.2%}")
    print(f"Wrong F1: {metrics['wrong_f1']:.2%}")
    print(f"Error Label Hit Rate: {metrics['error_label_hit_rate']:.2%}")
    print(f"Error Label F1: {metrics['error_label_f1']:.2%}")
    print(f"Exact Match: {metrics['exact_match']:.2%}")
    print(f"Errors by stage: {summary['error_stage_counts']}")
    if args.verbose:
        if args.write_csv:
            print(f"Results output: {output_path}")
            print(f"Mismatches/errors output: {bad_path}")
        print(f"Summary output: {summary_path}")
        if args.write_error_report:
            print(f"Dashboard: {error_dir / 'index.html'}")


def run_benchmark(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    input_path = normalize_path(args.input)
    error_dir = normalize_path(args.error_dir)
    output_path = normalize_path(args.output) if args.output else error_dir / DEFAULT_OUTPUT_NAME
    if args.rebuild_report_only:
        results = rebuild_results_from_error_report(input_path, error_dir, args.limit)
        bad_path, summary_path = write_all_outputs(output_path, results, len(results), input_path, write_csv=args.write_csv)
        summary = compute_summary(results, len(results), input_path, output_path)
        if args.write_error_report:
            write_error_report(error_dir, results, summary)
        print_run_summary(summary, args, output_path, bad_path, summary_path, error_dir)
        return

    if args.resume and not args.write_csv:
        raise SystemExit("--resume cần --write-csv vì resume đọc lại các dòng đã chạy từ results.csv.")
    if not args.write_csv and not args.output:
        remove_stale_csv_outputs(output_path)

    rows = read_benchmark_rows(input_path, args.limit)
    indexed_rows = [(index + 1, row) for index, row in enumerate(rows)]
    if args.indices:
        selected = parse_indices(args.indices)
        indexed_rows = [(index, row) for index, row in indexed_rows if index in selected]

    existing = read_existing_results(output_path) if args.resume else {}
    results = [existing[index] for index in sorted(existing)]
    completed_indices = {int(row["index"]) for row in results if str(row.get("index", "")).isdigit()}
    pending_rows = [(index, row) for index, row in indexed_rows if index not in completed_indices]

    if args.verbose:
        print(f"Input: {input_path}")
        print(f"CSV output: {output_path if args.write_csv else '(disabled)'}")
        print(f"Rows in limit: {len(rows)}")
        print(f"Pending rows: {len(pending_rows)}")
        print(f"Workers: {args.workers}")

    if args.workers <= 1:
        for index, row in pending_rows:
            result = run_one_row(index, len(rows), row, args, emit_log=args.verbose)
            results.append(result)
            write_all_outputs(output_path, results, len(rows), input_path, write_csv=args.write_csv)
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        pending_indices = [index for index, _ in pending_rows]
        completed_for_log: dict[int, dict[str, Any]] = {}
        next_log_position = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_one_row, index, len(rows), row, args, emit_log=False)
                for index, row in pending_rows
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed_for_log[int(result["index"])] = result
                while next_log_position < len(pending_indices):
                    next_index = pending_indices[next_log_position]
                    if next_index not in completed_for_log:
                        break
                    message = completed_for_log[next_index].get("_progress_message")
                    if message and args.verbose:
                        print(message, flush=True)
                    next_log_position += 1
                write_all_outputs(output_path, results, len(rows), input_path, write_csv=args.write_csv)

    bad_path, summary_path = write_all_outputs(output_path, results, len(rows), input_path, write_csv=args.write_csv)
    summary = compute_summary(results, len(rows), input_path, output_path)
    if args.write_error_report:
        write_error_report(error_dir, results, summary)

    print_run_summary(summary, args, output_path, bad_path, summary_path, error_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct base-model verifier benchmark.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Benchmark CSV path.")
    parser.add_argument(
        "--output",
        default="",
        help="CSV path when --write-csv is enabled; also controls the summary filename. Default: <error-dir>/results.csv.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to run. Default: all rows.")
    parser.add_argument("--indices", default="", help="Optional 1-based row indices or ranges, e.g. '1,6,21-23'.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds per row.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between rows in sequential mode.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel LLM calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per row.")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Sleep between retry attempts.")
    parser.add_argument("--model", default="", help="Override model. Default: BASE_VERIFIER_MODEL or OPENROUTER_MODEL.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override completion max_tokens for base verifier calls.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rows in the output CSV. Requires --write-csv.")
    parser.add_argument("--error-dir", default=str(DEFAULT_ERROR_DIR), help="Directory for reports and dashboard.")
    parser.add_argument("--no-error-report", dest="write_error_report", action="store_false", help="Do not write per-case report.")
    parser.add_argument("--write-csv", action="store_true", help="Write results.csv and results_wrong.csv. Default is off.")
    parser.add_argument(
        "--rebuild-report-only",
        action="store_true",
        help="Rebuild summary/dashboard from existing per-case ErrorBaseVerify folders without calling the LLM.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print setup details, per-row logs, and artifact paths.")
    parser.set_defaults(write_error_report=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
