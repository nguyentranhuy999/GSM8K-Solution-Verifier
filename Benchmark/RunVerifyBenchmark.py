from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Benchmark" / "GSM8K Benchmark.csv"
DEFAULT_ERROR_DIR = ROOT / "ErrorVerify"
DEFAULT_OUTPUT_NAME = "results.csv"


def normalize_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def input_file(root_dir: Path, name: str) -> Path:
    return root_dir / "Input" / name


def output_file(root_dir: Path, name: str) -> Path:
    return root_dir / "Output" / name


def main_path(root_dir: Path) -> Path:
    return root_dir / "Main" / "Grader.py"


def read_snapshot(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def clear_pipeline_outputs(root_dir: Path) -> None:
    output_names = [
        "ProblemEntities.yaml",
        "Code.txt",
        "Plan.yaml",
        "PlanEntities.yaml",
        "TeacherPlan.yaml",
        "TeacherAnswerEntities.yaml",
        "TeacherTrace.yaml",
        "StudentPlan.yaml",
        "StudentAnswerEntities.yaml",
        "StudentTrace.yaml",
        "Diagnosis.yaml",
        "Wrong.yaml",
        "Error.yaml",
        "LLMChecker.yaml",
        "Log.yaml",
        "Hint.txt",
    ]
    for name in output_names:
        path = output_file(root_dir, name)
        if path.exists():
            path.unlink()


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
            "offical response": find_column(
                fieldnames,
                ("offical response", "official response", "teacher answer", "correct solution"),
            ),
            "student answer": find_column(fieldnames, ("student answer", "student response")),
            "type": find_column(fieldnames, ("type", "label", "labels")),
            "wrong": find_column(fieldnames, ("wrong", "is wrong")),
        }
        rows = [
            {
                canonical_name: row.get(source_name, "")
                for canonical_name, source_name in columns.items()
            }
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


LABEL_ALIASES = {
    "all right": "all right",
    "answer by word": "answer by word",
    "combine step": "combine step",
    "combine steps": "combine step",
    "different calculation": "different calculation",
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
    "units missing": "unit missing",
    "word problem": "word problem",
    "wrong caculation": "wrong calculation",
    "wrong calculation": "wrong calculation",
    "wrong calculations": "wrong calculation",
    "wrong relationship": "wrong relationship",
    "wrong target": "wrong target",
    "wrong units conversions": "wrong unit conversion",
    "wrong unit conversion": "wrong unit conversion",
    "wrong units conversion": "wrong unit conversion",
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


def parse_wrong_value(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return text


def parse_wrong_yaml(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return parse_wrong_value(stripped)


def predicted_labels_from_diagnosis(text: str) -> set[str]:
    if not text.strip():
        return set()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return set()
    if data is None:
        return set()

    labels: set[str] = set()
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("diagnosis", data.get("errors", []))
        if isinstance(items, str):
            return parse_label_set(items)
        if isinstance(items, dict):
            items = [items]
    else:
        return set()

    if not isinstance(items, list):
        return set()

    for item in items:
        if isinstance(item, dict):
            label = item.get("diagnosis") or item.get("label") or item.get("type")
        else:
            label = item
        normalized = normalize_label(label)
        if normalized:
            labels.add(normalized)
    return labels


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
        "llmchecker_invoked",
        "llmchecker_llm_called",
        "llmchecker_mode",
        "llmchecker_reason",
        "llmchecker_failed_stage",
        "error_stage",
        "error",
        "duration_seconds",
        "pipeline_stdout",
        "pipeline_stderr",
        "diagnosis_yaml",
        "wrong_yaml",
        "problem_entities_yaml",
        "teacher_trace_yaml",
        "teacher_plan_yaml",
        "teacher_entities_yaml",
        "student_trace_yaml",
        "student_plan_yaml",
        "student_entities_yaml",
        "plan_yaml",
        "plan_entities_yaml",
        "llmchecker_yaml",
        "log_yaml",
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
    return [
        row
        for row in rows
        if row.get("error") or row.get("exact_match") != "yes"
    ]


def report_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("error") or row.get("exact_match") != "yes" or row.get("llmchecker_llm_called") == "yes"
    ]


def write_bad_results(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    bad_path = derived_output_path(output_path, "_wrong")
    write_results(bad_path, bad_result_rows(rows))
    return bad_path


def remove_stale_csv_outputs(output_path: Path) -> None:
    for path in [output_path, derived_output_path(output_path, "_wrong")]:
        if path.exists():
            path.unlink()


def clean_error_report_dir(error_dir: Path) -> None:
    error_dir.mkdir(parents=True, exist_ok=True)
    stale_names = {
        "Summary.md",
        "index.html",
    }
    for path in list(error_dir.iterdir()):
        if path.is_dir() and path.name.isdigit():
            shutil.rmtree(path)
        elif path.is_file() and path.name in stale_names:
            path.unlink()


def clean_benchmark_run_outputs(error_dir: Path, output_path: Path) -> None:
    """
    Dọn artefact benchmark cũ trước khi bắt đầu run mới.

    `write_error_report()` cũng dọn case folder ở cuối run, nhưng nếu benchmark
    bị dừng giữa chừng thì dashboard/folder cũ vẫn dễ gây hiểu nhầm. Vì vậy
    non-resume run sẽ clean ngay từ đầu.
    """
    clean_error_report_dir(error_dir)
    for path in {
        output_path,
        derived_output_path(output_path, "_wrong"),
        summary_output_path(output_path),
    }:
        if path.exists() and path.is_file():
            path.unlink()


def score_attempted_rows(attempted: list[dict[str, Any]]) -> dict[str, Any]:
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

        expected_wrong_yes = parse_wrong_value(row.get("expected_wrong")) == "yes"
        predicted_wrong_yes = parse_wrong_value(row.get("predicted_wrong")) == "yes"
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
    wrong_yes_precision = safe_divide(wrong_yes_tp, wrong_yes_tp + wrong_yes_fp)
    wrong_yes_recall = safe_divide(wrong_yes_tp, wrong_yes_tp + wrong_yes_fn)

    return {
        "metrics": {
            "wrong_accuracy": safe_divide(len(wrong_matches), attempted_rows),
            "wrong_f1": f1_score(wrong_yes_precision, wrong_yes_recall),
            "error_label_hit_rate": safe_divide(error_label_partial_match_rows, error_label_expected_rows),
            "error_label_f1": f1_score(error_label_precision_micro, error_label_recall_micro),
            "exact_match": safe_divide(len(exact_matches), attempted_rows),
        },
        "support": {
            "attempted_rows": attempted_rows,
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
    }


def compute_summary(rows: list[dict[str, Any]], total_limit: int, input_path: Path, output_path: Path) -> dict[str, Any]:
    completed = len(rows)
    error_rows = [row for row in rows if row.get("error")]
    attempted = [row for row in rows if not row.get("error")]
    attempted_rows = len(attempted)
    llmchecker_invoked_rows = [row for row in attempted if row.get("llmchecker_invoked") == "yes"]
    llmchecker_llm_called_rows = [row for row in attempted if row.get("llmchecker_llm_called") == "yes"]
    llmchecker_fallback_rows = [row for row in attempted if row.get("llmchecker_failed_stage")]
    no_llmchecker_llm_rows = [
        row for row in attempted
        if row.get("llmchecker_llm_called") != "yes"
    ]
    symbolic_pipeline_pass_rows = [
        row for row in attempted
        if not row.get("llmchecker_failed_stage")
    ]
    symbolic_pipeline_fallback_rows = [
        row for row in attempted
        if row.get("llmchecker_failed_stage")
    ]
    overall_score = score_attempted_rows(attempted)
    no_llmchecker_llm_score = score_attempted_rows(no_llmchecker_llm_rows)
    symbolic_pipeline_score = score_attempted_rows(symbolic_pipeline_pass_rows)
    fallback_score = score_attempted_rows(symbolic_pipeline_fallback_rows)

    fallback_stage_counts: dict[str, int] = {}
    for row in symbolic_pipeline_fallback_rows:
        stage = str(row.get("llmchecker_failed_stage") or "unknown")
        fallback_stage_counts[stage] = fallback_stage_counts.get(stage, 0) + 1

    error_stage_counts: dict[str, int] = {}
    for row in error_rows:
        stage = str(row.get("error_stage") or "unknown")
        error_stage_counts[stage] = error_stage_counts.get(stage, 0) + 1

    return {
        "pipeline": "Main/Grader.py",
        "model": os.getenv("OPENROUTER_MODEL", ""),
        "input": str(input_path),
        "output": str(output_path),
        "total_limit": total_limit,
        "completed_rows": completed,
        "pending_rows": max(total_limit - completed, 0),
        "attempted_rows": attempted_rows,
        "error_rows": len(error_rows),
        "metrics": overall_score["metrics"],
        "support": {
            **overall_score["support"],
            "llmchecker_invoked_rows": len(llmchecker_invoked_rows),
            "llmchecker_llm_called_rows": len(llmchecker_llm_called_rows),
            "llmchecker_fallback_rows": len(llmchecker_fallback_rows),
        },
        "no_llmchecker_llm": {
            "description": "Metrics over attempted rows where LLMChecker did not call an LLM API.",
            "metrics": no_llmchecker_llm_score["metrics"],
            "support": {
                **no_llmchecker_llm_score["support"],
                "excluded_llmchecker_llm_called_rows": len(llmchecker_llm_called_rows),
            },
        },
        "symbolic_pipeline": {
            "description": "Metrics split by whether the deterministic symbolic pipeline completed before LLMChecker fallback.",
            "pass": {
                "metrics": symbolic_pipeline_score["metrics"],
                "support": symbolic_pipeline_score["support"],
            },
            "fallback": {
                "metrics": fallback_score["metrics"],
                "support": fallback_score["support"],
                "stage_counts": fallback_stage_counts,
            },
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


def classify_error_stage(stdout: str, stderr: str, error: str = "") -> str:
    text = f"{stdout}\n{stderr}\n{error}"
    if "TimeoutExpired" in text:
        return "timeout"
    for stage in [
        "Tutor",
        "Solver",
        "ProblemFormalizer",
        "TeacherAnswerFormalizer",
        "StudentAnswerFormalizer",
        "InsideChecker",
        "Mapper",
        "CompareChecker",
        "LLMChecker",
    ]:
        if f"Fail {stage}" in text:
            return stage
    running = re.findall(
        r"Running (Tutor|Solver|ProblemFormalizer|TeacherAnswerFormalizer|StudentAnswerFormalizer|InsideCheckerTeacher|InsideCheckerStudent|Mapper|CompareChecker|LLMCheckerFallback|LLMCheckerReview)\.\.\.",
        text,
    )
    if running and "Reason:" in text:
        return running[-1]
    return "unknown"


def copy_project_to_workspace() -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="gsm8k_verify_")
    workspace = Path(temp_dir.name) / "repo"
    shutil.copytree(
        ROOT,
        workspace,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            "BenchmarkOuput",
            "Error",
            "ErrorVerify",
            "ErrorCase*",
        ),
    )
    return temp_dir


def run_pipeline(root_dir: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(main_path(root_dir))],
        cwd=str(root_dir),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def snapshot_outputs(root_dir: Path) -> dict[str, str]:
    return {
        "diagnosis_yaml": read_snapshot(output_file(root_dir, "Diagnosis.yaml")),
        "wrong_yaml": read_snapshot(output_file(root_dir, "Wrong.yaml")),
        "problem_entities_yaml": read_snapshot(output_file(root_dir, "ProblemEntities.yaml")),
        "teacher_trace_yaml": read_snapshot(output_file(root_dir, "TeacherTrace.yaml")),
        "teacher_plan_yaml": read_snapshot(output_file(root_dir, "TeacherPlan.yaml")),
        "teacher_entities_yaml": read_snapshot(output_file(root_dir, "TeacherAnswerEntities.yaml")),
        "student_trace_yaml": read_snapshot(output_file(root_dir, "StudentTrace.yaml")),
        "student_plan_yaml": read_snapshot(output_file(root_dir, "StudentPlan.yaml")),
        "student_entities_yaml": read_snapshot(output_file(root_dir, "StudentAnswerEntities.yaml")),
        "plan_yaml": read_snapshot(output_file(root_dir, "Plan.yaml")),
        "plan_entities_yaml": read_snapshot(output_file(root_dir, "PlanEntities.yaml")),
        "llmchecker_yaml": read_snapshot(output_file(root_dir, "LLMChecker.yaml")),
        "log_yaml": read_snapshot(output_file(root_dir, "Log.yaml")),
    }


def yaml_dict_from_text(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def llmchecker_metadata(row: dict[str, Any]) -> dict[str, str]:
    checker_data = yaml_dict_from_text(str(row.get("llmchecker_yaml") or ""))
    log_data = yaml_dict_from_text(str(row.get("log_yaml") or ""))
    grader_call = checker_data.get("grader_call")
    if not isinstance(grader_call, dict):
        grader_call = {}

    def text_value(value: Any) -> str:
        return "" if value is None else str(value)

    invoked = bool(grader_call.get("called_by_grader") or checker_data)
    llm_called = bool(
        grader_call.get("llm_called")
        or checker_data.get("llm_called")
        or str(checker_data.get("raw_response") or "").strip()
    )

    return {
        "llmchecker_invoked": "yes" if invoked else "no",
        "llmchecker_llm_called": "yes" if llm_called else "no",
        "llmchecker_mode": text_value(
            grader_call.get("mode") or checker_data.get("mode") or log_data.get("Grader_llmchecker_mode")
        ),
        "llmchecker_reason": text_value(
            grader_call.get("reason")
            or checker_data.get("reason")
            or log_data.get("Grader_llmchecker_reason")
        ),
        "llmchecker_failed_stage": text_value(
            grader_call.get("failed_stage")
            or log_data.get("Grader_failed_stage")
        ),
    }


def record_progress_message(result_row: dict[str, Any], message: str, *, emit_log: bool) -> None:
    result_row["_progress_message"] = message
    if emit_log:
        print(message, flush=True)


def run_one_row(
    *,
    index: int,
    total: int,
    row: dict[str, str],
    timeout: int,
    isolated: bool,
    emit_log: bool = True,
) -> dict[str, Any]:
    question = (row.get("question") or "").strip()
    teacher_answer = (row.get("offical response") or row.get("official response") or "").strip()
    student_answer = (row.get("student answer") or "").strip()
    expected_wrong = parse_wrong_value(row.get("wrong"))
    expected_labels = parse_label_set(row.get("type"))
    if not expected_labels and expected_wrong == "no":
        expected_labels = {"all right"}

    start_time = time.monotonic()
    result_row: dict[str, Any] = {
        "index": index,
        "question": question,
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
        "llmchecker_invoked": "no",
        "llmchecker_llm_called": "no",
        "llmchecker_mode": "",
        "llmchecker_reason": "",
        "llmchecker_failed_stage": "",
        "error_stage": "",
        "error": "",
        "duration_seconds": "",
        "pipeline_stdout": "",
        "pipeline_stderr": "",
    }

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        root_dir = ROOT
        if isolated:
            temp_dir = copy_project_to_workspace()
            root_dir = Path(temp_dir.name) / "repo"

        input_file(root_dir, "Problem.txt").parent.mkdir(parents=True, exist_ok=True)
        input_file(root_dir, "Problem.txt").write_text(question + "\n", encoding="utf-8")
        input_file(root_dir, "TeacherAnswer.txt").write_text(teacher_answer + "\n", encoding="utf-8")
        input_file(root_dir, "StudentAnswer.txt").write_text(student_answer + "\n", encoding="utf-8")
        clear_pipeline_outputs(root_dir)

        completed = run_pipeline(root_dir, timeout)
        result_row["pipeline_stdout"] = completed.stdout.strip()
        result_row["pipeline_stderr"] = completed.stderr.strip()
        result_row.update(snapshot_outputs(root_dir))
        result_row.update(llmchecker_metadata(result_row))

        if completed.returncode != 0:
            result_row["error_stage"] = classify_error_stage(
                result_row["pipeline_stdout"],
                result_row["pipeline_stderr"],
            )
            raise RuntimeError(f"Verify pipeline failed with exit code {completed.returncode}")

        predicted_labels = predicted_labels_from_diagnosis(result_row.get("diagnosis_yaml", ""))
        predicted_wrong = parse_wrong_yaml(result_row.get("wrong_yaml", ""))
        label_score, tp, fp, fn = score_labels(expected_labels, predicted_labels)
        wrong_match = expected_wrong == predicted_wrong
        label_exact = expected_labels == predicted_labels

        result_row.update(
            {
                "predicted_wrong": predicted_wrong,
                "wrong_match": "yes" if wrong_match else "no",
                "predicted_labels": labels_to_text(predicted_labels),
                "label_tp": labels_to_text(tp),
                "label_fp": labels_to_text(fp),
                "label_fn": labels_to_text(fn),
                "label_score": str(label_score),
                "label_exact_match": "yes" if label_exact else "no",
                "exact_match": "yes" if (wrong_match and label_exact) else "no",
            }
        )

        status = "OK" if result_row["exact_match"] == "yes" else "MISMATCH"
        llm_note = ""
        if result_row.get("llmchecker_llm_called") == "yes":
            failed_stage = result_row.get("llmchecker_failed_stage")
            llm_note = f", llmchecker={failed_stage or result_row.get('llmchecker_mode')}"
        record_progress_message(
            result_row,
            f"[{index}/{total}] {status}: expected={labels_to_text(expected_labels)!r}/{expected_wrong}, "
            f"predicted={labels_to_text(predicted_labels)!r}/{predicted_wrong}{llm_note}",
            emit_log=emit_log,
        )
    except subprocess.TimeoutExpired as exc:
        result_row["pipeline_stdout"] = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        result_row["pipeline_stderr"] = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        result_row["error_stage"] = "timeout"
        result_row["error"] = f"TimeoutExpired: verify pipeline exceeded {timeout}s"
        record_progress_message(
            result_row,
            f"[{index}/{total}] ERROR: {result_row['error']}",
            emit_log=emit_log,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        if not result_row.get("error_stage"):
            result_row["error_stage"] = classify_error_stage(
                str(result_row.get("pipeline_stdout", "")),
                str(result_row.get("pipeline_stderr", "")),
                f"{type(exc).__name__}: {exc}",
            )
        result_row["error"] = f"{type(exc).__name__}: {exc}"
        record_progress_message(
            result_row,
            f"[{index}/{total}] ERROR ({result_row['error_stage']}): {result_row['error']}",
            emit_log=emit_log,
        )
        if temp_dir is not None:
            try:
                result_row.update(snapshot_outputs(Path(temp_dir.name) / "repo"))
                result_row.update(llmchecker_metadata(result_row))
            except Exception:
                pass
    finally:
        result_row["duration_seconds"] = f"{time.monotonic() - start_time:.3f}"
        if temp_dir is not None:
            temp_dir.cleanup()

    return result_row


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
  <title>Verify Benchmark Dashboard</title>
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
    <h1>Verify Benchmark Dashboard</h1>
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
    const state = {{ selected: 0, tab: 'Diagnosis.yaml', query: '', stage: '' }};
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
      [...listEl.querySelectorAll('.case[data-i]')].forEach(el => el.onclick = () => {{ state.selected = Number(el.dataset.i); state.tab = 'Diagnosis.yaml'; render(); }});
    }}
    function renderDetail() {{
      const items = filteredCases();
      const c = items[state.selected];
      if (!c) {{ detailEl.innerHTML = ''; return; }}
      const files = c.files || {{}};
      if (!files[state.tab]) state.tab = Object.keys(files)[0] || 'Diagnosis.yaml';
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
            f"LLMChecker invoked: {row.get('llmchecker_invoked')}",
            f"LLMChecker called LLM: {row.get('llmchecker_llm_called')}",
            f"LLMChecker mode: {row.get('llmchecker_mode')}",
            f"LLMChecker reason: {row.get('llmchecker_reason')}",
            f"LLMChecker failed stage: {row.get('llmchecker_failed_stage')}",
            f"Error stage: {row.get('error_stage')}",
            f"Error: {row.get('error')}",
            "",
            "stdout:",
            str(row.get("pipeline_stdout") or "").strip() or "(empty)",
            "",
            "stderr:",
            str(row.get("pipeline_stderr") or "").strip() or "(empty)",
        ]
    ).rstrip() + "\n"


def write_error_report(error_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    clean_error_report_dir(error_dir)

    cases: list[dict[str, Any]] = []
    for row in report_case_rows(rows):
        index = str(row.get("index"))
        case_dir = error_dir / index
        case_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "Summary.txt": case_summary_text(row),
            "Diagnosis.yaml": row.get("diagnosis_yaml", ""),
            "Wrong.yaml": row.get("wrong_yaml", ""),
            "ProblemEntities.yaml": row.get("problem_entities_yaml", ""),
            "TeacherTrace.yaml": row.get("teacher_trace_yaml", ""),
            "TeacherPlan.yaml": row.get("teacher_plan_yaml", ""),
            "TeacherAnswerEntities.yaml": row.get("teacher_entities_yaml", ""),
            "StudentTrace.yaml": row.get("student_trace_yaml", ""),
            "StudentPlan.yaml": row.get("student_plan_yaml", ""),
            "StudentAnswerEntities.yaml": row.get("student_entities_yaml", ""),
            "Plan.yaml": row.get("plan_yaml", ""),
            "PlanEntities.yaml": row.get("plan_entities_yaml", ""),
            "LLMChecker.yaml": row.get("llmchecker_yaml", ""),
        }
        for name, content in files.items():
            fallback = "(empty)\n" if name.endswith(".txt") else "# unavailable\n"
            write_case_file(case_dir / name, content, fallback=fallback)

        if row.get("error"):
            kind = "pipeline error"
        elif row.get("exact_match") != "yes":
            kind = "mismatch"
        elif row.get("llmchecker_failed_stage"):
            kind = "llm fallback"
        else:
            kind = "llm review"
        stage = row.get("error_stage") or row.get("llmchecker_failed_stage")
        if not stage and row.get("llmchecker_llm_called") == "yes":
            stage = f"llmchecker {row.get('llmchecker_mode')}"
        cases.append(
            {
                "id": index,
                "kind": kind,
                "stage": stage or ("exact mismatch" if row.get("exact_match") != "yes" else "ok"),
                "question": row.get("question", ""),
                "expected": f"{row.get('expected_labels')} / {row.get('expected_wrong')}",
                "predicted": f"{row.get('predicted_labels')} / {row.get('predicted_wrong')}",
                "score": row.get("label_score", ""),
                "files": files,
            }
        )

    metrics = summary.get("metrics", {})
    support = summary.get("support", {})
    no_llm = summary.get("no_llmchecker_llm", {})
    no_llm_metrics = no_llm.get("metrics", {}) if isinstance(no_llm, dict) else {}
    no_llm_support = no_llm.get("support", {}) if isinstance(no_llm, dict) else {}
    symbolic = summary.get("symbolic_pipeline", {})
    symbolic_pass = symbolic.get("pass", {}) if isinstance(symbolic, dict) else {}
    symbolic_fallback = symbolic.get("fallback", {}) if isinstance(symbolic, dict) else {}
    symbolic_pass_metrics = symbolic_pass.get("metrics", {}) if isinstance(symbolic_pass, dict) else {}
    symbolic_pass_support = symbolic_pass.get("support", {}) if isinstance(symbolic_pass, dict) else {}
    symbolic_fallback_metrics = symbolic_fallback.get("metrics", {}) if isinstance(symbolic_fallback, dict) else {}
    symbolic_fallback_support = symbolic_fallback.get("support", {}) if isinstance(symbolic_fallback, dict) else {}
    symbolic_fallback_stages = symbolic_fallback.get("stage_counts", {}) if isinstance(symbolic_fallback, dict) else {}
    summary_text = "\n".join(
        [
            "# Verify Benchmark Summary",
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
            f"- LLMChecker invoked rows: {support.get('llmchecker_invoked_rows')}",
            f"- LLMChecker called LLM rows: {support.get('llmchecker_llm_called_rows')}",
            f"- LLMChecker fallback rows: {support.get('llmchecker_fallback_rows')}",
            "",
            "## No LLMChecker LLM Metrics",
            "",
            f"- Attempted: {no_llm_support.get('attempted_rows')}",
            f"- Wrong Accuracy: {no_llm_metrics.get('wrong_accuracy'):.4f}",
            f"- Wrong F1: {no_llm_metrics.get('wrong_f1'):.4f}",
            f"- Error Label Hit Rate: {no_llm_metrics.get('error_label_hit_rate'):.4f}",
            f"- Error Label F1: {no_llm_metrics.get('error_label_f1'):.4f}",
            f"- Exact Match: {no_llm_metrics.get('exact_match'):.4f}",
            "",
            "## Symbolic Pipeline Metrics",
            "",
            f"- Symbolic pass attempted: {symbolic_pass_support.get('attempted_rows')}",
            f"- Symbolic pass Wrong Accuracy: {symbolic_pass_metrics.get('wrong_accuracy'):.4f}",
            f"- Symbolic pass Wrong F1: {symbolic_pass_metrics.get('wrong_f1'):.4f}",
            f"- Symbolic pass Error Label Hit Rate: {symbolic_pass_metrics.get('error_label_hit_rate'):.4f}",
            f"- Symbolic pass Error Label F1: {symbolic_pass_metrics.get('error_label_f1'):.4f}",
            f"- Symbolic pass Exact Match: {symbolic_pass_metrics.get('exact_match'):.4f}",
            f"- Fallback attempted: {symbolic_fallback_support.get('attempted_rows')}",
            f"- Fallback Wrong Accuracy: {symbolic_fallback_metrics.get('wrong_accuracy'):.4f}",
            f"- Fallback Wrong F1: {symbolic_fallback_metrics.get('wrong_f1'):.4f}",
            f"- Fallback Error Label Hit Rate: {symbolic_fallback_metrics.get('error_label_hit_rate'):.4f}",
            f"- Fallback Error Label F1: {symbolic_fallback_metrics.get('error_label_f1'):.4f}",
            f"- Fallback Exact Match: {symbolic_fallback_metrics.get('exact_match'):.4f}",
            f"- Fallback by stage: {json.dumps(symbolic_fallback_stages, ensure_ascii=False, sort_keys=True)}",
            f"- Updated at: {summary.get('updated_at')}",
            "",
        ]
    )
    (error_dir / "Summary.md").write_text(summary_text, encoding="utf-8")
    (error_dir / "index.html").write_text(dashboard_html(cases, summary), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")

    input_path = normalize_path(args.input)
    error_dir = normalize_path(args.error_dir)
    output_path = normalize_path(args.output) if args.output else error_dir / DEFAULT_OUTPUT_NAME
    if args.resume and not args.write_csv:
        raise SystemExit("--resume cần --write-csv vì resume đọc lại các dòng đã chạy từ results.csv.")
    if not args.resume and args.write_error_report:
        clean_benchmark_run_outputs(error_dir, output_path)
    elif not args.write_csv and not args.output:
        remove_stale_csv_outputs(output_path)

    rows = read_benchmark_rows(input_path, args.limit)
    indexed_rows = [(i + 1, row) for i, row in enumerate(rows)]

    if args.indices:
        selected = parse_indices(args.indices)
        indexed_rows = [(index, row) for index, row in indexed_rows if index in selected]

    existing = read_existing_results(output_path) if args.resume else {}
    results = [existing[index] for index in sorted(existing)]
    completed_indices = {int(row["index"]) for row in results if str(row.get("index", "")).isdigit()}
    pending_rows = [(index, row) for index, row in indexed_rows if index not in completed_indices]

    original_inputs = {
        name: input_file(ROOT, name).read_text(encoding="utf-8") if input_file(ROOT, name).exists() else None
        for name in ["Problem.txt", "TeacherAnswer.txt", "StudentAnswer.txt"]
    }

    if args.verbose:
        print(f"Input: {input_path}")
        print(f"CSV output: {output_path if args.write_csv else '(disabled)'}")
        print(f"Rows in limit: {len(rows)}")
        print(f"Pending rows: {len(pending_rows)}")
        print(f"Workers: {args.workers}")

    try:
        if args.workers <= 1:
            for index, row in pending_rows:
                result = run_one_row(
                    index=index,
                    total=len(rows),
                    row=row,
                    timeout=args.timeout,
                    isolated=False,
                    emit_log=args.verbose,
                )
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
                    executor.submit(
                        run_one_row,
                        index=index,
                        total=len(rows),
                        row=row,
                        timeout=args.timeout,
                        isolated=True,
                        emit_log=False,
                    )
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
    finally:
        if args.workers <= 1 and args.restore_input:
            for name, content in original_inputs.items():
                path = input_file(ROOT, name)
                if content is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_text(content, encoding="utf-8")

    bad_path, summary_path = write_all_outputs(output_path, results, len(rows), input_path, write_csv=args.write_csv)
    summary = compute_summary(results, len(rows), input_path, output_path)
    if args.write_error_report:
        write_error_report(error_dir, results, summary)

    metrics = summary["metrics"]
    print(f"Completed rows: {summary['completed_rows']}")
    print(f"Attempted rows: {summary['attempted_rows']}")
    print(f"Error rows: {summary['error_rows']}")
    print(f"Wrong Accuracy: {metrics['wrong_accuracy']:.2%}")
    print(f"Wrong F1: {metrics['wrong_f1']:.2%}")
    print(f"Error Label Hit Rate: {metrics['error_label_hit_rate']:.2%}")
    print(f"Error Label F1: {metrics['error_label_f1']:.2%}")
    print(f"Exact Match: {metrics['exact_match']:.2%}")
    print(f"LLMChecker called LLM rows: {summary['support']['llmchecker_llm_called_rows']}")
    print(f"LLMChecker fallback rows: {summary['support']['llmchecker_fallback_rows']}")
    no_llm = summary.get("no_llmchecker_llm", {})
    no_llm_metrics = no_llm.get("metrics", {}) if isinstance(no_llm, dict) else {}
    no_llm_support = no_llm.get("support", {}) if isinstance(no_llm, dict) else {}
    print(f"No LLMChecker LLM attempted rows: {no_llm_support.get('attempted_rows', 0)}")
    print(
        "No LLMChecker LLM metrics: "
        f"Wrong Accuracy {no_llm_metrics.get('wrong_accuracy', 0):.2%}, "
        f"Wrong F1 {no_llm_metrics.get('wrong_f1', 0):.2%}, "
        f"Error Label Hit Rate {no_llm_metrics.get('error_label_hit_rate', 0):.2%}, "
        f"Error Label F1 {no_llm_metrics.get('error_label_f1', 0):.2%}, "
        f"Exact Match {no_llm_metrics.get('exact_match', 0):.2%}"
    )
    symbolic = summary.get("symbolic_pipeline", {})
    symbolic_pass = symbolic.get("pass", {}) if isinstance(symbolic, dict) else {}
    symbolic_fallback = symbolic.get("fallback", {}) if isinstance(symbolic, dict) else {}
    symbolic_pass_metrics = symbolic_pass.get("metrics", {}) if isinstance(symbolic_pass, dict) else {}
    symbolic_pass_support = symbolic_pass.get("support", {}) if isinstance(symbolic_pass, dict) else {}
    symbolic_fallback_metrics = symbolic_fallback.get("metrics", {}) if isinstance(symbolic_fallback, dict) else {}
    symbolic_fallback_support = symbolic_fallback.get("support", {}) if isinstance(symbolic_fallback, dict) else {}
    print(f"Symbolic pipeline pass rows: {symbolic_pass_support.get('attempted_rows', 0)}")
    print(
        "Symbolic pass metrics: "
        f"Wrong Accuracy {symbolic_pass_metrics.get('wrong_accuracy', 0):.2%}, "
        f"Wrong F1 {symbolic_pass_metrics.get('wrong_f1', 0):.2%}, "
        f"Error Label Hit Rate {symbolic_pass_metrics.get('error_label_hit_rate', 0):.2%}, "
        f"Error Label F1 {symbolic_pass_metrics.get('error_label_f1', 0):.2%}, "
        f"Exact Match {symbolic_pass_metrics.get('exact_match', 0):.2%}"
    )
    print(f"Symbolic fallback rows: {symbolic_fallback_support.get('attempted_rows', 0)}")
    print(
        "Fallback metrics: "
        f"Wrong Accuracy {symbolic_fallback_metrics.get('wrong_accuracy', 0):.2%}, "
        f"Wrong F1 {symbolic_fallback_metrics.get('wrong_f1', 0):.2%}, "
        f"Error Label Hit Rate {symbolic_fallback_metrics.get('error_label_hit_rate', 0):.2%}, "
        f"Error Label F1 {symbolic_fallback_metrics.get('error_label_f1', 0):.2%}, "
        f"Exact Match {symbolic_fallback_metrics.get('exact_match', 0):.2%}"
    )
    print(f"Errors by stage: {summary['error_stage_counts']}")
    if args.verbose:
        if args.write_csv:
            print(f"Results output: {output_path}")
            print(f"Mismatches/errors output: {bad_path}")
        print(f"Summary output: {summary_path}")
        if args.write_error_report:
            print(f"Dashboard: {error_dir / 'index.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher-reference verification benchmark and score predicted error labels."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Benchmark CSV path.")
    parser.add_argument(
        "--output",
        default="",
        help="CSV path when --write-csv is enabled; also controls the summary filename. Default: <error-dir>/results.csv.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to run.")
    parser.add_argument("--indices", default="", help="Optional 1-based row indices or ranges, e.g. '1,6,21-23'.")
    parser.add_argument("--timeout", type=int, default=420, help="Timeout seconds per row.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between rows in sequential mode.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel verification runs.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rows in the output CSV. Requires --write-csv.")
    parser.add_argument("--no-restore-input", dest="restore_input", action="store_false", help="Leave Input/*.txt as last row.")
    parser.add_argument("--error-dir", default=str(DEFAULT_ERROR_DIR), help="Directory for reports and dashboard.")
    parser.add_argument("--no-error-report", dest="write_error_report", action="store_false", help="Do not write per-case report.")
    parser.add_argument("--write-csv", action="store_true", help="Write results.csv and results_wrong.csv. Default is off.")
    parser.add_argument("--verbose", action="store_true", help="Print setup details, per-row logs, and artifact paths.")
    parser.set_defaults(restore_input=True)
    parser.set_defaults(write_error_report=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
