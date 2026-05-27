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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Benchmark" / "GSM8K Benchmark.csv"
DEFAULT_OUTPUT = ROOT / "Output" / "BenchmarkOuput" / "solver_pipeline_benchmark_results.csv"

PROBLEM_PATH = ROOT / "Input" / "Problem.txt"
PLAN_ENTITIES_PATH = ROOT / "Output" / "PlanEntities.yaml"
SOLVER_PATH = ROOT / "Main" / "Solver.py"


def normalize_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def decimal_from_text(value: Any) -> Optional[Decimal]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    fraction_match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if fraction_match:
        denominator = Decimal(fraction_match.group(2))
        if denominator == 0:
            return None
        return Decimal(fraction_match.group(1)) / denominator

    number_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not number_matches:
        return None

    try:
        return Decimal(number_matches[-1])
    except InvalidOperation:
        return None


def answers_match(expected: Any, actual: Any, tolerance: Decimal) -> bool:
    expected_num = decimal_from_text(expected)
    actual_num = decimal_from_text(actual)
    if expected_num is not None and actual_num is not None:
        return abs(expected_num - actual_num) <= tolerance

    return str(expected).strip().lower() == str(actual).strip().lower()


def yaml_number_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def problem_path(root_dir: Path) -> Path:
    return root_dir / "Input" / "Problem.txt"


def plan_entities_path(root_dir: Path) -> Path:
    return root_dir / "Output" / "PlanEntities.yaml"


def problem_entities_path(root_dir: Path) -> Path:
    return root_dir / "Output" / "ProblemEntities.yaml"


def plan_path(root_dir: Path) -> Path:
    return root_dir / "Output" / "Plan.yaml"


def solver_path(root_dir: Path) -> Path:
    return root_dir / "Main" / "Solver.py"


def read_snapshot(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def read_target_answer(root_dir: Path) -> tuple[str, str]:
    path = plan_entities_path(root_dir)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    with path.open(encoding="utf-8") as file:
        entities = yaml.safe_load(file)

    if not isinstance(entities, dict):
        raise ValueError("Output/PlanEntities.yaml must be a dictionary")

    targets = [
        (name, entity)
        for name, entity in entities.items()
        if isinstance(entity, dict) and entity.get("location") == "target"
    ]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly 1 target entity, got {len(targets)}")

    target_name, target_entity = targets[0]
    return target_name, yaml_number_to_text(target_entity.get("value"))


def official_answer_column(fieldnames: list[str]) -> str:
    for candidate in ("offical answer", "official answer", "answer"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Missing official answer column. Found: {fieldnames}")


def final_answer_from_response(row: dict[str, str]) -> Optional[str]:
    response = row.get("offical response") or row.get("official response") or ""
    if not response:
        return None

    matches = re.findall(
        r"(?:final answer is|answer is|therefore[^.\n]*?)(?:[^-0-9]*)(-?\d[\d,]*(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    return matches[-1]


def official_answer_for_row(row: dict[str, str], answer_column: str) -> str:
    answer = row[answer_column]
    response_answer = final_answer_from_response(row)
    if response_answer is None:
        return answer

    answer_number = decimal_from_text(answer)
    response_number = decimal_from_text(response_answer)
    if answer_number is None or response_number is None:
        return answer

    answer_digits = re.sub(r"\D", "", answer)
    response_digits = re.sub(r"\D", "", response_answer)
    if (
        answer_digits
        and response_digits.startswith(answer_digits)
        and response_number != answer_number
        and abs(response_number) > abs(answer_number)
    ):
        return response_answer

    return answer


def read_benchmark_rows(input_path: Path, limit: Optional[int]) -> tuple[list[dict[str, str]], str]:
    with input_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if "question" not in fieldnames:
            raise ValueError(f"Missing 'question' column. Found: {fieldnames}")

        answer_column = official_answer_column(fieldnames)
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    return rows, answer_column


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
            continue
        selected.add(int(chunk))
    return selected


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
        "official_answer",
        "target_entity",
        "pipeline_answer",
        "correct",
        "solver_stdout",
        "solver_stderr",
        "problem_entities_yaml",
        "plan_yaml",
        "plan_entities_yaml",
        "error_stage",
        "error",
        "duration_seconds",
    ]


def write_results(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = result_fieldnames()
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def derived_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def summary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def write_wrong_results(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    wrong_path = derived_output_path(output_path, "_wrong")
    wrong_rows = [
        row
        for row in rows
        if row.get("error", "") or row.get("correct") != "yes"
    ]
    write_results(wrong_path, wrong_rows)
    return wrong_path


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    total_limit: int,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    completed = len(rows)
    error_count = len([row for row in rows if row.get("error", "")])
    attempted = completed - error_count
    correct_count = len([row for row in rows if row.get("correct") == "yes"])
    wrong_count = len(
        [
            row
            for row in rows
            if not row.get("error", "") and row.get("correct") != "yes"
        ]
    )
    accuracy_attempted = correct_count / attempted if attempted else 0.0
    accuracy_completed = correct_count / completed if completed else 0.0

    error_stage_counts: dict[str, int] = {}
    for row in rows:
        if not row.get("error", ""):
            continue
        stage = str(row.get("error_stage") or "unknown")
        error_stage_counts[stage] = error_stage_counts.get(stage, 0) + 1

    return {
        "pipeline": "Main/Solver.py",
        "model": os.getenv("OPENROUTER_MODEL", ""),
        "input": str(input_path),
        "output": str(output_path),
        "total_limit": total_limit,
        "completed_rows": completed,
        "pending_rows": max(total_limit - completed, 0),
        "attempted_rows": attempted,
        "correct_rows": correct_count,
        "wrong_rows": wrong_count,
        "error_rows": error_count,
        "formalizer_error_rows": error_stage_counts.get("ProblemFormalizer", 0),
        "planner_error_rows": error_stage_counts.get("Planner", 0),
        "executor_error_rows": error_stage_counts.get("Executor", 0),
        "target_read_error_rows": error_stage_counts.get("target_read", 0),
        "timeout_error_rows": error_stage_counts.get("timeout", 0),
        "error_stage_counts": error_stage_counts,
        "accuracy": accuracy_attempted,
        "accuracy_percent": round(accuracy_attempted * 100, 4),
        "accuracy_attempted": accuracy_attempted,
        "accuracy_attempted_percent": round(accuracy_attempted * 100, 4),
        "accuracy_completed": accuracy_completed,
        "accuracy_completed_percent": round(accuracy_completed * 100, 4),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_summary(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    total_limit: int,
    input_path: Path,
) -> Path:
    summary_path = summary_output_path(output_path)
    summary = compute_summary(
        rows=rows,
        total_limit=total_limit,
        input_path=input_path,
        output_path=output_path,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary_path


def write_all_outputs(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    total_limit: int,
    input_path: Path,
) -> tuple[Path, Path]:
    rows.sort(key=lambda item: int(item["index"]))
    write_results(output_path, rows)
    wrong_path = write_wrong_results(output_path, rows)
    summary_path = write_summary(
        output_path=output_path,
        rows=rows,
        total_limit=total_limit,
        input_path=input_path,
    )
    return wrong_path, summary_path


def run_solver(root_dir: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(solver_path(root_dir))],
        cwd=str(root_dir),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def classify_error_stage(stdout: str, stderr: str, error: str = "") -> str:
    text = f"{stdout}\n{stderr}\n{error}"
    if "TimeoutExpired" in text:
        return "timeout"
    if "Fail ProblemFormalizer" in text:
        return "ProblemFormalizer"
    if "Fail Planner" in text:
        return "Planner"
    if "Fail Executor" in text:
        return "Executor"
    running_stages = re.findall(r"Running (ProblemFormalizer|Planner|Executor)\.\.\.", text)
    if running_stages and "Reason:" in text:
        return running_stages[-1]
    if "PlanEntities.yaml" in text or "target" in text:
        return "target_read"
    return "unknown"


def copy_project_to_workspace() -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="gsm8k_solver_")
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
        ),
    )
    return temp_dir


def record_progress_message(result_row: dict[str, Any], message: str, *, emit_log: bool) -> None:
    result_row["_progress_message"] = message
    if emit_log:
        print(message, flush=True)


def run_one_row(
    *,
    index: int,
    total: int,
    row: dict[str, str],
    answer_column: str,
    timeout: int,
    tolerance: Decimal,
    isolated: bool,
    emit_log: bool = True,
) -> dict[str, Any]:
    question = row["question"].strip()
    official_answer = official_answer_for_row(row, answer_column)
    start_time = time.monotonic()
    result_row: dict[str, Any] = {
        "index": index,
        "question": question,
        "official_answer": official_answer,
        "target_entity": "",
        "pipeline_answer": "",
        "correct": "no",
        "solver_stdout": "",
        "solver_stderr": "",
        "problem_entities_yaml": "",
        "plan_yaml": "",
        "plan_entities_yaml": "",
        "error_stage": "",
        "error": "",
        "duration_seconds": "",
    }

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    try:
        root_dir = ROOT
        if isolated:
            temp_dir = copy_project_to_workspace()
            root_dir = Path(temp_dir.name) / "repo"

        current_problem_path = problem_path(root_dir)
        current_problem_path.parent.mkdir(parents=True, exist_ok=True)
        current_problem_path.write_text(question + "\n", encoding="utf-8")

        completed = run_solver(root_dir, timeout)
        result_row["solver_stdout"] = completed.stdout.strip()
        result_row["solver_stderr"] = completed.stderr.strip()
        result_row["problem_entities_yaml"] = read_snapshot(problem_entities_path(root_dir))
        result_row["plan_yaml"] = read_snapshot(plan_path(root_dir))
        result_row["plan_entities_yaml"] = read_snapshot(plan_entities_path(root_dir))

        if completed.returncode != 0:
            result_row["error_stage"] = classify_error_stage(
                result_row["solver_stdout"],
                result_row["solver_stderr"],
            )
            raise RuntimeError(f"Solver failed with exit code {completed.returncode}")

        try:
            target_entity, pipeline_answer = read_target_answer(root_dir)
        except Exception:
            result_row["error_stage"] = "target_read"
            raise

        correct = answers_match(official_answer, pipeline_answer, tolerance)
        result_row.update(
            {
                "target_entity": target_entity,
                "pipeline_answer": pipeline_answer,
                "correct": "yes" if correct else "no",
            }
        )

        status = "OK" if correct else "WRONG"
        record_progress_message(
            result_row,
            f"[{index}/{total}] {status}: expected={official_answer!r}, "
            f"got={pipeline_answer!r} ({target_entity})",
            emit_log=emit_log,
        )
    except subprocess.TimeoutExpired as exc:
        result_row["solver_stdout"] = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        result_row["solver_stderr"] = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        result_row["error_stage"] = "timeout"
        result_row["error"] = f"TimeoutExpired: solver exceeded {timeout}s"
        record_progress_message(
            result_row,
            f"[{index}/{total}] ERROR: {result_row['error']}",
            emit_log=emit_log,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        if not result_row.get("error_stage"):
            result_row["error_stage"] = classify_error_stage(
                str(result_row.get("solver_stdout", "")),
                str(result_row.get("solver_stderr", "")),
                f"{type(exc).__name__}: {exc}",
            )
        result_row["error"] = f"{type(exc).__name__}: {exc}"
        record_progress_message(
            result_row,
            f"[{index}/{total}] ERROR ({result_row['error_stage']}): {result_row['error']}",
            emit_log=emit_log,
        )
    finally:
        result_row["duration_seconds"] = f"{time.monotonic() - start_time:.3f}"
        if temp_dir is not None:
            temp_dir.cleanup()

    return result_row


def run_benchmark(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")

    input_path = normalize_path(args.input)
    output_path = normalize_path(args.output)
    rows, answer_column = read_benchmark_rows(input_path, args.limit)
    indexed_rows = [
        (zero_based_index + 1, row)
        for zero_based_index, row in enumerate(rows)
    ]

    selected_indices: Optional[set[int]] = None
    if args.indices:
        selected_indices = parse_indices(args.indices)
        indexed_rows = [
            (index, row)
            for index, row in indexed_rows
            if index in selected_indices
        ]

    existing = read_existing_results(output_path) if args.resume else {}
    results = [existing[index] for index in sorted(existing)]
    completed = {
        int(row["index"])
        for row in results
        if str(row.get("index", "")).isdigit()
    }
    pending_rows = [
        (index, row)
        for index, row in indexed_rows
        if index not in completed
    ]

    original_problem = PROBLEM_PATH.read_text(encoding="utf-8") if PROBLEM_PATH.exists() else None

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Answer column: {answer_column}")
    print(f"Limit: {len(rows)}")
    if selected_indices is not None:
        print(f"Selected indices: {sorted(selected_indices)}")
    print(f"Pending rows: {len(pending_rows)}")
    print(f"Timeout per row: {args.timeout}s")
    print(f"Workers: {args.workers}")
    if args.workers > 1:
        print("Parallel mode: each row runs in an isolated temporary workspace.")

    try:
        if args.workers <= 1:
            for index, row in pending_rows:
                result_row = run_one_row(
                    index=index,
                    total=len(rows),
                    row=row,
                    answer_column=answer_column,
                    timeout=args.timeout,
                    tolerance=Decimal(str(args.tolerance)),
                    isolated=False,
                )
                results.append(result_row)
                write_all_outputs(
                    output_path=output_path,
                    rows=results,
                    total_limit=len(rows),
                    input_path=input_path,
                )

                if args.sleep > 0:
                    time.sleep(args.sleep)
        else:
            if args.sleep > 0:
                print("--sleep is ignored when --workers is greater than 1.")

            pending_indices = [index for index, _ in pending_rows]
            completed_for_ordered_log: dict[int, dict[str, Any]] = {}
            next_log_position = 0

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_one_row,
                        index=index,
                        total=len(rows),
                        row=row,
                        answer_column=answer_column,
                        timeout=args.timeout,
                        tolerance=Decimal(str(args.tolerance)),
                        isolated=True,
                        emit_log=False,
                    )
                    for index, row in pending_rows
                ]

                for future in as_completed(futures):
                    result_row = future.result()
                    results.append(result_row)
                    completed_for_ordered_log[int(result_row["index"])] = result_row

                    while next_log_position < len(pending_indices):
                        next_index = pending_indices[next_log_position]
                        if next_index not in completed_for_ordered_log:
                            break

                        message = completed_for_ordered_log[next_index].get("_progress_message")
                        if message:
                            print(message, flush=True)
                        next_log_position += 1

                    write_all_outputs(
                        output_path=output_path,
                        rows=results,
                        total_limit=len(rows),
                        input_path=input_path,
                    )
    finally:
        if args.workers <= 1 and args.restore_input and original_problem is not None:
            PROBLEM_PATH.write_text(original_problem, encoding="utf-8")

    attempted = len([row for row in results if row.get("error", "") == ""])
    correct_count = sum(1 for row in results if row.get("correct") == "yes")
    wrong_path, summary_path = write_all_outputs(
        output_path=output_path,
        rows=results,
        total_limit=len(rows),
        input_path=input_path,
    )
    summary = compute_summary(
        rows=results,
        total_limit=len(rows),
        input_path=input_path,
        output_path=output_path,
    )

    print()
    print(f"Completed rows: {len(results)}")
    print(f"Attempted rows: {attempted}")
    print(f"Error rows: {summary['error_rows']}")
    print(f"Formalizer errors: {summary['formalizer_error_rows']}")
    print(f"Errors by stage: {summary['error_stage_counts']}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy on attempted rows: {summary['accuracy_attempted']:.2%}")
    print(f"Accuracy on completed rows: {summary['accuracy_completed']:.2%}")
    print(f"Wrong rows output: {wrong_path}")
    print(f"Summary output: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Main/Solver.py over GSM8K benchmark rows and compare the target "
            "entity value in Output/PlanEntities.yaml with the official answer."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Benchmark CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Result CSV path.")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to run.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional 1-based row indices or ranges to run, e.g. '1,6,21-23'.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout seconds per row.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between rows.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel solver runs. Values >1 use isolated temporary "
            "workspace copies so Input/Output files do not collide."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=str,
        default="0.000001",
        help="Numeric comparison tolerance.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing rows in the output CSV and continue from missing rows.",
    )
    parser.add_argument(
        "--no-restore-input",
        dest="restore_input",
        action="store_false",
        help="Leave Input/Problem.txt as the last benchmark question.",
    )
    parser.set_defaults(restore_input=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
