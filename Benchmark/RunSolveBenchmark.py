from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Benchmark" / "GSM8K Benchmark.csv"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_DIRECT_OUTPUT = ROOT / "Output" / "gpt4o_mini_benchmark_results.csv"
DEFAULT_COT_OUTPUT = ROOT / "Output" / "gpt4o_mini_cot_benchmark_results.csv"


def default_output_for_mode(mode: str) -> Path:
    if mode == "cot":
        return DEFAULT_COT_OUTPUT
    return DEFAULT_DIRECT_OUTPUT


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def decimal_from_text(value: Any) -> Optional[Decimal]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace(",", "")
    fraction_match = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if fraction_match:
        numerator = Decimal(fraction_match.group(1))
        denominator = Decimal(fraction_match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator

    number_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not number_matches:
        return None

    try:
        return Decimal(number_matches[-1])
    except InvalidOperation:
        return None


def answers_match(expected: Any, actual: Any) -> bool:
    expected_num = decimal_from_text(expected)
    actual_num = decimal_from_text(actual)
    if expected_num is not None and actual_num is not None:
        return abs(expected_num - actual_num) <= Decimal("0.000001")

    return str(expected).strip().lower() == str(actual).strip().lower()


def extract_answer(raw_response: str) -> str:
    text = strip_code_fence(raw_response)

    final_line_match = re.search(
        r"FINAL_ANSWER\s*:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+/\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if final_line_match:
        return final_line_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("answer", "final_answer", "result"):
                if key in parsed:
                    return str(parsed[key]).strip()
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                for key in ("answer", "final_answer", "result"):
                    if key in parsed:
                        return str(parsed[key]).strip()
        except json.JSONDecodeError:
            pass

    answer_match = re.search(
        r"(?:answer|final answer|result)\s*[:=]\s*(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+/\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip()

    number_matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+/\d+", text)
    if not number_matches:
        raise ValueError(f"Could not extract final answer from response: {raw_response}")

    return number_matches[-1].strip()


def build_direct_prompt(question: str) -> str:
    return f"""
Solve the following grade-school math word problem.

Return ONLY a valid JSON object with this exact schema:
{{"answer": "final numeric answer"}}

Rules:
- Put only the final numeric answer in "answer".
- Do not include units.
- Do not include reasoning.
- Do not include markdown.

Question:
{question}
""".strip()


def build_cot_prompt(question: str) -> str:
    return f"""
Solve the following grade-school math word problem step by step.

Rules:
- Show the calculation steps briefly.
- Keep all arithmetic explicit.
- The final line must be exactly:
FINAL_ANSWER: <final numeric answer>
- Do not include units in FINAL_ANSWER.

Question:
{question}
""".strip()


def build_prompt(question: str, mode: str) -> str:
    if mode == "cot":
        return build_cot_prompt(question)
    return build_direct_prompt(question)


def call_openrouter(
    question: str,
    *,
    api_key: str,
    model: str,
    mode: str,
    timeout: int,
    max_tokens: int,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "GSM8K Benchmark Runner",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You solve grade-school math word problems accurately.",
            },
            {"role": "user", "content": build_prompt(question, mode)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected OpenRouter response: {data}") from exc

    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"OpenRouter returned empty content: {data}")

    return content


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


def write_results(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "question",
        "official_answer",
        "model_answer",
        "correct",
        "raw_response",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derived_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def summary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def compute_summary(
    *,
    rows: list[dict[str, Any]],
    total_limit: int,
    model: str,
    input_path: Path,
    output_path: Path,
    mode: str,
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
    accuracy = correct_count / attempted if attempted else 0.0
    progress = completed / total_limit if total_limit else 0.0

    return {
        "model": model,
        "mode": mode,
        "input": str(input_path),
        "output": str(output_path),
        "total_limit": total_limit,
        "completed_rows": completed,
        "pending_rows": max(total_limit - completed, 0),
        "successful_api_rows": attempted,
        "correct_rows": correct_count,
        "wrong_rows": wrong_count,
        "error_rows": error_count,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 4),
        "progress": progress,
        "progress_percent": round(progress * 100, 4),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_wrong_results(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    wrong_path = derived_output_path(output_path, "_wrong")
    wrong_rows = [
        row
        for row in rows
        if row.get("error", "") or row.get("correct") != "yes"
    ]
    write_results(wrong_path, wrong_rows)
    return wrong_path


def write_summary(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    total_limit: int,
    model: str,
    mode: str,
    input_path: Path,
) -> Path:
    summary_path = summary_output_path(output_path)
    summary = compute_summary(
        rows=rows,
        total_limit=total_limit,
        model=model,
        mode=mode,
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
    model: str,
    mode: str,
    input_path: Path,
) -> tuple[Path, Path]:
    write_results(output_path, rows)
    wrong_path = write_wrong_results(output_path, rows)
    summary_path = write_summary(
        output_path=output_path,
        rows=rows,
        total_limit=total_limit,
        model=model,
        mode=mode,
        input_path=input_path,
    )
    return wrong_path, summary_path


def run_one_row(
    *,
    index: int,
    total: int,
    row: dict[str, str],
    api_key: str,
    model: str,
    mode: str,
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    question = row["question"]
    official_answer = row["offical answer"]
    result_row: dict[str, Any] = {
        "index": index,
        "question": question,
        "official_answer": official_answer,
        "model_answer": "",
        "correct": "no",
        "raw_response": "",
        "error": "",
    }

    try:
        raw_response = call_openrouter(
            question,
            api_key=api_key,
            model=model,
            mode=mode,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        model_answer = extract_answer(raw_response)
        is_correct = answers_match(official_answer, model_answer)
        result_row.update(
            {
                "model_answer": model_answer,
                "correct": "yes" if is_correct else "no",
                "raw_response": raw_response,
            }
        )
        status = "OK" if is_correct else "WRONG"
        print(
            f"[{index}/{total}] {status}: expected={official_answer!r}, got={model_answer!r}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        result_row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{index}/{total}] ERROR: {result_row['error']}", flush=True)

    return result_row


def run_benchmark(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = args.model or DEFAULT_MODEL
    max_tokens = args.max_tokens if args.max_tokens is not None else 1000
    if args.mode == "direct" and args.max_tokens is None:
        max_tokens = 300

    if not api_key:
        raise EnvironmentError("Missing OPENROUTER_API_KEY in .env or environment.")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    output_path = Path(args.output or default_output_for_mode(args.mode))
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    existing = read_existing_results(output_path) if args.resume else {}
    results = [existing[index] for index in sorted(existing)]

    with input_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "question" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'question' column. Found: {reader.fieldnames}")
        if "offical answer" not in (reader.fieldnames or []):
            raise ValueError(
                f"Missing 'offical answer' column. Found: {reader.fieldnames}"
            )

        rows = list(reader)[: args.limit]

    completed = {int(row["index"]) for row in results if str(row.get("index", "")).isdigit()}
    pending_rows = [
        (zero_based_index + 1, row)
        for zero_based_index, row in enumerate(rows)
        if zero_based_index + 1 not in completed
    ]

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model}")
    print(f"Mode: {args.mode}")
    print(f"Limit: {len(rows)}")
    print(f"Workers: {args.workers}")
    print(f"Pending rows: {len(pending_rows)}")

    if args.sleep > 0 and args.workers > 1:
        print("--sleep is ignored when --workers is greater than 1.")

    if args.workers <= 1:
        for index, row in pending_rows:
            result_row = run_one_row(
                index=index,
                total=len(rows),
                row=row,
                api_key=api_key,
                model=model,
                mode=args.mode,
                timeout=args.timeout,
                max_tokens=max_tokens,
            )
            results.append(result_row)
            results.sort(key=lambda item: int(item["index"]))
            write_all_outputs(
                output_path=output_path,
                rows=results,
                total_limit=len(rows),
                model=model,
                mode=args.mode,
                input_path=input_path,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_one_row,
                    index=index,
                    total=len(rows),
                    row=row,
                    api_key=api_key,
                    model=model,
                    mode=args.mode,
                    timeout=args.timeout,
                    max_tokens=max_tokens,
                )
                for index, row in pending_rows
            ]

            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda item: int(item["index"]))
                write_all_outputs(
                    output_path=output_path,
                    rows=results,
                    total_limit=len(rows),
                    model=model,
                    mode=args.mode,
                    input_path=input_path,
                )

    attempted = len([row for row in results if row.get("error", "") == ""])
    correct_count = sum(1 for row in results if row.get("correct") == "yes")
    accuracy = correct_count / attempted if attempted else 0.0
    wrong_path, summary_path = write_all_outputs(
        output_path=output_path,
        rows=results,
        total_limit=len(rows),
        model=model,
        mode=args.mode,
        input_path=input_path,
    )
    print()
    print(f"Completed rows: {len(results)}")
    print(f"Successful API rows: {attempted}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Wrong rows output: {wrong_path}")
    print(f"Summary output: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct-answer GPT benchmark on GSM8K question rows."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Benchmark CSV path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Result CSV path. Defaults depend on --mode.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to run.")
    parser.add_argument(
        "--mode",
        choices=("direct", "cot"),
        default="direct",
        help="direct asks for final JSON only; cot asks for step-by-step work plus FINAL_ANSWER.",
    )
    parser.add_argument("--model", default=None, help="Override OPENROUTER_MODEL.")
    parser.add_argument("--timeout", type=int, default=45, help="Request timeout seconds.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens. Defaults to 300 for direct, 1000 for cot.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel OpenRouter requests. Use 3-8 carefully to avoid rate limits.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing rows in the output CSV and continue from missing rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
