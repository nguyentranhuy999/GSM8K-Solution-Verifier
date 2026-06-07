"""
Main/Grader.py

Pipeline chấm lời giải học sinh bằng lời giải giáo viên.

Grader là pipeline riêng với Tutor/Solver:
1. Formalizer/ProblemFormalizer.py tạo danh sách entity từ đề bài.
2. Formalizer/StudentAnswerFormalizer.py tạo StudentPlan và hoàn thiện StudentAnswerEntities.
3. Formalizer/TeacherAnswerFormalizer.py tạo TeacherPlan và hoàn thiện TeacherAnswerEntities.
4. Verifier/InsideChecker.py --mode teacher kiểm tra nội bộ lời giải giáo viên.
5. Verifier/InsideChecker.py --mode student kiểm tra nội bộ lời giải học sinh.
6. Formalizer/Mapper.py --reference teacher map StudentAnswerEntities với TeacherAnswerEntities.
7. Verifier/CompareChecker.py --reference teacher so sánh hai nhánh và ghi Diagnosis/Wrong.

Nếu bất kỳ stage symbolic nào fail, Grader gọi Verifier/LLMChecker.py --mode teacher
để fallback bằng raw problem/student/teacher. Nếu symbolic pass, Grader gọi
Verifier/LLMChecker.py --mode review; mode này chỉ gọi LLM thật khi cần kiểm tra lại
different calculation + Wrong=No. Có thể bật thêm --enable-refine để phân loại lại
wrong relationship bằng LLMChecker, nhưng mặc định tắt để benchmark backbone thuần.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Output"


@dataclass(frozen=True)
class Stage:
    name: str
    command: List[str]


class StageFailure(Exception):
    def __init__(self, stage: Stage, returncode: int) -> None:
        self.stage = stage
        self.returncode = returncode
        super().__init__(f"{stage.name} failed with exit code {returncode}")


GRADER_STAGES = [
    Stage(
        name="ProblemFormalizer",
        command=[
            sys.executable,
            str(ROOT_DIR / "Formalizer" / "ProblemFormalizer.py"),
            "--copy-targets",
            "grader",
        ],
    ),
    Stage(
        name="StudentAnswerFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "StudentAnswerFormalizer.py")],
    ),
    Stage(
        name="TeacherAnswerFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "TeacherAnswerFormalizer.py")],
    ),
    Stage(
        name="InsideCheckerTeacher",
        command=[
            sys.executable,
            str(ROOT_DIR / "Verifier" / "InsideChecker.py"),
            "--mode",
            "teacher",
        ],
    ),
    Stage(
        name="InsideCheckerStudent",
        command=[
            sys.executable,
            str(ROOT_DIR / "Verifier" / "InsideChecker.py"),
            "--mode",
            "student",
        ],
    ),
    Stage(
        name="Mapper",
        command=[
            sys.executable,
            str(ROOT_DIR / "Formalizer" / "Mapper.py"),
            "--reference",
            "teacher",
        ],
    ),
    Stage(
        name="CompareChecker",
        command=[
            sys.executable,
            str(ROOT_DIR / "Verifier" / "CompareChecker.py"),
            "--reference",
            "teacher",
        ],
    ),
]


GRADER_OUTPUTS = [
    "ProblemEntities.yaml",
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
]


def llm_checker_stage(mode: str, name: str) -> Stage:
    return Stage(
        name=name,
        command=[
            sys.executable,
            str(ROOT_DIR / "Verifier" / "LLMChecker.py"),
            "--mode",
            mode,
        ],
    )


def run_stage(stage: Stage) -> None:
    print(f"Running {stage.name}...")
    completed = subprocess.run(
        stage.command,
        cwd=str(ROOT_DIR),
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise StageFailure(stage, completed.returncode)


def read_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def write_yaml_dict(path: Path, data: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def update_log(updates: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "Log.yaml"
    log_data = read_yaml_dict(path)
    log_data.update(updates)
    write_yaml_dict(path, log_data)


def read_yaml_any(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None


def diagnosis_has_label(label: str) -> bool:
    raw = read_yaml_any(OUTPUT_DIR / "Diagnosis.yaml")
    if isinstance(raw, dict):
        items = raw.get("diagnosis", [])
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    for item in items:
        if isinstance(item, dict) and item.get("diagnosis") == label:
            return True
        if isinstance(item, str) and item.strip() == label:
            return True
    return False


def annotate_llmchecker(
    *,
    mode: str,
    reason: str,
    failed_stage: Optional[str] = None,
    failed_returncode: Optional[int] = None,
) -> bool:
    path = OUTPUT_DIR / "LLMChecker.yaml"
    data = read_yaml_dict(path)
    llm_called = bool(str(data.get("raw_response") or "").strip())
    data["grader_call"] = {
        "called_by_grader": True,
        "mode": mode,
        "reason": reason,
        "failed_stage": failed_stage,
        "failed_returncode": failed_returncode,
        "llm_called": llm_called,
    }
    write_yaml_dict(path, data)
    return llm_called


def run_llmchecker(
    *,
    mode: str,
    reason: str,
    failed_stage: Optional[str] = None,
    failed_returncode: Optional[int] = None,
) -> bool:
    update_log(
        {
            "Grader_llmchecker_stage_run": True,
            "Grader_llmchecker_mode": mode,
            "Grader_llmchecker_reason": reason,
            "Grader_failed_stage": failed_stage,
            "Grader_failed_returncode": failed_returncode,
        }
    )
    if mode == "teacher":
        stage_name = "LLMCheckerFallback"
    elif mode == "refine":
        stage_name = "LLMCheckerRefine"
    else:
        stage_name = "LLMCheckerReview"
    try:
        run_stage(llm_checker_stage(mode, stage_name))
    except StageFailure as exc:
        update_log(
            {
                "Grader": "Fail Grader while running LLMChecker",
                "Grader_llmchecker_error_stage": exc.stage.name,
                "Grader_llmchecker_error_returncode": exc.returncode,
            }
        )
        raise SystemExit(exc.returncode) from exc
    llm_called = annotate_llmchecker(
        mode=mode,
        reason=reason,
        failed_stage=failed_stage,
        failed_returncode=failed_returncode,
    )
    update_log(
        {
            "Grader_llmchecker_llm_called": llm_called,
            "Grader_llmchecker_called": llm_called,
        }
    )
    return llm_called


def clear_grader_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in GRADER_OUTPUTS:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade a student answer against a teacher answer.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Không xoá output cũ trước khi chạy.",
    )
    parser.add_argument(
        "--enable-refine",
        action="store_true",
        help="Bật LLMChecker refine cho diagnosis wrong relationship. Mặc định tắt.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.keep_existing:
        clear_grader_outputs()

    try:
        for stage in GRADER_STAGES:
            run_stage(stage)
    except StageFailure as exc:
        update_log(
            {
                "Grader": "Symbolic pipeline failed; using LLMChecker fallback",
                "Grader_symbolic_pipeline": "fail",
                "Grader_failed_stage": exc.stage.name,
                "Grader_failed_returncode": exc.returncode,
            }
        )
        run_llmchecker(
            mode="teacher",
            reason=f"{exc.stage.name} failed, so Grader used raw teacher/student fallback.",
            failed_stage=exc.stage.name,
            failed_returncode=exc.returncode,
        )
        update_log({"Grader": "Pass Grader with LLMChecker fallback"})
        print("Pass Grader with LLMChecker fallback")
        return

    update_log({"Grader_symbolic_pipeline": "pass"})
    run_llmchecker(
        mode="review",
        reason="Symbolic pipeline passed; review only if different calculation is marked with Wrong=No.",
    )
    if args.enable_refine and diagnosis_has_label("wrong relationship"):
        run_llmchecker(
            mode="refine",
            reason="Symbolic pipeline produced wrong relationship; refine into a more specific semantic label.",
        )

    print("Pass Grader")


if __name__ == "__main__":
    run()
