"""
Main/Grader.py

Pipeline chấm lời giải học sinh bằng lời giải giáo viên.

Grader là pipeline riêng với Tutor/Solver:
1. Formalizer/ProblemFormalizer.py tạo danh sách entity từ đề bài.
2. Formalizer/StudentAnswerFormalizer.py tạo StudentPlan và hoàn thiện StudentAnswerEntities.
3. Formalizer/TeacherAnswerFormalizer.py tạo TeacherPlan và hoàn thiện TeacherAnswerEntities.
4. Verifier/InsideChecker.py --mode student kiểm tra nội bộ lời giải học sinh.
5. Formalizer/Mapper.py --reference teacher map StudentAnswerEntities với TeacherAnswerEntities.
6. Verifier/CompareChecker.py --reference teacher so sánh hai nhánh và ghi Diagnosis/Wrong.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Output"


@dataclass(frozen=True)
class Stage:
    name: str
    command: List[str]


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
    "StudentPlan.yaml",
    "StudentAnswerEntities.yaml",
    "Diagnosis.yaml",
    "Wrong.yaml",
    "Error.yaml",
    "LLMChecker.yaml",
]


def run_stage(stage: Stage) -> None:
    print(f"Running {stage.name}...")
    completed = subprocess.run(
        stage.command,
        cwd=str(ROOT_DIR),
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.keep_existing:
        clear_grader_outputs()

    for stage in GRADER_STAGES:
        run_stage(stage)

    print("Pass Grader")


if __name__ == "__main__":
    run()
