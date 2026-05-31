"""
Main/Tutor.py

Pipeline tutor tự giải và tự chấm bài học sinh.

Luồng này dùng solver làm lời giải chuẩn:
1. Main/Solver.py tạo Plan.yaml và PlanEntities.yaml từ đề bài.
2. Formalizer/StudentAnswerFormalizer.py tạo StudentPlan và StudentAnswerEntities.
3. Verifier/InsideChecker.py --mode student kiểm tra nội bộ lời giải học sinh.
4. Formalizer/Mapper.py map StudentAnswerEntities với PlanEntities.
5. Verifier/CompareChecker.py so sánh lời giải solver với lời giải học sinh.
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


TUTOR_STAGES = [
    Stage(
        name="Solver",
        command=[sys.executable, str(ROOT_DIR / "Main" / "Solver.py")],
    ),
    Stage(
        name="StudentAnswerFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "StudentAnswerFormalizer.py")],
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
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "Mapper.py")],
    ),
    Stage(
        name="CompareChecker",
        command=[sys.executable, str(ROOT_DIR / "Verifier" / "CompareChecker.py")],
    ),
]


TUTOR_OUTPUTS = [
    "ProblemEntities.yaml",
    "Code.txt",
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


def clear_tutor_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in TUTOR_OUTPUTS:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-solve a problem and grade the student answer.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Không xoá output cũ trước khi chạy.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.keep_existing:
        clear_tutor_outputs()

    for stage in TUTOR_STAGES:
        run_stage(stage)

    print("Pass Tutor")


if __name__ == "__main__":
    run()
