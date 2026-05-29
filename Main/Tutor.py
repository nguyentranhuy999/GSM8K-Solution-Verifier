"""
Main/Tutor.py

Pipeline tạo lời giải/reference.

Mode solver:
1. Main/Solver.py
   - ProblemFormalizer
   - Planner
   - Executor

Mode teacher:
1. Formalizer/ProblemFormalizer.py
2. Formalizer/TeacherAnswerFormalizer.py

Output reference chính:
- Output/ProblemEntities.yaml
- Output/Plan.yaml
- Output/PlanEntities.yaml
- Output/Code.txt                  # nếu dùng solver/code planner
- Output/TeacherPlan.yaml          # nếu dùng teacher
- Output/TeacherAnswerEntities.yaml # nếu dùng teacher
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


SOLVER_REFERENCE_STAGES = [
    Stage(
        name="Solver",
        command=[sys.executable, str(ROOT_DIR / "Main" / "Solver.py")],
    ),
]


TEACHER_REFERENCE_STAGES = [
    Stage(
        name="ProblemFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "ProblemFormalizer.py")],
    ),
    Stage(
        name="TeacherAnswerFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "TeacherAnswerFormalizer.py")],
    ),
]


REFERENCE_OUTPUTS = [
    "ProblemEntities.yaml",
    "Code.txt",
    "Plan.yaml",
    "PlanEntities.yaml",
    "TeacherPlan.yaml",
    "TeacherAnswerEntities.yaml",
    "Error.yaml",
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


def clear_reference_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in REFERENCE_OUTPUTS:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference solution for a GSM8K problem.")
    parser.add_argument(
        "--reference",
        choices=["solver", "teacher"],
        default="solver",
        help="solver: tự giải làm chuẩn; teacher: dùng Input/TeacherAnswer.txt làm lời giải chuẩn.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Không xoá các output reference cũ trước khi chạy.",
    )
    return parser.parse_args()


def stages_for_reference(reference: str) -> List[Stage]:
    if reference == "teacher":
        return TEACHER_REFERENCE_STAGES
    return SOLVER_REFERENCE_STAGES


def run() -> None:
    args = parse_args()
    if not args.keep_existing:
        clear_reference_outputs()

    for stage in stages_for_reference(args.reference):
        run_stage(stage)

    print("Pass Tutor")


if __name__ == "__main__":
    run()
