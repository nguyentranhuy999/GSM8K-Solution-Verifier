"""
Main/Main.py

Pipeline đầy đủ.

Luồng mặc định:
1. Main/Solver.py
   - ProblemFormalizer
   - Planner
   - Executor
2. Formalizer/StudentAnswerFormalizer.py
3. Verifier/InsideChecker.py --mode student
4. Formalizer/Mapper.py
5. Verifier/CompareChecker.py

Luồng giáo viên:
python3 Main/Main.py --reference teacher
1. Formalizer/ProblemFormalizer.py
2. Formalizer/TeacherAnswerFormalizer.py
3. Formalizer/StudentAnswerFormalizer.py
4. Verifier/InsideChecker.py --mode student
5. Formalizer/Mapper.py
6. Verifier/CompareChecker.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[1]


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

COMPARE_STAGES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solution comparison pipeline.")
    parser.add_argument(
        "--reference",
        choices=["solver", "teacher"],
        default="solver",
        help="solver: tự giải làm chuẩn; teacher: dùng Input/TeacherAnswer.txt làm lời giải chuẩn.",
    )
    return parser.parse_args()


def stages_for_reference(reference: str) -> List[Stage]:
    if reference == "teacher":
        return TEACHER_REFERENCE_STAGES + COMPARE_STAGES
    return SOLVER_REFERENCE_STAGES + COMPARE_STAGES


def run() -> None:
    args = parse_args()
    for stage in stages_for_reference(args.reference):
        run_stage(stage)

    print("Pass Main")


if __name__ == "__main__":
    run()
