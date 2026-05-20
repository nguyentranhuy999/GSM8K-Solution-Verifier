"""
Main/Main.py

Pipeline đầy đủ:
1. Main/Solver.py
   - ProblemFormalizer
   - Planner
   - Executor
   - Executor tự chạy InsideChecker/repair cho lời giải chuẩn.
2. Formalizer/StudentAnswerFormalizer.py
3. Verifier/InsideChecker.py --mode student
4. Formalizer/Mapper.py
5. Verifier/CompareChecker.py

Ghi chú:
- Mapper.py là bước bắt buộc trước CompareChecker.py vì CompareChecker cần
  trường map trong PlanEntities.yaml và StudentAnswerEntities.yaml.
"""

from __future__ import annotations

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


STAGES = [
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


def run() -> None:
    for stage in STAGES:
        run_stage(stage)

    print("Pass Main")


if __name__ == "__main__":
    run()
