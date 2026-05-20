"""
Main/Solver.py

Pipeline giải bài toán đến bước solver/reference:
1. Formalizer/ProblemFormalizer.py
2. Formalizer/Solver/Planer.py
3. Formalizer/Solver/Executor.py

File này chỉ chạy pipeline lời giải chuẩn/LLM. Nó không chạy phần student,
mapper, compare, hint.

Ghi chú:
- Executor.py đã tự gọi Verifier/InsideChecker.py và repair bằng LLM nếu có lỗi,
  nên Solver.py không chạy InsideChecker thêm lần nữa.
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
        name="ProblemFormalizer",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "ProblemFormalizer.py")],
    ),
    Stage(
        name="Planner",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "Solver" / "Planer.py")],
    ),
    Stage(
        name="Executor",
        command=[sys.executable, str(ROOT_DIR / "Formalizer" / "Solver" / "Executor.py")],
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

    print("Pass Solver")


if __name__ == "__main__":
    run()
