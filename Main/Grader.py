"""
Main/Grader.py

Pipeline chấm lời giải học sinh.

Mặc định Grader chỉ dùng reference đã có sẵn:
- Output/ProblemEntities.yaml
- Output/Plan.yaml
- Output/PlanEntities.yaml

Chạy mặc định:
1. Formalizer/StudentAnswerFormalizer.py
2. Verifier/InsideChecker.py --mode student
3. Formalizer/Mapper.py
4. Verifier/CompareChecker.py

Nếu muốn build reference trong cùng một lệnh:
- python3 Main/Grader.py --reference solver
- python3 Main/Grader.py --reference teacher

Khi đó Grader sẽ gọi Main/Tutor.py trước rồi mới chấm.
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


def tutor_stage(reference: str) -> Stage:
    return Stage(
        name="Tutor",
        command=[
            sys.executable,
            str(ROOT_DIR / "Main" / "Tutor.py"),
            "--reference",
            reference,
        ],
    )


GRADING_STAGES = [
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


GRADING_OUTPUTS = [
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


def clear_grading_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in GRADING_OUTPUTS:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def ensure_reference_exists() -> None:
    missing = [
        str(path.relative_to(ROOT_DIR))
        for path in [
            OUTPUT_DIR / "ProblemEntities.yaml",
            OUTPUT_DIR / "Plan.yaml",
            OUTPUT_DIR / "PlanEntities.yaml",
        ]
        if not path.exists()
    ]
    if missing:
        files = ", ".join(missing)
        raise SystemExit(
            "Thiếu reference để chấm: "
            f"{files}. Hãy chạy `python3 Main/Tutor.py` trước, hoặc chạy "
            "`python3 Main/Grader.py --reference solver` / "
            "`python3 Main/Grader.py --reference teacher`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade a student answer against a reference plan.")
    parser.add_argument(
        "--reference",
        choices=["existing", "solver", "teacher"],
        default="existing",
        help=(
            "existing: dùng Output/Plan.yaml hiện có; "
            "solver: gọi Tutor tự giải trước; "
            "teacher: gọi Tutor dùng Input/TeacherAnswer.txt trước."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Không xoá output chấm cũ trước khi chạy.",
    )
    return parser.parse_args()


def stages_for_args(args: argparse.Namespace) -> List[Stage]:
    stages: List[Stage] = []
    if args.reference in {"solver", "teacher"}:
        stages.append(tutor_stage(args.reference))
    else:
        ensure_reference_exists()
    stages.extend(GRADING_STAGES)
    return stages


def run() -> None:
    args = parse_args()
    if not args.keep_existing:
        clear_grading_outputs()

    for stage in stages_for_args(args):
        run_stage(stage)

    print("Pass Grader")


if __name__ == "__main__":
    run()
