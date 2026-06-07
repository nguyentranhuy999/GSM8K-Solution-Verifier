#!/usr/bin/env python3
"""Plot verifier benchmark metrics against the base LLM verifier.

Default inputs:
  - ErrorVerify/results_summary.json
  - ErrorBaseVerify/results_summary.json

Default outputs:
  - Report/benchmark_comparison.svg for the full pipeline
  - Report/benchmark_comparison_no_llm.svg for --pipeline-view no-llm

The script intentionally uses only the Python standard library so it can run in
the current project without extra plotting dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Metric = Tuple[str, str]


DEFAULT_METRICS: List[Metric] = [
    ("wrong_accuracy", "Wrong Accuracy"),
    ("wrong_f1", "Wrong F1"),
    ("error_label_hit_rate", "Label Hit Rate"),
    ("error_label_f1", "Label F1"),
    ("exact_match", "Exact Match"),
]


PIPELINE_COLOR = "#2563eb"
BASE_COLOR = "#f97316"
GRID_COLOR = "#d9dee8"
TEXT_COLOR = "#172033"
MUTED_TEXT_COLOR = "#667085"
BACKGROUND = "#ffffff"


PIPELINE_VIEWS = {
    "full": {
        "path": (),
        "display_name": "Current pipeline",
        "default_out": Path("Report/benchmark_comparison.svg"),
    },
    "no-llm": {
        "path": ("no_llmchecker_llm",),
        "display_name": "Pipeline without LLMChecker API calls",
        "default_out": Path("Report/benchmark_comparison_no_llm.svg"),
    },
    "symbolic-pass": {
        "path": ("symbolic_pipeline", "pass"),
        "display_name": "Symbolic pipeline pass",
        "default_out": Path("Report/benchmark_comparison_symbolic_pass.svg"),
    },
    "symbolic-fallback": {
        "path": ("symbolic_pipeline", "fallback"),
        "display_name": "Symbolic pipeline fallback",
        "default_out": Path("Report/benchmark_comparison_symbolic_fallback.svg"),
    },
}


def load_summary(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("metrics"), dict):
        raise ValueError(f"Missing metrics object in {path}")
    return data


def select_pipeline_view(summary: Dict, view: str) -> Dict:
    if view not in PIPELINE_VIEWS:
        raise ValueError(f"Unknown pipeline view: {view}")

    config = PIPELINE_VIEWS[view]
    selected = summary
    for key in config["path"]:
        selected = selected.get(key)
        if not isinstance(selected, dict):
            path = ".".join(config["path"])
            raise ValueError(f"Missing pipeline view {path!r} in summary")

    result = dict(selected)
    result["display_name"] = config["display_name"]
    result["model"] = summary.get("model")

    support = result.get("support")
    if isinstance(support, dict):
        result.setdefault("attempted_rows", support.get("attempted_rows"))
        result.setdefault("error_rows", support.get("error_rows"))

    if view != "full":
        result.setdefault("pipeline", config["display_name"])
    return result


def pct(value: float) -> float:
    return value * 100.0


def fmt_pct(value: float) -> str:
    return f"{pct(value):.2f}%"


def metric_values(data: Dict, metrics: Iterable[Metric]) -> Dict[str, float]:
    values = {}
    raw = data.get("metrics", {})
    for key, _label in metrics:
        if key not in raw:
            raise ValueError(f"Missing metric {key!r} in summary")
        values[key] = float(raw[key])
    return values


def summary_label(data: Dict, fallback: str) -> str:
    pipeline = str(data.get("display_name") or data.get("pipeline") or fallback)
    model = str(data.get("model") or "")
    attempted = data.get("attempted_rows")
    error_rows = data.get("error_rows")

    parts = [pipeline]
    if model:
        parts.append(model)
    if attempted is not None:
        parts.append(f"attempted={attempted}")
    if error_rows is not None:
        parts.append(f"errors={error_rows}")
    return " | ".join(parts)


def render_svg(
    pipeline: Dict,
    base: Dict,
    metrics: List[Metric],
    out_path: Path,
    title: str,
) -> None:
    pipeline_values = metric_values(pipeline, metrics)
    base_values = metric_values(base, metrics)

    width = 1180
    height = 720
    margin_left = 90
    margin_right = 60
    margin_top = 120
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline_y = margin_top + plot_height
    chart_top = margin_top

    group_count = len(metrics)
    group_width = plot_width / group_count
    bar_width = min(58, group_width * 0.25)
    gap = 12

    def x_for(index: int, side: str) -> float:
        center = margin_left + group_width * index + group_width / 2
        if side == "pipeline":
            return center - bar_width - gap / 2
        return center + gap / 2

    def y_for(value: float) -> float:
        return baseline_y - pct(value) / 100.0 * plot_height

    lines: List[str] = []
    add = lines.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
    )
    add(f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>')

    add(
        f'<text x="{margin_left}" y="46" font-family="Inter, Arial, sans-serif" '
        f'font-size="28" font-weight="700" fill="{TEXT_COLOR}">{html.escape(title)}</text>'
    )
    add(
        f'<text x="{margin_left}" y="76" font-family="Inter, Arial, sans-serif" '
        f'font-size="14" fill="{MUTED_TEXT_COLOR}">'
        f'{html.escape(summary_label(pipeline, "Pipeline"))}</text>'
    )
    add(
        f'<text x="{margin_left}" y="98" font-family="Inter, Arial, sans-serif" '
        f'font-size="14" fill="{MUTED_TEXT_COLOR}">'
        f'{html.escape(summary_label(base, "Base model"))}</text>'
    )

    for tick in range(0, 101, 20):
        y = baseline_y - tick / 100.0 * plot_height
        stroke = "#9aa4b2" if tick == 0 else GRID_COLOR
        add(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="{stroke}" stroke-width="1"/>'
        )
        add(
            f'<text x="{margin_left - 16}" y="{y + 5:.2f}" '
            f'text-anchor="end" font-family="Inter, Arial, sans-serif" '
            f'font-size="13" fill="{MUTED_TEXT_COLOR}">{tick}%</text>'
        )

    add(
        f'<text x="{margin_left - 58}" y="{chart_top + plot_height / 2}" '
        f'transform="rotate(-90 {margin_left - 58} {chart_top + plot_height / 2})" '
        f'text-anchor="middle" font-family="Inter, Arial, sans-serif" '
        f'font-size="13" fill="{MUTED_TEXT_COLOR}">Score</text>'
    )

    for index, (key, label) in enumerate(metrics):
        p_val = pipeline_values[key]
        b_val = base_values[key]
        for side, value, color in (
            ("pipeline", p_val, PIPELINE_COLOR),
            ("base", b_val, BASE_COLOR),
        ):
            x = x_for(index, side)
            y = y_for(value)
            bar_height = baseline_y - y
            add(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" rx="4" fill="{color}"/>'
            )
            add(
                f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 8, chart_top + 14):.2f}" '
                f'text-anchor="middle" font-family="Inter, Arial, sans-serif" '
                f'font-size="12" font-weight="700" fill="{TEXT_COLOR}">{fmt_pct(value)}</text>'
            )

        delta = pct(p_val - b_val)
        delta_color = "#15803d" if delta >= 0 else "#b42318"
        delta_text = f"{delta:+.2f} pts"
        center = margin_left + group_width * index + group_width / 2
        add(
            f'<text x="{center:.2f}" y="{baseline_y + 28:.2f}" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="14" '
            f'font-weight="700" fill="{TEXT_COLOR}">{html.escape(label)}</text>'
        )
        add(
            f'<text x="{center:.2f}" y="{baseline_y + 50:.2f}" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="13" '
            f'fill="{delta_color}">Pipeline {html.escape(delta_text)} vs base</text>'
        )

    legend_y = height - 54
    legend_x = margin_left
    add(
        f'<rect x="{legend_x}" y="{legend_y - 13}" width="18" height="18" '
        f'rx="4" fill="{PIPELINE_COLOR}"/>'
    )
    add(
        f'<text x="{legend_x + 28}" y="{legend_y + 2}" '
        f'font-family="Inter, Arial, sans-serif" font-size="14" '
        f'fill="{TEXT_COLOR}">Current pipeline</text>'
    )
    add(
        f'<rect x="{legend_x + 190}" y="{legend_y - 13}" width="18" height="18" '
        f'rx="4" fill="{BASE_COLOR}"/>'
    )
    add(
        f'<text x="{legend_x + 218}" y="{legend_y + 2}" '
        f'font-family="Inter, Arial, sans-serif" font-size="14" '
        f'fill="{TEXT_COLOR}">Base model</text>'
    )

    add("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(pipeline: Dict, base: Dict, metrics: List[Metric]) -> None:
    pipeline_values = metric_values(pipeline, metrics)
    base_values = metric_values(base, metrics)

    pipeline_name = pipeline.get("display_name", "Pipeline")
    print(f"Pipeline view: {pipeline_name}")
    print("Metric                 Pipeline     Base model   Delta")
    print("---------------------  -----------  -----------  ----------")
    for key, label in metrics:
        p_val = pipeline_values[key]
        b_val = base_values[key]
        delta = pct(p_val - b_val)
        print(f"{label:<21}  {fmt_pct(p_val):>10}  {fmt_pct(b_val):>10}  {delta:+.2f} pts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot current verifier benchmark metrics against base model metrics."
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("ErrorVerify/results_summary.json"),
        help="Current pipeline results_summary.json path.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("ErrorBaseVerify/results_summary.json"),
        help="Base model results_summary.json path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output SVG path.",
    )
    parser.add_argument(
        "--pipeline-view",
        choices=sorted(PIPELINE_VIEWS),
        default="full",
        help=(
            "Which pipeline metrics to compare: full, no-llm, symbolic-pass, "
            "or symbolic-fallback."
        ),
    )
    parser.add_argument(
        "--title",
        default="Verifier Benchmark Comparison",
        help="Chart title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_summary = load_summary(args.pipeline)
    pipeline = select_pipeline_view(pipeline_summary, args.pipeline_view)
    base = load_summary(args.base)
    base["display_name"] = "Base model"
    out_path = args.out or PIPELINE_VIEWS[args.pipeline_view]["default_out"]

    render_svg(
        pipeline=pipeline,
        base=base,
        metrics=DEFAULT_METRICS,
        out_path=out_path,
        title=args.title,
    )
    print_table(pipeline, base, DEFAULT_METRICS)
    print(f"\nWrote chart: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
