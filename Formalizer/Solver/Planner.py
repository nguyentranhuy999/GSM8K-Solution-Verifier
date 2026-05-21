"""
Formalizer/Solver/Planner.py

Nhiệm vụ:
- Đọc đề bài từ Input/Problem.txt
- Đọc các thực thể đề bài từ Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để sinh kế hoạch giải bài toán
- Ghi kế hoạch vào Output/Plan.yaml
- Sau khi có plan, dùng code Python để thêm các thực thể result vào Output/PlanEntities.yaml
- Ghi trạng thái Pass Planner / Fail Planner vào Output/Log.yaml

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional

Ghi chú:
- LLM chỉ sinh kế hoạch symbolic, không phải nơi thực thi số học chính.
- Các bước trong plan được đặt step1, step2, step3, ...
- Mỗi step gồm:
  - expr
  - result
  - result_unit
  - result_grand_unit
- Code có hỗ trợ đọc alias grand_result_unit từ output LLM, nhưng khi lưu sẽ chuẩn hóa thành result_grand_unit.
"""

from __future__ import annotations

import json
import os
import re
import sys
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "Input" / "Problem.txt"
OUTPUT_DIR = ROOT_DIR / "Output"
PROBLEM_ENTITIES_PATH = OUTPUT_DIR / "ProblemEntities.yaml"
PLAN_PATH = OUTPUT_DIR / "Plan.yaml"
PLAN_ENTITIES_PATH = OUTPUT_DIR / "PlanEntities.yaml"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_RETRIES = 3


class PlannerError(Exception):
    """Lỗi riêng cho Planner."""


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_yaml_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise PlannerError(f"Không tìm thấy file: {path}")
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlannerError(f"File YAML không hợp lệ: {path} - {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise PlannerError(f"File YAML phải có dạng dictionary: {path}")

    return data


def write_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    ensure_dirs()
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def write_log(status: str, message: str = "") -> None:
    ensure_dirs()
    log_data = read_yaml_file(LOG_PATH, required=False)
    log_data["Planner"] = status
    if message:
        log_data["Planner_message"] = message
    elif "Planner_message" in log_data:
        del log_data["Planner_message"]
    write_yaml_file(LOG_PATH, log_data)


def read_problem() -> str:
    if not INPUT_PATH.exists():
        raise PlannerError(f"Không tìm thấy file: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise PlannerError("Input/Problem.txt đang rỗng.")

    return text


def validate_problem_entities(entities: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    required_fields = {"value", "unit", "location", "grand_unit"}
    normalized: Dict[str, Dict[str, Any]] = {}
    target_count = 0

    if not entities:
        raise PlannerError("Output/ProblemEntities.yaml đang rỗng.")

    for name, entity in entities.items():
        if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise PlannerError(f"Tên entity không hợp lệ: {name!r}")

        if not isinstance(entity, dict):
            raise PlannerError(f"Entity {name} phải là dictionary.")

        missing = required_fields - set(entity.keys())
        if missing:
            raise PlannerError(f"Entity {name} thiếu trường: {sorted(missing)}")

        location = entity.get("location")
        if location not in {"input", "target"}:
            raise PlannerError(f"Entity {name} có location không hợp lệ: {location!r}")

        if location == "target":
            target_count += 1

        normalized[name] = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": location,
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }

    if target_count != 1:
        raise PlannerError(f"ProblemEntities phải có đúng 1 target, hiện có {target_count}.")

    return normalized


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


def build_system_prompt() -> str:
    return """
Bạn là một Planner sinh kế hoạch giải toán symbolic từ đề bài và danh sách entity đã formalize.

Nhiệm vụ:
- Sinh các bước tính toán cần thiết để đi từ entity input tới entity target.
- Không tự thêm input entity mới không có trong ProblemEntities.
- Được tạo entity trung gian bằng trường result.
- Bước cuối cùng phải tạo ra đúng entity target đã có trong ProblemEntities.
- Không thực hiện giải thích, chỉ trả YAML thuần.

Mỗi bước có đúng 4 trường:
- expr: biểu thức tính toán symbolic, chỉ dùng tên entity/result. Ví dụ: morning_coffee_price + afternoon_coffee_price
- result: tên entity được tạo ra bởi bước đó.
- result_unit: đơn vị của result. Với scalar có thể để rỗng/null.
- result_grand_unit: grand unit của result theo target. Với scalar có thể để rỗng/null.

Quy tắc step:
- Tên bước phải là step1, step2, step3, ... liên tục, không bỏ số.
- step sau được dùng result của step trước.
- result phải là snake_case.
- expr chỉ được dùng tên entity/result đã có và toán tử đơn giản: +, -, *, /, parentheses.
- Tuyệt đối không viết số literal trong expr, kể cả 0, 1, 2, 36, 60, 0.5, 1/3, hay số mũ như ** 2.
- Nếu một hệ số là dữ kiện trong đề, hệ số đó phải là entity input trong ProblemEntities.
- Nếu cần hệ số trung gian như 1, hãy tạo từ entity đã có. Ví dụ: identity_multiplier = growth_multiplier / growth_multiplier.
- Nếu cần đổi đơn vị, chỉ dùng đúng tên conversion factor đã có trong ProblemEntities, ví dụ unit_conversion_inches_per_foot.
- Không tự đặt tên chung chung như conversion_factor nếu tên đó không tồn tại trong ProblemEntities.

Quy tắc không ảo giác:
- Chỉ dùng entity trong ProblemEntities hoặc result của step trước.
- Không tạo bước không cần thiết.
- Không đưa value vào Plan.yaml.
- Không tính toán ra số trong Plan.yaml và không đưa số literal vào expr.
- Không nhân đôi dữ kiện đếm nếu các thành phần đã được liệt kê đầy đủ.
  Ví dụ: "Nancy buys 2 coffees a day", rồi đề cho giá morning coffee và afternoon coffee.
  Khi đó daily cost là morning_coffee_price + afternoon_coffee_price, KHÔNG nhân thêm coffees_per_day.
  Chỉ dùng entity đếm như coffees_per_day nếu đề cho giá của một item đơn lẻ mà chưa liệt kê từng item.
- Với phần trăm tăng/giảm như "20% heavier", "20% more", "20% less":
  không viết base * (1 + percentage) hoặc base * (1 - percentage).
  Hãy tách thành bước riêng: increase_amount = base * percentage, rồi final = base + increase_amount.
  Với giảm: decrease_amount = base * percentage, rồi final = base - decrease_amount.
- Với discount dạng "N% off if/for customers who buy at least K items":
  K chỉ là ngưỡng đủ điều kiện nhận discount, không phải số lượng thật được mua.
  Nếu đề cho số lượng thật M và M >= K, hãy tính savings bằng tổng giá không discount của M items nhân discount percentage.
  Không dùng minimum_*_for_discount để tính giá mua lẻ hoặc số lượng mua.
- Với rate theo thời gian như books_per_month/pages_per_day và đề hỏi cho một khoảng thời gian khác:
  phải đổi rate sang tổng của khoảng đó bằng conversion entity có sẵn.
  Ví dụ nếu có books_per_month và unit_conversion_months_per_year, tính total_books_needed = books_per_month * unit_conversion_months_per_year.
  Không được lấy books_per_month trừ/cộng trực tiếp với tổng theo năm.
- Nếu đề cho khoảng thời gian cụ thể như "in 7 days", "for 20 days", ProblemEntities sẽ có entity như total_days/target_days.
  Bắt buộc dùng entity đó trong plan.
  Không được thay total_days bằng unit_conversion_hours_per_day; conversion này là 24 hours/day, không phải số ngày cần tính.
- Với bài có dạng "she/he and her/his friends all get/do..." và hỏi "how many friends can she/he invite":
  target là số friends được mời, không tính người chủ.
  Trước tiên tính full cost per person gồm tất cả hoạt động/items mỗi người nhận.
  Sau đó có thể làm một trong hai cách:
  Cách A: remaining_budget = budget - full_cost_per_person, rồi friends = remaining_budget / full_cost_per_person.
  Cách B: total_people = budget / full_cost_per_person, rồi friends = total_people - host_count nếu ProblemEntities có host_count.
  Nếu đã dùng Cách A thì quotient đã là số friends, không được trừ host_count lần nữa.
  Không được đặt remaining_budget / full_cost_per_person là total_people rồi lại tính friends = total_people - host_count.
  Với total_people, result_unit và result_grand_unit phải là people, không phải dollars.
  Không được chia remaining_budget cho một thành phần riêng lẻ như mini_golf_price.
- Entity có tên dạng x_more_than_y, x_less_than_y, x_fewer_than_y là độ chênh lệch, không phải số lượng thật của x.
  Nếu cần tổng gồm cả x và y, phải tạo x trước rồi mới cộng tổng.
  Ví dụ pens_more_than_notebooks không phải số pens.
  Đúng: pens = notebooks + pens_more_than_notebooks, sau đó total_items = notebooks + pens.
  Sai: total_items = notebooks + pens_more_than_notebooks.
  Ví dụ books_borrowed_fewer_than_books_bought không phải số books_borrowed.
  Đúng: books_borrowed = books_bought - books_borrowed_fewer_than_books_bought.
  Nếu tổng gồm gifted, bought, borrowed thì total = books_gifted + books_bought + books_borrowed.
- Với chuỗi tăng/giảm theo hệ số như "each new X has r times as many as the last":
  nếu biết tổng nhiều kỳ và cần tìm kỳ đầu, dùng tổng cấp số nhân.
  Không được viết hằng 1 hoặc số mũ literal trực tiếp trong expr.
  Hãy tạo hệ số đầu bằng identity_multiplier = growth_multiplier / growth_multiplier.
  Hệ số kỳ sau tạo bằng phép nhân entity/result, ví dụ third_multiplier = growth_multiplier * growth_multiplier.
  Sau đó cộng các multiplier đã tạo để chia tổng.
  Không được lấy total / count rồi nhân/chia với growth_multiplier; đó là trung bình, không phải kỳ đầu.

Quy tắc đơn vị:
- result_unit là đơn vị trực tiếp của result.
- result_grand_unit là đơn vị đối chiếu theo target.
- Nếu result là target thì result_unit và result_grand_unit phải khớp với unit và grand_unit của target.

Định dạng output bắt buộc:
step1:
  expr: entity_a + entity_b
  result: intermediate_entity
  result_unit: dollars
  result_grand_unit: dollars

step2:
  expr: intermediate_entity * days
  result: target_entity
  result_unit: dollars
  result_grand_unit: dollars

Chỉ trả YAML thuần, không Markdown, không ``` và không giải thích.
""".strip()


def build_user_prompt(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    entities_yaml = yaml.safe_dump(
        entities,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    retry_note = ""
    if previous_error:
        retry_note = f"""

Output trước bị reject vì lỗi:
{previous_error}

Hãy sinh lại plan, chỉ sửa lỗi đó và giữ schema.
""".rstrip()

    return f"""
Hãy sinh kế hoạch giải toán cho đề bài sau.

Đề bài:
{problem}

ProblemEntities.yaml:
{entities_yaml}
{retry_note}
""".strip()


def call_openrouter(
    problem: str,
    entities: Dict[str, Dict[str, Any]],
    previous_error: Optional[str] = None,
) -> str:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise PlannerError("Thiếu OPENROUTER_API_KEY trong file .env hoặc biến môi trường.")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "GSM8K-Solution-Verifier"),
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(problem, entities, previous_error=previous_error),
            },
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise PlannerError(f"Không gọi được OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        raise PlannerError(f"OpenRouter trả lỗi {response.status_code}: {response.text[:1000]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise PlannerError(f"Response OpenRouter không đúng định dạng: {response.text[:1000]}") from exc


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:yaml|yml)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_plan(text: str) -> Dict[str, Any]:
    clean_text = strip_markdown_fence(text)

    try:
        parsed = yaml.safe_load(clean_text)
    except yaml.YAMLError as exc:
        raise PlannerError(f"LLM trả về YAML không hợp lệ: {exc}") from exc

    if not isinstance(parsed, dict) or not parsed:
        raise PlannerError("Plan output phải là dictionary không rỗng.")

    return parsed


NUMERIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)

RELATIVE_DIFFERENCE_MARKERS = {
    "_more_than_": "+",
    "_greater_than_": "+",
    "_less_than_": "-",
    "_fewer_than_": "-",
}

RELATIVE_DELTA_PREFIXES = {
    "additional_": "+",
    "extra_": "+",
    "more_": "+",
    "fewer_": "-",
    "less_": "-",
}

RATE_PERIOD_CONVERSIONS = {
    "month": "unit_conversion_months_per_year",
}

TIME_UNITS = {
    "second",
    "seconds",
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
}


def extract_expr_tokens(expr: str) -> List[str]:
    """Lấy các token giống tên biến trong expr."""
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)


def extract_numeric_literals(expr: str) -> List[str]:
    """Lấy các số literal trong expr, bỏ qua chữ số nằm trong tên entity."""
    return [match.group(0) for match in NUMERIC_LITERAL_RE.finditer(expr)]


def validate_no_numeric_literals_in_expr(
    step_name: str,
    expr: str,
    *,
    problem: Optional[str] = None,
    target_name: Optional[str] = None,
    target: Optional[Dict[str, Any]] = None,
) -> None:
    numeric_literals = extract_numeric_literals(expr)
    if numeric_literals:
        expr_token_set = set(extract_expr_tokens(expr))
        if "1" in numeric_literals and any(
            marker in token
            for token in expr_token_set
            for marker in ("percent", "percentage")
        ):
            raise PlannerError(
                f"{step_name}.expr không được dùng số 1 để tạo multiplier phần trăm. "
                "Với phần trăm tăng/giảm, hãy tách bước: change_amount = base * percentage, "
                "rồi final = base + change_amount hoặc final = base - change_amount."
            )

        if (
            "1" in numeric_literals
            and target_name
            and target
            and looks_like_invited_friends_problem(problem, target_name, target)
        ):
            raise PlannerError(
                f"{step_name}.expr không được viết số 1 trực tiếp để trừ người chủ. "
                "Nếu ProblemEntities có host_count, hãy dùng trực tiếp host_count. "
                "Nếu chưa có, hãy tạo identity từ full_cost_per_person."
            )

        raise PlannerError(
            f"{step_name}.expr không được chứa số literal {numeric_literals}; "
            "expr chỉ được dùng entity/result và toán tử. "
            "Nếu cần hệ số 1, hãy tạo identity từ entity đã có, ví dụ multiplier / multiplier."
        )


def parse_relative_difference_entity_name(name: str) -> Optional[Tuple[str, str, str]]:
    for marker, operator_symbol in RELATIVE_DIFFERENCE_MARKERS.items():
        if marker not in name:
            continue
        left_name, right_name = name.split(marker, 1)
        if left_name and right_name:
            return left_name, right_name, operator_symbol
    return None


def parse_relative_delta_entity_name(name: str) -> Optional[Tuple[str, str]]:
    for prefix, operator_symbol in RELATIVE_DELTA_PREFIXES.items():
        if not name.startswith(prefix):
            continue
        quantity_name = name[len(prefix):]
        if quantity_name:
            return quantity_name, operator_symbol
    return None


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return f"{token[:-3]}y"
    if token.endswith("es") and len(token) > 2:
        return token[:-2]
    if token.endswith("s") and len(token) > 1:
        return token[:-1]
    return token


def entity_name_terms(name: str) -> set[str]:
    terms = {term for term in name.split("_") if term}
    terms.update(singularize_token(term) for term in list(terms))
    return terms


def result_name_mentions_quantity(result_name: str, quantity_name: str) -> bool:
    if result_name == quantity_name:
        return True

    quantity_terms = entity_name_terms(quantity_name)
    result_terms = entity_name_terms(result_name)
    return bool(quantity_terms) and quantity_terms.issubset(result_terms)


def validate_relative_difference_entities_resolved(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    """
    Bắt lỗi dùng "x more than y" như số lượng thật của x.

    Entity pens_more_than_notebooks biểu diễn offset 50, không phải 50 pens.
    Plan phải có bước tạo pens trước khi dùng pens trong tổng.
    """
    for step_name, step in plan.items():
        expr = step["expr"]
        result = step["result"]
        tokens = extract_expr_tokens(expr)

        for token in tokens:
            if token not in problem_entities:
                continue

            parsed = parse_relative_difference_entity_name(token)
            delta_parsed = parse_relative_delta_entity_name(token)
            if parsed:
                compared_quantity, reference_quantity, operator_symbol = parsed
                relationship_hint = f" so với {reference_quantity!r}"
                example_expr = f"{compared_quantity} = {reference_quantity} {operator_symbol} {token}"
            elif delta_parsed:
                compared_quantity, operator_symbol = delta_parsed
                relationship_hint = ""
                example_expr = f"{compared_quantity} = base_quantity {operator_symbol} {token}"
            else:
                continue

            if result_name_mentions_quantity(result, compared_quantity):
                if operator_symbol not in expr:
                    raise PlannerError(
                        f"{step_name}.expr dùng {token!r} nhưng thiếu phép {operator_symbol!r} "
                        f"để tạo số lượng thật {compared_quantity!r}{relationship_hint}."
                    )
                continue

            raise PlannerError(
                f"{step_name}.expr dùng {token!r} như một số lượng thật. "
                f"{token!r} là độ chênh lệch, nên phải tạo entity {compared_quantity!r} trước: "
                f"{example_expr}. "
                f"Sau đó mới dùng {compared_quantity!r} để tính {result!r}."
            )


def result_step_by_entity(plan: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    return {
        step["result"]: (step_name, step)
        for step_name, step in plan.items()
    }


def target_step(plan: Dict[str, Dict[str, Any]], target_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    for step_name, step in plan.items():
        if step.get("result") == target_name:
            return step_name, step
    return None


def step_lineage_to_result(
    plan: Dict[str, Dict[str, Any]],
    result_name: str,
) -> Dict[str, Dict[str, Any]]:
    by_result = result_step_by_entity(plan)
    lineage: Dict[str, Dict[str, Any]] = {}

    def visit(entity_name: str) -> None:
        produced = by_result.get(entity_name)
        if not produced:
            return

        step_name, step = produced
        if step_name in lineage:
            return

        lineage[step_name] = step
        for token in extract_expr_tokens(step["expr"]):
            visit(token)

    visit(result_name)
    return lineage


def ast_contains_token(node: ast.AST, token_name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == token_name for child in ast.walk(node))


def ast_expr_tokens(node: ast.AST) -> List[str]:
    return [
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    ]


def ast_subtracts_token(node: ast.AST, token_name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Sub):
            if ast_contains_token(child.right, token_name):
                return True
    return False


def expr_subtracts_token(expr: str, token_name: str) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    return ast_subtracts_token(tree.body, token_name)


def division_numerators_by_denominator(expr: str, denominator_token: str) -> List[ast.AST]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []

    numerators: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if ast_contains_token(node.right, denominator_token):
                numerators.append(node.left)

    return numerators


def expr_has_division_by_token(expr: str, token_name: str) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if ast_contains_token(node.right, token_name):
                return True
    return False


def looks_like_invited_friends_problem(
    problem: Optional[str],
    target_name: str,
    target: Dict[str, Any],
) -> bool:
    if not problem:
        return False

    text = problem.lower()
    target_text = f"{target_name} {normalize_empty(target.get('unit')) or ''}".lower()
    if "friend" not in target_text:
        return False
    if "invite" not in text:
        return False

    return bool(
        re.search(
            r"\b(?:she|he|they|we|i|[a-z]+)\s+and\s+"
            r"(?:her|his|their|our|my)\s+friends\b",
            text,
        )
    )


def full_money_cost_results(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    target_name: str,
) -> List[str]:
    results: List[str] = []
    for step in plan.values():
        result = step["result"]
        if result == target_name:
            continue
        if not is_money_unit(step.get("result_unit")):
            continue

        money_inputs = [
            token
            for token in extract_expr_tokens(step["expr"])
            if token in problem_entities and is_money_unit(problem_entities[token].get("unit"))
        ]
        if len(set(money_inputs)) >= 2:
            results.append(result)

    return results


def validate_invited_friends_uses_full_per_person_cost(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]

    if not looks_like_invited_friends_problem(problem, target_name, target):
        return

    candidate_cost_results = full_money_cost_results(plan, problem_entities, target_name)
    if not candidate_cost_results:
        return

    lineage = step_lineage_to_result(plan, target_name)
    for step in lineage.values():
        if any(expr_has_division_by_token(step["expr"], cost_result) for cost_result in candidate_cost_results):
            return

    raise PlannerError(
        "Bài hỏi số friends được mời trong nhóm gồm người chủ và friends. "
        f"Plan phải chia cho full cost per person {candidate_cost_results}, "
        "không được chia cho một thành phần riêng lẻ như mini_golf_price. "
        "Có thể tính remaining_budget = budget - full_cost_per_person rồi friends = remaining_budget / full_cost_per_person."
    )


def normalize_invited_friends_people_units(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]

    if not looks_like_invited_friends_problem(problem, target_name, target):
        return

    candidate_cost_results = full_money_cost_results(plan, problem_entities, target_name)
    if not candidate_cost_results:
        return

    for step in plan.values():
        result = step["result"]
        if "people" not in entity_name_terms(result):
            continue
        if any(expr_has_division_by_token(step["expr"], cost_result) for cost_result in candidate_cost_results):
            step["result_unit"] = "people"
            step["result_grand_unit"] = "people"


def is_money_unit(unit: Any) -> bool:
    unit_text = str(normalize_empty(unit) or "").lower()
    return unit_text in {"dollar", "dollars", "usd", "$", "cent", "cents"}


def is_count_unit(unit: Any) -> bool:
    unit_text = str(normalize_empty(unit) or "").lower()
    if not unit_text:
        return False
    if is_money_unit(unit_text):
        return False
    if unit_text in {"day", "days", "hour", "hours", "minute", "minutes", "week", "weeks"}:
        return False
    return True


def normalize_item_target_result_grand_units(
    plan: Dict[str, Dict[str, Any]],
    target_unit: Any,
    target_grand_unit: Any,
) -> None:
    target_unit_text = str(normalize_empty(target_unit) or "").strip().lower()
    if target_unit_text not in {"item", "items"}:
        return

    grand_unit = normalize_empty(target_grand_unit) or target_unit
    for step in plan.values():
        if is_count_unit(step.get("result_unit")):
            step["result_grand_unit"] = grand_unit


def entity_unit(name: str, problem_entities: Dict[str, Dict[str, Any]], plan: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if name in problem_entities:
        return normalize_empty(problem_entities[name].get("unit"))

    for step in plan.values():
        if step.get("result") == name:
            return normalize_empty(step.get("result_unit"))

    return None


def result_expr(name: str, plan: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for step in plan.values():
        if step.get("result") == name:
            return normalize_empty(step.get("expr"))
    return None


def operand_or_dependencies_subtract_token(
    operand: ast.AST,
    plan: Dict[str, Dict[str, Any]],
    token_name: str,
    seen: Optional[set[str]] = None,
) -> bool:
    if ast_subtracts_token(operand, token_name):
        return True

    if seen is None:
        seen = set()

    for name in ast_expr_tokens(operand):
        if name in seen:
            continue
        seen.add(name)

        previous_expr = result_expr(name, plan)
        if not previous_expr:
            continue

        try:
            previous_tree = ast.parse(previous_expr, mode="eval")
        except SyntaxError:
            continue

        if operand_or_dependencies_subtract_token(previous_tree.body, plan, token_name, seen):
            return True

    return False


def validate_invited_friends_no_double_host_subtraction(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]

    if not looks_like_invited_friends_problem(problem, target_name, target):
        return
    if "host_count" not in problem_entities:
        return

    lineage = step_lineage_to_result(plan, target_name)
    subtracts_host = any(
        expr_subtracts_token(step["expr"], "host_count")
        for step in lineage.values()
    )
    if not subtracts_host:
        return

    candidate_cost_results = full_money_cost_results(plan, problem_entities, target_name)
    for step_name, step in lineage.items():
        for cost_result in candidate_cost_results:
            numerators = division_numerators_by_denominator(step["expr"], cost_result)
            if not numerators:
                continue

            if any(
                operand_or_dependencies_subtract_token(numerator, plan, cost_result)
                for numerator in numerators
            ):
                raise PlannerError(
                    "Plan đang trừ người chủ hai lần. "
                    f"{step_name}.expr chia phần budget còn lại cho {cost_result!r}, "
                    "tức là chi phí của người chủ đã bị trừ trước đó. "
                    "Nếu dùng remaining_budget = budget - full_cost_per_person thì bước chia ra luôn là friends, "
                    "không được trừ tiếp host_count. "
                    "Hoặc dùng total_people = budget / full_cost_per_person rồi mới friends = total_people - host_count."
                )


def validate_no_obvious_double_count(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    """
    Bắt lỗi phổ biến: đã cộng giá của từng item trong ngày rồi lại nhân với
    count "2 items per day". Ví dụ coffee: (morning + afternoon) * coffees_per_day.
    """
    for step_name, step in plan.items():
        if not is_money_unit(step.get("result_unit")):
            continue

        expr = step["expr"]
        if "*" not in expr:
            continue

        tokens = extract_expr_tokens(expr)
        count_tokens = [
            token
            for token in tokens
            if token in problem_entities and is_count_unit(problem_entities[token].get("unit"))
        ]
        if not count_tokens:
            continue

        for token in tokens:
            previous_expr = result_expr(token, plan)
            if not previous_expr:
                continue

            previous_tokens = extract_expr_tokens(previous_expr)
            money_inputs = [
                previous_token
                for previous_token in previous_tokens
                if previous_token in problem_entities and is_money_unit(problem_entities[previous_token].get("unit"))
            ]
            if len(set(money_inputs)) >= 2:
                raise PlannerError(
                    f"{step_name}.expr có vẻ nhân đôi dữ kiện đếm {count_tokens}. "
                    f"Step trước {token!r} đã cộng các giá tiền thành phần {sorted(set(money_inputs))}."
                )


def validate_important_scalar_inputs_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    important_markers = ("fraction", "percent", "percentage", "ratio", "multiplier", "factor", "rate")
    unused = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        if normalize_empty(entity.get("unit")) is not None:
            continue
        if not any(marker in name for marker in important_markers):
            continue
        if name not in plan_tokens:
            unused.append(name)

    if unused:
        raise PlannerError(
            f"Plan chưa dùng các scalar input quan trọng: {unused}. "
            "Nếu entity này biểu diễn fraction/ratio/multiplier/rate từ đề, expr phải tham chiếu trực tiếp entity đó."
        )


def normalized_unit(value: Any) -> Optional[str]:
    value = normalize_empty(value)
    if value is None:
        return None
    return str(value).strip().lower()


def is_discount_threshold_entity(name: str, entity: Dict[str, Any]) -> bool:
    if entity.get("location") != "input":
        return False
    if not is_count_unit(entity.get("unit")):
        return False

    terms = entity_name_terms(name)
    if "discount" not in terms and "discount" not in name:
        return False

    return bool({"minimum", "min", "threshold", "least"} & terms) or "_for_discount" in name


def target_looks_like_money_savings(target_name: str, target: Dict[str, Any]) -> bool:
    if not is_money_unit(target.get("unit")):
        return False

    terms = entity_name_terms(target_name)
    return bool({"saving", "savings", "save", "saved", "discount"} & terms)


def looks_like_percentage_threshold_discount_problem(
    problem: Optional[str],
    problem_entities: Dict[str, Dict[str, Any]],
) -> bool:
    if not problem:
        return False

    text = problem.lower()
    if "discount" not in text:
        return False
    if "at least" not in text:
        return False

    has_discount_percentage = any(
        entity.get("location") == "input"
        and normalize_empty(entity.get("unit")) is None
        and "discount" in entity_name_terms(name)
        and ("percent" in entity_name_terms(name) or "percentage" in entity_name_terms(name))
        for name, entity in problem_entities.items()
    )
    return "%" in text or has_discount_percentage


def threshold_has_actual_quantity(
    threshold_name: str,
    threshold: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
) -> bool:
    threshold_unit = normalized_unit(threshold.get("unit"))
    threshold_value = normalize_empty(threshold.get("value"))
    if threshold_unit is None or not is_number(threshold_value):
        return False

    threshold_number = parse_numeric(threshold_value)

    for name, entity in problem_entities.items():
        if name == threshold_name:
            continue
        if entity.get("location") != "input":
            continue
        if normalized_unit(entity.get("unit")) != threshold_unit:
            continue

        value = normalize_empty(entity.get("value"))
        if not is_number(value):
            continue
        if parse_numeric(value) >= threshold_number:
            return True

    return False


def validate_discount_threshold_not_used_as_quantity(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not looks_like_percentage_threshold_discount_problem(problem, problem_entities):
        return

    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    if not target_looks_like_money_savings(target_name, target):
        return

    threshold_names = [
        name
        for name, entity in problem_entities.items()
        if is_discount_threshold_entity(name, entity)
        and threshold_has_actual_quantity(name, entity, problem_entities)
    ]
    if not threshold_names:
        return

    lineage = step_lineage_to_result(plan, target_name)
    used_thresholds: set[str] = set()
    for step in lineage.values():
        used_thresholds.update(
            token
            for token in extract_expr_tokens(step["expr"])
            if token in threshold_names
        )

    if used_thresholds:
        raise PlannerError(
            f"Plan đang dùng ngưỡng discount {sorted(used_thresholds)} như số lượng mua thật. "
            "Với dạng 'N% off when buying at least K items', K chỉ là điều kiện đủ để nhận discount. "
            "Khi đề đã cho số lượng thật M và M >= K, hãy tính savings từ M items, "
            "ví dụ total_without_discount = unit_price * actual_quantity, "
            "rồi total_savings = total_without_discount * discount_percentage."
        )


def rate_period_from_entity_name(name: str) -> Optional[str]:
    match = re.search(r"_per_(month|day|hour|minute|second|week|year)s?$", name)
    if not match:
        return None
    return match.group(1)


def validate_time_rate_conversions_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    missing: List[Tuple[str, str]] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue

        period = rate_period_from_entity_name(name)
        if not period:
            continue

        conversion_name = RATE_PERIOD_CONVERSIONS.get(period)
        if not conversion_name or conversion_name not in problem_entities:
            continue

        if name in plan_tokens and conversion_name not in plan_tokens:
            missing.append((name, conversion_name))

    if missing:
        detail = ", ".join(f"{rate} cần {conversion}" for rate, conversion in missing)
        raise PlannerError(
            f"Plan dùng rate theo thời gian nhưng chưa đổi sang khoảng thời gian của đề: {detail}. "
            "Ví dụ books_per_month phải nhân unit_conversion_months_per_year nếu đề hỏi cho this year."
        )


def validate_required_horizon_inputs_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    missing: List[str] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        if name.startswith("unit_conversion_"):
            continue

        unit = str(normalize_empty(entity.get("unit")) or "").lower()
        if unit not in TIME_UNITS:
            continue
        if not (name.startswith("total_") or name.startswith("target_")):
            continue
        if name not in plan_tokens:
            missing.append(name)

    if missing:
        raise PlannerError(
            f"Plan chưa dùng input khoảng thời gian cần tính: {missing}. "
            "Nếu đề nói 'in 7 days' hoặc 'for 20 days', phải dùng entity này; "
            "không được thay bằng unit_conversion_hours_per_day."
        )


def target_entity_name(problem_entities: Dict[str, Dict[str, Any]]) -> str:
    targets = [name for name, entity in problem_entities.items() if entity.get("location") == "target"]
    if len(targets) != 1:
        raise PlannerError(f"Phải có đúng 1 target, hiện có {len(targets)}.")
    return targets[0]


def normalize_step_fields(step_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise PlannerError(f"{step_name} phải là dictionary.")

    # Hỗ trợ alias do prompt trước đó từng dùng nhầm tên.
    if "grand_result_unit" in step and "result_grand_unit" not in step:
        step["result_grand_unit"] = step.pop("grand_result_unit")

    required_fields = {"expr", "result", "result_unit", "result_grand_unit"}
    fields = set(step.keys())
    missing = required_fields - fields
    extra = fields - required_fields

    if missing:
        raise PlannerError(f"{step_name} thiếu trường: {sorted(missing)}")
    if extra:
        raise PlannerError(f"{step_name} có trường thừa: {sorted(extra)}")

    expr = step["expr"]
    result = step["result"]

    if not isinstance(expr, str) or not expr.strip():
        raise PlannerError(f"{step_name}.expr phải là string không rỗng.")
    if not isinstance(result, str) or not re.match(r"^[a-z][a-z0-9_]*$", result):
        raise PlannerError(f"{step_name}.result phải là snake_case hợp lệ.")

    result_unit = normalize_empty(step.get("result_unit"))
    result_grand_unit = normalize_empty(step.get("result_grand_unit"))

    if result_unit is not None:
        result_unit = str(result_unit).strip()
    if result_grand_unit is not None:
        result_grand_unit = str(result_grand_unit).strip()

    return {
        "expr": expr.strip(),
        "result": result.strip(),
        "result_unit": result_unit,
        "result_grand_unit": result_grand_unit,
    }


def validate_and_normalize_plan(
    raw_plan: Dict[str, Any],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    available_entities = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
    }
    normalized_plan: Dict[str, Dict[str, Any]] = {}

    expected_step_names = [f"step{i}" for i in range(1, len(raw_plan) + 1)]
    actual_step_names = list(raw_plan.keys())

    if actual_step_names != expected_step_names:
        raise PlannerError(
            f"Step phải liên tục theo thứ tự {expected_step_names}, hiện là {actual_step_names}."
        )

    for step_index, step_name in enumerate(expected_step_names, start=1):
        step = normalize_step_fields(step_name, raw_plan[step_name])

        if "=" in step["expr"]:
            raise PlannerError(f"{step_name}.expr không được chứa dấu '='; expr phải là biểu thức tính toán.")

        validate_no_numeric_literals_in_expr(
            step_name,
            step["expr"],
            problem=problem,
            target_name=target_name,
            target=target,
        )

        expr_tokens = extract_expr_tokens(step["expr"])
        unknown_tokens = [token for token in expr_tokens if token not in available_entities]
        if unknown_tokens:
            raise PlannerError(
                f"{step_name}.expr dùng entity chưa tồn tại hoặc chưa được tạo: {unknown_tokens}"
            )

        result = step["result"]
        if result in available_entities and result != target_name:
            raise PlannerError(
                f"{step_name}.result {result!r} đã tồn tại trong input entity, không nên ghi đè."
            )

        if step_index < len(expected_step_names) and result == target_name:
            raise PlannerError("Target chỉ nên được tạo ở bước cuối cùng.")

        available_entities.add(result)
        normalized_plan[step_name] = step

    last_step = normalized_plan[expected_step_names[-1]]
    if last_step["result"] != target_name:
        raise PlannerError(
            f"Bước cuối phải tạo target {target_name!r}, hiện tạo {last_step['result']!r}."
        )

    target_unit = normalize_empty(target.get("unit"))
    target_grand_unit = normalize_empty(target.get("grand_unit"))

    if normalize_empty(last_step.get("result_unit")) != target_unit:
        raise PlannerError(
            f"result_unit của bước cuối phải khớp target.unit: {target_unit!r}."
        )

    if normalize_empty(last_step.get("result_grand_unit")) != target_grand_unit:
        raise PlannerError(
            f"result_grand_unit của bước cuối phải khớp target.grand_unit: {target_grand_unit!r}."
        )

    normalize_item_target_result_grand_units(normalized_plan, target_unit, target_grand_unit)
    normalize_invited_friends_people_units(normalized_plan, problem_entities, problem)

    validate_no_obvious_double_count(normalized_plan, problem_entities)
    validate_important_scalar_inputs_used(normalized_plan, problem_entities)
    validate_required_horizon_inputs_used(normalized_plan, problem_entities)
    validate_time_rate_conversions_used(normalized_plan, problem_entities)
    validate_relative_difference_entities_resolved(normalized_plan, problem_entities)
    validate_discount_threshold_not_used_as_quantity(normalized_plan, problem_entities, problem)
    validate_invited_friends_no_double_host_subtraction(normalized_plan, problem_entities, problem)
    validate_invited_friends_uses_full_per_person_cost(normalized_plan, problem_entities, problem)

    return normalized_plan


def max_retries() -> int:
    raw = os.getenv("PLANNER_MAX_RETRIES")
    if raw is None:
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise PlannerError("PLANNER_MAX_RETRIES phải là số nguyên.") from exc
    if value < 1:
        raise PlannerError("PLANNER_MAX_RETRIES phải >= 1.")
    return value


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", ""))
            return True
        except ValueError:
            return False
    return False


def parse_numeric(value: Any) -> float | int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    number = float(str(value).replace(",", ""))
    if number.is_integer():
        return int(number)
    return number


def safe_eval_conversion_expr(expr: str, entities: Dict[str, Dict[str, Any]]) -> Optional[float | int]:
    """
    Thử tính value cho các bước đổi đơn vị đơn giản.

    Hàm này không bắt buộc mọi step đều tính được value.
    Nó chỉ tính khi:
    - expr chỉ gồm số, toán tử đơn giản, ngoặc, và entity đã có value số.
    - không dùng function/call/import/attribute.

    Nếu không đủ điều kiện, trả None.
    """
    allowed_chars = set("0123456789.+-*/() _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if any(ch not in allowed_chars for ch in expr):
        return None

    tokens = extract_expr_tokens(expr)
    local_values: Dict[str, float | int] = {}

    for token in tokens:
        entity = entities.get(token)
        if not entity:
            return None
        value = normalize_empty(entity.get("value"))
        if not is_number(value):
            return None
        local_values[token] = parse_numeric(value)

    try:
        result = eval(expr, {"__builtins__": {}}, local_values)  # noqa: S307 - sandboxed simple arithmetic
    except Exception:
        return None

    if not is_number(result):
        return None

    return parse_numeric(result)


def looks_like_unit_conversion(step: Dict[str, Any], existing_entities: Dict[str, Dict[str, Any]]) -> bool:
    """
    Heuristic để nhận diện bước đổi đơn vị.

    Vì Planner không phải executor chính, hàm này chỉ dùng để thêm value cho biến mới
    khi step trông giống đổi đơn vị và có thể tính an toàn.
    """
    expr = step["expr"]
    result_unit = normalize_empty(step.get("result_unit"))
    result_grand_unit = normalize_empty(step.get("result_grand_unit"))

    if result_unit is None:
        return False

    expr_tokens = extract_expr_tokens(expr)
    if len(expr_tokens) != 1:
        return False

    source_name = expr_tokens[0]
    source = existing_entities.get(source_name)
    if not source:
        return False

    source_unit = normalize_empty(source.get("unit"))
    if source_unit is None:
        return False

    if source_unit == result_unit:
        return False

    # Nếu grand unit đổi theo cùng target hoặc unit trực tiếp đổi khác nhau, coi là đổi đơn vị.
    return True or result_grand_unit is not None


def merge_plan_results_into_plan_entities(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Thêm result của từng step vào PlanEntities.yaml bằng Python.

    Quy tắc:
    - Nếu result chưa có entity: thêm mới.
    - Nếu result đã có entity target: giữ nguyên location target.
    - Nếu là bước đổi đơn vị và có thể tính toán an toàn: thêm value cho biến result.
    - Với step bình thường: value để rỗng/null vì việc thực thi số học nên thuộc module executor/solver khác.
    """
    plan_entities: Dict[str, Dict[str, Any]] = {
        name: {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": entity.get("location"),
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }
        for name, entity in problem_entities.items()
    }

    for step_name, step in plan.items():
        result = step["result"]
        result_unit = normalize_empty(step.get("result_unit"))
        result_grand_unit = normalize_empty(step.get("result_grand_unit"))

        computed_value: Optional[float | int] = None
        if looks_like_unit_conversion(step, plan_entities):
            computed_value = safe_eval_conversion_expr(step["expr"], plan_entities)

        if result not in plan_entities:
            plan_entities[result] = {
                "value": computed_value,
                "unit": result_unit,
                "location": step_name,
                "grand_unit": result_grand_unit,
            }
        else:
            old_location = plan_entities[result].get("location")
            plan_entities[result]["unit"] = result_unit
            if old_location != "target":
                plan_entities[result]["location"] = step_name
            plan_entities[result]["grand_unit"] = result_grand_unit

            # Không tự tính target value ở Planner, trừ khi chính nó là bước đổi đơn vị đơn giản.
            if computed_value is not None:
                plan_entities[result]["value"] = computed_value
            else:
                plan_entities[result]["value"] = normalize_empty(plan_entities[result].get("value"))

    return plan_entities


def run() -> None:
    try:
        ensure_dirs()
        problem = read_problem()
        raw_problem_entities = read_yaml_file(PROBLEM_ENTITIES_PATH, required=True)
        problem_entities = validate_problem_entities(raw_problem_entities)

        previous_error: Optional[str] = None
        last_validation_error: Optional[Exception] = None
        plan: Optional[Dict[str, Dict[str, Any]]] = None

        for _ in range(max_retries()):
            raw_response = call_openrouter(
                problem,
                problem_entities,
                previous_error=previous_error,
            )

            try:
                raw_plan = parse_plan(raw_response)
                candidate_plan = validate_and_normalize_plan(raw_plan, problem_entities, problem=problem)
            except PlannerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            plan = candidate_plan
            break

        if plan is None:
            raise PlannerError(str(last_validation_error))

        write_yaml_file(PLAN_PATH, plan)

        plan_entities = merge_plan_results_into_plan_entities(plan, problem_entities)
        write_yaml_file(PLAN_ENTITIES_PATH, plan_entities)

        write_log("Pass Planner")
        print("Pass Planner")
    except Exception as exc:
        write_log("Fail Planner", str(exc))
        print("Fail Planner")
        print(f"Reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
