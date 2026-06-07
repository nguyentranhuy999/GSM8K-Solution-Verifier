"""
Formalizer/Solver/Planner.py

Nhiệm vụ:
- Đọc đề bài từ Input/Problem.txt
- Đọc các thực thể đề bài từ Output/ProblemEntities.yaml
- Gọi LLM qua OpenRouter để sinh lời giải dạng pseudo-code trong Output/Code.txt
- Dùng code Python map số literal trong pseudo-code về ProblemEntities để sinh Plan.yaml
- Ghi kế hoạch vào Output/Plan.yaml
- Sau khi có plan, dùng code Python để thêm các thực thể result vào Output/PlanEntities.yaml
- Ghi trạng thái Pass Planner / Fail Planner vào Output/Log.yaml

Yêu cầu .env:
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.0-flash-001  # optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions  # optional

Ghi chú:
- LLM giải bài ở dạng pseudo-code trước, không trực tiếp chọn entity để ghép công thức.
- Planner map số trong pseudo-code về entity input, rồi mới tạo plan symbolic.
- LLM không phải nơi thực thi số học chính.
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
from decimal import Decimal, InvalidOperation
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
CODE_PATH = OUTPUT_DIR / "Code.txt"
LOG_PATH = OUTPUT_DIR / "Log.yaml"

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 5


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


def write_text_file(path: Path, text: str) -> None:
    ensure_dirs()
    path.write_text(text.strip() + "\n", encoding="utf-8")


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

        normalized_entity = {
            "value": normalize_empty(entity.get("value")),
            "unit": normalize_empty(entity.get("unit")),
            "location": location,
            "grand_unit": normalize_empty(entity.get("grand_unit")),
        }
        source = normalize_empty(entity.get("source"))
        if source is not None:
            normalized_entity["source"] = str(source).strip()

        normalized[name] = normalized_entity

    if target_count != 1:
        raise PlannerError(f"ProblemEntities phải có đúng 1 target, hiện có {target_count}.")

    return normalized


def neutral_source_hint(name: str, entity: Dict[str, Any]) -> Optional[str]:
    value = normalize_empty(entity.get("value"))
    unit = normalize_empty(entity.get("unit"))

    if name == "identity_multiplier" and value == 1:
        return "implicit multiplicative identity 1"
    if name == "percentage_scale" and value == 100:
        return "scale factor for converting a fraction to a percentage: 100"
    if name.startswith("unit_conversion_") and value is not None:
        match = re.match(r"^unit_conversion_(.+)_per_(.+)$", name)
        if match:
            numerator_unit = match.group(1).replace("_", " ")
            denominator_unit = match.group(2).replace("_", " ")
            return (
                "standard conversion factor: "
                f"{value} {numerator_unit} per {denominator_unit}"
            )
        unit_text = f" {unit}" if unit else ""
        return f"standard conversion factor {value}{unit_text}"
    return None


def alias_problem_entities(
    problem_entities: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Tạo view trung lập cho Planner.

    LLM chỉ thấy e1/e2/.../target để giảm xu hướng suy luận từ tên entity thật
    như pens_more_than_notebooks. Sau khi LLM trả plan, code dịch alias về tên
    entity thật trước khi ghi Output/Plan.yaml.
    """
    target_name = target_entity_name(problem_entities)
    aliased: Dict[str, Dict[str, Any]] = {}
    alias_to_real: Dict[str, str] = {}
    real_to_alias: Dict[str, str] = {}
    next_index = 1

    for real_name, entity in problem_entities.items():
        if real_name == target_name:
            alias = "target"
        else:
            alias = f"e{next_index}"
            next_index += 1

        alias_entity = dict(entity)
        if normalize_empty(alias_entity.get("source")) is None:
            hint = neutral_source_hint(real_name, alias_entity)
            if hint:
                alias_entity["source"] = hint

        aliased[alias] = alias_entity
        alias_to_real[alias] = real_name
        real_to_alias[real_name] = alias

    return aliased, alias_to_real, real_to_alias


def normalize_empty(value: Any) -> Any:
    if value == "" or value == "null" or value == "None":
        return None
    return value


# Legacy direct-plan prompt path. run() now uses call_openrouter_code() and
# plan_from_code(), but these helpers are kept for comparison while the Planner
# design is still changing.
def build_system_prompt() -> str:
    return """
Bạn là một Planner sinh kế hoạch giải toán symbolic từ đề bài và danh sách entity đã formalize.

Bạn sẽ nhận các entity dưới dạng handle trung tính:
- Input entity được đặt tên e1, e2, e3, ...
- Target entity luôn được đặt tên target.
- Các handle e1/e2/... không có ý nghĩa ngữ nghĩa. Không suy luận quan hệ toán học từ tên handle.
- Muốn hiểu một handle biểu diễn số nào, hãy đọc value, unit, grand_unit, source và đối chiếu với Đề bài gốc.
- Khi viết expr/result, chỉ dùng các handle này và result trung gian bạn tự tạo. Không dùng tên entity thật nếu nó không xuất hiện trong input.

Nhiệm vụ:
- Giải bài toán từ Đề bài gốc.
- Dùng ProblemEntities như dictionary các hằng số đã được grounded trong đề: khi lời giải cần một số, dùng handle entity tương ứng thay vì viết số literal.
- ProblemEntities KHÔNG phải danh sách các biến bắt buộc phải dùng. Không dùng một handle chỉ vì nó tồn tại.
- Không tự thêm input entity mới không có trong ProblemEntities.
- Được tạo entity trung gian bằng trường result.
- Bước cuối cùng phải có result là target.
- Không thực hiện giải thích, chỉ trả YAML thuần.

Mỗi bước có đúng 4 trường:
- expr: biểu thức tính toán symbolic, chỉ dùng handle entity/result. Ví dụ: e1 + e2
- result: tên entity được tạo ra bởi bước đó.
- result_unit: đơn vị của result. Với scalar có thể để rỗng/null.
- result_grand_unit: grand unit của result theo target. Với scalar có thể để rỗng/null.

Quy tắc step:
- Tên bước phải là step1, step2, step3, ... liên tục, không bỏ số.
- step sau được dùng result của step trước.
- result phải là snake_case.
- expr chỉ được dùng tên entity/result đã có và toán tử đơn giản: +, -, *, /, parentheses.
- Tuyệt đối không viết số literal trong expr, kể cả 0, 1, 2, 36, 60, 0.5, 1/3, hay số mũ như ** 2.
- Nếu lời giải cần một hệ số/dữ kiện số trong đề, hệ số đó phải là entity input trong ProblemEntities.
- Nếu ProblemEntities có handle có source là "implicit multiplicative identity 1" thì dùng handle này cho hệ số 1 trong các biểu thức như total = base * (identity + multiplier), remainder = total * (identity - fraction).
- Nếu cần hệ số trung gian như 1, hãy tạo từ entity đã có. Ví dụ: identity_multiplier = growth_multiplier / growth_multiplier.
- Nếu cần đổi đơn vị trực tiếp từ một entity sang unit khác, tạo một step riêng chỉ có source entity trong expr.
  Ví dụ: expr: road_length, result: road_length_kilometers, result_unit: kilometers.
  Không cần viết conversion factor trong expr; Executor sẽ tự dùng unit_conversion_* phù hợp để tính reported_expr/value.
- Với rate theo thời gian hoặc các phép nhân tổng quát, vẫn dùng conversion factor nếu đó là quan hệ cần tính trong bài.
- Không tự đặt tên chung chung như conversion_factor nếu tên đó không tồn tại trong ProblemEntities.

Quy tắc không ảo giác:
- Chỉ dùng entity trong ProblemEntities hoặc result của step trước.
- Không cần dùng mọi input entity. Entity có thể chỉ là context, paraphrase, summary hoặc constraint của câu chữ.
- Hãy chọn các entity theo lời giải toán từ đề bài; bỏ qua entity nếu dùng nó sẽ đếm trùng hoặc không cần thiết.
- ProblemEntities có thể có field source. Đây là binding gốc của số trong đề bài.
  Handle chỉ là nhãn để viết expr, không phải bằng chứng về quan hệ toán học.
  Nếu một handle có value đúng nhưng source/problem text cho thấy nó chỉ là điều kiện, ngưỡng, context hoặc count đã được liệt kê bởi các thành phần khác, không dùng handle đó trong phép tính.
- Không tạo bước không cần thiết.
- Không tạo bước chỉ copy một entity sang entity khác. Nếu một step đã tính ra
  giá trị cuối cùng thì đặt result của chính step đó là target.
- Không đưa value vào Plan.yaml.
- Không tính toán ra số trong Plan.yaml và không đưa số literal vào expr.
- Với các source clause dạng "A unit represents B unit", "A unit corresponds to B unit",
  "A unit is equivalent to B unit", đây là quan hệ tỉ lệ giữa hai đầu của cùng clause.
  Nếu cần đổi một đại lượng cùng loại với A sang loại B, dùng known_A / scale_A * scale_B.
  Không được diễn giải B là B_per_A nếu A không phải 1 đơn vị.
- Không nhân đôi dữ kiện đếm nếu các thành phần đã được liệt kê đầy đủ.
  Ví dụ: "Nancy buys 2 coffees a day", rồi đề cho giá morning coffee và afternoon coffee.
  Khi đó daily cost là morning_coffee_price + afternoon_coffee_price, KHÔNG nhân thêm coffees_per_day.
  Chỉ dùng entity đếm như coffees_per_day nếu đề cho giá của một item đơn lẻ mà chưa liệt kê từng item.
- Với phần trăm tăng/giảm như "20% heavier", "20% more", "20% less":
  không viết base * (1 + percentage) hoặc base * (1 - percentage).
  Hãy tách thành bước riêng: increase_amount = base * percentage, rồi final = base + increase_amount.
  Với giảm: decrease_amount = base * percentage, rồi final = base - decrease_amount.
- Nếu target hỏi "what percentage" và ProblemEntities có percentage_scale:
  trước tiên tính fraction dạng comic_books / total_books, sau đó nhân percentage_scale để trả về số phần trăm.
  Ví dụ 24 / 120 = 0.2 thì target percentage phải là 0.2 * percentage_scale = 20, không trả 0.2.
- Với discount dạng "N% off if/for customers who buy at least K items":
  K chỉ là ngưỡng đủ điều kiện nhận discount, không phải số lượng thật được mua.
  Nếu đề cho số lượng thật M và M >= K, hãy tính savings bằng tổng giá không discount của M items nhân discount percentage.
  Không dùng minimum_*_for_discount để tính giá mua lẻ hoặc số lượng mua.
- Với rate theo thời gian như books_per_month/pages_per_day và đề hỏi cho một khoảng thời gian khác:
  phải đổi rate sang tổng của khoảng đó bằng conversion entity có sẵn.
  Ví dụ nếu có books_per_month và unit_conversion_months_per_year, tính total_books_needed = books_per_month * unit_conversion_months_per_year.
  Ví dụ nếu có food_cost_per_day và unit_conversion_days_per_year, tính yearly_food_cost = food_cost_per_day * unit_conversion_days_per_year.
  Ví dụ nếu có lessons_per_week và unit_conversion_weeks_per_year, tính yearly_lessons = lessons_per_week * unit_conversion_weeks_per_year.
  Không được lấy books_per_month trừ/cộng trực tiếp với tổng theo năm.
- Với quan hệ "X has/made half as many as Y", nghĩa là X = Y * fraction.
  Nếu cần tìm Y từ X thì phải tính Y = X / fraction, không được X * fraction.
  Với "X has twice/three times as many as Y", nghĩa là X = Y * multiplier.
  Nếu cần tìm Y từ X thì phải chia cho multiplier.
  Với "X occupy half as many as Y", base là Y; không lấy half của phần còn lại trừ khi đề nói rõ "half of the remaining".
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
- Nếu target hỏi tổng tiền cho cả hai/nhiều item và một item sau "was P% more expensive":
  price_after_increase = base_price + base_price * percentage.
  total_cost phải cộng cả base_price và price_after_increase, không được trả riêng price_after_increase.
- Với "split the remaining" giữa hai loại/nhóm và ProblemEntities có split_count:
  phần của một loại = remaining / split_count.
- Với "has N roommates" và bill được "divide equally", tổng số người chia thường là roommates + identity_multiplier
  vì người trong đề cũng ở cùng các roommates.
- Với bài "give to each ... same amount", target là số tiền mỗi người nhận thêm. Không được dùng target trong expr.
  Công thức cân bằng dạng một người cho N người nhận: amount_each = (giver_money - receiver_money) / (number_of_receivers + identity_multiplier).
- Với bài chia tiền có quan hệ tuyến tính như "second took $80 more than the first" và "third took twice what second took",
  không được tham chiếu first_share khi first_share là target. Hãy gom coefficient và offset:
  coefficient = identity_multiplier + identity_multiplier + multiplier, offset = more_than + more_than * multiplier,
  first_share = (total_amount - offset) / coefficient.
- Với sales có "large costs P times small", "sold Q times as many small as large", và total earnings:
  không được coi Q là số lượng small đã bán. Hãy tính large_price = small_price * price_multiplier,
  earnings_per_large_group = large_price + small_price * quantity_multiplier,
  large_count = total_earnings / earnings_per_large_group, rồi target small_count = large_count * quantity_multiplier.
- Với đổi mệnh giá bills/coins, count pieces phải được tính qua dollar amount:
  pieces_new_denominator = source_amount_dollars / value_of_each_new_bill.
  Không được lấy fraction của số pieces bill cũ rồi coi đó là số pieces bill mới.
- Với "family consists of her/him, her/his younger sibling, parents, grandparents" và ProblemEntities có các *_count,
  hãy dùng các count đó để tính số người. "parents" thường là 2 người; "grandfather/grandmother" là 1 người.
  Nếu discount áp dụng cho people 18 years old or younger, người trong đề và younger sibling thuộc nhóm discount
  khi tuổi người trong đề <= ngưỡng; parents/grandparents thường dùng regular ticket nếu đề không nói họ được discount.
  Không được tính discount trên total ticket cost của cả family. Hãy tách discounted_count và regular_count.
- Với phân bổ một tổng tài nguyên theo fraction cho nhóm A rồi hỏi nhóm B nhận bao nhiêu/per B:
  nếu đề không cho một fraction/amount riêng cho nhóm B, lượng của B là total - amount_A.
  Không được copy amount_A sang amount_B nếu điều đó làm tổng vượt quá resource ban đầu.
- Với "twins", số con/offspring tương ứng là 2, không dùng calves_from_single_pregnancy * calves_from_single_pregnancy để biểu diễn twins.
- Với bài về animals/llamas/calves sau khi birth, herd sau sinh gồm cả adult animals ban đầu và calves mới sinh.
  Nếu trade calves for adult animals, tổng herd thay đổi theo: original adults + total calves - calves_traded + new_adult_animals.
  Ví dụ original_adult_animals = pregnant_group_1 + pregnant_group_2; total_after_trade = original_adult_animals + total_calves - traded_calves + new_adult_animals.
  Không được chỉ lấy remaining calves + new adult animals rồi bỏ mất adult animals ban đầu.

Quy tắc đơn vị:
- result_unit là đơn vị trực tiếp của result.
- result_grand_unit là đơn vị đối chiếu theo target.
- Nếu result là target thì result_unit và result_grand_unit phải khớp với unit và grand_unit của target.

Định dạng output bắt buộc:
step1:
  expr: e1 + e2
  result: intermediate_entity
  result_unit: dollars
  result_grand_unit: dollars

step2:
  expr: intermediate_entity * e3
  result: target
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

ProblemEntityHandles.yaml:
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
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
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


def build_code_system_prompt() -> str:
    return """
Bạn là Solver viết lời giải toán dưới dạng pseudo-code Python tối giản.

Mục tiêu:
- Giải bài toán từ đề bài gốc như khi giải text thuần.
- Chỉ viết các phép gán cần thiết để ra đáp án.
- Không dùng bảng entity, không dùng tên entity formalized.
- Code chỉ là intermediate để hệ thống map số sang entity sau đó.

Quy tắc output:
- Chỉ trả code thuần, không Markdown, không ``` và không giải thích.
- Mỗi dòng là một phép gán dạng snake_case = expression.
- Dòng cuối phải là answer = expression.
- Expression chỉ dùng biến đã gán trước đó, số literal trong đề bài, và toán tử + - * / với ngoặc.
- Không dùng import, function, print, if/else, list, dict, loop, comparison.
- Code phải có 2 pha rõ ràng:
  Pha 1: bind từng dữ kiện số trong đề vào biến có tên mang nghĩa nguồn.
  Pha 2: tính toán chỉ bằng các biến đã bind, không viết số literal trực tiếp trong dòng tính toán.
- Nếu cùng một số xuất hiện với 2 ý nghĩa khác nhau, phải tạo 2 biến khác nhau.
  Ví dụ 7 tokens spent và seven times as many phải là spent_on_ski_ball = 7 và parent_token_multiplier = 7.
- Với fraction viết bằng chữ như a third, one third, a fourth, half:
  bind thành biến fraction trước, ví dụ pacman_fraction = 1 / 3, candy_crush_fraction = 1 / 4.
  Sau đó dùng phép nhân: wasted_on_pacman = starting_tokens * pacman_fraction.
  Không viết starting_tokens / 3 hoặc starting_tokens / 4 trong dòng tính toán.
- Với phần trăm N%, bind thành biến percentage trước, ví dụ discount_percentage = 20 / 100 hoặc discount_percentage = 0.2.
  Sau đó dùng biến này trong phép tính.
- Không ghi kết quả đã tính ra như một số literal nếu số đó không được cho trực tiếp trong đề.
  Ví dụ đúng: pens = notebooks + 50; answer = notebooks + pens.
  Ví dụ sai: answer = 110.
- Nếu số trong đề viết bằng chữ, hãy viết thành chữ số.
- Với phần trăm N%, hãy dùng decimal multiplier tương ứng nếu cần nhân, ví dụ 20% -> 0.2.
- Với "X more than Y", "X fewer than Y", "X less than Y", phải tạo quantity thật trước rồi mới tính tổng nếu đề hỏi tổng.
  Ví dụ: notebooks = 30; pens = notebooks + 50; answer = notebooks + pens.
- Với rate/count summary, không nhân đôi nếu đề đã liệt kê từng component.
  Ví dụ mua morning coffee và afternoon coffee mỗi ngày thì daily_cost = morning_price + afternoon_price.
""".strip()


def build_code_user_prompt(problem: str, previous_error: Optional[str] = None) -> str:
    retry_note = ""
    if previous_error:
        retry_note = f"""

Code trước bị reject vì lỗi:
{previous_error}

Hãy viết lại code. Giải từ đề bài trước, không shortcut thành final number.
""".rstrip()

    return f"""
Hãy viết pseudo-code Python tối giản để giải bài toán sau.

Đề bài:
{problem}
{retry_note}
""".strip()


def call_openrouter_code(problem: str, previous_error: Optional[str] = None) -> str:
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
            {"role": "system", "content": build_code_system_prompt()},
            {"role": "user", "content": build_code_user_prompt(problem, previous_error=previous_error)},
        ],
        "temperature": 0,
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
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


IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


def replace_expr_tokens(expr: str, mapping: Dict[str, str]) -> str:
    """Thay token entity/result trong expr mà không đụng vào substring."""
    return IDENTIFIER_RE.sub(lambda match: mapping.get(match.group(0), match.group(0)), expr)


def unique_result_name(name: str, reserved: set[str]) -> str:
    if name not in reserved:
        return name

    base = f"{name}_result"
    candidate = base
    index = 2
    while candidate in reserved:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def dealias_plan(
    alias_plan: Dict[str, Dict[str, Any]],
    alias_to_real: Dict[str, str],
    real_problem_entities: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Dịch plan sinh trên e1/e2/.../target về tên entity thật.

    Intermediate result của LLM được giữ nguyên trừ khi đụng tên input/target thật.
    """
    reserved = set(real_problem_entities.keys())
    result_mapping: Dict[str, str] = {}

    for step in alias_plan.values():
        alias_result = step["result"]
        if alias_result == "target":
            real_result = alias_to_real["target"]
        else:
            real_result = unique_result_name(alias_result, reserved)

        result_mapping[alias_result] = real_result
        reserved.add(real_result)

    token_mapping = {**alias_to_real, **result_mapping}
    real_plan: Dict[str, Dict[str, Any]] = {}

    for step_name, step in alias_plan.items():
        real_plan[step_name] = {
            "expr": replace_expr_tokens(step["expr"], token_mapping),
            "result": result_mapping[step["result"]],
            "result_unit": step["result_unit"],
            "result_grand_unit": step["result_grand_unit"],
        }

    return real_plan


NUMERIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)

ANSWER_NAMES = {"answer", "final_answer", "result", "target"}
CODE_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
ALLOWED_CODE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
ALLOWED_CODE_UNARYOPS = (ast.UAdd, ast.USub)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:python|py)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_code_assignments(text: str) -> List[Tuple[str, str, ast.AST]]:
    clean_text = strip_code_fence(text)
    assignments: List[Tuple[str, str, ast.AST]] = []

    for raw_line in clean_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("return "):
            line = f"answer = {line[len('return '):].strip()}"
        if line.startswith("print(") and line.endswith(")"):
            line = f"answer = {line[len('print('):-1].strip()}"

        match = CODE_ASSIGNMENT_RE.match(line)
        if not match:
            raise PlannerError(f"Code line không phải assignment hợp lệ: {raw_line!r}")

        lhs = normalize_code_name(match.group(1))
        expr = match.group(2).strip()
        if not lhs:
            raise PlannerError(f"Tên biến code không hợp lệ: {match.group(1)!r}")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise PlannerError(f"Expression code không parse được: {expr!r}") from exc

        validate_code_expr_ast(tree.body, expr)
        assignments.append((lhs, expr, tree.body))

    if not assignments:
        raise PlannerError("LLM không trả assignment code nào.")

    last_lhs = assignments[-1][0]
    if last_lhs not in ANSWER_NAMES:
        raise PlannerError("Dòng cuối của Code.txt phải gán vào answer.")

    return assignments


def normalize_code_name(name: str) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
    name = re.sub(r"_+", "_", name)
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return ""
    return name


def validate_code_expr_ast(node: ast.AST, expr: str) -> None:
    if isinstance(node, ast.Expression):
        validate_code_expr_ast(node.body, expr)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ALLOWED_CODE_BINOPS):
            raise PlannerError(f"Code expr chỉ được dùng + - * /: {expr!r}")
        validate_code_expr_ast(node.left, expr)
        validate_code_expr_ast(node.right, expr)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_CODE_UNARYOPS):
            raise PlannerError(f"Code expr chỉ được dùng unary +/-: {expr!r}")
        validate_code_expr_ast(node.operand, expr)
        return
    if isinstance(node, ast.Name):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", node.id):
            raise PlannerError(f"Biến code không hợp lệ trong expr: {node.id!r}")
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return
    raise PlannerError(f"Code expr chứa cú pháp không được hỗ trợ: {expr!r}")


def ast_is_numeric_only(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ALLOWED_CODE_UNARYOPS):
        return ast_is_numeric_only(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_CODE_BINOPS):
        return ast_is_numeric_only(node.left) and ast_is_numeric_only(node.right)
    return False


def ast_contains_numeric_literal(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant)
        and isinstance(child.value, (int, float))
        and not isinstance(child.value, bool)
        for child in ast.walk(node)
    )


def numeric_literals_in_ast(node: ast.AST) -> List[str]:
    return [
        repr(child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, (int, float))
        and not isinstance(child.value, bool)
    ]


def eval_numeric_ast(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        value = eval_numeric_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = eval_numeric_ast(node.left)
        right = eval_numeric_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise PlannerError("Code expr chia cho 0.")
            return left / right
    raise PlannerError("Không eval được numeric AST.")


def decimal_from_entity_value(value: Any) -> Optional[Decimal]:
    value = normalize_empty(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def decimals_equal(left: Decimal, right: Decimal) -> bool:
    return abs(left - right) <= Decimal("0.000000001")


def text_terms(text: Any) -> set[str]:
    return {
        singularize_token(term)
        for term in re.findall(r"[a-zA-Z]+", str(text or "").lower())
        if term
    }


def input_entity_candidates_for_number(
    value: Decimal,
    problem_entities: Dict[str, Dict[str, Any]],
) -> List[str]:
    candidates: List[str] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        entity_value = decimal_from_entity_value(entity.get("value"))
        if entity_value is not None and decimals_equal(value, entity_value):
            candidates.append(name)
    return candidates


def score_number_entity_candidate(
    name: str,
    entity: Dict[str, Any],
    context_terms: set[str],
) -> int:
    entity_terms = set(entity_name_terms(name))
    entity_terms.update(text_terms(entity.get("unit")))
    entity_terms.update(text_terms(entity.get("grand_unit")))
    entity_terms.update(text_terms(entity.get("source")))

    score = 4 * len(context_terms & entity_terms)
    scalar_context_terms = {
        "fraction",
        "percentage",
        "percent",
        "multiplier",
        "times",
        "ratio",
        "rate",
        "scale",
    }
    scalar_entity_terms = {
        "fraction",
        "percentage",
        "percent",
        "multiplier",
        "times",
        "ratio",
        "rate",
        "scale",
    }
    entity_unit = normalize_empty(entity.get("unit"))
    if entity_unit is None:
        if context_terms & scalar_context_terms:
            score += 6
        else:
            score -= 5
    elif context_terms & scalar_context_terms:
        score -= 4
    else:
        score += 5

    if context_terms & scalar_context_terms and entity_terms & scalar_entity_terms:
        score += 8
    if "multiplier" in context_terms and {"multiplier", "times"} & entity_terms:
        score += 8

    if name.startswith("unit_conversion_"):
        score -= 6
        if {"convert", "conversion", "per"} & context_terms:
            score += 6
    if name == "identity_multiplier":
        score -= 2
        if {"one", "self", "host", "identity"} & context_terms:
            score += 4
    if "percentage" in entity_terms or "percent" in entity_terms:
        if {"percent", "percentage", "discount"} & context_terms:
            score += 4
    return score


def map_number_to_entity(
    value: Decimal,
    problem_entities: Dict[str, Dict[str, Any]],
    context_terms: set[str],
) -> Optional[str]:
    candidates = input_entity_candidates_for_number(value, problem_entities)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: score_number_entity_candidate(name, problem_entities[name], context_terms),
    )


class NumericLiteralEntityRewriter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        problem_entities: Dict[str, Dict[str, Any]],
        context_terms: set[str],
    ) -> None:
        self.problem_entities = problem_entities
        self.context_terms = context_terms
        self.unmapped_literals: List[str] = []

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            return node

        mapped = map_number_to_entity(
            Decimal(str(node.value)),
            self.problem_entities,
            self.context_terms,
        )
        if not mapped:
            self.unmapped_literals.append(repr(node.value))
            return node

        return ast.copy_location(ast.Name(id=mapped, ctx=ast.Load()), node)


def rewrite_numeric_literals_to_entities(
    expr_ast: ast.AST,
    *,
    lhs: str,
    raw_expr: str,
    problem_entities: Dict[str, Dict[str, Any]],
) -> ast.AST:
    """
    LLM đôi khi viết literal trong dòng tính toán dù số đó đã có entity input.
    Ở tầng Planner, ta chỉ cho phép repair nếu literal map được về entity thật
    của đề bài/conversion; derived number vẫn bị reject.
    """
    rewriter = NumericLiteralEntityRewriter(
        problem_entities=problem_entities,
        context_terms=code_context_terms(lhs, raw_expr),
    )
    rewritten = rewriter.visit(ast.fix_missing_locations(expr_ast))
    ast.fix_missing_locations(rewritten)
    if rewriter.unmapped_literals:
        raise PlannerError(
            f"Assignment {lhs} = {raw_expr} đang dùng số literal trực tiếp trong dòng tính toán "
            f"và không map được literal {rewriter.unmapped_literals} sang input entity. "
            "Hãy bind dữ kiện số/fraction/percentage thành biến riêng trước, rồi chỉ dùng biến trong phép tính. "
            "Ví dụ a third phải là pacman_fraction = 1 / 3; wasted = tokens * pacman_fraction, "
            "không viết wasted = tokens / 3."
        )
    return rewritten


def op_symbol(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return "+"
    if isinstance(op, ast.Sub):
        return "-"
    if isinstance(op, ast.Mult):
        return "*"
    if isinstance(op, ast.Div):
        return "/"
    raise PlannerError(f"Toán tử không hỗ trợ: {type(op).__name__}")


def op_precedence(node: ast.AST) -> int:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            return 2
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return 1
    if isinstance(node, ast.UnaryOp):
        return 3
    return 4


def parenthesize_child(parent: ast.BinOp, child: ast.AST, rendered: str, *, right: bool) -> str:
    if not isinstance(child, ast.BinOp):
        return rendered
    if op_precedence(child) < op_precedence(parent):
        return f"({rendered})"
    if right and isinstance(parent.op, (ast.Sub, ast.Div)) and op_precedence(child) == op_precedence(parent):
        return f"({rendered})"
    return rendered


def code_context_terms(lhs: str, expr: str) -> set[str]:
    terms = text_terms(lhs)
    terms.update(text_terms(expr))
    return terms


def ast_to_symbolic_expr(
    node: ast.AST,
    *,
    lhs: str,
    raw_expr: str,
    var_to_entity: Dict[str, str],
    problem_entities: Dict[str, Dict[str, Any]],
) -> str:
    context_terms = code_context_terms(lhs, raw_expr)

    if ast_is_numeric_only(node):
        value = eval_numeric_ast(node)
        mapped = map_number_to_entity(value, problem_entities, context_terms)
        if mapped:
            return mapped
        if isinstance(node, ast.Constant):
            raise PlannerError(
                f"Không map được số literal {node.value!r} trong code sang input entity."
            )

    if isinstance(node, ast.Constant):
        value = Decimal(str(node.value))
        mapped = map_number_to_entity(value, problem_entities, context_terms)
        if not mapped:
            raise PlannerError(f"Không map được số literal {node.value!r} trong code sang input entity.")
        return mapped

    if isinstance(node, ast.Name):
        code_name = normalize_code_name(node.id)
        if code_name in var_to_entity:
            return var_to_entity[code_name]
        if code_name in problem_entities and problem_entities[code_name].get("location") == "input":
            return code_name
        raise PlannerError(f"Code dùng biến chưa được gán hoặc chưa map được: {node.id!r}")

    if isinstance(node, ast.UnaryOp):
        operand = ast_to_symbolic_expr(
            node.operand,
            lhs=lhs,
            raw_expr=raw_expr,
            var_to_entity=var_to_entity,
            problem_entities=problem_entities,
        )
        return f"-{operand}" if isinstance(node.op, ast.USub) else operand

    if isinstance(node, ast.BinOp):
        left = ast_to_symbolic_expr(
            node.left,
            lhs=lhs,
            raw_expr=raw_expr,
            var_to_entity=var_to_entity,
            problem_entities=problem_entities,
        )
        right = ast_to_symbolic_expr(
            node.right,
            lhs=lhs,
            raw_expr=raw_expr,
            var_to_entity=var_to_entity,
            problem_entities=problem_entities,
        )
        left = parenthesize_child(node, node.left, left, right=False)
        right = parenthesize_child(node, node.right, right, right=True)
        return f"{left} {op_symbol(node.op)} {right}"

    raise PlannerError(f"Không chuyển được code expr sang symbolic expr: {raw_expr!r}")


def single_name_in_ast(node: ast.AST) -> Optional[str]:
    return normalize_code_name(node.id) if isinstance(node, ast.Name) else None


def unit_matches_result_name(unit: Any, result_name: str) -> bool:
    unit_terms = text_terms(unit)
    if not unit_terms:
        return False
    return bool(unit_terms & entity_name_terms(result_name))


def unit_info_for_token(
    token: str,
    problem_entities: Dict[str, Dict[str, Any]],
    plan: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if token in problem_entities:
        entity = problem_entities[token]
        return normalize_empty(entity.get("unit")), normalize_empty(entity.get("grand_unit"))
    for step in plan.values():
        if step.get("result") == token:
            return normalize_empty(step.get("result_unit")), normalize_empty(step.get("result_grand_unit"))
    return None, None


def infer_plan_units(
    symbolic_expr: str,
    result_name: str,
    problem_entities: Dict[str, Dict[str, Any]],
    plan: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    if result_name == target_name:
        return normalize_empty(target.get("unit")), normalize_empty(target.get("grand_unit"))

    token_infos = [
        unit_info_for_token(token, problem_entities, plan)
        for token in extract_expr_tokens(symbolic_expr)
    ]
    token_infos = [(unit, grand) for unit, grand in token_infos if unit is not None or grand is not None]
    if not token_infos:
        return None, None

    for unit, grand in token_infos:
        if unit_matches_result_name(unit, result_name):
            return unit, grand or unit

    non_scalar_units = [unit for unit, _ in token_infos if unit is not None]
    non_scalar_grands = [grand for _, grand in token_infos if grand is not None]

    unique_units = list(dict.fromkeys(non_scalar_units))
    unique_grands = list(dict.fromkeys(non_scalar_grands))

    if len(unique_units) == 1:
        return unique_units[0], unique_grands[0] if unique_grands else unique_units[0]
    if len(unique_grands) == 1:
        return unique_grands[0], unique_grands[0]

    return unique_units[0] if unique_units else None, unique_grands[0] if unique_grands else None


def unit_base(unit: Any) -> Optional[str]:
    unit_text = normalize_empty(unit)
    if unit_text is None:
        return None
    terms = text_terms(unit_text)
    for term in terms:
        if term in UNIT_BASE_ALIASES:
            return UNIT_BASE_ALIASES[term]
    return singularize_token(str(unit_text).strip().lower())


def parse_conversion_entity_name(name: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^unit_conversion_(.+)_per_(.+)$", name)
    if not match:
        return None
    numerator = match.group(1).replace("_", " ")
    denominator = match.group(2).replace("_", " ")
    return numerator, denominator


def result_name_mentions_unit(result_name: str, unit: str) -> bool:
    unit_terms = text_terms(unit)
    if not unit_terms:
        return False
    return bool(entity_name_terms(result_name) & unit_terms)


def canonicalize_direct_unit_conversion(
    symbolic_expr: str,
    result_name: str,
    problem_entities: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Chuyển dạng source * unit_conversion_X_per_Y thành step đổi đơn vị chuẩn:
    expr chỉ còn source, còn result_unit là X/Y tương ứng.

    Chỉ áp dụng khi unit của source khớp thật với conversion entity. Các rate
    kiểu books_per_month * months_per_year vẫn giữ nguyên phép nhân.
    """
    patterns = [
        re.match(
            r"^(?P<source>[a-zA-Z_][a-zA-Z0-9_]*)\s*\*\s*(?P<conversion>unit_conversion_[a-zA-Z0-9_]+)$",
            symbolic_expr,
        ),
        re.match(
            r"^(?P<conversion>unit_conversion_[a-zA-Z0-9_]+)\s*\*\s*(?P<source>[a-zA-Z_][a-zA-Z0-9_]*)$",
            symbolic_expr,
        ),
        re.match(
            r"^(?P<source>[a-zA-Z_][a-zA-Z0-9_]*)\s*/\s*(?P<conversion>unit_conversion_[a-zA-Z0-9_]+)$",
            symbolic_expr,
        ),
    ]

    for index, match in enumerate(patterns):
        if not match:
            continue

        source_name = match.group("source")
        conversion_name = match.group("conversion")
        source = problem_entities.get(source_name)
        conversion = problem_entities.get(conversion_name)
        conversion_units = parse_conversion_entity_name(conversion_name)
        if not source or not conversion or not conversion_units:
            return symbolic_expr, None, None

        numerator_unit, denominator_unit = conversion_units
        source_base = unit_base(source.get("unit"))
        numerator_base = unit_base(numerator_unit)
        denominator_base = unit_base(denominator_unit)
        if source_base is None:
            return symbolic_expr, None, None

        is_division = index == 2
        if is_division:
            if source_base != numerator_base:
                return symbolic_expr, None, None
            result_unit = denominator_unit
        else:
            if source_base != denominator_base:
                return symbolic_expr, None, None
            result_unit = numerator_unit

        if not result_name_mentions_unit(result_name, result_unit):
            target_name = target_entity_name(problem_entities)
            target = problem_entities[target_name]
            target_unit = unit_base(target.get("unit"))
            if target_unit != unit_base(result_unit):
                return symbolic_expr, None, None

        return source_name, result_unit, result_unit

    return symbolic_expr, None, None


def rename_plan_result(
    plan: Dict[str, Dict[str, Any]],
    old_result: str,
    new_result: str,
    problem_entities: Dict[str, Dict[str, Any]],
) -> bool:
    target = problem_entities[new_result]
    for step in plan.values():
        if step.get("result") != old_result:
            continue
        step["result"] = new_result
        step["result_unit"] = normalize_empty(target.get("unit"))
        step["result_grand_unit"] = normalize_empty(target.get("grand_unit"))
        return True
    return False


def plan_from_code(
    code_text: str,
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    assignments = parse_code_assignments(code_text)
    target_name = target_entity_name(problem_entities)
    var_to_entity: Dict[str, str] = {}
    plan: Dict[str, Dict[str, Any]] = {}

    for index, (lhs, raw_expr, expr_ast) in enumerate(assignments):
        is_last = index == len(assignments) - 1
        wants_target = lhs in ANSWER_NAMES or is_last
        context_terms = code_context_terms(lhs, raw_expr)

        if ast_is_numeric_only(expr_ast):
            mapped = map_number_to_entity(eval_numeric_ast(expr_ast), problem_entities, context_terms)
            if mapped and not wants_target:
                var_to_entity[lhs] = mapped
                continue
            raise PlannerError(
                f"Assignment {lhs} = {raw_expr} chỉ là số literal/derived number. "
                "Code phải giữ phép tính symbolic thay vì ghi đáp án hoặc subtotal đã tính sẵn."
            )

        if ast_contains_numeric_literal(expr_ast):
            expr_ast = rewrite_numeric_literals_to_entities(
                expr_ast,
                lhs=lhs,
                raw_expr=raw_expr,
                problem_entities=problem_entities,
            )
            if ast_contains_numeric_literal(expr_ast):
                raise PlannerError(
                    f"Assignment {lhs} = {raw_expr} vẫn còn số literal "
                    f"{numeric_literals_in_ast(expr_ast)} sau khi repair."
                )

        copied_name = single_name_in_ast(expr_ast)
        if copied_name is not None:
            if copied_name not in var_to_entity:
                raise PlannerError(f"Code copy từ biến chưa map được: {copied_name!r}")
            source_entity = var_to_entity[copied_name]
            if wants_target:
                if source_entity == target_name:
                    var_to_entity[lhs] = target_name
                    continue
                if rename_plan_result(plan, source_entity, target_name, problem_entities):
                    for var_name, entity_name in list(var_to_entity.items()):
                        if entity_name == source_entity:
                            var_to_entity[var_name] = target_name
                    var_to_entity[lhs] = target_name
                    continue
                target = problem_entities[target_name]
                plan[f"step{len(plan) + 1}"] = {
                    "expr": source_entity,
                    "result": target_name,
                    "result_unit": normalize_empty(target.get("unit")),
                    "result_grand_unit": normalize_empty(target.get("grand_unit")),
                }
                var_to_entity[lhs] = target_name
                continue
            var_to_entity[lhs] = source_entity
            continue

        symbolic_expr = ast_to_symbolic_expr(
            expr_ast,
            lhs=lhs,
            raw_expr=raw_expr,
            var_to_entity=var_to_entity,
            problem_entities=problem_entities,
        )

        result_name = target_name if wants_target else unique_result_name(lhs, set(problem_entities) | set(var_to_entity.values()))
        (
            symbolic_expr,
            result_unit_override,
            result_grand_unit_override,
        ) = canonicalize_direct_unit_conversion(symbolic_expr, result_name, problem_entities)
        if result_unit_override is not None or result_grand_unit_override is not None:
            result_unit = result_unit_override
            result_grand_unit = result_grand_unit_override
        else:
            result_unit, result_grand_unit = infer_plan_units(symbolic_expr, result_name, problem_entities, plan)
        step_name = f"step{len(plan) + 1}"
        plan[step_name] = {
            "expr": symbolic_expr,
            "result": result_name,
            "result_unit": result_unit,
            "result_grand_unit": result_grand_unit,
        }
        var_to_entity[lhs] = result_name

    if not plan:
        raise PlannerError("Code không tạo ra bước tính toán nào cho Plan.yaml.")

    normalized_plan = validate_and_normalize_plan(plan, problem_entities, problem=problem)
    validate_relative_difference_entities_resolved(normalized_plan, problem_entities)
    return normalized_plan

RELATIVE_DIFFERENCE_MARKERS = {
    "_more_than_": "+",
    "_greater_than_": "+",
    "_less_than_": "-",
    "_fewer_than_": "-",
}

RELATIVE_DELTA_PREFIXES = {
    "fewer_": "-",
    "less_": "-",
}

RATE_PERIOD_CONVERSIONS = {
    "day": "unit_conversion_days_per_year",
    "week": "unit_conversion_weeks_per_year",
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

UNIT_BASE_ALIASES = {
    "inch": "inch",
    "inches": "inch",
    "foot": "foot",
    "feet": "foot",
    "yard": "yard",
    "yards": "yard",
    "mile": "mile",
    "miles": "mile",
    "millimeter": "millimeter",
    "millimeters": "millimeter",
    "centimeter": "centimeter",
    "centimeters": "centimeter",
    "meter": "meter",
    "meters": "meter",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "second": "second",
    "seconds": "second",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "cent": "cent",
    "cents": "cent",
    "dollar": "dollar",
    "dollars": "dollar",
    "ounce": "ounce",
    "ounces": "ounce",
    "pound": "pound",
    "pounds": "pound",
    "ton": "ton",
    "tons": "ton",
    "item": "item",
    "items": "item",
    "pair": "pair",
    "pairs": "pair",
    "dozen": "dozen",
    "dozens": "dozen",
}

STANDARD_CONVERSION_FACTORS = [
    ("inch", "foot", "unit_conversion_inches_per_foot", 12),
    ("foot", "yard", "unit_conversion_feet_per_yard", 3),
    ("foot", "mile", "unit_conversion_feet_per_mile", 5280),
    ("yard", "mile", "unit_conversion_yards_per_mile", 1760),
    ("millimeter", "centimeter", "unit_conversion_millimeters_per_centimeter", 10),
    ("centimeter", "meter", "unit_conversion_centimeters_per_meter", 100),
    ("meter", "kilometer", "unit_conversion_meters_per_kilometer", 1000),
    ("second", "minute", "unit_conversion_seconds_per_minute", 60),
    ("minute", "hour", "unit_conversion_minutes_per_hour", 60),
    ("hour", "day", "unit_conversion_hours_per_day", 24),
    ("day", "week", "unit_conversion_days_per_week", 7),
    ("day", "year", "unit_conversion_days_per_year", 365),
    ("week", "year", "unit_conversion_weeks_per_year", 52),
    ("month", "year", "unit_conversion_months_per_year", 12),
    ("cent", "dollar", "unit_conversion_cents_per_dollar", 100),
    ("ounce", "pound", "unit_conversion_ounces_per_pound", 16),
    ("pound", "ton", "unit_conversion_pounds_per_ton", 2000),
    ("item", "pair", "unit_conversion_items_per_pair", 2),
    ("item", "dozen", "unit_conversion_items_per_dozen", 12),
]


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

            if entity_name_terms(result) & {"offset", "difference", "delta", "adjustment"}:
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
        if name in {"identity_multiplier", "percentage_scale"}:
            continue
        if name.startswith("unit_conversion_"):
            continue
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


def target_is_percentage(target_name: str, target: Dict[str, Any]) -> bool:
    terms = entity_name_terms(target_name)
    unit = str(normalize_empty(target.get("unit")) or "").lower()
    grand_unit = str(normalize_empty(target.get("grand_unit")) or "").lower()

    return (
        "percentage" in terms
        or "percent" in terms
        or unit in {"percent", "percentage"}
        or grand_unit in {"percent", "percentage"}
    )


def validate_percentage_target_uses_scale(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    if "percentage_scale" not in problem_entities:
        return

    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    if not target_is_percentage(target_name, target):
        return

    lineage = step_lineage_to_result(plan, target_name)
    used_tokens: set[str] = set()
    for step in lineage.values():
        used_tokens.update(extract_expr_tokens(step["expr"]))

    if "percentage_scale" not in used_tokens:
        raise PlannerError(
            "Target hỏi giá trị percentage nên plan phải nhân fraction với percentage_scale. "
            "Không trả trực tiếp fraction dạng 0.2 nếu đáp án cần là 20 percent."
        )


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


def validate_matching_rate_inputs_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    target_unit = str(normalize_empty(target.get("unit")) or "").lower()
    if not target_unit:
        return

    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    missing: list[str] = []
    for name, entity in problem_entities.items():
        if entity.get("location") != "input":
            continue
        if "_per_" not in name:
            continue
        if str(normalize_empty(entity.get("unit")) or "").lower() != target_unit:
            continue
        if name not in plan_tokens:
            missing.append(name)

    if missing:
        raise PlannerError(
            f"Plan chưa dùng rate input cùng đơn vị với target {target_name!r}: {missing}. "
            "Ví dụ nếu target là pages và có pages_per_hour, phải dùng pages_per_hour để đổi thời gian đọc thành số pages."
        )


def validate_multiplier_not_used_as_additive_total(
    plan: Dict[str, Dict[str, Any]],
) -> None:
    for step_name, step in plan.items():
        expr = step["expr"]
        result = step["result"]
        result_terms = entity_name_terms(result)
        if result_terms & {"combined", "all", "group"}:
            continue
        if "+" not in expr:
            continue

        tokens = extract_expr_tokens(expr)
        multiplier_tokens = [token for token in tokens if "multiplier" in entity_name_terms(token)]
        if not multiplier_tokens:
            continue

        for token in set(tokens):
            if token in multiplier_tokens:
                continue
            if tokens.count(token) < 2:
                continue
            if any(re.search(rf"\b{re.escape(token)}\b\s*\*\s*\b{re.escape(multiplier)}\b", expr) or
                   re.search(rf"\b{re.escape(multiplier)}\b\s*\*\s*\b{re.escape(token)}\b", expr)
                   for multiplier in multiplier_tokens):
                raise PlannerError(
                    f"{step_name}.expr có vẻ cộng base với base * multiplier để tạo {result!r}. "
                    "Với quan hệ 'X has twice/three times as many as Y', số lượng X là Y * multiplier, "
                    "không phải Y + Y * multiplier. Chỉ cộng thêm base khi result là tổng/all/combined group."
                )


def validate_no_duplicate_per_input_addition(
    plan: Dict[str, Dict[str, Any]],
) -> None:
    for step_name, step in plan.items():
        expr = step["expr"]
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            if not isinstance(node.left, ast.Name) or not isinstance(node.right, ast.Name):
                continue
            if node.left.id != node.right.id:
                continue
            if "_per_" not in node.left.id:
                continue

            raise PlannerError(
                f"{step_name}.expr đang cộng trùng rate/input {node.left.id!r}. "
                "Nếu đề nói '3 extra days for each grade', entity *_per_* đã là số ngày mỗi grade; "
                "không được tự nhân đôi bằng cách cộng nó với chính nó."
            )


def validate_discount_group_size_not_applied_to_money(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    group_tokens = {
        name
        for name, entity in problem_entities.items()
        if "group" in entity_name_terms(name) and is_count_unit(entity.get("unit"))
    }
    if not group_tokens:
        return

    for step_name, step in plan.items():
        expr = step["expr"]
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            denominator_tokens = set(ast_expr_tokens(node.right))
            used_group_tokens = denominator_tokens & group_tokens
            if not used_group_tokens:
                continue

            numerator_tokens = set(ast_expr_tokens(node.left))
            money_numerators = [
                token
                for token in numerator_tokens
                if token in problem_entities and is_money_unit(problem_entities[token].get("unit"))
            ]
            # Result tokens from earlier money steps are not in problem_entities, so use name hint as fallback.
            money_numerators.extend(
                token
                for token in numerator_tokens
                if any(money_word in token for money_word in ("cost", "price", "money", "payment", "earnings"))
            )
            if money_numerators:
                raise PlannerError(
                    f"{step_name}.expr chia tiền {money_numerators} cho group size {sorted(used_group_tokens)}. "
                    "Với discount theo mỗi group of N seats/items, group size phải áp dụng lên số lượng item, "
                    "không áp dụng trực tiếp lên tổng tiền."
                )


def validate_roommate_equal_split_includes_host(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return
    text = problem.lower()
    if "roommate" not in text or "host_count" not in problem_entities:
        return
    if not re.search(r"\bdivide\b|\bequally\b|\bshare\b", text):
        return

    target_name = target_entity_name(problem_entities)
    lineage = step_lineage_to_result(plan, target_name)
    used_tokens: set[str] = set()
    for step in lineage.values():
        used_tokens.update(extract_expr_tokens(step["expr"]))

    if "host_count" not in used_tokens:
        raise PlannerError(
            "Bài chia bill giữa người trong đề và roommates phải dùng host_count. "
            "Nếu có N roommates và chia đều, mẫu số thường là roommates + host_count."
        )


def validate_sales_quantity_multiplier_not_used_as_fixed_count(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return

    text = problem.lower()
    if not re.search(r"\bsold\b", text):
        return
    if not re.search(r"\bearned\b|\bsales\b|\brevenue\b", text):
        return
    if not re.search(r"\btimes\s+the\s+price\b|\btimes\s+as\s+many\b|\bas\s+many\b", text):
        return

    quantity_multiplier_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and normalize_empty(entity.get("unit")) is None
        and "multiplier" in entity_name_terms(name)
        and ({"sold", "quantity", "count", "many"} & entity_name_terms(name))
    }
    if not quantity_multiplier_tokens:
        return

    money_input_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input" and is_money_unit(entity.get("unit"))
    }
    for step_name, step in plan.items():
        if not is_money_unit(step.get("result_unit")):
            continue
        result_terms = entity_name_terms(step["result"])
        if not (result_terms & {"earning", "earnings", "revenue", "sales", "cost", "total"}):
            continue
        expr_tokens = set(extract_expr_tokens(step["expr"]))
        if not (expr_tokens & quantity_multiplier_tokens):
            continue
        if not (expr_tokens & money_input_tokens):
            continue

        raise PlannerError(
            f"{step_name}.expr đang dùng quantity multiplier như số lượng bán cố định. "
            "Với bài sales có total earnings, price multiplier và quantity multiplier, "
            "hãy lập earnings_per_group rồi chia total_earnings để tìm count; "
            "không được coi multiplier như count thật."
        )


def validate_bill_change_uses_denominations(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return

    text = problem.lower()
    if not re.search(r"\b(?:bill|bills|coin|coins)\b", text):
        return
    if not re.search(r"\b(?:change|changed|requested|convert|converted)\b", text):
        return

    denomination_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and is_money_unit(entity.get("unit"))
        and (
            {"value", "denomination"} & entity_name_terms(name)
            or re.search(r"\b(?:bill|coin)s?\b", name)
        )
    }
    if not denomination_tokens:
        return

    plan_tokens: set[str] = set()
    for step in plan.values():
        plan_tokens.update(extract_expr_tokens(step["expr"]))

    missing_denominations = sorted(denomination_tokens - plan_tokens)
    if missing_denominations:
        raise PlannerError(
            f"Bài đổi mệnh giá bill/coin chưa dùng denomination values: {missing_denominations}. "
            "Muốn tính pieces mới, phải đổi pieces cũ thành dollar amount bằng value_of_each_source_bill "
            "rồi chia cho value_of_each_target_bill."
        )

    fraction_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and normalize_empty(entity.get("unit")) is None
        and ({"fraction", "percent", "percentage"} & entity_name_terms(name))
    }

    for step_name, step in plan.items():
        if str(normalize_empty(step.get("result_unit")) or "").lower() not in {"piece", "pieces"}:
            continue
        expr_tokens = set(extract_expr_tokens(step["expr"]))
        if not (expr_tokens & fraction_tokens):
            continue
        if expr_tokens & denomination_tokens:
            continue
        raise PlannerError(
            f"{step_name}.expr đang lấy fraction của số pieces bill cũ để tạo pieces bill mới. "
            "Fraction trong đổi mệnh giá áp lên dollar amount; hãy nhân với source bill value "
            "rồi chia cho target bill value."
        )


def validate_fraction_allocation_not_copied_when_exceeds_total(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return
    if not re.search(r"\b(?:fed|gave|given|distributed|shared|split)\b", problem.lower()):
        return

    fraction_values = {
        name: parse_numeric(entity.get("value"))
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and normalize_empty(entity.get("unit")) is None
        and ({"fraction", "percent", "percentage"} & entity_name_terms(name))
        and is_number(entity.get("value"))
    }
    if not fraction_values:
        return

    for step_name, step in plan.items():
        copied_tokens = extract_expr_tokens(step["expr"])
        if len(copied_tokens) != 1:
            continue

        source_name = copied_tokens[0]
        source_expr = result_expr(source_name, plan)
        if not source_expr:
            continue

        source_tokens = set(extract_expr_tokens(source_expr))
        copied_fraction_tokens = [
            token
            for token in source_tokens
            if token in fraction_values and float(fraction_values[token]) > 0.5
        ]
        if not copied_fraction_tokens:
            continue

        source_terms = entity_name_terms(source_name)
        result_terms = entity_name_terms(step["result"])
        if source_terms == result_terms:
            continue
        if not ({"fed", "given", "distributed", "shared", "split"} & (source_terms | result_terms)):
            continue

        raise PlannerError(
            f"{step_name}.expr copy allocation {source_name!r} sang {step['result']!r} "
            f"dù allocation trước dùng fraction > 1/2 ({copied_fraction_tokens}). "
            "Với một tổng tài nguyên hữu hạn, nhóm còn lại nên là total - allocated_amount "
            "nếu đề không cho amount/fraction riêng cho nhóm đó."
        )


def validate_family_discount_context_counts_used(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return

    text = problem.lower()
    if "family" not in text or "ticket" not in text or "discount" not in text:
        return

    context_counts = {
        name
        for name in problem_entities
        if name in {"self_count", "sibling_count", "parents_count", "grandparents_count"}
    }
    if len(context_counts) < 2:
        return

    target_name = target_entity_name(problem_entities)
    lineage = step_lineage_to_result(plan, target_name)
    used_tokens: set[str] = set()
    for step in lineage.values():
        used_tokens.update(extract_expr_tokens(step["expr"]))

    unused_counts = sorted(context_counts - used_tokens)
    if unused_counts:
        raise PlannerError(
            f"Bài ticket/discount theo family đang bỏ sót count ngữ cảnh: {unused_counts}. "
            "Với cụm 'family consists of ...', phải dùng self/sibling/parents/grandparents count "
            "để tính số vé regular và discount."
        )


def validate_family_age_discount_not_applied_to_all(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not problem:
        return

    text = problem.lower()
    if "family" not in text or "ticket" not in text or "discount" not in text:
        return
    if not re.search(r"\b(?:or younger|or under|and younger|younger than|under)\b", text):
        return

    adult_context_counts = {"parents_count", "grandparents_count"}
    child_context_counts = {"self_count", "sibling_count"}
    if not (adult_context_counts & set(problem_entities)):
        return
    if not (child_context_counts & set(problem_entities)):
        return

    for step_name, step in plan.items():
        expr_tokens = set(extract_expr_tokens(step["expr"]))
        if not any("discount" in entity_name_terms(token) for token in expr_tokens | {step["result"]}):
            continue

        for token in expr_tokens:
            if token in problem_entities:
                continue
            source_lineage = step_lineage_to_result(plan, token)
            source_tokens: set[str] = set()
            for source_step in source_lineage.values():
                source_tokens.update(extract_expr_tokens(source_step["expr"]))

            if not (adult_context_counts & source_tokens):
                continue
            if not (child_context_counts & source_tokens):
                continue
            if not is_money_unit(step.get("result_unit")):
                continue

            raise PlannerError(
                f"{step_name}.expr đang áp discount lên tổng ticket cost có cả parents/grandparents. "
                "Với discount theo age threshold, hãy tách discounted_count và regular_count; "
                "chỉ nhóm đủ tuổi mới được nhân discount_percentage."
            )


def looks_like_animal_birth_trade_problem(problem: Optional[str]) -> bool:
    if not problem:
        return False

    text = problem.lower()
    return bool(
        re.search(r"\b(?:pregnant|birth|calf|calves|offspring|babies)\b", text)
        and re.search(r"\b(?:trade|traded|sell|sells|sold|herd)\b", text)
    )


def validate_herd_after_birth_includes_original_adults(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
    problem: Optional[str],
) -> None:
    if not looks_like_animal_birth_trade_problem(problem):
        return

    target_name = target_entity_name(problem_entities)
    target = problem_entities[target_name]
    target_unit = str(normalize_empty(target.get("unit")) or "").lower()
    if not target_unit:
        return

    adult_input_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and str(normalize_empty(entity.get("unit")) or "").lower() == target_unit
        and not ({"new", "bought", "buy", "sold", "sell", "traded", "trade"} & entity_name_terms(name))
    }
    if not adult_input_tokens:
        return

    adult_count_tokens = set(adult_input_tokens)
    child_tokens: set[str] = set()
    new_adult_tokens = {
        name
        for name, entity in problem_entities.items()
        if entity.get("location") == "input"
        and str(normalize_empty(entity.get("unit")) or "").lower() == target_unit
        and ({"new", "bought", "buy"} & entity_name_terms(name))
    }

    for step_name, step in plan.items():
        result = step["result"]
        result_unit = str(normalize_empty(step.get("result_unit")) or "").lower()
        expr_tokens = set(extract_expr_tokens(step["expr"]))
        result_terms = entity_name_terms(result)

        if {"calf", "calves", "offspring", "baby", "babies"} & result_terms:
            child_tokens.add(result)

        if result_unit == target_unit and expr_tokens & adult_count_tokens:
            adult_count_tokens.add(result)

        if result_unit != target_unit:
            continue
        if not ({"total", "herd", "after", "remaining", "now"} & result_terms):
            continue
        if not ((expr_tokens & child_tokens) and (expr_tokens & new_adult_tokens)):
            continue
        if expr_tokens & adult_count_tokens:
            continue

        adult_hint = " + ".join(sorted(adult_input_tokens))
        raise PlannerError(
            f"{step_name}.expr đang tính herd sau birth/trade từ calves và new adult animals "
            "nhưng bỏ mất adult animals ban đầu. Tổng herd sau sinh phải gồm original adults + calves, "
            "rồi mới trừ calves traded/sold và cộng new adult animals. "
            f"Hãy tạo original_adult_animals = {adult_hint}, rồi dùng biến đó trong tổng herd."
        )


def validate_percentage_name_uses_percentage_scale(
    plan: Dict[str, Dict[str, Any]],
    problem_entities: Dict[str, Dict[str, Any]],
) -> None:
    if "percentage_scale" not in problem_entities:
        return
    validate_percentage_target_uses_scale(plan, problem_entities)


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
            if target_name in unknown_tokens:
                raise PlannerError(
                    f"{step_name}.expr đang tham chiếu target {target_name!r} trước khi tạo ra nó. "
                    "Nếu target là ẩn trong phương trình, phải biến đổi algebra để bước cuối mới tạo target. "
                    "Ví dụ equalization/give-to-each: target = (giver_money - receiver_money) / (receiver_count + identity_multiplier). "
                    "Ví dụ linear shares: gom coefficient và offset rồi target = (total - offset) / coefficient."
                )
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


def unit_base_keys(unit: Any) -> set[str]:
    unit = normalize_empty(unit)
    if unit is None:
        return set()
    text = str(unit).strip().lower().replace("-", "_").replace(" ", "_")
    return {UNIT_BASE_ALIASES[text]} if text in UNIT_BASE_ALIASES else set()


def conversion_factor_for_units(
    source_unit: Any,
    result_unit: Any,
    entities: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[float | int, str]]:
    source_bases = unit_base_keys(source_unit)
    result_bases = unit_base_keys(result_unit)
    if not source_bases or not result_bases:
        return None

    for smaller_unit, larger_unit, conversion_name, default_factor in STANDARD_CONVERSION_FACTORS:
        factor = default_factor
        if conversion_name in entities:
            value = normalize_empty(entities[conversion_name].get("value"))
            if is_number(value):
                factor = parse_numeric(value)

        if smaller_unit in source_bases and larger_unit in result_bases:
            return factor, "/"
        if larger_unit in source_bases and smaller_unit in result_bases:
            return factor, "*"

    return None


def safe_eval_semantic_unit_conversion(
    step: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
) -> Optional[float | int]:
    expr_tokens = extract_expr_tokens(step["expr"])
    if len(expr_tokens) != 1:
        return None

    source_name = expr_tokens[0]
    source = entities.get(source_name)
    if not source:
        return None

    source_unit = normalize_empty(source.get("unit"))
    result_unit = normalize_empty(step.get("result_unit"))
    if source_unit is None or result_unit is None or source_unit == result_unit:
        return None

    conversion = conversion_factor_for_units(source_unit, result_unit, entities)
    if conversion is None:
        return None

    source_value = normalize_empty(source.get("value"))
    if not is_number(source_value):
        return None

    factor, operator = conversion
    if operator == "/":
        result = parse_numeric(source_value) / factor
    else:
        result = parse_numeric(source_value) * factor

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

    return conversion_factor_for_units(source_unit, result_unit, existing_entities) is not None


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
            **(
                {"source": str(source).strip()}
                if (source := normalize_empty(entity.get("source"))) is not None
                else {}
            ),
        }
        for name, entity in problem_entities.items()
    }

    for step_name, step in plan.items():
        result = step["result"]
        result_unit = normalize_empty(step.get("result_unit"))
        result_grand_unit = normalize_empty(step.get("result_grand_unit"))

        computed_value: Optional[float | int] = None
        if looks_like_unit_conversion(step, plan_entities):
            computed_value = safe_eval_semantic_unit_conversion(step, plan_entities)

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
        code_text: Optional[str] = None
        last_code_response: Optional[str] = None

        for _ in range(max_retries()):
            raw_response = call_openrouter_code(
                problem,
                previous_error=previous_error,
            )
            last_code_response = raw_response

            try:
                candidate_plan = plan_from_code(raw_response, problem_entities, problem=problem)
            except PlannerError as exc:
                previous_error = str(exc)
                last_validation_error = exc
                continue

            plan = candidate_plan
            code_text = strip_code_fence(raw_response)
            break

        if plan is None:
            if last_code_response:
                write_text_file(CODE_PATH, strip_code_fence(last_code_response))
            raise PlannerError(str(last_validation_error))

        if code_text is not None:
            write_text_file(CODE_PATH, code_text)
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
