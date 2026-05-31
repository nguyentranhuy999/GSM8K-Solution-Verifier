# GSM8K Solution Verifier

Pipeline formalize, solve, and verify lời giải bài toán dạng GSM8K. Dự án dùng LLM để chuyển đề bài/lời giải học sinh thành các file YAML có cấu trúc, sau đó dùng Python validator/executor để kiểm tra quan hệ tính toán và so sánh lời giải chuẩn với lời giải học sinh.

## Mục Tiêu

- Trích xuất entity số từ đề bài.
- Sinh kế hoạch giải symbolic cho lời giải chuẩn.
- Thực thi kế hoạch để lấy đáp án target.
- Formalize lời giải học sinh thành các bước symbolic.
- Kiểm tra lỗi nội tại trong lời giải học sinh.
- Map entity giữa lời giải chuẩn và lời giải học sinh.
- So sánh hai lời giải và ghi diagnosis.

## Cấu Trúc Chính

```text
Input/
  Problem.txt                 # Đề bài hiện tại
  StudentAnswer.txt           # Lời giải học sinh hiện tại
  TeacherAnswer.txt           # Lời giải chuẩn giáo viên nếu dùng reference teacher

Main/
  Solver.py                   # Chạy pipeline giải chuẩn
  Tutor.py                    # Tự giải bằng solver rồi chấm lời giải học sinh
  Grader.py                   # Chấm lời giải học sinh bằng lời giải giáo viên có sẵn

Formalizer/
  ProblemFormalizer.py        # Đề bài -> ProblemEntities.yaml
  TeacherAnswerFormalizer.py  # Lời giải giáo viên -> TeacherPlan.yaml + TeacherAnswerEntities.yaml
  StudentAnswerFormalizer.py  # Bài làm học sinh -> StudentPlan.yaml + StudentAnswerEntities.yaml
  Mapper.py                   # Map entity student <-> plan
  Solver/
    Planner.py                # ProblemEntities.yaml -> Plan.yaml
    Executor.py               # Execute Plan.yaml -> PlanEntities.yaml, có InsideChecker/repair

Verifier/
  InsideChecker.py            # Check consistency nội tại
  CompareChecker.py           # So sánh lời giải chuẩn và lời giải học sinh

Benchmark/
  GSM8K Benchmark.csv
  RunSolveBenchmark.py        # Chạy Solver.py trên benchmark

Output/
  ProblemEntities.yaml
  Plan.yaml
  PlanEntities.yaml
  StudentPlan.yaml
  StudentAnswerEntities.yaml
  Diagnosis.yaml
  Error.yaml
  Wrong.yaml
  Log.yaml
```

## Cài Đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tạo file `.env` ở root:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
```

`OPENROUTER_MODEL` có thể đổi sang model khác OpenRouter hỗ trợ. Nếu không đặt, một số module có default là `google/gemini-2.0-flash-001`.

## Chạy Pipeline Giải Chuẩn

Ghi đề bài vào:

```text
Input/Problem.txt
```

Chạy:

```bash
python3 Main/Solver.py
```

Pipeline này chạy:

```text
ProblemFormalizer -> Planner -> Executor
```

`Executor.py` tự gọi `InsideChecker.py --mode llm` và repair bằng LLM nếu `Output/Error.yaml` có lỗi.

Output quan trọng:

- `Output/ProblemEntities.yaml`: entity từ đề bài.
- `Output/Plan.yaml`: kế hoạch giải symbolic.
- `Output/PlanEntities.yaml`: entity sau khi execute, gồm target answer.
- `Output/Error.yaml`: lỗi nội tại của plan nếu có.
- `Output/Log.yaml`: trạng thái pass/fail từng stage.

Target answer là entity duy nhất có:

```yaml
location: target
```

trong `Output/PlanEntities.yaml`.

## Chạy Tutor Pipeline

Ghi đề bài vào:

```text
Input/Problem.txt
```

Ghi lời giải học sinh vào:

```text
Input/StudentAnswer.txt
```

Chạy:

```bash
python3 Main/Tutor.py
```

Tutor tự giải rồi tự chấm:

```text
Solver.py
StudentAnswerFormalizer.py
InsideChecker.py --mode student
Mapper.py
CompareChecker.py
```

Output quan trọng:

- `Output/Plan.yaml`: lời giải solver tự lập.
- `Output/PlanEntities.yaml`: entity của lời giải solver sau execute.
- `Output/StudentPlan.yaml`: các bước học sinh làm.
- `Output/StudentAnswerEntities.yaml`: entity từ lời giải học sinh.
- `Output/Diagnosis.yaml`: nhãn lỗi/kết quả so sánh.
- `Output/Wrong.yaml`: `Yes` hoặc `No`.

## Chạy Grader Pipeline

Ghi đề bài, lời giải học sinh và lời giải giáo viên có sẵn vào:

```text
Input/Problem.txt
Input/StudentAnswer.txt
Input/TeacherAnswer.txt
```

Chạy:

```bash
python3 Main/Grader.py
```

Grader chấm dựa trên lời giải giáo viên:

```text
ProblemFormalizer.py
StudentAnswerFormalizer.py
TeacherAnswerFormalizer.py
InsideChecker.py --mode student
Mapper.py --reference teacher
CompareChecker.py --reference teacher
```

`Grader.py` là luồng teacher-vs-student riêng. Nó không gọi `Tutor.py` và không
dùng `Plan.yaml`/`PlanEntities.yaml` làm reference để chấm.

Output quan trọng:

- `Output/TeacherPlan.yaml`: các bước giáo viên làm.
- `Output/TeacherAnswerEntities.yaml`: entity từ lời giải giáo viên.
- `Output/StudentPlan.yaml`: các bước học sinh làm.
- `Output/StudentAnswerEntities.yaml`: entity từ lời giải học sinh.
- `Output/Diagnosis.yaml`: nhãn lỗi/kết quả so sánh.
- `Output/Wrong.yaml`: `Yes` hoặc `No`.

## Format Entity YAML

Mỗi entity có các trường chính:

```yaml
entity_name:
  value: 123
  unit: dollars
  location: input
  grand_unit: dollars
```

Sau khi execute/formalize student, entity có thêm:

```yaml
expr: entity_a + entity_b
formalized_expr: ...
map: other_entity_name
```

Quy ước `location`:

- `input`: dữ kiện trực tiếp trong đề.
- `target`: đại lượng cần tìm.
- `step1`, `step2`, ...: entity trung gian được tạo bởi step.

Target luôn nên giữ `location: target`, kể cả khi value được tính ở step cuối.

## Format Plan YAML

```yaml
step1:
  expr: morning_coffee_price + afternoon_coffee_price
  result: daily_cost
  result_unit: dollars
  result_grand_unit: dollars

step2:
  expr: daily_cost * days
  result: total_cost
  result_unit: dollars
  result_grand_unit: dollars
```

Sau `Executor.py`, mỗi step được thêm:

```yaml
reported_expr: 3.0 + 2.5 = 5.5
```

## Benchmark Solver

Chạy solver trên benchmark và so sánh target value với cột `offical answer`:

```bash
python3 Benchmark/RunSolveBenchmark.py --limit 200 --workers 4
```

Output mặc định:

```text
Error/results_summary.json
Error/Summary.md
Error/index.html
Error/<id>/ProblemEntities.yaml
Error/<id>/Plan.yaml
Error/<id>/PlanEntities.yaml
```

CSV không ghi mặc định. Nếu cần CSV:

```bash
python3 Benchmark/RunSolveBenchmark.py --limit 200 --workers 4 --write-csv
```

CSV benchmark khi bật `--write-csv` có các snapshot để debug:

- `problem_entities_yaml`
- `plan_yaml`
- `plan_entities_yaml`
- `solver_stdout`
- `solver_stderr`
- `error_stage`

Các option hữu ích:

```bash
--limit 200          # số bài chạy
--workers 4         # chạy song song bằng workspace tạm riêng
--timeout 300       # timeout mỗi bài
--resume            # tiếp tục từ output CSV hiện có
--tolerance 0.000001
```

Với `--workers > 1`, mỗi row chạy trong một copy workspace tạm để tránh va chạm `Input/` và `Output/`.

## Debug

Nếu một bài sai hoặc fail, đọc theo thứ tự:

1. `Output/Log.yaml`: stage nào fail.
2. `Output/ProblemEntities.yaml`: entity đề bài có đúng không.
3. `Output/Plan.yaml`: plan có quan hệ đúng không.
4. `Output/PlanEntities.yaml`: target value có đúng không.
5. `Output/Error.yaml`: lỗi nội tại của lời giải chuẩn.
6. `Output/StudentPlan.yaml`: học sinh có bị gộp/bỏ bước không.
7. `Output/Diagnosis.yaml`: nhãn lỗi cuối cùng.

Trong benchmark, mở file CSV và xem các cột snapshot YAML của row sai.

## Một Số Lưu Ý Thiết Kế

- LLM sinh YAML nhưng Python validator giữ các invariant quan trọng.
- Prompt là ràng buộc mềm; validator là ràng buộc cứng.
- `ProblemFormalizer` chặn input value tự tính như `49` nếu đề chỉ nói `7` và `seven times`.
- `Planner` có retry khi plan bị validator reject.
- `Executor` không được sửa entity gốc từ `ProblemEntities.yaml`; nó chỉ execute/repair plan và entity trung gian.
- `InsideChecker` dùng cho cả reference mode và student mode.

## Các Biến Môi Trường Hữu Ích

```env
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions

PLANNER_MAX_RETRIES=3
EXECUTOR_MAX_REPAIR_ITERATIONS=5
STUDENT_FORMALIZER_MAX_RETRIES=3

# Strict mode: bắt ProblemFormalizer phải trích mọi số/scalar trực tiếp trong đề.
PROBLEM_FORMALIZER_REQUIRE_ALL_DIRECT_VALUES=1
```

## Trạng Thái Dự Án

Dự án hiện phù hợp cho prototype/research pipeline. Các bước chính đã có validation và repair loop, nhưng vẫn nên thêm test tự động cho các case dễ lỗi:

- fraction/scalar: `1/3`, `a fourth`, `twice`
- target giữ `location: target`
- không sinh số tự tính trong `ProblemEntities.yaml`
- không gộp step trong `StudentPlan.yaml`
- benchmark không bị stale output
