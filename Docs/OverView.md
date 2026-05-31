# GSM8K Solution Verifier Overview

Tài liệu này tổng hợp nội dung thiết kế ban đầu trong `Docs/Prompt.pdf` và các
thay đổi hiện tại đã ghi trong `Docs/Changes.md`.

Mục tiêu của dự án là biến đề toán GSM8K và lời giải học sinh thành các file
YAML có cấu trúc, sau đó dùng code Python để thực thi, kiểm tra nội tại, map và
so sánh lời giải chuẩn với lời giải học sinh.

## Mục tiêu hệ thống

Hệ thống xử lý một bài toán theo hai nhánh:

- Nhánh lời giải chuẩn: formalize đề bài, lập kế hoạch giải, thực thi kế hoạch
  để lấy đáp án target.
- Nhánh lời giải học sinh: formalize các bước học sinh làm, kiểm tra lỗi nội
  tại, map với lời giải chuẩn, rồi so sánh hai lời giải.

Thiết kế hiện tại là hybrid:

- LLM dùng để hiểu ngôn ngữ tự nhiên và sinh YAML/plan ban đầu.
- Python validator, executor và deterministic planner dùng để khóa schema, tính
  toán, bắt lỗi và giảm phụ thuộc vào LLM.

## Pipeline chính

### Solver pipeline

`Main/Solver.py` chạy pipeline lời giải chuẩn:

```text
ProblemFormalizer -> Planner -> Executor
```

Output chính:

- `Output/ProblemEntities.yaml`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/Error.yaml`
- `Output/Log.yaml`

`Executor.py` tự gọi `Verifier/InsideChecker.py --mode llm` và tự repair bằng
LLM nếu InsideChecker báo lỗi, nên `Main/Solver.py` không gọi InsideChecker thêm
lần nữa.

### Tutor pipeline

`Main/Tutor.py` chạy luồng tự giải và tự chấm: hệ thống tự tạo lời giải chuẩn
bằng solver, sau đó chấm lời giải học sinh theo lời giải đó.

```text
Solver
StudentAnswerFormalizer
InsideChecker --mode student
Mapper
CompareChecker
```

Output chính:

- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`
- `Output/Diagnosis.yaml`
- `Output/Wrong.yaml`

### Grader pipeline

`Main/Grader.py` chạy pipeline chấm lời giải học sinh bằng lời giải giáo viên.
Đây là luồng riêng với `Tutor.py`/`Solver.py`, không dùng solver reference để
chấm.

```text
ProblemFormalizer
StudentAnswerFormalizer
TeacherAnswerFormalizer
InsideChecker --mode student
Mapper --reference teacher
CompareChecker --reference teacher
```

Output chính:

- `Output/TeacherPlan.yaml`
- `Output/TeacherAnswerEntities.yaml`
- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`
- `Output/Diagnosis.yaml`
- `Output/Wrong.yaml`

## Data contract

### Entity YAML

Mỗi entity ban đầu có 4 trường chính:

```yaml
entity_name:
  value: 123
  unit: dollars
  location: input
  grand_unit: dollars
```

Ý nghĩa:

- `value`: giá trị số. Với `target`, value để rỗng.
- `unit`: đơn vị trực tiếp của entity.
- `location`: `input`, `target`, hoặc `stepN` sau khi entity trung gian được tạo.
- `grand_unit`: đơn vị đối chiếu theo target.

Sau khi execute hoặc formalize lời giải học sinh, entity có thể có thêm:

```yaml
expr: entity_a + entity_b
formalized_expr: ...
map: other_entity
```

Quy ước quan trọng hiện tại:

- `ProblemEntities.yaml` chỉ cho phép `location` là `input` hoặc `target`.
- Target phải giữ `location: target`, kể cả sau khi đã tính được value.
- Các helper entity như `unit_conversion_*`, `identity_multiplier`,
  `percentage_scale` là input do code thêm để giữ `expr` không có số literal.

### Plan YAML

Mỗi step có 4 trường:

```yaml
step1:
  expr: morning_coffee_price + afternoon_coffee_price
  result: daily_cost
  result_unit: dollars
  result_grand_unit: dollars
```

Sau khi `Executor.py` chạy, step có thêm:

```yaml
reported_expr: 3.0 + 2.5 = 5.5
```

Quy ước hiện tại:

- Step phải liên tục theo thứ tự `step1`, `step2`, ...
- `expr` chỉ được dùng entity/result và toán tử đơn giản.
- `expr` không được chứa số literal như `1`, `2`, `0.5`, `36`.
- Bước cuối phải tạo đúng target.

## Module overview

### `Formalizer/ProblemFormalizer.py`

Vai trò theo prompt gốc:

- Đọc `Input/Problem.txt`.
- Gọi LLM qua OpenRouter để trích xuất các entity số trực tiếp trong đề.
- Không tính entity trung gian.
- Tạo đúng một target có value rỗng.
- Ghi `Output/ProblemEntities.yaml`.
- Copy entity sang `Output/PlanEntities.yaml` và
  `Output/StudentAnswerEntities.yaml`.

Mở rộng hiện tại và mục đích:

- Validate schema entity bằng Python để reject YAML sai format ngay tại
  `ProblemFormalizer`, thay vì để lỗi lan sang Planner/Executor.
- Parse/validate số trực tiếp bằng code như digit, decimal `.50`, fraction,
  mixed fraction, số viết bằng chữ, percent, multiplier words, `twins`. Mục đích
  là giảm lỗi LLM bỏ sót số trực tiếp hoặc không hiểu số viết bằng chữ.
- Reject value input không được nói trực tiếp trong đề để giữ đúng ranh giới:
  `ProblemEntities.yaml` chỉ chứa dữ kiện trực tiếp, không chứa kết quả suy ra.
- Tự bỏ complement fraction do LLM tự tính, ví dụ `girls_fraction = 0.6` khi đề
  chỉ cho `2/5 are boys`. Mục đích là chặn LLM “giúp” giải trước bài toán ở
  formalizer.
- Bỏ placeholder sai như `age = 0` nếu đề không có số 0. Mục đích là tránh việc
  LLM bịa số để lấp thông tin quan hệ như `younger brother`.
- Sửa lỗi LLM đổi đơn vị quá sớm, ví dụ giữ `12000 meters` thay vì tự đổi thành
  `12 kilometers`. Mục đích là để unit conversion thuộc Planner/Executor, không
  nằm trong ProblemFormalizer.
- Thêm helper entity như `unit_conversion_*`, `identity_multiplier`,
  `percentage_scale`, `host_count`, `split_count`, family context counts. Mục
  đích là cho Planner có biến symbolic để dùng, không phải viết số literal trong
  `expr`.
- Validate target name cho một số dạng câu hỏi dễ nhầm như invite friends,
  `not in`, `give ... each`. Mục đích là bắt lỗi LLM chọn target trung gian hoặc
  target sai đại lượng được hỏi.

### `Formalizer/Solver/Planner.py`

Vai trò theo prompt gốc:

- Đọc đề bài và `Output/ProblemEntities.yaml`.
- Gọi LLM sinh kế hoạch symbolic.
- Ghi `Output/Plan.yaml`.
- Thêm các result entity vào `Output/PlanEntities.yaml` bằng Python.
- Với step đổi đơn vị đơn giản, có thể tính value cho result ngay ở Planner.

Mở rộng hiện tại và mục đích:

- Enforce `expr` không chứa numeric literal. Mục đích là giữ plan hoàn toàn
  symbolic, giúp Executor, Mapper và CompareChecker xử lý nhất quán.
- Normalize alias cũ `grand_result_unit` về `result_grand_unit` để tương thích
  với output prompt cũ nhưng vẫn lưu schema thống nhất.
- Validate step liên tục và target chỉ được tạo ở bước cuối để tránh plan thiếu
  bước, đảo bước hoặc tạo target sớm rồi tiếp tục tính vòng.
- Giữ target `location: target` trong `PlanEntities.yaml` để không làm mất thông
  tin đâu là đáp án cuối khi target đã được tính value.
- Thêm validator logic/backbone để bắt các lỗi LLM hay lặp lại:
  - double count item: tránh nhân thêm count khi giá từng item đã được liệt kê;
  - scalar quan trọng không được bỏ: tránh bỏ fraction/percent/multiplier;
  - rate theo thời gian phải đổi đúng horizon: tránh lấy rate tháng cộng/trừ với
    tổng năm;
  - quan hệ `more_than`, `less_than`, `fewer_than` phải resolve trước: tránh dùng
    độ chênh như số lượng thật;
  - percentage target phải nhân `percentage_scale`: tránh trả `0.2` khi đáp án
    cần `20 percent`;
  - discount threshold không phải quantity thật: tránh dùng ngưỡng discount như
    số lượng mua;
  - roommate split tính cả người chủ khi phù hợp: tránh chia bill thiếu một
    người;
  - invite friends không trừ host hai lần: tránh tính số bạn mời bị lệch;
  - family ticket discount không áp discount cho cả parents/grandparents: tránh
    áp rule age threshold cho toàn bộ family;
  - herd/calves phải cộng cả adult animals ban đầu: tránh chỉ tính calves mới;
  - bill/coin change phải đổi qua dollar amount rồi chia denomination: tránh lấy
    fraction của số tờ cũ làm số tờ mới;
  - allocation fraction không được copy lượng nhóm A sang nhóm B nếu vượt tổng:
    tránh phân bổ tổng tài nguyên quá 100%.
- Thêm deterministic planner cho các schema `give to each ... same amount`,
  linear shares, sales multiplier, bill/coin change, allocation fraction. Mục
  đích là xử lý các pattern đại số phổ biến mà LLM thường sinh plan sai, nhất là
  lỗi dùng target như biến đã tồn tại.

Điểm khác lớn nhất so với prompt gốc: Planner không còn hoàn toàn phụ thuộc LLM.
Nếu nhận ra schema deterministic, code tự sinh plan trước; nếu không match thì
mới gọi LLM.

### `Formalizer/Solver/Executor.py`

Vai trò theo prompt gốc:

- Đọc `Output/Plan.yaml` và `Output/PlanEntities.yaml`.
- Thực thi từng step bằng Python.
- Gắn `reported_expr` vào `Plan.yaml`.
- Gắn `value`, `expr`, `formalized_expr` vào `PlanEntities.yaml`.
- Gọi InsideChecker và repair nếu có lỗi.

Mở rộng hiện tại và mục đích:

- Dùng evaluator an toàn cho arithmetic expression để Executor tính số bằng
  Python nhưng không chạy code tùy ý từ YAML.
- Tự gọi `Verifier/InsideChecker.py --mode llm` để sau khi tính xong có một lớp
  kiểm tra nội tại trước khi coi lời giải chuẩn là hợp lệ.
- Nếu InsideChecker còn lỗi, gọi LLM repair plan/entities rồi chạy lại. Mục đích
  là cho pipeline tự rollback các lỗi cấu trúc nhỏ thay vì dừng ngay.
- Nếu `Output/Error.yaml` chỉ có `extra step`, Executor coi như pass để tránh
  repair vô ích vì extra step thường không làm sai đáp án chính.
- LLM repair dùng `OPENROUTER_MAX_TOKENS` để kiểm soát token giống các module
  LLM khác.

### `Formalizer/StudentAnswerFormalizer.py`

Vai trò theo prompt gốc:

- Đọc đề bài, `ProblemEntities.yaml` và `Input/StudentAnswer.txt`.
- Gọi LLM sinh `Output/StudentPlan.yaml`.
- Mỗi step gồm `expr`, `result`, `result_unit`, `result_grand_unit`,
  `reported_expr`.
- Thêm result entity vào `Output/StudentAnswerEntities.yaml`.
- Ghi diagnosis đặc biệt như spelling errors, word problem, answer by word.

Mở rộng hiện tại và mục đích:

- `reported_expr` phải giữ đúng phép tính học sinh viết hoặc ngụ ý theo từng
  dòng để checker nhìn thấy lỗi thật của học sinh.
- Không được thay bằng phép tính tương đương hoặc gộp bước, vì gộp/bỏ bước sẽ
  làm mất thông tin học sinh đã suy luận như thế nào.
- Validate số trong `reported_expr` phải xuất hiện trong bài làm học sinh để LLM
  không tự thêm số hoặc tự sửa phép tính.
- Extract equation học sinh viết và bắt StudentPlan giữ đúng thứ tự để so sánh
  được với lời giải thực tế trong file txt.
- `expr` vẫn map về entity chuẩn, nhưng `reported_expr` giữ số học sinh dùng sai
  nếu học sinh đọc sai đề. Mục đích là tách quan hệ symbolic khỏi lỗi đọc số.
- Ghi vào `Output/Diagnosis.yaml`, không dùng spelling sai `Diagonosis.yaml`, để
  các checker downstream đọc đúng một file diagnosis.

### `Main/Solver.py`, `Main/Tutor.py` và `Main/Grader.py`

Các file này là phần bổ sung ngoài prompt gốc.

Mở rộng hiện tại và mục đích:

- `Main/Solver.py` gom pipeline lời giải chuẩn thành một command duy nhất:
  `ProblemFormalizer -> Planner -> Executor`. Mục đích là tránh phải chạy từng
  stage thủ công khi chỉ muốn giải bài toán.
- `Main/Solver.py` không gọi InsideChecker riêng vì `Executor.py` đã tự gọi
  InsideChecker và repair. Mục đích là tránh check trùng và tránh repair hai
  lần trên cùng một plan.
- `Main/Tutor.py` chạy luồng tự giải và tự chấm:
  `Solver -> StudentAnswerFormalizer -> Mapper -> CompareChecker`. Mục đích là
  kiểm tra bài học sinh bằng lời giải do hệ thống tự sinh.
- `Main/Grader.py` chạy phần chấm teacher-vs-student:
  `ProblemFormalizer -> StudentAnswerFormalizer -> TeacherAnswerFormalizer ->
  Mapper teacher -> CompareChecker teacher`. Mục đích là benchmark verifier dựa
  trên lời giải giáo viên, không bị lẫn lỗi từ solver.
- Các stage được chạy bằng subprocess theo thứ tự. Mục đích là mỗi module vẫn có
  thể chạy độc lập, nhưng khi cần thì có pipeline tổng hợp.

### `Formalizer/Mapper.py`

Vai trò theo prompt gốc:

- Map entity trong `StudentAnswerEntities.yaml` với entity trong
  `PlanEntities.yaml`.
- Entity chung từ `ProblemEntities.yaml` được auto map.
- Entity trung gian được map theo quan hệ trong `expr` hoặc `formalized_expr`.
- Mapper chạy bằng Python, không dùng LLM.

Mục tiêu của Mapper là giúp CompareChecker hiểu entity nào của học sinh tương
ứng với entity nào trong lời giải chuẩn, kể cả khi tên step/result khác nhau.

### `Verifier/InsideChecker.py`

Vai trò theo prompt gốc:

- Check lỗi nội tại của một plan/entities.
- Có hai mode:
  - `llm`: đọc `Plan.yaml` và `PlanEntities.yaml`, ghi `Error.yaml`.
  - `student`: đọc `StudentPlan.yaml` và `StudentAnswerEntities.yaml`, ghi
    `Diagnosis.yaml`.

Các lỗi chính:

- `wrong target`
- `wrong calculation`
- `unit missing`
- `only final answer`
- `wrong relationship`
- `do not convert units`
- `missing step`
- `misreading`
- `logic error`
- `extra step`

Policy `Wrong.yaml`:

- Nếu có lỗi khác `extra step`, ghi `Yes`.
- Nếu chỉ có `extra step`, ghi `No`.

### `Verifier/CompareChecker.py`

Vai trò theo prompt gốc:

- So sánh lời giải chuẩn và lời giải học sinh sau khi đã map entity.
- Đọc:
  - `Output/Plan.yaml`
  - `Output/PlanEntities.yaml`
  - `Output/StudentPlan.yaml`
  - `Output/StudentAnswerEntities.yaml`
- Ghi `Output/Diagnosis.yaml`.

Các lỗi/kết quả chính:

- `wrong units conversion`
- `combine step`
- `step separation`
- `reverse steps`
- `all right`
- `wrong relationship`
- `different calculation`

Policy `Wrong.yaml`:

- `wrong relationship` ghi `Yes`.
- Các trường hợp còn lại như cách làm khác, gộp/tách/đảo bước nhưng đúng quan hệ
  thường ghi `No`.

### `Verifier/LLMChecker.py`

`Prompt.pdf` chỉ bắt đầu nhắc đến `Verifier/LLMChecker.py` nhưng chưa mô tả đầy
đủ trong phần được trích. Vì vậy tài liệu này chưa coi LLMChecker là thành phần
pipeline chính.

## Benchmark

`Benchmark/RunSolveBenchmark.py` là phần bổ sung ngoài prompt gốc.

Mở rộng hiện tại và mục đích:

- Chạy `Main/Solver.py` trên benchmark GSM8K để kiểm tra solver trên nhiều bài,
  không chỉ một bài trong `Input/Problem.txt`.
- So sánh target value với cột `offical answer` hoặc `official answer` để đo %
  đúng của pipeline lời giải chuẩn.
- Ghi result CSV, wrong CSV và summary JSON để biết số bài pass, số bài sai, số
  bài lỗi formalize/planner/executor.
- Lưu các snapshot debug trong CSV như `problem_entities_yaml`, `plan_yaml`,
  `plan_entities_yaml`, `solver_stdout`, `solver_stderr`, `error_stage`. Mục
  đích là khi một bài sai có thể nhìn ngay lỗi nằm ở stage nào và YAML lúc đó ra
  sao.
- Hỗ trợ `--workers > 1` bằng workspace tạm riêng cho từng row. Mục đích là chạy
  song song mà không bị các bài ghi đè cùng `Input/` và `Output/`.
- Hỗ trợ `--indices` để chạy lại một nhóm bài lỗi cụ thể. Mục đích là fix
  backbone theo vòng lặp nhỏ thay vì chạy lại toàn bộ benchmark mỗi lần.

Option đáng chú ý:

```bash
python3 Benchmark/RunSolveBenchmark.py --limit 200 --workers 4
python3 Benchmark/RunSolveBenchmark.py --limit 100 --indices 27,36,51
```

Với `--workers > 1`, mỗi row chạy trong workspace tạm riêng để tránh ghi đè
`Input/` và `Output/`.

## LLM usage

Các điểm có gọi LLM:

- `ProblemFormalizer.py`: formalize đề bài thành entity.
- `Planner.py`: sinh plan nếu deterministic planner không match.
- `Executor.py`: repair plan/entities khi InsideChecker báo lỗi.
- `StudentAnswerFormalizer.py`: formalize lời giải học sinh.

Các phần không dùng LLM:

- `Mapper.py`
- `InsideChecker.py`
- `CompareChecker.py`
- Executor arithmetic evaluation.

## Environment

Các module LLM đọc cấu hình OpenRouter từ `.env` hoặc environment:

```env
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MAX_TOKENS=4096
```

Nếu không đặt `OPENROUTER_MODEL`, một số module default về
`google/gemini-2.0-flash-001`.

Khi tài khoản OpenRouter thiếu credit, có thể tạm giảm max tokens bằng:

```bash
OPENROUTER_MAX_TOKENS=3000 python3 Main/Solver.py
```

## Thiết kế hiện tại

Các nguyên tắc đang được giữ:

- LLM hiểu ngôn ngữ, Python giữ kỷ luật schema và tính toán.
- `ProblemEntities.yaml` không chứa entity trung gian do LLM tự tính.
- `Plan.yaml` không chứa số literal trong `expr`; mọi số phải là entity.
- Target không bị đổi `location` sau khi tính.
- Lời giải học sinh phải giữ đúng phép tính học sinh viết trong
  `reported_expr`, kể cả khi học sinh sai.

Các khác biệt có chủ ý so với prompt gốc:

- Có helper entity do code thêm, ví dụ `identity_multiplier`,
  `unit_conversion_*`, `percentage_scale`.
- Planner có deterministic fallback cho một số schema.
- Có benchmark runner và output snapshot để debug.
- Có nhiều validator backbone hơn prompt ban đầu để giảm lỗi LLM lặp lại.

## File quan trọng

```text
Input/
  Problem.txt
  StudentAnswer.txt

Main/
  Solver.py
  Tutor.py
  Grader.py

Formalizer/
  ProblemFormalizer.py
  StudentAnswerFormalizer.py
  Mapper.py
  Solver/
    Planner.py
    Executor.py

Verifier/
  InsideChecker.py
  CompareChecker.py

Benchmark/
  RunSolveBenchmark.py
  GSM8K Benchmark.csv

Output/
  ProblemEntities.yaml
  Plan.yaml
  PlanEntities.yaml
  StudentPlan.yaml
  StudentAnswerEntities.yaml
  Error.yaml
  Diagnosis.yaml
  Wrong.yaml
  Log.yaml
```
