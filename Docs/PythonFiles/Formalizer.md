# Formalizer Files

Nhóm `Formalizer/` biến input ngôn ngữ tự nhiên hoặc plan thành các YAML có cấu
trúc. Đây là phần nặng nhất của dự án vì nó phải cân bằng giữa:

- LLM để hiểu đề/lời giải;
- Python để khóa schema, tính toán, validate và chống hallucination.

## `Formalizer/ProblemFormalizer.py`

### Vai Trò

`ProblemFormalizer.py` đọc `Input/Problem.txt`, gọi LLM qua OpenRouter và sinh:

- `Output/ProblemEntities.yaml`
- bản copy ban đầu sang các entity file downstream tùy mode:
  `PlanEntities.yaml`, `StudentAnswerEntities.yaml`, hoặc
  `TeacherAnswerEntities.yaml`

Mục tiêu của file này là trích xuất các số được nói trực tiếp trong đề bài,
không giải bài toán và không tạo kết quả trung gian do phải tính mới có.

### Contract Output

Mỗi entity phải có bốn trường:

```yaml
entity_name:
  value: 123
  unit: dollars
  location: input
  grand_unit: dollars
```

Quy tắc:

- `location` chỉ được là `input` hoặc `target`.
- Có đúng một target.
- Target có `value` rỗng.
- Entity input phải là số xuất hiện trực tiếp trong đề hoặc helper constant do
  code thêm.

### Cách Hoạt Động Thực Tế

Các bước chính trong `run()`:

1. Đọc đề bài.
2. Gọi LLM sinh YAML entity.
3. Parse YAML, normalize empty value.
4. Validate schema và value.
5. Thêm/sửa helper entity cần thiết.
6. Ghi `ProblemEntities.yaml`.
7. Copy sang `PlanEntities.yaml` và `StudentAnswerEntities.yaml`.
8. Ghi trạng thái vào `Output/Log.yaml`.

### Validator Và Normalize

Code hiện tại không tin hoàn toàn vào LLM. Sau khi LLM trả YAML, Python kiểm tra:

- tên entity phải là snake_case;
- entity phải đủ `value`, `unit`, `location`, `grand_unit`;
- không có field thừa;
- `location` phải đúng;
- target phải duy nhất;
- input value phải xuất hiện trực tiếp trong đề.

File này có các parser số bằng code:

- số digit: `12`, `3.5`, `.50`;
- phân số: `1/3`;
- mixed fraction: `3 3/8`;
- số viết bằng chữ: `two`, `twenty`, `one hundred`;
- percent dạng `%`, `percent`, hoặc chữ;
- multiplier words: `twice`, `double`, `triple`, `twins`.

### Helper Entity

Khác với prompt gốc, code có thể thêm helper constants bằng Python:

- `unit_conversion_*`: ví dụ `unit_conversion_inches_per_foot`;
- `identity_multiplier = 1`;
- `percentage_scale = 100`;
- `host_count = 1`;
- `split_count = 2`;
- family context counts như `self_count`, `sibling_count`,
  `parents_count`, `grandparents_count`.

Các helper này không phải là kết quả trung gian do LLM tính ra. Chúng là hằng số
backbone để Planner có thể viết `expr` chỉ bằng biến, không viết số literal.

### Các Lỗi LLM Được Sửa Bằng Code

`ProblemFormalizer.py` có logic sửa hoặc reject các lỗi thường gặp:

- LLM tự tính complement fraction, ví dụ đề cho `2/5 are boys` rồi tự thêm
  `girls_fraction = 3/5`.
- LLM tạo placeholder `0` dù đề không nói số 0.
- LLM đổi đơn vị quá sớm, ví dụ đề nói `12000 meters` nhưng entity thành
  `12 kilometers`.
- LLM chọn target sai đại lượng, ví dụ bài hỏi "friends can invite" nhưng target
  lại là cost.
- LLM đặt grand unit sai trong các bài item/pair/relation.

### Tại Sao Thiết Kế Như Vậy

Nếu để LLM tự do formalize, nó hay "giải hộ" ở bước entity extraction. Điều đó
làm Planner và Checker khó biết số nào là dữ kiện gốc, số nào là kết quả suy ra.

Vì vậy file này giữ ranh giới:

- LLM đọc đề và đề xuất entity.
- Python kiểm tra entity có thật sự grounded trong đề không.
- Helper constants được thêm có chủ đích để phục vụ symbolic plan.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc chỉ yêu cầu gọi LLM để trích entity số trực tiếp và copy YAML sang
hai file downstream. Code hiện tại mở rộng mạnh:

- thêm validator số trực tiếp;
- thêm parser số viết bằng chữ/fraction/percent;
- thêm helper constants;
- thêm normalize unit/grand_unit;
- thêm target-name validator;
- tăng retry và max token;
- chuẩn hóa log và schema.

Điểm giữ nguyên: `ProblemEntities.yaml` vẫn không được chứa kết quả trung gian
do bài toán phải tính mới ra.

## `Formalizer/Solver/Planner.py`

### Vai Trò

`Planner.py` tạo kế hoạch symbolic cho lời giải chuẩn.

Input:

- `Input/Problem.txt`
- `Output/ProblemEntities.yaml`

Output:

- `Output/Code.txt`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- log trong `Output/Log.yaml`

### Thiết Kế Hiện Tại

Planner hiện tại không còn chỉ yêu cầu LLM sinh `Plan.yaml` trực tiếp. Luồng mới:

```text
problem text
  -> LLM sinh pseudo-code Python tối giản vào Code.txt
  -> Python parse code assignments
  -> Python map số literal trong code về ProblemEntities
  -> Python tạo Plan.yaml symbolic
  -> Python thêm result entity vào PlanEntities.yaml
```

Mục đích của thay đổi này là giảm ảnh hưởng của tên entity. Khi LLM nhìn
`ProblemEntities.yaml` với tên như `pens_more_than_notebooks`, nó có thể suy luận
sai từ tên biến. Luồng pseudo-code cho LLM giải theo đề bài trước, sau đó code
mới map số về entity.

### Pseudo-Code Contract

LLM phải trả code thuần:

```python
morning_price = 3.00
afternoon_price = 2.50
days = 20
daily_cost = morning_price + afternoon_price
answer = daily_cost * days
```

Quy tắc:

- mỗi dòng là phép gán;
- dòng cuối là `answer = ...`;
- phase 1 bind dữ kiện số trong đề vào biến;
- phase 2 tính toán chỉ bằng biến đã bind;
- không import, không function, không loop, không if/else;
- không shortcut thành final answer literal.

### Map Code Sang Plan

Các hàm chính:

- `parse_code_assignments()`: parse từng dòng assignment.
- `validate_code_expr_ast()`: chỉ cho phép toán tử số học an toàn.
- `input_entity_candidates_for_number()`: tìm entity có value phù hợp với số trong code.
- `score_number_entity_candidate()`: chấm candidate theo value, unit, source/context.
- `map_number_to_entity()`: quyết định số literal trong code map về entity nào.
- `ast_to_symbolic_expr()`: đổi AST code thành expr symbolic.
- `plan_from_code()`: tạo `step1`, `step2`, ... từ code.

### Alias Entity

File vẫn giữ một đường prompt direct-plan legacy. Trong nhánh đó, entity được
ẩn tên thật thành:

```text
e1, e2, e3, ..., target
```

Mục đích là giảm việc LLM overfit tên biến. Tuy nhiên `run()` hiện ưu tiên
`call_openrouter_code()` và `plan_from_code()`.

### Validator Plan

Planner kiểm tra nhiều thứ trước khi ghi file:

- step phải liên tục `step1`, `step2`, ...
- mỗi step có `expr`, `result`, `result_unit`, `result_grand_unit`;
- `expr` không được chứa số literal;
- step cuối phải tạo đúng target;
- result name phải hợp lệ;
- không dùng entity chưa tồn tại;
- target giữ `location: target` trong `PlanEntities.yaml`.

### Validator Logic/Backbone

Planner có nhiều validator để bắt lỗi phổ biến:

- không double-count item đã liệt kê;
- không dùng threshold discount như quantity thật;
- rate theo ngày/tuần/tháng/năm phải đổi đúng horizon;
- `more_than`, `less_than`, `fewer_than` phải resolve thành quantity thật;
- target phần trăm phải nhân `percentage_scale`;
- roommate split phải tính cả host nếu đề nói người đó cũng chia;
- invite friends không được trừ host hai lần;
- bill/coin change phải đổi qua value tiền rồi mới tính số tờ/coin mới;
- family discount không áp discount cho cả family nếu chỉ trẻ em được discount;
- animal birth/trade phải giữ cả adult animals ban đầu;
- fraction allocation không được copy amount nhóm A sang nhóm B nếu vượt tổng.

### Merge Plan Vào Entities

`merge_plan_results_into_plan_entities()` thêm result entity vào
`PlanEntities.yaml`:

- `value` thường để rỗng để Executor tính;
- `unit` lấy từ `result_unit`;
- `location` là `stepN` nếu entity chưa là target;
- target giữ `location: target`;
- `grand_unit` lấy từ `result_grand_unit`;
- một số step đổi đơn vị có thể tính value sớm nếu đủ thông tin.

### Tại Sao Thiết Kế Như Vậy

Planner là nơi LLM dễ hallucinate nhất vì nó vừa phải hiểu đề vừa phải dùng đúng
entity. Thiết kế mới tách thành hai việc:

- LLM giải bài như text thuần bằng code đơn giản;
- Python chịu trách nhiệm biến code thành plan theo entity contract.

Cách này hy sinh một chút sự đơn giản, nhưng giảm lỗi "model nhìn tên biến rồi
suy luận sai".

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc: LLM nhận problem + entities và sinh `Plan.yaml`.

Code hiện tại:

- LLM sinh `Code.txt`;
- Python parse code và map số sang entity;
- vẫn có direct-plan prompt legacy nhưng không phải đường chính;
- có validator dày hơn nhiều;
- thêm rule cấm numeric literal trong `expr`;
- thêm nhiều logic chống lỗi benchmark.

## `Formalizer/Solver/Executor.py`

### Vai Trò

`Executor.py` là nơi tính toán numeric chính của solver reference.

Input:

- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`

Output:

- update `Output/Plan.yaml` với `reported_expr`;
- update `Output/PlanEntities.yaml` với `value`, `expr`, `formalized_expr`;
- gọi `Verifier/InsideChecker.py --mode llm`;
- nếu có lỗi, gọi LLM repair.

### Cách Hoạt Động

Các bước chính:

1. Đọc và normalize plan/entities.
2. Collapse target passthrough step nếu có bước copy target không cần thiết.
3. Prune stale step entities.
4. Với từng step:
   - lấy value của token trong `expr`;
   - thay token bằng value;
   - tính bằng evaluator an toàn;
   - tạo `reported_expr`;
   - gán value cho result entity.
5. Tạo `formalized_expr` bằng cách bung result trung gian về input entity.
6. Ghi lại `Plan.yaml` và `PlanEntities.yaml`.
7. Chạy InsideChecker.
8. Nếu InsideChecker báo lỗi không phải chỉ `extra step`, gọi LLM repair và lặp.

### Safe Evaluation

Executor không dùng `eval()` trực tiếp. Nó parse AST và chỉ cho phép:

- số;
- unary `+`/`-`;
- binary `+`, `-`, `*`, `/`;
- ngoặc.

Điều này tránh việc YAML do LLM sinh có thể chạy code tùy ý.

### Unit Conversion

Executor có logic `semantic_unit_conversion()`:

- nếu step là đổi đơn vị trực tiếp, ví dụ `expr: road_length`,
  Executor có thể tìm `unit_conversion_*` phù hợp;
- conversion factor được lấy từ entity helper hoặc bảng semantic unit;
- `reported_expr` vẫn thể hiện phép tính số cụ thể.

### Formalized Expression

Ví dụ:

```yaml
daily_cost:
  expr: morning_price + afternoon_price
  formalized_expr: morning_price + afternoon_price

total_cost:
  expr: daily_cost * days
  formalized_expr: (morning_price + afternoon_price) * days
```

`formalized_expr` giúp Mapper và CompareChecker so sánh các lời giải khác cách
biểu diễn.

### Repair Loop

Sau khi execute, Executor chạy InsideChecker. Nếu có lỗi:

- đọc `Input/Problem.txt`;
- đọc `Plan.yaml`, `PlanEntities.yaml`, `Error.yaml`;
- gọi LLM repair;
- parse output repair;
- apply lại plan/entities;
- execute lại.

`extra step` được coi là lỗi nhẹ; nếu chỉ có extra step thì không repair.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc đã yêu cầu Executor tính `reported_expr`, update value, thêm
`formalized_expr`, gọi InsideChecker và repair. Code hiện tại mở rộng:

- dùng AST safe evaluator;
- thêm collapse target passthrough;
- thêm prune stale entities;
- thêm semantic unit conversion;
- thêm retry/repair control;
- bỏ repair nếu chỉ có `extra step`.

## `Formalizer/StudentAnswerFormalizer.py`

### Vai Trò

Formalize lời giải học sinh trong `Input/StudentAnswer.txt` thành:

- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`
- có thể ghi `Output/Diagnosis.yaml`
- có thể ghi `Output/Wrong.yaml`

### Nguyên Tắc Quan Trọng

File này không sửa bài học sinh.

Nếu học sinh viết:

```text
3.00 + 2.00 = 5.00
```

thì `reported_expr` phải giữ:

```yaml
reported_expr: 3.00 + 2.00 = 5.00
```

kể cả đề bài đúng là `2.50`.

`expr` vẫn có thể map về entity đúng trong đề:

```yaml
expr: morning_coffee_price + afternoon_coffee_price
```

Sự khác nhau giữa `expr` và `reported_expr` là cơ sở để InsideChecker phát hiện
`misreading`, `logic error`, `wrong calculation`.

### Prompt Và Output

LLM trả YAML với hai key:

```yaml
StudentPlan.yaml:
  step1:
    expr: ...
    result: ...
    result_unit: ...
    result_grand_unit: ...
    reported_expr: ...
  target: ...
Diagnosis.yaml:
  - diagnosis: spelling errors
    step:
    entity:
```

Diagnosis ban đầu chỉ cho các lỗi biểu đạt như:

- `spelling errors`;
- `word problem`;
- `answer by word`.

### Validator

Code kiểm tra:

- plan phải là dict;
- step liên tục;
- mỗi step đủ 5 field;
- `expr` chỉ dùng entity có sẵn hoặc result trước;
- `result` hợp lệ;
- `reported_expr` phải có dấu `=`;
- số trong `reported_expr` phải grounded trong bài học sinh;
- `expr` chỉ dùng entity có sẵn hoặc result của step trước.

Các hàm đáng chú ý:

- `extract_equations_from_student_answer()`: lấy các phép tính regex nhận diện
  được trong text để đưa vào prompt như hint tham khảo.
- `validate_reported_expr_grounded_in_student_answer()`: chống LLM tự thêm số.
- `merge_student_plan_into_entities()`: thêm result entity vào
  `StudentAnswerEntities.yaml`.

### Write Diagnosis/Wrong

`write_diagnosis_and_wrong()` ghi diagnosis vào `Diagnosis.yaml` và cập nhật
`Wrong.yaml`. Các lỗi biểu đạt nhẹ thường ghi `Wrong: No`; lỗi toán học chính
sẽ được InsideChecker/CompareChecker thêm sau.

Diagnosis được merge, không ghi đè mất lỗi cũ.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc yêu cầu tạo StudentPlan và cập nhật entities. Code hiện tại siết thêm:

- không cho LLM tự sửa phép tính học sinh;
- yêu cầu LLM đọc toàn bộ file txt và giữ thứ tự suy luận;
- đưa equation regex vào prompt như hint mềm, không dùng exact-match code thuần
  để reject output;
- bắt số trong `reported_expr` phải xuất hiện trong bài học sinh;
- merge diagnosis thay vì stage sau ghi đè stage trước;
- chuẩn hóa `Diagnosis.yaml` thay vì typo `Diagonosis.yaml`.

## `Formalizer/TeacherAnswerFormalizer.py`

### Vai Trò

Formalize lời giải chuẩn của giáo viên trong `Input/TeacherAnswer.txt`.

Input:

- `Input/Problem.txt`
- `Input/TeacherAnswer.txt`
- `Output/ProblemEntities.yaml`

Output:

- `Output/TeacherPlan.yaml`
- `Output/TeacherAnswerEntities.yaml`

### Cách Hoạt Động

File này tái sử dụng nhiều helper từ `StudentAnswerFormalizer.py`:

- đọc YAML/text;
- parse output LLM;
- validate plan;
- extract equation;
- merge plan vào entities.

Khác với student formalizer, teacher formalizer yêu cầu:

- không tự tạo lời giải khác;
- giữ đúng thứ tự phép tính giáo viên viết;
- step cuối phải tạo target thật của đề bài;
- target trong `TeacherPlan.yaml` phải bằng target trong `ProblemEntities.yaml`.

Sau khi validate:

1. Ghi `TeacherPlan.yaml`.
2. Ghi `TeacherAnswerEntities.yaml`.

File này không ghi `Plan.yaml`/`PlanEntities.yaml` nữa. `Main/Grader.py` dùng
trực tiếp `TeacherPlan.yaml` và `TeacherAnswerEntities.yaml` làm reference khi
chấm theo lời giải giáo viên.

### Tại Sao Cần File Này

Ban đầu hệ thống chỉ có luồng tự giải bằng solver. Khi muốn so sánh với lời giải
chuẩn giáo viên, cần một luồng reference thứ hai.

`TeacherAnswerFormalizer.py` giúp benchmark/verifier dùng official solution làm
reference thay vì tin vào solver tự sinh.

### Khác Biệt So Với `Prompt.pdf`

File này không nằm trong prompt gốc. Nó được thêm để hỗ trợ luồng:

```text
ProblemFormalizer -> TeacherAnswerFormalizer -> StudentAnswerFormalizer -> Mapper -> CompareChecker
```

## `Formalizer/Mapper.py`

### Vai Trò

Map entity giữa:

- `Output/PlanEntities.yaml`
- `Output/StudentAnswerEntities.yaml`

Sau khi map, cả hai file entity được update thêm field:

```yaml
map: other_entity_name
```

### Nguyên Tắc Map

Prompt gốc yêu cầu:

1. Các entity chung ban đầu từ `ProblemEntities.yaml` auto-map.
2. Entity trung gian map theo quan hệ trong `expr`.
3. Không phụ thuộc cứng vào cùng step, vì học sinh có thể gộp/tách/đảo bước.

Code hiện tại giữ nguyên ý tưởng đó nhưng triển khai thành nhiều vòng:

- auto-map prefix đến target;
- map theo cùng result name;
- map theo signature của `expr`;
- map theo `formalized_expr`;
- suy luận ngược từ target expression nếu map xuôi chưa đủ.

### Expression Signature

Mapper dùng AST để canonicalize expression:

- `a + b` tương đương `b + a`;
- `a * b` tương đương `b * a`;
- tên entity student đã map sẽ được thay sang tên plan trước khi so sánh;
- metadata unit/grand_unit được check nhẹ để tránh map nhầm.

### Vì Sao Không Dùng LLM

Mapper là phần structural; nếu dùng LLM sẽ khó đảm bảo nhất quán và dễ tốn token.
Code deterministic giúp:

- reproducible;
- dễ debug;
- không phụ thuộc prompt;
- không làm mờ lỗi toán học của student.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc đã yêu cầu map bằng Python và map cả hai phía. Code hiện tại mở rộng:

- canonical AST cho expression;
- map bằng nhiều candidate signature;
- infer missing internal maps từ target expr;
- cho phép metadata unit mềm để student thiếu unit vẫn map được;
- ghi map vào cả PlanEntities và StudentAnswerEntities.
