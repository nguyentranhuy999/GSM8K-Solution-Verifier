# Changes Compared With `docs/Prompt.pdf`

File `docs/Prompt.pdf` mô tả backbone ban đầu cho các module:
`ProblemFormalizer`, `Planner`, `Executor`, `StudentAnswerFormalizer`,
`Mapper`, `InsideChecker`, `CompareChecker` và `LLMChecker`.

Các thay đổi dưới đây là phần code hiện tại đã mở rộng hoặc chỉnh khác so với
prompt ban đầu, chủ yếu để chạy benchmark ổn định hơn và giảm lỗi do LLM sinh
YAML/plan sai.

## Thay đổi chung

- Thêm `OPENROUTER_MAX_TOKENS` cho các lần gọi OpenRouter. Default hiện tại là
  `4096` ở `ProblemFormalizer`, `Planner`, `Executor` và
  `StudentAnswerFormalizer`.
- Tăng số lần retry ở một số module LLM, ví dụ `ProblemFormalizer` và `Planner`
  đang default `5` retries.
- Thêm nhiều validator bằng Python trước/sau khi nhận output LLM. Prompt ban đầu
  chủ yếu dựa vào LLM làm đúng schema; code hiện tại reject output sai rồi gọi
  lại LLM bằng lỗi cụ thể.
- Chuẩn hóa spelling file diagnosis về `Output/Diagnosis.yaml`. Prompt PDF có
  chỗ ghi nhầm `Diagonosis.yaml`.

## `Formalizer/ProblemFormalizer.py`

So với prompt ban đầu, module này không chỉ gọi LLM rồi ghi
`ProblemEntities.yaml`, mà còn có lớp normalize/validator sau LLM.

Các thay đổi chính:

- Validate schema entity: đúng 4 trường `value`, `unit`, `location`,
  `grand_unit`; chỉ chấp nhận `input` hoặc `target`; target phải có value rỗng.
- Parse/validate số trực tiếp trong đề bằng code:
  - số digit, decimal dạng `.50`;
  - phân số như `1/3`, mixed fraction như `3 3/8`;
  - số viết bằng chữ như `two`, `twenty`, `one hundred`;
  - percent viết bằng `%`, `percent`, hoặc chữ;
  - multiplier words như `twice`, `double`, `triple`, `twins`.
- Reject entity input có value không được nói trực tiếp trong đề, trừ các helper
  constant do code tự thêm.
- Tự sửa một số lỗi LLM thường gặp:
  - bỏ complement fraction tự tính như `girls_fraction = 0.6` khi đề chỉ cho
    `2/5 are boys`;
  - bỏ placeholder `age/count = 0` nếu đề không hề nói số 0;
  - sửa entity bị LLM đổi đơn vị sớm, ví dụ đề nói `12000 meters` nhưng LLM đổi
    thành `12 kilometers`.
- Thêm helper entity không có trong prompt PDF ban đầu:
  - `unit_conversion_*`, ví dụ `unit_conversion_meters_per_kilometer`,
    `unit_conversion_days_per_year`, `unit_conversion_items_per_pair`;
  - `identity_multiplier = 1` cho các bài cần hệ số 1 symbolic;
  - `percentage_scale = 100` khi target hỏi percent/percentage;
  - `host_count = 1`, `split_count = 2`;
  - family context counts như `self_count`, `sibling_count`, `parents_count`,
    `grandparents_count` khi đề có dạng `family consists of ...`.
- Thêm validator target name cho một số câu hỏi dễ formalize nhầm, ví dụ:
  - bài hỏi `friends can invite` thì target phải là friends, không phải cost;
  - bài hỏi `not in/not on/not at` thì target nên chứa `not`;
  - bài hỏi `give ... each` thì target phải thể hiện per/every/each.
- Normalize unit/grand_unit thêm:
  - target dạng pair/items;
  - relation `_more_than`, `_less_than`, `_fewer_than`.

Điểm khác quan trọng: prompt ban đầu nói `ProblemEntities` chỉ gồm số trực tiếp
trong đề và target. Code hiện tại vẫn giữ nguyên nguyên tắc đó cho entity do LLM
sinh ra, nhưng có thêm helper constants bằng Python để Planner không phải dùng
số literal trong `expr`.

## `Formalizer/Solver/Planner.py`

So với prompt ban đầu, Planner hiện tại không chỉ gọi LLM sinh plan, mà có thêm
validator mạnh và deterministic fallback cho một số schema toán học phổ biến.

Các thay đổi chính:

- Enforce `expr` chỉ dùng entity/result symbolic, không được có số literal như
  `1`, `2`, `0.5`, `36`.
- Hỗ trợ alias `grand_result_unit` từ prompt cũ nhưng normalize về
  `result_grand_unit`.
- Validate step liên tục `step1`, `step2`, ... và bước cuối phải tạo đúng target.
- Giữ `location: target` của target trong `PlanEntities.yaml`; không ghi đè thành
  `stepN`.
- Thêm nhiều validator logic/backbone:
  - không double-count giá item đã liệt kê;
  - scalar quan trọng phải được dùng;
  - rate theo ngày/tuần/tháng phải đổi đúng horizon;
  - relation `x_more_than_y`, `x_fewer_than_y` phải được resolve trước khi dùng;
  - percentage target phải nhân `percentage_scale`;
  - discount threshold không được dùng như quantity thật;
  - roommate split phải tính cả người chủ khi phù hợp;
  - invite friends không được trừ host hai lần;
  - family ticket discount không được áp discount cho cả parents/grandparents;
  - herd/calves sau khi sinh phải cộng cả adult animals ban đầu;
  - bill/coin change phải đổi qua dollar amount rồi chia denomination;
  - allocation fraction không được copy lượng nhóm A sang nhóm B nếu vượt tổng.
- Thêm deterministic planner cho các pattern mà LLM hay dùng target trước khi
  target được tạo:
  - `give to each ... same amount`;
  - linear shares như first/second/third với `more than` và multiplier;
  - sales có price multiplier + quantity multiplier + total earnings;
  - đổi mệnh giá bills/coins;
  - phân bổ fraction còn lại cho target group.

Điểm khác quan trọng: prompt ban đầu để Planner hoàn toàn do LLM sinh. Code hiện
tại ưu tiên deterministic plan khi nhận ra schema; nếu không match thì mới gọi
LLM.

## `Formalizer/Solver/Executor.py`

Executor vẫn làm đúng core trong PDF: đọc `Plan.yaml`, tính `reported_expr`,
update `PlanEntities.yaml`, thêm `expr` và `formalized_expr`.

Các phần đã mở rộng:

- Dùng evaluator an toàn cho biểu thức arithmetic thay vì `eval` tùy ý.
- Tự gọi `Verifier/InsideChecker.py --mode llm`.
- Nếu InsideChecker báo lỗi, gọi LLM repair plan/entities rồi chạy lại.
- Nếu `Output/Error.yaml` chỉ có lỗi `extra step`, Executor coi là pass để tránh
  rollback vô ích.
- Repair LLM cũng dùng `OPENROUTER_MAX_TOKENS`.

## `Formalizer/StudentAnswerFormalizer.py`

So với prompt PDF, module này đã được siết để không “làm mượt” bài làm học sinh
quá mức.

Các thay đổi chính:

- `reported_expr` phải giữ đúng phép tính học sinh viết hoặc ngụ ý trong từng
  dòng, không thay bằng phép tính tương đương.
- Validate số trong `reported_expr` phải xuất hiện trong bài làm học sinh.
- Extract các equation học sinh viết và bắt StudentPlan giữ đúng thứ tự, không
  gộp/bỏ bước.
- Vẫn map `expr` về entity chuẩn của đề, nhưng `reported_expr` giữ số học sinh
  dùng sai nếu học sinh đọc sai đề.
- Ghi diagnosis vào `Output/Diagnosis.yaml` và `Wrong.yaml`.

## `Main/Solver.py` và `Main/Main.py`

Hai file pipeline này không có trong prompt PDF ban đầu dưới dạng file riêng.

- `Main/Solver.py` chạy pipeline lời giải chuẩn:
  `ProblemFormalizer -> Planner -> Executor`.
- `Main/Solver.py` không gọi InsideChecker riêng vì Executor đã gọi và repair.
- `Main/Main.py` chạy pipeline đầy đủ:
  `Solver -> StudentAnswerFormalizer -> InsideChecker --mode student -> Mapper -> CompareChecker`.

## `Benchmark/RunSolveBenchmark.py`

File benchmark không nằm trong prompt PDF ban đầu.

Các chức năng đã thêm:

- Chạy `Main/Solver.py` trên benchmark GSM8K.
- So sánh target value với cột `offical answer`/`official answer`.
- Ghi CSV kết quả, wrong CSV và summary JSON.
- Ghi snapshot debug:
  - `problem_entities_yaml`;
  - `plan_yaml`;
  - `plan_entities_yaml`;
  - stdout/stderr;
  - stage lỗi.
- Hỗ trợ:
  - `--limit`;
  - `--workers`;
  - `--indices`;
  - `--resume`;
  - `--timeout`;
  - `--tolerance`.
- Với `--workers > 1`, mỗi bài chạy trong workspace tạm riêng để tránh ghi đè
  `Input/` và `Output/`.
- Có logic đọc final answer từ `offical response` để xử lý một số row CSV có cột
  answer bị thiếu digit.

## `README.md`

`README.md` cũng là phần bổ sung ngoài prompt PDF.

README hiện mô tả:

- cấu trúc thư mục;
- cách tạo `.env`;
- cách chạy Solver pipeline;
- cách chạy full pipeline;
- format YAML entity/plan;
- cách chạy benchmark;
- các lỗi thường gặp và lưu ý phát triển.

## Những điểm vẫn giữ nguyên theo prompt ban đầu

- Entity vẫn xoay quanh các trường chính: `value`, `unit`, `location`,
  `grand_unit`.
- Plan vẫn gồm `expr`, `result`, `result_unit`, `result_grand_unit`.
- Executor vẫn là nơi tính numeric value chính, không để Planner tự tính value
  target.
- StudentAnswerFormalizer vẫn formalize lời giải học sinh thành
  `StudentPlan.yaml` và update `StudentAnswerEntities.yaml`.
- Mapper và Checker vẫn chạy bằng Python, không dùng LLM cho mapping/checking.

## Ghi chú hiện tại

- Code hiện tại có xu hướng “hybrid”: LLM dùng để hiểu ngôn ngữ, còn Python
  validator/deterministic planner dùng để khóa schema và bắt lỗi backbone.
- Một số helper entity như `identity_multiplier`, `unit_conversion_*`,
  `percentage_scale` là khác biệt có chủ ý so với prompt ban đầu. Chúng giúp
  giữ `expr` chỉ gồm biến, không phải số literal.
- Khi chạy benchmark, OpenRouter có thể trả lỗi credit/token. Code vẫn để default
  `OPENROUTER_MAX_TOKENS = 4096`; có thể override tạm bằng biến môi trường nếu
  cần chạy ít token hơn.
