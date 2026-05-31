# Benchmark Files

Nhóm `Benchmark/` dùng để chạy nhiều bài từ file CSV, đo pass/fail, ghi summary
và sinh dashboard lỗi. Các script benchmark không nằm trong thiết kế gốc
`Docs/Prompt.pdf`; chúng được thêm sau để kiểm tra backbone trên nhiều case.

## `Benchmark/RunSolveBenchmark.py`

### Vai Trò

Chạy solver reference trên nhiều bài GSM8K và so đáp án target với official
answer.

Pipeline mỗi bài:

```text
write Input/Problem.txt
-> python3 Main/Solver.py
-> đọc Output/PlanEntities.yaml
-> lấy entity có location: target
-> so value với official answer
```

### Input CSV

Mặc định đọc:

```text
Benchmark/GSM8K Benchmark.csv
```

Script tìm các cột:

- `question` hoặc `problem`;
- `offical answer` hoặc `official answer`;
- có fallback đọc final answer từ `offical response` nếu cột answer thiếu.

### Output

Mặc định hiện tại ưu tiên folder lỗi/dashboard hơn là CSV đầy đủ.

Các output thường gặp:

- summary JSON;
- dashboard HTML;
- thư mục error theo id;
- nếu bật `--write-csv` thì ghi thêm `results.csv` và `results_wrong.csv`.

Mỗi case lỗi có thể chứa snapshot:

- `ProblemEntities.yaml`
- `Code.txt`
- `Plan.yaml`
- `PlanEntities.yaml`
- stdout/stderr;
- chẩn đoán `Codex.txt` hoặc summary case.

### Các Metric

`compute_summary()` tính:

- tổng số bài;
- số bài đã chạy;
- số bài error;
- số bài đúng target;
- accuracy theo attempted rows;
- error stage counts.

So đáp án dùng Decimal và tolerance để tránh lỗi format số.

### Chạy Song Song

Nếu `--workers > 1`, script copy project sang workspace tạm cho mỗi bài. Lý do:

- pipeline chính đọc/ghi cố định `Input/` và `Output/`;
- nếu nhiều worker dùng chung root thì các file YAML sẽ ghi đè nhau;
- workspace tạm giúp mỗi bài có `Input/Output` riêng.

Log vẫn được in đúng thứ tự id, không theo thứ tự worker hoàn thành.

### Diagnose Case Sai

`diagnose_result_row()` đọc snapshot để đoán vì sao sai, ví dụ:

- formalize fail;
- không có target;
- target value rỗng;
- đáp án khác official answer;
- plan sai relation;
- executor/checker báo lỗi.

Đây là chẩn đoán heuristic để người đọc mở dashboard nhanh hơn, không phải nhãn
chính thức của pipeline.

### Khác Biệt So Với `Prompt.pdf`

File này không có trong prompt gốc. Nó được thêm để đánh giá solver reference
trên benchmark và gom lỗi vào dashboard thay vì phải mở từng file `Output/`.

## `Benchmark/RunVerifyBenchmark.py`

### Vai Trò

Chạy pipeline verify với reference là lời giải giáo viên. Script này dùng để đo
toàn bộ hệ thống chấm bài học sinh.

Pipeline mỗi bài:

```text
write Input/Problem.txt
write Input/TeacherAnswer.txt
write Input/StudentAnswer.txt
clear Output
python3 Main/Grader.py
read Diagnosis.yaml + Wrong.yaml
compare với cột type + wrong trong benchmark
```

### Vì Sao Dùng `Grader.py`

Luồng này không dùng solver tự sinh reference. Nó dùng official teacher answer
từ benchmark để tạo reference. Mục đích là đo verifier/grader thay vì đo chất
lượng solver.

Sau khi tách `Main/` thành `Tutor.py` và `Grader.py`, script này đã được sửa từ:

```text
Main/Main.py --reference teacher
```

thành:

```text
Main/Grader.py
```

### Input CSV

Đọc các cột:

- `question`;
- `offical response` hoặc `official response`;
- `student answer`;
- `type`;
- `wrong`.

`type` có thể chứa nhiều nhãn, được parse thành set label.

### Output Case

Khi mismatch hoặc pipeline error, script lưu thư mục theo id trong `ErrorVerify/`
hoặc error dir được truyền vào.

Mỗi folder case có thể chứa:

- `Diagnosis.yaml`
- `Wrong.yaml`
- `ProblemEntities.yaml`
- `TeacherPlan.yaml`
- `TeacherAnswerEntities.yaml`
- `StudentPlan.yaml`
- `StudentAnswerEntities.yaml`
- `Plan.yaml`
- `PlanEntities.yaml`
- `Summary.txt`

### Scoring

Mỗi row có:

- expected wrong;
- predicted wrong;
- expected labels;
- predicted labels;
- label true positives;
- label false positives;
- label false negatives;
- label score;
- exact match.

Exact match yêu cầu:

- wrong match;
- label set match hoàn toàn.

Ngoài exact match, script cũng tính metric riêng cho nhóm
`error_causing_labels`, gồm `unit missing` và `only final answer`, để đo khả năng
bắt đúng loại lỗi có thể làm bài bị xem là sai mà không bị nhiễu bởi các label
trình bày như `combine step` hoặc `reverse steps`.

### Chạy Song Song

Giống solve benchmark, nếu `--workers > 1`, script tạo workspace tạm để tránh
đụng `Input/Output`.

### CSV Và Dashboard

Theo thay đổi hiện tại, CSV đầy đủ không còn là output mặc định. Nếu cần CSV:

```bash
python3 Benchmark/RunVerifyBenchmark.py --write-csv
```

Mặc định script tập trung vào:

- summary;
- per-id folder;
- dashboard HTML.

### Khác Biệt So Với `Prompt.pdf`

File này không có trong prompt gốc. Nó được thêm để benchmark pipeline verify
theo official teacher solution.

## `Benchmark/RunBaseVerifyBenchmark.py`

### Vai Trò

Chạy model base trực tiếp để làm baseline verifier. Đây là luồng không dùng
pipeline symbolic.

Input cho model:

- original problem;
- official teacher solution;
- student solution.

Output model phải là YAML:

```yaml
BaseVerifier.yaml:
  wrong: Yes
  diagnosis:
    - diagnosis: wrong relationship
      step:
      entity:
  reason: ...
```

### Mục Đích

Script này trả lời câu hỏi:

> Nếu không formalize entity/plan, chỉ đưa text cho model base tự chấm thì kết
> quả thế nào?

Nó là baseline để so với pipeline symbolic.

### Prompt Và Label Rubric

Script có whitelist label:

| Label | Ý nghĩa |
|---|---|
| `all right` | Lời giải đúng, không có lỗi đáng kể. |
| `answer by word` | Học sinh trả lời chủ yếu bằng chữ, không viết phép tính rõ ràng; thường không sai nếu kết quả đúng. |
| `combine step` | Học sinh gộp nhiều bước chuẩn thành một bước hợp lệ. |
| `different calculation` | Học sinh dùng cách tính khác nhưng vẫn đúng. |
| `do not convert units` | Cần đổi đơn vị nhưng học sinh không đổi. |
| `extra step` | Có bước thừa nhưng không nhất thiết làm sai kết quả. |
| `logic error` | Lỗi suy luận tổng quát, không chỉ là tính nhầm hay đọc sai số. |
| `misreading` | Đọc sai dữ kiện/điều kiện trong đề. |
| `missing step` | Thiếu bước cần thiết làm lời giải không đủ hoặc sai. |
| `only final answer` | Chỉ đưa đáp án cuối, không có hoặc gần như không có lập luận. |
| `reverse steps` | Làm các bước đúng nhưng khác thứ tự lời giải chuẩn. |
| `spelling errors` | Lỗi chính tả/diễn đạt không làm đổi toán học. |
| `step separation` | Tách một bước chuẩn thành nhiều bước nhỏ hợp lệ. |
| `unit missing` | Thiếu đơn vị trong lời giải hoặc kết quả. |
| `word problem` | Diễn giải bằng lời/tối nghĩa, khó map thành bước tính rõ ràng. |
| `wrong calculation` | Công thức đúng nhưng tính số sai. |
| `wrong relationship` | Dùng sai quan hệ/toán tử/công thức. |
| `wrong target` | Giải ra đại lượng khác với thứ đề hỏi. |
| `wrong unit conversion` | Có đổi đơn vị nhưng dùng sai hệ số hoặc sai chiều đổi. |

`LABEL_RUBRIC` giải thích từng nhãn để model không chỉ đoán label theo tên.

### Normalize Output

Code normalize:

- label typo, ví dụ `wrong caculation` -> `wrong calculation`;
- `diffirent caculation` -> `different calculation`;
- `wrong units conversion` -> `wrong unit conversion`;
- `wrong` thành `yes`/`no`.

Nếu model trả label ngoài whitelist, row bị error/retry.

### Metric

Summary hiện tại chỉ báo cáo 5 metric chính để tránh nhiễu:

| Metric | Ý nghĩa |
|---|---|
| `wrong_accuracy` | Tỷ lệ dự đoán đúng bài sai hay không sai (`wrong=yes/no`). Đây là metric dễ hiểu nhất. |
| `wrong_f1` | F1 cho lớp `wrong=yes`, cân bằng giữa bỏ sót bài sai và báo sai nhầm bài đúng. |
| `error_label_hit_rate` | Trong các bài có label lỗi chính, tỷ lệ bài bắt được ít nhất một label lỗi đúng. |
| `error_label_f1` | F1 cho riêng nhóm label gây lỗi. Metric này đo chất lượng phân loại loại lỗi chính. |
| `exact_match` | Tỷ lệ khớp hoàn toàn cả `wrong` và toàn bộ set label. Đây là metric nghiêm ngặt để tham khảo. |

Các field như `completed_rows`, `attempted_rows`, `error_rows`,
`error_stage_counts`, `support` chỉ là metadata/debug để biết benchmark chạy đủ
không và số dòng hỗ trợ phía sau metric. Chúng không được xem là metric báo cáo
chính.

`support` hiện chứa các count để giải thích metric:

| Field | Ý nghĩa |
|---|---|
| `exact_match_rows` | Số bài khớp hoàn toàn cả `wrong` và label set. |
| `wrong_match_rows` | Số bài dự đoán đúng `wrong=yes/no`. |
| `wrong_tp` | Expected `wrong=yes`, predicted cũng `yes`. |
| `wrong_fp` | Expected `wrong=no`, predicted `yes`. |
| `wrong_tn` | Expected `wrong=no`, predicted cũng `no`. |
| `wrong_fn` | Expected `wrong=yes`, predicted `no`. |
| `error_label_expected_rows` | Số bài ground truth có ít nhất một label trong nhóm lỗi chính. |
| `error_label_partial_match_rows` | Số bài bắt được ít nhất một label lỗi chính đúng. |
| `error_label_tp` | Tổng số error-causing label bắt đúng. |
| `error_label_fp` | Tổng số error-causing label bị dự đoán thừa. |
| `error_label_fn` | Tổng số error-causing label bị bỏ sót. |

Lý do chỉ giữ 5 metric: exact label set quá nghiêm ngặt, còn toàn bộ taxonomy
label có nhiều nhãn trình bày như `combine step`, `reverse steps`,
`spelling errors`. Bộ 5 metric trên trả lời đủ các câu hỏi chính:

- hệ thống có biết bài sai hay đúng không;
- hệ thống có bắt được bài sai không;
- hệ thống có bắt được ít nhất một lỗi chính không;
- hệ thống phân loại lỗi chính tốt không;
- hệ thống có khớp hoàn toàn benchmark không.

Nhóm `error_causing_labels` dùng để đo riêng các nhãn có thể làm bài bị xem là
sai hoặc không đủ điều kiện chấm đúng:

| Label | Vì sao nằm trong nhóm lỗi |
|---|---|
| `do not convert units` | Không đổi đơn vị khi đề bắt buộc đổi, dễ làm phép tính sai đại lượng. |
| `logic error` | Suy luận sai nên ảnh hưởng trực tiếp đến lời giải. |
| `misreading` | Dùng sai dữ kiện đề bài, thường kéo toàn bộ lời giải sai. |
| `missing step` | Thiếu bước cần thiết nên lời giải không đủ hoặc không thể verify đúng. |
| `only final answer` | Không có lập luận/phép tính để kiểm chứng, có thể bị xem là không đủ lời giải. |
| `unit missing` | Thiếu đơn vị; trong benchmark này được tính vào nhóm lỗi cần bắt. |
| `wrong calculation` | Quan hệ đúng nhưng tính số sai, làm đáp án sai. |
| `wrong relationship` | Công thức/toán tử sai, là lỗi toán học cốt lõi. |
| `wrong target` | Tính ra đại lượng khác với câu hỏi. |
| `wrong unit conversion` | Có đổi đơn vị nhưng đổi sai hệ số hoặc sai chiều. |

Metric nhóm lỗi nằm giữa `wrong_accuracy` và `exact_match`: nó chi tiết hơn việc
chỉ biết bài sai hay đúng, nhưng không rộng và nhiễu như toàn bộ taxonomy label.

### Rebuild Report Không Gọi LLM

Có thể rebuild summary/dashboard từ folder error hiện có:

```bash
python3 Benchmark/RunBaseVerifyBenchmark.py --rebuild-report-only
```

Flag này không gọi API, không tốn token. Nó đọc lại các `Summary.txt` trong
`ErrorBaseVerify/<id>/` và tái dựng các row exact match còn lại từ CSV.

### Output

Mặc định không ghi CSV, trừ khi có `--write-csv`.

Output chính:

- `ErrorBaseVerify/results_summary.json`
- `ErrorBaseVerify/Summary.md`
- `ErrorBaseVerify/index.html`
- folder mismatch/error theo id.

### Khác Biệt So Với `Prompt.pdf`

File này không có trong prompt gốc. Nó được thêm để có baseline LLM trực tiếp.
Nó cũng giúp kiểm tra giả thuyết: formalize symbolic có thật sự tốt hơn việc
đưa text thuần cho model hay không.

## Ghi Chú Chung Về Benchmark

### Vì Sao Có Workspace Tạm

Pipeline chính không được thiết kế stateless. Nó đọc/ghi file cố định:

```text
Input/*.txt
Output/*.yaml
```

Do đó benchmark song song phải copy project sang workspace tạm để mỗi worker
không ghi đè file của worker khác.

### Vì Sao Không Ghi CSV Mặc Định

Khi đã có folder theo id và dashboard HTML, CSV đầy đủ thường làm thư mục rối và
dễ bị git track nhầm. Vì vậy các runner hiện ưu tiên:

- summary;
- dashboard;
- per-case debug files.

CSV vẫn bật được bằng `--write-csv` khi cần phân tích bằng spreadsheet.

### Vì Sao Cần Dashboard HTML

Mỗi bài lỗi có nhiều file YAML. Dashboard giúp:

- lọc theo stage;
- xem expected/predicted nhanh;
- click từng file trong case;
- tránh phải mở thủ công hàng chục folder.
