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
python3 Main/Grader.py --reference teacher
read Diagnosis.yaml + Wrong.yaml
compare với cột type + wrong trong benchmark
```

### Vì Sao Dùng `Grader.py --reference teacher`

Luồng này không dùng solver tự sinh reference. Nó dùng official teacher answer
từ benchmark để tạo reference. Mục đích là đo verifier/grader thay vì đo chất
lượng solver.

Sau khi tách `Main/` thành `Tutor.py` và `Grader.py`, script này đã được sửa từ:

```text
Main/Main.py --reference teacher
```

thành:

```text
Main/Grader.py --reference teacher
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

- `all right`
- `answer by word`
- `combine step`
- `different calculation`
- `do not convert units`
- `extra step`
- `logic error`
- `misreading`
- `missing step`
- `only final answer`
- `reverse steps`
- `spelling errors`
- `step separation`
- `unit missing`
- `word problem`
- `wrong calculation`
- `wrong relationship`
- `wrong target`
- `wrong unit conversion`

`LABEL_RUBRIC` giải thích từng nhãn để model không chỉ đoán label theo tên.

### Normalize Output

Code normalize:

- label typo, ví dụ `wrong caculation` -> `wrong calculation`;
- `diffirent caculation` -> `different calculation`;
- `wrong units conversion` -> `wrong unit conversion`;
- `wrong` thành `yes`/`no`.

Nếu model trả label ngoài whitelist, row bị error/retry.

### Metric

Ngoài exact match, script hiện có thêm metric mềm:

- `label_partial_match_rows`;
- `label_precision_micro`;
- `label_recall_micro`;
- `label_f1_micro`;
- `wrong_accuracy_attempted`;
- `wrong_yes_precision`;
- `wrong_yes_recall`;
- `wrong_yes_f1`;
- label distribution:
  - expected label counts;
  - predicted label counts;
  - true positive counts;
  - false positive counts;
  - false negative counts.

Lý do thêm metric mềm: exact label set quá nghiêm ngặt. Model có thể phát hiện
đúng bài sai (`wrong=yes`) nhưng gắn nhãn chi tiết không khớp benchmark.

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

