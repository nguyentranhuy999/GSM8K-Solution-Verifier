# Project Structure

Tài liệu này mô tả cấu trúc hiện tại của dự án `GSM8K-Solution-Verifier`: mỗi thư mục làm gì, các pipeline chính chạy theo thứ tự nào, file YAML nào là input/output của từng stage, và ranh giới trách nhiệm giữa các module.

## Mục Tiêu Kiến Trúc

Dự án tách bài toán kiểm tra lời giải GSM8K thành nhiều stage nhỏ:

1. Formalize đề bài thành danh sách thực thể số.
2. Formalize hoặc tự sinh lời giải thành plan symbolic.
3. Execute/validate plan bằng code Python.
4. Formalize lời giải học sinh hoặc giáo viên thành plan riêng.
5. Map entity giữa hai lời giải.
6. So sánh quan hệ tính toán và ghi diagnosis.

Nguyên tắc chính là mỗi file chỉ nên làm phần việc của nó:

- `Formalizer/*` chuyển text hoặc YAML thô thành cấu trúc symbolic.
- `Solver/*` tạo và thực thi lời giải chuẩn do hệ thống tự sinh.
- `Verifier/*` kiểm tra lỗi nội tại hoặc so sánh hai lời giải.
- `Main/*` chỉ nối các stage thành pipeline.
- `Benchmark/*` chỉ chạy nhiều case và lưu artifact debug.

## Cây Thư Mục Tổng Quan

```text
.
├── Benchmark/
│   ├── GSM8K Benchmark.csv
│   ├── RunSolveBenchmark.py
│   ├── RunBaseSolveBenchmark.py
│   ├── RunVerifyBenchmark.py
│   └── RunBaseVerifyBenchmark.py
├── Docs/
│   ├── Prompt.pdf
│   ├── Benchmark.pdf
│   ├── Changes.md
│   ├── OverView.md
│   ├── Overview.html
│   └── PythonFiles/
├── Formalizer/
│   ├── ProblemFormalizer.py
│   ├── StudentAnswerFormalizer.py
│   ├── TeacherAnswerFormalizer.py
│   ├── Mapper.py
│   └── Solver/
│       ├── Planner.py
│       └── Executor.py
├── Input/
│   ├── Problem.txt
│   ├── StudentAnswer.txt
│   └── TeacherAnswer.txt
├── Main/
│   ├── Solver.py
│   ├── Tutor.py
│   └── Grader.py
├── Output/
│   ├── ProblemEntities.yaml
│   ├── Code.txt
│   ├── Plan.yaml
│   ├── PlanEntities.yaml
│   ├── StudentPlan.yaml
│   ├── StudentAnswerEntities.yaml
│   ├── TeacherPlan.yaml
│   ├── TeacherAnswerEntities.yaml
│   ├── Diagnosis.yaml
│   ├── Wrong.yaml
│   ├── Error.yaml
│   ├── LLMChecker.yaml
│   └── Log.yaml
├── Verifier/
│   ├── InsideChecker.py
│   ├── CompareChecker.py
│   └── LLMChecker.py
├── Error/
├── ErrorVerify/
├── ErrorBaseVerify/
├── README.md
├── Structure.md
├── requirements.txt
└── .env
```

`Error/`, `ErrorVerify/`, `ErrorBaseVerify/` là artifact benchmark/debug. Các folder này đã được đưa vào `.gitignore`.

## Input

`Input/Problem.txt`

Đề bài hiện tại cần giải hoặc cần chấm.

`Input/StudentAnswer.txt`

Lời giải học sinh hiện tại. Dùng bởi `StudentAnswerFormalizer.py`, `Tutor.py`, `Grader.py`, `RunVerifyBenchmark.py`.

`Input/TeacherAnswer.txt`

Lời giải chuẩn của giáo viên. Dùng trong luồng `Grader.py` và fallback `LLMChecker.py`.

## Output

`Output/ProblemEntities.yaml`

Danh sách entity được trích từ đề bài. Đây là output của `ProblemFormalizer.py`.

Entity có các trường chính:

```yaml
some_entity:
  value: 30
  unit: items
  location: input
  grand_unit: items
  source: "optional text evidence"
```

Quy ước `location`:

- `input`: dữ kiện trực tiếp trong đề.
- `target`: đại lượng cần tìm.
- `step1`, `step2`, ...: thực thể được tạo ra bởi một bước tính.

`Output/Code.txt`

Pseudo-code do `Formalizer/Solver/Planner.py` sinh ra khi hệ thống tự giải bài. File này giúp debug xem LLM đã hiểu lời giải toán như thế nào trước khi map sang `Plan.yaml`.

`Output/Plan.yaml`

Plan symbolic của lời giải chuẩn do Solver tự sinh. File này thuộc luồng `Solver.py`/`Tutor.py`.

`Output/PlanEntities.yaml`

Entity của lời giải chuẩn do Solver tự sinh. Ban đầu được copy từ `ProblemEntities.yaml`, sau đó `Executor.py` thêm value, `expr`, `formalized_expr` cho các result entity.

`Output/StudentPlan.yaml`

Plan symbolic của lời giải học sinh, do `StudentAnswerFormalizer.py` tạo.

`Output/StudentAnswerEntities.yaml`

Entity của lời giải học sinh. Ban đầu có thể được copy từ `ProblemEntities.yaml`, sau đó `StudentAnswerFormalizer.py` thêm các result entity từ `StudentPlan.yaml`.

`Output/TeacherPlan.yaml`

Plan symbolic của lời giải giáo viên, do `TeacherAnswerFormalizer.py` tạo.

`Output/TeacherAnswerEntities.yaml`

Entity của lời giải giáo viên. Dùng làm reference trong luồng `Grader.py`.

`Output/Diagnosis.yaml`

Danh sách lỗi hoặc nhãn kết luận. File này được append/merge bởi nhiều checker, không nên bị ghi đè mất lỗi cũ.

Ví dụ:

```yaml
- diagnosis: wrong calculation
  step: step2
  entity: total_cost
```

`Output/Wrong.yaml`

Kết luận bài học sinh sai hay không:

- `Yes`: có lỗi làm sai kết quả hoặc sai quan hệ cần chấm.
- `No`: không sai, hoặc chỉ có lỗi trình bày/khác cách làm.

`Output/Error.yaml`

Lỗi nội tại của plan solver/reference trong luồng `Executor.py` và `InsideChecker.py --mode llm`. File này chủ yếu phục vụ repair solver.

`Output/LLMChecker.yaml`

Log/debug chi tiết nếu chạy `Verifier/LLMChecker.py`. Fallback này không nằm trong pipeline mặc định của `Tutor.py` hoặc `Grader.py` hiện tại.

`Output/Log.yaml`

Trạng thái pass/fail của từng stage. Đây là nơi nên ghi warning hoặc lỗi không fatal để debug.

## Pipeline Chính

### Solver

Chạy:

```bash
python3 Main/Solver.py
```

Mục tiêu: tự giải đề bài và tạo lời giải chuẩn symbolic.

Thứ tự:

```text
ProblemFormalizer
-> Planner
-> Executor
```

Luồng file:

```text
Input/Problem.txt
-> Output/ProblemEntities.yaml
-> Output/Code.txt
-> Output/Plan.yaml
-> Output/PlanEntities.yaml
```

Chi tiết:

- `ProblemFormalizer.py` đọc đề bài, trích entity input và target.
- `Planner.py` sinh pseudo-code trước, sau đó map literal về entity để tạo `Plan.yaml`.
- `Executor.py` thực thi `Plan.yaml`, điền value vào `PlanEntities.yaml`, rồi gọi `InsideChecker.py`.
- Nếu `InsideChecker.py` phát hiện lỗi trong solver plan, `Executor.py` có cơ chế repair bằng LLM.

`Solver.py` không chấm lời giải học sinh.

### Tutor

Chạy:

```bash
python3 Main/Tutor.py
```

Mục tiêu: hệ thống tự giải bài, rồi dùng lời giải tự sinh để chấm lời giải học sinh.

Thứ tự:

```text
Solver
-> StudentAnswerFormalizer
-> InsideChecker --mode student
-> Mapper
-> CompareChecker
```

Luồng reference:

```text
Output/Plan.yaml
Output/PlanEntities.yaml
```

Luồng học sinh:

```text
Input/StudentAnswer.txt
-> Output/StudentPlan.yaml
-> Output/StudentAnswerEntities.yaml
```

So sánh:

```text
PlanEntities.yaml <-> StudentAnswerEntities.yaml
-> Diagnosis.yaml
-> Wrong.yaml
```

`Tutor.py` phù hợp khi không có lời giải giáo viên và muốn hệ thống tự tạo reference.

### Grader

Chạy:

```bash
python3 Main/Grader.py
```

Mục tiêu: chấm lời giải học sinh bằng lời giải giáo viên có sẵn.

Thứ tự hiện tại:

```text
ProblemFormalizer --copy-targets grader
-> StudentAnswerFormalizer
-> TeacherAnswerFormalizer
-> InsideChecker --mode teacher
-> InsideChecker --mode student
-> Mapper --reference teacher
-> CompareChecker --reference teacher
```

Luồng file:

```text
Input/Problem.txt
Input/StudentAnswer.txt
Input/TeacherAnswer.txt

-> Output/ProblemEntities.yaml
-> Output/StudentPlan.yaml
-> Output/StudentAnswerEntities.yaml
-> Output/TeacherPlan.yaml
-> Output/TeacherAnswerEntities.yaml
-> Output/Diagnosis.yaml
-> Output/Wrong.yaml
```

Điểm quan trọng:

- `Grader.py` là pipeline riêng với `Tutor.py`.
- `Grader.py` không cần Solver làm reference.
- Reference của `Grader.py` là `TeacherAnswerEntities.yaml`, không phải `PlanEntities.yaml`.
- `Mapper.py --reference teacher` map student entity với teacher entity.
- `CompareChecker.py --reference teacher` so sánh student với teacher.

## Formalizer

### ProblemFormalizer.py

Trách nhiệm:

- Đọc `Input/Problem.txt`.
- Gọi LLM để trích entity có số xuất hiện trực tiếp trong đề.
- Không tính sẵn các entity trung gian nếu đề không cho trực tiếp.
- Tạo đúng target entity với `location: target`.
- Thêm một số entity mặc định hoặc chuẩn hóa hữu ích, ví dụ unit conversion, multiplier words, fraction words.
- Ghi `Output/ProblemEntities.yaml`.
- Tùy flag `--copy-targets`, copy entity sang các file entity khác.

Không nên đưa logic giải bài vào file này. Nếu `ProblemFormalizer.py` bắt đầu tính target hoặc tính trung gian quá nhiều, downstream sẽ khó phân biệt dữ kiện đề bài và kết quả suy luận.

### StudentAnswerFormalizer.py

Trách nhiệm:

- Đọc `Input/StudentAnswer.txt`.
- Đọc `Output/ProblemEntities.yaml`.
- Gọi LLM để tạo `Output/StudentPlan.yaml`.
- Mỗi step phản ánh đúng phép tính học sinh viết hoặc ngụ ý.
- Không sửa lỗi tính toán của học sinh.
- `reported_expr` giữ phép tính số học học sinh báo cáo.
- `expr` là biểu thức symbolic bằng entity.
- Materialize numeric literal phát sinh trong lời giải thành entity cục bộ, ví dụ `student_answer_number_2`.
- Merge plan vào `Output/StudentAnswerEntities.yaml`.

File này có validator để reject các cấu trúc dễ gây ảo giác, ví dụ:

- Step không liên tục.
- `expr` dùng entity chưa tồn tại.
- Step copy/tautology kiểu `110 = 110`.
- `expr` không đúng schema.

### TeacherAnswerFormalizer.py

Trách nhiệm tương tự `StudentAnswerFormalizer.py`, nhưng input là `Input/TeacherAnswer.txt`.

Output:

- `Output/TeacherPlan.yaml`
- `Output/TeacherAnswerEntities.yaml`

Điểm khác biệt:

- Teacher answer là reference trong `Grader.py`.
- Target của `TeacherPlan.yaml` phải là target của đề bài.
- Step cuối của teacher plan phải tạo target.

### Mapper.py

Trách nhiệm:

- Đọc `StudentAnswerEntities.yaml`.
- Đọc reference entity:
  - mặc định: `PlanEntities.yaml`
  - nếu `--reference teacher`: `TeacherAnswerEntities.yaml`
- Thêm trường `map` vào cả hai phía.
- Không dùng LLM.

Ý tưởng:

- Entity input/target ban đầu được auto-map theo tên.
- Entity trung gian được map bằng `expr` hoặc `formalized_expr`.
- Mapper cố gắng map cả forward lẫn backward để xử lý trường hợp học sinh đổi thứ tự bước.
- Nếu không map được thì để `map: null`.

## Solver Internals

### Planner.py

Trách nhiệm:

- Đọc `Input/Problem.txt` và `Output/ProblemEntities.yaml`.
- Gọi LLM để sinh pseudo-code vào `Output/Code.txt`.
- Dùng code Python map số literal trong pseudo-code về entity.
- Sinh `Output/Plan.yaml`.
- Khởi tạo/cập nhật `Output/PlanEntities.yaml`.

Thiết kế này giúp LLM tập trung giải bài trước, thay vì bị nhiễu bởi tên entity dài. Việc biến lời giải thành plan symbolic được code kiểm soát nhiều hơn.

### Executor.py

Trách nhiệm:

- Đọc `Output/Plan.yaml` và `Output/PlanEntities.yaml`.
- Thực thi các step bằng evaluator an toàn.
- Điền `reported_expr`, `value`, `expr`, `formalized_expr`.
- Gọi `InsideChecker.py` để kiểm tra plan solver.
- Nếu có `Error.yaml`, gọi LLM repair plan và lặp lại.

Executor là nơi tính toán thật của solver. Planner không nên tự điền value cuối cùng nếu value đó chưa được execute.

## Verifier

### InsideChecker.py

Trách nhiệm:

- Kiểm tra lỗi nội tại trong một plan/entity pair.
- Không so sánh hai lời giải với nhau.

Mode:

```bash
python3 Verifier/InsideChecker.py --mode llm
python3 Verifier/InsideChecker.py --mode student
python3 Verifier/InsideChecker.py --mode teacher
```

Input theo mode:

- `llm`: `Plan.yaml` + `PlanEntities.yaml`
- `student`: `StudentPlan.yaml` + `StudentAnswerEntities.yaml`
- `teacher`: `TeacherPlan.yaml` + `TeacherAnswerEntities.yaml`

Các nhóm lỗi có thể phát hiện:

- `wrong target`
- `wrong calculation`
- `unit missing`
- `only final answer`
- `wrong relationship`
- `do not convert units`
- `missing step`
- `misreading`
- `logic error`
- `double count`
- `extra step`

Trong teacher mode, lỗi reference nên được ghi vào `Log.yaml` để debug, không nên làm mất diagnosis của học sinh.

### CompareChecker.py

Trách nhiệm:

- So sánh reference với lời giải học sinh sau khi `Mapper.py` đã map entity.
- Không dùng LLM.
- Ghi `Diagnosis.yaml` và `Wrong.yaml`.

Reference mặc định:

```text
Plan.yaml
PlanEntities.yaml
```

Reference teacher:

```text
TeacherPlan.yaml
TeacherAnswerEntities.yaml
```

Các nhãn so sánh chính:

- `all right`
- `combine step`
- `step separation`
- `reverse steps`
- `different calculation`
- `wrong relationship`

### LLMChecker.py

Trách nhiệm:

- Fallback bằng LLM khi pipeline symbolic không formalize được hoặc khi kết quả mơ hồ.
- Đọc đề bài, lời giải học sinh, lời giải giáo viên.
- Có thể chạy mode `teacher`, `review`, hoặc `auto`.

Hiện tại file này không nằm trong pipeline mặc định của `Tutor.py` hoặc `Grader.py`. Nó là fallback chạy riêng khi cần.

## Benchmark

### RunSolveBenchmark.py

Chạy `Main/Solver.py` trên nhiều bài trong `Benchmark/GSM8K Benchmark.csv`.

Mục tiêu:

- Đo pipeline tự giải có ra đúng official answer không.
- Lưu artifact lỗi vào `Error/`.
- Tạo `Summary.md`, `index.html`, và `results_summary.json` trong error folder.

### RunVerifyBenchmark.py

Chạy pipeline `Main/Grader.py` trên benchmark verifier.

Mục tiêu:

- Đánh giá pipeline symbolic teacher-vs-student.
- Lưu case lỗi vào `ErrorVerify/<id>/`.
- Tạo dashboard `ErrorVerify/index.html`.
- Tạo summary metric trong `ErrorVerify/results_summary.json`.

Metric chính hiện hướng tới:

- `Wrong Accuracy`
- `Wrong F1`
- `Error Label F1`
- `Exact Match`
- `Any Error Label Hit`

### RunBaseVerifyBenchmark.py

Chạy base LLM verifier trực tiếp, không qua pipeline symbolic.

Input vẫn là benchmark CSV, nhưng model đọc thẳng:

- đề bài
- lời giải học sinh
- lời giải giáo viên
- rubric label

Output nằm trong `ErrorBaseVerify/`.

File này dùng để so sánh pipeline symbolic với model base.

### RunBaseSolveBenchmark.py

Chạy baseline tự giải bằng LLM trực tiếp, không qua Solver symbolic. Dùng để so sánh chất lượng solver pipeline với model base.

## Quan Hệ Giữa Các File YAML

### Solver/Tutor Reference

```text
ProblemEntities.yaml
  -> Plan.yaml
  -> PlanEntities.yaml
```

`PlanEntities.yaml` là reference khi chạy Tutor.

### Student Branch

```text
ProblemEntities.yaml
Input/StudentAnswer.txt
  -> StudentPlan.yaml
  -> StudentAnswerEntities.yaml
```

`StudentAnswerEntities.yaml` được map với reference rồi đưa vào `CompareChecker.py`.

### Teacher Branch

```text
ProblemEntities.yaml
Input/TeacherAnswer.txt
  -> TeacherPlan.yaml
  -> TeacherAnswerEntities.yaml
```

`TeacherAnswerEntities.yaml` là reference khi chạy Grader.

## Ranh Giới Trách Nhiệm Quan Trọng

`ProblemFormalizer.py`

Chỉ trích dữ kiện đề bài và target. Không nên giải bài.

`Planner.py`

Tự sinh lời giải chuẩn của hệ thống. Có thể dùng LLM để hiểu bài, nhưng output cuối phải map được về plan symbolic.

`Executor.py`

Tính toán thật cho solver plan. Nếu quan hệ đúng thì value phải do code tính, không phụ thuộc LLM.

`StudentAnswerFormalizer.py`

Mô tả đúng những gì học sinh làm. Không sửa số sai, không tự biến lời giải học sinh thành lời giải đúng.

`TeacherAnswerFormalizer.py`

Mô tả lời giải giáo viên làm reference. Không ghi sang `Plan.yaml`/`PlanEntities.yaml` như Solver.

`InsideChecker.py`

Kiểm tra một lời giải có tự nhất quán không. Không map student với teacher.

`Mapper.py`

Chỉ nối entity tương ứng giữa hai phía. Không quyết định đúng/sai.

`CompareChecker.py`

So sánh hai lời giải đã map. Đây là nơi kết luận các lỗi quan hệ giữa student và reference.

`LLMChecker.py`

Fallback ngoài pipeline symbolic. Không nên dùng để che lỗi backbone khi đang benchmark pipeline chính.

## Debug Theo Triệu Chứng

Nếu `ProblemEntities.yaml` thiếu dữ kiện:

- Xem `ProblemFormalizer.py`.
- Kiểm tra prompt trích entity.
- Kiểm tra validator số chữ, phân số, unit conversion.

Nếu `Plan.yaml` tự giải sai:

- Xem `Code.txt` trước.
- Nếu `Code.txt` đã hiểu sai bài, lỗi ở prompt/LLM Planner.
- Nếu `Code.txt` đúng nhưng `Plan.yaml` sai, lỗi ở phần map pseudo-code sang entity.

Nếu `PlanEntities.yaml` value sai:

- Xem `Executor.py`.
- Kiểm tra `expr`, `reported_expr`, `formalized_expr`.
- Xem `Error.yaml` và `Log.yaml`.

Nếu `StudentPlan.yaml` có step linh tinh:

- Xem `StudentAnswerFormalizer.py`.
- Kiểm tra `reported_expr` có đúng lời học sinh viết không.
- Kiểm tra có copy step kiểu `110 = 110` không.
- Kiểm tra số literal trong `expr` đã được materialize thành `student_answer_number_*` chưa.

Nếu `TeacherPlan.yaml` fail vì student:

- Kiểm tra `TeacherAnswerFormalizer.py` có đang ghi đúng `TeacherPlan.yaml` và `TeacherAnswerEntities.yaml` không.
- Kiểm tra `InsideChecker.py --mode teacher` có đang đọc teacher files không.

Nếu `Diagnosis.yaml` bị mất lỗi:

- Kiểm tra các hàm merge diagnosis trong `InsideChecker.py` và `CompareChecker.py`.
- Stage sau không nên ghi đè toàn bộ diagnosis nếu stage trước đã phát hiện lỗi.

Nếu `Wrong.yaml` từ `Yes` thành `No`:

- Kiểm tra logic ghi `Wrong.yaml` trong checker sau cùng.
- Rule mong muốn hiện tại là nếu đã `Yes` thì không stage sau nào nên hạ xuống `No`.

Nếu benchmark hiển thị lỗi cũ:

- Xóa hoặc reset folder error tương ứng trước khi chạy lại.
- `Error/`, `ErrorVerify/`, `ErrorBaseVerify/` là artifact sinh ra, không phải source truth.

## File Sinh Ra Không Nên Commit

Các folder sau là output/debug artifact và đã nằm trong `.gitignore`:

```text
Error/
ErrorVerify/
ErrorBaseVerify/
```

Các file trong `Output/` thường là trạng thái chạy gần nhất. Có thể dùng để debug, nhưng cần cẩn thận nếu commit vì chúng phụ thuộc input hiện tại.

## Tóm Tắt Luồng Nên Nhớ

Tự giải bài:

```text
Main/Solver.py
Problem -> ProblemEntities -> Code/Plan -> PlanEntities
```

Tự giải rồi chấm học sinh:

```text
Main/Tutor.py
Solver reference <-> StudentAnswer
```

Chấm bằng lời giải giáo viên:

```text
Main/Grader.py
TeacherAnswer reference <-> StudentAnswer
```

Benchmark pipeline symbolic:

```text
Benchmark/RunVerifyBenchmark.py
-> Main/Grader.py
-> ErrorVerify/
```

Benchmark model base:

```text
Benchmark/RunBaseVerifyBenchmark.py
-> ErrorBaseVerify/
```
