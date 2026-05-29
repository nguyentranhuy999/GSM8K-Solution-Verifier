# Main Entrypoints

Nhóm `Main/` là lớp điều phối pipeline. Các file trong nhóm này không tự xử lý
schema toán học; chúng gọi các module ở `Formalizer/` và `Verifier/` theo đúng
thứ tự.

## `Main/Solver.py`

### Vai Trò

`Main/Solver.py` chạy nhánh tự giải bài toán để tạo lời giải chuẩn/reference từ
đề bài trong `Input/Problem.txt`.

Pipeline:

```text
ProblemFormalizer -> Planner -> Executor
```

Các output chính:

- `Output/ProblemEntities.yaml`
- `Output/Code.txt`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/Error.yaml`
- `Output/Log.yaml`

### Cách Hoạt Động

File định nghĩa dataclass `Stage` gồm:

- `name`: tên stage để in log.
- `command`: command Python cần chạy.

`STAGES` gồm ba stage:

1. `Formalizer/ProblemFormalizer.py`
2. `Formalizer/Solver/Planner.py`
3. `Formalizer/Solver/Executor.py`

Hàm `run_stage()` gọi `subprocess.run()` với `cwd` là root project. Nếu stage trả
exit code khác 0 thì `Solver.py` dừng ngay bằng `SystemExit`.

Hàm `run()` chạy lần lượt các stage và in `Pass Solver` khi xong.

### Tại Sao Thiết Kế Như Vậy

`Solver.py` là entrypoint nhỏ, có một trách nhiệm: tạo reference solver. Điều này
giúp debug rõ lỗi đang nằm ở bước nào:

- lỗi trích entity nằm ở `ProblemFormalizer`;
- lỗi lập plan nằm ở `Planner`;
- lỗi tính toán/rewrite nằm ở `Executor`.

`Solver.py` không gọi `InsideChecker.py` trực tiếp vì `Executor.py` đã tự gọi
`InsideChecker.py --mode llm` sau khi execute và tự repair nếu cần.

### Khác Biệt So Với `Prompt.pdf`

`Prompt.pdf` mô tả từng module backbone, nhưng chưa có một file điều phối riêng
tên `Main/Solver.py`. File này được thêm sau để chạy solver pipeline bằng một
command duy nhất.

## `Main/Tutor.py`

### Vai Trò

`Main/Tutor.py` là entrypoint tạo reference. Reference có thể đến từ hai nguồn:

- `solver`: hệ thống tự giải bằng `Main/Solver.py`.
- `teacher`: dùng lời giải giáo viên trong `Input/TeacherAnswer.txt`.

Command:

```bash
python3 Main/Tutor.py --reference solver
python3 Main/Tutor.py --reference teacher
```

### Output Reference

Mode `solver` tạo:

- `Output/ProblemEntities.yaml`
- `Output/Code.txt`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/Error.yaml`

Mode `teacher` tạo:

- `Output/ProblemEntities.yaml`
- `Output/TeacherPlan.yaml`
- `Output/TeacherAnswerEntities.yaml`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`

Trong mode teacher, `TeacherAnswerFormalizer.py` ghi cả file debug riêng
`TeacherPlan.yaml`/`TeacherAnswerEntities.yaml`, đồng thời ghi đè reference
chuẩn vào `Plan.yaml`/`PlanEntities.yaml` để các checker phía sau dùng chung
contract cũ.

### Cách Hoạt Động

`Tutor.py` có hai danh sách stage:

- `SOLVER_REFERENCE_STAGES`: gọi `Main/Solver.py`.
- `TEACHER_REFERENCE_STAGES`: gọi `ProblemFormalizer.py` rồi
  `TeacherAnswerFormalizer.py`.

Trước khi chạy, `clear_reference_outputs()` xoá các file reference cũ:

- `ProblemEntities.yaml`
- `Code.txt`
- `Plan.yaml`
- `PlanEntities.yaml`
- `TeacherPlan.yaml`
- `TeacherAnswerEntities.yaml`
- `Error.yaml`

Có thể dùng `--keep-existing` để không xoá các output này trước khi chạy.

### Tại Sao Thiết Kế Như Vậy

Sau khi dự án có hai nguồn reference, nếu tiếp tục nhét cả vào `Main.py` thì
pipeline khó đọc. `Tutor.py` tách riêng phần "tạo lời giải chuẩn" khỏi phần
"chấm lời giải học sinh".

Nó cũng làm rõ boundary:

- `Tutor.py` chỉ tạo reference.
- `Grader.py` chỉ chấm student answer dựa trên reference.

### Khác Biệt So Với `Prompt.pdf`

`Prompt.pdf` không có `Tutor.py`. File này được thêm sau khi hệ thống cần hai
luồng reference:

- tự giải bằng solver;
- dùng lời giải chuẩn của giáo viên.

## `Main/Grader.py`

### Vai Trò

`Main/Grader.py` là entrypoint chấm lời giải học sinh. Nó dùng reference đã có
trong:

- `Output/ProblemEntities.yaml`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`

Sau đó chạy nhánh student:

```text
StudentAnswerFormalizer
InsideChecker --mode student
Mapper
CompareChecker
```

Command mặc định:

```bash
python3 Main/Grader.py
```

Command chạy full pipeline một lệnh:

```bash
python3 Main/Grader.py --reference solver
python3 Main/Grader.py --reference teacher
```

### Cách Hoạt Động

`Grader.py` có ba mode reference:

- `existing`: không tạo reference mới, chỉ kiểm tra các file reference đã tồn tại.
- `solver`: gọi `Main/Tutor.py --reference solver` trước khi chấm.
- `teacher`: gọi `Main/Tutor.py --reference teacher` trước khi chấm.

Trước khi chấm, `clear_grading_outputs()` xoá các file output cũ của grading:

- `StudentPlan.yaml`
- `StudentAnswerEntities.yaml`
- `Diagnosis.yaml`
- `Wrong.yaml`
- `Error.yaml`
- `LLMChecker.yaml`

Với `--keep-existing`, `Grader.py` không xoá các file này.

Nếu chạy mode `existing`, `ensure_reference_exists()` bắt buộc phải có:

- `Output/ProblemEntities.yaml`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`

Nếu thiếu, chương trình dừng và gợi ý chạy `Tutor.py` hoặc `Grader.py --reference`.

### Tại Sao Thiết Kế Như Vậy

Tách `Grader.py` giúp chạy lại nhiều bài học sinh trên cùng một reference mà
không phải gọi lại solver. Điều này tiết kiệm token và giúp debug:

1. Chạy `Tutor.py` để tạo reference một lần.
2. Sửa `Input/StudentAnswer.txt`.
3. Chạy `Grader.py` nhiều lần để kiểm tra formalizer/checker student.

### Khác Biệt So Với `Prompt.pdf`

`Prompt.pdf` chưa có `Main/Grader.py`. Trước đó dự án từng dùng `Main/Main.py`
cho full pipeline. Hiện tại full pipeline được thay bằng:

```bash
python3 Main/Grader.py --reference solver
```

hoặc:

```bash
python3 Main/Grader.py --reference teacher
```

## Quan Hệ Giữa Ba File

```text
Main/Solver.py
  chỉ tạo reference bằng solver

Main/Tutor.py
  tạo reference bằng solver hoặc teacher

Main/Grader.py
  chấm student answer, có thể gọi Tutor trước nếu cần
```

Thiết kế hiện tại giữ nguyên nguyên tắc "mỗi file làm một phần":

- `Solver.py`: solver engine entrypoint.
- `Tutor.py`: reference builder.
- `Grader.py`: grading entrypoint.

