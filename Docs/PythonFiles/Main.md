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

`Main/Tutor.py` là entrypoint tutor tự giải và tự chấm lời giải học sinh.
Nó dùng solver làm lời giải chuẩn, sau đó so sánh bài học sinh với lời giải do
hệ thống tự sinh.

Command:

```bash
python3 Main/Tutor.py
```

### Pipeline

```text
Solver
StudentAnswerFormalizer
InsideChecker --mode student
Mapper
CompareChecker
```

### Output

- `Output/ProblemEntities.yaml`
- `Output/Code.txt`
- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`
- `Output/Diagnosis.yaml`
- `Output/Wrong.yaml`
- `Output/Error.yaml`

### Cách Hoạt Động

`Tutor.py` gọi `Main/Solver.py` trước. `Solver.py` tự chạy
`ProblemFormalizer -> Planner -> Executor`, tạo `Plan.yaml` và
`PlanEntities.yaml`.

Sau đó `Tutor.py` chạy nhánh student:

- `StudentAnswerFormalizer.py`: formalize bài học sinh.
- `InsideChecker.py --mode student`: kiểm tra lỗi nội tại trong lời giải học sinh.
- `Mapper.py`: map `StudentAnswerEntities.yaml` với `PlanEntities.yaml`.
- `CompareChecker.py`: so sánh lời giải solver với lời giải học sinh.

Trước khi chạy, `clear_tutor_outputs()` xoá các file output cũ:

- `ProblemEntities.yaml`
- `Code.txt`
- `Plan.yaml`
- `PlanEntities.yaml`
- `TeacherPlan.yaml`
- `TeacherAnswerEntities.yaml`
- `StudentPlan.yaml`
- `StudentAnswerEntities.yaml`
- `Diagnosis.yaml`
- `Wrong.yaml`
- `Error.yaml`
- `LLMChecker.yaml`

Có thể dùng `--keep-existing` để không xoá các output này trước khi chạy.

### Tại Sao Thiết Kế Như Vậy

`Tutor.py` đại diện cho luồng hệ thống tự làm tutor: tự giải bài toán và dùng
lời giải đó để chấm bài học sinh. Luồng này hữu ích khi không có lời giải giáo
viên hoặc khi muốn đo toàn bộ backbone solve-and-verify.

Boundary hiện tại:

- `Solver.py`: chỉ giải bài toán.
- `Tutor.py`: tự giải rồi tự chấm.
- `Grader.py`: chấm dựa trên lời giải giáo viên có sẵn.

### Khác Biệt So Với `Prompt.pdf`

`Prompt.pdf` không có `Tutor.py`. File này được thêm để gom luồng
`Solver -> StudentAnswerFormalizer -> Mapper -> CompareChecker` thành một lệnh
khi muốn hệ thống tự giải và tự chấm.

## `Main/Grader.py`

### Vai Trò

`Main/Grader.py` là entrypoint chấm lời giải học sinh bằng lời giải giáo viên.
Nó là pipeline riêng với `Tutor.py`/`Solver.py`, không dùng reference solver
trong `Output/Plan.yaml` và `Output/PlanEntities.yaml` để chấm.

Pipeline hiện tại:

```text
ProblemFormalizer
StudentAnswerFormalizer
TeacherAnswerFormalizer
InsideChecker --mode student
Mapper --reference teacher
CompareChecker --reference teacher
```

Command chạy:

```bash
python3 Main/Grader.py
```

### Cách Hoạt Động

`Grader.py` chạy hai nhánh formalize song song về mặt contract:

- `ProblemFormalizer.py` tạo `ProblemEntities.yaml`, rồi copy entity gốc sang
  `StudentAnswerEntities.yaml` và `TeacherAnswerEntities.yaml`.
- `StudentAnswerFormalizer.py` đọc bài học sinh, tạo `StudentPlan.yaml` và thêm
  entity trung gian vào `StudentAnswerEntities.yaml`.
- `TeacherAnswerFormalizer.py` đọc lời giải giáo viên, tạo `TeacherPlan.yaml`
  và thêm entity trung gian vào `TeacherAnswerEntities.yaml`.
- `Mapper.py --reference teacher` map entity student với entity teacher.
- `CompareChecker.py --reference teacher` so sánh hai plan/entity sau map.

Trước khi chấm, `clear_grader_outputs()` xoá các file output cũ:

- `ProblemEntities.yaml`
- `Plan.yaml`
- `PlanEntities.yaml`
- `TeacherPlan.yaml`
- `TeacherAnswerEntities.yaml`
- `StudentPlan.yaml`
- `StudentAnswerEntities.yaml`
- `Diagnosis.yaml`
- `Wrong.yaml`
- `Error.yaml`
- `LLMChecker.yaml`

Với `--keep-existing`, `Grader.py` không xoá các file này.

### Tại Sao Thiết Kế Như Vậy

Tách `Grader.py` theo teacher-vs-student giúp debug rõ hơn:

- `Tutor.py` tự giải và tự chấm theo lời giải solver.
- `Grader.py` chỉ chấm bài bằng hai lời giải đã cho trong input.
- `Plan.yaml`/`PlanEntities.yaml` không còn là contract bắt buộc của grader,
  nên lỗi từ solver không lẫn vào benchmark verifier.

### Khác Biệt So Với `Prompt.pdf`

`Prompt.pdf` chưa có `Main/Grader.py`. Trước đó dự án từng dùng `Main/Main.py`
cho full pipeline. Hiện tại grader được tách thành luồng riêng:
`ProblemFormalizer -> StudentAnswerFormalizer -> TeacherAnswerFormalizer ->
Mapper teacher -> CompareChecker teacher`.

## Quan Hệ Giữa Ba File

```text
Main/Solver.py
  chỉ giải bài toán bằng solver

Main/Tutor.py
  tự giải bằng solver rồi chấm student answer

Main/Grader.py
  chấm student answer trực tiếp với teacher answer có sẵn
```

Thiết kế hiện tại giữ nguyên nguyên tắc "mỗi file làm một phần":

- `Solver.py`: solver engine entrypoint.
- `Tutor.py`: self-solve grading entrypoint.
- `Grader.py`: grading entrypoint.
