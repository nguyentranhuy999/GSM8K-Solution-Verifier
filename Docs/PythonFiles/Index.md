# Python Files Documentation

Tài liệu này mô tả các file Python hiện có trong dự án theo đúng trạng thái code
hiện tại, không chỉ theo thiết kế ban đầu trong `Docs/Prompt.pdf`.

## Cách Đọc

- `Main.md`: các entrypoint chạy pipeline.
- `Formalizer.md`: các module biến đề bài/lời giải thành YAML symbolic.
- `Verifier.md`: các checker, mapper-review và fallback LLM.
- `Benchmark.md`: các script chạy benchmark và sinh report.
- `../Overview.html`: bản dashboard trực quan để click xem nhanh pipeline và vai trò từng file.

## Danh Sách File Python

| File | Nhóm | Vai trò ngắn |
|---|---|---|
| `Main/Solver.py` | Main | Chạy solver reference: ProblemFormalizer -> Planner -> Executor. |
| `Main/Tutor.py` | Main | Tạo reference bằng solver hoặc bằng lời giải giáo viên. |
| `Main/Grader.py` | Main | Chấm lời giải học sinh dựa trên reference đã có hoặc tự gọi Tutor. |
| `Formalizer/ProblemFormalizer.py` | Formalizer | Trích entity số trực tiếp trong đề bài. |
| `Formalizer/Solver/Planner.py` | Formalizer | Sinh pseudo-code, map số sang entity, tạo `Plan.yaml`. |
| `Formalizer/Solver/Executor.py` | Formalizer | Thực thi plan, tính value, formalized expr, gọi InsideChecker và repair. |
| `Formalizer/StudentAnswerFormalizer.py` | Formalizer | Formalize bài làm học sinh, giữ đúng phép tính học sinh viết. |
| `Formalizer/TeacherAnswerFormalizer.py` | Formalizer | Formalize lời giải giáo viên thành reference plan. |
| `Formalizer/Mapper.py` | Formalizer | Map entity giữa lời giải chuẩn và lời giải học sinh. |
| `Verifier/InsideChecker.py` | Verifier | Check lỗi nội tại trong một plan/entities. |
| `Verifier/CompareChecker.py` | Verifier | So sánh reference với student sau khi map. |
| `Verifier/LLMChecker.py` | Verifier | Fallback/review bằng LLM, chưa nối mặc định vào pipeline benchmark chính. |
| `Benchmark/RunSolveBenchmark.py` | Benchmark | Chạy solver trên GSM8K và so target answer. |
| `Benchmark/RunVerifyBenchmark.py` | Benchmark | Chạy pipeline grader với teacher reference trên benchmark verify. |
| `Benchmark/RunBaseVerifyBenchmark.py` | Benchmark | Chạy model base trực tiếp để làm baseline verifier. |

## Khác Biệt Chung So Với `Docs/Prompt.pdf`

`Prompt.pdf` thiết kế backbone ban đầu gồm các module chính:
`ProblemFormalizer`, `Planner`, `Executor`, `StudentAnswerFormalizer`,
`Mapper`, `InsideChecker`, `CompareChecker`, `LLMChecker`.

Code hiện tại đã mở rộng theo hướng hybrid:

- LLM chịu trách nhiệm hiểu ngôn ngữ tự nhiên và sinh bản nháp.
- Python chịu trách nhiệm khóa schema, tính toán, validate, map, repair và benchmark.
- Một số file mới như `Main/Tutor.py`, `Main/Grader.py`,
  `TeacherAnswerFormalizer.py`, các benchmark runner không nằm trong prompt gốc.
- `Diagnosis.yaml` là tên đúng chính tả hiện tại; prompt gốc có chỗ ghi
  `Diagonosis.yaml`.
- `Planner.py` không còn chỉ bắt LLM sinh `Plan.yaml` trực tiếp. Hiện tại LLM
  sinh pseudo-code vào `Output/Code.txt`, sau đó Python map số literal trong code
  về entity để tạo plan symbolic.
- `ProblemFormalizer.py` vẫn giữ nguyên nguyên tắc không tạo kết quả trung gian
  do LLM tính ra, nhưng code có thêm helper constants như `unit_conversion_*`,
  `identity_multiplier`, `percentage_scale`, `host_count`, `split_count`.
  Những helper này được thêm bằng code để giữ `expr` chỉ dùng biến, không dùng
  số literal.

