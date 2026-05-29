# Verifier Files

Nhóm `Verifier/` kiểm tra lỗi sau khi dữ liệu đã được formalize thành plan và
entities. Đây là phần cố gắng chạy deterministic bằng Python nhiều nhất có thể.

## `Verifier/InsideChecker.py`

### Vai Trò

`InsideChecker.py` kiểm tra lỗi nội tại trong một lời giải, tức là chỉ nhìn một
cặp plan/entities tại một thời điểm.

Có hai mode:

```bash
python3 Verifier/InsideChecker.py --mode llm
python3 Verifier/InsideChecker.py --mode student
```

Mode `llm` đọc:

- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`

và ghi:

- `Output/Error.yaml`

Mode `student` đọc:

- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`

và ghi:

- `Output/Diagnosis.yaml`
- `Output/Wrong.yaml`

### Các Lỗi Được Check

Các nhãn chính:

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

### Cách Check

Các check chính:

- `check_wrong_target()`: target entity phải được tính đúng, và trong student
  mode, `target:` cuối `StudentPlan.yaml` phải trỏ đến entity hợp lý.
- `check_negative_count_target()`: bắt target dạng count âm.
- `check_wrong_calculation()`: parse `reported_expr`, tính lại RHS bằng Python
  và so với kết quả học sinh/LLM báo.
- `check_unit_missing()`: chỉ student mode; entity có `unit: missing` bị ghi
  `unit missing`.
- `check_only_final_answer()`: plan chỉ có một step rỗng expr bị coi là chỉ trả
  đáp án.
- `check_wrong_relationship()`: bắt cộng/trừ các đơn vị không tương thích,
  hoặc trường hợp đáng ra phải đổi đơn vị.
- `check_double_count_summary_counts()`: bắt lỗi nhân đôi count summary khi
  component đã được liệt kê.
- `check_missing_step()`: bắt dùng entity chưa tồn tại, chưa có value, hoặc phụ
  thuộc vào step tương lai.
- `check_misreading_and_logic_error()`: so số trong `reported_expr` với value
  của entity trong `expr`; nếu khác input thì là `misreading`, nếu khác result
  trung gian thì là `logic error`.
- `check_extra_step()`: phát hiện result không được dùng về sau.

### Evaluation An Toàn

Giống Executor, InsideChecker dùng AST evaluator thay vì `eval()`.

Nó chỉ cho phép arithmetic đơn giản để kiểm tra `reported_expr`.

### Ghi Diagnosis Và Wrong

Mode `llm` ghi lỗi vào `Error.yaml` để Executor có thể repair.

Mode `student` append lỗi vào `Diagnosis.yaml`, không xoá lỗi trước đó.

`Wrong.yaml` được cập nhật theo nguyên tắc:

- có lỗi nghiêm trọng khác `extra step` thì `Yes`;
- chỉ lỗi nhẹ/extra step thì `No`;
- nếu trước đó đã là `Yes`, checker không hạ xuống `No`.

### Tại Sao Thiết Kế Như Vậy

InsideChecker là lớp "unit test" cho một lời giải. Nó không cần biết lời giải
chuẩn và lời giải học sinh có giống nhau không; nó chỉ hỏi:

- plan này tự thân có hợp lệ không?
- phép tính báo cáo có đúng không?
- có dùng dữ kiện sai không?
- có thiếu bước không?
- có quan hệ đơn vị vô lý không?

Việc tách checker nội tại khỏi CompareChecker giúp debug rõ hơn: lỗi nằm trong
student formalization hay nằm ở so sánh với reference.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc đã mô tả InsideChecker tương đối đầy đủ. Code hiện tại mở rộng:

- chuẩn hóa `Diagnosis.yaml`;
- merge diagnosis thay vì overwrite;
- giữ `Wrong.yaml = Yes` nếu stage trước đã phát hiện sai;
- thêm check count âm và double-count summary;
- dùng AST safe evaluator;
- xử lý `extra step` như lỗi nhẹ để Executor không repair vô ích.

## `Verifier/CompareChecker.py`

### Vai Trò

`CompareChecker.py` so sánh lời giải chuẩn với lời giải học sinh sau khi Mapper
đã thêm field `map`.

Input:

- `Output/Plan.yaml`
- `Output/PlanEntities.yaml`
- `Output/StudentPlan.yaml`
- `Output/StudentAnswerEntities.yaml`

Output:

- append vào `Output/Diagnosis.yaml`
- update `Output/Wrong.yaml`
- log vào `Output/Log.yaml`

### Các Check Chính

`check_wrong_units_conversion()`:

- nếu student entity đã map với plan entity;
- value khác;
- metadata unit/grand_unit thuộc nhóm convertible;
- thì ghi `wrong units conversion`.

`check_wrong_relationship()`:

- nếu student entity map được với plan entity nhưng `expr` khác quan hệ;
- ưu tiên bắt lỗi quan hệ thật thay vì chỉ khác cách trình bày.

`check_different_calculation()`:

- nếu target student có đáp án đúng nhưng `formalized_expr` khác reference;
- dùng random substitution để kiểm tra hai biểu thức có tương đương không.

`check_all_right()`:

- nếu các entity map/value/unit/expr tương thích toàn bộ;
- ghi `all right`.

`check_step_structure_change()`:

- nếu target expression tương đương nhưng số bước khác:
  - ít bước hơn: `combine step`;
  - nhiều bước hơn: `step separation`;
  - bằng số bước nhưng thứ tự khác: `reverse steps`.

### Expression Equivalence

CompareChecker dùng hai hướng:

- so text normalized cho trường hợp đơn giản;
- random substitution cho `formalized_expr`.

Ý tưởng: nếu thay ngẫu nhiên các input entity bằng vài bộ số khác nhau mà hai
expression luôn ra cùng kết quả, thì hai cách tính được coi là tương đương.

### Thứ Tự Ưu Tiên Lỗi

Code hiện tại chạy:

1. wrong unit conversion;
2. wrong relationship;
3. different calculation;
4. nếu chưa có core error mới check all right/step structure.

Điều này tránh trường hợp một lời giải sai quan hệ nhưng vẫn bị ghi thêm quá
nhiều label trình bày.

`remove_structural_label_if_all_right()` xoá nhãn structural nếu đã có
`all right`.

### Ghi Wrong

Hiện tại `Wrong.yaml` chỉ thành `Yes` nếu có `wrong relationship`. Các nhãn như
`combine step`, `step separation`, `reverse steps`, `different calculation`,
`all right` thường là `No`.

Nếu `Wrong.yaml` đã là `Yes` từ InsideChecker, CompareChecker không hạ xuống
`No`.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc đã có các checker so sánh. Code hiện tại mở rộng:

- merge diagnosis thay vì overwrite;
- giữ `Wrong.yaml = Yes` nếu đã có lỗi trước;
- dùng AST/random substitution an toàn hơn;
- tách core error trước structural labels;
- xoá structural label nếu all right;
- có thêm normalize label/metadata để giảm false positive khi học sinh thiếu
  unit nhưng relation vẫn map được.

## `Verifier/LLMChecker.py`

### Vai Trò

`LLMChecker.py` là fallback bằng LLM. Nó không nằm trong pipeline mặc định hiện
tại; chỉ chạy khi gọi trực tiếp hoặc khi sau này nối vào.

Command:

```bash
python3 Verifier/LLMChecker.py --mode teacher
python3 Verifier/LLMChecker.py --mode review
python3 Verifier/LLMChecker.py --mode auto
```

Input mặc định:

- `Input/Problem.txt`
- `Input/StudentAnswer.txt`
- `Input/TeacherAnswer.txt`

Output:

- `Output/Diagnosis.yaml`
- `Output/Wrong.yaml`
- `Output/LLMChecker.yaml`
- `Output/Log.yaml`

### Modes

`teacher`:

- luôn gọi LLM để so sánh trực tiếp problem + student answer + teacher answer;
- dùng khi symbolic pipeline không formalize được.

`review`:

- chỉ gọi LLM nếu hiện tại:
  - `Diagnosis.yaml` có `different calculation`;
  - `Wrong.yaml` là `No`.
- dùng để review trường hợp CompareChecker cho là cách tính khác nhưng đúng,
  tránh bỏ sót quan hệ sai.

`auto`:

- nếu đủ điều kiện review thì chạy review;
- nếu không thì chạy teacher fallback.

### Output Schema LLM

LLM phải trả YAML dạng:

```yaml
LLMChecker.yaml:
  wrong: Yes
  diagnosis:
    - diagnosis: wrong relationship
      step:
      entity:
  reason: ...
```

Code normalize:

- label aliases, ví dụ `wrong caculation` -> `wrong calculation`;
- `wrong` thành `Yes`/`No`;
- bỏ `all right` nếu có lỗi nghiêm trọng;
- nếu `wrong=Yes` mà không có diagnosis hợp lệ thì reject.

### Debug File `LLMChecker.yaml`

`Output/LLMChecker.yaml` lưu:

- mode chạy;
- wrong/diagnosis/reason từ LLM;
- raw response;
- snapshot các file plan/entities nếu review mode.

File này không phải contract chính của pipeline; nó là debug artifact để xem LLM
fallback đã quyết định như thế nào.

### Tại Sao Không Nối Mặc Định

LLMChecker làm tăng token và có thể làm benchmark pipeline symbolic bị lẫn với
baseline LLM. Hiện tại nó được giữ standalone để:

- debug fallback riêng;
- đo pipeline symbolic riêng;
- tránh stage cuối dùng LLM ghi đè kết quả deterministic.

### Khác Biệt So Với `Prompt.pdf`

Prompt gốc chỉ yêu cầu một LLMChecker fallback. Code hiện tại cụ thể hóa:

- ba mode `teacher`, `review`, `auto`;
- whitelist label;
- merge diagnosis;
- không hạ `Wrong.yaml` từ `Yes` về `No`;
- debug output riêng `LLMChecker.yaml`;
- điều kiện review riêng cho `different calculation + Wrong No`.

