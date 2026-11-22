# Bộ học sinh mô phỏng (A6, P1-08) và thí nghiệm vòng kín (C5, P1-09)

Học sinh ảo có tham số thật đã biết, để đo hệ thống chọn bài có rơi đúng vùng ZPD không (P1-09). Mọi kết quả chạy trên
bộ này là **kết quả mô phỏng**, không phải kết quả trên người học thật.

- **Mã:** `lmsrag/sim/model.py` (mô hình, hiệu chỉnh), `lmsrag/sim/env.py` (môi trường), `lmsrag/sim/calibrate.py`
  (kiểm tra và bảng báo cáo). Cấu hình: `configs/sim.yaml`. Test: `lmsrag/tests/test_sim.py`.
- **Đầu vào (không commit):** kho bài tập `lmsrag/artifacts/v2/exercise_bank_v2.jsonl` (P1-04, sau quyết định P1-06)
  và phần **history** của `interactions_junyi2015.parquet` (P1-03).
- **Báo cáo:** `reports/tables/a6_sim_calibration.csv`, `a6_sim_learning.csv`, `a6_sim_items.csv`.

## Mô hình

```text
Xác suất đúng      P(u, i) = g_i + (1 − g_i − s_i) · σ(a_i · (θ_{u,k} − b_i)),   k = skill_tag của bài i
Độ khó bài         b_i = logit(1 − p_correct_hist_i), p cắt về [0,01; 0,99]       (bài có số liệu Junyi, bậc 1)
                   b_i = Q_b(difficulty_i), ánh xạ phân vị                         (bài bậc 2 và 3)
Độ phân biệt       a_i ~ LogNormal(0; 0,3), rút một lần theo world_seed = 2026
Guess, slip        g_i = 0,20 nếu answer_type = radio (chọn đáp án), 0,02 với bài điền; s_i = 0,05
Người học          θ_{u,k} ~ Normal(μ_k, 1), độc lập giữa các kỹ năng
Hiệu chỉnh μ_k     (1/|I_k|) Σ_{i∈I_k} E_θ[P(θ, i)] = tỷ lệ đúng của kỹ năng k trong history Junyi
Tốc độ học         η_u ~ Uniform(0,0218; 0,0871), một giá trị cho mọi kỹ năng của người học (xem "Tốc độ học")
Học sau mỗi lượt   θ_{u,k} ← θ_{u,k} + η_u · G(P),  P tính trước lượt đó, cập nhật dù đúng hay sai
```

| Quy tắc học | G(P) | Lý do |
|---|---|---|
| `flat` | 1 | Học như nhau dù bài dễ hay khó |
| `info` | 4 · P · (1 − P) | Học nhiều nhất ở bài vừa sức theo lý thuyết thông tin, đỉnh tại P = 0,5 |
| `zpd` | exp(−(P − 0,7)² / (2 · 0,15²)) | Giả định ZPD của đề tài |

Quy tắc `zpd` cho hệ nhắm vào 0,7 lợi thế sẵn có. Vì vậy kết luận "hiểu đúng vùng ZPD" chỉ dựa vào tỷ lệ bài có
`true_p` trong [0,6; 0,8] (không phụ thuộc quy tắc học), còn kết luận "học nhanh hơn" phải nêu rõ đúng với quy tắc nào.

Chi tiết cài đặt:

- **Ánh xạ phân vị cho bài không có số liệu Junyi.** Lấy các bài bậc 1, sắp xếp riêng `difficulty` và b, nội suy
  tuyến tính, giữ nguyên ở hai đầu. Ánh xạ đơn điệu, nên trong mọi nguồn `difficulty` cao hơn thì b cao hơn. Với bài
  bậc 1, `difficulty` là hạng của 1 − p, nên ánh xạ trả lại đúng b của chúng. Ánh xạ fit trên cả kho, trước khi lọc kỹ năng.
- **Kỳ vọng theo θ** tính bằng cầu phương Gauss–Hermite 64 nút; μ_k tìm bằng chia đôi trên [−10; 10] (tỷ lệ đúng tăng theo μ).
- **Bài vào mô phỏng:** bài có `skill_tag` có ít nhất 200 lượt trong history Junyi (`min_junyi_responses`). Bài thuộc
  nhiều kỹ năng chỉ dùng `skill_tag` (phần tử đầu của `skill_tags`).
- **"Tỷ lệ đúng ban đầu" của mô phỏng** là trung bình đều trên các bài của kỹ năng; Junyi thì theo số lượt làm từng
  bài. Hai cách lấy trung bình này khác nhau khi chính sách chọn bài không đều.

## Môi trường (`env.py`)

| Hàm | Ai được gọi | Trả về |
|---|---|---|
| `reset(seed)` | chính sách | rút θ_{u,k} và η_u cho `n_learners` người học |
| `catalog` | chính sách | bảng bài công khai: `item_id`, `skill_tag`, `source`, `difficulty`, `difficulty_level`, `difficulty_tier`, `difficulty_reliable`, `answer_type`, `khan_slug`, `problem_type`; không có a, b, g, s |
| `observe(u)` | chính sách | các lượt đã làm: `(t, item_id, correct)`; không có θ, η hay P |
| `step(u, item_id)` | chính sách | 0/1, rồi cập nhật θ |
| `true_p(u, item_id)` | chỉ để tính chỉ số | P hiện tại |
| `true_theta(u, skill_tag)` | chỉ để tính chỉ số | θ hiện tại (cho chỉ số "tăng mastery" của P1-09) |

Số ngẫu nhiên: `reset(seed)` rút mọi người học từ một bộ sinh, nên cùng seed thì cùng người học ở cả ba quy tắc học.
Mỗi người học có luồng đúng/sai riêng (`SeedSequence(seed).spawn`), nên đúng/sai của một người chỉ phụ thuộc seed và
các bài người đó được giao, không phụ thuộc thứ tự chính sách đi qua các người học. `world_seed` cố định bài (a_i);
P1-09 chỉnh tham số trên `world_seed + 1` = 2027 và chạy cuối trên 2026.

## Quy mô

| Nguồn | Bậc độ khó | Bài trong kho | Bài vào mô phỏng | b: p10 / trung vị / p90 |
|---|---:|---:|---:|---|
| Khan, từ Junyi | 1 | 1.273 | 1.273 | −1,93 / −1,01 / 0,00 |
| Khan, từ số bước gợi ý | 2 | 791 | 476 | −1,13 / −0,88 / −0,67 |
| GSM8K | 3 | 1.892 | 1.193 | −1,48 / −1,04 / 0,00 |
| **Tổng** | | **3.956** | **2.942** | |

83 kỹ năng (59 tag chuẩn, 24 tag `junyi:<topic>`); số bài mỗi kỹ năng: trung vị 16, ít nhất 4, nhiều nhất 462
(Percent-Of). Bỏ 1.014 bài: 950 bài của 42 tag không có lượt nào trong Junyi (lớn nhất là Unit-Rate, 627 bài GSM8K;
các tag `khan:<slug>`), và 64 bài của 5 tag có dưới 200 lượt (Parallel-and-Perpendicular-Lines 80 lượt,
Area-Irregular-Figure 126, Solving-for-a-variable 130, Finding-y-intercept-from-Linear-Equation 103, Intercept 30).

## Kiểm tra

Chạy `python -m lmsrag.sim.calibrate --config configs/sim.yaml` (khoảng 20 giây). Ngày 20/11/2026:

- **Tỷ lệ đúng ban đầu theo kỹ năng so với Junyi:** 10.000 người học ảo (seed 0), mỗi người làm một bài ngẫu nhiên
  của từng kỹ năng qua `step`. |mô phỏng − Junyi| trung bình 0,0035, lớn nhất **0,0125**
  (Multiplying-non-Monomial-Polynomials); không kỹ năng nào lệch ≥ 0,05. Theo công thức (cầu phương), lệch lớn nhất
  là 0,0045. Đó là ở Area-Parallelogram: Junyi đúng 0,955, vượt mức trần 1 − s = 0,95 của mô hình, nên μ chạm biên 10.
  Tỷ lệ đúng Junyi của 83 kỹ năng: từ 0,357 (Percent-Of) đến 0,955, trung vị 0,767. Bảng: `a6_sim_calibration.csv`.
- **Test (`test_sim.py`):** cùng seed cho cùng chuỗi, kể cả khi đổi thứ tự đi qua người học; đúng/sai khớp `true_p`
  trên 10.000 lượt trong 4 sai số chuẩn, cả trên toàn bộ và theo từng khoảng P; `observe` chỉ trả `(t, item_id,
  correct)`; `catalog` không có a, b, g, s; mỗi lượt chỉ θ của kỹ năng vừa làm thay đổi, đúng một lượng η · G(P).
## Tốc độ học (chọn η)

Kế hoạch: η_u ~ Uniform(0,02; 0,08), "chọn sao cho học sinh trung bình cần khoảng 30 lượt để lên một đơn vị θ".

- **Cách đo:** số lượt đến khi θ_{u,k} tăng 1 đơn vị; 1.000 người học, mỗi người một kỹ năng ngẫu nhiên, bài rút ngẫu
  nhiên trong kỹ năng đó (như chính sách `random` của P1-09). "Học sinh trung bình" = trung vị gộp cả ba quy tắc học,
  vì ba quy tắc dùng chung một phân phối η. Người không đạt trong 400 lượt được tính là trên trung vị.
- **Cách tìm** (`python -m lmsrag.sim.calibrate --tune-eta`): giữ tỷ lệ cận trên / cận dưới = 4 như kế hoạch, co giãn
  cả khoảng bằng chia đôi trên log(hệ số), tìm trên `world_seed + 1` = 2027 để world chạy cuối không tham gia chọn η.
  Mỗi lần thử dùng cùng người học và cùng lượt rút bài (seed 0), nên các lần so được với nhau. Nhật ký:
  `a6_sim_eta_tuning.csv`.
- **Kết quả:** hệ số 1,089, làm tròn thành **η ~ U(0,0218; 0,0871)**. Khoảng của kế hoạch cho 33 lượt; khoảng mới cho
  30 lượt, cả trên world 2027 (dùng để tìm) và world 2026 (kiểm tra).

Theo từng quy tắc, với η mới (`a6_sim_learning.csv`, world 2026):

| Quy tắc | Bài ngẫu nhiên trong kỹ năng: trung vị | Trung bình (người đạt) | Đạt trong 400 lượt | Nếu P luôn = 0,7: trung bình qua người học |
|---|---:|---:|---:|---:|
| `flat` | 19 | 21,9 | 100% | 21,2 |
| `info` | 40 | 51,2 | 100% | 25,3 |
| `zpd` | 38 | 49,1 | 99,0% | 21,2 |

Với `info` và `zpd`, bài ngẫu nhiên cho học chậm gấp đôi bài vừa sức. Khoảng cách đó chính là chỗ để chính sách chọn
bài tạo khác biệt trong P1-09. Với 50 lượt, người học trung bình tăng khoảng 1,3–2,6 đơn vị θ, tùy quy tắc và chính sách.

## Thí nghiệm vòng kín (P1-09, C5)

`lmsrag/sim/run.py`, chính sách baseline ở `lmsrag/sim/baselines.py`, cấu hình ở mục `closed_loop` của `sim.yaml`.

- **Thiết kế:** 1.000 học sinh × 50 lượt × 5 seed (0–4) × 3 quy tắc học. Mỗi học sinh luyện một kỹ năng, rút đều
  trong các kỹ năng mà học sinh đó **chưa thành thạo** trước lượt 1 (`skill_assignment: uniform_unmastered`, chọn ở
  mục "Chọn thiết kế" bên dưới). Cùng một seed thì mọi chính sách và mọi quy tắc gặp cùng học sinh, cùng kỹ năng, nên
  chênh lệch giữa hai chính sách là ghép cặp theo học sinh.
- **Giao diện chính sách** (như P2-08): `choose(observed_history, candidate_items) -> item_id`.
  `observed_history` là `observe(u)`; `candidate_items` là các dòng công khai của `catalog` thuộc kỹ năng đang luyện.
  Bài trả về phải nằm trong danh sách đó, nếu không thì `run.py` báo lỗi. `reset(seed)` được gọi trước mỗi seed nếu
  chính sách có hàm này. `ProposedPolicy` được tạo bằng `ProposedPolicy.from_config(cfg, catalog)` nếu có, nếu không
  thì bằng `ProposedPolicy()`.
- **Chính sách:** cả ba baseline đưa bài học sinh đã gặp ít lần nhất trước, nên một bài chỉ lặp lại khi các bài cùng
  hạng đã được đưa đủ.
  - `random`: một bài ngẫu nhiên.
  - `fixed`: luôn mức khó 3. Kỹ năng không có bài mức 3 thì dùng mức gần nhất có bài; nếu cách đều hai phía thì lấy
    mức thấp hơn.
  - `most_popular`: bài có nhiều lượt nhất trong history Junyi, tính theo đơn vị (template, dạng bài), rồi theo
    template. Junyi ghi dạng bài lúc theo mã, lúc theo số thứ tự (`khan_problem_types.csv`), nên một đơn vị cộng cả
    hai cách ghi; 64,6% bài Khan có số lượt ở mức đơn vị. Bài GSM8K có số lượt 0 nên xếp cuối.
  - `proposed`: P2-08 của Người 2, hạn 14/12, chưa có.
  - `oracle` (dòng tham chiếu, `references`): đọc P thật của mọi bài ứng viên và đưa bài có P gần 0,7 nhất,
    được lặp bài. Đây là trần của tỷ lệ trúng ZPD có thể đạt, không phải một chính sách, và không nằm trong H6.
- **Chỉ số:**

| Chỉ số | Định nghĩa |
|---|---|
| `zpd_hit_rate` | Tỷ lệ lượt có `true_p` trong [0,6; 0,8], tính trước khi trả lời |
| `mastery_gain` | θ sau lượt 50 trừ θ trước lượt 1, trên kỹ năng được luyện |
| `turns_to_mastery` | Lượt đầu tiên mà sau đó σ(a_m(θ − b_m)) ≥ 0,95 ở bài trung vị m (b trung vị) của kỹ năng; chưa đạt thì ghi 50 (`not_mastered` = 1); đạt từ đầu thì ghi 0 (`mastered_at_start` = 1) |

  **Lệch so với kế hoạch:** kế hoạch ghi "P đúng ở bài trung vị ≥ 0,95". Điều này không bao giờ xảy ra, vì P < 1 − s
  = 0,95. Ngưỡng vì vậy đặt trên phần của P không kể guess và slip, tương đương P ≥ 0,904 với bài điền và P ≥ 0,9125 với
  bài radio.
- **Báo cáo:** trung bình qua 5 seed, độ lệch chuẩn qua seed, khoảng 95% bootstrap theo học sinh (1.000 lần lấy lại
  học sinh trong từng seed, cùng chỉ số lấy mẫu cho mọi chính sách). Bảng chênh lệch so với `fixed` (H6) có thêm
  p bootstrap hai phía để P2-10 hiệu chỉnh Holm. Mọi bảng có cột `label` = "Kết quả mô phỏng", và tiêu đề hình bắt đầu
  bằng chữ đó.
- **Quy trình:** `--world tune` chạy trên world 2027, dùng để chỉnh tham số của `proposed`; bảng có hậu tố `_tune`.
  `--world final` chạy một lần trên world 2026 sau Mốc B, bắt buộc đủ bốn chính sách và không nhận `--policies` hay
  `--n-learners`. Chạy đủ mất khoảng 3,5 phút cho ba baseline và oracle.

### Chọn thiết kế (cách giao kỹ năng)

Kế hoạch không nói học sinh luyện kỹ năng nào. Bản đầu rút đều trong 83 kỹ năng, và 25,5% học sinh luyện một kỹ năng
đã thành thạo từ đầu. Với họ, hầu như không bài nào có P trong vùng ZPD, nên mọi chính sách đều trượt và chênh lệch bị
loãng; ngoài ra không ai giao luyện một kỹ năng đã thành thạo. `lmsrag/sim/design_check.py` so bốn cách trên world
2027 (2 seed; ba baseline và oracle; `proposed` không tham gia nên việc chọn không thể có lợi cho nó), theo quy tắc ghi
**trước** khi chạy:

1. hợp lệ: không học sinh nào luyện kỹ năng đã thành thạo từ đầu;
2. độ nhạy: (oracle − `fixed`) trúng ZPD chia cho nửa độ rộng khoảng 95% ghép cặp, trung bình qua ba quy tắc;
3. hòa (trong 10%) thì chọn cách đơn giản hơn.

| `skill_assignment` | Thành thạo từ đầu | `fixed` trúng ZPD (flat / info / zpd) | oracle trúng ZPD | Độ nhạy |
|---|---:|---|---|---:|
| `uniform` (bản đầu) | 26,2% | 0,17 / 0,22 / 0,18 | 0,45 / 0,52 / 0,45 | 25,9 |
| `junyi` (theo số lượt Junyi) | 54,8% | 0,09 / 0,12 / 0,10 | 0,26 / 0,31 / 0,27 | 17,6 |
| **`uniform_unmastered`** | **0%** | 0,23 / 0,29 / 0,22 | 0,59 / 0,67 / 0,57 | **34,8** |
| `junyi_unmastered` | 0% | 0,18 / 0,24 / 0,20 | 0,50 / 0,59 / 0,51 | 31,0 |

Chọn `uniform_unmastered`. Rút theo số lượt Junyi gần thực tế hơn, nhưng dồn học sinh vào các kỹ năng dễ, phổ biến
(55% thành thạo từ đầu), và kém nhạy hơn cả khi đã loại kỹ năng thành thạo. Độ nhạy của cả bốn cách đều dư cho H6
(khoảng cách oracle − `fixed` lớn gấp 18–35 lần nửa khoảng tin cậy), nên điểm quyết định thực ra là quy tắc 1.
Bảng đầy đủ: `c5_design_check_tune.csv`. Việc chọn dùng tham số thật để dựng thí nghiệm (θ trước lượt 1), giống giáo
viên giao bài; chính sách vẫn chỉ thấy `observe(u)`.

### Lần chạy trên world chỉnh tham số (2027), 20/11/2026

`c5_closed_loop_tune.csv`, 5 seed, khoảng 95% bootstrap theo học sinh. Đây là kết quả mô phỏng, chưa phải lần chạy
cuối, và chưa có `proposed`:

| Quy tắc | Chính sách | Trúng ZPD | Tăng θ | Lượt đến thành thạo | Chưa đạt |
|---|---|---|---|---|---:|
| flat | random | 0,226 [0,221; 0,231] | 2,71 [2,68; 2,73] | 34,2 [33,7; 34,6] | 36,8% |
| flat | fixed | 0,231 [0,226; 0,237] | 2,71 [2,68; 2,73] | 34,2 [33,7; 34,6] | 36,8% |
| flat | most_popular | 0,223 [0,218; 0,228] | 2,71 [2,68; 2,73] | 34,2 [33,7; 34,6] | 36,8% |
| flat | *oracle (trần)* | *0,585 [0,576; 0,594]* | *2,71 [2,68; 2,73]* | *34,2 [33,7; 34,6]* | *36,8%* |
| info | random | 0,290 [0,285; 0,296] | 1,52 [1,50; 1,54] | 42,3 [41,9; 42,6] | 67,3% |
| info | fixed | 0,295 [0,288; 0,301] | 1,51 [1,50; 1,53] | 42,2 [41,8; 42,5] | 67,2% |
| info | most_popular | 0,288 [0,282; 0,294] | 1,51 [1,50; 1,53] | 42,3 [42,0; 42,7] | 67,2% |
| info | *oracle (trần)* | *0,664 [0,655; 0,673]* | *1,92 [1,91; 1,94]* | *38,8 [38,4; 39,2]* | *53,1%* |
| zpd | random | 0,234 [0,228; 0,240] | 1,43 [1,41; 1,45] | 41,0 [40,6; 41,4] | 62,6% |
| zpd | fixed | 0,229 [0,223; 0,235] | 1,42 [1,41; 1,44] | 40,9 [40,5; 41,3] | 62,6% |
| zpd | most_popular | 0,232 [0,226; 0,237] | 1,43 [1,41; 1,44] | 41,0 [40,6; 41,4] | 62,7% |
| zpd | *oracle (trần)* | *0,571 [0,561; 0,580]* | *2,08 [2,06; 2,10]* | *36,5 [36,1; 36,9]* | *45,9%* |

- **Ba baseline gần như bằng nhau:** trúng ZPD chênh so với `fixed` không quá 0,8 điểm phần trăm. Với `flat`, tăng θ
  và số lượt giống hệt nhau ở mọi dòng, kể cả oracle, vì G = 1 nên θ tăng η_u mỗi lượt bất kể bài nào. Đây cũng là một
  kiểm tra của mã.
- **Trần:** oracle trúng ZPD 0,57–0,66, hơn `fixed` 0,34–0,37 (khoảng 95% rộng ±0,007). Trần thấp hơn 1 vì có kỹ năng
  mà không bài nào có P trong vùng: ở θ = μ_k, 77% kỹ năng có ít nhất một bài trong [0,6; 0,8], và P của các bài trong
  một kỹ năng trải rộng 0,21 (trung vị). Khi có `proposed`, báo cáo thêm phần khoảng cách nó lấp được:
  (proposed − fixed) / (oracle − fixed).
- **Lượt đến thành thạo là chỉ số yếu ở 50 lượt:** 37–67% học sinh chưa đạt, nên trung bình gần như là số lượt bị cắt
  ở 50. Kết luận nên dựa vào trúng ZPD và tăng θ; chỉ số này giữ theo kế hoạch, kèm tỷ lệ chưa đạt.

## Giới hạn

- **Độ khó công khai của bài bậc 2 và 3 được coi là đúng.** b của các bài này suy ra trực tiếp từ `difficulty`, nên
  trong mô phỏng `difficulty` sắp thứ tự b hoàn hảo. Trên người thật, P1-06 đo Spearman với điểm của người chỉ 0,59
  (Khan từ số bước) và 0,42 (GSM8K). Vì vậy, chính sách chọn bài theo `difficulty` sẽ trông tốt hơn trên bài bậc 2–3
  so với thực tế. Có thể cộng nhiễu vào b theo bậc; bản này chưa làm.
- Ở bài bậc 1, `p_correct_hist` tính trên chính history Junyi, nên b và μ_k dùng chung dữ liệu. Đây là hiệu chỉnh, không
  phải kiểm định độc lập.
- Kỹ năng độc lập: luyện kỹ năng này không làm tăng θ của kỹ năng khác, và không có quên.
- Khoảng guess 0,20 áp chung cho mọi bài radio, không theo số phương án.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only junyi2015   # P1-03, nếu chưa có parquet
python -m lmsrag.corpus.build_corpus --config configs/data.yaml               # P1-04
python -m lmsrag.sim.calibrate --config configs/sim.yaml --tune-eta           # chọn η, khoảng 15 giây (một lần)
python -m lmsrag.sim.calibrate --config configs/sim.yaml                      # P1-08, khoảng 20 giây
python -m lmsrag.sim.design_check --config configs/sim.yaml                    # P1-09: chọn cách giao kỹ năng, khoảng 5 phút
python -m lmsrag.sim.run --config configs/sim.yaml --world tune                 # P1-09, khoảng 3,5 phút
python -m lmsrag.sim.run --config configs/sim.yaml --world final                # P1-09, một lần sau Mốc B
python -m pytest lmsrag/tests/test_sim.py lmsrag/tests/test_sim_run.py -q
```
