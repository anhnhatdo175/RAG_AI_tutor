# Bộ học sinh mô phỏng (A6, P1-08)

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
python -m pytest lmsrag/tests/test_sim.py -q
```
