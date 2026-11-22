# Kho ngữ liệu v2 (Khan, GSM8K, MATH)

- **Nguồn:**
  - Khan Academy: các template của khan-exercises tại commit `001b27a5^` (tháng 5/2016), sinh bài bằng
    `lmsrag/data_tools/render_khan.py` và lọc bằng `build_khan_corpus.py`. Đầu vào: `lmsrag/data/khan_v2/khan_problems.jsonl`.
  - GSM8K: train và test, tải bằng `python -m lmsrag.data_tools.download_data`.
  - MATH: tập test, 4 môn Prealgebra, Algebra, Number Theory, Counting & Probability, Level 1–3, bỏ bài có `[asy]`.
- **Giấy phép:** Khan CC BY-NC-SA 3.0 (chỉ phi thương mại, phải ghi nguồn, chia sẻ cùng giấy phép); GSM8K MIT; MATH MIT.
  Trường `licence` có ở mọi chunk.
- **Đầu ra (không commit, sinh lại bằng lệnh bên dưới):** `lmsrag/artifacts/v2/corpus_chunks.jsonl`,
  `lmsrag/artifacts/v2/exercise_bank_v2.jsonl`. Báo cáo: `reports/tables/a3_corpus_summary.csv`,
  `reports/figures/a3_difficulty_by_source.png`.

## Quy mô

| Nguồn | Bài (tài liệu) | Chunk `worked_example` | Chunk `hint_step` | Kho bài tập |
|---|---:|---:|---:|---:|
| Khan | 2.064 (283 template) | 2.349 | 10.229 | 2.064 |
| GSM8K | 1.892 | 1.892 | – | 1.892 |
| MATH | 452 | 461 | – | – |
| **Tổng** | **4.408** | **4.702** | **10.229** | **3.956** |

Số liệu theo từ điển v1.1 (`map_version` 1.1, sau vòng κ đầu). Với v1.0: GSM8K 1.983, MATH 501 bài; xem mục Kỹ năng.

- Mỗi bài là một chunk `worked_example` có `text = "Problem: …\nSolution: …\nAnswer: …"`. Bài dài hơn 512 token
  (đếm bằng tokenizer của `BAAI/bge-small-en-v1.5`, kể cả token đặc biệt) được cắt bằng `layer2_embedding/chunk.py`:
  176 bài (171 Khan, 5 MATH), tức 4,0% số bài. Sau khi cắt, chunk dài nhất có đúng 512 token.
- Mỗi bước gợi ý của Khan là một chunk `hint_step`, chỉ dùng cho hội thoại Socratic, không dùng trong đánh giá truy xuất.
  12 bước dài quá 512 token cũng được cắt, có id dạng `#h3.0`, `#h3.1`.
- `chunk_id` = `<doc_id>#<số>`; vì `doc_id` đã có tên nguồn ở đầu (`khan:`, `gsm8k:`, `math:`), id này tương đương
  `<source>:<…>#<số>` trong lược đồ. Id của MATH giữ dạng của bộ phân tích cũ, ví dụ `math:test_prealgebra_126`.
  Hai lần build cho cùng tệp, nên id không đổi.

### Lọc GSM8K và MATH

| Bước | GSM8K | MATH |
|---|---:|---:|
| Đầu vào | 8.792 | 5.000 |
| Đúng 4 môn và Level 1–3 | – | 1.568 |
| Bỏ bài có `[asy]` | – | 1.479 |
| `clean_records` (chuẩn hóa, khử trùng chính xác và gần trùng MinHash) | gộp: 10.271 → 10.265 | |
| Không gắn được kỹ năng bằng `SkillMapper.tag_text`, bỏ | 6.899 | 1.022 |
| **Còn lại** | **1.892** | **452** |

78% bài GSM8K không gắn được kỹ năng: quy tắc hiện tại chỉ nhận dạng các dạng bài gần với kỹ năng ASSISTments. Khử
gần trùng chỉ chạy trên GSM8K và MATH; với Khan thì không, vì các bài cùng template chỉ khác số và sẽ bị xóa nhầm.

## Kỹ năng

- **Khan:** tra ngược từ điển kỹ năng v1 `lmsrag/data/mappings/skill_map_v1.csv` (P1-05, cột `canonical_tag`). Một
  template có thể thuộc nhiều kỹ năng: `skill_tags` là danh sách, còn `skill_tag` là phần tử đầu theo thứ tự chữ cái.
  Template không có trong từ điển dùng `junyi:<topic>`, giống quy tắc của adapter Junyi, để chunk và lượt làm Junyi có
  cùng tag. Nếu vẫn không có, dùng quy tắc văn bản; cuối cùng là `khan:<slug>`. Từ điển v1 cũng liệt kê 32 template đã
  rà là không thuộc kỹ năng ASSISTments nào (ma trận, hóa học, lượng giác, phân thức đại số…), với tag `khan:<slug>`.
  Từ v1.1, một cặp có thể chỉ phủ một số dạng bài của template (cột `khan_types`), nên tag của bài Khan tính theo
  (template, dạng bài): ví dụ `mean_median_and_mode` dạng `median` chỉ mang tag `Median`.
  Kết quả: từ điển 1.302 bài, Junyi topic 562, `khan:<slug>` 184 (`template_only`), quy tắc 16. Bài quy tắc và một
  phần `template_only` thuộc các template mà người rà đã bỏ khỏi kỹ năng duy nhất của chúng (`comparing_whole_numbers`,
  `comparing_whole_numbers_2`, `multiplying_polynomials_0.5`; từ v1.1 thêm `integer_sums`, `geometric_sequences_2`,
  `converting_between_point_slope_and_slope_intercept`, `simplifying_expressions_with_exponents`, `exponent_rules`).
  Với v1.0: từ điển 1.338, Junyi topic 534, quy tắc 8.
  Bản nháp (`assistments_to_khan_draft.csv`) cho: từ điển 1.195, Junyi topic 586, quy tắc 205, `khan:<slug>` 78.
- **GSM8K và MATH:** `SkillMapper.tag_text`. Vòng κ đầu cho thấy quy tắc chỉ đúng 62% (31/50 cặp), nên v1.1 thu hẹp
  các mẫu bắt theo từ khóa (bỏ câu định dạng đáp số "Express your answer as…" trước khi so; "average" phải là điều
  được hỏi; làm tròn phải là yêu cầu chính; phân số có chữ ở mẫu không phải cộng trừ phân số; một phép tính đơn lẻ
  không phải thứ tự phép tính…) và tách tag của ba kỹ năng trước dùng chung tag với kỹ năng khác
  (`Order of Operations +,-,/,* () positive reals`, `Simplifying Expressions positive exponents`,
  `Finding Slope from Ordered Pairs`). Trên 50 cặp của mẫu κ: 18/19 cặp sai đổi tag, cặp còn lại hết sai nhờ tách tag;
  2/31 cặp đúng đổi sang tag cũng hợp lệ. Trên toàn bộ 10.265 bài: 179 bài đổi tag, 140 bài mất tag (chủ yếu
  GSM8K chỉ cho sẵn một giá trị trung bình). Độ chính xác của quy tắc mới phải đo lại trên một mẫu mới.
- Có 144 tag có chunk (85 tag chuẩn); 49 tag có chunk ở ít nhất 3 mức khó (37 tag chuẩn); v1.0: 139 (80), 50 (38). Số tag tăng và số tag đủ 3
  mức giảm so với bản nháp (105/65, 54/40) vì các template trước đây bị quy tắc gắn chung vào một tag nay mỗi template
  là một tag `khan:<slug>`.

**Độ phủ ASSISTments** (mẫu số: các lượt sau làm sạch, trước bước lọc kỹ năng; bảng `a1_skill_counts_*.csv`):

| | Kỹ năng có ít nhất một template Khan dùng được | Tag của kỹ năng có ít nhất một chunk (mọi nguồn) |
|---|---:|---:|
| ASSISTments 2009, từ điển v1.1 | **76,8%** | **81,9%** |
| ASSISTments 2012, từ điển v1.1 | **81,5%** | **88,3%** |
| ASSISTments 2009, từ điển v1.0 | 77,1% | 81,6% |
| ASSISTments 2012, từ điển v1.0 | 82,4% | 87,1% |
| ASSISTments 2009, bản nháp | 80,0% | 80,9% |
| ASSISTments 2012, bản nháp | 84,1% | 82,4% |

Tên kỹ năng được so không phân biệt hoa thường và khoảng trắng: log 2009 có tên như `Circumference ` (dấu cách cuối).
Lần đo đầu ở P1-04 so chính xác từng ký tự nên cho 77,6% với bản nháp; số đúng là 80,0%, sát số kỳ vọng 80,4% của kế
hoạch. 0,4 điểm còn lại có thể đến từ việc siết bộ lọc hình (`MANUAL_FIGURE`, 9 template hình học) sau đợt khảo sát,
vốn ảnh hưởng tới 0,8 điểm; chưa tách riêng được.

Cột thứ nhất giảm so với bản nháp vì người rà bỏ các cặp mà template không luyện đúng kỹ năng, khiến 5 kỹ năng không
còn template dùng được: Ordering Integers (1,19% lượt 2009 / 0,80% lượt 2012), Unit Rate (1,06% / 0,97%), Estimation
(0,56% / 0,11%), Rate (0,03% / 0,08%) và Interpreting Coordinate Graphs (0,01% / 0,14%). Bốn kỹ năng mới của 2012
(phiếu B) thêm 0,37% lượt 2012. Cột thứ hai tăng vì adapter giờ lấy tag từ từ điển v1, nên các kỹ năng có template
đúng nhưng `SkillMapper` không có quy tắc (ví dụ Substitution, Combining Like Terms) có tag và có chunk.

## Độ khó (`lmsrag/corpus/difficulty.py`)

| `difficulty_source` | Bài | Độ khó gốc |
|---|---:|---|
| `junyi_history` | 1.273 Khan (172 template) | 1 − tỷ lệ đúng trên phần **history** của mẫu Junyi. Có 920 bài tính theo (template, dạng bài) khi đơn vị có từ 200 lượt, co về tỷ lệ đúng của template với m = 20; 353 bài tính theo mức template khi chỉ template đủ 200 lượt. Trường `p_correct_hist`, `difficulty_n` |
| `khan_steps` | 791 Khan | hồi quy isotonic (tăng dần) từ số bước gợi ý sang độ khó gốc, fit trên 1.273 bài ở dòng trên: 1 bước → 0,19; 5 bước → 0,29; 9 bước → 0,42; từ 11 bước → 0,45 |
| `gsm8k_steps` | 1.892 | số phép tính `<<…>>` trong lời giải gốc |
| `math_level` | 452 | Level 1–3 |

- `difficulty` = hạng phần trăm giữa (mid-rank) trong từng nguồn, nên mỗi nguồn trải đều trên (0, 1).
- `difficulty_level` = ngũ phân vị của `difficulty` trên toàn kho. Vì GSM8K và MATH có nhiều giá trị trùng, GSM8K không
  có bài ở mức 4 và MATH chỉ có các mức 1, 2, 4 (hình `a3_difficulty_by_source.png`).
- **Giả định đã kiểm (P1-06, `a3_difficulty_BIEN_BAN.md`):** coi hạng trong từng nguồn là so sánh được giữa các nguồn.
  Hai người chấm 100 bài (κ có trọng số 0,89). Spearman giữa `difficulty` và điểm của người: Khan `junyi_history`
  0,81 [0,66; 0,88], Khan `khan_steps` 0,59 [0,19; 0,82], GSM8K 0,42 [0,06; 0,74], MATH 0,47 [0,05; 0,75]. Cùng một
  `difficulty_level`, bài GSM8K và MATH khó hơn bài Khan khoảng 1 điểm (thang 1–5) ở các mức thấp, và năm mức của GSM8K
  gần như không phân tầng (2,25 → 2,83). Giả định **không đứng**: chỉ nên so độ khó trong cùng một nguồn, hoặc hiệu
  chỉnh thang chung trên một mẫu riêng.
- **Cách dùng độ khó (quyết định sau P1-06):** mỗi chunk và mỗi bài tập có `difficulty_tier` (1: Khan từ Junyi, 1.273 bài;
  2: Khan từ số bước gợi ý, 791; 3: GSM8K và MATH, 2.344) và `difficulty_reliable` (bậc 1). Chọn bài theo mức khó ưu
  tiên bậc 1, rồi bậc 2, rồi bậc 3; không so độ khó giữa hai nguồn. Kỹ năng chuẩn có bài ở ≥ 3 mức trong cùng một nguồn:
  17 (bậc 1), 21 (tới bậc 2), 33 (tới bậc 3). Thử thêm người học Junyi ngoài mẫu không cải thiện độ khớp với người nên
  không dùng (`difficulty.junyi_outside_sample: false`).
- **Đáp án dạng căn:** 54 bài Khan trong kho bài tập có `answer_type` `radical` hoặc `cuberoot`; theo quy ước của Khan,
  `answer` chỉ lưu số dưới dấu căn ("147" nghĩa là √147 = 7√3). Bộ chấm phải dựng lại dấu căn trước khi so.
- **Bộ lọc Khan (P1-06):** 92,0% [81,2; 96,9] bài làm được mà không cần hình; 100% [92,9; 100] đáp án lưu đúng.
- Thống kê `junyi_*_survey.csv` (tính trên toàn log) **không** được dùng ở đây.

## Kho bài tập

`exercise_bank_v2.jsonl`: bài Khan và GSM8K có đáp án chấm tự động, cùng trường với chunk nhưng `text` chỉ gồm đề
(`chunk_id` = `<doc_id>#q`). Đáp án Khan chấm theo `answer_type`; GSM8K là số.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml   # cần interactions_junyi2015.parquet và a1_skill_counts_*
python -m lmsrag.corpus.build_corpus --config configs/data.yaml   # khoảng 35 giây
```
