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
| GSM8K | 1.983 | 1.983 | – | 1.983 |
| MATH | 501 | 510 | – | – |
| **Tổng** | **4.548** | **4.842** | **10.229** | **4.047** |

- Mỗi bài là một chunk `worked_example` có `text = "Problem: …\nSolution: …\nAnswer: …"`. Bài dài hơn 512 token
  (đếm bằng tokenizer của `BAAI/bge-small-en-v1.5`, kể cả token đặc biệt) được cắt bằng `layer2_embedding/chunk.py`:
  176 bài (171 Khan, 5 MATH), tức 3,9% số bài. Sau khi cắt, chunk dài nhất có đúng 512 token.
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
| Không gắn được kỹ năng bằng `SkillMapper.tag_text`, bỏ | 6.808 | 973 |
| **Còn lại** | **1.983** | **501** |

77% bài GSM8K không gắn được kỹ năng: quy tắc hiện tại chỉ nhận dạng các dạng bài gần với kỹ năng ASSISTments. Khử
gần trùng chỉ chạy trên GSM8K và MATH; với Khan thì không, vì các bài cùng template chỉ khác số và sẽ bị xóa nhầm.

## Kỹ năng

- **Khan:** tra ngược từ điển ASSISTments → Khan (hiện dùng bản nháp `assistments_to_khan_draft.csv`). Một template có
  thể thuộc nhiều kỹ năng: `skill_tags` là danh sách, còn `skill_tag` là phần tử đầu theo thứ tự chữ cái. Template không
  có trong từ điển dùng `junyi:<topic>`, giống quy tắc của adapter Junyi, để chunk và lượt làm Junyi có cùng tag. Nếu
  vẫn không có, dùng quy tắc văn bản; cuối cùng là `khan:<slug>`.
  Kết quả: từ điển 1.195 bài, Junyi topic 586, quy tắc 205, `khan:<slug>` 78 (10 template, sẽ rà ở P1-05).
- **GSM8K và MATH:** `SkillMapper.tag_text`, như pipeline cũ.
- Có 105 tag có chunk (65 tag chuẩn); 54 tag có chunk ở ít nhất 3 mức khó (40 tag chuẩn).

**Độ phủ ASSISTments** (mẫu số: các lượt sau làm sạch, trước bước lọc kỹ năng; bảng `a1_skill_counts_*.csv`):

| | Kỹ năng có ít nhất một template Khan dùng được | Tag của kỹ năng có ít nhất một chunk (mọi nguồn) |
|---|---:|---:|
| ASSISTments 2009 | 77,6% | 80,9% |
| ASSISTments 2012 | 84,1% | 82,4% |

Số kỳ vọng ghi trong kế hoạch là 80,4% (2009) và 84,1% (2012). Con số 2012 khớp. Với 2009, đã thử các mẫu số khác
(bản gốc 73,5%, trước loại trùng 73,2%, sau loại trùng 78,1%, theo sự kiện 86,4%) nhưng không mẫu số nào cho 80,4%.
Việc siết bộ lọc hình sau đợt khảo sát (`MANUAL_FIGURE`, 9 template hình học) chỉ giải thích được 0,8 điểm. Bảng
dùng định nghĩa ghi trên. Con số sẽ thay đổi sau khi có từ điển v1.

## Độ khó (`lmsrag/corpus/difficulty.py`)

| `difficulty_source` | Bài | Độ khó gốc |
|---|---:|---|
| `junyi_history` | 1.273 Khan (172 template) | 1 − tỷ lệ đúng trên phần **history** của mẫu Junyi. Có 920 bài tính theo (template, dạng bài) khi đơn vị có từ 200 lượt, co về tỷ lệ đúng của template với m = 20; 353 bài tính theo mức template khi chỉ template đủ 200 lượt. Trường `p_correct_hist`, `difficulty_n` |
| `khan_steps` | 791 Khan | hồi quy isotonic (tăng dần) từ số bước gợi ý sang độ khó gốc, fit trên 1.273 bài ở dòng trên: 1 bước → 0,19; 5 bước → 0,29; 9 bước → 0,42; từ 11 bước → 0,45 |
| `gsm8k_steps` | 1.983 | số phép tính `<<…>>` trong lời giải gốc |
| `math_level` | 501 | Level 1–3 |

- `difficulty` = hạng phần trăm giữa (mid-rank) trong từng nguồn, nên mỗi nguồn trải đều trên (0, 1).
- `difficulty_level` = ngũ phân vị của `difficulty` trên toàn kho. Vì GSM8K và MATH có nhiều giá trị trùng, GSM8K không
  có bài ở mức 4 và MATH chỉ có các mức 1, 2, 4 (hình `a3_difficulty_by_source.png`).
- **Giả định cần kiểm:** coi hạng trong từng nguồn là so sánh được giữa các nguồn. Giả định này được kiểm ở P1-06 bằng
  Spearman với đánh giá của người.
- Thống kê `junyi_*_survey.csv` (tính trên toàn log) **không** được dùng ở đây.

## Kho bài tập

`exercise_bank_v2.jsonl`: bài Khan và GSM8K có đáp án chấm tự động, cùng trường với chunk nhưng `text` chỉ gồm đề
(`chunk_id` = `<doc_id>#q`). Đáp án Khan chấm theo `answer_type`; GSM8K là số.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml   # cần interactions_junyi2015.parquet và a1_skill_counts_*
python -m lmsrag.corpus.build_corpus --config configs/data.yaml   # khoảng 35 giây
```
