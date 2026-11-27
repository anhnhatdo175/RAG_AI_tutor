# ASSISTments 2012–2013 School Data with Affect

- **Nguồn và đường dẫn tải:** ASSISTments, bộ "2012-13 School Data with Affect",
  https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect. Tệp dùng:
  `2012-2013-data-with-predictions-4-final.csv` (3,0 GB), lưu tại `lmsrag/data/raw/assistments_2012/` (không commit).
- **Ngày tải:** 16/11/2026 (đợt khảo sát dữ liệu).
- **SHA-256 tệp gốc:** `1D06AEE9E649C5BA9DB49052FE9A86E68D0C3CCB4709A7487289D7721F5464DB`
- **Giấy phép và trích dẫn bắt buộc:** chỉ dùng cho nghiên cứu, không phân phối lại; theo điều khoản trên trang
  ASSISTments. Bản đã làm sạch chỉ để trong Kaggle Dataset riêng tư.

## Quy mô

| | Trước làm sạch | Sau làm sạch |
|---|---:|---:|
| Dòng (mỗi dòng một `problem_log_id`) | 6.123.270 | 2.289.624 |
| Người học | 46.674 | 17.289 |
| Bài (`problem_id`) | 179.999 | 41.100 |
| Kỹ năng | 198 chuỗi (196 sau khi bỏ khoảng trắng thừa) | 109 (97 tag chuẩn) |
| Thời gian | 01/09/2012 – 31/08/2013 | như bên trái |
| Tỷ lệ đúng | – | 0,701 |

Số liệu tính với từ điển kỹ năng v1 (`skill_map_v1.csv`, P1-05). Với `SkillMapper` (trước P1-05): 2.148.066 dòng,
16.887 người học, 93 kỹ năng. Các số kỳ vọng của thẻ P1-02 (bản gốc, có kỹ năng, `original` = 1, 17.863 người học
trước ánh xạ) không phụ thuộc từ điển nên không đổi.

107 trên 110 tên kỹ năng của ASSISTments 2009 xuất hiện lại trong bộ này.

## Quy tắc loại bỏ

Bảng đầy đủ: `reports/tables/a1_cleaning_assist2012.csv`. Thứ tự giống ASSISTments 2009 (P1-01).

| Bước | Quy tắc | Dòng trước | Dòng sau | Bị loại | % | Người học sau |
|---|---|---:|---:|---:|---:|---:|
| 0 | Bản gốc | 6.123.270 | 6.123.270 | 0 | 0 | 46.674 |
| – | *Thông tin:* dòng có kỹ năng trên bản gốc | 6.123.270 | – | 2.630.080 | 42,95 | – |
| – | *Thông tin:* `problem_log_id` trùng trên bản gốc | 6.123.270 | – | 0 | 0 | – |
| 1 | Bỏ `original == 0` | 6.123.270 | 5.819.737 | 303.533 | 4,96 | 46.544 |
| 2 | Bỏ dòng không có kỹ năng | 5.819.737 | 2.543.887 | 3.275.850 | 56,29 | 28.812 |
| 3 | Bỏ `correct` lẻ (0,25; 0,5; 0,75) | 2.543.887 | 2.543.677 | 210 | 0,01 | 28.812 |
| 4 | Loại trùng theo `problem_log_id` | 2.543.677 | 2.543.677 | 0 | 0 | 28.812 |
| 5 | Bỏ người học dưới 20 dòng | 2.543.677 | 2.453.473 | 90.204 | 3,55 | 17.863 |
| 6 | Bỏ dòng có kỹ năng không có tag chuẩn trong `skill_map_v1.csv` | 2.453.473 | 2.298.220 | 155.253 | 6,33 | 17.825 |
| 7 | Áp lại ngưỡng 20 dòng sau bước 6 | 2.298.220 | 2.289.624 | 8.596 | 0,37 | 17.289 |

Bước 6 loại 87 kỹ năng, lớn nhất là Histogram as Table or Graph (12.329 dòng), Properties and Classification
Quadrilaterals (11.010) và Point Plotting (10.383): kỹ năng cần hình, không có template dùng được và không có quy
tắc văn bản. Với `SkillMapper`, bước này loại 103 kỹ năng và 291.150 dòng (11,87%); từ điển v1 thêm tag cho các kỹ
năng có template Khan đúng sau khi rà, như Finding y-intercept from Linear Equation (21.026 dòng), Substitution
(19.920), Combining Like Terms (18.212) và 4 kỹ năng mới của phiếu B.

### Phần giữ so với phần bị loại ở bước 6

Bảng: `reports/tables/a1_kept_vs_dropped_assist2012.csv`. Tỷ lệ đúng phần giữ 0,701, phần loại 0,700. Chênh lệch
giữ − loại = +0,001, khoảng tin cậy 95% [−0,004; 0,006] (bootstrap theo cụm người học, 1.000 lần, `data_seed` 2026):
**không có lệch**. Với `SkillMapper`, chênh lệch là +0,017 [0,013; 0,020], vì khi đó phần bị loại chủ yếu là kỹ năng
đại số chưa có quy tắc; từ điển v1 đã đưa các kỹ năng này trở lại.

## Loại bài tập (cột `type`)

Bảng: `reports/tables/a1_sections_assist2012.csv`, tính sau làm sạch.

| `section_type` | Dòng | Người học | Tỷ lệ đúng |
|---|---:|---:|---:|
| MasterySection (Skill Builder) | 1.476.345 | 15.700 | 0,703 |
| LinearSection | 686.338 | 11.525 | 0,703 |
| RandomChildOrderSection | 119.887 | 4.471 | 0,677 |
| RandomIterateSection | 4.660 | 1.324 | 0,657 |
| PlacementsSection | 1.869 | 273 | 0,398 |
| ChooseConditionSection | 514 | 106 | 0,562 |
| NumericLimitSection | 11 | 2 | 0,545 |

**Tập con ngoài MasterySection** (mục A1 của cô): 813.279 dòng, 12.714 người học. Đây là phần không chịu quy tắc dừng
"đúng 3 câu liên tiếp". Lọc bằng `section_type != "MasterySection"` trên `interactions_assist2012.parquet`.

## Cách chia và seed

Giống 2009: theo từng người học, theo `ts`, tỷ lệ 70/15/15 theo sự kiện. Kết quả: history 1.610.433 dòng, validation
343.747, test 335.444. Hai lần chạy cho cùng một tệp.

## Đặc điểm cần biết khi đọc kết quả

- **`ts` là thời gian thật:** `start_time` đổi ra mili giây kể từ 1970. Khi hai bài có cùng `start_time`, thứ tự
  được cố định theo `problem_log_id`.
- **Skill Builder** (`reports/tables/a1_skill_builder_assist2012.csv`): trong MasterySection, 70,8% chuỗi kết thúc
  bằng 3 lượt đúng; ngoài MasterySection chỉ 23,7%. AUC của "Last Answer" trên test là 0,633 trong MasterySection
  và 0,645 ngoài MasterySection.
- **Tỷ lệ đúng theo phần:** 0,704 ở history, 0,694 ở validation, 0,695 ở test.
- **Chỉ 43% dòng gốc có kỹ năng.** Mọi kết luận về 2012 chỉ áp dụng cho phần có gắn kỹ năng.
- Cột cảm xúc `Average_confidence(...)` không được dùng.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only assist2012   # khoảng 1 phút, đọc theo khối 400.000 dòng
```

**Data v1** (tag `data-v1`): mã băm các tệp đầu ra nằm trong `lmsrag/data/cards/data_v1.sha256`; kiểm bằng
`sha256sum -c lmsrag/data/cards/data_v1.sha256` từ gốc repo. Bản này build bằng pandas 3.0.6, pyarrow 26.0.1, numpy
2.4.6; build lại ngày 27/11/2026 ra đúng từng byte, kể cả các bảng trong `reports/tables/`. Với phiên bản thư viện khác,
parquet sẽ khác byte (vì ghi phiên bản pandas vào metadata), nên khi đó phải so nội dung.
