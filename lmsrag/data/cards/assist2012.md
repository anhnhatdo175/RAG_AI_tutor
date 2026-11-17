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
| Dòng (mỗi dòng một `problem_log_id`) | 6.123.270 | 2.148.066 |
| Người học | 46.674 | 16.887 |
| Bài (`problem_id`) | – | 38.096 |
| Kỹ năng | 198 chuỗi (196 sau khi bỏ khoảng trắng thừa) | 93 (81 tag chuẩn) |
| Thời gian | 01/09/2012 – 31/08/2013 | như bên trái |
| Tỷ lệ đúng | – | 0,703 |

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
| 6 | Bỏ dòng có kỹ năng không ánh xạ được tag chuẩn | 2.453.473 | 2.162.323 | 291.150 | 11,87 | 17.823 |
| 7 | Áp lại ngưỡng 20 dòng sau bước 6 | 2.162.323 | 2.148.066 | 14.257 | 0,66 | 16.887 |

Bước 6 loại 103 kỹ năng. Phần lớn chưa có quy tắc trong `SkillMapper` vì chỉ xuất hiện ở 2012, chẳng hạn
Finding y-intercept from Linear Equation (21.026 dòng), Substitution (19.920) và Combining Like Terms (18.212).
Từ điển v1 (P1-05) sẽ bổ sung các kỹ năng từ 1.000 dòng trở lên, sau đó chạy lại adapter này.

### Phần giữ so với phần bị loại ở bước 6

Bảng: `reports/tables/a1_kept_vs_dropped_assist2012.csv`. Tỷ lệ đúng phần giữ 0,703, phần loại 0,686. Chênh lệch
giữ − loại = +0,017, khoảng tin cậy 95% [0,013; 0,020] (bootstrap theo cụm người học, 1.000 lần, `data_seed` 2026).
Chiều chênh lệch ngược với 2009, vì ở đây phần bị loại chủ yếu là kỹ năng đại số chưa có quy tắc, không phải kỹ năng
đọc hình.

## Loại bài tập (cột `type`)

Bảng: `reports/tables/a1_sections_assist2012.csv`, tính sau làm sạch.

| `section_type` | Dòng | Người học | Tỷ lệ đúng |
|---|---:|---:|---:|
| MasterySection (Skill Builder) | 1.395.481 | 15.384 | 0,705 |
| LinearSection | 635.264 | 11.127 | 0,704 |
| RandomChildOrderSection | 111.775 | 4.188 | 0,679 |
| RandomIterateSection | 3.154 | 1.066 | 0,635 |
| PlacementsSection | 1.869 | 273 | 0,398 |
| ChooseConditionSection | 512 | 105 | 0,561 |
| NumericLimitSection | 11 | 2 | 0,545 |

**Tập con ngoài MasterySection** (mục A1 của cô): 752.585 dòng, 12.265 người học. Đây là phần không chịu quy tắc dừng
"đúng 3 câu liên tiếp". Lọc bằng `section_type != "MasterySection"` trên `interactions_assist2012.parquet`.

## Cách chia và seed

Giống 2009: theo từng người học, theo `ts`, tỷ lệ 70/15/15 theo sự kiện. Kết quả: history 1.511.165 dòng, validation
322.498, test 314.403. Hai lần chạy cho cùng một tệp.

## Đặc điểm cần biết khi đọc kết quả

- **`ts` là thời gian thật:** `start_time` đổi ra mili giây kể từ 1970. Khi hai bài có cùng `start_time`, thứ tự
  được cố định theo `problem_log_id`.
- **Skill Builder** (`reports/tables/a1_skill_builder_assist2012.csv`): trong MasterySection, 71,6% chuỗi kết thúc
  bằng 3 lượt đúng; ngoài MasterySection chỉ 24,1%. AUC của "Last Answer" trên test là 0,633 trong MasterySection
  và 0,646 ngoài MasterySection.
- **Tỷ lệ đúng theo phần:** 0,706 ở history, 0,694 ở validation, 0,698 ở test.
- **Chỉ 43% dòng gốc có kỹ năng.** Mọi kết luận về 2012 chỉ áp dụng cho phần có gắn kỹ năng.
- Cột cảm xúc `Average_confidence(...)` không được dùng.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only assist2012   # khoảng 1 phút, đọc theo khối 400.000 dòng
```
