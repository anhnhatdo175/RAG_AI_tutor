# ASSISTments 2009–2010 Skill Builder

- **Nguồn và đường dẫn tải:** ASSISTments, bộ "2009-2010 ASSISTment Data – Skill Builder",
  https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data. Tệp dùng: `skill_builder_data.csv`
  (bản gốc, chưa phải bản `_corrected`), lưu tại `lmsrag/data/assistments/skill_builder_data.csv`.
- **Ngày tải:** tệp có trong repo từ 25/09/2026 (commit `955ad28`). *Ngày tải gốc: Người 1 điền.*
- **SHA-256 tệp gốc:** `F22E3FB7872C1784CE93B0F9EBABBE0CBCAC4F896FD8B4A11667B9715D77DBDC`
- **Giấy phép và trích dẫn bắt buộc:** dữ liệu được ASSISTments cho phép dùng cho nghiên cứu, không phân phối lại.
  Trích dẫn: Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge with an online
  system that tutors as it assesses. *User Modeling and User-Adapted Interaction, 19*(3), 243–266.
  Vì vậy bộ đã làm sạch chỉ để trong Kaggle Dataset riêng tư.

## Quy mô

| | Trước làm sạch | Sau làm sạch |
|---|---:|---:|
| Dòng | 525.534 | 242.764 |
| Sự kiện (`order_id` duy nhất) | 346.860 | 224.106 |
| Người học | 4.217 | 2.065 |
| Bài (`problem_id`) | 26.688 | 14.363 |
| Kỹ năng (`skill_name`) | 110 | 96 (83 tag chuẩn) |
| Tỷ lệ đúng | – | 0,656 |

## Quy tắc loại bỏ

Bảng đầy đủ: `reports/tables/a1_cleaning_assist2009.csv`. Thứ tự giữ đúng như pipeline cũ
(`lmsrag/data.py`: `read_assistments` → `finalize_logs` → `filter_mapped_logs`) để số liệu so được với báo cáo trước.

| Bước | Quy tắc | Dòng trước | Dòng sau | Bị loại | % | Người học sau |
|---|---|---:|---:|---:|---:|---:|
| 0 | Bản gốc | 525.534 | 525.534 | 0 | 0 | 4.217 |
| – | *Thông tin:* dòng trùng theo (`order_id`, `skill_id`) trên bản gốc | 525.534 | – | 123.778 | 23,55 | – |
| 1 | Bỏ `original == 0` (câu hỏi phụ) | 525.534 | 449.220 | 76.314 | 14,52 | 4.217 |
| 2 | Bỏ thiếu `skill_id` hoặc `skill_name` | 449.220 | 421.318 | 27.902 | 6,21 | 4.151 |
| 3 | Bỏ `correct` ngoài {0, 1} | 421.318 | 421.318 | 0 | 0 | 4.151 |
| 4 | Loại trùng theo (`order_id`, `skill_id`) | 421.318 | 300.389 | 120.929 | 28,70 | 4.151 |
| 5 | Bỏ người học dưới 20 dòng | 300.389 | 283.687 | 16.702 | 5,56 | 2.219 |
| 6 | Bỏ dòng có kỹ năng không ánh xạ được tag chuẩn | 283.687 | 245.048 | 38.639 | 13,62 | 2.219 |
| 7 | Áp lại ngưỡng 20 dòng sau bước 6 | 245.048 | 242.764 | 2.284 | 0,93 | 2.065 |

Bước 7 là bước mới so với pipeline cũ: sau khi bỏ dòng không ánh xạ, 154 người học còn dưới 20 dòng. Ở bước 4,
số bị loại (120.929) nhỏ hơn 123.778 vì một phần dòng trùng đã bị bỏ ở bước 1–2.

### Phần giữ so với phần bị loại ở bước 6

Bảng: `reports/tables/a1_kept_vs_dropped.csv`.

- Tỷ lệ đúng phần giữ 0,656 (245.048 dòng, 2.219 người học); phần bị loại 0,692 (38.639 dòng, 1.382 người học).
- Chênh lệch giữ − loại = **−0,036**, khoảng tin cậy 95% [−0,045; −0,027] (bootstrap theo cụm người học, 1.000 lần,
  `data_seed` 2026). Chi-bình phương χ² = 196,5, p < 10⁻⁴³; kiểm định này tính trên dòng nên coi các dòng độc lập,
  chỉ dùng để tham khảo, còn khoảng bootstrap mới là căn cứ.
- **Kết luận: có lệch.** Bỏ 14 kỹ năng đọc hình làm tỷ lệ đúng của phần còn lại thấp hơn khoảng 3,6 điểm phần trăm.
  Lệch đến từ các kỹ năng dễ: Scatter Plot (0,880), Box and Whisker (0,748), Table (0,742), Venn Diagram (0,718).
  Hai kỹ năng khó hơn mức chung là Circle Graph (0,526) và Number Line (0,595).
- Phân bố tỷ lệ đúng theo người học (người có ít nhất 5 dòng trong phần đó): trung vị 0,655 ở phần giữ và
  0,714 ở phần loại.

## Cách chia và seed

Theo từng người học, theo thứ tự `ts`: 70% sự kiện đầu là `history`, 15% tiếp là `validation`, 15% cuối là
`test` (`lmsrag/datasets/splits.py`). Đơn vị chia là sự kiện (`user_id`, `ts`), không phải dòng, để các dòng của
cùng một câu nhiều kỹ năng luôn nằm cùng một phần. Kết quả: history 171.138 dòng, validation 36.121, test 35.505.
`data_seed` = 2026, dùng cho bootstrap; việc chia không có thành phần ngẫu nhiên. Hai lần chạy cho cùng một tệp.

## Đặc điểm cần biết khi đọc kết quả

- **`ts` không phải thời gian.** Đây là `order_id`, chỉ cho biết thứ tự.
- **100% dòng thuộc MasterySection (Skill Builder).** Hệ thống giao bài cùng kỹ năng cho đến khi học sinh đúng 3 câu
  liên tiếp, nên chuỗi (người học, kỹ năng) thường kết thúc bằng lượt đúng. Baseline "Last Answer on Skill" vì vậy
  được lợi một cách nhân tạo. Số đo (`reports/tables/a1_skill_builder.csv`):
  - 29.976 chuỗi, độ dài trung vị 5;
  - 82,7% chuỗi kết thúc bằng lượt đúng;
  - 45,5% chuỗi kết thúc bằng 3 lượt đúng liên tiếp (65,3% nếu chỉ xét chuỗi dài từ 3 lượt);
  - tỷ lệ đúng tăng từ 0,592 ở lượt đầu lên khoảng 0,69 ở lượt 4–5;
  - AUC của "Last Answer on Skill" là 0,649 trên mọi dòng có lượt trước và 0,651 trên test.

  So sánh với tập con ngoài MasterySection của ASSISTments 2012 sẽ được bổ sung sau P1-02.
- **Tỷ lệ đúng giảm dần theo phần:** 0,661 ở history, 0,654 ở validation, 0,632 ở test. Học sinh gặp bài khó hơn về
  sau, nên mọi so sánh phải dùng cùng một phần dữ liệu.
- **Thời gian phản hồi:** 0,11% dòng có `rt_ms` thiếu (giá trị gốc ≤ 0); các giá trị còn lại được cắt về
  [500; 120.000] ms.
- **Ánh xạ kỹ năng:** tag lấy từ `SkillMapper` hiện tại. Sẽ chạy lại khi có từ điển `skill_map_v1.csv` (P1-05).

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only assist2009   # khoảng 7 giây
```

Đầu ra: `lmsrag/artifacts/v2/interactions_assist2009.parquet` cùng ba bảng `reports/tables/a1_cleaning_assist2009.csv`,
`a1_kept_vs_dropped.csv` và `a1_skill_builder.csv`.
