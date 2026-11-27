# ASSISTments 2009–2010 Skill Builder

- **Nguồn và đường dẫn tải:** ASSISTments, bộ "2009-2010 ASSISTment Data – Skill Builder",
  https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data. Tệp dùng: `skill_builder_data.csv`
  (bản gốc, chưa phải bản `_corrected`), lưu tại `lmsrag/data/assistments/skill_builder_data.csv`.
- **Ngày tải:** 25/09/2026 (commit `955ad28`).
- **SHA-256 tệp gốc:** `F22E3FB7872C1784CE93B0F9EBABBE0CBCAC4F896FD8B4A11667B9715D77DBDC` với xuống dòng CRLF, tức bản
  checkout trên Windows khi `core.autocrlf=true`. Git lưu tệp với LF, nên bản checkout trên Linux hoặc Kaggle có
  SHA-256 `408698925FC3488672431E4C19B9B16808B6A25760F8E41AD4D3D6257AD0832A`. Hai bản chỉ khác ký tự xuống dòng và
  `pandas.read_csv` đọc ra cùng một bảng (525.534 × 30).
- **Giấy phép và trích dẫn bắt buộc:** dữ liệu được ASSISTments cho phép dùng cho nghiên cứu, không phân phối lại.
  Trích dẫn: Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge with an online
  system that tutors as it assesses. *User Modeling and User-Adapted Interaction, 19*(3), 243–266.
  Vì vậy bộ đã làm sạch chỉ để trong Kaggle Dataset riêng tư.

## Quy mô

| | Trước làm sạch | Sau làm sạch |
|---|---:|---:|
| Dòng | 525.534 | 248.060 |
| Sự kiện (`order_id` duy nhất) | 346.860 | 225.423 |
| Người học | 4.217 | 2.078 |
| Bài (`problem_id`) | 26.688 | 14.555 |
| Kỹ năng (`skill_name`) | 110 | 97 (84 tag chuẩn) |
| Tỷ lệ đúng | – | 0,656 |

Số liệu tính với từ điển kỹ năng v1 (`skill_map_v1.csv`, P1-05). Với `SkillMapper` (trước P1-05): 242.764 dòng,
2.065 người học, 38.639 dòng không ánh xạ (13,62%), khớp số kỳ vọng của thẻ P1-01.

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
| 6 | Bỏ dòng có kỹ năng không có tag chuẩn trong `skill_map_v1.csv` | 283.687 | 250.183 | 33.504 | 11,81 | 2.219 |
| 7 | Áp lại ngưỡng 20 dòng sau bước 6 | 250.183 | 248.060 | 2.123 | 0,85 | 2.078 |

Bước 7 là bước mới so với pipeline cũ: sau khi bỏ dòng không ánh xạ, 141 người học còn dưới 20 dòng. Ở bước 4,
số bị loại (120.929) nhỏ hơn 123.778 vì một phần dòng trùng đã bị bỏ ở bước 1–2.

### Phần giữ so với phần bị loại ở bước 6

Bảng: `reports/tables/a1_kept_vs_dropped.csv`.

- Tỷ lệ đúng phần giữ 0,656 (250.183 dòng, 2.219 người học); phần bị loại 0,696 (33.504 dòng, 1.293 người học).
- Chênh lệch giữ − loại = **−0,039**, khoảng tin cậy 95% [−0,048; −0,029] (bootstrap theo cụm người học, 1.000 lần,
  `data_seed` 2026). Chi-bình phương χ² = 202,3, p < 10⁻⁴⁵; kiểm định này tính trên dòng nên coi các dòng độc lập,
  chỉ dùng để tham khảo, còn khoảng bootstrap mới là căn cứ.
- **Kết luận: có lệch.** Bỏ 13 kỹ năng (phần lớn là kỹ năng đọc hình) làm tỷ lệ đúng của phần còn lại thấp hơn khoảng
  3,9 điểm phần trăm. Lệch đến từ các kỹ năng dễ: Nets of 3D Figures (0,972), Congruence (0,894), Scatter Plot (0,880),
  Estimation (0,791), Box and Whisker (0,748), Table (0,742). Kỹ năng khó hơn mức chung: Circle Graph (0,526),
  Interpreting Coordinate Graphs (0,531) và Number Line (0,595).
- Phân bố tỷ lệ đúng theo người học (người có ít nhất 5 dòng trong phần đó): trung vị 0,655 ở phần giữ và
  0,717 ở phần loại.

## Cách chia và seed

Theo từng người học, theo thứ tự `ts`: 70% sự kiện đầu là `history`, 15% tiếp là `validation`, 15% cuối là
`test` (`lmsrag/datasets/splits.py`). Đơn vị chia là sự kiện (`user_id`, `ts`), không phải dòng, để các dòng của
cùng một câu nhiều kỹ năng luôn nằm cùng một phần. Kết quả: history 174.687 dòng, validation 36.985, test 36.388.
`data_seed` = 2026, dùng cho bootstrap; việc chia không có thành phần ngẫu nhiên. Hai lần chạy cho cùng một tệp.

## Đặc điểm cần biết khi đọc kết quả

- **`ts` không phải thời gian.** Đây là `order_id`, chỉ cho biết thứ tự.
- **100% dòng thuộc MasterySection (Skill Builder).** Hệ thống giao bài cùng kỹ năng cho đến khi học sinh đúng 3 câu
  liên tiếp, nên chuỗi (người học, kỹ năng) thường kết thúc bằng lượt đúng. Giả thuyết ban đầu là baseline
  "Last Answer on Skill" được lợi nhân tạo; số đo bên dưới và so sánh với 2012 **không ủng hộ** giả thuyết này.
  Số đo (`reports/tables/a1_skill_builder.csv`):
  - 30.413 chuỗi, độ dài trung vị 5;
  - 82,5% chuỗi kết thúc bằng lượt đúng;
  - 45,5% chuỗi kết thúc bằng 3 lượt đúng liên tiếp (65,3% nếu chỉ xét chuỗi dài từ 3 lượt);
  - tỷ lệ đúng tăng từ 0,590 ở lượt đầu lên khoảng 0,69 ở lượt 4–5;
  - AUC của "Last Answer on Skill" là 0,650 trên mọi dòng có lượt trước và 0,652 trên test.

  So với ASSISTments 2012 (`reports/tables/a1_skill_builder_assist2012.csv`, chi tiết trong `assist2012.md`):

  | | 2009 (toàn bộ Skill Builder) | 2012 MasterySection | 2012 ngoài MasterySection |
  |---|---:|---:|---:|
  | Chuỗi kết thúc bằng 3 lượt đúng | 45,5% | 70,8% | 23,7% |
  | Chuỗi kết thúc bằng lượt đúng | 82,5% | 89,8% | 67,7% |
  | AUC "Last Answer", test | 0,652 | 0,633 | 0,645 |

  Cơ chế dừng thấy rõ ở hình dạng chuỗi, nhưng không làm AUC của "Last Answer" cao hơn. Ảnh hưởng thật nằm ở
  phân bố: trong Skill Builder, chuỗi dừng ngay khi học sinh đạt yêu cầu, nên thiếu các lượt luyện sau khi đã thành
  thạo. Mọi chỉ số tính trên 2009 nên được đối chiếu với tập con ngoài MasterySection của 2012.
- **Tỷ lệ đúng giảm dần theo phần:** 0,661 ở history, 0,654 ở validation, 0,635 ở test. Học sinh gặp bài khó hơn về
  sau, nên mọi so sánh phải dùng cùng một phần dữ liệu.
- **Thời gian phản hồi:** 0,11% dòng có `rt_ms` thiếu (giá trị gốc ≤ 0); các giá trị còn lại được cắt về
  [500; 120.000] ms.
- **Ánh xạ kỹ năng:** tag lấy từ `skill_map_v1.csv` (cột `canonical_tag`; `configs/data.yaml`, khóa `skill_map`), so
  tên kỹ năng không phân biệt hoa thường và khoảng trắng, vì log 2009 có tên như `Circumference ` (dấu cách cuối).
  Kỹ năng không có trong từ điển dùng `SkillMapper`. So với `SkillMapper`, v1 thêm tag cho Pattern Finding
  (5.135 dòng), là kỹ năng có template Khan đúng sau khi rà.

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only assist2009   # khoảng 7 giây
```

Đầu ra: `lmsrag/artifacts/v2/interactions_assist2009.parquet` cùng ba bảng `reports/tables/a1_cleaning_assist2009.csv`,
`a1_kept_vs_dropped.csv` và `a1_skill_builder.csv`.

**Data v1** (tag `data-v1`): mã băm các tệp đầu ra nằm trong `lmsrag/data/cards/data_v1.sha256`; kiểm bằng
`sha256sum -c lmsrag/data/cards/data_v1.sha256` từ gốc repo. Bản này build bằng pandas 3.0.6, pyarrow 26.0.1, numpy
2.4.6; build lại ngày 27/11/2026 ra đúng từng byte, kể cả các bảng trong `reports/tables/`. Với phiên bản thư viện khác,
parquet sẽ khác byte (vì ghi phiên bản pandas vào metadata), nên khi đó phải so nội dung.
