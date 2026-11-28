# Junyi Academy 2015 (problem log)

- **Nguồn và đường dẫn tải:** Junyi Academy, bộ log công bố cùng bài báo EDM 2015. Bản sao trên Hugging Face:
  https://huggingface.co/datasets/Lucy9999/junyi2015. Tệp dùng: `junyi_ProblemLog_original.csv` (2,7 GB) và
  `junyi_Exercise_table.csv`, lưu tại `lmsrag/data/raw/junyi_2015/` (không commit).
- **Ngày tải:** 16/11/2026 (đợt khảo sát dữ liệu).
- **SHA-256 tệp gốc:** log `A7C99EDE1F9B19648FDCD953B6CA0B0282BD31F2F913BD9D26B23AA9FCB36A25`; bảng bài tập
  `13292CC2C809985ABBACEDABA434E39AEB02A7FCB2A32F17B60C4D3F4D769546`.
- **Giấy phép và trích dẫn bắt buộc:** chỉ dùng phi thương mại cho nghiên cứu. Trích dẫn: Chang, H.-S., Hsu, H.-J., &
  Chen, K.-T. (2015). Modeling Exercise Relationships in E-Learning: A Unified Approach. *Proceedings of the 8th
  International Conference on Educational Data Mining (EDM 2015)*. Bản đã làm sạch chỉ để trong Kaggle Dataset riêng tư.

## Vì sao dùng bộ này

Junyi chạy chính các template của khan-exercises (bản tiếng Trung), nên tên `exercise` trùng với tên template Khan.
Đây là bộ log duy nhất của đề tài nối được một lượt làm với nội dung câu hỏi, qua `khan_slug` và
`lmsrag/data/khan_v2/`.

## Quy mô

| | Bản gốc | Người học từ 20 lượt | Mẫu dùng trong thực nghiệm |
|---|---:|---:|---:|
| Dòng | 25.925.992 | 25.013.101 | 5.111.317 |
| Người học | 247.606 | 96.521 | 20.000 |
| Tỷ lệ đúng | 0,828 | 0,831 | 0,832 |
| Bài tập (`exercise`) | 722 | – | 706 |
| Item (`exercise:problem_type`) | 2.835 | – | 2.474 |
| Tag kỹ năng | – | – | 111 (72 tag chuẩn, 39 `junyi:<topic>`) |

Bảng bài tập có 837 bài thuộc 40 topic; log dùng 722 bài trong số đó. Dữ liệu từ 12/10/2012 đến 11/01/2015.
Tỷ lệ đúng của mẫu (0,832) gần với tỷ lệ của mọi người học đủ điều kiện (0,831), nên mẫu không bị lệch rõ.

## Quy tắc loại bỏ

Bảng đầy đủ: `reports/tables/a2_cleaning_junyi2015.csv`.

| Bước | Quy tắc | Dòng trước | Dòng sau | Bị loại | Người học sau |
|---|---|---:|---:|---:|---:|
| 0 | Bản gốc | 25.925.992 | 25.925.992 | 0 | 247.606 |
| – | *Thông tin:* `time_taken` ≤ 0, đặt thành thiếu và giữ dòng | – | – | 75.822 (0,29%) | – |
| – | *Thông tin:* trùng (`user_id`, `time_done`) | – | – | 1.439 (0,01%) | – |
| 1 | Bỏ bài tập không có trong bảng bài tập | 25.925.992 | 25.925.992 | 0 | 247.606 |
| 2 | Bỏ dòng thiếu `correct` | 25.925.992 | 25.925.992 | 0 | 247.606 |
| 3 | Bỏ người học dưới 20 lượt | 25.925.992 | 25.013.101 | 912.891 | 96.521 |
| 4 | Lấy mẫu 20.000 người học (`data_seed` 2026) | 25.013.101 | 5.111.317 | 19.901.784 | 20.000 |

Mẫu giống hệt nhau giữa các lần chạy, vì người học được sắp xếp trước khi rút bằng `numpy.random.default_rng(2026)`.

## Nối với kho Khan

Bảng: `reports/tables/a2_coverage_junyi2015.csv`.

| Chỉ số | Toàn bộ log | Mẫu |
|---|---:|---:|
| Lượt làm trên bài có template Khan | 65,8% | 66,1% |
| Lượt làm trên template **dùng được** trong kho v2 (283 template) | 32,7% | 32,9% |
| Trong các lượt có template, `problem_type` là dạng bài của template | 89,3% | 89,2% |

`problem_type` của Junyi là `id` của dạng bài trong HTML, hoặc số thứ tự nếu dạng bài không có `id`. Bảng nối là
`lmsrag/data/khan_v2/khan_problem_types.csv` (449 template, 1.069 dạng bài), sinh bằng
`python -m lmsrag.data_tools.khan_problem_types`. Khoảng 11% còn lại không khớp vì Junyi chạy các bản template cũ
hơn bản Khan năm 2016, ví dụ `subtraction-2` hay `addition_3` nằm trong template `addition_4`.

## Trường của bảng tương tác

- `item_id` = `exercise:problem_type`; `source_skill` = `exercise`; `khan_slug` = `exercise` nếu có template.
- `skill_tag`: tag chuẩn qua chiều ngược của từ điển kỹ năng v1 `skill_map_v1.csv` (P1-05, cột `canonical_tag`).
  Nếu không có thì dùng `junyi:<topic>`. 44,7% lượt có tag chuẩn (72 tag); số còn lại thuộc 39 tag `junyi:<topic>`.
  Từ v1.1, template gồm nhiều kỹ năng được tách theo dạng bài (cột `khan_types`): `mean_median_and_mode:median` mang
  tag `Median` (v1.0 gắn `Mean` cho cả ba dạng), `discount_tax_and_tip_word_problems:simple-tax` mang `Percent-Of`
  (v1.0: `Percent-Discount`). Với v1.0: 44,8% (69 tag).
  Template không dùng được (cần hình, không có gợi ý) vẫn gắn tag cho lượt làm, nên các cặp này được rà riêng ở phiếu D.
  Với bản nháp `assistments_to_khan_draft.csv`: 40,5% (62 tag).
- `ts` = `time_done` (micro giây) chia 1000; `rt_ms` = `time_taken` × 1000, cắt về [500; 120.000] (0,21% thiếu).
- `correct` là cờ của Junyi: đúng ở lần đầu và không dùng gợi ý. Chỉ 18 dòng có `correct` = 1 mà `attempts` > 1.
- `section_type` = `review` nếu `review_mode` (2,2% lượt), ngược lại `practice`.

## Cách chia và seed

Giống ASSISTments: theo từng người học, theo `ts`, tỷ lệ 70/15/15. Kết quả: history 3.586.673 dòng, validation
767.370, test 757.274.

## Đặc điểm cần biết khi đọc kết quả

- **Tỷ lệ đúng chung rất cao (0,832).** Đoán "luôn đúng" đã đạt độ chính xác 0,832, nên mọi bảng phải kèm baseline đoán
  theo đa số và ưu tiên AUC hoặc log-loss hơn độ chính xác.
- Tỷ lệ đúng theo phần: 0,839 ở history, 0,826 ở validation, 0,807 ở test.
- Học sinh làm bản tiếng Trung; kho v2 là bản tiếng Anh của cùng template. Nội dung tương đương, nhưng câu chữ khác.
- Thống kê trong `lmsrag/data/khan_v2/junyi_*_survey.csv` được tính trên toàn log, chỉ dùng cho khảo sát; độ khó dùng
  trong thực nghiệm phải tính từ phần history (P1-04).

## Lệnh tái lập

```powershell
python -m lmsrag.datasets.build --config configs/data.yaml --only junyi2015   # khoảng 3 phút
```

**Data v1** (tag `data-v1`): mã băm các tệp đầu ra nằm trong `lmsrag/data/cards/data_v1.sha256`; kiểm bằng
`sha256sum -c lmsrag/data/cards/data_v1.sha256` từ gốc repo. Bản này build bằng pandas 3.0.6, pyarrow 26.0.1, numpy
2.4.6; build lại ngày 27/11/2026 ra đúng từng byte, kể cả các bảng trong `reports/tables/`. Với phiên bản thư viện khác,
parquet sẽ khác byte (vì ghi phiên bản pandas vào metadata), nên khi đó phải so nội dung.
