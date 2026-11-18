# Biên bản kiểm định từ điển kỹ năng v1 (P1-05, vòng 1)

Lưu lại để làm bằng chứng cho báo cáo: thiết kế mẫu, các tệp đã nộp, kết quả trước và sau phân xử, và các thay đổi
trong phiếu. Mọi số trong tệp này tái tạo được bằng `python -m lmsrag.labels.agreement`.

## 1. Thiết kế

| Mục | Giá trị |
|---|---|
| Từ điển được kiểm định | `lmsrag/data/mappings/skill_map_v1.csv`, `map_version` 1.0 |
| Mẫu | 150 cặp: 100 cặp (kỹ năng, template Khan) rút theo số lượt làm ASSISTments chia đều cho các template dùng được của kỹ năng; 50 cặp (kỹ năng, đoạn đầu của bài GSM8K/MATH gắn tag bằng quy tắc) |
| Seed | `data_seed` = 2026 (`configs/data.yaml`); thứ tự các dòng đã xáo trộn |
| Sinh phiếu | `python -m lmsrag.labels.make_sheets`, ngày 18/11/2026 17:41; khóa cách tạo cặp ở `a4_mapping_key.csv` (không gửi cho người gán) |
| Nhãn | `1` bài luyện đúng kỹ năng, `0` không; tiêu chí và 6 ví dụ luyện tập ở `a4_mapping_HUONG_DAN.md` |
| Người gán | Người 2: Thuật (`a4_mapping_nguoi2.csv`); Người 3: Dương (`a4_mapping_nguoi3.csv`); gán độc lập |
| Phân xử | Người 1, chỉ với các dòng hai người gán khác nhau (`a4_mapping_adjudication.csv`) |

## 2. Tệp đã nộp

Bản gốc giữ nguyên byte trong `a4_submissions/`. Hai tệp dùng để tính là bản sao y của bản nộp (cùng SHA-256).

| Tệp gốc | Lưu lúc | SHA-256 | Dùng làm |
|---|---|---|---|
| `a4_mapping_thuat_da_gan_nhan.csv` | 18/11/2026 22:08 | `ad62a2171210e2ea8daa1465931e404ffb7ca89828a7f5212cafbbb9792b103b` | Bản lưu trước của Người 2; cùng 150 nhãn với bản nộp, khác 15 ghi chú. Không dùng để tính |
| `a4_mapping_thuat.csv` | 18/11/2026 22:15 | `c1c7704167522f285ae094c44db443efa024746a57a506701719d52b4264a0af` | `a4_mapping_nguoi2.csv` |
| `a4_mapping_duong.csv` | 18/11/2026 23:15 | `bbb08e973913baeee0c3a9060af9cf3d30ab5b09b99be69bf6eca7a5cc8aa13e` | `a4_mapping_nguoi3.csv` |

Kiểm tra lại: `sha256sum lmsrag/data/labels/a4_submissions/* lmsrag/data/labels/a4_mapping_nguoi*.csv`.

Kiểm tra khi nhận: cả hai tệp đủ 150 dòng, không dòng nào trống nhãn, chỉ có nhãn 0/1; mọi cột ngoài `label`, `note`
trùng với phiếu gốc.

## 3. Kết quả trước phân xử

Bảng đầy đủ (cả khoảng tin cậy bootstrap 1.000 lần theo cặp): `reports/tables/a4_mapping_audit.csv`.

| Nhóm | n | κ Cohen [95%] | Tỷ lệ đồng ý | PABAK | Tỷ lệ nhãn 1 (N2 / N3) | Cần phân xử |
|---|---|---|---|---|---|---|
| Tất cả | 150 | 0,892 [0,775; 0,976] | 96,7% | 0,933 | 80,0% / 82,0% | 5 |
| Template Khan | 100 | 0,756 [0,475; 0,960] | 96,0% | 0,920 | 90,0% / 92,0% | 4 |
| GSM8K/MATH gắn bằng quy tắc | 50 | 0,958 [0,860; 1,000] | 98,0% | 0,960 | 60,0% / 62,0% | 1 |

κ ≥ 0,6 nên không phải sửa hướng dẫn hay gán lại. κ của nhóm Khan thấp hơn PABAK vì gần như mọi cặp đều đúng
(nghịch lý tỷ lệ); tỷ lệ đồng ý thô 96% cho thấy hai người khớp nhau.

## 4. Phân xử (Người 1)

| item | Kỹ năng | N2 | N3 | Nhãn cuối | Lý do |
|---|---|---|---|---|---|
| m062 | Write Linear Equation from Ordered Pairs | 0 | 1 | 1 | Tìm hệ số góc và tung độ gốc từ hai điểm là phần chính của kỹ năng; chỉ khác cách ghi đáp số |
| m097 | D.4.8-understanding-concept-of-probabilities | 0 | 1 | 1 | Tính xác suất một biến cố đơn giản thuộc khái niệm xác suất cơ bản; giải thích kỹ năng trong phiếu hẹp hơn kỹ năng gốc |
| m107 | Multiplication and Division Positive Decimals | 0 | 1 | 1 | Phải chia tiếp sang phần thập phân để có đáp số 0.6 |
| m114 | Proportion | 1 | 0 | 1 | Viết tỉ số dưới dạng phân số là bước nền của tỉ lệ; ranh giới với Ratio |
| m115 | Simplifying Expressions positive exponents | 0 | 1 | 0 | Bài dùng số mũ âm, ngoài phạm vi số mũ dương của kỹ năng |

## 5. Kết quả sau phân xử

Độ chính xác = tỷ lệ cặp có nhãn cuối là 1, khoảng Wilson 95%.

| Nhóm | n | Độ chính xác [Wilson 95%] |
|---|---|---|
| Tất cả | 150 | 82,0% [75,1; 87,3] |
| Template Khan | 100 | 92,0% [85,0; 95,9] |
| — `manual_review` | 96 | 91,7% [84,4; 95,7] |
| — `manual_add` | 3 | 3/3 [43,9; 100] |
| — `rule_confirmed` | 1 | 1/1 [20,7; 100] |
| GSM8K/MATH, `rule_text` | 50 | 62,0% [48,2; 74,1] |

27 cặp có nhãn cuối 0 (26 cặp cả hai cùng gán 0, 1 cặp qua phân xử), liệt kê ở
`reports/tables/a4_mapping_errors.csv`: 8 cặp template Khan, 19 cặp gắn bằng quy tắc.

**Nhận xét.** Từ điển kỹ năng–template Khan (phần đã rà tay ở phiếu A–D) đạt độ chính xác cao. Điểm yếu là quy tắc
gắn tag cho bài GSM8K/MATH: khoảng 4/10 bài bị gắn sai kỹ năng, phần lớn vì quy tắc bắt theo từ khóa (ví dụ "solve for
x" thành phương trình 1–2 bước, phép chia số âm thành thứ tự phép tính). Sửa 19 bài trong mẫu không sửa được phần còn
lại, nên phải sửa quy tắc và đo lại trên một mẫu mới.

## 6. Thay đổi trong phiếu và hướng dẫn

| Khi nào | Thay đổi | Ảnh hưởng tới nhãn |
|---|---|---|
| Trước khi gửi | Thêm cột `skill_meaning` (giải thích 60 kỹ năng, `a4_skill_glossary.csv`), đổi TeX sang dạng dễ đọc (cột `problem`, giữ bản gốc ở `problem_tex`), thêm phần bối cảnh và 6 ví dụ luyện tập vào hướng dẫn; mẫu 150 cặp không đổi | — |
| Sau vòng 1 | Lỗi đổi TeX: `\pink{{?}}` hiện thành "πnk?" ở m030 và m099 (cả hai người đều phát hiện, Người 2 ghi chú). Đã sửa trong `make_sheets.py` và thêm test; trong 150 đề chỉ hai đề này đổi | Không: cả hai người vẫn đọc được phép tính và cùng gán 1 |
| Sau vòng 1 | m097: giải thích của D.4.8 trong `a4_skill_glossary.csv` hẹp hơn kỹ năng gốc, khiến Người 2 gán 0 | Đã phân xử; sửa giải thích trước vòng sau |

## 7. Việc tiếp theo

1. Sửa từ điển theo 8 cặp template Khan có nhãn 0, sửa quy tắc gắn tag GSM8K/MATH, tăng `map_version`.
2. Chạy lại P1-03 và P1-04.
3. Rút mẫu mới (seed khác hoặc loại trừ 150 cặp này) để đo lại độ chính xác của quy tắc đã sửa.
