# Biên bản kiểm định độ khó và bộ lọc Khan (P1-06)

Lưu lại để làm bằng chứng cho báo cáo. Mọi số trong tệp này tái tạo được bằng
`python -m lmsrag.labels.difficulty_agreement`; bảng đầy đủ ở `reports/tables/a3_difficulty_audit.csv`.

## 1. Thiết kế

| Mục | Giá trị |
|---|---|
| Đối tượng kiểm định | `difficulty` (hạng phần trăm trong từng nguồn) và bộ lọc bài Khan của kho v2, từ điển kỹ năng v1.1 |
| Mẫu | 100 bài `worked_example`, 20 bài mỗi `difficulty_level`: 50 Khan (50 template khác nhau), 25 GSM8K, 25 MATH. GSM8K không có bài ở mức 4, MATH chỉ có ở mức 1, 2, 4, nên cách chia theo mức cố định trong `ALLOCATION` (`difficulty_sheets.py`) |
| Seed | `data_seed` = 2026; thứ tự các dòng đã xáo trộn |
| Sinh phiếu | `python -m lmsrag.labels.difficulty_sheets`, ngày 19/11/2026; khóa ở `a3_difficulty_key.csv` (độ khó tự động và nguồn, không gửi cho người chấm) |
| Câu hỏi | `rating` 1–5: độ khó với học sinh THCS đã học phần kiến thức đó. Bài Khan thêm `figure_ok` (làm được mà không cần hình) và `answer_ok` (đáp án lưu đúng). Thang và ví dụ ở `a3_difficulty_HUONG_DAN.md` |
| Người chấm | Người 2: Thuật (`a3_difficulty_nguoi2.csv`); Người 3: Dương (`a3_difficulty_nguoi3.csv`); chấm độc lập |
| Phân xử | Người 1, chỉ với ô `figure_ok`/`answer_ok` hai người khác nhau. Điểm độ khó không phân xử: thẻ dùng trung bình hai người |

## 2. Tệp đã nộp

Bản gốc giữ nguyên byte trong `a3_submissions/`; hai tệp dùng để tính là bản sao y (cùng SHA-256).

| Tệp gốc | Lưu lúc | SHA-256 | Dùng làm |
|---|---|---|---|
| `a3_difficulty_thuat.csv` | 19/11/2026 22:41 | `34fdab5f1d45b093f6b00e42b7721744e10bc3e2d462a4b1017f325294e9cc84` | `a3_difficulty_nguoi2.csv` |
| `a3_difficulty_duong.csv` | 19/11/2026 22:41 | `700bd45e7b0452be405d3eddca81ae7c156016aec20edcd3bb76461212c336b7` | `a3_difficulty_nguoi3.csv` |

Kiểm tra khi nhận: đủ 100 dòng, không ô nào trống, `rating` trong 1–5, hai cột kiểm tra chỉ có ở 50 dòng Khan, mọi cột
khác trùng với phiếu gốc.

## 3. Độ khớp giữa hai người chấm

| Nhóm | n | κ có trọng số bậc hai [95%] | Spearman | Trùng điểm | Lệch ≥ 2 mức |
|---|---:|---|---:|---:|---:|
| Tất cả | 100 | 0,893 [0,835; 0,936] | 0,897 | 79% | 0 |
| Khan | 50 | 0,913 [0,848; 0,962] | 0,920 | 80% | 0 |
| GSM8K | 25 | 0,636 [0,367; 0,852] | 0,681 | 76% | 0 |
| MATH | 25 | 0,905 [0,762; 0,978] | 0,903 | 80% | 0 |

Hai người chấm rất khớp, nên điểm trung bình của họ đủ tin cậy để làm chuẩn đối chiếu. GSM8K thấp hơn vì điểm dồn ở
mức 2–3 (ít phương sai), không phải vì hai người hiểu thang khác nhau: không bài nào lệch quá 1 mức.

## 4. Độ khó tự động so với người (kết quả chính)

Spearman giữa `difficulty` và điểm trung bình hai người, khoảng bootstrap 95% theo bài (1.000 lần, `data_seed`).

| Nhóm | Cách tính độ khó | n | Spearman [95%] |
|---|---|---:|---|
| Tất cả | | 100 | 0,581 [0,442; 0,691] |
| Khan | tỷ lệ đúng trên log Junyi (`junyi_history`) | 31 | **0,805** [0,663; 0,878] |
| Khan | hồi quy từ số bước gợi ý (`khan_steps`) | 19 | 0,592 [0,188; 0,818] |
| Khan, cả hai | | 50 | 0,754 [0,617; 0,838] |
| GSM8K | số phép tính trong lời giải (`gsm8k_steps`) | 25 | **0,417** [0,064; 0,742] |
| MATH | Level 1–3 (`math_level`) | 25 | **0,467** [0,049; 0,750] |

Điểm trung bình của người theo (mức khó, nguồn), thang 1–5:

| `difficulty_level` | Khan | GSM8K | MATH | Cả ba |
|---|---:|---:|---:|---:|
| 1 | 1,19 | 2,25 | 2,17 | 1,80 |
| 2 | 1,50 | 2,33 | 2,67 | 2,10 |
| 3 | 2,19 | 2,64 | – | 2,35 |
| 4 | 2,57 | – | 3,35 | 3,08 |
| 5 | 3,29 | 2,83 | – | 3,15 |

**Nhận xét.**

1. Độ khó lấy từ dữ liệu làm bài thật (Junyi) khớp tốt với người. Hai cách tính dựa trên đặc điểm bài (số phép tính
   của GSM8K, Level của MATH) khớp yếu: khoảng tin cậy rộng và chạm gần 0. Số phép tính của GSM8K đo độ dài lời
   giải hơn là độ khó; MATH chỉ có ba giá trị Level (1–3) nên thứ tự bên trong một Level không có thông tin.
2. Giả định của P1-04 ("hạng trong từng nguồn so sánh được giữa các nguồn") **không đứng**. Cùng một mức khó, bài
   GSM8K và MATH khó hơn bài Khan khoảng 1 điểm ở các mức thấp. Ở GSM8K, điểm của người gần như không đổi theo mức (2,25
   → 2,83), nên năm mức của GSM8K gần như không phân tầng được độ khó.
3. Trong cả kho, độ khó vẫn có thứ tự đúng (điểm trung bình tăng đều từ mức 1 đến 5), nhưng khoảng cách giữa mức 4 và
   5 rất nhỏ (3,08 so với 3,15).

**Đề xuất** (thẻ P1-06 yêu cầu khi một nguồn có Spearman thấp rõ rệt). Nhóm chọn trước khi chạy thí nghiệm, không chọn
theo kết quả trên tập test:

- (a) Khi chọn bài theo ZPD, chỉ so độ khó **trong cùng một nguồn**, hoặc chỉ dùng bài Khan có `difficulty_source =
  junyi_history` cho phần cần độ khó chính xác; GSM8K và MATH dùng làm ví dụ giải mẫu, không làm căn cứ chọn mức khó.
- (b) Hạ trọng số của GSM8K và MATH trong các chỉ số đánh giá phụ thuộc độ khó (ví dụ tỷ lệ bài đúng mức ở gold set),
  và báo cáo các chỉ số đó tách theo nguồn.
- (c) Nếu muốn ghép một thang chung cho cả ba nguồn, phải hiệu chỉnh trên **một mẫu hiệu chỉnh riêng** do người chấm,
  rồi đo lại trên mẫu khác. Không dùng 100 bài này cho cả hai việc.

## 5. Bộ lọc Khan

| Kiểm tra | n | Đồng ý giữa hai người | PABAK | Cần phân xử | Đạt sau phân xử [Wilson 95%] |
|---|---:|---:|---:|---:|---|
| `figure_ok` (làm được không cần hình) | 50 | 100% | 1,00 | 0 | 92,0% [81,2; 96,9] |
| `answer_ok` (đáp án lưu đúng) | 50 | 98% | 0,96 | 1 | 100% [92,9; 100] |
| Cả hai | 50 | | | | 92,0% [81,2; 96,9] |

κ của `answer_ok` bằng 0 dù hai người đồng ý 98%: gần như mọi bài là "có" (nghịch lý tỷ lệ), nên báo cáo PABAK.

Bốn bài cả hai người cùng cho là cần hình: d046 `inscribed_angles_1` (không biết D nằm trên cung nào), d053
`shaded_areas` (không biết phần nào tô), d059 `trigonometry_2` (không biết góc vuông ở đâu), d063
`solving_similar_triangles_2` (không biết các điểm thẳng hàng). `shaded_areas` và `solving_similar_triangles_1` cũng
bị ghi "thiếu hình" ở phiếu κ của P1-05 (m047, m021).

## 6. Phân xử (Người 1)

| item | Kiểm tra | N2 | N3 | Nhãn cuối | Lý do |
|---|---|---|---|---|---|
| d066 | `answer_ok` | có | không | có | `answer_type` là `cuberoot`: đáp án lưu "192" nghĩa là ∛192 = 4∛3, đúng |

Ghi chú: d066 lúc đầu được quyết là "không", vì phiếu chỉ hiện con số 192 mà không có dấu căn, và Người 1 được giải
thích (sai) rằng bộ tách đáp án làm mất hệ số. Khi đối chiếu `answer_type` thì thấy đây là quy ước của Khan: ô trả lời
dạng căn chỉ lưu số dưới dấu căn. Hai bài cùng quy ước trong phiếu (d014 √10, d096 √49) đều được cả hai người chấm
"có", nên nhãn cuối của d066 được đổi thành "có" cho nhất quán.

## 7. Thay đổi trong phiếu

| Khi nào | Thay đổi | Ảnh hưởng tới nhãn |
|---|---|---|
| Trước khi gửi | Mở rộng bộ đổi TeX cho lời giải (`\boxed`, `\binom`, `\cancel`, phân số tách dòng, dấu ngoặc tập hợp…); phiếu gửi đi không còn ký tự `\` | — |
| Sau khi nhận | Cột `answer` không hiện dấu căn cho đáp án `radical`/`cuberoot` (d014, d066, d096). Đã sửa trong `difficulty_sheets.py` (`display_answer`) và thêm test | Có ở d066 (một người chấm "không"); đã phân xử |
| Sau khi nhận | Thêm PABAK cho hai cột kiểm tra Khan | — |

## 8. Việc tiếp theo

1. ~~Nhóm chọn một trong các đề xuất (a)–(c) ở mục 4~~: đã chọn (a) có thứ tự ưu tiên, mục 9.
2. Đánh dấu `needs_figure` cho 4 template ở mục 5, xem lại các template cùng họ (`solving_similar_triangles_1`,
   `inscribed_angles_*`), rồi build lại kho. Làm sau khi đã chốt các số ở trên, để 92% vẫn là độ chính xác của bộ lọc
   lúc được đo.
3. Bộ chấm tự động của kho bài tập phải hiểu `answer_type` `radical`/`cuberoot` (54 bài): "147" nghĩa là √147.

## 9. Quyết định sau kiểm định (20/11/2026)

**Phương án được chọn: (a) có thứ tự ưu tiên.** Chọn trước khi chạy P1-07 và P1-09, dựa trên các số ở mục 4, không
dựa trên kết quả của tập test hay gold set.

| Bậc (`difficulty_tier`) | Bài | Cách tính độ khó | Spearman với người (mục 4) | Dùng thế nào |
|---|---:|---|---|---|
| 1 (`difficulty_reliable` = true) | 1.273 | Khan, tỷ lệ đúng trên Junyi | 0,81 | Nguồn chính khi chọn bài theo mức khó (ZPD) |
| 2 | 791 | Khan, từ số bước gợi ý | 0,59 | Khi kỹ năng không đủ bài bậc 1 |
| 3 | 2.344 | GSM8K (số phép tính), MATH (Level) | 0,42; 0,47 | Ví dụ giải mẫu; độ khó chỉ so trong cùng nguồn, kết quả ghi là "độ khó kém tin cậy" |

Quy tắc chung: **không bao giờ so độ khó giữa hai nguồn khác nhau**, và mọi chỉ số phụ thuộc độ khó báo cáo tách theo
bậc. Hai trường `difficulty_tier` và `difficulty_reliable` có trong mọi chunk và mọi bài của kho bài tập (build
`build_corpus.py`).

Phạm vi phủ (kỹ năng chuẩn có bài ở ít nhất 3 mức khó **trong cùng một nguồn**, `a3_corpus_summary.csv`): chỉ bậc 1:
17 kỹ năng; tới bậc 2: 21; tới bậc 3: 33. (Nếu cho phép ghép các mức của nhiều nguồn thì là 37, nhưng như vậy là so độ
khó giữa các nguồn.)

**Thử nghiệm không giữ lại: thêm người học Junyi ngoài mẫu.** Để tăng số bài bậc 1, đã thử tính độ khó Khan bằng cả
câu trả lời của 76.521 người học đủ điều kiện nhưng không nằm trong mẫu 20.000 (19,9 triệu lượt; những người này không
thuộc split nào nên không lộ dữ liệu test). Thử nghiệm được quyết theo cơ chế (thêm dữ liệu), không theo điểm của người
chấm, rồi kiểm một lần trên 100 bài của P1-06:

| Trên 100 bài P1-06 | n | Độ khó v1 (mẫu 20.000) | Thêm người học ngoài mẫu |
|---|---:|---|---|
| Tất cả | 100 | 0,581 [0,442; 0,691] | 0,538 [0,389; 0,664] |
| Khan | 50 | 0,754 [0,617; 0,838] | 0,725 [0,569; 0,829] |
| Khan bậc 1 theo v1 | 31 | 0,805 | 0,814 |

Bài bậc 1 tăng từ 1.273 lên 1.494, kỹ năng bậc 1 có ≥ 3 mức từ 17 lên 20, nhưng độ khớp với người không tăng. Trong 6
bài của mẫu chuyển sang bậc 1, 2 bài gần điểm của người hơn và 4 bài xa hơn (d053, d056, d088, d096); các template này
ít được luyện trong mẫu, nên người làm chúng có thể không đại diện. Vì cách mới không cho thấy cải thiện và khiến các số
P1-06 không còn mô tả đúng kho được dùng, kho giữ độ khó v1. Code giữ lại sau khóa `difficulty.junyi_outside_sample:
false` (`configs/data.yaml`); kho build lại có mọi trường của 14.931 chunk trùng với bản đã kiểm định, chỉ thêm `difficulty_tier`.
