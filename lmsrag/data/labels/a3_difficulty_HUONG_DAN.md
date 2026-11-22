# Hướng dẫn chấm độ khó (P1-06)

Dành cho **Người 2** (tệp `a3_difficulty_nguoi2.csv`) và **Người 3** (tệp `a3_difficulty_nguoi3.csv`). Hạn nộp: 2/12.
Thời gian: khoảng 60–75 phút cho 100 bài, vì có lời giải để đọc. Nên chia làm 2 lần.

## Bối cảnh

Hệ thống chọn bài cho học sinh theo độ khó: học sinh mới học được giao bài dễ, học sinh đã vững được giao bài khó hơn.
Mỗi bài trong kho đã có một độ khó tính tự động (từ tỷ lệ làm đúng của học sinh thật, số bước giải, hoặc mức của bộ
đề). Việc của bạn là chấm độ khó **theo cảm nhận của bạn**, để nhóm kiểm tra xem con số tự động có khớp với người
không. Phiếu không cho biết độ khó tự động; đừng cố đoán nó.

Bài lấy từ ba kho đề tiếng Anh. Với các bài của Khan Academy, bạn kiểm thêm hai điều về chất lượng đề (mục 3).

## 1. Các cột trong phiếu

| Cột | Nội dung |
|---|---|
| `problem` | Đề bài. Công thức đã đổi sang dạng dễ đọc: `(3)/(4)` là 3/4, `x^2` là x², `√(16)`, `C(5, 2)` là tổ hợp chập 2 của 5. Bài trắc nghiệm có các lựa chọn ở cuối đề |
| `solution` | Lời giải mẫu, để bạn thấy bài cần những bước gì |
| `answer` | Đáp án được lưu trong kho |
| `rating` | **Bạn điền**: 1 đến 5 |
| `figure_ok`, `answer_ok` | **Bạn điền** ở các dòng còn trống (bài Khan): `có` hoặc `không`. Dòng ghi sẵn `–` thì để nguyên |
| `note` | Ghi chú nếu cần |
| `problem_tex`, `solution_tex`, `answer_tex` | Bản gốc dạng TeX, chỉ để đối chiếu khi bản dễ đọc khó hiểu; có thể ẩn các cột này |

## 2. Chấm `rating`

> Với một học sinh trung học cơ sở bình thường **đã học phần kiến thức bài cần**, bài này khó đến mức nào?

| `rating` | Mức | Dấu hiệu | Ví dụ (không có trong phiếu) |
|---|---|---|---|
| `1` | Rất dễ | Một phép tính hoặc một bước quen thuộc, làm nhẩm được | 36 + 47; 10% của 50 |
| `2` | Dễ | Hai bước theo cách quen thuộc | Giải 3x − 4 = 11; mua 3 bút giá 5 nghìn, đưa 20 nghìn, được trả lại bao nhiêu |
| `3` | Trung bình | Ba, bốn bước, hoặc phải tự chọn cách làm; bài có lời nhiều dữ kiện | Giải 2(x + 3) = 5x − 9; tỉ lệ: 4 người làm 6 ngày, 3 người làm mấy ngày |
| `4` | Khó | Nhiều bước, có bước biến đổi không hiển nhiên, dễ sai | Bài có lời phải lập hệ hai phương trình; xác suất rút 2 bi liên tiếp không hoàn lại |
| `5` | Rất khó | Cần một ý tưởng không quen, kiến thức cuối cấp hoặc vượt THCS, chuỗi lập luận dài | Số dư khi chia 2^100 cho 7; đếm số cách xếp có điều kiện |

- Chấm **độ khó toán học** của bài: số bước, mức trừu tượng, có cần ý tưởng hay không. Không chấm độ khó tiếng Anh,
  không chấm độ dài của đề.
- Lời giải chỉ để tham khảo. Lời giải của Khan viết rất chi tiết, kể cả cho bài rất dễ; một lời giải dài không có
  nghĩa là bài khó.
- Chấm từng bài riêng, theo bảng trên. Đừng so với bài vừa chấm: không cần mỗi mức có số bài bằng nhau.
- Dùng cả thang 1–5. Phân vân giữa hai mức thì chọn một, ghi lý do vào `note`. Không để trống `rating`.

## 3. Hai cột kiểm tra bài Khan

Chỉ điền ở các dòng mà hai cột này đang trống. Đây là kiểm tra bộ lọc đã chọn bài Khan vào kho.

| Cột | `có` | `không` |
|---|---|---|
| `figure_ok` | Đọc chữ là đủ để hiểu và làm bài | Đề cần một hình, đồ thị, bảng hay trục số mà phiếu không có, và thiếu nó thì không làm được |
| `answer_ok` | Đáp án ở cột `answer` đúng với đề | Đáp án sai, thiếu (bài có hai đáp án mà chỉ ghi một), hoặc không đọc ra được |

- Với `answer_ok`, coi là đúng nếu **giá trị** bằng nhau dù khác cách viết: 3/8 và 0.375; 0.6666666666666666 và 2/3;
  44,1% và 0.441; "x = 1" và "1". Không cần tự giải lại từ đầu, chỉ cần đối chiếu đề, lời giải và đáp án.
- Đề nhắc tới hình nhưng vẫn làm được bằng chữ (ví dụ "the rectangle below is 5 cm by 4 cm") thì `figure_ok` là `có`.
- Phân vân thì vẫn chọn, rồi ghi lý do vào `note`.

## 4. Ví dụ luyện tập (không có trong phiếu)

Đọc 5 ví dụ này trước khi bắt đầu.

| Bài | `rating` | Vì sao |
|---|---|---|
| What is 25% of 80? | `1` | Một phép tính quen thuộc |
| Tom has 3 boxes with 12 pencils each. He gives away 10 pencils. How many are left? | `2` | Hai bước: nhân rồi trừ |
| The ratio of boys to girls is 3 : 5 and there are 24 girls. How many students are there? | `3` | Tìm một phần, rồi số bạn nam, rồi tổng; phải hiểu tỉ số |
| A shop raises a price by 20%, then lowers the new price by 20%. By what percent did the price change overall? | `4` | Nhiều bước, và dễ sai nếu nghĩ là "không đổi" |
| How many positive integers less than 1000 are divisible by 3 or 5 but not by 15? | `5` | Cần nguyên lý bao hàm – loại trừ và lập luận cẩn thận |

Ví dụ cho `answer_ok`: đề "Express 0.75 as a fraction", đáp án `0.75` → `có` (cùng giá trị). Đề "Solve x² = 9",
đáp án `3` → `không` (thiếu −3).

## 5. Quy tắc để kết quả có giá trị

- **Chấm độc lập.** Không trao đổi với người kia về từng bài cho đến khi cả hai đã nộp. Được hỏi Người 1 về hướng
  dẫn, nhưng không hỏi về một bài cụ thể.
- **Không mở** `a3_difficulty_key.csv` và không tra độ khó của bài trong kho (`corpus_chunks.jsonl`): các tệp này có
  độ khó tự động, sẽ làm điểm của bạn không còn độc lập.
- Chỉ điền `rating`, `figure_ok`, `answer_ok` và `note`; không sửa, xóa hay sắp xếp lại các cột hoặc dòng khác.

## 6. Lưu và nộp

Mở bằng Excel, điền xong thì lưu đè đúng tên tệp, chọn kiểu **CSV UTF-8 (Comma delimited)**. Kiểu "CSV (Comma
delimited)" sẽ làm mất dấu tiếng Việt. Gửi tệp cho Người 1.

---

## Dành cho Người 1: tổng hợp và phân xử

Khi đã có cả hai tệp:

```powershell
python -m lmsrag.labels.difficulty_agreement
```

Lệnh ghi `reports/tables/a3_difficulty_audit.csv`: Spearman giữa độ khó tự động và điểm trung bình của hai người (kèm
khoảng bootstrap 95%), tách theo nguồn và theo `difficulty_source`; độ khớp giữa hai người (Spearman, κ có trọng số
bậc hai trên thang 1–5, tỷ lệ trùng và lệch tối đa 1 mức); điểm trung bình theo (mức khó, nguồn). Điểm độ khó không
phân xử: thẻ dùng trung bình của hai người.

Các ô `figure_ok`, `answer_ok` hai người trả lời khác nhau được liệt kê ở `a3_difficulty_adjudication.csv`. Điền
`final` (`có`/`không`) và `note_final`, rồi chạy lại để có độ chính xác của bộ lọc Khan kèm khoảng Wilson 95%.

Nếu Spearman của một nguồn thấp rõ rệt, ghi đề xuất vào báo cáo: dùng riêng nguồn đó, hoặc hạ trọng số của nó trong
đánh giá. Không chấm lại để tăng số.
