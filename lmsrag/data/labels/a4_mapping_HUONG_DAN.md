# Hướng dẫn gán nhãn: kiểm định từ điển kỹ năng (P1-05)

Dành cho **Người 2** (tệp `a4_mapping_nguoi2.csv`) và **Người 3** (tệp `a4_mapping_nguoi3.csv`). Hạn nộp: 2/12.
Thời gian: khoảng 45–60 phút cho 150 dòng. Nên chia làm 2 lần.

## Bối cảnh (không cần biết gì thêm về dữ liệu)

Hệ thống gợi ý bài cho học sinh theo **kỹ năng**. Tên kỹ năng lấy từ ASSISTments, một hệ thống luyện toán của Mỹ, nên
bằng tiếng Anh, ví dụ "Percent Of". Bài toán lấy từ ba kho đề tiếng Anh: Khan Academy, GSM8K (bài toán có lời văn) và
MATH. Một từ điển ghép mỗi kỹ năng với các bài toán; việc của bạn là kiểm tra những cặp ghép đó có đúng không. Bạn chỉ
cần đọc hiểu toán trung học cơ sở bằng tiếng Anh.

## Các cột trong phiếu

| Cột | Nội dung |
|---|---|
| `skill` | Tên kỹ năng (tiếng Anh) |
| `skill_meaning` | **Giải thích kỹ năng bằng tiếng Việt.** Đọc cột này trước khi chấm |
| `source`, `title` | Bài lấy từ kho nào, tên dạng bài (chỉ có với Khan Academy) |
| `problem`, `answer` | Đề bài và đáp án. Công thức đã đổi sang dạng dễ đọc: `(3)/(4)` là phân số 3/4, `x^2` là x², `×` `÷` `√` `≤` như thường lệ. Với bài trắc nghiệm, các lựa chọn nằm ở các dòng cuối của đề |
| `label`, `note` | **Bạn điền** |
| `problem_tex` | Đề bài gốc (dạng TeX), chỉ để đối chiếu khi bản dễ đọc khó hiểu; có thể ẩn cột này |

## Bạn cần làm gì

Mỗi dòng là một cặp: **một kỹ năng** (cột `skill`, giải thích ở `skill_meaning`) và **một bài toán** (cột `problem`).
Trả lời một câu hỏi:

> Một học sinh đang luyện kỹ năng `skill`, nếu được giao bài này, **có đang luyện đúng kỹ năng đó không?**

| `label` | Khi nào |
|---|---|
| `1` | Bài luyện đúng kỹ năng, kể cả khi bài dễ hơn hoặc khó hơn mức thường gặp |
| `0` | Bài luyện kỹ năng khác, hoặc kỹ năng chỉ là một bước phụ không đáng kể của bài |

- Chỉ xét **bài được in trên dòng**. Không đoán các bài khác cùng nguồn.
- Độ khó không phải tiêu chí. Ví dụ: kỹ năng "Addition and Subtraction Fractions" với bài cộng hai phân số khác mẫu
  rất dài vẫn là `1`.
- Bài nhiều bước: là `1` nếu kỹ năng là **phần chính** của bài. Ví dụ: kỹ năng "Percent Of" với bài "Tìm 15% của 80
  rồi cộng thêm 5" là `1`; kỹ năng "Addition Whole Numbers" với cùng bài đó là `0`, vì phép cộng chỉ là bước phụ.
- Đề nhắc tới hình mà không có hình: vẫn xét kỹ năng, ghi "thiếu hình" vào `note`.
- Phân vân thì vẫn chọn `0` hoặc `1`, rồi ghi lý do vào `note`. Không để trống `label`.
## Ví dụ luyện tập (không có trong phiếu)

Đọc 6 ví dụ này trước khi bắt đầu.

| Kỹ năng | Bài | `label` | Vì sao |
|---|---|---|---|
| Percent Of (tìm a% của một số) | What is 20% of 45? | `1` | Đúng dạng tìm a% của một số |
| Percent Of | A shirt costs $40. It is on sale for 25% off. What is the sale price? | `1` | Phần chính là tính 25% của 40, rồi trừ đi; dù bài này cũng hợp với kỹ năng giảm giá |
| Addition Whole Numbers (cộng số tự nhiên) | A shirt costs $40. It is on sale for 25% off. What is the sale price? | `0` | Có phép trừ ở bước cuối nhưng phần chính là phần trăm |
| Equation Solving Two or Fewer Steps (giải phương trình 1–2 bước) | Solve 3x − 4 = 11. | `1` | Hai bước: cộng 4, chia 3 |
| Equation Solving Two or Fewer Steps | Solve 2(x + 3) = 5x − 9. | `0` | Cần phá ngoặc và chuyển vế, nhiều hơn hai bước; đây là kỹ năng "More Than Two Steps" |
| Mean (trung bình cộng) | What is the median of 3, 8, 5, 9, 4? | `0` | Hỏi trung vị, không phải trung bình cộng |

Các ví dụ cho thấy hai điều: tên kỹ năng phải khớp với **phần chính** của bài, và hai kỹ năng gần nhau (trung bình và
trung vị; 2 bước và nhiều bước) phải được tách rõ.

## Quy tắc để kết quả có giá trị

- **Gán độc lập.** Không trao đổi với người kia về từng dòng cho đến khi cả hai đã nộp. Được hỏi Người 1 về hướng dẫn,
  nhưng không hỏi về một dòng cụ thể.
- **Không mở** `a4_mapping_key.csv`, `lmsrag/data/mappings/skill_map_v1.csv` và các tệp `a4_review_*`. Các tệp này cho biết
  cặp được tạo bằng cách nào và quyết định của người rà trước, nên sẽ làm nhãn của bạn không còn độc lập.
- Chỉ điền cột `label` và `note`; không sửa, xóa hay sắp xếp lại các cột hoặc dòng khác.

## Lưu và nộp

Mở bằng Excel, điền xong thì lưu đè đúng tên tệp, chọn kiểu **CSV UTF-8 (Comma delimited)**. Kiểu "CSV (Comma
delimited)" sẽ làm mất dấu tiếng Việt. Gửi tệp cho Người 1, hoặc commit lên một nhánh riêng (ví dụ `p2/a4-labels`).

---

## Dành cho Người 1: phân xử

Khi đã có cả hai tệp:

```powershell
python -m lmsrag.labels.agreement
```

Lệnh in κ và tỷ lệ đồng thuận **trước phân xử**, ghi bảng `reports/tables/a4_mapping_audit.csv`, và liệt kê các dòng hai
người gán khác nhau vào `a4_mapping_adjudication.csv`. Điền `label_final` (0 hoặc 1) và `note_final` cho từng dòng đó,
rồi chạy lại lệnh để có **độ chính xác sau phân xử** kèm khoảng Wilson 95%, tách theo `method`.

Nếu κ dưới 0,6: ghi lại lý do (bảng có thêm PABAK và tỷ lệ nhãn 1 của từng người để kiểm tra có phải do gần như mọi
cặp đều đúng) và cách đã sửa hướng dẫn. **Không gán lại để tăng số.** Sau đó sửa từ điển theo các cặp có nhãn cuối là
0, tăng `map_version`, rồi chạy lại P1-03 và P1-04.
