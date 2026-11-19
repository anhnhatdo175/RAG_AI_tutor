# Rubric v1: gán nhãn gold set truy xuất (P1-07)

Ngày tạo: 20/11/2026, cùng ngày sinh `queries_v1.csv`. Tệp này **không sửa sau khi đã dùng để gán**. Nếu vòng gán thử
cho κ dưới 0,6, nhóm viết `rubric_v2.md` (kèm ví dụ) và giữ nguyên tệp này để báo cáo.

## Truy vấn

Mỗi truy vấn là một học sinh đang luyện một kỹ năng, với một mức thành thạo (mastery), cần một bài giải mẫu ở một mức
khó mục tiêu. Có 36 kỹ năng × 5 mức mastery = 180 truy vấn. Mức khó mục tiêu tính bằng **quy tắc cố định**:

> `target_level = clip(round_half_up(1 + 5 × mastery), 1, 5)`

| mastery | 0,2 | 0,35 | 0,5 | 0,65 | 0,8 |
|---|---|---|---|---|---|
| `target_level` | 2 | 3 | 4 | 4 | 5 |

Ví dụ truy vấn: *"A student with mastery 0.35 on Addition and Subtraction Fractions needs a worked example at difficulty
level 3 of 5."*

## Nhãn cho mỗi cặp (truy vấn, chunk)

| Nhãn | Điều kiện |
|---|---|
| `2` | Đúng kỹ năng; độ khó lệch tối đa 1 mức so với mức mục tiêu theo cảm nhận người gán; nội dung đủ để làm căn cứ sinh một bài mới cùng dạng |
| `1` | Đúng kỹ năng, nhưng độ khó lệch từ 2 mức trở lên, hoặc nội dung quá sơ sài để làm căn cứ |
| `0` | Sai kỹ năng, hoặc cần hình mới hiểu được |

Xét theo thứ tự:

1. **Đúng kỹ năng không?** Cùng tiêu chí với phiếu từ điển kỹ năng (P1-05): bài luyện đúng kỹ năng nếu kỹ năng là
   **phần chính** của bài, không chỉ là một bước phụ. Sai kỹ năng → `0`.
2. **Có cần hình không?** Đề cần một hình, đồ thị, bảng hay trục số mà chunk không có, và thiếu nó thì không làm được
   → `0`. Đề nhắc tới hình nhưng vẫn làm được bằng chữ thì xét tiếp.
3. **Độ khó.** Tự ước lượng mức khó của bài trên thang 1–5, **cùng thang và cùng ví dụ mốc** với phiếu chấm độ khó
   (P1-06, `lmsrag/data/labels/a3_difficulty_HUONG_DAN.md`, mục 2): độ khó toán học với một học sinh THCS đã học phần
   kiến thức đó, không xét độ dài lời giải hay độ khó tiếng Anh. Lệch so với `target_level` từ 2 mức trở lên → `1`.
4. **Nội dung có đủ làm căn cứ không?** Có đề và lời giải cho thấy cách làm, đủ để một người viết được bài mới cùng
   dạng. Chỉ có đáp số, lời giải đứt đoạn, hoặc chunk chỉ là một mảnh của bài dài mà thiếu phần chính → `1`.
5. Qua cả bốn bước → `2`.

Không có mức khó tự động hay tên hệ truy xuất trên phiếu; đừng cố đoán chúng.
