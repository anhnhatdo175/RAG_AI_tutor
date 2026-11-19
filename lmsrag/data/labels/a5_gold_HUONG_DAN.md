# Hướng dẫn gán nhãn gold set (P1-07)

Dành cho **Người 1** (tệp `a5_gold_pilot_nguoi1.csv`, `a5_gold_nguoi1.csv`) và **Người 3** (tệp
`a5_gold_pilot_nguoi3.csv`, `a5_gold_nguoi3.csv`). Tiêu chí gán nằm ở **`lmsrag/data/gold/rubric_v1.md`**; đọc kỹ
rubric trước khi bắt đầu. Tài liệu này chỉ nói cách làm.

| Đợt | Tệp | Hạn | Khối lượng |
|---|---|---|---|
| Gán thử | `a5_gold_pilot_<người>.csv` (20 truy vấn) | 1–2/12 | khoảng 250–300 cặp, 1–1,5 giờ |
| Gán đầy đủ | `a5_gold_<người>.csv` (160 truy vấn còn lại) | 3–5/12 | khoảng 2.000 cặp, chia 4–5 buổi ngắn |

Gán thử xong thì nộp ngay: nhóm tính κ, và nếu κ dưới 0,6 thì ba người họp 30 phút để viết rubric v2 rồi gán lại phần
thử, **trước khi** vào đợt đầy đủ.

## Các cột trong phiếu

| Cột | Nội dung |
|---|---|
| `query_id`, `skill`, `skill_meaning` | Truy vấn và kỹ năng (giải thích bằng tiếng Việt) |
| `mastery`, `target_level` | Mức thành thạo của học sinh và mức khó mục tiêu 1–5 |
| `content` | Chunk cần chấm: đề, lời giải, đáp án (công thức đã đổi sang dạng dễ đọc) |
| `label` | **Bạn điền**: `0`, `1` hoặc `2` theo rubric |
| `note` | Ghi chú nếu cần, nhất là khi phân vân |
| `content_tex` | Bản gốc dạng TeX, chỉ để đối chiếu; có thể ẩn |

Các dòng của cùng một truy vấn nằm liền nhau (10–15 dòng). Đọc truy vấn một lần, rồi chấm từng chunk. Chấm từng chunk
theo rubric, **không** so các chunk với nhau: không cần có chunk nào được `2`, cũng không giới hạn số `2`.

## Quy tắc để kết quả có giá trị

- **Gán độc lập.** Không trao đổi với người kia về từng cặp cho đến khi cả hai đã nộp.
- **Không mở** `lmsrag/data/gold/pool_provenance.csv` (chunk nào do hệ nào tìm ra) và không tra độ khó của chunk trong
  `corpus_chunks.jsonl`. Hai thông tin này phải ẩn để nhãn không thiên về hệ nào.
- Chỉ điền `label` và `note`; không sửa, xóa hay sắp xếp lại các cột hoặc dòng khác.
- Mỗi buổi gán không quá khoảng 1,5 giờ; mệt dễ làm lệch nhãn.

## Lưu và nộp

Mở bằng Excel, lưu đè đúng tên tệp, chọn kiểu **CSV UTF-8 (Comma delimited)**. Gửi tệp cho Người 1, hoặc đặt vào
`lmsrag/data/labels/`.

---

## Dành cho Người 1: các lệnh

```powershell
python -m lmsrag.labels.gold_queries     # 180 truy vấn -> lmsrag/data/gold/queries_v1.csv (đã chạy 20/11)
python -m lmsrag.labels.gold_pool        # sau khi có pool_runs của Người 2: pool, provenance, phiếu
python -m lmsrag.labels.gold_finalize    # κ gán thử; sau đợt đầy đủ: phiếu phân xử cho Người 2; cuối cùng: gold_v1.csv
```

Người 2 phân xử trong `a5_gold_adjudication.csv` (điền `label_final`), không mở `pool_provenance.csv`. Khi mọi cặp có
nhãn cuối, `gold_finalize` ghi `gold_v1.csv` và mã SHA-256 (vào `gold_v1.sha256` và `prereg/hypotheses.md`); sau đó
commit, gắn tag `gold-v1` và push tag. Sau tag, không ai sửa `gold_v1.csv`.
