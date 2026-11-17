# Nhật ký số liệu theo thẻ việc

Một dòng mỗi thẻ: ngày, mã thẻ, commit, số liệu chính, ghi chú.

| Ngày | Thẻ | Commit | Số liệu chính | Ghi chú |
|---|---|---|---|---|
| 17/11/2026 | P1-01 | (PR P1-01) | ASSISTments 2009: 525.534 → 242.764 dòng, 2.065 người học, tỷ lệ đúng 0,656. Dòng trùng (order_id, skill_id) trên bản gốc 123.778; không ánh xạ 38.639 (13,62%). Giữ − loại = −0,036 [−0,045; −0,027]. Last Answer AUC 0,651 (test) | Khớp cả 3 số kỳ vọng. Thêm bước 7 (áp lại ngưỡng 20 dòng sau lọc kỹ năng, loại 2.284 dòng, 154 người học) |
| 17/11/2026 | P1-02 | (PR P1-02) | ASSISTments 2012: 6.123.270 → 2.148.066 dòng, 16.887 người học, tỷ lệ đúng 0,703. Có kỹ năng 2.630.080; có kỹ năng và original=1: 2.543.887; người học ≥20 dòng trước ánh xạ 17.863. Ngoài MasterySection: 752.585 dòng, 12.265 người học. Last Answer AUC test: 0,633 (Mastery) và 0,646 (ngoài Mastery) | Khớp cả 4 số kỳ vọng. Dòng correct lẻ chỉ có 210 sau bước 2 (thẻ ghi 0,09% là tính trên toàn tệp). Giả thuyết "Last Answer được lợi nhân tạo" không được ủng hộ |
| 17/11/2026 | P1-03 | (PR P1-03) | Junyi 2015: 25.925.992 dòng, 247.606 người học, 96.521 người học ≥20 lượt, tỷ lệ đúng 0,828. Mẫu 20.000 người học: 5.111.317 dòng, tỷ lệ đúng 0,832 (= baseline đa số). Có template Khan: 65,8%; problem_type khớp: 89,3%; template dùng được: 32,7% | Khớp cả 5 số kỳ vọng. Lần đo đầu cho 58% vì so với số thứ tự của bản render; đã sửa bằng bảng id dạng bài khan_problem_types.csv |
| 18/11/2026 | P1-04 | (PR P1-04) | Kho v2: 4.548 bài (Khan 2.064, GSM8K 1.983, MATH 501), 15.071 chunk (4.842 worked_example, 10.229 hint_step), kho bài tập 4.047. Độ khó Khan: 1.273 junyi_history, 791 khan_steps. 40 tag chuẩn có chunk ở ≥3 mức. Độ phủ: 2009 77,6%, 2012 84,1% (có template Khan dùng được) | 2012 khớp kỳ vọng; 2009 thấp hơn 80,4% (kỳ vọng tạm), chưa tái lập được, ghi trong corpus_v2.md. 77% GSM8K không gắn được kỹ năng nên bị bỏ |
