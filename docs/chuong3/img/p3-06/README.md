# Ảnh P3-06 cho mục 3.4

Chụp ngày 29/11/2026 trên Moodle 5.3.1+ (moodle-docker, PostgreSQL) với dịch vụ P3-05 trong Docker. Các bài trong ảnh do `qwen3:8b` sinh trên Kaggle T4 trong lần chạy ngày 28/11. Chụp bằng Playwright với Edge headless, cửa sổ 1366 px. Khi chụp đã tắt bảng hiệu năng và thông tin debug của Moodle dev. Giờ trên trang là UTC (múi giờ của container).

| Ảnh | Bước trong luồng | Chú thích đề xuất |
| --- | --- | --- |
| `01-settings.jpg` | Cài đặt | Trang cài đặt plugin: URL dịch vụ, token, chuỗi băm id |
| `02-teacher-generate-and-list.jpg` | 1–2 | Giáo viên chọn kỹ năng, mức khó, số bài; bên dưới là danh sách bài của khóa và trạng thái |
| `03-teacher-edit.jpg` | 3 | Giáo viên sửa bài Draft. Ô đáp án mới để trống vì Moodle không bao giờ nhận đáp án cũ |
| `04-teacher-after-edit-draft.jpg` | 3 | Sau khi sửa, bài quay về Draft và phải duyệt lại |
| `05-teacher-approved.jpg` | 3 | Bài đã duyệt, học sinh làm được |
| `06-learner-practice-list.jpg` | 4 | Học sinh chỉ thấy các bài đã duyệt |
| `07-learner-task.jpg` | 5 | Đề bài, lý do giao bài và mã nguồn trích dẫn (`gsm8k-train-5252`) |
| `08-learner-result.jpg` | 5 | Dịch vụ chấm: đúng, mastery `Percent-Of` 0,500 → 0,550 |
| `09-learner-profile.jpg` | 6 | Hồ sơ năng lực của học sinh: mastery theo kỹ năng, lịch sử trả lời |
| `10-teacher-profiles.jpg` | 6 | Giáo viên xem danh sách học sinh mô phỏng và số lượt làm |
| `11-learner-no-permission.jpg` | Phân quyền | Học sinh mở trang giáo viên thì bị chặn (`local/aitutor:manage`) |

Ở ảnh 03, đáp án do LLM sinh là 125 (kiểm tra tay ngày 28/11, xem `reports/RESULTS.md`, dòng P3-06), trong khi đáp án đúng là 25% × 200 + 30% × 300 = 140. Giáo viên sửa thành 140 trước khi duyệt, và ảnh 08 cho thấy dịch vụ chấm theo đáp án đã sửa. Hai bài `Unit-Rate` trùng đề là từ lượt sinh trước bản sửa "mỗi bài một bài gốc" (commit `2f6dd8e`).
