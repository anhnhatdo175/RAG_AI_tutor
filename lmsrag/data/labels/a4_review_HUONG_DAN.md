# Hướng dẫn rà soát bản nháp từ điển kỹ năng (P1-05, bước 1 — Người 1)

Ba phiếu trong thư mục này được sinh bằng `python -m lmsrag.labels.review_draft`. Mở bằng Excel; tệp dùng
UTF-8 có BOM nên tiếng Việt và ký hiệu TeX hiển thị đúng. **Chỉ điền các cột `decision`, `khan_slugs`
(hoặc `assistments_skill`) và `note`**; không sửa các cột khác, vì script sẽ đọc lại theo cột `row`.

Câu hỏi khi rà mỗi dòng: *Một học sinh đang luyện kỹ năng ASSISTments này, nếu được giao bài mẫu của template, có
đang luyện đúng kỹ năng đó không?*

## Phiếu A · `a4_review_A_draft_pairs.csv` (296 dòng)

Mỗi dòng là một cặp (kỹ năng ASSISTments, template Khan) trong bản nháp, kèm tiêu đề template, một bài mẫu, số
lượt làm ở 2009/2012 và tỷ lệ đúng Junyi.

| `decision` | Khi nào |
|---|---|
| `keep` | Template luyện đúng kỹ năng |
| `drop` | Template không luyện kỹ năng này; xóa cặp |
| `fix` | Kỹ năng đúng nhưng nên dùng template khác; ghi slug đúng vào `khan_slugs` (nhiều slug cách nhau bằng `;`) |
| `none` | (dòng `no_khan_template`) xác nhận không có template phù hợp; ghi lý do vào `note` |

- Làm **8 dòng `priority = 1` trước**. Đây là các cặp mà kế hoạch yêu cầu xem kỹ: Table → `reading_tables_*`;
  Unit Rate → `rate_problems_2`; Write Linear Equation from Graph → `slope_intercept_form`;
  Estimation → `rounding_*`; Volume/Surface Area Cylinder, Volume Sphere → `solid_geometry`.
- Các dòng `template_status` khác `usable` (`needs_figure`, `not_autogradable`, `no_hints`) không có bài trong kho.
  Vẫn đánh giá cặp đó có đúng kỹ năng không; script tự bỏ qua chúng khi dựng kho.
- Các dòng còn lại đã được xếp theo số lượt làm giảm dần. Ưu tiên rà kỹ những dòng đầu, vì chúng ảnh hưởng nhiều lượt nhất.

## Phiếu B · `a4_review_B_new_2012_skills.csv` (19 dòng)

Đây là các kỹ năng chỉ có ở ASSISTments 2012, từ 1.000 dòng trở lên, mà bản nháp chưa có. `candidate_1..3` là
template gợi ý theo số từ trùng nhau; **gợi ý chỉ để tham khảo**, nhiều dòng sẽ không có template phù hợp.

| `decision` | Khi nào |
|---|---|
| `add` | Có template phù hợp; ghi slug vào `khan_slugs` (có thể khác các gợi ý) |
| `none` | Không có template phù hợp; ghi lý do vào `note` (ví dụ "cần hình", "chỉ có trong ASSISTments") |

Danh sách đầy đủ template dùng được: `lmsrag/data/khan_v2/khan_templates.csv` (cột `status = usable`).

## Phiếu C · `a4_review_C_untagged_templates.csv` (45 dòng)

Đây là các template dùng được mà từ điển và Junyi chưa gắn kỹ năng. 10 dòng đầu có `tag_now = khan:<slug>`, tức
hiện chưa có tag nào (ma trận, hóa học…); 35 dòng sau đang được gắn bằng quy tắc văn bản.

| `decision` | Khi nào |
|---|---|
| `add` | Template thuộc một kỹ năng ASSISTments; ghi tên kỹ năng vào `assistments_skill` |
| `none` | Không thuộc kỹ năng nào của đề tài (ví dụ ma trận, cân bằng phương trình hóa học); giữ tag hiện tại |

## Sau khi rà xong

Lưu đè ba tệp này, giữ nguyên tên, rồi báo lại. Bước tiếp theo sẽ dựng `skill_map_v1.csv` từ quyết định của bạn,
chạy lại P1-03 và P1-04, rồi sinh phiếu 150 cặp cho Người 2 và Người 3 gán độc lập để tính κ. Ước tính thời gian:
khoảng 1,5–2 giờ cho phiếu A, 20 phút cho B, 15 phút cho C.
