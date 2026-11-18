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
hiện chưa có tag nào (ma trận, hóa học…); 35 dòng sau có `tag_now = rule:<tag>`, tức đang được gắn bằng quy tắc
văn bản theo từ khóa. Quy tắc chạy trên từng bài nên các bài cùng template có thể khác tag; cột ghi tag phổ biến nhất.

| `decision` | Khi nào |
|---|---|
| `keep` | (dòng `rule:`) tag hiện tại đúng kỹ năng |
| `add` | Template thuộc một kỹ năng ASSISTments khác tag hiện tại; ghi tên kỹ năng vào `assistments_skill`, đúng như cột `source_skill` của `reports/tables/a1_skill_counts_*.csv` (có dấu cách, ví dụ `Addition Whole Numbers`); nhiều kỹ năng cách nhau bằng `;` |
| `none` | Không thuộc kỹ năng nào của đề tài (ví dụ ma trận, hóa học); template sẽ mang tag `khan:<slug>`, bỏ tag quy tắc |

## Phiếu D · `a4_review_D_recheck_drops.csv` (35 dòng, xem lại sau phiếu A)

Sinh bằng `python -m lmsrag.labels.review_recheck`. Đây là các cặp bị `drop` ở phiếu A thuộc một trong hai loại dễ nhầm:

- **Template không dùng được** (`needs_figure`, `no_hints`, `not_autogradable`; 24 dòng). Chỉ xét kỹ năng:
  học sinh làm bài này có đang luyện kỹ năng đó không? Việc template cần hình **không** phải lý do để `drop`.
  Template loại này không vào kho, nhưng vẫn gắn tag cho lượt làm Junyi (cột `junyi_responses`; ví dụ
  `addition_1` có gần 900.000 lượt).
- **Template dùng được nhưng có nhiều dạng bài** (11 dòng). Phiếu A chỉ in dạng đầu tiên. Ví dụ
  `mean_median_and_mode` có dạng tính trung vị, `discount_tax_and_tip_word_problems` có dạng giảm giá.

Cột `all_types` in một bài mẫu của **mọi** dạng bài. Với template `no_hints`, đề nằm trong widget nên chỉ thấy đáp án;
dựa vào tên template (`template_title`). Điền `decision` = `keep` hoặc `drop` và `note`. Để trống thì giữ quyết định ở
phiếu A. Quy tắc: giữ cặp nếu có ít nhất một dạng bài luyện đúng kỹ năng **và** các dạng còn lại cùng chủ đề gần
(ví dụ trung bình, trung vị, yếu vị trong một bài); bỏ nếu các dạng còn lại là kỹ năng khác hoặc khó hơn hẳn. Hai kỹ
năng có chung `canonical_tag` (Rate và Unit Rate đều là `Unit-Rate`) phải được xét cùng nhau, vì cặp của kỹ năng này
cũng đưa bài vào tag của kỹ năng kia.

## Sau khi rà xong

Lưu đè ba tệp này, giữ nguyên tên. Trong Excel, chọn **CSV UTF-8 (Comma delimited)**; kiểu "CSV (Comma
delimited)" lưu theo bảng mã ANSI và làm mất dấu tiếng Việt trong `note`. Ước tính thời gian: khoảng 1,5–2 giờ cho
phiếu A, 20 phút cho B, 15 phút cho C.

Bước 2 dựng từ điển v1 từ ba phiếu:

```powershell
python -m lmsrag.labels.build_skill_map
```

Script kiểm tra mọi quyết định, slug (so với `khan_templates.csv`) và tên kỹ năng (so với `reports/tables/a1_skill_counts_*.csv`);
nếu có lỗi thì liệt kê và không ghi tệp. Đầu ra: `lmsrag/data/mappings/skill_map_v1.csv` (cột `review` ghi phiếu và
quyết định của mỗi dòng, ví dụ `A:keep`, `B:add`, `C:none`) và `skill_map_v1_rejected.csv` (các cặp bị bỏ). Sau đó chạy
lại P1-03 và P1-04 (`configs/data.yaml` đã trỏ tới v1). Bước 3 là phiếu 150 cặp cho Người 2 và Người 3 gán độc lập để tính κ.
