<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Chuỗi tiếng Việt cho local_aitutor.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'Gia sư AI';
$string['aitutor:manage'] = 'Sinh và duyệt bài của Gia sư AI';
$string['aitutor:practice'] = 'Luyện tập bài của Gia sư AI';

// Cài đặt.
$string['serviceurl'] = 'URL dịch vụ';
$string['serviceurl_desc'] = 'Địa chỉ gốc của dịch vụ gia sư. Từ container Moodle Docker, dịch vụ chạy trên máy host có địa chỉ http://host.docker.internal:8800.';
$string['servicetoken'] = 'Token dịch vụ';
$string['servicetoken_desc'] = 'Trùng với SERVICE_TOKEN trong file .env của dịch vụ. Gửi kèm dạng "Authorization: Bearer".';
$string['idsalt'] = 'Chuỗi băm id';
$string['idsalt_desc'] = 'Khóa bí mật dùng để đổi id người dùng và khóa học của Moodle thành bí danh trước khi gửi đi. Không đổi sau khi học sinh đã bắt đầu, nếu không lịch sử của họ trên dịch vụ sẽ mất.';
$string['timeout'] = 'Thời gian chờ (giây)';
$string['timeout_desc'] = 'Sinh bài bằng LLM chạy local có thể mất vài phút.';
$string['skills'] = 'Kỹ năng gợi ý';
$string['skills_desc'] = 'Mỗi dòng một mã kỹ năng, hiện trong form của giáo viên. Giáo viên vẫn gõ được mã khác.';

// Lỗi.
$string['error_notconfigured'] = 'Gia sư AI chưa được cấu hình: hãy nhập URL và token dịch vụ trong trang cài đặt plugin.';
$string['error_nosalt'] = 'Gia sư AI chưa được cấu hình: hãy nhập chuỗi băm id trong trang cài đặt plugin.';
$string['error_unreachable'] = 'Không kết nối được dịch vụ gia sư ({$a}).';
$string['error_httperror'] = 'Dịch vụ gia sư từ chối yêu cầu ({$a}).';
$string['error_badresponse'] = 'Dịch vụ gia sư trả về dữ liệu không phải JSON.';
$string['formexpired'] = 'Form này không còn hợp lệ. Hãy tải lại trang rồi thử lại.';
$string['notenrolled'] = 'Người dùng này không ghi danh trong khóa học.';

// Trang.
$string['teacherpage'] = 'Gia sư AI: bài tập';
$string['practicepage'] = 'Gia sư AI: luyện tập';
$string['profilepage'] = 'Gia sư AI: hồ sơ năng lực';
$string['task_sync'] = 'Gửi bài làm quiz và các lượt đang chờ sang dịch vụ gia sư';

// Trang giáo viên.
$string['generateheader'] = 'Sinh bài tập';
$string['skilltag'] = 'Kỹ năng';
$string['skilltag_help'] = 'Mã kỹ năng mà dịch vụ biết, ví dụ Unit-Rate. Dịch vụ cần ít nhất một bài gốc của kỹ năng này ở mức khó đã chọn.';
$string['skilltag_choose'] = 'Chọn hoặc gõ kỹ năng';
$string['level'] = 'Mức khó';
$string['level1'] = '1 (rất dễ)';
$string['level2'] = '2 (dễ)';
$string['level3'] = '3 (trung bình)';
$string['level4'] = '4 (khó)';
$string['level5'] = '5 (rất khó)';
$string['count'] = 'Số bài';
$string['grade'] = 'Lớp (không bắt buộc)';
$string['grade_help'] = 'Để trống để dịch vụ tự chọn lớp của bài gốc.';
$string['generate'] = 'Sinh bài';
$string['generated'] = 'Đã sinh {$a->count} bài; {$a->review} bài cần duyệt lại. Bài phải được duyệt thì học sinh mới thấy.';
$string['tasklist'] = 'Bài tập của khóa học';
$string['notasks'] = 'Chưa có bài nào. Hãy dùng form phía trên.';
$string['attempts'] = 'Số lượt làm';
$string['needsreview_help'] = 'Kết quả của mô hình không qua được bước kiểm tra. Hãy sửa bài (thêm đáp án nếu cần) trước khi duyệt.';
$string['edittask'] = 'Sửa bài';
$string['prompt'] = 'Đề bài';
$string['hints'] = 'Gợi ý';
$string['hints_help'] = 'Mỗi dòng một gợi ý, tối đa 10.';
$string['reason'] = 'Lý do giao bài';
$string['answerkey'] = 'Đáp án mới';
$string['answerkey_help'] = 'Đáp án hiện tại nằm trên dịch vụ và không bao giờ hiện trong Moodle. Để trống để giữ nguyên.';
$string['toomanyhints'] = 'Tối đa 10 gợi ý.';
$string['review_approve'] = 'Duyệt';
$string['review_edit'] = 'Sửa';
$string['review_reject'] = 'Loại';
$string['review_approve_done'] = 'Đã duyệt bài: học sinh có thể luyện tập.';
$string['review_reject_done'] = 'Đã loại bài.';
$string['review_edit_done'] = 'Đã lưu bài ở trạng thái nháp. Hãy duyệt để giao cho học sinh.';

// Trạng thái.
$string['status_draft'] = 'Nháp';
$string['status_needs_review'] = 'Cần duyệt';
$string['status_approved'] = 'Đã duyệt';
$string['status_rejected'] = 'Bị loại';
$string['status_pending'] = 'Đang chờ gửi';
$string['status_graded'] = 'Đã chấm';
$string['status_processing'] = 'Đang chấm';
$string['statusinfo_pending'] = 'Không kết nối được dịch vụ. Bài làm của em đã được lưu và sẽ được chấm tự động.';
$string['statusinfo_needs_review'] = 'Giáo viên sẽ kiểm tra câu trả lời này.';
$string['statusinfo_processing'] = 'Bài làm đang được chấm. Tải lại trang sau ít phút.';
$string['statusinfo_rejected'] = 'Dịch vụ không nhận câu trả lời này.';

// Trang luyện tập.
$string['nopractice'] = 'Khóa học chưa có bài nào được duyệt.';
$string['lastresult'] = 'Kết quả gần nhất';
$string['start'] = 'Làm bài';
$string['tryagain'] = 'Làm lại';
$string['youranswer'] = 'Câu trả lời';
$string['submitanswer'] = 'Nộp bài';
$string['savedforlater'] = 'Dịch vụ gia sư đang không hoạt động. Bài làm đã được lưu và sẽ được chấm sau.';
$string['result'] = 'Kết quả';
$string['correct'] = 'Đúng';
$string['incorrect'] = 'Chưa đúng';
$string['newmastery'] = 'Mức thành thạo';
$string['masterybefore'] = 'Trước';
$string['masteryafter'] = 'Sau';
$string['showhints'] = 'Xem gợi ý';
$string['sources'] = 'Nguồn';
$string['backtolist'] = 'Quay lại danh sách bài';

// Trang hồ sơ.
$string['nolearners'] = 'Khóa học chưa có học sinh.';
$string['viewprofile'] = 'Xem hồ sơ';
$string['totalattempts'] = 'Số câu trả lời đã được dịch vụ chấm: {$a}';
$string['coldstart'] = 'Chưa có câu trả lời nào: mọi kỹ năng bắt đầu ở mức thành thạo 0,5.';
$string['weakskills'] = 'Kỹ năng yếu';
$string['strongskills'] = 'Kỹ năng mạnh';
$string['mastery'] = 'Mức thành thạo';
$string['practisedskills'] = 'Kỹ năng đã luyện';
$string['allskills'] = 'Tất cả kỹ năng ({$a})';
$string['history'] = 'Các câu trả lời gần nhất';

// Quyền riêng tư.
$string['privacy:metadata:attempt'] = 'Câu trả lời của học sinh cho bài của Gia sư AI.';
$string['privacy:metadata:attempt:userid'] = 'Học sinh.';
$string['privacy:metadata:attempt:taskid'] = 'Bài được trả lời.';
$string['privacy:metadata:attempt:answer'] = 'Câu trả lời.';
$string['privacy:metadata:attempt:correct'] = 'Câu trả lời có đúng không.';
$string['privacy:metadata:attempt:feedback'] = 'Phản hồi của dịch vụ.';
$string['privacy:metadata:attempt:masteryafter'] = 'Mức thành thạo kỹ năng sau câu trả lời.';
$string['privacy:metadata:attempt:timecreated'] = 'Thời điểm trả lời.';
$string['privacy:metadata:task'] = 'Bài được sinh cho một khóa học.';
$string['privacy:metadata:task:createdby'] = 'Giáo viên đã sinh bài.';
$string['privacy:metadata:service'] = 'Câu trả lời được gửi sang dịch vụ gia sư để chấm và cập nhật mô hình người học. Người dùng chỉ được nhận diện bằng bí danh có khóa.';
$string['privacy:metadata:service:pseudonym'] = 'Bí danh HMAC của id người dùng Moodle.';
$string['privacy:metadata:service:answer'] = 'Câu trả lời.';
