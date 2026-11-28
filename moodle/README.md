# Moodle Docker và plugin `local_aitutor` (P3-06)

Plugin `local_aitutor` nối Moodle với dịch vụ gia sư `lmsrag/service` (P3-05) qua REST. Moodle chỉ gọi HTTP tới dịch vụ. Đáp án (`answer_key`) và việc chấm bài nằm hoàn toàn ở dịch vụ. Id người dùng và id khóa học được đổi thành bí danh HMAC trước khi gửi.

```
Giáo viên ─ teacher.php ─┐                       ┌─ /generate, /tasks/{id}/review
Học sinh  ─ practice.php ┼─ classes/client.php ──┼─ /submit
           ─ profile.php ┘    (\curl, Bearer)    ├─ /profile/{user_id}
Cron ─ task\sync_attempts ───────────────────────┴─ /sync
```

Luồng bắt buộc: giáo viên đặt kỹ năng, mức khó và số bài → `/generate` → giáo viên **duyệt** bài (dịch vụ chỉ nhận bài làm cho bài đã duyệt) → học sinh mở trang luyện tập và làm bài → `/submit` chấm ở dịch vụ → trang hiện đúng/sai, phản hồi và mastery trước/sau.

Các mục "nên có" đã làm: trang hồ sơ năng lực, giáo viên sửa hoặc loại bài trước khi giao, hiện lý do giao bài và nguồn trích dẫn, tác vụ định kỳ gửi bài làm quiz sang `/sync`.

## 1. Chuẩn bị

- Docker Desktop đang chạy, Git, và **Git Bash** (script `bin/moodle-docker-compose` là bash).
- Ổ chứa dữ liệu Docker còn ít nhất khoảng 10 GB trống. Nếu ổ C: gần đầy, chuyển đĩa ảo của Docker sang ổ khác trước: Docker Desktop → *Settings → Resources → Advanced → Disk image location* → chọn ví dụ `D:\DockerData` → *Apply & restart*.
- Dịch vụ P3-05 chạy trên máy host ở cổng 8800 (xem README gốc, mục Docker):

  ```powershell
  Copy-Item deploy\.env.example .env   # đặt SERVICE_TOKEN
  docker compose -f deploy\docker-compose.yml up --build -d
  Invoke-RestMethod http://localhost:8800/health   # {"status":"ok"}
  ```

## 2. Dựng Moodle bằng `moodlehq/moodle-docker`

Dùng nhánh LTS hiện hành **Moodle 5.3** (`MOODLE_503_STABLE`, ra 10/2026). Nhánh `_STABLE` luôn có bản vá mới nhất, nên hãy `git pull` thường xuyên (Moodle HQ đã cảnh báo lỗ hổng PHP2SHELL công bố ngày 21/10/2026). Không dùng ảnh Bitnami.

Trong Git Bash:

```bash
cd /d
git clone https://github.com/moodlehq/moodle-docker.git
git clone -b MOODLE_503_STABLE --depth 1 https://github.com/moodle/moodle.git /d/moodle-src

export MOODLE_DOCKER_WWWROOT=/d/moodle-src
export MOODLE_DOCKER_DB=pgsql
cd /d/moodle-docker
cp config.docker-template.php $MOODLE_DOCKER_WWWROOT/config.php
```

Chép plugin vào cây Moodle. Từ Moodle 5.1 mã web nằm trong thư mục `public/`, nên plugin đặt ở `public/local/aitutor`. Với Moodle 4.5 thì đặt ở `local/aitutor`.

```bash
cp -r /d/RAG_AI_tutor/moodle/local_aitutor $MOODLE_DOCKER_WWWROOT/public/local/aitutor
```

Khởi động và cài CSDL:

```bash
bin/moodle-docker-compose up -d
bin/moodle-docker-wait-for-db
bin/moodle-docker-compose exec webserver php admin/cli/install_database.php \
  --agree-license --fullname="Moodle Demo" --shortname="demo" \
  --adminpass="test" --adminemail="admin@example.com"
```

Nếu lệnh báo không tìm thấy `admin/cli/install_database.php` (cấu trúc `public/` của 5.1+), chạy lại với `public/admin/cli/install_database.php`. Các lệnh CLI bên dưới cũng đổi tiền tố như vậy.

Moodle chạy tại <http://localhost:8000> (tài khoản `admin` / `test`). Dừng bằng `bin/moodle-docker-compose stop` (giữ dữ liệu); `down` thì xóa container.

## 3. Cài đặt plugin

1. Đăng nhập admin → *Site administration → Notifications* → nâng cấp để cài `local_aitutor`.
2. *Site administration → Plugins → Local plugins → AI tutor*:
   - **Service URL**: `http://host.docker.internal:8800` (container gọi sang dịch vụ trên máy host).
   - **Service token**: trùng `SERVICE_TOKEN` trong `.env` của dịch vụ.
   - **Pseudonym salt**: một chuỗi ngẫu nhiên dài, ví dụ lấy từ `openssl rand -hex 32`. Không đổi sau khi học sinh đã làm bài.
   - **Suggested skills**: mỗi dòng một mã kỹ năng có trong ngân hàng bài của dịch vụ.
3. (Tùy chọn) cài gói tiếng Việt: *Site administration → Language → Language packs → Tiếng Việt*. Plugin có sẵn `lang/vi`.

Plugin gọi dịch vụ bằng `\curl` với `ignoresecurity`, vì `host.docker.internal` là địa chỉ nội bộ và cổng 8800 nằm ngoài danh sách cổng mặc định của Moodle. URL này chỉ admin đặt được.

## 4. Dữ liệu demo

```bash
bin/moodle-docker-compose exec webserver php public/local/aitutor/cli/setup_demo.php --password='Demo#2026'
```

Script tạo khóa **"Toán – Demo"** (`TOAN-DEMO`), giáo viên `gv.demo` và 10 học sinh mô phỏng `hs01`…`hs10` với tên giả. Không có dữ liệu người thật. Chạy lại script không tạo bản trùng.

## 5. Chạy luồng bắt buộc

| Bước | Ai | Ở đâu | Kết quả mong đợi |
| --- | --- | --- | --- |
| 1 | `gv.demo` | Khóa Toán – Demo → *More → AI tutor: tasks* (`/local/aitutor/teacher.php?courseid=<id>`) | Form sinh bài |
| 2 | `gv.demo` | Chọn kỹ năng, mức khó 1–5, số bài → *Generate* | Bài mới ở trạng thái *Draft* hoặc *Needs review* |
| 3 | `gv.demo` | Bấm ✓ (*Approve*); bài *Needs review* thì sửa ✎ trước (thêm đáp án nếu cần) | Trạng thái *Approved* |
| 4 | `hs01` | *More → AI tutor: practice* | Danh sách bài đã duyệt |
| 5 | `hs01` | *Start* → nhập câu trả lời → *Submit answer* | Đúng/sai, phản hồi, bảng mastery trước/sau |
| 6 | `hs01` hoặc `gv.demo` | *More → AI tutor: profile* | Mastery theo kỹ năng, kỹ năng yếu/mạnh, lịch sử |

Mức khó 1–5 được gửi sang dịch vụ thành `difficulty` 0,1 / 0,3 / 0,5 / 0,7 / 0,9. Dịch vụ tìm bài gốc trong khoảng ±0,10 quanh giá trị đó. Nếu báo lỗi 404 "No mapped source exercise…", hãy chọn kỹ năng hoặc mức khó khác.

Nếu dịch vụ tắt khi học sinh nộp bài, bài làm vẫn được lưu ở trạng thái *Waiting to be sent*, và tác vụ định kỳ sẽ gửi lại sau.

## 6. Bài làm từ quiz và tác vụ `/sync`

Tác vụ `\local_aitutor\task\sync_attempts` chạy 15 phút một lần:

1. Đọc các lượt quiz đã nộp (`$DB`, bảng `question_attempts` và `quiz_attempts`) của những câu hỏi có **ID number** dạng `aitutor:<task_id>` trong ngân hàng câu hỏi. Câu hỏi đó phải ứng với một bài đã duyệt của plugin.
2. Gửi mọi lượt đang chờ (từ quiz hoặc từ trang luyện tập lúc dịch vụ tắt) sang `/sync`, mỗi lô 100 lượt. Lượt bị dịch vụ từ chối được đánh dấu *Rejected* kèm lý do, và không chặn các lượt khác.

Chạy tay để kiểm tra:

```bash
bin/moodle-docker-compose exec webserver php admin/cli/scheduled_task.php --execute='\local_aitutor\task\sync_attempts'
```

## 7. Cấu trúc plugin

| File | Vai trò |
| --- | --- |
| `version.php`, `settings.php` | Phiên bản; cài đặt URL, token, chuỗi băm, timeout, kỹ năng gợi ý |
| `db/access.php` | `local/aitutor:manage` (giáo viên), `local/aitutor:practice` (học sinh) |
| `db/install.xml` | Bảng `local_aitutor_task` (bài của khóa, không có đáp án), `local_aitutor_attempt` (bài làm, khóa idempotency duy nhất) |
| `db/tasks.php`, `classes/task/sync_attempts.php` | Tác vụ định kỳ gửi `/sync` |
| `classes/client.php` | Gọi REST tới dịch vụ, tạo bí danh HMAC |
| `classes/store.php` | Lưu bài, lượt làm, hiển thị trạng thái và mastery |
| `classes/form/*` | Form sinh bài, sửa bài, trả lời |
| `classes/privacy/provider.php` | Khai báo và xuất/xóa dữ liệu cá nhân (GDPR) |
| `teacher.php`, `practice.php`, `profile.php` | Ba trang chính |
| `lib.php` | Thêm liên kết vào menu khóa học |
| `cli/setup_demo.php` | Tạo dữ liệu demo |
| `lang/en`, `lang/vi` | Chuỗi tiếng Anh và tiếng Việt |

Nộp trùng: mỗi lần mở form trả lời sẽ sinh một khóa ngẫu nhiên. Gửi lại cùng một form (bấm hai lần, tải lại trang) vẫn dùng chung khóa idempotency, nên dịch vụ trả kết quả đã lưu và không cập nhật mastery lần hai.

LTI 1.3 không làm: plugin cục bộ gọi REST là đủ cho phạm vi đồ án. LTI để làm hướng mở rộng (Chương 3).
