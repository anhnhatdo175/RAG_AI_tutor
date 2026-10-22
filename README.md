# LMS Adaptive Profile Based RAG

Đề tài xây dựng learner profile vector từ dữ liệu ASSISTments và dùng profile để cá nhân hóa truy xuất tài liệu trong LMS.

## Chạy pipeline duy nhất

```powershell
python -m adaptiverag.run_experiment
```

Pipeline chính nằm trong `adaptiverag/`: temporal split 70/15/15, profile vector, Known-skill retrieval, chọn alpha/beta trên validation và đánh giá trên test.

Kết quả cuối nằm trong `adaptiverag/artifacts/final/`. Corpus tài liệu hiện tại là synthetic instructional scaffold; cần thẩm định hoặc thay bằng tài liệu khóa học thật trước khi triển khai.

## Chạy pipeline `lmsrag` (Layer 1–2)

```powershell
python -m pip install -r requirements.txt
python -m lmsrag.download_data
python -m lmsrag.layer1_data.build
python -m lmsrag.layer2_embedding.build --embedder lsa
```

ASSISTments và Khan Academy sample data được lưu trong repository. GSM8K và MATH được tải vào `lmsrag/data/raw/`; các tệp đó và output sinh ra trong `lmsrag/artifacts/` không được commit, vì có thể tạo lại bằng các lệnh trên. Cần kết nối Internet khi tải corpus lần đầu.

## Layer 3–4: truy xuất thích ứng và gợi ý bài tập

Layer 3 dùng `AdaptiveRetriever`: bắt buộc lọc chính xác theo subject + grade, có thể giới hạn thêm `target_skill`, sau đó dense search bằng chính embedder đã tạo index và Cross-Encoder reranking. Lần đầu dùng Cross-Encoder cần tải model đã cấu hình từ Hugging Face. Ví dụ:

```python
from lmsrag.layer2_embedding.index import load_and_verify
from lmsrag.layer3_retrieval.retriever import (
    AdaptiveRetriever, CrossEncoderReranker, load_query_encoder,
)

store, _ = load_and_verify()
retriever = AdaptiveRetriever(store, load_query_encoder("lsa"), CrossEncoderReranker())
hits = retriever.retrieve(
    "How do I add fractions with different denominators?",
    target_subject="Mathematics",
    target_grade="lower_secondary",
    target_skill="Addition-and-Subtraction-Fractions",
)
```

Layer 4 dùng `HybridRecommender` để xếp hạng bài trong exercise bank theo semantic score, profile similarity được điều chỉnh bởi ZPD và CF. Các thành phần được đưa về [0,1]; `Score_profile` trong công thức hybrid là `normalized_profile_cosine * zpd_score`. Nếu chưa có mastery theo kỹ năng, hệ thống dùng accuracy toàn cục của người học; với cold start dùng mastery trung tính 0.5 và ghi rõ nguồn fallback trong kết quả. Do log ASSISTments không dùng cùng ID bài với kho GSM8K/MATH/Khan, CF ở đây là **canonical-skill-level**, không phải item-level. CF chỉ fit từ tương tác `history`, learner state dùng output lịch sử của Layer 1. Khung khởi tạo:

```python
import json
from lmsrag import config, data
from lmsrag.layer1_data.state_store import StateStore
from lmsrag.layer2_embedding.index import load_and_verify
from lmsrag.layer3_retrieval.retriever import load_query_encoder
from lmsrag.layer4_recommender import HybridRecommender
from lmsrag.skill_mapping import SkillMapper

store, _ = load_and_verify()
bank = [json.loads(line) for line in (config.ART / "exercise_bank.jsonl").read_text(encoding="utf-8").splitlines() if line]
catalog = json.loads(config.SKILL_CATALOG_JSON.read_text(encoding="utf-8"))
logs = data.load_logs(SkillMapper())
recommender = HybridRecommender(store, bank, catalog, logs)
state_store = StateStore(config.ART / "learner_state.sqlite")
state = state_store.get(user_id=123)  # for a read-only/offline demonstration
encoder = load_query_encoder("lsa")
query_vector = encoder.encode_queries(["Practice adding fractions"])[0]
recommendations = recommender.recommend(
    user_id=123,
    state=state,
    query_vector=query_vector,
    target_subject="Mathematics",
    target_grade="lower_secondary",
    target_skill="Addition-and-Subtraction-Fractions",
    top_k=5,
)
state_store.close()
```

Bài đã có trong `state.seen` được loại mặc định; có thể truyền `seen_item_ids` để ghi đè.

Để chuẩn bị context Layer 5, `LearnerState.context_snapshot()` cung cấp accuracy, consistency, pace, canonical mastery và danh sách kỹ năng yếu/mạnh ở dạng JSON-safe. Để chạy feedback Layer 6, dùng một DB live riêng để Layer 1 có thể rebuild snapshot lịch sử mà không xóa tiến độ online. Khởi tạo DB live một lần từ snapshot:

```python
state_store = StateStore(config.ART / "learner_state_live.sqlite")
if state_store.count() == 0:
    state_store.initialize_from_snapshot(config.ART / "learner_state.sqlite")
```

Ghi kết quả chấm theo `canonical_skill_tag` của bài, không ánh xạ ngược tùy tiện sang một source skill ID ASSISTments. `StateStore.apply_canonical_event` cập nhật EMA, trọng số profile theo difficulty bucket, danh sách weak/strong canonical skills và `seen`; các đề xuất tiếp theo sẽ ưu tiên canonical mastery/profile vừa cập nhật:

```python
updated_state = state_store.apply_canonical_event(
    user_id=123,
    skill_tag="Addition-and-Subtraction-Fractions",
    score=0.0,  # auto-graded score in [0, 1]
    rt_ms=25_000,
    alpha=0.25,
    prior=0.5,
    bucket=1,  # 0 easy, 1 medium, 2 hard
    tau_base=25_000,
    task_id="T-123",
)
```

Đối với truy xuất, `load_query_encoder()` đọc loại embedder từ manifest; truyền loại khác sẽ báo lỗi thay vì tạo query vector trong không gian không khớp. Các learner state cũ vẫn đọc được; canonical fields mặc định rỗng và Layer 4 tiếp tục dùng EMA source-skill đã ánh xạ cho đến khi có event canonical.

## Layer 5–6: sinh nội dung và phản hồi thích ứng

Mô tả kiến trúc và trạng thái implementation Layer 5–6, gồm context đa nguồn, JSON prompting, trợ giảng Socratic, chấm điểm, EMA/profile, smoke test Ollama thật và các phần còn cần tích hợp nằm trong [Chương 2.6–2.7](docs/chuong-2-tang-5-6.md).

Implementation có thể gọi trực tiếp qua `lmsrag.layer5_generation` (`GenerationContext`, `LLMGenerator`, `SocraticTutor`) và `lmsrag.layer6_feedback` (`FeedbackLoop`, `GradingResult`). Layer 5 dùng backend đã cấu hình; Layer 6 nhận `StateStore` live và trả learner state/snapshot mới sau khi chấm.

Để chạy inference LLM local thật qua Layer 5 (không dùng mock), cài Ollama, pull model rồi chạy:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_MODELS = "D:\OllamaModels"
ollama pull qwen3:4b
ollama serve
```

Ở một PowerShell khác trong thư mục repository:

```powershell
./.venv/Scripts/python.exe -m lmsrag.layer5_generation.live_smoke_test --model qwen3:4b --base-url http://127.0.0.1:11435 --output output/layer5_live_llm_smoke_test.json
```

Lệnh này gọi model local thật, kiểm tra schema qua Layer 5 và lưu model, endpoint, thời gian chạy cùng kết quả vào JSON.

## Chạy luồng học terminal Layer 1–6

Build dữ liệu và index một lần (Layer 1–2):

```powershell
python -m pip install -r requirements.txt
python -m lmsrag.download_data
python -m lmsrag.layer1_data.build
python -m lmsrag.layer2_embedding.build --embedder lsa
```

Khởi động Ollama trong một PowerShell riêng:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama pull qwen3:4b
ollama serve
```

Trong PowerShell khác, tại thư mục repository, cấu hình backend và bắt đầu phiên học:

```powershell
$env:LLM_BACKEND = "ollama"
$env:GENERATION_MODEL = "qwen3:4b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11435"
python -m lmsrag.app --user-id 123 --skill Addition-and-Subtraction-Fractions
```

Có thể truyền `--query`, `--subject`, hoặc `--grade`; mặc định query được tạo từ skill, môn là `Mathematics`, lớp là `lower_secondary`, và mode là `practice`. Skill cần dùng canonical tag đã có trong corpus. Nhập câu trả lời để chấm và cập nhật learner state; nhập `:quit` để giữ bài đang chờ cho lần chạy tiếp theo.

Ứng dụng nối retrieval, gợi ý bài, sinh nội dung và feedback cho từng lượt học. Snapshot offline `learner_state.sqlite` không bị ghi đè; state online và bài đang chờ được lưu tại `lmsrag/artifacts/learner_state_live.sqlite`. Bài chưa có answer key tự động sẽ không được phát; câu trả lời không chấm được sẽ chuyển sang `needs_review` mà không cập nhật mastery. MVP hiện chấm một lần cho mỗi bài và chưa tự chấm bài tự luận/Socratic nhiều lượt. Lần đầu tạo retriever có thể cần tải Cross-Encoder model từ Hugging Face.
