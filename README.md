# LMS AI Tutor — RAG Benchmark trên SciQ

## 1. Cài đặt

```bash
.venv-1\Scripts\Activate.ps1
pip install -r requirements.txt
```

Để cấu hình local, sao chép `.env.example` thành `.env`. Không commit `.env`
vì file này có thể chứa secret hoặc đường dẫn máy cá nhân.

## Chạy local trên Google Colab GPU

Trong Colab, bật `Runtime > Change runtime type > T4 GPU`, rồi cài các
dependency:

```python
!pip install -q -r requirements.txt
```

Sau khi clone project và mount Drive, đặt đường dẫn SciQ trong Colab, ví dụ:

```python
import os
os.environ["SCIQ_DATA_DIR"] = "/content/drive/MyDrive/SciQ"
os.environ["LLM_BACKEND"] = "transformers"
```

Kiểm tra GPU trước khi chạy:

```python
!nvidia-smi
```

Backend mặc định dùng Hugging Face Transformers, không gọi API và không cần
Ollama:

```
LLM_BACKEND=transformers
GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct
JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct
SCIQ_DATA_DIR=D:\SciQ
```

Hai model được load 4-bit NF4 trên GPU. Cần lưu `output/` vào Google Drive
hoặc thư mục persistent khác nếu muốn giữ checkpoint sau khi Colab ngắt.
Nếu Colab hết VRAM, giảm `TRANSFORMERS_CONTEXT_LENGTH` xuống `2048` hoặc dùng
Qwen2.5-3B cho generation.

Backend Ollama vẫn có thể dùng trên máy cá nhân bằng cách đặt
`LLM_BACKEND=ollama`, `GENERATION_MODEL=qwen2.5:3b` và
`JUDGE_MODEL=qwen2.5:1.5b`.

## 2. Chạy pipeline theo thứ tự

```bash
# Bước 1: đọc CSV, xây corpus các đoạn support duy nhất
python -m src.ingestion

# Bước 2: embed corpus bằng model local, xây FAISS index (có thể mất vài phút tuỳ corpus size)
python -m src.retriever

# Bước 3: chạy benchmark RAG vs LLM thuần trên tập câu hỏi mẫu
python -m src.evaluate

# Bước 4: vẽ biểu đồ so sánh
python -m src.make_charts

# Bước 5: chấm faithfulness trên toàn bộ benchmark_results.csv
python -m src.evaluate_faithfulness
```

Kết quả nằm trong thư mục `output/`:
- `benchmark_results.csv` — toàn bộ câu hỏi, câu trả lời, verdict, latency,
  precision/recall và các đoạn retrieved support thực tế từng dòng; được cập
  nhật sau mỗi câu để có thể tiếp tục khi lỗi
- `chart_accuracy.png`, `chart_precision_recall.png`, `chart_latency.png`
- `faithfulness_judgments.json`, `faithfulness_judgments.csv` — phán quyết và lý do
  của judge cho từng câu/phương pháp
- `chart_faithfulness.png`, `faithfulness_report.md` — biểu đồ và bảng tỷ lệ để
  chèn vào báo cáo

## 3. Ghi chú quan trọng

- `config.SAMPLE_SIZE` (mặc định 200) là số câu hỏi lấy mẫu để benchmark — tăng lên nếu muốn kết quả ổn định hơn, nhưng sẽ tốn nhiều API call hơn.
- Benchmark tự đọc `output/benchmark_results.csv` và bỏ qua các câu đã hoàn tất. Chạy lại `python -m src.evaluate` sau lỗi sẽ tiếp tục từ checkpoint.
- Mặc định pipeline chạy model local trực tiếp trên GPU Colab bằng
  Transformers: Qwen2.5-7B-Instruct để sinh và Qwen2.5-3B-Instruct làm judge.
  Hai model khác nhau nên judge độc lập tương đối.
- Có thể đổi server Ollama bằng `OLLAMA_BASE_URL`; model được chọn qua
  `GENERATION_MODEL` và `JUDGE_MODEL`.
- Các file benchmark/faithfulness cũ được tạo bằng model hoặc context cũ không
  nên dùng lại để báo cáo. Hãy xóa hoặc đổi tên checkpoint trong `output/` trước
  khi chạy lại toàn bộ pipeline.
- Thí nghiệm chuẩn dùng `SAMPLE_SIZE=200`, tương ứng 200 câu cho mỗi phương
  pháp và 400 judgment faithfulness.
- Faithfulness có checkpoint JSON nên có thể chạy lại sau khi bị gián đoạn.
- Faithfulness của RAG đối chiếu với các đoạn retrieved support thực tế đã được
  đưa vào prompt; support gold chỉ dùng cho Precision@K/Recall@K.
- Vì model có thể đã "biết" một phần nội dung SciQ qua dữ liệu huấn luyện, Accuracy của nhánh LLM thuần có thể cao hơn dự kiến. Đây là giới hạn chung khi benchmark trên dataset công khai lâu năm — nên phân tích thêm khía cạnh "độ bám sát nguồn" (so `rag_answer` với `support` gốc) trong báo cáo, không chỉ dựa vào đúng/sai.
- Toàn bộ module được tách rời (`ingestion.py` chỉ phụ trách đọc dữ liệu) để sau này thay `D:\SciQ` bằng nội dung khóa học thật của LMS, chỉ cần viết lại hàm `load_all()`/`build_corpus()`, phần còn lại giữ nguyên.
