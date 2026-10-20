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
