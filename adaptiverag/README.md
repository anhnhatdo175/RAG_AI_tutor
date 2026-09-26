# Adaptive Profile Based RAG

Đây là pipeline duy nhất của đề tài, sử dụng ASSISTments Skill Builder và đánh giá track Known-skill. Skill hiện tại do LMS cung cấp và được dùng làm constraint; learner profile được dùng để cá nhân hóa difficulty của tài liệu.

## Chạy thí nghiệm

```powershell
python -m adaptiverag.run_experiment
```

Pipeline thực hiện: đọc interaction, chia temporal 70/15/15, xây profile vector từ history, tạo embedding cho corpus, chọn alpha/beta trên validation, sau đó đánh giá Traditional RAG và Adaptive RAG trên test.

```text
Traditional: final_score = semantic_score
Adaptive:    final_score = alpha * semantic_score + beta * profile_score
```

ASSISTments không có đầy đủ problem statement nên query dùng skill name và problem ID. Corpus tài liệu hiện tại là synthetic instructional scaffold và cần được thay hoặc thẩm định bằng tài liệu học tập thật.

## Cấu trúc

- `build_knowledge_base.py`: tạo corpus scaffold.
- `canonical_pipeline.py`: profile, retrieval, evaluation và báo cáo.
- `run_experiment.py`: entrypoint duy nhất.
- `skill_builder_data.csv`: dữ liệu ASSISTments.
- `corpus_manifest.json`, `materials/`: corpus và metadata.
- `artifacts/final/`: chỉ chứa kết quả cuối.

## Kết quả cuối

- `known_skill_report.json`, `known_skill_report.md`
- `known_skill_comparison.csv`, `known_skill_comparison.png`
- `known_skill_per_student.csv`, `known_skill_per_student.png`
- `profiles.json`
