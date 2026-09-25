# ASSISTments Adaptive RAG data preparation

This directory contains the controlled data slice used to develop the
profile-aware retrieval experiment.

## Retrieval pipeline

[`adaptive_rag.py`](./adaptive_rag.py) implements:

1. section-aware chunking;
2. SentenceTransformers embeddings;
3. a cosine-similarity FAISS index (`IndexFlatIP` with normalized vectors);
4. metadata filtering by `skill_id`, `difficulty`, and `content_type`;
5. Traditional RAG retrieval (`alpha=1`, `beta=0`);
6. profile-aware reranking using mastery and recommended difficulty;
7. Adaptive RAG scoring;
8. alpha/beta grid experiments;
9. retrieval error analysis.

Traditional and Adaptive RAG use the same chunks and FAISS index. The only
change is the profile term:

```text
traditional_score = semantic_score
adaptive_score = alpha * semantic_score + beta * profile_match_score
```

The profile score rewards the learner's skill, recommended difficulty, and a
content type suited to the mastery state. `target_student_ids` is not used as
a ranking feature, preventing direct identity leakage.

### Google Colab commands

Install dependencies:

```python
!pip install -q sentence-transformers faiss-cpu
```

Build chunks, embeddings, and FAISS:

```python
!python adaptiverag/adaptive_rag.py build \
  --model BAAI/bge-small-en-v1.5
```

Run Traditional RAG (`beta=0`):

```python
!python adaptiverag/adaptive_rag.py retrieve \
  --query "A cylinder has diameter 10 and height 4. Which radius belongs in V = pi r^2 h?" \
  --student-id 78557 \
  --top-k 5 \
  --alpha 1.0 --beta 0.0
```

Run Adaptive RAG:

```python
!python adaptiverag/adaptive_rag.py retrieve \
  --query "A cylinder has diameter 10 and height 4. Which radius belongs in V = pi r^2 h?" \
  --student-id 78557 \
  --top-k 5 \
  --alpha 0.5 --beta 0.5
```

Run the alpha/beta grid:

```python
!python adaptiverag/adaptive_rag.py experiment \
  --questions adaptiverag/questions.example.jsonl \
  --alphas 1.0,0.75,0.5,0.25 \
  --betas 0.0,0.25,0.5,0.75,1.0 \
  --evaluation-mode known-skill
```

The experiment has two non-leaking evaluation tracks:

- `known-skill` (default): the LMS has already routed the query to its
  `skill_id`; retrieval must choose the appropriate difficulty and content
  type. This is the primary Traditional-vs-Adaptive comparison.
- `end-to-end`: neither `skill_id` nor difficulty is passed to retrieval.
  This additionally measures whether the retriever discovers the correct
  skill.

`expected_difficulty` and `expected_content_type` are evaluation labels only;
they are never used as metadata filters. The report includes skill
Recall@k, skill top-1 accuracy, difficulty hit@k, difficulty top-1 accuracy,
difficulty MRR, and content-type top-1 accuracy. This prevents a result from
being called successful merely because a correct difficulty appears somewhere
in the top-k list.

Outputs are written under `adaptiverag/artifacts/`:

- `chunks.jsonl`: chunk text and all retrieval metadata;
- `embeddings.npy`: normalized embedding matrix;
- `faiss.index`: FAISS inner-product index;
- `alpha_beta_results.json`: one row per question and alpha/beta pair;
- `error_analysis.json`: metrics and failed retrieval cases grouped by
  expected difficulty and alpha/beta setting.

The example question file is only a schema/example. For the final experiment,
create a larger held-out set with one JSON object per line:

```json
{
  "student_id": "78523",
  "question": "Find the area ...",
  "skill_id": "297",
  "expected_difficulty": "beginner",
  "expected_content_type": "worked_example"
}
```

Use historical attempts to build the profile and later attempts/questions for
evaluation. Do not use `target_student_ids` as the gold label for retrieval.

## Selected learners

The selection is reproducible from `skill_builder_data.csv`. The default
selection in `build_knowledge_base.py` is:

```text
78523, 78561, 78571, 78544, 78557
```

The five learners were chosen because each has at least 100 valid,
skill-tagged attempts and they share six skills with at least ten observations
per learner and skill:

| skill_id | skill_name |
|---:|---|
| 11 | Venn Diagram |
| 70 | Percent Of |
| 297 | Area Trapezoid |
| 303 | Volume Cylinder |
| 307 | Volume Rectangular Prism |
| 317 | Greatest Common Factor |

Using common skills is important for the later experiment: Traditional RAG
and Adaptive RAG can retrieve from the same knowledge base while only the
learner profile changes.

## Generated artifacts

- `learner_profiles.json`: one profile per selected learner. Each skill entry
  contains attempts, correct answers, accuracy, average attempt count, average
  response time, mastery level, and recommended material difficulty.
- `materials/*.md`: synthetic learning materials. Each of the six skills has
  three difficulty levels and five content types (`concept`,
  `worked_example`, `misconception`, `strategy`, and `transfer_practice`),
  producing 90 retrievable documents.
  Each document has metadata for `doc_id`, `skill_id`, `skill_name`,
  `difficulty`, `content_type`, source dataset, and target learners.
- `corpus_manifest.json`: machine-readable document index for ingestion,
  embedding, FAISS, and later citation.
- `selection_report.json`: selection rule, common skills, and data-quality
  counters.

The materials are deterministic synthetic scaffolds, not official ASSISTments
content. They are marked `needs_subject_matter_review` and should be checked
by a mathematics instructor before being presented to students. Documents for
all difficulty levels are generated even when no selected learner currently
has that recommendation; this gives Traditional RAG and Adaptive RAG the same
complete candidate corpus and avoids making corpus size depend on the selected
users.

## Rebuild

From the repository root:

```powershell
.venv\Scripts\python.exe adaptiverag\build_knowledge_base.py
```

The script reads the official CSV using `latin-1` encoding, because the
download contains bytes that are not valid UTF-8. It ignores rows without a
skill ID and computes profiles only from valid numeric values for the selected
learners. Re-running it replaces only generated `assistments-skill-*.md`
documents and the three JSON artifacts.

## Important experimental boundary

The profile is an aggregate of all selected learners' available attempts. It
is suitable for constructing the initial knowledge base and checking the
pipeline. For the final evaluation, split attempts chronologically:

```text
history attempts -> learner profile
later attempts   -> evaluation questions
```

This prevents future performance from leaking into the profile.
