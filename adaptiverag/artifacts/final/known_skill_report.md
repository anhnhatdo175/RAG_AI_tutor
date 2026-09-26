# Temporal Vector-Profile RAG Report

## Experimental setup

- Dataset interactions: 10127
- Temporal split: history=7086, validation=1519, test=1522
- Learner profiles: 5; corpus chunks: 90
- Selected configuration: alpha=0.25, beta=0.75

## Main results

| System | Skill Recall@5 | Skill MRR | Difficulty Hit@5 | Difficulty MRR |
|---|---:|---:|---:|---:|
| Traditional RAG | 0.9297 | 0.9297 | 0.5204 | 0.1590 |
| Adaptive Profile-Based RAG | 0.9297 | 0.9297 | 0.7260 | 0.3471 |

## Figures

- `known_skill_comparison.png`: overall metric comparison.
- `known_skill_per_student.png`: difficulty MRR by learner.

## Limitations

- ASSISTments export does not contain complete problem statements; queries use skill name and problem ID.
- The instructional corpus is synthetic and requires subject-matter review.
- Results measure retrieval ranking, not final generated-answer quality.
