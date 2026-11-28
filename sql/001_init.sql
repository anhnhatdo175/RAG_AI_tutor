CREATE TABLE IF NOT EXISTS learners (
    learner_hash VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    item_id VARCHAR(200) NOT NULL,
    skill_tag VARCHAR(200) NOT NULL,
    difficulty DOUBLE PRECISION NOT NULL,
    prompt TEXT NOT NULL,
    response_type VARCHAR(40) NOT NULL,
    answer_key TEXT,
    hints JSON NOT NULL,
    generation JSON NOT NULL,
    mode VARCHAR(20) NOT NULL,
    status VARCHAR(24) NOT NULL,
    created_by_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS task_citations (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    chunk_id VARCHAR(300) NOT NULL,
    source VARCHAR(80) NOT NULL,
    doc_id VARCHAR(300) NOT NULL,
    CONSTRAINT uq_task_citations_task_chunk UNIQUE (task_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS assignments (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    teacher_hash VARCHAR(64) NOT NULL,
    course_hash VARCHAR(64) NOT NULL,
    status VARCHAR(24) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_assignments_task_course UNIQUE (task_id, course_hash)
);

CREATE TABLE IF NOT EXISTS submissions (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL REFERENCES tasks(task_id),
    learner_hash VARCHAR(64) NOT NULL REFERENCES learners(learner_hash),
    idempotency_key VARCHAR(200) NOT NULL,
    request_hash VARCHAR(64) NOT NULL,
    answer JSON NOT NULL,
    status VARCHAR(24) NOT NULL,
    score DOUBLE PRECISION,
    feedback TEXT,
    mastery_before JSON,
    mastery_after JSON,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_submissions_learner_idempotency UNIQUE (learner_hash, idempotency_key)
);

CREATE TABLE IF NOT EXISTS mastery_state (
    learner_hash VARCHAR(64) PRIMARY KEY REFERENCES learners(learner_hash),
    state_json JSON NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS applied_feedback_events (
    learner_hash VARCHAR(64) NOT NULL REFERENCES learners(learner_hash),
    event_id VARCHAR(200) NOT NULL,
    event_fingerprint VARCHAR(64) NOT NULL,
    task_id VARCHAR(200),
    applied_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (learner_hash, event_id),
    CONSTRAINT uq_feedback_learner_task UNIQUE (learner_hash, task_id)
);

CREATE TABLE IF NOT EXISTS tutor_turns (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(200) NOT NULL,
    learner_hash VARCHAR(64) NOT NULL REFERENCES learners(learner_hash),
    task_id VARCHAR(64) NOT NULL REFERENCES tasks(task_id),
    turn_index INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    score DOUBLE PRECISION,
    response JSON,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_log (
    decision_id VARCHAR(64) PRIMARY KEY,
    learner_hash VARCHAR(64) REFERENCES learners(learner_hash),
    endpoint VARCHAR(40) NOT NULL,
    query TEXT,
    retrieved_chunks JSON NOT NULL,
    scores JSON NOT NULL,
    reasons JSON NOT NULL,
    model VARCHAR(200),
    stage_latency_ms JSON NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(300) PRIMARY KEY,
    doc_id VARCHAR(300) NOT NULL,
    source VARCHAR(80) NOT NULL,
    skill_tag VARCHAR(200) NOT NULL,
    text TEXT NOT NULL,
    metadata JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS ix_assignments_course_status ON assignments(course_hash, status);
CREATE INDEX IF NOT EXISTS ix_submissions_task_learner ON submissions(task_id, learner_hash);
CREATE INDEX IF NOT EXISTS ix_tutor_turns_session ON tutor_turns(session_id, learner_hash, task_id);
CREATE INDEX IF NOT EXISTS ix_decision_log_created_at ON decision_log(created_at);
CREATE INDEX IF NOT EXISTS ix_chunks_doc_id ON chunks(doc_id);
