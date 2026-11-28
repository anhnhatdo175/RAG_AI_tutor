<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor;

/**
 * Moodle-side records: tasks known to a course and learner attempts.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class store {
    /** Difficulty sent to the service for each teacher level (the service matches within 0.10). */
    const LEVELS = [1 => 0.1, 2 => 0.3, 3 => 0.5, 4 => 0.7, 5 => 0.9];

    /**
     * Service difficulty for a level 1..5.
     *
     * @param int $level
     * @return float
     */
    public static function difficulty_for_level(int $level): float {
        return self::LEVELS[min(5, max(1, $level))];
    }

    /**
     * Nearest level 1..5 for a service difficulty.
     *
     * @param float $difficulty
     * @return int
     */
    public static function level_for_difficulty(float $difficulty): int {
        return (int)min(5, max(1, (int)floor($difficulty * 5) + 1));
    }

    /**
     * Insert or update a task returned by /generate or /review.
     *
     * @param int $courseid
     * @param array $task public task from the service
     * @param int $userid teacher who created it
     * @return \stdClass the stored record
     */
    public static function save_task(int $courseid, array $task, int $userid): \stdClass {
        global $DB;
        $now = time();
        $record = $DB->get_record('local_aitutor_task', ['taskid' => $task['task_id']]);
        $fields = [
            'itemid' => (string)($task['item_id'] ?? ''),
            'skilltag' => (string)($task['skill_tag'] ?? ''),
            'difficulty' => (float)($task['difficulty'] ?? 0),
            'prompt' => (string)($task['prompt'] ?? ''),
            'hints' => json_encode(array_values($task['hints'] ?? []), JSON_UNESCAPED_UNICODE),
            'reason' => (string)($task['personalization_reason'] ?? ''),
            'docids' => json_encode(array_values($task['supporting_doc_ids'] ?? []), JSON_UNESCAPED_UNICODE),
            'status' => (string)($task['status'] ?? 'draft'),
            'timemodified' => $now,
        ];
        if ($record) {
            foreach ($fields as $name => $value) {
                $record->$name = $value;
            }
            $DB->update_record('local_aitutor_task', $record);
            return $record;
        }
        $record = (object)($fields + [
            'courseid' => $courseid,
            'taskid' => (string)$task['task_id'],
            'createdby' => $userid,
            'timecreated' => $now,
        ]);
        $record->id = $DB->insert_record('local_aitutor_task', $record);
        return $record;
    }

    /**
     * Store a learner answer before it is sent, so nothing is lost if the service is down.
     * The same idempotency key always maps to the same row.
     *
     * @param int $courseid
     * @param int $userid
     * @param string $taskid
     * @param string $key idempotency key
     * @param string $answer
     * @param int $rtms
     * @param string $source practice or quiz
     * @return \stdClass
     */
    public static function record_attempt(int $courseid, int $userid, string $taskid, string $key,
            string $answer, int $rtms, string $source = 'practice'): \stdClass {
        global $DB;
        $existing = $DB->get_record('local_aitutor_attempt', ['idempotencykey' => $key]);
        if ($existing) {
            return $existing;
        }
        $now = time();
        $record = (object)[
            'courseid' => $courseid,
            'userid' => $userid,
            'taskid' => $taskid,
            'idempotencykey' => $key,
            'source' => $source,
            'answer' => $answer,
            'rtms' => max(0, $rtms),
            'status' => 'pending',
            'timecreated' => $now,
            'timemodified' => $now,
        ];
        $record->id = $DB->insert_record('local_aitutor_attempt', $record);
        return $record;
    }

    /**
     * Copy a /submit or /sync result onto the attempt row.
     *
     * @param \stdClass $attempt
     * @param array $result
     * @return \stdClass
     */
    public static function apply_result(\stdClass $attempt, array $result): \stdClass {
        global $DB;
        $attempt->status = (string)($result['status'] ?? 'graded');
        $attempt->correct = isset($result['correct']) ? (int)(bool)$result['correct'] : null;
        $attempt->score = isset($result['score']) ? (float)$result['score'] : null;
        $attempt->feedback = (string)($result['feedback'] ?? '');
        $attempt->masterybefore = isset($result['mastery_before']) ? json_encode($result['mastery_before']) : null;
        $attempt->masteryafter = isset($result['mastery_after']) ? json_encode($result['mastery_after']) : null;
        $attempt->error = isset($result['error']) ? (string)$result['error'] : null;
        $attempt->timemodified = time();
        $DB->update_record('local_aitutor_attempt', $attempt);
        return $attempt;
    }

    /**
     * Mark an attempt as refused by the service (it will not be retried).
     *
     * @param \stdClass $attempt
     * @param string $error
     * @return \stdClass
     */
    public static function reject_attempt(\stdClass $attempt, string $error): \stdClass {
        global $DB;
        $attempt->status = 'rejected';
        $attempt->error = $error;
        $attempt->timemodified = time();
        $DB->update_record('local_aitutor_attempt', $attempt);
        return $attempt;
    }

    /**
     * Decode a JSON list or map stored in a text column.
     *
     * @param string|null $json
     * @return array
     */
    public static function decode(?string $json): array {
        if ($json === null || $json === '') {
            return [];
        }
        $value = json_decode($json, true);
        return is_array($value) ? $value : [];
    }

    /**
     * Bootstrap badge for a task or attempt status.
     *
     * @param string $status
     * @return string HTML
     */
    public static function status_badge(string $status): string {
        $classes = [
            'approved' => 'success', 'graded' => 'success',
            'draft' => 'secondary', 'pending' => 'secondary', 'processing' => 'info',
            'needs_review' => 'warning', 'rejected' => 'danger',
        ];
        $colour = $classes[$status] ?? 'secondary';
        $label = get_string_manager()->string_exists('status_' . $status, 'local_aitutor')
            ? get_string('status_' . $status, 'local_aitutor') : s($status);
        return \html_writer::span($label, "badge bg-{$colour} badge-{$colour}");
    }

    /**
     * Table rows comparing mastery before and after an answer.
     *
     * @param \stdClass $attempt
     * @return \html_table|null
     */
    public static function mastery_table(\stdClass $attempt): ?\html_table {
        $before = self::decode($attempt->masterybefore ?? null);
        $after = self::decode($attempt->masteryafter ?? null);
        if (!$after) {
            return null;
        }
        $table = new \html_table();
        $table->head = [
            get_string('skilltag', 'local_aitutor'),
            get_string('masterybefore', 'local_aitutor'),
            get_string('masteryafter', 'local_aitutor'),
        ];
        foreach ($after as $skill => $value) {
            $old = $before[$skill] ?? null;
            $delta = $old === null ? '' : sprintf(' (%+.3f)', $value - $old);
            $table->data[] = [
                s($skill),
                $old === null ? '-' : format_float($old, 3),
                format_float($value, 3) . $delta,
            ];
        }
        return $table;
    }
}
