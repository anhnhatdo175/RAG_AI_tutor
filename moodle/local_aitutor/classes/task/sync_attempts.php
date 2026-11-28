<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor\task;

use local_aitutor\client;
use local_aitutor\client_exception;
use local_aitutor\store;

/**
 * Collect finished quiz answers for AI tutor tasks and send every pending attempt to /sync.
 *
 * A quiz question is linked to a service task by its question bank ID number
 * "aitutor:<task_id>". Practice-page answers that could not be sent (service down)
 * are pending too and are retried here.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class sync_attempts extends \core\task\scheduled_task {
    /** Events per /sync request (the service accepts at most 500). */
    const BATCH = 100;

    /** Prefix of the question ID number that links a quiz question to a task. */
    const IDNUMBER_PREFIX = 'aitutor:';

    /**
     * Task name shown in the admin pages.
     *
     * @return string
     */
    public function get_name() {
        return get_string('task_sync', 'local_aitutor');
    }

    /**
     * Run the task.
     */
    public function execute() {
        try {
            $client = new client();
        } catch (client_exception $e) {
            mtrace('local_aitutor: service not configured, nothing sent.');
            return;
        }
        $collected = $this->collect_quiz_attempts();
        mtrace("local_aitutor: {$collected} new quiz answers collected.");
        $this->send_pending($client);
    }

    /**
     * Copy finished quiz answers on linked questions into the attempt table.
     *
     * @return int number of new attempts
     */
    protected function collect_quiz_attempts(): int {
        global $DB;
        $since = (int)get_config('local_aitutor', 'lastquizsync');
        $like = $DB->sql_like('qbe.idnumber', ':prefix');
        $sql = "SELECT qa.id AS qaid, quiza.userid, quiz.course, qbe.idnumber, qa.responsesummary,
                       quiza.timestart, quiza.timefinish
                  FROM {question_attempts} qa
                  JOIN {quiz_attempts} quiza ON quiza.uniqueid = qa.questionusageid
                  JOIN {quiz} quiz ON quiz.id = quiza.quiz
                  JOIN {question_versions} qv ON qv.questionid = qa.questionid
                  JOIN {question_bank_entries} qbe ON qbe.id = qv.questionbankentryid
                 WHERE quiza.state = :state
                   AND quiza.timefinish > :since
                   AND {$like}
              ORDER BY quiza.timefinish ASC";
        $params = ['state' => 'finished', 'since' => $since, 'prefix' => self::IDNUMBER_PREFIX . '%'];

        $count = 0;
        $latest = $since;
        $rows = $DB->get_recordset_sql($sql, $params);
        foreach ($rows as $row) {
            $latest = max($latest, (int)$row->timefinish);
            $taskid = substr($row->idnumber, strlen(self::IDNUMBER_PREFIX));
            if ($taskid === '' || $row->responsesummary === null) {
                continue;
            }
            $key = 'quiz-' . $row->qaid;
            if ($DB->record_exists('local_aitutor_attempt', ['idempotencykey' => $key])) {
                continue;
            }
            $rtms = max(0, ((int)$row->timefinish - (int)$row->timestart) * 1000);
            store::record_attempt((int)$row->course, (int)$row->userid, $taskid, $key,
                (string)$row->responsesummary, $rtms, 'quiz');
            $count++;
        }
        $rows->close();
        set_config('lastquizsync', $latest, 'local_aitutor');
        return $count;
    }

    /**
     * Send pending attempts in batches; stop at the first network or server error.
     *
     * @param client $client
     */
    protected function send_pending(client $client): void {
        global $DB;
        $sent = 0;
        $lastid = 0;
        while (true) {
            $attempts = $DB->get_records_select('local_aitutor_attempt', 'status = :status AND id > :lastid',
                ['status' => 'pending', 'lastid' => $lastid], 'id ASC', '*', 0, self::BATCH);
            if (!$attempts) {
                break;
            }
            $events = [];
            $bykey = [];
            foreach ($attempts as $attempt) {
                $lastid = max($lastid, (int)$attempt->id);
                $events[] = [
                    'user_id' => client::pseudonym('user', (int)$attempt->userid),
                    'task_id' => $attempt->taskid,
                    'answer' => (string)$attempt->answer,
                    'idempotency_key' => $attempt->idempotencykey,
                    'rt_ms' => (int)$attempt->rtms,
                ];
                $bykey[$attempt->idempotencykey] = $attempt;
            }
            try {
                $response = $client->sync($events);
            } catch (client_exception $e) {
                mtrace('local_aitutor: /sync failed, will retry next run: ' . $e->getMessage());
                return;
            }
            foreach ($response['results'] ?? [] as $result) {
                $key = $result['idempotency_key'] ?? null;
                if ($key !== null && isset($bykey[$key])) {
                    store::apply_result($bykey[$key], $result);
                    $sent++;
                }
            }
        }
        mtrace("local_aitutor: {$sent} attempts sent to /sync.");
    }
}
