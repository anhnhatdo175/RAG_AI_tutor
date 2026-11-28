<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor;

defined('MOODLE_INTERNAL') || die();

global $CFG;
require_once($CFG->libdir . '/filelib.php');

/**
 * REST client for the tutoring service (lmsrag/service, P3-05).
 *
 * Moodle ids never leave Moodle as they are: users, teachers and courses are sent as
 * HMAC pseudonyms keyed with the idsalt setting.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class client {
    /** @var string */
    protected $baseurl;

    /** @var string */
    protected $token;

    /** @var int seconds */
    protected $timeout;

    /**
     * Constructor; reads the plugin settings unless values are given.
     *
     * @param string|null $baseurl
     * @param string|null $token
     * @param int|null $timeout
     * @throws client_exception when the URL or token is not set
     */
    public function __construct(?string $baseurl = null, ?string $token = null, ?int $timeout = null) {
        $config = get_config('local_aitutor');
        $this->baseurl = rtrim((string)($baseurl ?? ($config->serviceurl ?? '')), '/');
        $this->token = (string)($token ?? ($config->servicetoken ?? ''));
        $this->timeout = max(5, (int)($timeout ?? ($config->timeout ?? 300)));
        if ($this->baseurl === '' || $this->token === '') {
            throw new client_exception('notconfigured');
        }
    }

    /**
     * Pseudonym sent to the service instead of a Moodle id.
     *
     * @param string $kind user, course
     * @param int $id Moodle id
     * @return string
     * @throws client_exception when the salt is not set
     */
    public static function pseudonym(string $kind, int $id): string {
        $salt = (string)get_config('local_aitutor', 'idsalt');
        if ($salt === '') {
            throw new client_exception('nosalt');
        }
        return 'moodle-' . substr(hash_hmac('sha256', $kind . ':' . $id, $salt), 0, 32);
    }

    /**
     * GET /health (no token).
     *
     * @return array
     */
    public function health(): array {
        return $this->request('GET', '/health', null, false);
    }

    /**
     * POST /generate.
     *
     * @param int $teacherid Moodle user id of the teacher
     * @param int $courseid Moodle course id
     * @param string $skilltag
     * @param float $difficulty 0..1
     * @param int $count 1..10
     * @param string|null $grade
     * @return array list of public tasks (no answer key)
     */
    public function generate(int $teacherid, int $courseid, string $skilltag, float $difficulty,
            int $count, ?string $grade = null): array {
        $body = [
            'teacher_id' => self::pseudonym('user', $teacherid),
            'course_id' => self::pseudonym('course', $courseid),
            'skill_tag' => $skilltag,
            'difficulty' => $difficulty,
            'count' => $count,
            'mode' => 'practice',
        ];
        if ($grade !== null && $grade !== '') {
            $body['grade'] = $grade;
        }
        $response = $this->request('POST', '/generate', $body);
        return $response['tasks'] ?? [];
    }

    /**
     * POST /tasks/{id}/review.
     *
     * @param string $taskid
     * @param string $action approve, edit or reject
     * @param array|null $edit prompt, hints, personalization_reason, answer_key
     * @return array the reviewed public task
     */
    public function review(string $taskid, string $action, ?array $edit = null): array {
        $body = ['action' => $action];
        if ($edit !== null) {
            $body['edit'] = $edit;
        }
        $response = $this->request('POST', '/tasks/' . rawurlencode($taskid) . '/review', $body);
        return $response['task'] ?? [];
    }

    /**
     * POST /submit; graded on the service, which owns the answer key.
     *
     * @param int $userid Moodle user id
     * @param string $taskid
     * @param string $answer
     * @param string $idempotencykey same key returns the stored result
     * @param int $rtms response time in ms
     * @return array status, correct, score, feedback, mastery_before, mastery_after
     */
    public function submit(int $userid, string $taskid, string $answer, string $idempotencykey, int $rtms = 0): array {
        return $this->request('POST', '/submit', [
            'user_id' => self::pseudonym('user', $userid),
            'task_id' => $taskid,
            'answer' => $answer,
            'idempotency_key' => $idempotencykey,
            'rt_ms' => max(0, $rtms),
        ]);
    }

    /**
     * GET /profile/{user_id}.
     *
     * @param int $userid Moodle user id
     * @return array
     */
    public function profile(int $userid): array {
        return $this->request('GET', '/profile/' . rawurlencode(self::pseudonym('user', $userid)));
    }

    /**
     * POST /sync with already pseudonymised events.
     *
     * @param array $events list of user_id, task_id, answer, idempotency_key, rt_ms
     * @return array accepted, rejected, results
     */
    public function sync(array $events): array {
        return $this->request('POST', '/sync', ['events' => array_values($events)]);
    }

    /**
     * Send one request and decode the JSON answer.
     *
     * @param string $method GET or POST
     * @param string $path
     * @param array|null $body
     * @param bool $auth send the Bearer token
     * @return array
     * @throws client_exception
     */
    protected function request(string $method, string $path, ?array $body = null, bool $auth = true): array {
        // The URL is set by a site administrator only; the service usually runs on the
        // Docker host (host.docker.internal, a private address on port 8800), which
        // Moodle's default cURL blocklist would refuse.
        $curl = new \curl(['ignoresecurity' => true]);
        $headers = ['Accept: application/json'];
        if ($auth) {
            $headers[] = 'Authorization: Bearer ' . $this->token;
        }
        $options = [
            'CURLOPT_TIMEOUT' => $this->timeout,
            'CURLOPT_CONNECTTIMEOUT' => 10,
        ];
        $url = $this->baseurl . $path;

        if ($method === 'GET') {
            $curl->setHeader($headers);
            $raw = $curl->get($url, [], $options);
        } else {
            $headers[] = 'Content-Type: application/json';
            $curl->setHeader($headers);
            $raw = $curl->post($url, json_encode($body, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES), $options);
        }

        if ($curl->get_errno()) {
            throw new client_exception('unreachable', (string)$curl->error);
        }
        $info = $curl->get_info();
        $status = (int)($info['http_code'] ?? 0);
        $data = json_decode((string)$raw, true);

        if ($status < 200 || $status >= 300) {
            $detail = (string)$raw;
            if (is_array($data) && array_key_exists('detail', $data)) {
                $detail = is_string($data['detail']) ? $data['detail'] : json_encode($data['detail']);
            }
            throw new client_exception('httperror', $status . ': ' . \core_text::substr($detail, 0, 500), $status);
        }
        if (!is_array($data)) {
            throw new client_exception('badresponse', \core_text::substr((string)$raw, 0, 200));
        }
        return $data;
    }
}
