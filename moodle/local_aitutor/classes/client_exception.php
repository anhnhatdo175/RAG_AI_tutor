<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor;

/**
 * Error raised when the tutoring service cannot be reached or refuses a request.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class client_exception extends \moodle_exception {
    /** @var string Short error code: notconfigured, nosalt, unreachable, httperror, badresponse. */
    public $reason;

    /** @var int HTTP status (0 when there was no HTTP answer). */
    public $httpstatus;

    /**
     * Constructor.
     *
     * @param string $reason short error code
     * @param string $detail text shown to the user ({$a} in the language string)
     * @param int $httpstatus HTTP status of the service answer
     */
    public function __construct(string $reason, string $detail = '', int $httpstatus = 0) {
        $this->reason = $reason;
        $this->httpstatus = $httpstatus;
        parent::__construct('error_' . $reason, 'local_aitutor', '', $detail, $detail);
    }

    /**
     * Whether the request may succeed later without any change (network or server trouble).
     *
     * @return bool
     */
    public function is_retryable(): bool {
        return $this->reason === 'unreachable' || $this->httpstatus >= 500;
    }
}
