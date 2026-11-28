<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor\form;

defined('MOODLE_INTERNAL') || die();

global $CFG;
require_once($CFG->libdir . '/formslib.php');

/**
 * Learner answer for one task. The attempt key is created when the form is shown,
 * so a resubmitted form reuses the same idempotency key.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class answer_form extends \moodleform {
    /**
     * Form definition.
     */
    protected function definition() {
        $mform = $this->_form;

        foreach (['courseid' => PARAM_INT, 'taskid' => PARAM_ALPHANUMEXT,
                'attemptkey' => PARAM_ALPHANUM, 'starttime' => PARAM_INT] as $name => $type) {
            $mform->addElement('hidden', $name);
            $mform->setType($name, $type);
        }

        $mform->addElement('text', 'answer', get_string('youranswer', 'local_aitutor'), ['size' => 40]);
        $mform->setType('answer', PARAM_RAW_TRIMMED);
        $mform->addRule('answer', null, 'required', null, 'client');
        $mform->addRule('answer', get_string('maximumchars', '', 2000), 'maxlength', 2000, 'client');

        $this->add_action_buttons(false, get_string('submitanswer', 'local_aitutor'));
    }

    /**
     * Validation.
     *
     * @param array $data
     * @param array $files
     * @return array
     */
    public function validation($data, $files) {
        $errors = parent::validation($data, $files);
        if (trim((string)$data['answer']) === '') {
            $errors['answer'] = get_string('required');
        }
        if (strlen((string)$data['attemptkey']) < 16) {
            $errors['answer'] = get_string('formexpired', 'local_aitutor');
        }
        return $errors;
    }
}
