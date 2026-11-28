<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor\form;

defined('MOODLE_INTERNAL') || die();

global $CFG;
require_once($CFG->libdir . '/formslib.php');

/**
 * Teacher edit of a generated task before approval (review action "edit").
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class edit_form extends \moodleform {
    /**
     * Form definition.
     */
    protected function definition() {
        $mform = $this->_form;

        foreach (['courseid' => PARAM_INT, 'taskid' => PARAM_ALPHANUMEXT, 'action' => PARAM_ALPHA] as $name => $type) {
            $mform->addElement('hidden', $name);
            $mform->setType($name, $type);
        }

        $mform->addElement('header', 'editheader', get_string('edittask', 'local_aitutor'));

        $mform->addElement('textarea', 'prompt', get_string('prompt', 'local_aitutor'), ['rows' => 6, 'cols' => 80]);
        $mform->setType('prompt', PARAM_TEXT);
        $mform->addRule('prompt', null, 'required', null, 'client');

        $mform->addElement('textarea', 'hints', get_string('hints', 'local_aitutor'), ['rows' => 4, 'cols' => 80]);
        $mform->setType('hints', PARAM_TEXT);
        $mform->addHelpButton('hints', 'hints', 'local_aitutor');

        $mform->addElement('textarea', 'reason', get_string('reason', 'local_aitutor'), ['rows' => 2, 'cols' => 80]);
        $mform->setType('reason', PARAM_TEXT);

        $mform->addElement('text', 'answerkey', get_string('answerkey', 'local_aitutor'), ['size' => 40]);
        $mform->setType('answerkey', PARAM_RAW_TRIMMED);
        $mform->addHelpButton('answerkey', 'answerkey', 'local_aitutor');

        $this->add_action_buttons(true, get_string('savechanges'));
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
        if (trim((string)$data['prompt']) === '') {
            $errors['prompt'] = get_string('required');
        }
        if (count(self::hint_lines((string)$data['hints'])) > 10) {
            $errors['hints'] = get_string('toomanyhints', 'local_aitutor');
        }
        return $errors;
    }

    /**
     * One hint per non-empty line.
     *
     * @param string $text
     * @return array
     */
    public static function hint_lines(string $text): array {
        return array_values(array_filter(array_map('trim', preg_split('/\R/', $text)), 'strlen'));
    }
}
