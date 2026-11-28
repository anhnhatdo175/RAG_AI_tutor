<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor\form;

defined('MOODLE_INTERNAL') || die();

global $CFG;
require_once($CFG->libdir . '/formslib.php');

/**
 * Teacher constraints for /generate: skill, level, number of tasks.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class generate_form extends \moodleform {
    /**
     * Form definition.
     */
    protected function definition() {
        $mform = $this->_form;

        $mform->addElement('hidden', 'courseid');
        $mform->setType('courseid', PARAM_INT);

        $mform->addElement('header', 'generateheader', get_string('generateheader', 'local_aitutor'));

        $skills = preg_split('/\R/', (string)get_config('local_aitutor', 'skills'));
        $skills = array_values(array_filter(array_map('trim', $skills), 'strlen'));
        $options = $skills ? array_combine($skills, $skills) : [];
        $mform->addElement('autocomplete', 'skilltag', get_string('skilltag', 'local_aitutor'), $options, [
            'tags' => true,
            'multiple' => false,
            'noselectionstring' => get_string('skilltag_choose', 'local_aitutor'),
        ]);
        $mform->setType('skilltag', PARAM_TEXT);
        $mform->addRule('skilltag', null, 'required', null, 'client');
        $mform->addHelpButton('skilltag', 'skilltag', 'local_aitutor');

        $levels = [];
        foreach (array_keys(\local_aitutor\store::LEVELS) as $i) {
            $levels[$i] = get_string('level' . $i, 'local_aitutor');
        }
        $mform->addElement('select', 'level', get_string('level', 'local_aitutor'), $levels);
        $mform->setDefault('level', 2);

        $counts = array_combine(range(1, 10), range(1, 10));
        $mform->addElement('select', 'count', get_string('count', 'local_aitutor'), $counts);
        $mform->setDefault('count', 3);

        $mform->addElement('text', 'grade', get_string('grade', 'local_aitutor'), ['size' => 10]);
        $mform->setType('grade', PARAM_TEXT);
        $mform->addHelpButton('grade', 'grade', 'local_aitutor');

        $this->add_action_buttons(false, get_string('generate', 'local_aitutor'));
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
        $skill = trim((string)($data['skilltag'] ?? ''));
        if ($skill === '' || \core_text::strlen($skill) > 200) {
            $errors['skilltag'] = get_string('required');
        }
        return $errors;
    }
}
