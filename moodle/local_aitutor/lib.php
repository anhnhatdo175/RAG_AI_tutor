<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Navigation hooks for local_aitutor.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Add the AI tutor pages to the course navigation ("More" menu).
 *
 * @param navigation_node $navigation
 * @param stdClass $course
 * @param context $context
 */
function local_aitutor_extend_navigation_course(navigation_node $navigation, stdClass $course, context $context) {
    if (has_capability('local/aitutor:manage', $context)) {
        $navigation->add(
            get_string('teacherpage', 'local_aitutor'),
            new moodle_url('/local/aitutor/teacher.php', ['courseid' => $course->id]),
            navigation_node::TYPE_CUSTOM,
            null,
            'local_aitutor_teacher',
            new pix_icon('i/settings', '')
        );
    }
    if (has_capability('local/aitutor:practice', $context)) {
        $navigation->add(
            get_string('practicepage', 'local_aitutor'),
            new moodle_url('/local/aitutor/practice.php', ['courseid' => $course->id]),
            navigation_node::TYPE_CUSTOM,
            null,
            'local_aitutor_practice',
            new pix_icon('i/edit', '')
        );
    }
    if (has_any_capability(['local/aitutor:practice', 'local/aitutor:manage'], $context)) {
        $navigation->add(
            get_string('profilepage', 'local_aitutor'),
            new moodle_url('/local/aitutor/profile.php', ['courseid' => $course->id]),
            navigation_node::TYPE_CUSTOM,
            null,
            'local_aitutor_profile',
            new pix_icon('i/report', '')
        );
    }
}

/**
 * Set up a course page of the plugin and check the capability.
 *
 * @param int $courseid
 * @param string $script e.g. /local/aitutor/teacher.php
 * @param array $params extra URL parameters
 * @param string $capability
 * @param string $titlestring language string of the page title
 * @return array [stdClass $course, context_course $context, moodle_url $url]
 */
function local_aitutor_setup_page(int $courseid, string $script, array $params, string $capability,
        string $titlestring): array {
    global $PAGE;
    $course = get_course($courseid);
    require_login($course);
    $context = context_course::instance($course->id);
    require_capability($capability, $context);

    $url = new moodle_url($script, ['courseid' => $course->id] + $params);
    $PAGE->set_url($url);
    $PAGE->set_context($context);
    $PAGE->set_pagelayout('incourse');
    $title = get_string($titlestring, 'local_aitutor');
    $PAGE->set_title($title . ': ' . format_string($course->shortname));
    $PAGE->set_heading(format_string($course->fullname));
    $PAGE->navbar->add($title, new moodle_url($script, ['courseid' => $course->id]));
    return [$course, $context, $url];
}

/**
 * Service client, or null with an error notification when the plugin is not configured.
 *
 * @return \local_aitutor\client|null
 */
function local_aitutor_client_or_notify(): ?\local_aitutor\client {
    try {
        $client = new \local_aitutor\client();
        \local_aitutor\client::pseudonym('user', 0);
        return $client;
    } catch (\local_aitutor\client_exception $e) {
        \core\notification::error($e->getMessage());
        return null;
    }
}
