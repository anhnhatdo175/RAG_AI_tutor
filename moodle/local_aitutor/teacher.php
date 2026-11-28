<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Teacher page: generate tasks with constraints, then approve, edit or reject them.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/lib.php');

use local_aitutor\client_exception;
use local_aitutor\form\edit_form;
use local_aitutor\form\generate_form;
use local_aitutor\store;

$courseid = required_param('courseid', PARAM_INT);
$action = optional_param('action', '', PARAM_ALPHA);
$taskid = optional_param('taskid', '', PARAM_ALPHANUMEXT);

[$course, $context, $url] = local_aitutor_setup_page($courseid, '/local/aitutor/teacher.php', [],
    'local/aitutor:manage', 'teacherpage');
$listurl = new moodle_url('/local/aitutor/teacher.php', ['courseid' => $course->id]);
$client = local_aitutor_client_or_notify();

// Approve or reject a task.
if (in_array($action, ['approve', 'reject'], true) && $taskid !== '' && $client) {
    require_sesskey();
    $task = $DB->get_record('local_aitutor_task', ['courseid' => $course->id, 'taskid' => $taskid], '*', MUST_EXIST);
    try {
        $reviewed = $client->review($task->taskid, $action);
        store::save_task($course->id, $reviewed, $USER->id);
        redirect($listurl, get_string('review_' . $action . '_done', 'local_aitutor'), null,
            \core\output\notification::NOTIFY_SUCCESS);
    } catch (client_exception $e) {
        redirect($listurl, $e->getMessage(), null, \core\output\notification::NOTIFY_ERROR);
    }
}

// Edit a task (review action "edit": it goes back to draft and must be approved again).
if ($action === 'edit' && $taskid !== '') {
    $task = $DB->get_record('local_aitutor_task', ['courseid' => $course->id, 'taskid' => $taskid], '*', MUST_EXIST);
    $editform = new edit_form(new moodle_url('/local/aitutor/teacher.php'));
    $editform->set_data([
        'courseid' => $course->id,
        'taskid' => $task->taskid,
        'action' => 'edit',
        'prompt' => $task->prompt,
        'hints' => implode("\n", store::decode($task->hints)),
        'reason' => $task->reason,
    ]);
    if ($editform->is_cancelled()) {
        redirect($listurl);
    }
    if (($data = $editform->get_data()) && $client) {
        $edit = [
            'prompt' => trim($data->prompt),
            'hints' => edit_form::hint_lines((string)$data->hints),
            'personalization_reason' => trim((string)$data->reason),
        ];
        if ((string)$data->answerkey !== '') {
            $edit['answer_key'] = (string)$data->answerkey;
        }
        try {
            $reviewed = $client->review($task->taskid, 'edit', $edit);
            store::save_task($course->id, $reviewed, $USER->id);
            redirect($listurl, get_string('review_edit_done', 'local_aitutor'), null,
                \core\output\notification::NOTIFY_SUCCESS);
        } catch (client_exception $e) {
            \core\notification::error($e->getMessage());
        }
    }
    echo $OUTPUT->header();
    echo $OUTPUT->heading(get_string('edittask', 'local_aitutor'));
    $editform->display();
    echo $OUTPUT->footer();
    exit;
}

// Generate tasks.
$generateform = new generate_form(new moodle_url('/local/aitutor/teacher.php'));
$generateform->set_data(['courseid' => $course->id]);
if (($data = $generateform->get_data()) && $client) {
    core_php_time_limit::raise(900);
    try {
        $tasks = $client->generate((int)$USER->id, (int)$course->id, trim((string)$data->skilltag),
            store::difficulty_for_level((int)$data->level), (int)$data->count, trim((string)$data->grade));
        $review = 0;
        foreach ($tasks as $task) {
            store::save_task($course->id, $task, $USER->id);
            $review += ($task['status'] ?? '') === 'needs_review' ? 1 : 0;
        }
        $message = get_string('generated', 'local_aitutor', (object)['count' => count($tasks), 'review' => $review]);
        redirect($listurl, $message, null, $review ? \core\output\notification::NOTIFY_WARNING
            : \core\output\notification::NOTIFY_SUCCESS);
    } catch (client_exception $e) {
        \core\notification::error($e->getMessage());
    }
}

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('teacherpage', 'local_aitutor'));
$generateform->display();

// Task list.
echo $OUTPUT->heading(get_string('tasklist', 'local_aitutor'), 3);
$tasks = $DB->get_records('local_aitutor_task', ['courseid' => $course->id], 'timecreated DESC, id DESC');
if (!$tasks) {
    echo $OUTPUT->notification(get_string('notasks', 'local_aitutor'), 'info');
} else {
    $table = new html_table();
    $table->head = [
        get_string('prompt', 'local_aitutor'),
        get_string('skilltag', 'local_aitutor'),
        get_string('level', 'local_aitutor'),
        get_string('status'),
        get_string('attempts', 'local_aitutor'),
        get_string('actions'),
    ];
    foreach ($tasks as $task) {
        $actions = [];
        if (in_array($task->status, ['draft', 'needs_review'], true)) {
            foreach (['approve' => 'i/valid', 'edit' => 't/edit', 'reject' => 'i/invalid'] as $name => $icon) {
                $params = ['courseid' => $course->id, 'taskid' => $task->taskid, 'action' => $name];
                if ($name !== 'edit') {
                    $params['sesskey'] = sesskey();
                }
                $actions[] = $OUTPUT->action_icon(new moodle_url('/local/aitutor/teacher.php', $params),
                    new pix_icon($icon, get_string('review_' . $name, 'local_aitutor')));
            }
        }
        $attempts = $DB->count_records('local_aitutor_attempt', ['taskid' => $task->taskid]);
        $prompt = shorten_text(format_string($task->prompt), 140);
        $cell = html_writer::div($prompt);
        if ($task->status === 'needs_review') {
            $cell .= html_writer::div(get_string('needsreview_help', 'local_aitutor'), 'small text-muted');
        }
        $table->data[] = [
            $cell,
            s($task->skilltag),
            store::level_for_difficulty((float)$task->difficulty),
            store::status_badge($task->status),
            $attempts,
            implode(' ', $actions),
        ];
    }
    echo html_writer::table($table);
}

echo $OUTPUT->footer();
