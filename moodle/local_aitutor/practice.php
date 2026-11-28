<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Learner page: list approved tasks, answer one, see the server-side grade and new mastery.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/lib.php');

use local_aitutor\client_exception;
use local_aitutor\form\answer_form;
use local_aitutor\store;

$courseid = required_param('courseid', PARAM_INT);
$taskid = optional_param('taskid', '', PARAM_ALPHANUMEXT);
$attemptid = optional_param('attempt', 0, PARAM_INT);

$params = $taskid !== '' ? ['taskid' => $taskid] : [];
[$course, $context, $url] = local_aitutor_setup_page($courseid, '/local/aitutor/practice.php', $params,
    'local/aitutor:practice', 'practicepage');

// Task list.
if ($taskid === '') {
    echo $OUTPUT->header();
    echo $OUTPUT->heading(get_string('practicepage', 'local_aitutor'));
    $tasks = $DB->get_records('local_aitutor_task', ['courseid' => $course->id, 'status' => 'approved'],
        'timecreated ASC, id ASC');
    if (!$tasks) {
        echo $OUTPUT->notification(get_string('nopractice', 'local_aitutor'), 'info');
        echo $OUTPUT->footer();
        exit;
    }
    $table = new html_table();
    $table->head = [
        get_string('prompt', 'local_aitutor'),
        get_string('skilltag', 'local_aitutor'),
        get_string('level', 'local_aitutor'),
        get_string('lastresult', 'local_aitutor'),
        '',
    ];
    foreach ($tasks as $task) {
        $last = $DB->get_records('local_aitutor_attempt', ['userid' => $USER->id, 'taskid' => $task->taskid],
            'timecreated DESC, id DESC', '*', 0, 1);
        $last = $last ? reset($last) : null;
        $result = '-';
        if ($last) {
            $result = $last->status === 'graded'
                ? ($last->correct ? get_string('correct', 'local_aitutor') : get_string('incorrect', 'local_aitutor'))
                : store::status_badge($last->status);
        }
        $link = html_writer::link(
            new moodle_url('/local/aitutor/practice.php', ['courseid' => $course->id, 'taskid' => $task->taskid]),
            $last ? get_string('tryagain', 'local_aitutor') : get_string('start', 'local_aitutor'),
            ['class' => 'btn btn-primary btn-sm']
        );
        $table->data[] = [
            shorten_text(format_string($task->prompt), 120),
            s($task->skilltag),
            store::level_for_difficulty((float)$task->difficulty),
            $result,
            $link,
        ];
    }
    echo html_writer::table($table);
    echo $OUTPUT->footer();
    exit;
}

$task = $DB->get_record('local_aitutor_task',
    ['courseid' => $course->id, 'taskid' => $taskid, 'status' => 'approved'], '*', MUST_EXIST);
$taskurl = new moodle_url('/local/aitutor/practice.php', ['courseid' => $course->id, 'taskid' => $task->taskid]);

$form = new answer_form($taskurl);
if ($data = $form->get_data()) {
    $key = 'practice-' . $USER->id . '-' . $data->attemptkey;
    $rtms = $data->starttime > 0 ? max(0, (time() - (int)$data->starttime) * 1000) : 0;
    $attempt = store::record_attempt($course->id, (int)$USER->id, $task->taskid, $key,
        (string)$data->answer, $rtms);
    if ($attempt->status === 'pending') {
        try {
            $client = new \local_aitutor\client();
            $result = $client->submit((int)$USER->id, $task->taskid, (string)$attempt->answer, $key, (int)$attempt->rtms);
            store::apply_result($attempt, $result);
        } catch (client_exception $e) {
            if ($e->is_retryable()) {
                // Kept as pending; the scheduled task sends it to /sync later.
                \core\notification::warning(get_string('savedforlater', 'local_aitutor'));
            } else {
                store::reject_attempt($attempt, $e->getMessage());
                \core\notification::error($e->getMessage());
            }
        }
    }
    $redirect = new moodle_url($taskurl, ['attempt' => $attempt->id]);
    redirect($redirect);
}

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('practicepage', 'local_aitutor'));

// Result of the last answer (after the redirect).
if ($attemptid) {
    $attempt = $DB->get_record('local_aitutor_attempt', ['id' => $attemptid, 'userid' => $USER->id,
        'taskid' => $task->taskid]);
    if ($attempt) {
        $box = html_writer::tag('h4', get_string('result', 'local_aitutor'));
        $box .= html_writer::div(get_string('youranswer', 'local_aitutor') . ': ' . s($attempt->answer));
        if ($attempt->status === 'graded') {
            $type = $attempt->correct ? 'success' : 'danger';
            $label = $attempt->correct ? get_string('correct', 'local_aitutor') : get_string('incorrect', 'local_aitutor');
            $box .= html_writer::div($label, "alert alert-{$type} mt-2");
        } else {
            $box .= html_writer::div(store::status_badge($attempt->status), 'mt-2');
            $box .= html_writer::div(get_string('statusinfo_' . $attempt->status, 'local_aitutor'), 'small text-muted');
            if ($attempt->status === 'rejected' && (string)$attempt->error !== '') {
                $box .= html_writer::div(s($attempt->error), 'small text-danger');
            }
        }
        if ((string)$attempt->feedback !== '') {
            $box .= html_writer::div(format_text($attempt->feedback, FORMAT_MARKDOWN), 'mt-2');
        }
        if ($mastery = store::mastery_table($attempt)) {
            $box .= html_writer::tag('h5', get_string('newmastery', 'local_aitutor'), ['class' => 'mt-3']);
            $box .= html_writer::table($mastery);
        }
        echo $OUTPUT->box($box, 'generalbox p-3 mb-3');
    }
}

// The task.
$content = html_writer::div(format_text($task->prompt, FORMAT_MARKDOWN), 'aitutor-prompt');
$hints = store::decode($task->hints);
if ($hints) {
    $items = '';
    foreach ($hints as $hint) {
        $items .= html_writer::tag('li', format_text((string)$hint, FORMAT_MARKDOWN));
    }
    $content .= html_writer::tag('details',
        html_writer::tag('summary', get_string('showhints', 'local_aitutor')) . html_writer::tag('ol', $items),
        ['class' => 'mt-2']);
}
$why = '';
if ((string)$task->reason !== '') {
    $why .= html_writer::div(get_string('reason', 'local_aitutor') . ': ' . s($task->reason));
}
$docids = store::decode($task->docids);
if ($docids) {
    $why .= html_writer::div(get_string('sources', 'local_aitutor') . ': ' .
        implode(', ', array_map(fn($id) => html_writer::tag('code', s((string)$id)), $docids)));
}
if ($why !== '') {
    $content .= html_writer::div($why, 'small text-muted mt-2');
}
echo $OUTPUT->box($content, 'generalbox p-3 mb-3');

$form->set_data([
    'courseid' => $course->id,
    'taskid' => $task->taskid,
    'attemptkey' => random_string(24),
    'starttime' => time(),
]);
$form->display();

echo html_writer::div(html_writer::link(
    new moodle_url('/local/aitutor/practice.php', ['courseid' => $course->id]),
    get_string('backtolist', 'local_aitutor')), 'mt-3');
echo $OUTPUT->footer();
