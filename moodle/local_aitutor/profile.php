<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Competence profile from GET /profile: learners see their own, teachers pick a learner.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/lib.php');

use local_aitutor\client_exception;

$courseid = required_param('courseid', PARAM_INT);
$userid = optional_param('userid', 0, PARAM_INT);

$course = get_course($courseid);
require_login($course);
$context = context_course::instance($course->id);
$canmanage = has_capability('local/aitutor:manage', $context);
if (!$canmanage) {
    require_capability('local/aitutor:practice', $context);
    $userid = (int)$USER->id;
}

[$course, $context, $url] = local_aitutor_setup_page($courseid, '/local/aitutor/profile.php',
    $userid ? ['userid' => $userid] : [], $canmanage ? 'local/aitutor:manage' : 'local/aitutor:practice',
    'profilepage');

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('profilepage', 'local_aitutor'));

// Teachers: list the learners of the course.
if (!$userid) {
    $learners = get_enrolled_users($context, 'local/aitutor:practice', 0, 'u.*', 'u.lastname, u.firstname');
    if (!$learners) {
        echo $OUTPUT->notification(get_string('nolearners', 'local_aitutor'), 'info');
    } else {
        $table = new html_table();
        $table->head = [get_string('fullname'), get_string('attempts', 'local_aitutor'), ''];
        foreach ($learners as $learner) {
            $table->data[] = [
                fullname($learner),
                $DB->count_records('local_aitutor_attempt', ['courseid' => $course->id, 'userid' => $learner->id]),
                html_writer::link(new moodle_url($url, ['userid' => $learner->id]),
                    get_string('viewprofile', 'local_aitutor'), ['class' => 'btn btn-secondary btn-sm']),
            ];
        }
        echo html_writer::table($table);
    }
    echo $OUTPUT->footer();
    exit;
}

$learner = core_user::get_user($userid, '*', MUST_EXIST);
if ($canmanage && !is_enrolled($context, $learner)) {
    throw new moodle_exception('notenrolled', 'local_aitutor');
}
if ($canmanage) {
    echo $OUTPUT->heading(fullname($learner), 3);
}

$client = local_aitutor_client_or_notify();
$profile = null;
if ($client) {
    try {
        $profile = $client->profile($userid);
    } catch (client_exception $e) {
        \core\notification::error($e->getMessage());
    }
}

if ($profile) {
    $total = (int)($profile['total_attempts'] ?? 0);
    echo html_writer::div(get_string('totalattempts', 'local_aitutor', $total), 'mb-2');
    if ($total === 0) {
        echo $OUTPUT->notification(get_string('coldstart', 'local_aitutor'), 'info');
    }

    $mastery = $profile['mastery'] ?? [];
    $counts = $profile['attempts_by_skill'] ?? [];

    $bar = function(float $value): string {
        $pct = (int)round(max(0, min(1, $value)) * 100);
        $colour = $value >= 0.8 ? 'success' : ($value < 0.4 ? 'danger' : 'warning');
        return html_writer::div(
            html_writer::div($pct . '%', "progress-bar bg-{$colour}",
                ['role' => 'progressbar', 'style' => "width: {$pct}%", 'aria-valuenow' => $pct,
                    'aria-valuemin' => 0, 'aria-valuemax' => 100]),
            'progress', ['style' => 'min-width: 160px']);
    };

    foreach (['weak_skills' => 'weakskills', 'strong_skills' => 'strongskills'] as $key => $string) {
        $list = $profile[$key] ?? [];
        if ($list) {
            $names = array_map(fn($row) => s($row['skill_tag']) . ' (' . format_float((float)$row['mastery'], 2) . ')', $list);
            echo html_writer::div(html_writer::tag('strong', get_string($string, 'local_aitutor') . ': ') .
                implode(', ', $names), 'mb-1');
        }
    }

    // Practised skills first; the full list (every skill starts at 0.5) is collapsed.
    $practised = array_intersect_key($mastery, array_filter($counts));
    $table = new html_table();
    $table->head = [get_string('skilltag', 'local_aitutor'), get_string('mastery', 'local_aitutor'),
        get_string('attempts', 'local_aitutor')];
    foreach ($practised as $skill => $value) {
        $table->data[] = [s($skill), $bar((float)$value), (int)($counts[$skill] ?? 0)];
    }
    if ($table->data) {
        echo $OUTPUT->heading(get_string('practisedskills', 'local_aitutor'), 4);
        echo html_writer::table($table);
    }

    $all = new html_table();
    $all->head = $table->head;
    foreach ($mastery as $skill => $value) {
        $all->data[] = [s($skill), $bar((float)$value), (int)($counts[$skill] ?? 0)];
    }
    echo html_writer::tag('details',
        html_writer::tag('summary', get_string('allskills', 'local_aitutor', count($mastery))) .
        html_writer::table($all), ['class' => 'mt-3']);
}

// Moodle-side history.
$attempts = $DB->get_records('local_aitutor_attempt', ['courseid' => $course->id, 'userid' => $userid],
    'timecreated DESC, id DESC', '*', 0, 20);
if ($attempts) {
    echo $OUTPUT->heading(get_string('history', 'local_aitutor'), 4);
    $table = new html_table();
    $table->head = [get_string('date'), get_string('skilltag', 'local_aitutor'), get_string('youranswer', 'local_aitutor'),
        get_string('result', 'local_aitutor')];
    foreach ($attempts as $attempt) {
        $skill = $DB->get_field('local_aitutor_task', 'skilltag', ['taskid' => $attempt->taskid]);
        $result = $attempt->status === 'graded'
            ? ($attempt->correct ? get_string('correct', 'local_aitutor') : get_string('incorrect', 'local_aitutor'))
            : \local_aitutor\store::status_badge($attempt->status);
        $table->data[] = [userdate($attempt->timecreated, get_string('strftimedatetimeshort')),
            s((string)$skill), s(shorten_text((string)$attempt->answer, 60)), $result];
    }
    echo html_writer::table($table);
}

echo $OUTPUT->footer();
