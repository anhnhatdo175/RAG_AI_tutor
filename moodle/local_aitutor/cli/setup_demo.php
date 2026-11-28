<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Create the demo data: course "Toán – Demo", one teacher, ten simulated learners.
 * All names are fictional; no real person's data is used.
 *
 * Usage: php local/aitutor/cli/setup_demo.php --password=Demo#2026
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('CLI_SCRIPT', true);

require(__DIR__ . '/../../../config.php');
require_once($CFG->libdir . '/clilib.php');
require_once($CFG->dirroot . '/course/lib.php');
require_once($CFG->dirroot . '/user/lib.php');
require_once($CFG->libdir . '/enrollib.php');

[$options, $unrecognized] = cli_get_params(
    ['password' => 'Demo#2026', 'help' => false],
    ['p' => 'password', 'h' => 'help']
);
if ($options['help']) {
    echo "Create the AI tutor demo course, one teacher and ten learners.\n\n" .
        "Options:\n  -p, --password  password for every demo account (default Demo#2026)\n";
    exit(0);
}

$admin = get_admin();
\core\session\manager::set_user($admin);

$shortname = 'TOAN-DEMO';
$course = $DB->get_record('course', ['shortname' => $shortname]);
if (!$course) {
    $course = create_course((object)[
        'fullname' => 'Toán – Demo',
        'shortname' => $shortname,
        'category' => core_course_category::get_default()->id,
        'summary' => 'Khóa demo cho plugin local_aitutor (học sinh mô phỏng).',
        'format' => 'topics',
        'numsections' => 1,
    ]);
    cli_writeln("Course created: {$course->fullname} (id {$course->id})");
} else {
    cli_writeln("Course exists: {$course->fullname} (id {$course->id})");
}

$people = [['gv.demo', 'Giáo viên', 'Demo', 'editingteacher']];
$names = ['An', 'Bình', 'Chi', 'Dũng', 'Giang', 'Hà', 'Khánh', 'Lan', 'Minh', 'Ngọc'];
foreach ($names as $i => $name) {
    $people[] = [sprintf('hs%02d', $i + 1), $name, 'Mô phỏng', 'student'];
}

$roles = $DB->get_records_menu('role', null, '', 'shortname, id');
foreach ($people as [$username, $firstname, $lastname, $role]) {
    $user = $DB->get_record('user', ['username' => $username, 'mnethostid' => $CFG->mnet_localhost_id]);
    if (!$user) {
        $id = user_create_user((object)[
            'username' => $username,
            'password' => $options['password'],
            'firstname' => $firstname,
            'lastname' => $lastname,
            'email' => $username . '@example.invalid',
            'auth' => 'manual',
            'confirmed' => 1,
            'mnethostid' => $CFG->mnet_localhost_id,
        ], true, false);
        $user = $DB->get_record('user', ['id' => $id], '*', MUST_EXIST);
        cli_writeln("User created: {$username}");
    }
    if (!enrol_try_internal_enrol($course->id, $user->id, $roles[$role])) {
        cli_error("Could not enrol {$username}: enable the manual enrolment plugin.");
    }
}

cli_writeln("Done. Teacher: gv.demo; learners: hs01..hs10; password: {$options['password']}");
