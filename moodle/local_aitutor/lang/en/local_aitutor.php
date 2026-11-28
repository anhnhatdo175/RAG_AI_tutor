<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * English strings for local_aitutor.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'AI tutor';
$string['aitutor:manage'] = 'Generate and review AI tutor tasks';
$string['aitutor:practice'] = 'Practise AI tutor tasks';

// Settings.
$string['serviceurl'] = 'Service URL';
$string['serviceurl_desc'] = 'Base URL of the tutoring service. From the Moodle Docker containers the service on the host is http://host.docker.internal:8800.';
$string['servicetoken'] = 'Service token';
$string['servicetoken_desc'] = 'Same value as SERVICE_TOKEN in the service .env file. Sent as "Authorization: Bearer".';
$string['idsalt'] = 'Pseudonym salt';
$string['idsalt_desc'] = 'Secret used to turn Moodle user and course ids into pseudonyms before they are sent. Do not change it after learners have started, or their history in the service is lost.';
$string['timeout'] = 'Request timeout (seconds)';
$string['timeout_desc'] = 'Task generation with a local LLM can take several minutes.';
$string['skills'] = 'Suggested skills';
$string['skills_desc'] = 'One skill tag per line, offered in the teacher form. Teachers can still type another tag.';

// Errors.
$string['error_notconfigured'] = 'The AI tutor is not configured: set the service URL and token in the plugin settings.';
$string['error_nosalt'] = 'The AI tutor is not configured: set the pseudonym salt in the plugin settings.';
$string['error_unreachable'] = 'The tutoring service cannot be reached ({$a}).';
$string['error_httperror'] = 'The tutoring service refused the request ({$a}).';
$string['error_badresponse'] = 'The tutoring service sent an answer that is not JSON.';
$string['formexpired'] = 'This form is no longer valid. Reload the page and try again.';
$string['notenrolled'] = 'This user is not enrolled in the course.';

// Pages.
$string['teacherpage'] = 'AI tutor: tasks';
$string['practicepage'] = 'AI tutor: practice';
$string['profilepage'] = 'AI tutor: profile';
$string['task_sync'] = 'Send quiz answers and pending attempts to the tutoring service';

// Teacher page.
$string['generateheader'] = 'Generate tasks';
$string['skilltag'] = 'Skill';
$string['skilltag_help'] = 'Skill tag known to the tutoring service, for example Unit-Rate. The service needs at least one source exercise of this skill at the chosen level.';
$string['skilltag_choose'] = 'Choose or type a skill';
$string['level'] = 'Level';
$string['level1'] = '1 (easy)';
$string['level2'] = '2 (medium)';
$string['level3'] = '3 (hard)';
$string['count'] = 'Number of tasks';
$string['grade'] = 'Grade (optional)';
$string['grade_help'] = 'Leave empty to let the service choose the grade of the source exercises.';
$string['generate'] = 'Generate';
$string['generated'] = '{$a->count} task(s) generated; {$a->review} need review. Approve a task before learners can see it.';
$string['tasklist'] = 'Tasks of this course';
$string['notasks'] = 'No task yet. Use the form above.';
$string['attempts'] = 'Attempts';
$string['needsreview_help'] = 'The model output failed a check. Edit the task (add an answer key if needed) before approving it.';
$string['edittask'] = 'Edit task';
$string['prompt'] = 'Task';
$string['hints'] = 'Hints';
$string['hints_help'] = 'One hint per line, at most 10.';
$string['reason'] = 'Why this task';
$string['answerkey'] = 'New answer key';
$string['answerkey_help'] = 'The current answer key stays on the service and is never shown in Moodle. Leave empty to keep it.';
$string['toomanyhints'] = 'At most 10 hints.';
$string['review_approve'] = 'Approve';
$string['review_edit'] = 'Edit';
$string['review_reject'] = 'Reject';
$string['review_approve_done'] = 'Task approved: learners can now practise it.';
$string['review_reject_done'] = 'Task rejected.';
$string['review_edit_done'] = 'Task saved as draft. Approve it to give it to learners.';

// Statuses.
$string['status_draft'] = 'Draft';
$string['status_needs_review'] = 'Needs review';
$string['status_approved'] = 'Approved';
$string['status_rejected'] = 'Rejected';
$string['status_pending'] = 'Waiting to be sent';
$string['status_graded'] = 'Graded';
$string['status_processing'] = 'Processing';
$string['statusinfo_pending'] = 'The service could not be reached. Your answer is saved and will be graded automatically.';
$string['statusinfo_needs_review'] = 'Your teacher will check this answer.';
$string['statusinfo_processing'] = 'Your answer is being graded. Reload the page in a moment.';
$string['statusinfo_rejected'] = 'The service did not accept this answer.';

// Practice page.
$string['nopractice'] = 'No task has been approved for this course yet.';
$string['lastresult'] = 'Last result';
$string['start'] = 'Start';
$string['tryagain'] = 'Try again';
$string['youranswer'] = 'Your answer';
$string['submitanswer'] = 'Submit answer';
$string['savedforlater'] = 'The tutoring service is not available. Your answer is saved and will be graded later.';
$string['result'] = 'Result';
$string['correct'] = 'Correct';
$string['incorrect'] = 'Not correct';
$string['newmastery'] = 'Your mastery';
$string['masterybefore'] = 'Before';
$string['masteryafter'] = 'After';
$string['showhints'] = 'Show hints';
$string['sources'] = 'Sources';
$string['backtolist'] = 'Back to the task list';

// Profile page.
$string['nolearners'] = 'No learner in this course.';
$string['viewprofile'] = 'View profile';
$string['totalattempts'] = 'Answers graded by the service: {$a}';
$string['coldstart'] = 'No answer yet: every skill starts at mastery 0.5.';
$string['weakskills'] = 'Weak skills';
$string['strongskills'] = 'Strong skills';
$string['mastery'] = 'Mastery';
$string['practisedskills'] = 'Practised skills';
$string['allskills'] = 'All skills ({$a})';
$string['history'] = 'Latest answers';

// Privacy.
$string['privacy:metadata:attempt'] = 'Answers given by learners on AI tutor tasks.';
$string['privacy:metadata:attempt:userid'] = 'The learner.';
$string['privacy:metadata:attempt:taskid'] = 'The task answered.';
$string['privacy:metadata:attempt:answer'] = 'The answer given.';
$string['privacy:metadata:attempt:correct'] = 'Whether the answer was correct.';
$string['privacy:metadata:attempt:feedback'] = 'The feedback returned by the service.';
$string['privacy:metadata:attempt:masteryafter'] = 'Skill mastery after the answer.';
$string['privacy:metadata:attempt:timecreated'] = 'When the answer was given.';
$string['privacy:metadata:task'] = 'Tasks generated for a course.';
$string['privacy:metadata:task:createdby'] = 'The teacher who generated the task.';
$string['privacy:metadata:service'] = 'Answers are sent to the tutoring service for grading and the learner model. Users are identified only by a keyed pseudonym.';
$string['privacy:metadata:service:pseudonym'] = 'HMAC pseudonym of the Moodle user id.';
$string['privacy:metadata:service:answer'] = 'The answer given.';
