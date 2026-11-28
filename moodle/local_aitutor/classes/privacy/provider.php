<?php
// This file is part of the RAG AI tutor project (P3-06).

namespace local_aitutor\privacy;

use core_privacy\local\metadata\collection;
use core_privacy\local\request\approved_contextlist;
use core_privacy\local\request\approved_userlist;
use core_privacy\local\request\contextlist;
use core_privacy\local\request\userlist;
use core_privacy\local\request\writer;

/**
 * Privacy provider: learner attempts stored in Moodle and the data sent to the tutoring service.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class provider implements
        \core_privacy\local\metadata\provider,
        \core_privacy\local\request\core_userlist_provider,
        \core_privacy\local\request\plugin\provider {

    /**
     * Describe the stored and exported data.
     *
     * @param collection $collection
     * @return collection
     */
    public static function get_metadata(collection $collection): collection {
        $collection->add_database_table('local_aitutor_attempt', [
            'userid' => 'privacy:metadata:attempt:userid',
            'taskid' => 'privacy:metadata:attempt:taskid',
            'answer' => 'privacy:metadata:attempt:answer',
            'correct' => 'privacy:metadata:attempt:correct',
            'feedback' => 'privacy:metadata:attempt:feedback',
            'masteryafter' => 'privacy:metadata:attempt:masteryafter',
            'timecreated' => 'privacy:metadata:attempt:timecreated',
        ], 'privacy:metadata:attempt');
        $collection->add_database_table('local_aitutor_task', [
            'createdby' => 'privacy:metadata:task:createdby',
        ], 'privacy:metadata:task');
        $collection->add_external_location_link('tutoringservice', [
            'pseudonym' => 'privacy:metadata:service:pseudonym',
            'answer' => 'privacy:metadata:service:answer',
        ], 'privacy:metadata:service');
        return $collection;
    }

    /**
     * Course contexts with attempts by the user.
     *
     * @param int $userid
     * @return contextlist
     */
    public static function get_contexts_for_userid(int $userid): contextlist {
        $contextlist = new contextlist();
        $contextlist->add_from_sql(
            "SELECT ctx.id
               FROM {context} ctx
               JOIN {local_aitutor_attempt} a ON a.courseid = ctx.instanceid AND ctx.contextlevel = :level
              WHERE a.userid = :userid",
            ['level' => CONTEXT_COURSE, 'userid' => $userid]
        );
        return $contextlist;
    }

    /**
     * Users with attempts in a course context.
     *
     * @param userlist $userlist
     */
    public static function get_users_in_context(userlist $userlist) {
        $context = $userlist->get_context();
        if ($context->contextlevel != CONTEXT_COURSE) {
            return;
        }
        $userlist->add_from_sql('userid', "SELECT userid FROM {local_aitutor_attempt} WHERE courseid = :courseid",
            ['courseid' => $context->instanceid]);
    }

    /**
     * Export attempts.
     *
     * @param approved_contextlist $contextlist
     */
    public static function export_user_data(approved_contextlist $contextlist) {
        global $DB;
        $userid = $contextlist->get_user()->id;
        foreach ($contextlist->get_contexts() as $context) {
            if ($context->contextlevel != CONTEXT_COURSE) {
                continue;
            }
            $rows = $DB->get_records('local_aitutor_attempt', ['courseid' => $context->instanceid, 'userid' => $userid],
                'timecreated ASC');
            $data = array_map(fn($row) => (object)[
                'taskid' => $row->taskid,
                'answer' => $row->answer,
                'status' => $row->status,
                'correct' => $row->correct,
                'feedback' => $row->feedback,
                'masteryafter' => $row->masteryafter,
                'timecreated' => \core_privacy\local\request\transform::datetime($row->timecreated),
            ], array_values($rows));
            writer::with_context($context)->export_data([get_string('pluginname', 'local_aitutor')],
                (object)['attempts' => $data]);
        }
    }

    /**
     * Delete all attempts in a context.
     *
     * @param \context $context
     */
    public static function delete_data_for_all_users_in_context(\context $context) {
        global $DB;
        if ($context->contextlevel == CONTEXT_COURSE) {
            $DB->delete_records('local_aitutor_attempt', ['courseid' => $context->instanceid]);
        }
    }

    /**
     * Delete one user's attempts.
     *
     * @param approved_contextlist $contextlist
     */
    public static function delete_data_for_user(approved_contextlist $contextlist) {
        global $DB;
        $userid = $contextlist->get_user()->id;
        foreach ($contextlist->get_contexts() as $context) {
            if ($context->contextlevel == CONTEXT_COURSE) {
                $DB->delete_records('local_aitutor_attempt', ['courseid' => $context->instanceid, 'userid' => $userid]);
            }
        }
    }

    /**
     * Delete several users' attempts in one context.
     *
     * @param approved_userlist $userlist
     */
    public static function delete_data_for_users(approved_userlist $userlist) {
        global $DB;
        $context = $userlist->get_context();
        if ($context->contextlevel != CONTEXT_COURSE || !$userlist->get_userids()) {
            return;
        }
        [$insql, $params] = $DB->get_in_or_equal($userlist->get_userids(), SQL_PARAMS_NAMED);
        $params['courseid'] = $context->instanceid;
        $DB->delete_records_select('local_aitutor_attempt', "courseid = :courseid AND userid {$insql}", $params);
    }
}
