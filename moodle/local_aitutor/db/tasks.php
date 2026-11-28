<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Scheduled tasks.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$tasks = [
    [
        'classname' => 'local_aitutor\task\sync_attempts',
        'blocking' => 0,
        'minute' => '*/15',
        'hour' => '*',
        'day' => '*',
        'month' => '*',
        'dayofweek' => '*',
    ],
];
