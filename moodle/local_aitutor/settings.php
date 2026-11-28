<?php
// This file is part of the RAG AI tutor project (P3-06).

/**
 * Admin settings: service URL, token, pseudonymisation salt.
 *
 * @package    local_aitutor
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_aitutor', get_string('pluginname', 'local_aitutor'));
    $ADMIN->add('localplugins', $settings);

    if ($ADMIN->fulltree) {
        $settings->add(new admin_setting_configtext(
            'local_aitutor/serviceurl',
            get_string('serviceurl', 'local_aitutor'),
            get_string('serviceurl_desc', 'local_aitutor'),
            'http://host.docker.internal:8800',
            PARAM_URL
        ));
        $settings->add(new admin_setting_configpasswordunmask(
            'local_aitutor/servicetoken',
            get_string('servicetoken', 'local_aitutor'),
            get_string('servicetoken_desc', 'local_aitutor'),
            ''
        ));
        $settings->add(new admin_setting_configpasswordunmask(
            'local_aitutor/idsalt',
            get_string('idsalt', 'local_aitutor'),
            get_string('idsalt_desc', 'local_aitutor'),
            ''
        ));
        $settings->add(new admin_setting_configtext(
            'local_aitutor/timeout',
            get_string('timeout', 'local_aitutor'),
            get_string('timeout_desc', 'local_aitutor'),
            300,
            PARAM_INT
        ));
        $settings->add(new admin_setting_configtextarea(
            'local_aitutor/skills',
            get_string('skills', 'local_aitutor'),
            get_string('skills_desc', 'local_aitutor'),
            "Addition-and-Subtraction-Fractions\nUnit-Rate"
        ));
    }
}
