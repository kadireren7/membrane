#include <cstdlib>
#include <string>

#include "windows_task.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D5: unit tests for windows_task.h's pure Task
 * Scheduler XML generation -- no filesystem access, no schtasks.exe
 * invocation (see windows_task.h's own top comment for the split).
 * Pure string generation/parsing -- runs on any platform, including
 * this project's normal Linux CI, exactly like test_systemd_unit.cpp/
 * test_launchd_unit.cpp don't need real systemd/launchd to run.
 */

static void	test_generate_task_xml_has_expected_shape(void)
{
	membrane_task_options_t	opts;

	opts.exec_path = "C:\\Program Files\\membrane\\membrane.exe";
	opts.log_path = "";
	std::string	xml = membrane_generate_task_xml(opts);

	TEST_ASSERT(xml.find(MEMBRANE_TASK_MARKER) != std::string::npos,
		"generated XML carries the MEMBRANE marker");
	TEST_ASSERT(xml.find("<Task version=\"1.2\"") != std::string::npos,
		"a real Task Scheduler 1.2 <Task> root element");
	TEST_ASSERT(xml.find("<Triggers />") != std::string::npos,
		"no triggers registered -- nothing auto-starts this task, matching "
		"systemd_unit.h/launchd_unit.h's own no-auto-start-without-being-"
		"asked policy");
	TEST_ASSERT(xml.find("<LogonType>InteractiveToken</LogonType>")
		!= std::string::npos, "runs as the interactive user, not a service "
		"account");
	TEST_ASSERT(xml.find("<RunLevel>LeastPrivilege</RunLevel>")
		!= std::string::npos, "no elevation requested -- matches this "
		"project's own no-root/no-elevated-capability policy");
	TEST_ASSERT(xml.find("<Command>cmd.exe</Command>") != std::string::npos,
		"runs via cmd.exe so real stdout/stderr redirection is possible");
	TEST_ASSERT(xml.find("C:\\Program Files\\membrane\\membrane.exe")
		!= std::string::npos, "the exec_path is carried into the "
		"Arguments element, escaped");
	TEST_ASSERT(xml.find("serve") != std::string::npos, "the bare 'serve' "
		"subcommand is appended -- no bind/port baked in, same reasoning "
		"as systemd_unit.h's own ExecStart=");
	TEST_ASSERT(xml.find("C:\\Windows\\Temp\\membrane-service.log")
		!= std::string::npos, "a default log_path is applied when none is "
		"given");
}

static void	test_generate_task_xml_custom_log_path(void)
{
	membrane_task_options_t	opts;

	opts.exec_path = "C:\\membrane\\membrane.exe";
	opts.log_path = "C:\\Users\\dev\\membrane.log";
	std::string	xml = membrane_generate_task_xml(opts);

	TEST_ASSERT(xml.find("C:\\Users\\dev\\membrane.log") != std::string::npos,
		"an explicit log_path overrides the default");
}

static void	test_generate_task_xml_escapes_special_characters(void)
{
	membrane_task_options_t	opts;

	opts.exec_path = "C:\\Program Files\\membrane & co\\membrane.exe";
	opts.log_path = "";
	std::string	xml = membrane_generate_task_xml(opts);

	TEST_ASSERT(xml.find("membrane &amp; co") != std::string::npos,
		"a literal '&' in the exec_path is XML-escaped, not left raw "
		"(which would produce invalid XML schtasks.exe would refuse to "
		"parse)");
}

static void	test_task_xml_is_membrane_managed(void)
{
	membrane_task_options_t	opts;

	opts.exec_path = "C:\\membrane\\membrane.exe";
	opts.log_path = "";
	std::string	generated = membrane_generate_task_xml(opts);

	TEST_ASSERT(membrane_task_xml_is_membrane_managed(generated) == true,
		"freshly generated XML is recognized as MEMBRANE-managed");
	TEST_ASSERT(membrane_task_xml_is_membrane_managed(
		"<Task version=\"1.2\"><RegistrationInfo><Description>Some other "
		"task</Description></RegistrationInfo></Task>") == false,
		"an unrelated task's XML (no marker) is NOT recognized as "
		"MEMBRANE-managed -- this is the refuse-to-overwrite check's own "
		"foundation, same policy as systemd_unit.h/launchd_unit.h's own "
		"managed checks");
	TEST_ASSERT(membrane_task_xml_is_membrane_managed("") == false,
		"an empty string is not managed");
}

static void	check_task_name_default(void)
{
	TEST_ASSERT(membrane_task_name() == "MembraneServer",
		"the default task name is the fixed MEMBRANE identifier");
}

static void	check_task_name_override(void)
{
	TEST_ASSERT(membrane_task_name() == "MembraneTestOverride",
		"MEMBRANE_TASK_NAME_OVERRIDE overrides the name outright -- "
		"tests/CI never touch a real user's own scheduled task under the "
		"real default name");
}

static void	test_task_name(void)
{
	const char	*old_override = getenv("MEMBRANE_TASK_NAME_OVERRIDE");
	std::string	saved_override = old_override != NULL ? old_override : "";
	bool		had_override = old_override != NULL;

	unsetenv("MEMBRANE_TASK_NAME_OVERRIDE");
	check_task_name_default();
	setenv("MEMBRANE_TASK_NAME_OVERRIDE", "MembraneTestOverride", 1);
	check_task_name_override();
	unsetenv("MEMBRANE_TASK_NAME_OVERRIDE");
	if (had_override)
		setenv("MEMBRANE_TASK_NAME_OVERRIDE", saved_override.c_str(), 1);
}

int	main(void)
{
	test_generate_task_xml_has_expected_shape();
	test_generate_task_xml_custom_log_path();
	test_generate_task_xml_escapes_special_characters();
	test_task_xml_is_membrane_managed();
	test_task_name();
	printf("test_windows_task: all tests passed\n");
	return (0);
}
