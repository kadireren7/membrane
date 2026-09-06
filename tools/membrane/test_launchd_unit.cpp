#include <cstdlib>
#include <string>

#include "launchd_unit.h"
#include "test_helpers.h"

/*
 * Mega Phase D, PR D4: unit tests for launchd_unit.h's pure plist
 * generation and path resolution -- no filesystem access, no launchctl
 * invocation (see launchd_unit.h's own top comment for the split).
 * Pure string generation/parsing -- runs on any platform, including
 * this project's normal Linux CI, exactly like test_systemd_unit.cpp
 * does not need real systemd to run.
 */

static void	test_generate_plist_has_expected_shape(void)
{
	membrane_launchd_options_t	opts;

	opts.exec_path = "/usr/local/bin/membrane";
	opts.log_path = "";
	std::string	content = membrane_generate_launchd_plist(opts);

	TEST_ASSERT(content.find(MEMBRANE_LAUNCHD_MARKER) != std::string::npos,
		"generated plist carries the MEMBRANE marker");
	TEST_ASSERT(content.find("<key>Label</key>") != std::string::npos,
		"has a Label key");
	TEST_ASSERT(content.find("<string>com.membrane.server</string>")
		!= std::string::npos, "Label value is the fixed MEMBRANE identifier");
	TEST_ASSERT(content.find("<string>/usr/local/bin/membrane</string>")
		!= std::string::npos, "ProgramArguments carries the exec_path as "
		"its own array element -- no shell quoting needed (real XML array "
		"structure, unlike systemd's ExecStart= string quoting)");
	TEST_ASSERT(content.find("<string>serve</string>") != std::string::npos,
		"ProgramArguments appends the bare 'serve' subcommand -- no bind/"
		"port baked in, same reasoning as systemd_unit.h's own ExecStart=");
	TEST_ASSERT(content.find("<key>RunAtLoad</key>\n\t<false/>")
		!= std::string::npos, "RunAtLoad is false -- nothing auto-starts "
		"at login, matching systemd_unit.h's own no-auto-start-at-boot "
		"policy");
	TEST_ASSERT(content.find("<key>SuccessfulExit</key>\n\t\t<false/>")
		!= std::string::npos, "KeepAlive/SuccessfulExit=false -- launchd's "
		"closest analog to Restart=on-failure (restart only on a "
		"non-zero exit)");
	TEST_ASSERT(content.find("<key>ThrottleInterval</key>\n\t<integer>2"
		"</integer>") != std::string::npos, "a bounded restart-throttle "
		"floor is set -- never an unthrottled restart loop");
	TEST_ASSERT(content.find("<key>StandardOutPath</key>") != std::string::npos
		&& content.find("<key>StandardErrorPath</key>") != std::string::npos,
		"stdout/stderr are both redirected to a real log path (launchd has "
		"no journalctl equivalent -- `membrane service logs` tails this "
		"file directly on macOS)");
	TEST_ASSERT(content.find("<string>/tmp/membrane-service.log</string>")
		!= std::string::npos, "a default log_path is applied when none is "
		"given");
}

static void	test_generate_plist_custom_log_path(void)
{
	membrane_launchd_options_t	opts;

	opts.exec_path = "/opt/membrane/bin/membrane";
	opts.log_path = "/Users/dev/Library/Logs/membrane/membrane.log";
	std::string	content = membrane_generate_launchd_plist(opts);

	TEST_ASSERT(content.find(
		"<string>/Users/dev/Library/Logs/membrane/membrane.log</string>")
		!= std::string::npos, "an explicit log_path overrides the default, "
		"used for both stdout and stderr");
}

/* A path containing a space -- ProgramArguments is a real XML array, so
 * unlike systemd_unit.h's own space-in-path test (which proves quoting),
 * this proves no quoting/escaping is needed at all: the space is simply
 * part of the <string> element's own text content. */
static void	test_generate_plist_exec_path_with_space(void)
{
	membrane_launchd_options_t	opts;

	opts.exec_path = "/Users/dev/my apps/membrane";
	opts.log_path = "";
	std::string	content = membrane_generate_launchd_plist(opts);

	TEST_ASSERT(content.find("<string>/Users/dev/my apps/membrane</string>")
		!= std::string::npos, "the exec_path is carried verbatim, space "
		"and all -- no shell-style quoting exists in this format");
}

static void	test_plist_is_membrane_managed(void)
{
	membrane_launchd_options_t	opts;

	opts.exec_path = "/usr/local/bin/membrane";
	opts.log_path = "";
	std::string	generated = membrane_generate_launchd_plist(opts);

	TEST_ASSERT(membrane_launchd_plist_is_membrane_managed(generated) == true,
		"a freshly generated plist is recognized as MEMBRANE-managed");
	TEST_ASSERT(membrane_launchd_plist_is_membrane_managed(
		"<plist version=\"1.0\"><dict><key>Label</key>"
		"<string>com.other.thing</string></dict></plist>") == false,
		"an unrelated plist (no marker) is NOT recognized as "
		"MEMBRANE-managed -- this is the refuse-to-overwrite check's own "
		"foundation, same policy as systemd_unit.h's own managed check");
	TEST_ASSERT(membrane_launchd_plist_is_membrane_managed("") == false,
		"an empty string is not managed");
}

static void	with_env_saved(const char *name, const char *value,
				void (*fn)(void))
{
	const char	*old = getenv(name);
	bool		had = old != NULL;
	std::string	saved = had ? old : "";

	if (value != NULL)
		setenv(name, value, 1);
	else
		unsetenv(name);
	fn();
	if (had)
		setenv(name, saved.c_str(), 1);
	else
		unsetenv(name);
}

static void	check_override_dir(void)
{
	TEST_ASSERT(membrane_launchd_plist_path()
		== "/tmp/membrane-launchd-test-dir/com.membrane.server.plist",
		"MEMBRANE_LAUNCHD_USER_DIR overrides the directory outright");
}

static void	check_home_default(void)
{
	std::string	path = membrane_launchd_plist_path();

	TEST_ASSERT(path.find("/Library/LaunchAgents/com.membrane.server.plist")
		!= std::string::npos,
		"uses $HOME/Library/LaunchAgents/com.membrane.server.plist -- no "
		"XDG fallback (unlike systemd_unit.h's Linux path resolution, "
		"macOS has no XDG_CONFIG_HOME convention)");
}

static void	test_plist_path_resolution(void)
{
	const char	*old_override = getenv("MEMBRANE_LAUNCHD_USER_DIR");
	std::string	saved_override = old_override != NULL ? old_override : "";
	bool		had_override = old_override != NULL;

	unsetenv("MEMBRANE_LAUNCHD_USER_DIR");
	with_env_saved("MEMBRANE_LAUNCHD_USER_DIR",
		"/tmp/membrane-launchd-test-dir", check_override_dir);
	unsetenv("MEMBRANE_LAUNCHD_USER_DIR");
	check_home_default();
	if (had_override)
		setenv("MEMBRANE_LAUNCHD_USER_DIR", saved_override.c_str(), 1);
}

int	main(void)
{
	test_generate_plist_has_expected_shape();
	test_generate_plist_custom_log_path();
	test_generate_plist_exec_path_with_space();
	test_plist_is_membrane_managed();
	test_plist_path_resolution();
	printf("test_launchd_unit: all tests passed\n");
	return (0);
}
