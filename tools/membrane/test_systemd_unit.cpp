#include <cstdlib>
#include <string>

#include "systemd_unit.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B1: unit tests for systemd_unit.h's pure unit-file
 * generation and path resolution -- no filesystem access, no systemctl
 * invocation (see systemd_unit.h's own top comment for the split).
 */

static void	test_generate_unit_file_has_expected_shape(void)
{
	membrane_unit_options_t	opts;

	opts.exec_path = "/usr/bin/membrane";
	opts.description = "";
	std::string	content = membrane_generate_unit_file(opts);

	TEST_ASSERT(content.find(MEMBRANE_UNIT_MARKER) != std::string::npos,
		"generated unit carries the MEMBRANE marker");
	TEST_ASSERT(content.find("[Unit]") != std::string::npos,
		"has a [Unit] section");
	TEST_ASSERT(content.find("[Service]") != std::string::npos,
		"has a [Service] section");
	TEST_ASSERT(content.find("[Install]") != std::string::npos,
		"has an [Install] section");
	TEST_ASSERT(content.find("Type=simple") != std::string::npos,
		"Type=simple");
	TEST_ASSERT(content.find("ExecStart=\"/usr/bin/membrane\" serve")
		!= std::string::npos,
		"ExecStart quotes the exec_path and appends the bare 'serve' "
		"subcommand -- no bind/port baked in, so changing server_config.h's "
		"config never requires regenerating the unit");
	TEST_ASSERT(content.find("Restart=on-failure") != std::string::npos,
		"Restart=on-failure -- never an aggressive restart loop");
	TEST_ASSERT(content.find("RestartSec=2") != std::string::npos,
		"RestartSec=2 -- a bounded restart delay");
	TEST_ASSERT(content.find("WantedBy=default.target") != std::string::npos,
		"WantedBy=default.target -- a normal --user unit, no root");
	TEST_ASSERT(content.find("Description=MEMBRANE local inference server")
		!= std::string::npos, "a default Description is applied when none "
		"is given");
}

static void	test_generate_unit_file_custom_description(void)
{
	membrane_unit_options_t	opts;

	opts.exec_path = "/opt/membrane/bin/membrane";
	opts.description = "My Custom Description";
	std::string	content = membrane_generate_unit_file(opts);

	TEST_ASSERT(content.find("Description=My Custom Description")
		!= std::string::npos, "an explicit description overrides the "
		"default");
}

/* A path containing a space -- ExecStart= is systemd's own quoting, not a
 * shell, so a quoted path with an embedded space is exactly what real
 * systemd expects here. */
static void	test_generate_unit_file_exec_path_with_space(void)
{
	membrane_unit_options_t	opts;

	opts.exec_path = "/home/user/my apps/membrane";
	opts.description = "";
	std::string	content = membrane_generate_unit_file(opts);

	TEST_ASSERT(content.find("ExecStart=\"/home/user/my apps/membrane\" "
		"serve") != std::string::npos, "the exec_path is quoted whole, "
		"space and all");
}

static void	test_unit_is_membrane_managed(void)
{
	membrane_unit_options_t	opts;

	opts.exec_path = "/usr/bin/membrane";
	opts.description = "";
	std::string	generated = membrane_generate_unit_file(opts);

	TEST_ASSERT(membrane_unit_is_membrane_managed(generated) == true,
		"a freshly generated unit is recognized as MEMBRANE-managed");
	TEST_ASSERT(membrane_unit_is_membrane_managed(
		"[Unit]\nDescription=Some other service\n") == false,
		"an unrelated unit file (no marker) is NOT recognized as "
		"MEMBRANE-managed -- this is the refuse-to-overwrite check's own "
		"foundation");
	TEST_ASSERT(membrane_unit_is_membrane_managed("") == false,
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
	TEST_ASSERT(membrane_unit_file_path()
		== "/tmp/membrane-unit-test-dir/membrane.service",
		"MEMBRANE_SYSTEMD_USER_DIR overrides the directory outright");
}

static void	check_xdg(void)
{
	TEST_ASSERT(membrane_unit_file_path()
		== "/tmp/xdgtest/systemd/user/membrane.service",
		"XDG_CONFIG_HOME/systemd/user/membrane.service is used when set");
}

static void	check_home_fallback(void)
{
	std::string	path = membrane_unit_file_path();

	TEST_ASSERT(path.find("/.config/systemd/user/membrane.service")
		!= std::string::npos,
		"falls back to $HOME/.config/systemd/user/membrane.service");
}

static void	test_unit_file_path_resolution(void)
{
	const char	*old_override = getenv("MEMBRANE_SYSTEMD_USER_DIR");
	std::string	saved_override = old_override != NULL ? old_override : "";
	bool		had_override = old_override != NULL;
	const char	*old_xdg = getenv("XDG_CONFIG_HOME");
	std::string	saved_xdg = old_xdg != NULL ? old_xdg : "";
	bool		had_xdg = old_xdg != NULL;

	unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
	with_env_saved("MEMBRANE_SYSTEMD_USER_DIR",
		"/tmp/membrane-unit-test-dir", check_override_dir);
	unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
	with_env_saved("XDG_CONFIG_HOME", "/tmp/xdgtest", check_xdg);
	unsetenv("XDG_CONFIG_HOME");
	check_home_fallback();
	if (had_override)
		setenv("MEMBRANE_SYSTEMD_USER_DIR", saved_override.c_str(), 1);
	if (had_xdg)
		setenv("XDG_CONFIG_HOME", saved_xdg.c_str(), 1);
}

int	main(void)
{
	test_generate_unit_file_has_expected_shape();
	test_generate_unit_file_custom_description();
	test_generate_unit_file_exec_path_with_space();
	test_unit_is_membrane_managed();
	test_unit_file_path_resolution();
	printf("test_systemd_unit: all tests passed\n");
	return (0);
}
