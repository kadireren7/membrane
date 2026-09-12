#include "service_state.h"

#include <cstdlib>
#include <sstream>

#include <sys/stat.h>

#ifndef _WIN32
# include <unistd.h>
#endif

#include "subprocess.h"
#include "systemd_unit.h"
#include "launchd_unit.h"
#include "windows_task.h"

/* See service_state.h's own top comment for this test-only hook's
 * contract. Returns false (probe left untouched) if the env var is
 * unset or names something other than the four recognized states. */
static bool	probe_from_test_override(membrane_service_probe_t *probe)
{
	const char	*override_val = getenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);

	if (override_val == NULL || override_val[0] == '\0')
		return (false);
	std::string	ov(override_val);

	probe->manager_name = "test-override";
	if (ov == "not_installed")
	{
		probe->manager_available = true;
		probe->installed = false;
		probe->active = false;
		probe->active_state_raw = "inactive";
	}
	else if (ov == "installed_inactive")
	{
		probe->manager_available = true;
		probe->installed = true;
		probe->active = false;
		probe->active_state_raw = "inactive";
	}
	else if (ov == "installed_active")
	{
		probe->manager_available = true;
		probe->installed = true;
		probe->active = true;
		probe->active_state_raw = "active";
	}
	else if (ov == "manager_unavailable")
	{
		probe->manager_available = false;
		probe->installed = false;
		probe->active = false;
		probe->active_state_raw = "unknown";
	}
	else
		return (false);
	return (true);
}

/*
 * Extracted verbatim (same real subprocess calls, same per-platform
 * #ifdef isolation) from doctor_cmd.cpp's pre-existing check_service() --
 * see service_state.h's own top comment for why this now lives in one
 * shared place instead of five independently-drifting readings of the
 * same real systemctl/launchctl/schtasks state.
 */
membrane_service_probe_t	membrane_probe_service(void)
{
	membrane_service_probe_t	probe;

	probe.manager_available = true;
	probe.installed = false;
	probe.active = false;
	probe.active_state_raw = "unknown";
	if (probe_from_test_override(&probe))
		return (probe);
#ifdef __APPLE__
	probe.manager_name = "launchctl";

	membrane_subprocess_result_t	print_result;
	bool	ran = membrane_run_subprocess({"launchctl", "print",
			"gui/" + std::to_string(getuid()) + "/" MEMBRANE_LAUNCHD_LABEL},
			&print_result, 5);

	if (!ran || print_result.exit_code == 127)
		probe.manager_available = false;
	else if (print_result.exit_code == 0)
	{
		probe.installed = true;
		std::istringstream	iss(print_result.stdout_output);
		std::string			line;

		while (std::getline(iss, line))
		{
			size_t	eq = line.find('=');

			if (eq == std::string::npos)
				continue ;
			std::string	key = line.substr(0, eq);
			size_t		ks = key.find_first_not_of(" \t");
			size_t		ke = key.find_last_not_of(" \t");

			if (ks == std::string::npos)
				continue ;
			key = key.substr(ks, ke - ks + 1);
			if (key == "state")
			{
				std::string	val = line.substr(eq + 1);
				size_t		vs = val.find_first_not_of(" \t");

				probe.active_state_raw = vs != std::string::npos
						? val.substr(vs) : "";
			}
		}
	}
#elif defined(_WIN32)
	probe.manager_name = "schtasks";

	membrane_subprocess_result_t	query_result;
	bool	ran = membrane_run_subprocess({"schtasks", "/query", "/tn",
			membrane_task_name(), "/fo", "list", "/v"}, &query_result, 5);

	if (!ran || query_result.exit_code == 1)
		probe.manager_available = false;
	else if (query_result.exit_code == 0)
	{
		probe.installed = true;
		std::istringstream	iss(query_result.stdout_output);
		std::string			line;

		while (std::getline(iss, line))
		{
			size_t	colon = line.find(':');

			if (colon == std::string::npos)
				continue ;
			if (line.substr(0, colon).find("Status") != std::string::npos)
			{
				std::string	val = line.substr(colon + 1);
				size_t		vs = val.find_first_not_of(" \t");

				probe.active_state_raw = vs != std::string::npos
						? val.substr(vs) : "";
			}
		}
	}
#else
	probe.manager_name = "systemctl";

	std::string	unit_path = membrane_unit_file_path();
	bool		unit_exists = false;

	if (!unit_path.empty())
	{
		struct stat	st;

		unit_exists = (stat(unit_path.c_str(), &st) == 0);
	}
	membrane_subprocess_result_t	show_result;
	bool	ran = membrane_run_subprocess({"systemctl", "--user", "show",
			MEMBRANE_UNIT_NAME, "--property=ActiveState,LoadState"},
			&show_result, 5);

	/* execvp() itself failing (systemctl not found on PATH at all) makes
	 * the CHILD exit 127 (subprocess.h's own documented execvp-failure
	 * convention) -- ran is still true (fork/pipe setup succeeded), only
	 * exit_code reveals it. */
	if (!ran || show_result.exit_code == 127)
		probe.manager_available = false;
	else if (show_result.exit_code == 0)
	{
		std::istringstream	iss(show_result.stdout_output);
		std::string			line;
		std::string			load_state = "unknown";

		while (std::getline(iss, line))
		{
			size_t	eq = line.find('=');

			if (eq == std::string::npos)
				continue ;
			std::string	key = line.substr(0, eq);
			std::string	val = line.substr(eq + 1);

			if (key == "ActiveState")
				probe.active_state_raw = val;
			else if (key == "LoadState")
				load_state = val;
		}
		probe.installed = (load_state == "loaded");
	}
	else
		probe.installed = unit_exists;
#endif
	if (!probe.manager_available)
	{
		probe.installed = false;
		probe.active = false;
		return (probe);
	}
	probe.active = (probe.active_state_raw == "active"
			|| probe.active_state_raw == "Running");
	return (probe);
}

bool	membrane_service_start_should_be_blocked(
			const membrane_service_probe_t &probe)
{
	/* Only blocks when the real service manager IS available but genuinely
	 * has no MEMBRANE unit/plist/task installed -- a manager that is
	 * itself unusable (systemctl/launchctl/schtasks missing entirely)
	 * falls through to the existing, already-tested IO_ERROR path
	 * service_cmd.cpp's own verb dispatch produces today. */
	return (probe.manager_available && !probe.installed);
}
