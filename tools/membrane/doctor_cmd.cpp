#include "doctor_cmd.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>

#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "registry_core.h"
#include "server_config.h"
#include "systemd_unit.h"
#include "subprocess.h"
#include "status_client.h"
#include "product_cli.h"

using json = nlohmann::json;

/*
 * See doctor_cmd.h's own top comment for the full contract.
 */

# define MEMBRANE_DOCTOR_STATUS_OK		"OK"
# define MEMBRANE_DOCTOR_STATUS_WARN	"WARN"
# define MEMBRANE_DOCTOR_STATUS_ERROR	"ERROR"

struct s_doctor_check
{
	std::string	name;
	std::string	status;
	json		detail;
};

static bool	status_worse(const std::string &a, const std::string &b)
{
	auto	rank = [](const std::string &s) -> int
	{
		if (s == MEMBRANE_DOCTOR_STATUS_ERROR)
			return (2);
		if (s == MEMBRANE_DOCTOR_STATUS_WARN)
			return (1);
		return (0);
	};

	return (rank(a) > rank(b));
}

/* /proc/self/exe of the CURRENTLY RUNNING `membrane` binary -- Linux-only,
 * matching this project's existing scope (same primitive service_cmd.cpp's
 * own resolve_exec_path() already uses). Real packaging (Section 17 of
 * the task) always installs `membrane` and `membrane-run` into the SAME
 * bindir, so looking for "membrane-run" right next to this running
 * binary is correct for every real installed case; a dev build with the
 * two binaries in separate build-tree subdirectories is a secondary,
 * gracefully-degraded case (reported as a WARN, never a hard failure --
 * Section 9's own "may aggregate" language never promised every check
 * can always run). MEMBRANE_RUN_EXEC_PATH overrides it outright, the
 * same test-hook convention MEMBRANE_MODELS_PATH/MEMBRANE_SERVER_
 * CONFIG_PATH/MEMBRANE_SYSTEMD_USER_DIR already established.
 */
static bool	resolve_membrane_run_path(std::string *out_path)
{
	const char	*override_path = getenv("MEMBRANE_RUN_EXEC_PATH");

	if (override_path != NULL && override_path[0] != '\0')
	{
		*out_path = override_path;
		return (true);
	}
	char	buf[PATH_MAX];
	ssize_t	n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

	if (n < 0)
		return (false);
	buf[n] = '\0';
	std::string	self_path(buf);
	size_t		slash = self_path.find_last_of('/');

	if (slash == std::string::npos)
		return (false);
	std::string	candidate = self_path.substr(0, slash) + "/membrane-run";
	struct stat	st;

	if (stat(candidate.c_str(), &st) != 0)
		return (false);
	*out_path = candidate;
	return (true);
}

static s_doctor_check	check_installation(void)
{
	s_doctor_check	c;

	c.name = "installation";
	c.status = MEMBRANE_DOCTOR_STATUS_OK;
	c.detail = {{"membrane_version", MEMBRANE_VERSION}};
	return (c);
}

/* Section 5/9: orchestrates the EXISTING membrane-run --doctor --json
 * (Phase 29's own hardware/host-memory diagnostic) via a real subprocess
 * call -- never a second CPU/GPU/RAM detection implementation. */
static s_doctor_check	check_hardware(void)
{
	s_doctor_check	c;

	c.name = "hardware";
	std::string	run_path;

	if (!resolve_membrane_run_path(&run_path))
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail = {{"message", "could not locate the membrane-run binary "
			"next to this one -- hardware/RAM detection skipped (set "
			"MEMBRANE_RUN_EXEC_PATH to override)"}};
		return (c);
	}
	membrane_subprocess_result_t	result;

	if (!membrane_run_subprocess({run_path, "--doctor", "--json"}, &result, 10)
		|| result.exit_code != 0)
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail = {{"message", "membrane-run --doctor --json did not "
			"complete successfully"}};
		return (c);
	}
	try
	{
		json	run_doctor = json::parse(result.stdout_output);

		c.status = MEMBRANE_DOCTOR_STATUS_OK;
		c.detail = {
			{"cpu", run_doctor.value("cpu", std::string("unknown"))},
			{"gpu_backend_available",
				run_doctor.value("gpu_backend_available", false)},
			{"gpu_devices_found", run_doctor.value("gpu_devices_found", 0)},
			{"host_total_bytes", run_doctor.value("host_total_bytes", 0ULL)},
			{"host_available_bytes",
				run_doctor.value("host_available_bytes", 0ULL)},
		};
		/* A real, low-memory warning -- matches host_memory_guard.h's
		 * own conservative spirit, never a hard failure (a small model
		 * may still fit fine). */
		uint64_t	avail = run_doctor.value("host_available_bytes", 0ULL);

		if (run_doctor.value("host_meminfo_available", false)
			&& avail > 0 && avail < (512ULL * 1024 * 1024))
		{
			c.status = MEMBRANE_DOCTOR_STATUS_WARN;
			c.detail["message"] = "host available memory is currently "
				"very low -- model loads may be rejected as infeasible";
		}
	}
	catch (const json::parse_error &)
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail = {{"message", "membrane-run --doctor --json returned "
			"unparseable output"}};
	}
	return (c);
}

static s_doctor_check	check_registry(json *out_models_for_reuse)
{
	s_doctor_check				c;
	std::string					registry_path
			= membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	err;

	c.name = "registry";
	if (registry_path.empty())
	{
		c.status = MEMBRANE_DOCTOR_STATUS_ERROR;
		c.detail = {{"message", "neither XDG_DATA_HOME nor HOME is set -- "
			"cannot locate the model registry"}};
		return (c);
	}
	if (!membrane_registry_load(registry_path, &reg, &err))
	{
		c.status = MEMBRANE_DOCTOR_STATUS_ERROR;
		c.detail = {{"message", err.message}, {"code", err.code}};
		return (c);
	}
	json	models = json::array();
	bool	any_stale = false;

	for (const auto &e : reg.entries)
	{
		struct stat	st;
		e_membrane_registry_stat_status	stat_status;
		uint64_t	size_bytes = 0;
		int64_t		mtime_ns = 0;

		if (stat(e.path.c_str(), &st) == 0)
		{
			stat_status = MEMBRANE_REGISTRY_STAT_OK;
			size_bytes = (uint64_t)st.st_size;
			mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000LL
				+ (int64_t)st.st_mtim.tv_nsec;
		}
		else
			stat_status = (errno == ENOENT) ? MEMBRANE_REGISTRY_STAT_MISSING
				: MEMBRANE_REGISTRY_STAT_ERROR;
		const char	*status = membrane_registry_check_identity(e,
				stat_status, size_bytes, mtime_ns);

		if (strcmp(status, MEMBRANE_REGISTRY_CHECK_OK) != 0)
			any_stale = true;
		models.push_back({{"name", e.name}, {"status", status}});
	}
	c.status = any_stale ? MEMBRANE_DOCTOR_STATUS_WARN
			: MEMBRANE_DOCTOR_STATUS_OK;
	c.detail = {{"count", reg.entries.size()}, {"models", models}};
	if (any_stale)
		c.detail["message"] = "one or more registered models are stale "
			"(moved/deleted/modified since being added) -- see "
			"`membrane model list`";
	*out_models_for_reuse = models;
	return (c);
}

static s_doctor_check	check_default_model(const json &known_models)
{
	s_doctor_check					c;
	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	c.name = "default_model";
	if (!config_path.empty())
		membrane_server_config_load(config_path, &cfg, &cfg_err);
	if (cfg.default_model.empty())
	{
		c.status = MEMBRANE_DOCTOR_STATUS_OK;
		c.detail = {{"configured", false}};
		return (c);
	}
	bool	registered = false;

	for (const auto &m : known_models)
		if (m["name"] == cfg.default_model)
			registered = true;
	c.detail = {{"configured", true}, {"name", cfg.default_model},
		{"registered", registered}};
	if (registered)
		c.status = MEMBRANE_DOCTOR_STATUS_OK;
	else
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail["message"] = "default_model '" + cfg.default_model
			+ "' is not (or no longer) registered -- a chat request "
			"relying on it will get a real 404, never a startup failure";
	}
	return (c);
}

static s_doctor_check	check_config(void)
{
	s_doctor_check					c;
	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;

	c.name = "config";
	if (config_path.empty())
	{
		c.status = MEMBRANE_DOCTOR_STATUS_ERROR;
		c.detail = {{"message", "neither XDG_CONFIG_HOME nor HOME is set"}};
		return (c);
	}
	if (!membrane_server_config_load(config_path, &cfg, &cfg_err))
	{
		c.status = MEMBRANE_DOCTOR_STATUS_ERROR;
		c.detail = {{"message", cfg_err.message}, {"code", cfg_err.code}};
		return (c);
	}
	c.status = MEMBRANE_DOCTOR_STATUS_OK;
	c.detail = {{"listen_address", cfg.listen_address}, {"port", cfg.port}};
	return (c);
}

/* Section 9/Section 11 (Mega Phase C audit finding): a bare `ubuntu:24.04`
 * container has no systemctl at all -- `membrane service install/start`
 * used to fail there with an unhelpful "(no output)" message and no
 * proactive warning. Checking for systemctl's own presence FIRST and
 * reporting it clearly closes that real, observed gap. */
static s_doctor_check	check_service(bool *out_installed, bool *out_active,
				int *out_port)
{
	s_doctor_check	c;

	c.name = "service";
	*out_installed = false;
	*out_active = false;

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

	/* execvp() itself failing (systemctl not found on PATH at all -- a
	 * real, observed condition in a bare `ubuntu:24.04` container, see
	 * this file's own top comment) makes the CHILD exit 127
	 * (subprocess.h's own documented execvp-failure convention) -- ran
	 * is still true (fork/pipe setup succeeded), only exit_code reveals
	 * it. Reported as its own clear WARN rather than the old unhelpful
	 * "(no output)" a real `membrane service start` attempt used to
	 * give in this exact situation. */
	if (!ran || show_result.exit_code == 127)
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail = {{"systemctl_available", false},
			{"message", "systemctl is not available in this environment "
				"-- `membrane service` commands cannot work here; use "
				"`membrane serve` directly instead"}};
		return (c);
	}
	std::string	active_state = "unknown";

	if (show_result.exit_code == 0)
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
				active_state = val;
			else if (key == "LoadState")
				load_state = val;
		}
		*out_installed = (load_state == "loaded");
	}
	else
		*out_installed = unit_exists;
	*out_active = (active_state == "active");
	c.status = MEMBRANE_DOCTOR_STATUS_OK;
	c.detail = {{"systemctl_available", true}, {"installed", *out_installed},
		{"active_state", active_state}};
	if (!*out_installed)
		c.detail["message"] = "not installed -- run `membrane service "
			"install` to run MEMBRANE as a background service";
	(void)out_port;
	return (c);
}

static s_doctor_check	check_http(bool service_installed, bool service_active)
{
	s_doctor_check					c;
	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	c.name = "http";
	if (!config_path.empty())
		membrane_server_config_load(config_path, &cfg, &cfg_err);
	json	status_json;
	bool	reachable = membrane_fetch_server_status(cfg.listen_address,
			cfg.port, &status_json);

	if (reachable)
	{
		c.status = MEMBRANE_DOCTOR_STATUS_OK;
		c.detail = {{"reachable", true}, {"endpoint", "http://"
			+ cfg.listen_address + ":" + std::to_string(cfg.port)},
			{"loaded_model", status_json.value("loaded_model", json(nullptr))},
			{"model_state", status_json.value("model_state",
				std::string("unknown"))}};
		return (c);
	}
	c.detail = {{"reachable", false}, {"endpoint", "http://"
		+ cfg.listen_address + ":" + std::to_string(cfg.port)}};
	if (service_installed && service_active)
	{
		c.status = MEMBRANE_DOCTOR_STATUS_WARN;
		c.detail["message"] = "the service reports active, but the HTTP "
			"endpoint is not answering yet -- it may still be starting, "
			"or check `membrane service logs`";
	}
	else
	{
		c.status = MEMBRANE_DOCTOR_STATUS_OK;
		c.detail["message"] = "not reachable -- normal if `membrane "
			"serve`/the service is not currently running";
	}
	return (c);
}

std::string	membrane_doctor_collect(json *out_root)
{
	std::vector<s_doctor_check>	checks;
	json							models;

	checks.push_back(check_installation());
	checks.push_back(check_hardware());
	checks.push_back(check_registry(&models));
	checks.push_back(check_default_model(models));
	checks.push_back(check_config());
	bool	service_installed = false;
	bool	service_active = false;
	int		unused_port = 0;

	checks.push_back(check_service(&service_installed, &service_active,
			&unused_port));
	checks.push_back(check_http(service_installed, service_active));

	std::string	overall = MEMBRANE_DOCTOR_STATUS_OK;

	for (const auto &c : checks)
		if (status_worse(c.status, overall))
			overall = c.status;
	*out_root = json();
	(*out_root)["schema_version"] = 1;
	(*out_root)["membrane_version"] = MEMBRANE_VERSION;
	(*out_root)["overall"] = overall;
	(*out_root)["checks"] = json::array();
	for (const auto &c : checks)
		(*out_root)["checks"].push_back({{"name", c.name},
			{"status", c.status}, {"detail", c.detail}});
	return (overall);
}

int	membrane_doctor_cmd_dispatch(const std::vector<std::string> &args,
				bool want_json)
{
	(void)args;

	json		root;
	std::string	overall = membrane_doctor_collect(&root);

	if (want_json)
		printf("%s\n", root.dump().c_str());
	else
	{
		printf("MEMBRANE doctor -- overall: %s\n\n", overall.c_str());
		for (const auto &c : root["checks"])
		{
			printf("[%s] %s\n", c["status"].get<std::string>().c_str(),
				c["name"].get<std::string>().c_str());
			if (c["detail"].contains("message"))
				printf("    %s\n",
					c["detail"]["message"].get<std::string>().c_str());
		}
	}
	return (overall == MEMBRANE_DOCTOR_STATUS_ERROR
		? MEMBRANE_EXIT_RUNTIME_ERROR : MEMBRANE_EXIT_SUCCESS);
}
