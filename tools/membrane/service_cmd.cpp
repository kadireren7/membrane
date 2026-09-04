#include "service_cmd.h"
#include "systemd_unit.h"
#include "server_config.h"
#include "subprocess.h"
#include "status_client.h"
#include "fs_util.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <limits.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "product_cli.h"

using json = nlohmann::json;

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", code}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane service: %s\n", message.c_str());
}

/* Section 6: resolves the CURRENTLY RUNNING membrane binary's own real
 * path via /proc/self/exe (Linux-only, matching this project's existing
 * Linux-only scope) -- never a caller-assembled guess. A build-tree
 * path (this repo's own out-of-tree build directories, or /tmp) is
 * usable but fragile (a later `rm -rf`/rebuild breaks the installed
 * unit) -- callers get a warning, never a silent trap, but installation
 * still proceeds (Section 6 explicitly allows either choice; blocking
 * outright would make this command untestable from a dev build, which
 * this project's own dev-local smoke testing needs). */
static bool	resolve_exec_path(const std::string &override_path,
				std::string *out_path, bool *out_looks_like_build_tree)
{
	if (!override_path.empty())
	{
		*out_path = override_path;
		*out_looks_like_build_tree = false;
		return (true);
	}
	char	buf[PATH_MAX];
	ssize_t	n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

	if (n < 0)
		return (false);
	buf[n] = '\0';
	*out_path = buf;
	*out_looks_like_build_tree = (out_path->find("/build") != std::string::npos
		|| out_path->find("/tmp/") == 0);
	return (true);
}

static int	cmd_install(const std::vector<std::string> &args, bool want_json)
{
	std::string	exec_path_override;
	bool		force = false;

	for (size_t i = 0; i < args.size(); ++i)
	{
		if (args[i] == "--exec-path" && i + 1 < args.size())
			exec_path_override = args[++i];
		else if (args[i] == "--force")
			force = true;
		else
		{
			print_err(want_json, "CLI_ERROR", std::string("unknown option '")
				+ args[i] + "'");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
	}
	std::string	exec_path;
	bool		looks_like_build_tree;

	if (!resolve_exec_path(exec_path_override, &exec_path,
			&looks_like_build_tree))
	{
		print_err(want_json, "IO_ERROR", "could not resolve the running "
			"membrane binary's own path (/proc/self/exe) -- pass "
			"--exec-path PATH explicitly");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (looks_like_build_tree && !want_json)
		fprintf(stderr, "membrane service install: WARNING -- '%s' looks "
			"like a development build path, not an installed location; "
			"the generated service will break if this path is later "
			"removed or rebuilt. Pass --exec-path once installed, or "
			"reinstall after packaging.\n", exec_path.c_str());
	std::string	unit_path = membrane_unit_file_path();

	if (unit_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set -- cannot locate the systemd --user unit "
			"directory");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	FILE	*existing = fopen(unit_path.c_str(), "rb");

	if (existing != NULL)
	{
		std::string	content;
		char		buf[4096];
		size_t		n;

		while ((n = fread(buf, 1, sizeof(buf), existing)) > 0)
			content.append(buf, n);
		fclose(existing);
		if (!membrane_unit_is_membrane_managed(content) && !force)
		{
			print_err(want_json, "UNIT_EXISTS", std::string("'") + unit_path
				+ "' already exists and was not created by `membrane "
				"service install` -- refusing to overwrite it. Remove it "
				"yourself first, or pass --force if you are sure.");
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
	}
	membrane_unit_options_t	unit_opts;

	unit_opts.exec_path = exec_path;
	std::string	unit_content = membrane_generate_unit_file(unit_opts);
	membrane_fs_error_t	fs_err;

	if (!membrane_atomic_write_file(unit_path, unit_content, &fs_err))
	{
		print_err(want_json, fs_err.code, fs_err.message);
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	/* Section 8: seed a default config if none exists yet -- `membrane
	 * serve` already tolerates a missing config (defaults apply), this
	 * just makes `membrane service status`/a user editing the file
	 * afterward see real values immediately. Never overwrites an
	 * existing config. */
	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;
	std::string						config_path
			= membrane_server_config_resolve_path();

	if (!config_path.empty()
		&& !membrane_server_config_load(config_path, &cfg, &cfg_err))
	{
		/* A malformed EXISTING config is a real problem -- surfaced, but
		 * does not block install (the unit is already written and
		 * valid; the user can fix the config separately). */
		if (!want_json)
			fprintf(stderr, "membrane service install: WARNING -- "
				"existing server config could not be read: %s\n",
				cfg_err.message.c_str());
	}
	else if (!config_path.empty())
	{
		FILE	*probe = fopen(config_path.c_str(), "rb");

		if (probe == NULL)
			membrane_server_config_save(config_path,
				membrane_server_config_defaults(), &cfg_err);
		else
			fclose(probe);
	}
	membrane_subprocess_result_t	reload_result;

	membrane_run_subprocess({"systemctl", "--user", "daemon-reload"},
		&reload_result);
	if (want_json)
	{
		json	j;

		j["ok"] = true;
		j["unit_path"] = unit_path;
		j["exec_path"] = exec_path;
		printf("%s\n", j.dump().c_str());
	}
	else
	{
		printf("Installed %s\n", unit_path.c_str());
		printf("  exec: %s\n", exec_path.c_str());
		printf("Next: membrane service start\n");
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_uninstall(bool want_json)
{
	std::string	unit_path = membrane_unit_file_path();

	if (unit_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	FILE	*existing = fopen(unit_path.c_str(), "rb");

	if (existing == NULL)
	{
		if (want_json)
			printf("{\"ok\":true,\"note\":\"nothing installed\"}\n");
		else
			printf("Nothing installed.\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	std::string	content;
	char		buf[4096];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), existing)) > 0)
		content.append(buf, n);
	fclose(existing);
	if (!membrane_unit_is_membrane_managed(content))
	{
		print_err(want_json, "NOT_MANAGED", std::string("'") + unit_path
			+ "' was not created by `membrane service install` -- "
			"refusing to remove it");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	membrane_subprocess_result_t	stop_result;

	membrane_run_subprocess({"systemctl", "--user", "stop",
		MEMBRANE_UNIT_NAME}, &stop_result);	/* best-effort */
	unlink(unit_path.c_str());
	membrane_subprocess_result_t	reload_result;

	membrane_run_subprocess({"systemctl", "--user", "daemon-reload"},
		&reload_result);
	if (want_json)
		printf("{\"ok\":true}\n");
	else
		printf("Uninstalled %s\n", unit_path.c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_systemctl_verb(const std::string &verb, bool want_json)
{
	membrane_subprocess_result_t	result;

	if (!membrane_run_subprocess({"systemctl", "--user", verb,
			MEMBRANE_UNIT_NAME}, &result))
	{
		print_err(want_json, "IO_ERROR", "could not run systemctl -- is "
			"it installed?");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (result.exit_code != 0)
	{
		print_err(want_json, "SYSTEMCTL_FAILED", "systemctl --user " + verb
			+ " " MEMBRANE_UNIT_NAME " failed: "
			+ (result.stderr_output.empty() ? "(no output)"
				: result.stderr_output));
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (want_json)
		printf("{\"ok\":true}\n");
	else
		printf("%s\n", verb.c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_status(bool want_json)
{
	membrane_subprocess_result_t	show_result;
	bool	have_systemctl = membrane_run_subprocess({"systemctl", "--user",
			"show", MEMBRANE_UNIT_NAME, "--property=ActiveState,SubState,"
			"MainPID,LoadState"}, &show_result);
	std::string	active_state = "unknown";
	std::string	sub_state = "unknown";
	std::string	load_state = "unknown";
	std::string	main_pid = "0";

	if (have_systemctl && show_result.exit_code == 0)
	{
		std::istringstream	iss(show_result.stdout_output);
		std::string			line;

		while (std::getline(iss, line))
		{
			size_t	eq = line.find('=');

			if (eq == std::string::npos)
				continue ;
			std::string	key = line.substr(0, eq);
			std::string	val = line.substr(eq + 1);

			if (key == "ActiveState")
				active_state = val;
			else if (key == "SubState")
				sub_state = val;
			else if (key == "LoadState")
				load_state = val;
			else if (key == "MainPID")
				main_pid = val;
		}
	}
	std::string	config_path = membrane_server_config_resolve_path();
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	if (!config_path.empty())
		membrane_server_config_load(config_path, &cfg, &cfg_err);
	json	http_status;
	bool	http_ok = membrane_fetch_server_status(cfg.listen_address,
			cfg.port, &http_status);

	if (want_json)
	{
		json	j;

		j["installed"] = (load_state == "loaded");
		j["active_state"] = active_state;
		j["sub_state"] = sub_state;
		j["main_pid"] = main_pid;
		j["http_reachable"] = http_ok;
		if (http_ok)
			j["http_status"] = http_status;
		printf("%s\n", j.dump().c_str());
	}
	else
	{
		printf("Service:\n");
		printf("  installed: %s\n",
			load_state == "loaded" ? "yes" : "no");
		printf("  state: %s (%s)\n", active_state.c_str(),
			sub_state.c_str());
		if (main_pid != "0")
			printf("  pid: %s\n", main_pid.c_str());
		if (http_ok)
			membrane_print_server_status_human(http_status);
		else
			printf("  http: not reachable at %s:%d\n",
				cfg.listen_address.c_str(), cfg.port);
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	cmd_logs(const std::vector<std::string> &args, bool want_json)
{
	int	lines = 50;

	for (size_t i = 0; i < args.size(); ++i)
	{
		if (args[i] == "-n" && i + 1 < args.size())
			lines = atoi(args[++i].c_str());
		else
		{
			print_err(want_json, "CLI_ERROR", std::string("unknown option '")
				+ args[i] + "'");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
	}
	if (lines < 1)
		lines = 1;
	if (lines > 10000)
		lines = 10000;
	membrane_subprocess_result_t	result;
	bool	ran = membrane_run_subprocess({"journalctl", "--user", "-u",
			MEMBRANE_UNIT_NAME, "-n", std::to_string(lines), "--no-pager"},
			&result, 15);

	if (!ran)
	{
		print_err(want_json, "IO_ERROR", "could not run journalctl -- is "
			"it available on this system?");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	printf("%s", result.stdout_output.c_str());
	if (result.exit_code != 0 && !result.stderr_output.empty())
		fprintf(stderr, "%s", result.stderr_output.c_str());
	return (result.exit_code == 0 ? MEMBRANE_EXIT_SUCCESS
		: MEMBRANE_EXIT_RUNTIME_ERROR);
}

int	membrane_service_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	if (args.empty())
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane service "
			"install|uninstall|start|stop|restart|status|logs ...");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::vector<std::string>	rest(args.begin() + 1, args.end());

	if (args[0] == "install")
		return (cmd_install(rest, want_json));
	if (args[0] == "uninstall")
		return (cmd_uninstall(want_json));
	if (args[0] == "start" || args[0] == "stop" || args[0] == "restart")
		return (cmd_systemctl_verb(args[0], want_json));
	if (args[0] == "status")
		return (cmd_status(want_json));
	if (args[0] == "logs")
		return (cmd_logs(rest, want_json));
	print_err(want_json, "CLI_ERROR", std::string("unknown subcommand '")
		+ args[0] + "' -- expected install|uninstall|start|stop|restart|"
		"status|logs");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
