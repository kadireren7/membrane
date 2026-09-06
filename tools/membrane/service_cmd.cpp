#include "service_cmd.h"
#include "systemd_unit.h"
#include "launchd_unit.h"
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

#ifdef __APPLE__
# include <mach-o/dyld.h>
#endif

#include <nlohmann/json.hpp>

#include "product_cli.h"

using json = nlohmann::json;

/*
 * Mega Phase D, PR D4: this file now dispatches between two real
 * per-user service-manager backends -- systemd --user (Linux, Mega
 * Phase B, PR B1) and launchd (macOS/Darwin, this PR) -- via #ifdef
 * __APPLE__ at each real point of platform divergence: unit/plist
 * generation+path resolution (systemd_unit.h/launchd_unit.h, already
 * fully separated), the real service-manager binary invoked, and how
 * "logs" is obtained (journalctl has no launchd equivalent -- launchd
 * redirects the job's own stdout/stderr straight to a real file via
 * StandardOutPath/StandardErrorPath, so "logs" tails that file
 * directly on macOS). server.h's own foreground membrane_server_run()
 * (`membrane serve`) is completely unchanged and identical on both
 * platforms -- only how it gets supervised as a background service
 * differs, matching Section 4's "never duplicate server/runtime logic".
 *
 * Known, disclosed semantic differences (docs/macos-metal.md has the
 * full narrative) -- launchd's command set has no exact 1:1 mapping to
 * systemctl's start/stop/restart/status:
 *   - "start": `launchctl bootstrap` (loads+starts; fails if already
 *     loaded) falling back to `launchctl kickstart -k` (force-restart;
 *     requires already loaded) -- together they make `start` idempotent
 *     whether or not the job was already loaded, without a special
 *     "is it loaded" pre-check of their own.
 *   - "stop": `launchctl bootout` fully unloads the job (closer to
 *     systemd's stop+disable combined) -- there is no launchd verb that
 *     stops a running job while leaving it loaded/re-startable via a
 *     bare "start" the way `systemctl --user stop` does; a subsequent
 *     `start` on macOS re-bootstraps it fresh, which is fine (idempotent)
 *     but is not byte-for-byte the same lifecycle as the Linux path.
 *   - "restart": `launchctl kickstart -k` (kill and relaunch in place);
 *     requires the job already be loaded (run `start` first if not).
 *   - "status": `launchctl print` has no machine-stable documented
 *     output format (unlike `systemctl show --property=...`) -- parsed
 *     defensively for "state = " / "pid = " lines, the same real,
 *     widely-relied-upon convention every other launchd tooling uses,
 *     but disclosed here as inherently less stable than the systemd
 *     path.
 */

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

#ifdef __APPLE__
/* macOS's own real per-session domain target for `launchctl bootstrap/
 * bootout/kickstart/print` -- gui/<uid> is the correct target for a
 * per-user LaunchAgent running in a real login GUI/session (as opposed
 * to system/ for root-owned LaunchDaemons, which this project never
 * uses -- Section 5: ordinary user permissions only, no root). */
static std::string	launchd_domain_target(void)
{
	return ("gui/" + std::to_string(getuid()));
}

static std::string	launchd_service_target(void)
{
	return (launchd_domain_target() + "/" MEMBRANE_LAUNCHD_LABEL);
}
#endif

/* Section 6: resolves the CURRENTLY RUNNING membrane binary's own real
 * path -- /proc/self/exe on Linux, _NSGetExecutablePath() on macOS
 * (there is no /proc filesystem on Darwin) -- never a caller-assembled
 * guess. A build-tree path (this repo's own out-of-tree build
 * directories, or /tmp) is usable but fragile (a later `rm -rf`/rebuild
 * breaks the installed unit/plist) -- callers get a warning, never a
 * silent trap, but installation still proceeds (Section 6 explicitly
 * allows either choice; blocking outright would make this command
 * untestable from a dev build, which this project's own dev-local
 * smoke testing needs). */
static bool	resolve_exec_path(const std::string &override_path,
				std::string *out_path, bool *out_looks_like_build_tree)
{
	if (!override_path.empty())
	{
		*out_path = override_path;
		*out_looks_like_build_tree = false;
		return (true);
	}
#ifdef __APPLE__
	char		buf[PATH_MAX];
	uint32_t	size = sizeof(buf);

	if (_NSGetExecutablePath(buf, &size) != 0)
		return (false);
	char	resolved[PATH_MAX];

	if (realpath(buf, resolved) == NULL)
		return (false);
	*out_path = resolved;
#else
	char	buf[PATH_MAX];
	ssize_t	n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

	if (n < 0)
		return (false);
	buf[n] = '\0';
	*out_path = buf;
#endif
	*out_looks_like_build_tree = (out_path->find("/build") != std::string::npos
		|| out_path->find("/tmp/") == 0);
	return (true);
}

/* One real, sensible default log location per platform -- launchd has
 * no journalctl equivalent, so this is where `membrane service logs`
 * reads from on macOS (see cmd_logs below). Not XDG (macOS has no such
 * convention); $HOME/Library/Logs/<app> is the real, standard macOS
 * per-user application log location. */
#ifdef __APPLE__
static std::string	default_log_path(void)
{
	const char	*home = getenv("HOME");

	if (home == NULL || home[0] == '\0')
		return ("/tmp/membrane-service.log");
	return (std::string(home) + "/Library/Logs/membrane/membrane.log");
}
#endif

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
			"membrane binary's own path -- pass --exec-path PATH "
			"explicitly");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (looks_like_build_tree && !want_json)
		fprintf(stderr, "membrane service install: WARNING -- '%s' looks "
			"like a development build path, not an installed location; "
			"the generated service will break if this path is later "
			"removed or rebuilt. Pass --exec-path once installed, or "
			"reinstall after packaging.\n", exec_path.c_str());
#ifdef __APPLE__
	std::string	unit_path = membrane_launchd_plist_path();
#else
	std::string	unit_path = membrane_unit_file_path();
#endif

	if (unit_path.empty())
	{
#ifdef __APPLE__
		print_err(want_json, "IO_ERROR", "HOME is not set -- cannot locate "
			"the LaunchAgents directory");
#else
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set -- cannot locate the systemd --user unit "
			"directory");
#endif
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
#ifdef __APPLE__
		bool	is_managed = membrane_launchd_plist_is_membrane_managed(
				content);
#else
		bool	is_managed = membrane_unit_is_membrane_managed(content);
#endif
		if (!is_managed && !force)
		{
			print_err(want_json, "UNIT_EXISTS", std::string("'") + unit_path
				+ "' already exists and was not created by `membrane "
				"service install` -- refusing to overwrite it. Remove it "
				"yourself first, or pass --force if you are sure.");
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
	}
#ifdef __APPLE__
	membrane_launchd_options_t	unit_opts;

	unit_opts.exec_path = exec_path;
	unit_opts.log_path = default_log_path();
	std::string	unit_content = membrane_generate_launchd_plist(unit_opts);
#else
	membrane_unit_options_t	unit_opts;

	unit_opts.exec_path = exec_path;
	std::string	unit_content = membrane_generate_unit_file(unit_opts);
#endif
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
#ifndef __APPLE__
	/* launchd has no analogous "reload the unit registry" step --
	 * unlike systemd, it discovers a plist's content fresh at the point
	 * it is actually bootstrapped (see cmd_start_stop_restart below), so
	 * there is nothing to reload here on macOS. */
	membrane_subprocess_result_t	reload_result;

	membrane_run_subprocess({"systemctl", "--user", "daemon-reload"},
		&reload_result);
#endif
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
#ifdef __APPLE__
	std::string	unit_path = membrane_launchd_plist_path();
#else
	std::string	unit_path = membrane_unit_file_path();
#endif

	if (unit_path.empty())
	{
#ifdef __APPLE__
		print_err(want_json, "IO_ERROR", "HOME is not set");
#else
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set");
#endif
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
#ifdef __APPLE__
	bool	is_managed = membrane_launchd_plist_is_membrane_managed(content);
#else
	bool	is_managed = membrane_unit_is_membrane_managed(content);
#endif
	if (!is_managed)
	{
		print_err(want_json, "NOT_MANAGED", std::string("'") + unit_path
			+ "' was not created by `membrane service install` -- "
			"refusing to remove it");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	membrane_subprocess_result_t	stop_result;

#ifdef __APPLE__
	/* best-effort: bootout fails harmlessly if it was never bootstrapped */
	membrane_run_subprocess({"launchctl", "bootout",
		launchd_service_target()}, &stop_result);
#else
	membrane_run_subprocess({"systemctl", "--user", "stop",
		MEMBRANE_UNIT_NAME}, &stop_result);	/* best-effort */
#endif
	unlink(unit_path.c_str());
#ifndef __APPLE__
	membrane_subprocess_result_t	reload_result;

	membrane_run_subprocess({"systemctl", "--user", "daemon-reload"},
		&reload_result);
#endif
	if (want_json)
		printf("{\"ok\":true}\n");
	else
		printf("Uninstalled %s\n", unit_path.c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}

#ifdef __APPLE__
static int	cmd_launchd_verb(const std::string &verb, bool want_json)
{
	membrane_subprocess_result_t	result;
	std::string	plist_path = membrane_launchd_plist_path();
	bool		ok = false;
	std::string	failure_detail;

	if (verb == "start")
	{
		/* bootstrap loads+starts a not-yet-loaded job; if it's already
		 * loaded, bootstrap fails (real launchctl behavior) and
		 * kickstart -k (force-restart) is the real fallback -- together
		 * these make `start` idempotent either way. */
		membrane_run_subprocess({"launchctl", "bootstrap",
			launchd_domain_target(), plist_path}, &result);
		if (result.exit_code == 0)
			ok = true;
		else
		{
			membrane_run_subprocess({"launchctl", "kickstart", "-k",
				launchd_service_target()}, &result);
			ok = (result.exit_code == 0);
		}
	}
	else if (verb == "stop")
	{
		membrane_run_subprocess({"launchctl", "bootout",
			launchd_service_target()}, &result);
		ok = (result.exit_code == 0);
	}
	else	/* restart */
	{
		membrane_run_subprocess({"launchctl", "kickstart", "-k",
			launchd_service_target()}, &result);
		ok = (result.exit_code == 0);
	}
	failure_detail = result.stderr_output.empty() ? "(no output)"
		: result.stderr_output;
	if (!ok)
	{
		print_err(want_json, "LAUNCHCTL_FAILED", "launchctl " + verb
			+ " failed: " + failure_detail);
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (want_json)
		printf("{\"ok\":true}\n");
	else
		printf("%s\n", verb.c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}
#else
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
#endif

static int	cmd_status(bool want_json)
{
	std::string	active_state = "unknown";
	std::string	sub_state = "unknown";
	std::string	load_state = "unknown";
	std::string	main_pid = "0";

#ifdef __APPLE__
	membrane_subprocess_result_t	print_result;
	bool	have_launchctl = membrane_run_subprocess({"launchctl", "print",
			launchd_service_target()}, &print_result);

	/* `launchctl print`'s output format has no documented stability
	 * guarantee (unlike `systemctl show --property=...`) -- parsed
	 * defensively for the real "state = " / "pid = " lines it does
	 * carry in practice, same real convention every other launchd
	 * tooling relies on. A non-zero exit means the job isn't bootstrapped
	 * at all (the real, expected result right after `service uninstall`,
	 * or before the first `service start`). */
	if (have_launchctl && print_result.exit_code == 0)
	{
		load_state = "loaded";
		std::istringstream	iss(print_result.stdout_output);
		std::string			line;

		while (std::getline(iss, line))
		{
			size_t	eq = line.find('=');

			if (eq == std::string::npos)
				continue ;
			std::string	key = line.substr(0, eq);
			std::string	val = line.substr(eq + 1);

			/* launchctl print pads keys with leading/trailing whitespace
			 * (e.g. "\tstate = running") -- trim before comparing. */
			size_t	key_start = key.find_first_not_of(" \t");
			size_t	key_end = key.find_last_not_of(" \t");
			if (key_start == std::string::npos)
				continue ;
			key = key.substr(key_start, key_end - key_start + 1);
			size_t	val_start = val.find_first_not_of(" \t");
			size_t	val_end = val.find_last_not_of(" \t");
			if (val_start != std::string::npos)
				val = val.substr(val_start, val_end - val_start + 1);
			else
				val = "";
			if (key == "state")
				active_state = val;
			else if (key == "pid")
				main_pid = val;
		}
	}
	else
		load_state = "not-found";
#else
	membrane_subprocess_result_t	show_result;
	bool	have_systemctl = membrane_run_subprocess({"systemctl", "--user",
			"show", MEMBRANE_UNIT_NAME, "--property=ActiveState,SubState,"
			"MainPID,LoadState"}, &show_result);

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
#endif
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
#ifdef __APPLE__
	/* launchd has no journalctl equivalent -- the job's own stdout/
	 * stderr are redirected straight to a real file (StandardOutPath/
	 * StandardErrorPath in the installed plist, see default_log_path()
	 * above), so "logs" is a real `tail -n` of that file. */
	membrane_subprocess_result_t	result;
	bool	ran = membrane_run_subprocess({"tail", "-n",
			std::to_string(lines), default_log_path()}, &result, 15);

	if (!ran)
	{
		print_err(want_json, "IO_ERROR", "could not run tail -- is it "
			"available on this system?");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	printf("%s", result.stdout_output.c_str());
	if (result.exit_code != 0 && !result.stderr_output.empty())
		fprintf(stderr, "%s", result.stderr_output.c_str());
	return (result.exit_code == 0 ? MEMBRANE_EXIT_SUCCESS
		: MEMBRANE_EXIT_RUNTIME_ERROR);
#else
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
#endif
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
#ifdef __APPLE__
		return (cmd_launchd_verb(args[0], want_json));
#else
		return (cmd_systemctl_verb(args[0], want_json));
#endif
	if (args[0] == "status")
		return (cmd_status(want_json));
	if (args[0] == "logs")
		return (cmd_logs(rest, want_json));
	print_err(want_json, "CLI_ERROR", std::string("unknown subcommand '")
		+ args[0] + "' -- expected install|uninstall|start|stop|restart|"
		"status|logs");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
