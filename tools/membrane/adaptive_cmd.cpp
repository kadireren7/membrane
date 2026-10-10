#include "adaptive_cmd.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "adaptive.h"
#include "adaptive_controller.h"
#include "cli_shared.h"
#include "product_cli.h"
#include "server_config.h"

using json = nlohmann::json;

/* See adaptive_cmd.h's top comment. */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
		printf("%s\n", json{{"ok", false}, {"error", {{"code", code},
			{"message", message}}}}.dump().c_str());
	else
		fprintf(stderr, "membrane adaptive: %s\n", message.c_str());
}

static const char	*usage(void)
{
	return ("usage: membrane adaptive status | enable [--mode observe|"
		"recommend|apply] [--runtime ID] [--allow CLASSES] [--interval S] "
		"[--cooldown S] [--yes] | disable");
}

static bool	fetch_live(const membrane_server_config_t &cfg, json *out)
{
	httplib::Client	cli(cfg.listen_address, cfg.port);

	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(3, 0);
	auto	res = cli.Get("/membrane/v1/adaptive");

	if (!res || res->status != 200)
		return (false);
	json	j = json::parse(res->body, nullptr, false);

	if (j.is_discarded() || !j.is_object())
		return (false);
	*out = j;
	return (true);
}

static bool	load_cfg(membrane_server_config_t *cfg, std::string *path,
				std::string *err)
{
	membrane_server_config_error_t	e;

	*cfg = membrane_server_config_defaults();
	*path = membrane_server_config_resolve_path();
	if (path->empty())
	{
		*err = "no server config path (set HOME or "
			"MEMBRANE_SERVER_CONFIG_PATH)";
		return (false);
	}
	if (!membrane_server_config_load(*path, cfg, &e))
	{
		*err = "server config unreadable: " + e.message;
		return (false);
	}
	return (true);
}

static std::string	ago(int64_t then_ms)
{
	int64_t	now = (int64_t)std::chrono::duration_cast<
			std::chrono::milliseconds>(std::chrono::system_clock::now()
				.time_since_epoch()).count();
	int64_t	s = (now - then_ms) / 1000;

	if (s < 0)
		s = 0;
	if (s < 120)
		return (std::to_string(s) + "s ago");
	if (s < 7200)
		return (std::to_string(s / 60) + "m ago");
	return (std::to_string(s / 3600) + "h ago");
}

static int	cmd_status(const membrane_server_config_t &cfg, bool want_json)
{
	json	live;
	bool	reachable = fetch_live(cfg, &live);
	membrane_adaptive_policy_t	pol;
	std::string	cerr = membrane_adaptive_policy_from_config(cfg.adaptive, &pol);
	char	allow[128];
	json	j;

	membrane_adaptive_classes_format(pol.allowed_classes, allow, sizeof(allow));
	j["schema_version"] = MEMBRANE_ADAPTIVE_SCHEMA_VERSION;
	j["mode"] = membrane_adaptive_mode_name(pol.mode);
	j["enabled"] = pol.enabled != 0;
	j["effective_mode"] = membrane_adaptive_mode_name(
			membrane_adaptive_effective_mode(&pol));
	j["runtime"] = pol.runtime_scope;
	j["service_reachable"] = reachable;
	j["config_error"] = cerr.empty() ? json(nullptr) : json(cerr);
	j["policy"] = {{"interval_s", pol.interval_s},
		{"cooldown_s", pol.cooldown_s}, {"allow", allow},
		{"min_pressure", membrane_pressure_name(pol.min_pressure)},
		{"max_actions_per_window", pol.max_actions_per_window},
		{"auto_restore", pol.auto_restore != 0}};
	j["live"] = reachable ? live : json(nullptr);
	if (want_json)
	{
		printf("%s\n", j.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	const char	*mode = membrane_adaptive_mode_name(
			membrane_adaptive_effective_mode(&pol));

	printf("Adaptive runtime: %s\n", mode);
	if (!cerr.empty())
		printf("  ! The saved adaptive settings are invalid, so adaptive "
			"control is OFF: %s\n", cerr.c_str());
	if (membrane_adaptive_effective_mode(&pol) == MEMBRANE_ADAPTIVE_OFF)
	{
		printf("  Runtime:           %s\n", pol.runtime_scope);
		printf("  Adaptive control is off (the default). Nothing is "
			"observed or changed automatically.\n");
		printf("\nNext: membrane adaptive enable   (recommend-only; never "
			"changes anything)\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("  Runtime:           %s\n", pol.runtime_scope);
	if (!reachable)
	{
		printf("  Service:           not running (the policy is saved and "
			"takes effect when `membrane serve` runs)\n");
		printf("\nNext: membrane serve   (or: membrane service start)\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (live.contains("last_observation_unix_ms")
		&& live["last_observation_unix_ms"].is_number_integer())
		printf("  Last observation:  %s\n", ago(live[
			"last_observation_unix_ms"].get<int64_t>()).c_str());
	else
		printf("  Last observation:  none yet (first sample within %ds)\n",
			pol.interval_s);
	printf("  Pressure:          %s\n", live.value("pressure",
			std::string("unknown")).c_str());
	if (live.contains("last_action") && live["last_action"].is_object())
		printf("  Last action:       %s [%s]\n", live["last_action"].value(
				"step", std::string("?")).c_str(), live["last_action"].value(
				"status", std::string("?")).c_str());
	else
		printf("  Last action:       none\n");
	if (live.value("cooldown_remaining_s", 0) > 0 && !live.value("suppressed",
			false))
		printf("  Cooldown:          %d s remaining\n",
			live.value("cooldown_remaining_s", 0));
	else
		printf("  Cooldown:          none\n");
	printf("  Suppressed:        %s\n", live.value("suppressed", false)
		? ("yes (" + live.value("suppress_reason", std::string("?")) + ")")
			.c_str() : "no");
	if (live.value("pending_reload", false))
		printf("  Pending:           a change applies when the model is "
			"next loaded\n");
	printf("  Decision:          %s%s%s\n", live.value("decision",
			std::string("none")).c_str(), live["reason"].is_string()
			? " -- " : "", live["reason"].is_string() ? live["reason"]
			.get<std::string>().c_str() : "");
	if (live.contains("step") && live["step"].is_string())
		printf("  %s:  %s\n", pol.mode == MEMBRANE_ADAPTIVE_APPLY
			? "Step             " : "Suggested step   ",
			live["step"].get<std::string>().c_str());
	if (live.contains("reason_chain") && live["reason_chain"].is_array()
		&& !live["reason_chain"].empty())
	{
		printf("  Why:\n");
		for (const auto &c : live["reason_chain"])
			printf("    %-14s %s\n", (c.value("stage", std::string("")) + ":")
					.c_str(), c.value("text", std::string("")).c_str());
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_adaptive_cmd_dispatch_ex(const std::vector<std::string> &args,
		bool want_json, const membrane_adaptive_cmd_io_t &io)
{
	membrane_server_config_t	cfg;
	std::string					path;
	std::string					err;

	if (args.empty())
	{
		print_err(want_json, "CLI_ERROR", usage());
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!load_cfg(&cfg, &path, &err))
	{
		print_err(want_json, "CONFIG_ERROR", err);
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	const std::string	&verb = args[0];

	if (verb == "status")
	{
		if (args.size() != 1)
		{
			print_err(want_json, "CLI_ERROR", usage());
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		return (cmd_status(cfg, want_json));
	}
	if (verb == "disable")
	{
		if (args.size() != 1)
		{
			print_err(want_json, "CLI_ERROR", usage());
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		if (!cfg.adaptive.present)
			cfg.adaptive.present = true;
		cfg.adaptive.enabled = false;
		membrane_server_config_error_t	e;

		if (!membrane_server_config_save(path, cfg, &e))
		{
			print_err(want_json, "CONFIG_ERROR", e.message);
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
		if (want_json)
			printf("%s\n", json{{"ok", true}, {"enabled", false},
				{"mode", "off"}}.dump().c_str());
		else
			printf("Adaptive runtime: off\n  Nothing is observed or changed "
				"automatically. (Your other adaptive settings are kept.)\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (verb != "enable")
	{
		print_err(want_json, "CLI_ERROR", std::string("unknown subcommand '")
			+ verb + "' -- " + usage());
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	/* ---- enable --------------------------------------------------- */
	membrane_server_adaptive_t	a = cfg.adaptive;
	bool						yes = false;

	if (!a.present)
		a = membrane_server_adaptive_t();
	a.present = true;
	a.enabled = true;
	a.mode = "recommend";
	for (size_t i = 1; i < args.size(); ++i)
	{
		const std::string	&o = args[i];
		bool				has = i + 1 < args.size();

		if (o == "--yes" || o == "-y")
			yes = true;
		else if (o == "--mode" && has)
			a.mode = args[++i];
		else if (o == "--runtime" && has)
			a.runtime = args[++i];
		else if (o == "--interval" && has)
			a.interval_s = atoi(args[++i].c_str());
		else if (o == "--cooldown" && has)
			a.cooldown_s = atoi(args[++i].c_str());
		else if (o == "--max-actions" && has)
			a.max_actions_per_window = atoi(args[++i].c_str());
		else if (o == "--min-pressure" && has)
			a.min_pressure = args[++i];
		else if (o == "--auto-restore")
			a.auto_restore = true;
		else if (o == "--allow" && has)
		{
			std::string	csv = args[++i];

			a.allow.clear();
			for (size_t s = 0; s <= csv.size();)
			{
				size_t	e = csv.find(',', s);

				if (e == std::string::npos)
					e = csv.size();
				if (e > s)
					a.allow.push_back(csv.substr(s, e - s));
				s = e + 1;
			}
		}
		else
		{
			print_err(want_json, "CLI_ERROR", "unknown option '" + o + "' -- "
				+ usage());
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
	}
	membrane_adaptive_policy_t	pol;
	std::string					why = membrane_adaptive_policy_from_config(a,
			&pol);

	if (!why.empty())
	{
		print_err(want_json, "INVALID_POLICY", why);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (pol.runtime_scope != std::string(MEMBRANE_RUNTIME_ID_NATIVE)
		&& pol.runtime_scope != std::string(MEMBRANE_RUNTIME_ID_OLLAMA))
	{
		print_err(want_json, "INVALID_POLICY", "adaptive control supports "
			"membrane-native and ollama only");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (pol.mode == MEMBRANE_ADAPTIVE_APPLY)
	{
		char	cls[128];
		bool	interactive = io.interactive_known ? io.interactive
			: membrane_cli_is_interactive();

		membrane_adaptive_classes_format(pol.allowed_classes, cls, sizeof(cls));
		if (!yes)
		{
			if (!interactive || want_json)
			{
				print_err(want_json, "CONFIRMATION_REQUIRED", "apply mode lets "
					"MEMBRANE change runtime state by itself; it needs "
					"explicit consent: re-run with --yes, or run it "
					"interactively. Nothing was changed.");
				return (MEMBRANE_EXIT_CLI_ERROR);
			}
			printf("WARNING: apply mode lets MEMBRANE change runtime state "
				"on its own.\n  Runtime:  %s\n  May change: %s\n  At most "
				"%d change(s) per %d s, %d s apart; every change is "
				"validated, audited and rolled back on failure.\n  Turn it "
				"off any time: membrane adaptive disable\n",
				pol.runtime_scope, cls, pol.max_actions_per_window,
				pol.window_s, pol.cooldown_s);
			bool	ok = io.confirm ? io.confirm("Enable apply mode?")
				: membrane_cli_ask_yes_no("Enable apply mode?", false, false);

			if (!ok)
			{
				printf("Declined. Adaptive settings were not changed.\n");
				return (MEMBRANE_EXIT_SUCCESS);
			}
		}
	}
	cfg.adaptive = a;
	membrane_server_config_error_t	e;

	if (!membrane_server_config_save(path, cfg, &e))
	{
		print_err(want_json, "CONFIG_ERROR", e.message);
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	if (want_json)
		printf("%s\n", json{{"ok", true}, {"enabled", true},
			{"mode", membrane_adaptive_mode_name(pol.mode)},
			{"runtime", pol.runtime_scope}}.dump().c_str());
	else
	{
		printf("Adaptive runtime: %s\n  Runtime: %s\n", membrane_adaptive_mode_name(
				pol.mode), pol.runtime_scope);
		if (pol.mode == MEMBRANE_ADAPTIVE_APPLY)
			printf("  MEMBRANE may now change runtime state within the "
				"allowed classes.\n");
		else if (pol.mode == MEMBRANE_ADAPTIVE_RECOMMEND)
			printf("  It will suggest steps; it never changes anything.\n");
		else
			printf("  It will only record observations.\n");
		printf("\nNext: membrane adaptive status   (a running service picks "
			"this up within %d s)\n", pol.interval_s);
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_adaptive_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json)
{
	return (membrane_adaptive_cmd_dispatch_ex(args, want_json,
			membrane_adaptive_cmd_io_t()));
}
