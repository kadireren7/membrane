#include "setup_cmd.h"

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include "membrane/posix_compat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "registry_core.h"
#include "server_config.h"
#include "model_cmd.h"
#include "service_cmd.h"
#include "doctor_cmd.h"
#include "product_cli.h"
#include "cli_shared.h"

using json = nlohmann::json;

/*
 * See setup_cmd.h's own top comment for the full contract.
 */

struct s_setup_opts
{
	std::string	model_name;
	bool		want_model_name = false;
	std::string	model_path;
	bool		want_model_path = false;
	bool		assume_yes = false;
	bool		no_service = false;
};

static bool	parse_setup_args(const std::vector<std::string> &args,
				s_setup_opts *o, std::string *err)
{
	for (size_t i = 0; i < args.size(); ++i)
	{
		if (args[i] == "--model-name" && i + 1 < args.size())
		{
			o->model_name = args[++i];
			o->want_model_name = true;
		}
		else if (args[i] == "--model" && i + 1 < args.size())
		{
			o->model_path = args[++i];
			o->want_model_path = true;
		}
		else if (args[i] == "--yes" || args[i] == "-y")
			o->assume_yes = true;
		else if (args[i] == "--no-service")
			o->no_service = true;
		else
		{
			*err = "unknown option '" + args[i] + "'";
			return (false);
		}
	}
	return (true);
}

static std::string	derive_model_name(const std::string &path)
{
	size_t	slash = path.find_last_of('/');
	std::string	base = (slash == std::string::npos) ? path
			: path.substr(slash + 1);
	size_t	dot = base.find_last_of('.');

	if (dot != std::string::npos)
		base = base.substr(0, dot);
	return (base.empty() ? "model" : base);
}

static void	print_step(int n, int total, const std::string &title)
{
	printf("\n== Step %d/%d: %s ==\n", n, total, title.c_str());
}

/* A small, direct GET -- same minimal httplib::Client pattern status_
 * client.cpp already established (0.5s connect / 2s read timeout),
 * reused here rather than growing status_client.h's own contract for
 * two one-off checks (Section 5's own /health, /v1/models verification
 * steps -- distinct from /v1/status, which status_client.h already
 * covers). */
static bool	verify_get(const std::string &bind, int port,
				const std::string &path)
{
	httplib::Client	cli(bind, port);

	cli.set_connection_timeout(0, 500000);
	cli.set_read_timeout(2, 0);
	auto	res = cli.Get(path);

	return (res != nullptr && res->status == 200);
}

/* Real bug found and fixed while testing this exact step: `membrane
 * service start` returns as soon as systemctl reports the unit
 * started, which is not a guarantee the listener has actually bound
 * the port yet (the same real race server.cpp's own svr.wait_until_
 * ready() already had to account for internally) -- an immediate,
 * single GET /health right after start can genuinely see "connection
 * refused" for a real, still-healthy service that just needed another
 * few hundred milliseconds. Bounded retry (up to ~3s, same order of
 * magnitude as test_server.cpp's own readiness poll), never an
 * unbounded wait. */
static bool	verify_get_with_retry(const std::string &bind, int port,
				const std::string &path)
{
	for (int attempt = 0; attempt < 15; ++attempt)
	{
		if (verify_get(bind, port, path))
			return (true);
		struct timespec	ts = {0, 200000000L};

		nanosleep(&ts, NULL);
	}
	return (false);
}

/* Section 13's own "JSON diagnostic if implemented" ask, taken
 * seriously: when want_json is true, EVERY prose narration line below
 * is suppressed (never mixed in with machine-readable output) and the
 * underlying reused commands (membrane_model_cmd_dispatch()/
 * membrane_service_cmd_dispatch()) are always called with want_json
 * false, so they never print their own JSON lines mid-run either --
 * control flow still relies purely on their real return codes, and
 * `summary` (accumulated as the run progresses, printed exactly once
 * at the very end) is the ONE JSON document a machine caller gets,
 * never a mix of narrative headers and scattered per-step JSON blobs.
 * (membrane_cli_narrate()/stdout_silencer_t/membrane_cli_dispatch_
 * silently_if_json() now live in cli_shared.h -- Mega Phase D, PR D6 --
 * reused as-is by use_cmd.cpp too.) */
int	membrane_setup_cmd_dispatch(const std::vector<std::string> &args,
				bool want_json)
{
	s_setup_opts	o;
	std::string		parse_err;

	if (!parse_setup_args(args, &o, &parse_err))
	{
		fprintf(stderr, "membrane setup: %s\n", parse_err.c_str());
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	const int	total_steps = 6;
	int			step = 0;
	json		summary;

	summary["membrane_version"] = MEMBRANE_VERSION;

	/* Step 1: version + real hardware/state summary -- Section 5, items
	 * 2-6. Reuses membrane_doctor_collect() (doctor_cmd.h) wholesale --
	 * never a second hardware/registry/service detection. */
	if (!want_json)
		print_step(++step, total_steps, "checking installation and host");
	membrane_cli_narrate(want_json, "MEMBRANE %s\n", MEMBRANE_VERSION);

	json		doctor_root;

	membrane_doctor_collect(&doctor_root);
	summary["doctor"] = doctor_root;

	for (const auto &c : doctor_root["checks"])
	{
		if (c["name"] == "hardware" && c["status"] == "OK")
		{
			const json	&d = c["detail"];

			membrane_cli_narrate(want_json, "CPU: %s\n",
				d.value("cpu", std::string("unknown")).c_str());
			membrane_cli_narrate(want_json, "GPU backend: %s (%d device(s))\n",
				d.value("gpu_backend_available", false) ? "available"
					: "none",
				d.value("gpu_devices_found", 0));
			uint64_t	total_b = d.value("host_total_bytes", 0ULL);

			if (total_b > 0)
				membrane_cli_narrate(want_json, "Host RAM: %.1f GiB total\n",
					(double)total_b / (1024.0 * 1024.0 * 1024.0));
		}
		if (c["name"] == "registry")
			membrane_cli_narrate(want_json, "Model registry: %d model(s) currently "
				"registered\n", (int)c["detail"].value("count", 0));
	}

	/* Step 2: resolve/register a model -- Section 5, items 7-10.
	 * Idempotent (Section 7): a name already pointing at the SAME
	 * canonical path is reported as already-done, never a raw
	 * DUPLICATE_NAME error; a name pointing somewhere ELSE is a real,
	 * clearly-reported conflict, never silently overwritten (Section
	 * 8's own "never destructively undo existing configuration" spirit
	 * extends to never silently clobbering a differently-pointed
	 * entry either). */
	if (!want_json)
		print_step(++step, total_steps, "model registration");

	std::string	model_path = o.model_path;

	if (!o.want_model_path)
		model_path = membrane_cli_ask_line("Path to a local GGUF model (leave empty "
			"to skip model registration): ");
	summary["model"] = json();
	if (model_path.empty())
	{
		membrane_cli_narrate(want_json, "No model path given -- skipping model "
			"registration (you can run `membrane model add NAME PATH` "
			"any time).\n");
		summary["model"]["registered"] = false;
	}
	else
	{
		char	resolved[PATH_MAX];

		if (realpath(model_path.c_str(), resolved) == NULL)
		{
			fprintf(stderr, "membrane setup: '%s' does not exist or is "
				"not accessible: %s\n", model_path.c_str(),
				strerror(errno));
			fprintf(stderr, "membrane setup: stopping here -- nothing "
				"else has been changed yet. Fix the path and re-run "
				"`membrane setup`.\n");
			return (MEMBRANE_EXIT_MODEL_ERROR);
		}
		std::string	canonical(resolved);
		std::string	model_name = o.want_model_name ? o.model_name
				: derive_model_name(canonical);
		std::string				registry_path
				= membrane_registry_resolve_path();
		membrane_registry_t		reg;
		membrane_registry_error_t	reg_err;
		bool	already_correct = false;

		if (!registry_path.empty()
			&& membrane_registry_load(registry_path, &reg, &reg_err))
		{
			const membrane_registry_entry_t	*existing
					= membrane_registry_find(reg, model_name);

			if (existing != NULL)
			{
				if (existing->path == canonical)
				{
					already_correct = true;
					membrane_cli_narrate(want_json, "'%s' is already registered "
						"pointing at this exact file -- nothing to "
						"do.\n", model_name.c_str());
				}
				else
				{
					fprintf(stderr, "membrane setup: '%s' is already "
						"registered, but pointing at a DIFFERENT file "
						"('%s') -- refusing to silently overwrite it. "
						"Choose a different --model-name, or run "
						"`membrane model remove %s` first.\n",
						model_name.c_str(), existing->path.c_str(),
						model_name.c_str());
					return (MEMBRANE_EXIT_CLI_ERROR);
				}
			}
		}
		if (!already_correct)
		{
			int	rc = membrane_cli_dispatch_silently_if_json(want_json,
					{"add", model_name, canonical},
					membrane_model_cmd_dispatch);

			if (rc != MEMBRANE_EXIT_SUCCESS)
			{
				fprintf(stderr, "membrane setup: registering '%s' "
					"failed -- stopping here, nothing else has been "
					"changed. Fix the issue above and re-run `membrane "
					"setup`, or register it manually with `membrane "
					"model add`.\n", model_name.c_str());
				return (rc);
			}
		}
		summary["model"]["registered"] = true;
		summary["model"]["name"] = model_name;
		summary["model"]["path"] = canonical;
		summary["model"]["already_registered"] = already_correct;
		/* Section 5, item 10: "optionally make it default" -- never
		 * silently overriding an EXISTING different default someone
		 * already chose deliberately. */
		std::string						config_path
				= membrane_server_config_resolve_path();
		membrane_server_config_t		cfg
				= membrane_server_config_defaults();
		membrane_server_config_error_t	cfg_err;

		if (!config_path.empty())
			membrane_server_config_load(config_path, &cfg, &cfg_err);
		if (cfg.default_model == model_name)
			membrane_cli_narrate(want_json, "'%s' is already the default model.\n",
				model_name.c_str());
		else if (!cfg.default_model.empty())
			membrane_cli_narrate(want_json, "Default model is already set to '%s' "
				"-- leaving it unchanged (run `membrane model use %s` "
				"yourself if you want to switch it).\n",
				cfg.default_model.c_str(), model_name.c_str());
		else if (membrane_cli_ask_yes_no("Make '" + model_name + "' the default "
				"model?", true, o.assume_yes))
			membrane_cli_dispatch_silently_if_json(want_json, {"use", model_name},
				membrane_model_cmd_dispatch);
		summary["model"]["default_model"] = cfg.default_model.empty()
				? model_name : cfg.default_model;
	}

	/* Step 3: background service -- Section 5, items 11-12. Reuses
	 * membrane_service_cmd_dispatch() wholesale (both install and
	 * start are already independently idempotent -- see this file's
	 * own top comment). Skipped cleanly, never attempted, when
	 * systemctl itself is not available in this environment (a real
	 * condition confirmed directly in a bare `ubuntu:24.04` container
	 * during this mega-phase's own first product audit) -- doctor's
	 * own "service" check already detected this. */
	if (!want_json)
		print_step(++step, total_steps, "background service");

	bool	systemctl_available = true;

	for (const auto &c : doctor_root["checks"])
		if (c["name"] == "service"
			&& c["detail"].value("systemctl_available", true) == false)
			systemctl_available = false;
	summary["service"] = json();
	if (o.no_service)
	{
		membrane_cli_narrate(want_json, "--no-service given -- skipping (use "
			"`membrane serve` directly, or `membrane service "
			"install`/`start` later).\n");
		summary["service"]["skipped"] = "no-service flag given";
	}
	else if (!systemctl_available)
	{
		membrane_cli_narrate(want_json, "systemctl is not available in this "
			"environment -- skipping service install/start. Use "
			"`membrane serve` directly instead.\n");
		summary["service"]["skipped"] = "systemctl not available";
	}
	else
	{
		bool	already_installed = false;
		bool	already_active = false;

		for (const auto &c : doctor_root["checks"])
			if (c["name"] == "service")
			{
				already_installed = c["detail"].value("installed", false);
				already_active = (c["detail"].value("active_state",
					std::string("")) == "active");
			}
		if (already_installed)
			membrane_cli_narrate(want_json, "Service already installed.\n");
		else
			membrane_cli_dispatch_silently_if_json(want_json, {"install"},
				membrane_service_cmd_dispatch);
		if (already_active)
			membrane_cli_narrate(want_json, "Service already running.\n");
		else
		{
			int	rc = membrane_cli_dispatch_silently_if_json(want_json, {"start"},
					membrane_service_cmd_dispatch);

			if (rc != MEMBRANE_EXIT_SUCCESS)
				fprintf(stderr, "membrane setup: service install "
					"succeeded, but starting it failed -- run "
					"`membrane service logs` to see why, then "
					"`membrane service start` again once fixed. Nothing "
					"else was changed.\n");
			summary["service"]["start_ok"] = (rc == MEMBRANE_EXIT_SUCCESS);
		}
		summary["service"]["installed"] = true;
	}

	/* Step 4: verify /health and /v1/models -- Section 5, items 13-14.
	 * Only meaningful if the service (or an already-running `membrane
	 * serve`) is actually reachable -- never treated as a hard failure
	 * of setup itself if it is not (e.g. --no-service was given). */
	if (!want_json)
		print_step(++step, total_steps, "verifying the endpoint");

	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	if (!config_path.empty())
		membrane_server_config_load(config_path, &cfg, &cfg_err);
	/* Skip the (bounded, but real) retry-poll entirely when we already
	 * know nothing was started this run (--no-service, or systemctl
	 * itself unavailable) -- no reason to spend ~3s discovering what
	 * the earlier step already told the user. */
	bool	health_ok = (!o.no_service && systemctl_available)
			&& verify_get_with_retry(cfg.listen_address, cfg.port, "/health");
	bool	models_ok = health_ok && verify_get(cfg.listen_address, cfg.port,
			"/v1/models");

	summary["verified"] = {{"health", health_ok}, {"models", models_ok}};
	if (health_ok && models_ok)
		membrane_cli_narrate(want_json, "/health and /v1/models both answered "
			"correctly.\n");
	else if (o.no_service || !systemctl_available)
		membrane_cli_narrate(want_json, "Endpoint not checked (no service running "
			"yet in this session) -- start it with `membrane service "
			"start` or `membrane serve`, then verify with `membrane "
			"status`.\n");
	else
		membrane_cli_narrate(want_json, "Endpoint did not answer yet -- it may "
			"still be starting. Check again shortly with `membrane "
			"status` or `membrane service logs`.\n");

	/* Step 5: final summary -- Section 5, item 15. */
	if (!want_json)
		print_step(++step, total_steps, "done");
	membrane_cli_narrate(want_json, "Endpoint: http://%s:%d/v1\n",
		cfg.listen_address.c_str(), cfg.port);
	membrane_cli_narrate(want_json, "Next: point any OpenAI-compatible client at "
		"that URL, or run `membrane status` / `membrane doctor` any "
		"time.\n");
	summary["ok"] = true;
	summary["endpoint"] = "http://" + cfg.listen_address + ":"
		+ std::to_string(cfg.port) + "/v1";
	if (want_json)
		printf("%s\n", summary.dump().c_str());
	return (MEMBRANE_EXIT_SUCCESS);
}
