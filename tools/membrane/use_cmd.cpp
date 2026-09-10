#include "use_cmd.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <sys/stat.h>

#include <nlohmann/json.hpp>

#include "registry_core.h"
#include "server_config.h"
#include "model_catalog.h"
#include "variant_selector.h"
#include "model_cmd.h"
#include "status_client.h"
#include "doctor_cmd.h"
#include "cli_shared.h"
#include "product_cli.h"
#include "runtime_session.h"
#include "fs_util.h"

using json = nlohmann::json;

/*
 * See use_cmd.h's own top comment for the full architectural contract
 * (Section 2/3 of the D6 task). This is the one file that ORCHESTRATES
 * catalog resolution + variant selection + the real install transaction
 * + the registry + server_config + the live-switch admin endpoint --
 * every one of those primitives is called through its own existing,
 * already-tested public API, never reimplemented.
 *
 * Error codes (Section 30 of the task) -- reused where this project
 * already has one, new only where no existing code fits:
 *   NOT_FOUND                     -- reused (registry_core.h/model_
 *                                     catalog.h's own existing convention
 *                                     for "no such name")
 *   NO_FEASIBLE_VARIANT           -- reused verbatim from variant_
 *                                     selector.h/cmd_install (Section 30
 *                                     asked for "NO_FITTING_VARIANT",
 *                                     but this project already has the
 *                                     identical concept under this name)
 *   DOWNLOAD_DECLINED             -- new (D6-specific: the user said no)
 *   NONINTERACTIVE_CONSENT_REQUIRED -- new (D6-specific, Section 6)
 *   DOWNLOAD_FAILED               -- new, a coarse wrapper: the real,
 *                                     specific cause (CHECKSUM_MISMATCH,
 *                                     a download_manager.h code, a
 *                                     registry code, ...) is whatever
 *                                     `membrane model install` itself
 *                                     already printed to STDERR during
 *                                     the (possibly stdout-silenced,
 *                                     never stderr-silenced) internal
 *                                     re-invocation below -- a real,
 *                                     disclosed simplification (docs/
 *                                     model-lifecycle.md) rather than
 *                                     refactoring cmd_install's ~300
 *                                     lines to return a structured error
 *                                     just for this one caller.
 *   MODEL_FILE_MISSING            -- new (Section 4: the registered path
 *                                     no longer exists)
 *   MODEL_SWITCH_FAILED           -- new, used only when the server
 *                                     itself did not supply its own more
 *                                     specific error_code (it usually
 *                                     does -- NO_FEASIBLE_CONTEXT/
 *                                     MODEL_LOAD_FAILED -- which is
 *                                     surfaced verbatim instead)
 *   SERVICE_UNAVAILABLE           -- new (the server answered an earlier
 *                                     /v1/status but is unreachable for
 *                                     the activation call itself -- a
 *                                     real, if rare, race; never fatal
 *                                     to the "selected" outcome, since
 *                                     default_model is already saved by
 *                                     that point)
 *   IO_ERROR/registry codes       -- reused verbatim from server_
 *                                     config.h/registry_core.h's own
 *                                     error structs wherever this file
 *                                     calls them directly
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
		fprintf(stderr, "membrane use: %s\n", message.c_str());
}

struct s_use_opts
{
	std::string	name;
	std::string	requested_quant;
	bool		assume_yes = false;
};

static bool	parse_use_args(const std::vector<std::string> &args,
				s_use_opts *o, std::string *err)
{
	for (size_t i = 0; i < args.size(); ++i)
	{
		if ((args[i] == "--quant" || args[i] == "--variant")
			&& i + 1 < args.size())
			o->requested_quant = args[++i];
		else if (args[i] == "--yes" || args[i] == "-y")
			o->assume_yes = true;
		else if (o->name.empty())
			o->name = args[i];
		else
		{
			*err = "unknown option '" + args[i] + "'";
			return (false);
		}
	}
	if (o->name.empty())
	{
		*err = "usage: membrane use MODEL [--quant QUANT] [--yes]";
		return (false);
	}
	return (true);
}

static bool	doctor_gpu_available(void)
{
	json	root;

	membrane_doctor_collect(&root);
	for (const auto &c : root["checks"])
		if (c["name"] == "hardware")
			return (c["detail"].value("gpu_backend_available", false));
	return (false);
}

/* Section 5/8 of the task: the not-yet-installed preview + consent gate.
 * Returns the resolved family/variant to install via *out_family_name/
 * *out_quant, or a nonzero exit code on any failure (already reported). */
static int	preview_and_consent(const membrane_catalog_family_t &fam,
				const membrane_catalog_variant_t &variant,
				const std::vector<membrane_variant_fit_t> &considered,
				bool want_json, bool assume_yes)
{
	if (!want_json)
	{
		std::string	fit_detail = "estimated to fit";

		for (const auto &c : considered)
			if (c.quant == variant.quant)
				fit_detail = c.reason;
		printf("Model: %s (not installed)\n", fam.display_name.c_str());
		printf("Recommended variant: %s\n", variant.quant.c_str());
		printf("Approx download: %.1f GiB\n",
			(double)variant.size_bytes / (1024.0 * 1024.0 * 1024.0));
		printf("Estimated hardware fit: %s\n", fit_detail.c_str());
		printf("Backend: %s\n", doctor_gpu_available()
			? "GPU detected (final choice made automatically at load time)"
			: "CPU only (no compatible GPU detected)");
	}
	/* Section 6: deliberately NOT membrane_cli_ask_yes_no()'s own "non-
	 * interactive defaults to default_yes" convention (right for
	 * `membrane setup`'s own OPTIONAL prompts, wrong here) -- a real
	 * network download triggered with no human and no --yes must fail
	 * clearly, never silently proceed. */
	if (assume_yes)
		return (MEMBRANE_EXIT_SUCCESS);
	if (!membrane_cli_is_interactive())
	{
		print_err(want_json, "NONINTERACTIVE_CONSENT_REQUIRED", "model is "
			"not installed. Re-run with --yes to allow download in "
			"non-interactive mode.");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!membrane_cli_ask_yes_no("Download and install this model?", true,
			false))
	{
		print_err(want_json, "DOWNLOAD_DECLINED", "download declined -- "
			"nothing was installed");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_use_cmd_dispatch(const std::vector<std::string> &args,
				bool want_json)
{
	s_use_opts	o;
	std::string	parse_err;

	if (!parse_use_args(args, &o, &parse_err))
	{
		print_err(want_json, "CLI_ERROR", parse_err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	std::string					registry_path
			= membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;

	if (registry_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_DATA_HOME nor HOME "
			"is set");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		print_err(want_json, reg_err.code, reg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	/* Section 23: resolution precedence -- (1) an exact, already-
	 * registered name always wins outright (never re-resolved against
	 * the catalog at all, even if a catalog family happens to share the
	 * name), (2) otherwise an exact catalog id/alias. */
	const membrane_registry_entry_t	*entry
			= membrane_registry_find(reg, o.name);
	std::string							resolved_name;
	bool								downloaded_this_run = false;
	std::string							installed_variant;

	if (entry == NULL)
	{
		membrane_catalog_t	cat = membrane_catalog_load();
		const membrane_catalog_family_t	*fam
				= membrane_catalog_resolve(cat, o.name);

		if (fam == NULL)
		{
			print_err(want_json, "NOT_FOUND", "'" + o.name + "' is not an "
				"installed model or a known catalog model -- try `membrane "
				"model search " + o.name + "`");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}

		membrane_host_meminfo_t				meminfo;
		membrane_variant_selector_input_t	hw;

		membrane_read_host_meminfo(&meminfo);
		hw.host_total_bytes = meminfo.total_bytes;
		hw.host_available_bytes = meminfo.available_bytes;
		hw.host_available_known = meminfo.ok;

		std::vector<membrane_variant_fit_t>	considered;
		const membrane_catalog_variant_t		*variant = NULL;

		if (!o.requested_quant.empty())
		{
			variant = membrane_catalog_find_variant(*fam, o.requested_quant);
			if (variant == NULL)
			{
				print_err(want_json, "NOT_FOUND", "'" + o.requested_quant
					+ "' is not an available variant of '" + fam->name
					+ "' -- see `membrane model info " + fam->name + "`");
				return (MEMBRANE_EXIT_CLI_ERROR);
			}
			membrane_select_variant(*fam, hw, &considered);
		}
		else
		{
			variant = membrane_select_variant(*fam, hw, &considered);
			if (variant == NULL)
			{
				std::string	alternatives;

				for (const auto &c : considered)
					alternatives += "\n  " + c.quant + ": " + c.reason;
				print_err(want_json, "NO_FEASIBLE_VARIANT", "no variant of "
					"'" + fam->name + "' is estimated to fit this host's "
					"available memory. Considered:" + alternatives
					+ "\nForce a specific variant anyway with --quant, or "
					"try a smaller model (`membrane model search`).");
				return (MEMBRANE_EXIT_MODEL_ERROR);
			}
		}

		int	consent_rc = preview_and_consent(*fam, *variant, considered,
				want_json, o.assume_yes);

		if (consent_rc != MEMBRANE_EXIT_SUCCESS)
			return (consent_rc);

		int	install_rc = membrane_cli_dispatch_silently_if_json(want_json,
				{"install", fam->name, "--quant", variant->quant},
				membrane_model_cmd_dispatch);

		if (install_rc != MEMBRANE_EXIT_SUCCESS)
		{
			print_err(want_json, "DOWNLOAD_FAILED", "installing '"
				+ fam->name + "' failed -- see the message above for the "
				"exact cause; no model was selected or activated");
			return (install_rc);
		}
		downloaded_this_run = true;
		installed_variant = variant->quant;
		resolved_name = fam->name;
		if (!membrane_registry_load(registry_path, &reg, &reg_err))
		{
			print_err(want_json, reg_err.code, reg_err.message);
			return (MEMBRANE_EXIT_MODEL_ERROR);
		}
		entry = membrane_registry_find(reg, resolved_name);
		if (entry == NULL)
		{
			print_err(want_json, "REGISTRY_FAILED", "install reported "
				"success but '" + resolved_name + "' is not in the "
				"registry -- this should not happen, please report it");
			return (MEMBRANE_EXIT_MODEL_ERROR);
		}
	}
	else
		resolved_name = o.name;

	/* Section 4, points 2/3: the registered file must still really exist
	 * (never trust the registry's own cached metadata alone) -- reuses
	 * registry_core.h's own membrane_registry_check_identity(), the
	 * exact same real stat()-based check `membrane doctor`'s own
	 * check_registry() already performs. */
	struct stat							st_buf;
	e_membrane_registry_stat_status	stat_status;

	if (stat(entry->path.c_str(), &st_buf) != 0)
		stat_status = (errno == ENOENT) ? MEMBRANE_REGISTRY_STAT_MISSING
				: MEMBRANE_REGISTRY_STAT_ERROR;
	else
		stat_status = MEMBRANE_REGISTRY_STAT_OK;
	if (stat_status != MEMBRANE_REGISTRY_STAT_OK)
	{
		print_err(want_json, "MODEL_FILE_MISSING", "the registered file "
			"for '" + resolved_name + "' is missing or unreadable at '"
			+ entry->path + "' -- re-install with `membrane model install "
			+ resolved_name + "`, or re-add it with `membrane model add`");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	bool	metadata_stale = std::string(membrane_registry_check_identity(
			*entry, stat_status, (uint64_t)st_buf.st_size,
			membrane_stat_mtime_ns(st_buf)))
		== MEMBRANE_REGISTRY_CHECK_MODIFIED;

	/* Set default_model directly via server_config.h (the exact field
	 * `membrane model use` itself already writes) -- not a second
	 * dispatch through cmd_use, since this file already did its own
	 * (stricter -- Section 4's real file-existence check) registry
	 * validation above. */
	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg;
	membrane_server_config_error_t	cfg_err;

	if (config_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_CONFIG_HOME nor "
			"HOME is set");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_server_config_load(config_path, &cfg, &cfg_err))
	{
		print_err(want_json, cfg_err.code, cfg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	cfg.default_model = resolved_name;
	if (!membrane_server_config_save(config_path, cfg, &cfg_err))
	{
		print_err(want_json, cfg_err.code, cfg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	json	status_json;
	bool	reachable = membrane_fetch_server_status(cfg.listen_address,
			cfg.port, &status_json);
	json	result;

	result["ok"] = true;
	result["model"] = resolved_name;
	result["installed"] = true;
	result["downloaded"] = downloaded_this_run;
	if (!installed_variant.empty())
		result["variant"] = installed_variant;
	result["default_model"] = resolved_name;
	result["service_running"] = reachable;
	if (metadata_stale)
		result["metadata_stale"] = true;

	if (!reachable)
	{
		result["active_model"] = nullptr;
		result["result"] = "selected";
		if (want_json)
			printf("%s\n", result.dump().c_str());
		else
		{
			printf("Selected model: %s\n", resolved_name.c_str());
			if (metadata_stale)
				printf("Note: the registered file's size/mtime changed "
					"since it was added -- cached metadata may be "
					"stale.\n");
			printf("Service is not running. Start it with: membrane "
				"service start\n");
		}
		return (MEMBRANE_EXIT_SUCCESS);
	}

	std::string	endpoint = status_json.value("endpoint", std::string());

	result["endpoint"] = endpoint;

	/* Mega Phase E, PR E2: /v1/status's own single `loaded_model` field
	 * was replaced by `resident_models` (an array -- more than one model
	 * can be resident at once now, see server.cpp's own handle_status()
	 * comment) -- "currently active" for THIS purpose means "already
	 * resident", checked by membership rather than a single equality. */
	bool		currently_active = false;
	std::string	first_resident_name;	/* diagnostic-only fallback for
									 * the "switch was unreachable" report
									 * below -- the FIRST resident model's
									 * name, if any (multiple may now be
									 * resident; this is a best-effort
									 * "what was resident before this
									 * attempt", never a claim that it is
									 * THE active one). */

	if (status_json.contains("resident_models")
		&& status_json["resident_models"].is_array())
	{
		for (const auto &m : status_json["resident_models"])
		{
			if (!m.contains("model") || !m["model"].is_string())
				continue ;
			if (first_resident_name.empty())
				first_resident_name = m["model"].get<std::string>();
			if (m["model"].get<std::string>() == resolved_name)
			{
				currently_active = true;
				break ;
			}
		}
	}

	/* Section 17: idempotent if already active -- checked here via the
	 * real /v1/status read, before ever calling the activate endpoint at
	 * all, so a re-run against an already-correct server does zero extra
	 * work (not merely "reload skipped once you get to the server side"
	 * -- no HTTP switch attempt happens at all). */
	if (currently_active)
	{
		result["active_model"] = resolved_name;
		result["backend"] = status_json.value("backend", std::string("?"));
		result["result"] = "already_active";
		if (want_json)
			printf("%s\n", result.dump().c_str());
		else
		{
			printf("Already active: %s\n", resolved_name.c_str());
			printf("Backend: %s\n",
				result["backend"].get<std::string>().c_str());
			printf("Endpoint: %s\n", endpoint.c_str());
		}
		return (MEMBRANE_EXIT_SUCCESS);
	}

	membrane_activate_result_t	act;
	bool	transport_ok = membrane_activate_model(cfg.listen_address,
			cfg.port, resolved_name, &act);

	if (!transport_ok)
	{
		result["active_model"] = first_resident_name.empty() ? json(nullptr)
				: json(first_resident_name);
		result["result"] = "switch_unreachable";
		result["error"] = {{"code", "SERVICE_UNAVAILABLE"},
			{"message", "the server did not respond to the activation "
				"request"}};
		if (want_json)
			printf("%s\n", result.dump().c_str());
		else
		{
			printf("Selected model: %s\n", resolved_name.c_str());
			printf("Warning: could not reach the server to switch the "
				"active model (SERVICE_UNAVAILABLE) -- the default is "
				"set; try `membrane status` or `membrane service "
				"restart`.\n");
		}
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (!act.ok)
	{
		result["ok"] = false;
		result["active_model"] = act.active_model.empty() ? json(nullptr)
				: json(act.active_model);
		result["result"] = "switch_failed";
		result["error"] = {{"code", act.error_code.empty()
				? "MODEL_SWITCH_FAILED" : act.error_code},
			{"message", act.error_message}};
		if (want_json)
			printf("%s\n", result.dump().c_str());
		else
		{
			printf("Selected model: %s (default set)\n",
				resolved_name.c_str());
			printf("Model switch failed: %s\n", act.error_message.c_str());
			if (!act.active_model.empty())
				printf("The server recovered its previous model ('%s') "
					"and remains available.\n", act.active_model.c_str());
			else
				printf("The server has no model currently loaded.\n");
			printf("Try: membrane doctor\n");
		}
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}

	result["active_model"] = act.active_model;
	result["backend"] = act.backend;
	result["result"] = "switched";
	if (want_json)
		printf("%s\n", result.dump().c_str());
	else
	{
		printf("Selected model: %s\n", resolved_name.c_str());
		printf("Active model: %s\n", act.active_model.c_str());
		printf("Backend: %s\n", act.backend.c_str());
		printf("Endpoint: %s\n", endpoint.c_str());
	}
	return (MEMBRANE_EXIT_SUCCESS);
}
