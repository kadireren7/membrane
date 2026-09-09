#ifndef MEMBRANE_USE_CMD_H
# define MEMBRANE_USE_CMD_H

# include <string>
# include <vector>

/*
 * Mega Phase D, PR D6: `membrane use MODEL` -- makes "I want to use this
 * model" a first-class product action (Section 0/3 of the task), so a
 * normal user never has to manually compose `model install` + `model
 * add` + `model use` + `service restart` + `status` themselves.
 *
 * This module is pure ORCHESTRATION -- it introduces no second registry,
 * no second install/download pipeline, and no second model-lifecycle
 * policy (Section 2 of the task: "do not duplicate existing install/
 * registry/service logic"). It calls, directly:
 *   - registry_core.h (membrane_registry_find/load) to check whether
 *     MODEL is already installed and whether its file still exists;
 *   - model_catalog.h/variant_selector.h (the exact functions cmd_install
 *     itself already uses) to resolve/preview a not-yet-installed
 *     catalog model before asking for consent;
 *   - membrane_model_cmd_dispatch({"install", ...}) (model_cmd.h) --
 *     the SAME real download/verify/register transaction `membrane model
 *     install` runs, re-invoked internally (via cli_shared.h's
 *     stdout_silencer_t, exactly like setup_cmd.cpp already does) rather
 *     than a second downloader;
 *   - server_config.h directly to read/write default_model (the same
 *     field `membrane model use` itself writes -- see model_cmd.cpp's
 *     own cmd_use(), which this module's "installed flow" subsumes);
 *   - status_client.h's membrane_fetch_server_status()/
 *     membrane_activate_model() to detect whether a server is running
 *     and to trigger a live model switch through server.cpp's own
 *     already-existing acquire_model_slot() (idempotent-if-already-
 *     active, recovers the previous model on a failed switch -- see
 *     server.cpp's own top comments; this module never reimplements
 *     that logic, only calls the new POST /membrane/v1/models/activate
 *     endpoint that wraps it).
 *
 * Resolution precedence (Section 23 of the task, deterministic, no
 * fuzzy/typo matching for a name that triggers a real network
 * install): (1) an exact, already-registered registry name, (2) an
 * exact catalog id or alias (model_catalog.h's own
 * membrane_catalog_resolve() already matches name OR alias in one exact,
 * case-insensitive step -- never ambiguous).
 *
 * Exit codes match membrane-run's own convention (product_cli.h):
 * MEMBRANE_EXIT_SUCCESS/CLI_ERROR/MODEL_ERROR/RUNTIME_ERROR. Fine-
 * grained failure identity lives in the JSON error.code string, reusing
 * this project's existing codes where one already exists (NOT_FOUND,
 * NO_FEASIBLE_VARIANT, CHECKSUM_MISMATCH, ...) -- see use_cmd.cpp's own
 * top comment for the exact new/reused code list (Section 30).
 */
int	membrane_use_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
