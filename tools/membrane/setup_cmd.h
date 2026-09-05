#ifndef MEMBRANE_SETUP_CMD_H
# define MEMBRANE_SETUP_CMD_H

# include <string>
# include <vector>

/*
 * Mega Phase C, PR C1, Sections 4-8 of the task: `membrane setup` --
 * the guided first-run onboarding flow. Orchestrates EXISTING
 * capabilities (membrane_doctor_collect() for hardware/state,
 * membrane_model_cmd_dispatch()/registry_core.h for model registration,
 * membrane_service_cmd_dispatch() for the background service,
 * status_client.h for live verification) rather than re-implementing
 * any of them.
 *
 * Interactive (a real TTY, prompts for whatever wasn't given on the
 * command line) and non-interactive (--yes, or stdin is not a TTY --
 * CI/automation) modes share the exact same underlying steps (Section
 * 6) -- only whether a missing input is PROMPTED FOR or left at a safe
 * default differs. Idempotent (Section 7): running it twice reports
 * "already registered"/"already installed"/"already running" cleanly,
 * never a raw duplicate-name error and never a second unit/registry
 * entry. Never destructively undoes existing user configuration
 * (Section 8) -- a failure partway through is reported with the exact
 * completed state and a concrete recovery command, never rolled back.
 */
int	membrane_setup_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
