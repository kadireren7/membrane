#ifndef MEMBRANE_SERVICE_STATE_H
# define MEMBRANE_SERVICE_STATE_H

# include <string>

/*
 * Post-v1 product-polish, PR 1: a real v1.0.0 user session found that
 * `membrane use` told the user to run `membrane service start` even
 * though the background service had never been installed -- and
 * `membrane service start` itself then let systemd's raw "Unit
 * membrane.service not found" be the primary UX. Both symptoms trace to
 * the same root cause: `membrane doctor`'s own check_service() was the
 * ONLY place that ever asked "is a real per-user unit/plist/task
 * installed right now" -- every other command that recommends
 * `service start`/`service install` guidance (use_cmd.cpp, model_cmd.cpp's
 * SERVICE_UNAVAILABLE message, service_cmd.cpp's own `start` verb)
 * either assumed "installed but stopped" unconditionally, or never
 * checked before shelling out to the real service manager.
 *
 * This module is the ONE real per-platform "is MEMBRANE's own
 * background service installed/active right now" probe -- extracted
 * verbatim from doctor_cmd.cpp's pre-existing check_service() (same
 * subprocess calls, same per-platform #ifdef isolation, never a second,
 * independently-drifting implementation). doctor_cmd.cpp, use_cmd.cpp,
 * model_cmd.cpp, and service_cmd.cpp's own `start` guard all call this
 * instead of reading systemctl/launchctl/schtasks state independently.
 *
 * Not pure (real subprocess calls, same category as status_client.h's
 * own real HTTP fetch) -- cheap and safe to call more than once per
 * command (bounded subprocess timeouts, same as before).
 */

typedef struct s_membrane_service_probe
{
	bool		manager_available;	/* false iff the real per-platform
									 * service-manager binary itself is
									 * unusable (systemctl/launchctl/
									 * schtasks not on PATH, or otherwise
									 * could not be invoked at all) --
									 * `membrane service` commands cannot
									 * work here regardless of installed/
									 * active below. */
	bool		installed;			/* true iff a real MEMBRANE-managed
									 * unit/plist/task exists right now.
									 * Always false when !manager_available. */
	bool		active;				/* true iff it is currently reported
									 * running -- only meaningful when
									 * installed is true. */
	std::string	manager_name;		/* "systemctl" | "launchctl" |
									 * "schtasks" */
	std::string	active_state_raw;	/* the real, raw state string from
									 * the platform service manager --
									 * "unknown" if it could not be
									 * determined at all. */
}	membrane_service_probe_t;

/*
 * Test-only hook, same established convention this project already uses
 * for hermetic tests (MEMBRANE_SYSTEMD_USER_DIR, MEMBRANE_MODELS_PATH,
 * MEMBRANE_SERVER_CONFIG_PATH, ...): when the MEMBRANE_SERVICE_PROBE_OVERRIDE
 * environment variable is set to one of "not_installed" |
 * "installed_inactive" | "installed_active" | "manager_unavailable",
 * membrane_probe_service() returns a synthetic result matching that name
 * instead of ever invoking a real systemctl/launchctl/schtasks subprocess
 * -- lets `membrane use`/`membrane service start`/`membrane model pin`'s
 * new state-aware guidance be exercised deterministically without
 * depending on whatever real per-user service state happens to exist on
 * the machine running the test. Never read/set outside a test process.
 */
# define MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV	"MEMBRANE_SERVICE_PROBE_OVERRIDE"

membrane_service_probe_t	membrane_probe_service(void);

/*
 * True iff `membrane service start` should refuse up front, before ever
 * invoking the real per-platform service manager -- Section 2 of the
 * task: a real, concise MEMBRANE error instead of letting systemd's raw
 * "Unit membrane.service not found." be the primary UX. Pure, directly
 * unit-testable with a synthetic probe (no subprocess involved).
 */
bool	membrane_service_start_should_be_blocked(
			const membrane_service_probe_t &probe);

#endif
