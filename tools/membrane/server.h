#ifndef MEMBRANE_SERVER_H
# define MEMBRANE_SERVER_H

# include <cstdint>
# include <string>

/*
 * Mega Phase A, PR A3: `membrane serve` -- a long-lived local HTTP process
 * exposing an OpenAI-compatible subset (GET /health, GET /v1/models, POST
 * /v1/chat/completions) on top of the SAME reusable runtime-session core
 * (runtime_session.h, PR A1) and model registry (registry_core.h, PR A2)
 * the CLI already uses -- never a second runtime/planner implementation
 * (Section 2 of the Mega Phase A task's own architectural rule).
 *
 * Security (Section 28): loopback (127.0.0.1) only unless the caller
 * explicitly opts into a different bind address via
 * membrane_server_options_t::allow_non_loopback -- this module itself
 * never silently widens exposure. No authentication is implemented (only
 * defensible for loopback-only binding); no telemetry, no external
 * network calls of any kind.
 */

typedef struct s_membrane_server_options
{
	std::string	bind_address;		/* default "127.0.0.1" */
	int			port;				/* default 8642 (unassigned per
									 * IANA as of this writing); 0 lets
									 * the OS pick an ephemeral port,
									 * used by tests */
	bool		allow_non_loopback;	/* must be explicitly set true by the
									 * caller (Section 28) -- this
									 * module does not expose its own
									 * "just trust me" flag parsing */
	std::string	registry_path;		/* "" = registry_core's own XDG
									 * default */
}	membrane_server_options_t;

/* Blocks until the server is stopped (SIGINT/SIGTERM -- Section 38 -- or
 * membrane_server_request_stop() from another thread/signal handler).
 * Returns a process exit code (0 on a clean stop, nonzero if the server
 * could not even start, e.g. the port is already in use). Prints the
 * real bound host:port to stdout once listening (Section 37 groundwork
 * for a future `membrane status`). */
int	membrane_server_run(const membrane_server_options_t &opts);

/* Signal-safe: only sets an atomic flag membrane_server_run()'s own
 * listener loop checks; never allocates, never calls into libc from
 * signal-handler context beyond the atomic store itself. */
void	membrane_server_request_stop(void);

#endif
