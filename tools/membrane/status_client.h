#ifndef MEMBRANE_STATUS_CLIENT_H
# define MEMBRANE_STATUS_CLIENT_H

# include <string>

# include <nlohmann/json.hpp>

/*
 * Mega Phase B, PR B1: the GET /v1/status HTTP fetch, factored out of
 * `membrane status` (Mega Phase A, PR A4) so `membrane service status`
 * (Section 10 of the task: "should combine: systemd state + HTTP
 * /v1/status if reachable") reuses the exact same real fetch -- never a
 * second implementation, never a status report synthesized from config
 * alone (Section 10: "Do not fake loaded-model status from config. Use
 * actual server state.").
 */

/* Returns true iff a real HTTP 200 response was received and parsed as
 * JSON. false (out left empty) covers every "not reachable" case alike
 * (connection refused, timeout, non-200, unparseable body) -- callers
 * distinguish those by their own "reachable at all" check (e.g. res !=
 * nullptr) if they need to, but this project's own status commands only
 * ever need the binary "did we get a real status object back". */
bool	membrane_fetch_server_status(const std::string &bind, int port,
			nlohmann::json *out);

/* Human-readable "MEMBRANE server" block, shared verbatim between
 * `membrane status` and the HTTP half of `membrane service status` --
 * one real implementation of the presentation too, not just the fetch. */
void	membrane_print_server_status_human(const nlohmann::json &status);

/*
 * Mega Phase D, PR D6: the CLI-side counterpart of server.cpp's new
 * loopback-only POST /membrane/v1/models/activate -- a thin wrapper
 * around acquire_model_slot() (idempotent if already active, recovers
 * the previous model on a failed switch; see server.cpp's own top
 * comments), never a second switch/lifecycle implementation. Deliberately
 * NOT part of the OpenAI-compatible surface (/v1/...) -- a MEMBRANE-
 * specific admin namespace, same loopback-only exposure as every other
 * route this server already registers.
 */
typedef struct s_membrane_activate_result
{
	bool		ok;				/* true iff the server itself reports the
								 * switch succeeded (mirrors server.cpp's
								 * own acquire_model_slot() return value
								 * verbatim -- NEVER set true just because
								 * a recovery attempt on a FAILED switch
								 * happened to succeed; Section 16 of the
								 * task: never claim success when the
								 * server actually recovered the OLD
								 * model instead of loading the
								 * requested one) */
	bool		already_active;	/* true iff no reload was needed at all */
	std::string	active_model;	/* the server's real, current active
								 * model name after this call -- may be
								 * the PREVIOUS model (recovery) even
								 * when ok is false, or empty if the
								 * server ended up with nothing loaded */
	std::string	backend;		/* real backend the server reports for
								 * active_model, "" if active_model is
								 * itself empty */
	std::string	error_code;		/* "" iff ok */
	std::string	error_message;	/* "" iff ok */
}	membrane_activate_result_t;

/* Returns false only on a TRANSPORT failure (could not reach the server
 * at all, e.g. it stopped between an earlier reachability check and this
 * call) -- *out is left untouched in that case. A real application-level
 * failure (the server responded, but the switch itself failed) is
 * reported via a real HTTP response instead: this function returns true
 * and *out.ok is false, with error_code/error_message/active_model/
 * backend describing the server's own real post-attempt state (see
 * s_membrane_activate_result's own field comments) -- the caller must
 * check both. */
bool	membrane_activate_model(const std::string &bind, int port,
			const std::string &model_name,
			membrane_activate_result_t *out);

/*
 * Mega Phase E, PR E2: the CLI-side counterpart of server.cpp's new
 * loopback-only POST /membrane/v1/models/pin and .../unpin (Section 13
 * of the task) -- same admin namespace, same transport-vs-application-
 * failure split as membrane_activate_model() above.
 */
typedef struct s_membrane_pin_result
{
	bool		ok;
	bool		pinned;			/* the server's real post-call pinned
								 * state -- meaningful iff ok */
	std::string	error_code;		/* "" iff ok */
	std::string	error_message;	/* "" iff ok */
}	membrane_pin_result_t;

/* Returns false only on a TRANSPORT failure (server unreachable). A real
 * application-level failure (e.g. the named model is not resident) is
 * reported via *out with ok == false, error_code/error_message set. */
bool	membrane_set_model_pin(const std::string &bind, int port,
			const std::string &model_name, bool pin,
			membrane_pin_result_t *out);

#endif
