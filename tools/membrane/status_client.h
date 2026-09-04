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

#endif
