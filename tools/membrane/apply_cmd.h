#ifndef MEMBRANE_APPLY_CMD_H
# define MEMBRANE_APPLY_CMD_H

# include <functional>
# include <string>
# include <vector>

# include <nlohmann/json.hpp>

# include "action.h"
# include "memory_intelligence.h"
# include "plan_cmd.h"

/*
 * Milestone J (active control): `membrane apply` -- the one MUTATING step
 * of
 *
 *     plan -> observe -> advise -> apply
 *
 * It reuses `membrane advise`'s own collection (membrane_advise_collect(),
 * so recommendations are never recomputed), compiles them -- plus any
 * explicit user request -- into an action plan with the pure compiler in
 * tools/membrane-run/action.h, and either shows that plan (DRY RUN, the
 * default: nothing is changed, nothing is sent that mutates) or, only
 * with explicit consent (`--yes`, or an interactive y/N on a TTY),
 * executes it through the transaction engine (stale-plan check, ordered
 * execution, rollback). See docs/active-control.md.
 *
 * What it controls:
 *   membrane-native  the running service, through its existing loopback
 *                    admin routes (activate/pin/unpin, plus the new
 *                    unload) and the persisted overrides block in
 *                    server.json that the service re-reads.
 *   ollama           exactly one action: UNLOAD_MODEL (documented
 *                    keep_alive=0 request, no prompt). Everything else
 *                    stays advisory -- Ollama exposes no persistent
 *                    control surface for it.
 *
 * Nothing here touches a non-loopback endpoint, an external credential
 * or a cloud service.
 */

/* Injection points so the command is testable without a TTY or a real
 * host. A default-constructed struct means "real behavior". */
struct membrane_apply_io_t
{
	bool	interactive_known = false;
	bool	interactive = false;			/* stdin is a TTY */
	std::function<bool(const std::string &)>	confirm;	/* y/N prompt */
	std::function<bool(const std::string &runtime,
		membrane_memory_intelligence_t *intel, std::string *err)>
			intelligence;				/* default: membrane_advise_collect */
	std::function<bool(const std::string &model,
		const membrane_plan_override_request_t &,
		membrane_plan_override_check_t *)>
			planner_check;				/* default: membrane_plan_check_
										 * overrides (Planner v2) */
	std::string	audit_path;				/* "" = resolve */
};

int	membrane_apply_cmd_dispatch_ex(const std::vector<std::string> &args,
		bool want_json, const membrane_apply_io_t &io);
int	membrane_apply_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json);

/*
 * Milestone K: the non-printing entry the adaptive controller uses. It runs
 * EXACTLY the pipeline `membrane apply --yes` runs after consent -- live
 * read, Planner v2 validation of SET_* values, the J compiler, the
 * stale-plan guards, the transaction engine, rollback and the audit
 * record -- so adaptive control has no mutation path of its own. The
 * caller supplies consent by policy (mode = apply); `source` and
 * `audit_extra` (reason chain) are written into the audit record.
 */
typedef struct s_membrane_apply_request
{
	std::string					runtime;
	membrane_action_request_t	req;
	std::string					source = "user";
	nlohmann::json				audit_extra = nlohmann::json::object();
}	membrane_apply_request_t;

typedef struct s_membrane_apply_outcome
{
	bool			runtime_reachable = false;
	bool			attempted = false;		/* a mutation was attempted */
	bool			applied = false;		/* status == applied */
	std::string		status;					/* membrane_apply_status_name */
	std::string		detail;
	size_t			action_count = 0;
	nlohmann::json	report = nlohmann::json::object();
}	membrane_apply_outcome_t;

/* false (with *err) only if the request could not even be set up (unknown
 * runtime, unreadable state). A refused / stale / failed apply returns
 * true with the outcome describing it. */
bool	membrane_apply_execute(const membrane_apply_request_t &r,
			const membrane_apply_io_t &io, membrane_apply_outcome_t *out,
			std::string *err);
bool	membrane_apply_read_live(const std::string &runtime,
			membrane_apply_live_t *live, std::string *err);
bool	membrane_apply_surface(const std::string &runtime,
			membrane_apply_surface_t *surface);

/* Resolves the local audit log path: MEMBRANE_APPLY_AUDIT_PATH, else
 * $XDG_STATE_HOME/membrane/actions.jsonl, else
 * $HOME/.local/state/membrane/actions.jsonl. */
std::string	membrane_apply_audit_path(void);

#endif
