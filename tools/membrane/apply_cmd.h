#ifndef MEMBRANE_APPLY_CMD_H
# define MEMBRANE_APPLY_CMD_H

# include <functional>
# include <string>
# include <vector>

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

/* Resolves the local audit log path: MEMBRANE_APPLY_AUDIT_PATH, else
 * $XDG_STATE_HOME/membrane/actions.jsonl, else
 * $HOME/.local/state/membrane/actions.jsonl. */
std::string	membrane_apply_audit_path(void);

#endif
