#ifndef MEMBRANE_ADAPTIVE_CMD_H
# define MEMBRANE_ADAPTIVE_CMD_H

# include <functional>
# include <string>
# include <vector>

/*
 * Milestone K: `membrane adaptive status|enable|disable` -- the normal
 * path for opting in to (or out of) adaptive runtime control. Users never
 * edit server.json by hand for this.
 *
 *   status    READ-ONLY: asks the running service (GET /membrane/v1/
 *             adaptive, loopback) for its live controller state; when the
 *             service is not running, shows the persisted policy only.
 *   enable    persists the policy. Default mode `recommend` (never
 *             mutates). `--mode apply` lets the controller change runtime
 *             state within the allowed classes, so it states that plainly
 *             and needs explicit consent: an interactive y/N, or `--yes`
 *             when not on a terminal (otherwise it fails closed, exit 2).
 *   disable   turns it off (the other preferences are kept).
 *
 * The running service re-reads the policy every tick: no restart needed.
 */

struct membrane_adaptive_cmd_io_t
{
	bool	interactive_known = false;
	bool	interactive = false;
	std::function<bool(const std::string &)>	confirm;
};

int	membrane_adaptive_cmd_dispatch_ex(const std::vector<std::string> &args,
		bool want_json, const membrane_adaptive_cmd_io_t &io);
int	membrane_adaptive_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json);

#endif
