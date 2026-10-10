#ifndef MEMBRANE_ORCHESTRATE_CMD_H
# define MEMBRANE_ORCHESTRATE_CMD_H

# include <string>
# include <vector>

# include "orchestrator.h"

/*
 * Milestone L: `membrane run MODEL` -- "run this model", without having to
 * know which runtime or backend. It gathers the facts for every runtime
 * through the common adapter table (runtime_adapter.h -- no per-runtime
 * branches beyond how each one's model is looked up), asks the pure
 * orchestrator (tools/membrane-run/orchestrator.h) which runtime fits, and
 * PRINTS WHY.
 *
 *   membrane run MODEL                 auto: choose, explain, then act
 *   membrane run MODEL --runtime ID    force one runtime (still explained)
 *   membrane run MODEL --explain       choose and explain; change nothing
 *   membrane run MODEL --intent I      general (default) | interactive |
 *                                      memory_saving | long_context
 *
 * Acting means: if membrane-native is chosen, it is exactly `membrane use
 * MODEL` (activate in the running service); if an external runtime is
 * chosen, MEMBRANE does NOT launch or control it -- it tells you the
 * endpoint and model id to point a client at. Nothing is downloaded or
 * installed by `run`.
 */

typedef struct s_membrane_orch_gather
{
	membrane_topology_t							topology;
	std::vector<membrane_orch_runtime_input_t>	inputs;
	std::vector<std::string>					endpoints;	/* per input */
}	membrane_orch_gather_t;

/* The hardware topology: host RAM plus EVERY enumerated GPU device. */
void	membrane_orch_build_topology(membrane_topology_t *t);

/* Name normalization used to match one model across runtimes' id schemes:
 * lowercase, a trailing ":latest" dropped, '-' '_' ':' '/' ' ' removed.
 * Pure. Exact (case-insensitive) equality is tried first by callers. */
std::string	membrane_orch_normalize_model(const std::string &name);

/* Collects the per-runtime facts for `model`. runtime_filter "" or "auto"
 * considers every runtime in the table; otherwise only that one. */
bool	membrane_orch_gather(const std::string &model,
			const std::string &runtime_filter, membrane_orch_gather_t *out,
			std::string *err);

int	membrane_run_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json);

#endif
