#ifndef MEMBRANE_RUNTIME_CMD_H
# define MEMBRANE_RUNTIME_CMD_H

# include <string>
# include <vector>

# include <nlohmann/json.hpp>

/*
 * Milestone H1 (runtime abstraction foundation): `membrane runtime
 * list|inspect` -- a READ-ONLY introspection surface over runtime_
 * capabilities.h's own static runtime table (tools/membrane-run/
 * runtime_capabilities.h). Orchestrates that existing, unchanged
 * module; never re-implements the capability matrix here, and never
 * touches the model registry, server config, or service state (no
 * install/download/activate/service mutation of any kind -- see test_
 * runtime_cmd.cpp's own read-only-guarantee tests).
 *
 * Distinct from:
 *   `membrane status`          -- is the background HTTP service up
 *   `membrane service status`  -- OS-level service unit state
 * `membrane runtime inspect` never starts, stops, or queries either --
 * see runtime_capabilities.h's own "availability != service running"
 * top comment. Distinct from `membrane plan MODEL`, which stays the
 * one place Planner v2's own decisions are computed; H1 deliberately
 * does not add a `--plan MODEL` option here (kept for a later
 * milestone -- see docs/runtime-abstraction.md).
 */
int	membrane_runtime_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

/* Exposed separately (same convention as doctor_cmd.h's membrane_
 * doctor_collect()) so tests -- and any future in-process caller --
 * can assert on the exact JSON shape without capturing stdout. */
nlohmann::json	membrane_runtime_list_json(void);

/* Returns true and fills *out iff runtime_id is a real, describable
 * runtime (H1: only "membrane-native"); false otherwise, with
 * *err_message set to a human-readable reason (distinguishing a
 * genuinely unknown id from a reserved-but-unimplemented one -- see
 * runtime_capabilities.h's own MEMBRANE_RUNTIME_ID_OLLAMA/_VLLM top
 * comment). */
bool	membrane_runtime_inspect_json(const std::string &runtime_id,
			nlohmann::json *out, std::string *err_message);

#endif
