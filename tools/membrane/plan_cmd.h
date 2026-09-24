#ifndef MEMBRANE_PLAN_CMD_H
# define MEMBRANE_PLAN_CMD_H

# include <vector>
# include <string>

/*
 * Milestone G1 (Planner v2 foundation): `membrane plan MODEL` -- a
 * READ-ONLY planning/introspection command. It orchestrates EXISTING,
 * unchanged primitives (registry_core.h/model_catalog.h for identity,
 * variant_selector.h for a catalog-only model's quant estimate,
 * gpu_device.h/runtime_session.h for one real hardware snapshot,
 * context_recommender.h for the actual context/GPU-layers/KV-precision/
 * KV-placement decision) into the common membrane_plan_t representation
 * (membrane_plan.h) and prints it -- it never re-implements any of
 * those algorithms itself.
 *
 * Read-only guarantee (Part 4 of the G1 task): this command calls only
 * membrane_registry_load()/membrane_catalog_load() (never _save() or
 * _add()), never membrane_model_cmd_dispatch()/membrane_use_cmd_
 * dispatch() (no install/download/activate), never server_config.h/
 * service_state.h (no service start, no config mutation). See test_
 * plan_cmd.cpp's own read-only-guarantee tests, which assert the
 * registry/config files on disk are byte-for-byte unchanged (or, for a
 * fresh isolated env, still absent) after a `membrane plan` call.
 *
 * Distinct from:
 *   `membrane use`   -- install/select/activate lifecycle (mutates)
 *   `membrane serve` -- execute inference (starts a server)
 * `membrane plan` never does either.
 */
int	membrane_plan_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

/*
 * Milestone I1: the exact Planner v2 resolution `membrane plan NAME` (no
 * flags) performs for an already-REGISTERED model, returned in-process
 * instead of rendered -- so `membrane observe` can report Planner v2's
 * own estimates (always as ESTIMATED, see observation.h) without a
 * second planner. Same read-only contract as the command: registry
 * load only, GGUF metadata read, one host/device snapshot. Registry-only
 * on purpose (no catalog fallback): observation is about what is on this
 * machine. Returns false with a stable err_code (IO_ERROR, NOT_FOUND,
 * MODEL_FILE_UNREADABLE, or a registry_core.h code) otherwise.
 */
struct s_membrane_plan_v2_result;

bool	membrane_plan_resolve_installed_v2(const std::string &name,
			struct s_membrane_plan_v2_result *out, std::string *err_code,
			std::string *err_message);

#endif
