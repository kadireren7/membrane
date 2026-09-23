#ifndef MEMBRANE_RUNTIME_PLAN_CMD_H
# define MEMBRANE_RUNTIME_PLAN_CMD_H

# include <cstdint>
# include <string>

# include <nlohmann/json.hpp>

# include "runtime_adapter.h"
# include "runtime_plan_assessment.h"

/*
 * Milestone H3: `membrane plan MODEL --runtime ID` -- the runtime-aware,
 * READ-ONLY half of `membrane plan`. plan_cmd.cpp still owns Planner v2
 * (unchanged); this file only renders runtime_plan_assessment.h's
 * classification of a plan (native) or of a runtime model's own metadata
 * plus explicit requests (external runtimes, capability-only).
 *
 * H3 does not apply planner decisions. For an external runtime the only
 * runtime I/O is the adapter's existing H2 read path (describe +
 * inspect_model: GET /api/version, POST /api/show for Ollama). No new
 * HTTP route, no inference, no Modelfile/tag/parameter change, and no
 * registry/config file is read or written for an external runtime --
 * its model identity is the runtime's own, never a MEMBRANE registry name.
 */

/* Explicit `membrane plan` flags, forwarded for the capability-only path
 * (no Planner v2 plan). Auto/adaptive values are "not requested". */
typedef struct s_membrane_runtime_plan_request
{
	bool		context_known;
	uint64_t	context;
	bool		gpu_layers_known;
	int32_t		gpu_layers;
	bool		kv_precision_known;
	int			kv_precision;
	std::string	quant;
}	membrane_runtime_plan_request_t;

/* The stable JSON document (schema_version MEMBRANE_RUNTIME_ASSESSMENT_
 * SCHEMA_VERSION). `model` and `planner_plan` are built by the caller;
 * planner_plan is null when no Planner v2 plan exists. */
nlohmann::json	membrane_runtime_assessment_json(
					const membrane_runtime_descriptor_t &runtime,
					const membrane_runtime_plan_assessment_t &a,
					const nlohmann::json &model,
					const nlohmann::json &planner_plan);

void	membrane_runtime_assessment_print_human(const nlohmann::json &doc);

/* Native path: classify Planner v2's selected (or first) plan against
 * membrane-native. `planner_plan_json` is plan_cmd.cpp's existing
 * `membrane plan --json` object, embedded verbatim. */
int		membrane_runtime_plan_native(const membrane_plan_t &plan,
			const nlohmann::json &planner_plan_json, bool want_json);

/* External path (capability-only): no registry lookup, no Planner v2. */
int		membrane_runtime_plan_external(const membrane_runtime_adapter_t &a,
			const std::string &runtime_model_id,
			const membrane_runtime_plan_request_t &req, bool want_json);

#endif
