#ifndef MEMBRANE_RUN_RUNTIME_PLAN_ASSESSMENT_H
# define MEMBRANE_RUN_RUNTIME_PLAN_ASSESSMENT_H

# include <stddef.h>
# include <stdint.h>

# include "membrane_plan.h"
# include "runtime_capabilities.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone H3 (runtime-aware plan assessment): answers, READ-ONLY,
 * "given this model and this runtime, what can MEMBRANE recommend, what
 * can the runtime actually control, and what remains unsupported?"
 *
 * This is NOT a planner. It never computes a context, GPU-layer count,
 * KV precision, KV placement or memory figure. It only CLASSIFIES values
 * that already exist -- a Planner v2 membrane_plan_t (membrane_plan.h,
 * consumed purely by reading it), or an explicit user request -- against
 * one runtime's membrane_runtime_capabilities_t (runtime_capabilities.h).
 * It never applies anything: nothing here, or in any caller, sends a
 * setting to a runtime. H3 does not apply planner decisions.
 *
 * Llama-free, no I/O, deterministic -- same pure pattern as
 * runtime_capabilities.h/membrane_plan.h.
 */

# define MEMBRANE_RUNTIME_ASSESSMENT_SCHEMA_VERSION	1

/*
 * How much planning evidence stands behind an assessment. Ordered from
 * least to most evidence; UNAVAILABLE is the zero (fail-closed) value.
 *
 *   PLANNER_EXACT    Planner v2 produced context/GPU/KV decisions from
 *                    REAL model hparams (an installed GGUF), for a variant
 *                    that is not an estimate.
 *   PLANNER_ESTIMATE Planner v2 produced a plan, but only as a disclosed
 *                    estimate (sibling-scaled hparams, or a catalog-size
 *                    fit with no context/GPU/KV decision).
 *   CAPABILITY_ONLY  No Planner v2 plan exists for this model on this
 *                    runtime (e.g. an Ollama model: the API exposes
 *                    architecture/parameter count/max context/quant, but
 *                    not what Planner v2's memory math needs). Only the
 *                    runtime's capabilities, the runtime's own model
 *                    metadata and any explicit user request are assessed.
 *                    This is a valid, non-error result.
 *   UNAVAILABLE      Nothing can be assessed: the runtime is not
 *                    available, or the model is unknown.
 */
typedef enum e_membrane_planning_level
{
	MEMBRANE_PLANNING_LEVEL_UNAVAILABLE = 0,
	MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY,
	MEMBRANE_PLANNING_LEVEL_PLANNER_ESTIMATE,
	MEMBRANE_PLANNING_LEVEL_PLANNER_EXACT
}	membrane_planning_level_t;

const char	*membrane_planning_level_name(membrane_planning_level_t l);

/*
 * Overall result -- see membrane_runtime_recommend_plan()'s doc comment
 * for the exact deterministic rules. UNSUPPORTED is the zero value.
 */
typedef enum e_membrane_runtime_actionability
{
	MEMBRANE_ACTIONABILITY_UNSUPPORTED = 0,
	MEMBRANE_ACTIONABILITY_ADVISORY_ONLY,
	MEMBRANE_ACTIONABILITY_PARTIALLY_ACTIONABLE,
	MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE
}	membrane_runtime_actionability_t;

const char	*membrane_runtime_actionability_name(
				membrane_runtime_actionability_t a);

/*
 * Per-dimension applicability, derived ONLY from the runtime's own
 * capability states (never from a guess about runtime internals):
 *
 *   control capability SUPPORTED   -> CONTROLLABLE
 *   control capability PARTIAL     -> PARTIALLY_CONTROLLABLE
 *   control capability UNSUPPORTED -> OBSERVABLE_ONLY if the dimension's
 *                                     matching observability capability
 *                                     is SUPPORTED/PARTIAL, else
 *                                     UNSUPPORTED
 *   control capability UNKNOWN     -> UNKNOWN
 *
 * "Controllable" means the runtime EXPOSES a control surface for it --
 * never that MEMBRANE used it. UNKNOWN is the zero value.
 */
typedef enum e_membrane_dimension_applicability
{
	MEMBRANE_APPLICABILITY_UNKNOWN = 0,
	MEMBRANE_APPLICABILITY_UNSUPPORTED,
	MEMBRANE_APPLICABILITY_OBSERVABLE_ONLY,
	MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE,
	MEMBRANE_APPLICABILITY_CONTROLLABLE
}	membrane_dimension_applicability_t;

const char	*membrane_dimension_applicability_name(
				membrane_dimension_applicability_t a);

/* Fixed dimension order -- also the output order (deterministic). The
 * first five are the dimensions Planner v2 can decide; DEVICE_SELECTION
 * and CONCURRENCY are informational only, because membrane_plan_t has no
 * device or concurrency DECISION (its device is a hardware snapshot), so
 * they are never "required". */
typedef enum e_membrane_assessment_dimension
{
	MEMBRANE_ASSESS_DIM_QUANT_VARIANT = 0,
	MEMBRANE_ASSESS_DIM_CONTEXT,
	MEMBRANE_ASSESS_DIM_GPU_LAYERS,
	MEMBRANE_ASSESS_DIM_KV_PRECISION,
	MEMBRANE_ASSESS_DIM_KV_PLACEMENT,
	MEMBRANE_ASSESS_DIM_DEVICE_SELECTION,
	MEMBRANE_ASSESS_DIM_CONCURRENCY,
	MEMBRANE_ASSESS_DIM_COUNT
}	membrane_assessment_dimension_t;

/* Where a dimension's value came from. NONE is the zero value. */
typedef enum e_membrane_assessment_value_source
{
	MEMBRANE_ASSESS_VALUE_NONE = 0,
	MEMBRANE_ASSESS_VALUE_PLANNER,			/* a membrane_plan_t decision/
											 * identity field; plan_source
											 * says which */
	MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST,	/* a user flag, no plan */
	MEMBRANE_ASSESS_VALUE_RUNTIME_METADATA	/* reported by the runtime
											 * itself (e.g. an Ollama
											 * model's quantization_level) */
}	membrane_assessment_value_source_t;

const char	*membrane_assessment_value_source_name(
				membrane_assessment_value_source_t s);

/* Top-level and per-dimension reason codes (stable; only add new ones). */
# define MEMBRANE_ASSESS_REASON_RUNTIME_UNAVAILABLE		"RUNTIME_UNAVAILABLE"
# define MEMBRANE_ASSESS_REASON_MODEL_UNKNOWN				"MODEL_UNKNOWN"
# define MEMBRANE_ASSESS_REASON_NO_PLANNER_PLAN			"NO_PLANNER_PLAN"
# define MEMBRANE_ASSESS_REASON_EXACT_MEMORY_PLAN_UNAVAILABLE	"EXACT_MEMORY_PLAN_UNAVAILABLE"
# define MEMBRANE_ASSESS_REASON_PLAN_ESTIMATE_ONLY			"PLAN_ESTIMATE_ONLY"
# define MEMBRANE_ASSESS_REASON_PLAN_NOT_FEASIBLE			"PLAN_NOT_FEASIBLE"
# define MEMBRANE_ASSESS_REASON_NO_REQUIRED_DIMENSIONS		"NO_REQUIRED_DIMENSIONS"
# define MEMBRANE_ASSESS_REASON_QUANT_FIXED_BY_RUNTIME_MODEL	"QUANT_FIXED_BY_RUNTIME_MODEL"
# define MEMBRANE_ASSESS_REASON_REQUESTED_QUANT_DIFFERS	"REQUESTED_QUANT_DIFFERS_FROM_RUNTIME_MODEL"
# define MEMBRANE_ASSESS_REASON_CONTEXT_EXCEEDS_MODEL_MAX	"CONTEXT_EXCEEDS_MODEL_MAXIMUM"
/* Per-dimension codes are "<H1 negotiation dimension>_<CAPABILITY STATE>",
 * e.g. CONTEXT_CONTROL_SUPPORTED, KV_PRECISION_CONTROL_UNSUPPORTED --
 * reusing runtime_capabilities.h's MEMBRANE_NEGOTIATION_DIM_* names. */

# define MEMBRANE_ASSESS_MAX_REASONS		12
# define MEMBRANE_ASSESS_REASON_CODE_MAX	56
# define MEMBRANE_ASSESS_DETAIL_MAX		200
# define MEMBRANE_ASSESS_VALUE_MAX			40

typedef struct s_membrane_assessment_reason
{
	char	code[MEMBRANE_ASSESS_REASON_CODE_MAX];
	char	detail[MEMBRANE_ASSESS_DETAIL_MAX];
}	membrane_assessment_reason_t;

typedef struct s_membrane_assessment_dimension_result
{
	membrane_assessment_dimension_t		dimension;
	int									planner_dimension;	/* Planner v2
										 * can decide this dimension */
	int									required;	/* the plan/request
										 * actually relies on it */
	int									value_known;
	char	value[MEMBRANE_ASSESS_VALUE_MAX];		/* display string */
	membrane_assessment_value_source_t	value_source;
	membrane_plan_source_t				plan_source;	/* iff value_source
										 * is PLANNER */
	membrane_capability_state_t			capability;	/* the runtime's
										 * CONTROL capability */
	membrane_dimension_applicability_t	applicability;
	char	reason_code[MEMBRANE_ASSESS_REASON_CODE_MAX];
}	membrane_assessment_dimension_result_t;

typedef struct s_membrane_runtime_plan_assessment
{
	int		schema_version;
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	char	runtime_model_id[MEMBRANE_PLAN_NAME_MAX];
	membrane_capability_provenance_t	capability_provenance;

	membrane_planning_level_t			planning_level;
	membrane_runtime_actionability_t	actionability;
	size_t								required_count;
	size_t								controllable_count;

	membrane_assessment_dimension_result_t	dimensions[MEMBRANE_ASSESS_DIM_COUNT];

	membrane_assessment_reason_t	reasons[MEMBRANE_ASSESS_MAX_REASONS];
	size_t							reason_count;
}	membrane_runtime_plan_assessment_t;

/*
 * Plain facts a caller already has. Everything is optional; nothing is
 * inferred from a missing field. `plan` (Planner v2's SELECTED plan) wins
 * over the explicit request fields: when plan != NULL, requested_* are
 * ignored, because Planner v2 already baked any explicit flag into it.
 */
typedef struct s_membrane_runtime_recommend_input
{
	const membrane_runtime_descriptor_t	*runtime;	/* required */
	const membrane_plan_t				*plan;		/* NULL: no Planner v2
										 * plan for this runtime/model */

	/* The runtime's own model identity + metadata (external runtimes). */
	int				model_known;
	const char		*runtime_model_id;
	const char		*model_quantization;		/* NULL/"" = unknown */
	int				model_max_context_known;
	uint64_t		model_max_context;			/* trained maximum */

	/* Explicit user requests (used only when plan == NULL). */
	int				requested_context_known;
	uint64_t		requested_context;
	int				requested_gpu_layers_known;
	int32_t			requested_gpu_layers;		/* MEMBRANE_JOINT_GPU_LAYERS_
												 * REQUEST_ALL or N >= 0 */
	int				requested_kv_precision_known;
	int				requested_kv_precision;		/* MEMBRANE_JOINT_KV_* */
	const char		*requested_quant;			/* NULL/"" = none */
}	membrane_runtime_recommend_input_t;

/*
 * Classifies (never applies) plan/request dimensions against one runtime.
 *
 * Required dimensions -- the SAME rule as H1's membrane_runtime_
 * negotiate_plan():
 *   plan != NULL: plan->decisions.has_decisions makes context, GPU layers,
 *     KV precision and KV placement required; plan->identity.variant_known
 *     makes quant/variant required.
 *   plan == NULL: each requested_* field that is set makes its dimension
 *     required. A quant reported by the runtime's own model metadata is
 *     shown, but is NOT required (it is a fact, not a recommendation).
 *
 * Planning level:
 *   runtime NULL or not AVAILABLE            -> UNAVAILABLE
 *   plan != NULL, has_decisions and the variant is not an estimate
 *                                            -> PLANNER_EXACT
 *   plan != NULL otherwise                   -> PLANNER_ESTIMATE
 *   plan == NULL and model_known             -> CAPABILITY_ONLY
 *   plan == NULL and !model_known            -> UNAVAILABLE
 *
 * Overall actionability (deterministic, no scoring):
 *   planning level UNAVAILABLE               -> UNSUPPORTED
 *   no required dimension                    -> ADVISORY_ONLY
 *   every required dimension CONTROLLABLE    -> FULLY_ACTIONABLE
 *   at least one, but not all, CONTROLLABLE  -> PARTIALLY_ACTIONABLE
 *   none CONTROLLABLE                        -> ADVISORY_ONLY
 * PARTIALLY_CONTROLLABLE / OBSERVABLE_ONLY / UNSUPPORTED / UNKNOWN never
 * count as controllable (H1's "only SUPPORTED satisfies" rule).
 *
 * Deterministic: identical inputs always produce an identical result.
 * Returns out->actionability.
 */
membrane_runtime_actionability_t	membrane_runtime_recommend_plan(
		const membrane_runtime_recommend_input_t *in,
		membrane_runtime_plan_assessment_t *out);

/* Stable snake_case names for JSON (e.g. "quant_variant", "kv_precision"). */
const char	*membrane_assessment_dimension_name(
				membrane_assessment_dimension_t d);

# ifdef __cplusplus
}
# endif

#endif
