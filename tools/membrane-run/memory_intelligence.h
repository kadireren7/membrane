#ifndef MEMBRANE_RUN_MEMORY_INTELLIGENCE_H
# define MEMBRANE_RUN_MEMORY_INTELLIGENCE_H

# include <stddef.h>
# include <stdint.h>

# include "observation.h"
# include "runtime_capabilities.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone I3 (memory intelligence): the last of Milestone I's three
 * layers --
 *
 *     observation snapshot (I1/I2)  +  runtime capabilities (H1/H2)
 *         -> membrane_memory_intelligence_assemble()
 *         -> findings[] + recommendations[]
 *
 * READ-ONLY, same as everything else Milestone I ships. This module never
 * touches the machine, never calls a planner, never applies anything --
 * membrane_memory_intelligence_assemble() is a PURE function of an
 * already-built membrane_observation_snapshot_t (observation.h) and an
 * already-described membrane_runtime_capabilities_t (runtime_capabilities.h)
 * -- llama-free, no I/O, same testable-without-a-host pattern as membrane_
 * plan.h/runtime_capabilities.h/observation.h.
 *
 * What this module is NOT:
 *   - not a second planner (it never computes a context/GPU-layer/KV
 *     figure -- Planner v2's own numbers already reached the snapshot as
 *     context_planned/kv_estimated_bytes/kv_planned_precision/
 *     kv_planned_placement, always tagged ESTIMATED there; this module
 *     only reads them, exactly as membrane_plan_t is only ever read by
 *     runtime_capabilities.h/runtime_plan_assessment.h);
 *   - not a capability negotiator (runtime_capabilities.h's own
 *     membrane_runtime_negotiate_plan()/runtime_plan_assessment.h's
 *     membrane_runtime_recommend_plan() already answer "can this runtime
 *     satisfy THIS plan" against a membrane_plan_t; this module answers a
 *     narrower, snapshot-shaped question instead -- "given what is
 *     observably true right now, does this runtime expose ANY control
 *     surface for the dimension a finding is about" -- by reading the
 *     matching membrane_runtime_capabilities_t field directly, never by
 *     re-deriving or duplicating H1/H3's own negotiation logic);
 *   - not a remediation engine (every recommendation is advisory text;
 *     mutates_state is hardcoded 0 everywhere in this module, never
 *     computed -- Milestone J, not this one, may one day apply anything).
 *
 * Provenance discipline (the module's central rule, Part 6 of the I3
 * task): every finding/recommendation traces back to fields the snapshot
 * already labeled measured/runtime_reported/estimated/static_metadata/
 * unknown (observation.h's own membrane_obs_provenance_t, reused here
 * verbatim -- no second provenance vocabulary). A finding is never
 * produced from a field that is !known (Part 25/test L: "unknown context
 * -> no fabricated finding"), and an ESTIMATED-provenance finding is never
 * ranked above a MEASURED CRITICAL one (Part 16/17).
 */

# define MEMBRANE_INTEL_SCHEMA_VERSION	1

# define MEMBRANE_INTEL_CODE_MAX		56
# define MEMBRANE_INTEL_SUMMARY_MAX		200
# define MEMBRANE_INTEL_EVIDENCE_MAX	200
# define MEMBRANE_INTEL_ACTION_MAX		220
# define MEMBRANE_INTEL_ID_MAX			128

# define MEMBRANE_INTEL_MAX_FINDINGS		12
# define MEMBRANE_INTEL_MAX_EVIDENCE		3
# define MEMBRANE_INTEL_MAX_RECOMMENDATIONS	8
# define MEMBRANE_INTEL_MAX_REASON_CODES	3
# define MEMBRANE_INTEL_MAX_TOP_REASONS		12

/*
 * Part 22: the overall status, deterministically derived from findings[]
 * (see membrane_memory_intelligence_assemble()'s own doc comment for the
 * exact rule) -- never hand-set by any single check. INSUFFICIENT_DATA is
 * NOT a severity tier above/below the others: it means the question
 * itself could not be answered (the runtime was unobservable, or every
 * dimension this module knows how to assess came back unknown), which is
 * different from "assessed and healthy" (OK). OK is deliberately the zero
 * value, matching this project's general "zero-init must never look like
 * an active problem" convention (but see membrane_memory_intelligence_t's
 * own top comment: a freshly zeroed result is never treated as a real OK
 * assessment by any caller -- only membrane_memory_intelligence_assemble()
 * legitimately produces one).
 */
typedef enum e_membrane_intel_status
{
	MEMBRANE_INTEL_STATUS_OK = 0,
	MEMBRANE_INTEL_STATUS_NOTICE,
	MEMBRANE_INTEL_STATUS_WARNING,
	MEMBRANE_INTEL_STATUS_CRITICAL,
	MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA
}	membrane_intel_status_t;

const char	*membrane_intel_status_name(membrane_intel_status_t s);

/*
 * Part 3: four levels, ordered least to most severe. Findings are
 * deliberately conservative -- CRITICAL is reserved for measured
 * (never estimated) pressure that is very likely to cause an allocation
 * failure imminently (see the threshold policy in memory_intelligence.c's
 * own top comment). An estimate-only finding never exceeds NOTICE
 * (Part 16).
 */
typedef enum e_membrane_intel_severity
{
	MEMBRANE_INTEL_SEVERITY_INFO = 0,
	MEMBRANE_INTEL_SEVERITY_NOTICE,
	MEMBRANE_INTEL_SEVERITY_WARNING,
	MEMBRANE_INTEL_SEVERITY_CRITICAL
}	membrane_intel_severity_t;

const char	*membrane_intel_severity_name(membrane_intel_severity_t s);

/* Part 3: "affected dimension" -- a small, flat, fixed vocabulary (not a
 * free-form string), so a consumer can group/filter findings without
 * string-matching summaries. */
typedef enum e_membrane_intel_dimension
{
	MEMBRANE_INTEL_DIM_HOST_MEMORY = 0,
	MEMBRANE_INTEL_DIM_VRAM,
	MEMBRANE_INTEL_DIM_CONTEXT,
	MEMBRANE_INTEL_DIM_KV,
	MEMBRANE_INTEL_DIM_MODEL_RESIDENCY,
	MEMBRANE_INTEL_DIM_RUNTIME_CAPABILITY,
	MEMBRANE_INTEL_DIM_TELEMETRY
}	membrane_intel_dimension_t;

const char	*membrane_intel_dimension_name(membrane_intel_dimension_t d);

/*
 * Reason/finding codes (Part 4). Only codes with real, wired evidence in
 * membrane_memory_intelligence_assemble() are defined here -- see this
 * header's own comment on each one for exactly which snapshot/capability
 * fields back it. Two of the task's *potential* codes are deliberately
 * NOT implemented, and why:
 *   MODEL_MEMORY_HIGH_RELATIVE_TO_AVAILABLE_RAM -- the snapshot's own
 *     model_file_size_bytes is a STATIC_METADATA fact about the file on
 *     disk, not about what is actually resident/loaded; conflating the
 *     two would misreport a merely-registered-but-unloaded model as
 *     memory pressure. KV_FOOTPRINT_ESTIMATE_HIGH already covers the
 *     loaded-model estimate case with real provenance.
 *   ESTIMATED_CONTEXT_MEMORY_NEAR_LIMIT -- the snapshot has no distinct
 *     "estimated total footprint at the requested context" figure
 *     separate from kv_estimated_bytes (Planner v2's own model/KV split
 *     is not surfaced to `membrane observe`); implementing this code
 *     today would just be KV_FOOTPRINT_ESTIMATE_HIGH under a second name.
 */
# define MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW			"HOST_MEMORY_HEADROOM_LOW"
# define MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL		"HOST_MEMORY_HEADROOM_CRITICAL"
# define MEMBRANE_INTEL_CODE_VRAM_HEADROOM_LOW					"VRAM_HEADROOM_LOW"
# define MEMBRANE_INTEL_CODE_VRAM_HEADROOM_CRITICAL				"VRAM_HEADROOM_CRITICAL"
# define MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM			"CONTEXT_NEAR_MODEL_MAXIMUM"
# define MEMBRANE_INTEL_CODE_CONTEXT_EXCEEDS_MODEL_MAXIMUM		"CONTEXT_EXCEEDS_MODEL_MAXIMUM"
# define MEMBRANE_INTEL_CODE_KV_FOOTPRINT_ESTIMATE_HIGH			"KV_FOOTPRINT_ESTIMATE_HIGH"
# define MEMBRANE_INTEL_CODE_MULTIPLE_MODELS_RESIDENT				"MULTIPLE_MODELS_RESIDENT"
# define MEMBRANE_INTEL_CODE_KV_PRECISION_CONTROL_UNAVAILABLE		"KV_PRECISION_CONTROL_UNAVAILABLE"
# define MEMBRANE_INTEL_CODE_KV_PLACEMENT_CONTROL_UNAVAILABLE		"KV_PLACEMENT_CONTROL_UNAVAILABLE"
# define MEMBRANE_INTEL_CODE_GPU_LAYER_CONTROL_PARTIAL				"GPU_LAYER_CONTROL_PARTIAL"
# define MEMBRANE_INTEL_CODE_CONTEXT_CONTROL_AVAILABLE				"CONTEXT_CONTROL_AVAILABLE"
# define MEMBRANE_INTEL_CODE_RUNTIME_CONTROL_LIMITED				"RUNTIME_CONTROL_LIMITED"
# define MEMBRANE_INTEL_CODE_OBSERVATION_PARTIAL					"OBSERVATION_PARTIAL"
# define MEMBRANE_INTEL_CODE_TELEMETRY_INCOMPLETE					"TELEMETRY_INCOMPLETE"

/* Recommendation codes (Part 7). Every one implemented has at least one
 * finding/capability rule wired to it in memory_intelligence.c; none are
 * a blanket "switch to native" (Part 7's own explicit prohibition). */
# define MEMBRANE_INTEL_REC_REDUCE_CONTEXT						"REDUCE_CONTEXT"
# define MEMBRANE_INTEL_REC_USE_LOWER_MEMORY_VARIANT				"USE_LOWER_MEMORY_VARIANT"
# define MEMBRANE_INTEL_REC_REDUCE_GPU_LAYERS						"REDUCE_GPU_LAYERS"
# define MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION				"USE_LOWER_KV_PRECISION"
# define MEMBRANE_INTEL_REC_CHANGE_KV_PLACEMENT					"CHANGE_KV_PLACEMENT"
# define MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL					"UNLOAD_UNUSED_MODEL"
# define MEMBRANE_INTEL_REC_SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL	"SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL"
# define MEMBRANE_INTEL_REC_NO_ACTION_NEEDED						"NO_ACTION_NEEDED"

typedef struct s_membrane_intel_finding
{
	char						code[MEMBRANE_INTEL_CODE_MAX];
	membrane_intel_severity_t	severity;
	char						summary[MEMBRANE_INTEL_SUMMARY_MAX];
	/* Part 6: the strength of evidence behind THIS finding -- copied
	 * from the snapshot field(s) it was derived from, never guessed. */
	membrane_obs_provenance_t	provenance;
	membrane_intel_dimension_t	dimension;
	char	evidence[MEMBRANE_INTEL_MAX_EVIDENCE][MEMBRANE_INTEL_EVIDENCE_MAX];
	size_t	evidence_count;
}	membrane_intel_finding_t;

/*
 * Part 8: whether/how the affected runtime can actually act on a
 * recommendation, derived ONLY from a membrane_capability_state_t
 * (runtime_capabilities.h) -- never a second guess about runtime
 * internals. ADVISORY_ONLY is the zero value (a capability that is
 * UNKNOWN maps here too: "we do not know if this runtime can do this",
 * never silently upgraded to CONTROLLABLE -- same fail-closed-default
 * convention runtime_capabilities.h's own membrane_capability_state_t
 * documents for UNKNOWN).
 */
typedef enum e_membrane_intel_applicability
{
	MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY = 0,
	MEMBRANE_INTEL_APPLICABILITY_UNAVAILABLE_ON_RUNTIME,
	MEMBRANE_INTEL_APPLICABILITY_PARTIALLY_CONTROLLABLE,
	MEMBRANE_INTEL_APPLICABILITY_CONTROLLABLE
}	membrane_intel_applicability_t;

const char	*membrane_intel_applicability_name(
				membrane_intel_applicability_t a);

typedef struct s_membrane_intel_recommendation
{
	char	code[MEMBRANE_INTEL_CODE_MAX];
	char	action[MEMBRANE_INTEL_ACTION_MAX];		/* human-readable, no
											 * fake precision, no scare
											 * language (Part 19) */
	char	reason_codes[MEMBRANE_INTEL_MAX_REASON_CODES][MEMBRANE_INTEL_CODE_MAX];
	size_t	reason_code_count;
	char	affected_runtime[MEMBRANE_RUNTIME_ID_MAX];
	/* Part 8: the exact capability this recommendation's applicability
	 * was derived from -- echoed verbatim so a consumer never has to
	 * re-derive applicability from a runtime id string. */
	membrane_capability_state_t	runtime_capability;
	membrane_intel_applicability_t	applicability;
	int		mutates_state;				/* always 0 (Part 21/23) */
}	membrane_intel_recommendation_t;

/*
 * Part 2: the top-level, read-only result. A freshly memset(0) instance
 * is NOT a valid "OK, no problems" answer -- schema_version/runtime_id
 * are both empty/0, which membrane_memory_intelligence_assemble() never
 * produces (it always sets schema_version and, when the input snapshot
 * carries one, runtime_id); a caller that sees schema_version == 0 has a
 * bug in its own control flow, not a real assessment.
 */
typedef struct s_membrane_memory_intelligence
{
	int		schema_version;
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	int		model_known;
	char	model_id[MEMBRANE_INTEL_ID_MAX];

	membrane_intel_status_t	status;

	membrane_intel_finding_t	findings[MEMBRANE_INTEL_MAX_FINDINGS];
	size_t						finding_count;

	membrane_intel_recommendation_t
				recommendations[MEMBRANE_INTEL_MAX_RECOMMENDATIONS];
	size_t		recommendation_count;

	/* Part 2: the union of every finding's own code, in the same order
	 * findings[] appears in -- a flat "why" list for a consumer that
	 * wants reason codes without walking findings[] itself. Never a
	 * second, independently-derived reason vocabulary. */
	char	reasons[MEMBRANE_INTEL_MAX_TOP_REASONS][MEMBRANE_INTEL_CODE_MAX];
	size_t	reason_count;

	int		mutates_state;			/* always 0 (Part 21/23/26) */
}	membrane_memory_intelligence_t;

/*
 * The one entry point. Pure: identical (snap, caps) always produce an
 * identical result; touches neither the machine nor either input.
 *
 * model_known/model_id: 1/the first resident model's name if the
 * snapshot reports one (membrane_obs_resident_model_t::name, whatever its
 * provenance -- runtime_reported for both native and Ollama providers),
 * else 1/the configured model if known (CONFIGURED provenance -- intent,
 * not residency, Part 11's own distinction; still worth naming), else 0.
 *
 * Status (Part 22), deterministic from findings[] alone:
 *   snap->status == UNAVAILABLE, OR every dimension this module can
 *     assess (host memory, VRAM, context, KV, residency) came back
 *     !known -> INSUFFICIENT_DATA. The only finding produced is
 *     TELEMETRY_INCOMPLETE; no recommendations are produced (Part 22:
 *     "do not fail because fields are unknown", but also never invent an
 *     assessment there is no data for).
 *   otherwise: the highest severity among findings[] --
 *     any CRITICAL -> CRITICAL; else any WARNING -> WARNING; else any
 *     NOTICE -> NOTICE; else (only INFO findings, or none) -> OK.
 *
 * caps may be NULL (equivalent to an all-UNKNOWN capability matrix --
 * every capability-derived finding/recommendation degrades to its
 * UNKNOWN-capability branch, never crashes, never fabricates SUPPORTED).
 *
 * Findings/recommendations are appended in a FIXED phase order that is
 * already the Part 17 ranking (critical measured -> warning measured ->
 * runtime-reported residency/context -> estimate-based -> capability
 * limitations -> informational) -- see memory_intelligence.c's own top
 * comment for the exact phase list. No numeric score is computed or
 * stored anywhere (Part 17's own "not a numeric opaque score" rule).
 */
membrane_intel_status_t	membrane_memory_intelligence_assemble(
		const membrane_observation_snapshot_t *snap,
		const membrane_runtime_capabilities_t *caps,
		membrane_memory_intelligence_t *out);

# ifdef __cplusplus
}
# endif

#endif
