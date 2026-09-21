#ifndef MEMBRANE_RUN_MEMBRANE_PLAN_H
# define MEMBRANE_RUN_MEMBRANE_PLAN_H

# include <stddef.h>
# include <stdint.h>

# include "context_recommender.h"
# include "joint_planner.h"
# include "gpu_device.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone G1 (Planner v2 foundation): ONE coherent, product-facing
 * planning result -- the common container Part 2/3 of the G1 task asks
 * for. This module is deliberately NOT a second planner: every number
 * in membrane_plan_t is either copied verbatim from an existing,
 * unchanged authoritative result (membrane_ctxrec_result_t, itself a
 * thin bounded loop around the existing, unchanged
 * membrane_joint_plan_resolve()/membrane_host_memory_guard_resolve())
 * or supplied by the caller as plain identity/variant facts this module
 * has no way to compute itself (model name, catalog metadata, variant
 * fit). membrane_plan_assemble() is a pure ADAPTER: it reads exactly
 * one already-resolved membrane_ctxrec_result_t/request (Part 8: one
 * hardware/model snapshot per invocation) and reshapes it, plus the
 * caller's identity/variant facts, into this common representation.
 *
 * Llama-free, no I/O, no /proc read, no GGUF parse, no device access --
 * same testable-without-a-model/GPU pattern as every other *_policy.h/
 * context_recommender.h module in this project (test_membrane_plan.c
 * feeds hand-built membrane_ctxrec_result_t fixtures directly).
 */

# define MEMBRANE_PLAN_SCHEMA_VERSION	1

# define MEMBRANE_PLAN_NAME_MAX		128
# define MEMBRANE_PLAN_PATH_MAX		512
# define MEMBRANE_PLAN_QUANT_MAX		32
# define MEMBRANE_PLAN_LABEL_MAX		64

/*
 * Part 7 (provenance): every controlled field in membrane_plan_t is
 * paired with one of these -- a small, explicit enum, not a generic
 * key/value provenance framework (Part 3's own "do not overengineer"
 * instruction). Kept deliberately short; add a new value only if an
 * existing one genuinely does not describe the source.
 */
typedef enum e_membrane_plan_source
{
	MEMBRANE_PLAN_SOURCE_UNKNOWN = 0,	/* not yet determined / not
										 * applicable to this field */
	MEMBRANE_PLAN_SOURCE_EXPLICIT_USER,	/* a real --ctx/--kv/--gpu-
										 * layers/--quant flag on THIS
										 * `membrane plan` invocation */
	MEMBRANE_PLAN_SOURCE_CATALOG_METADATA,	/* the built-in catalog
										 * (model_catalog.h) -- a
										 * catalog-only model's own
										 * recorded facts */
	MEMBRANE_PLAN_SOURCE_MODEL_METADATA,	/* real GGUF metadata read
										 * from the installed model file
										 * (gpu_device.h's
										 * membrane_gpu_estimate_model()) */
	MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO,	/* MEMBRANE's own automatic
										 * selection from real hardware
										 * facts (host meminfo, device
										 * enumeration) */
	MEMBRANE_PLAN_SOURCE_PLANNER_DECISION,	/* the joint planner/context
										 * recommender's own resolved
										 * value, not a direct echo of any
										 * single input */
	MEMBRANE_PLAN_SOURCE_FALLBACK_DEFAULT	/* no explicit/catalog/model/
										 * hardware signal existed; a
										 * project-wide default was used */
}	membrane_plan_source_t;

const char	*membrane_plan_source_name(membrane_plan_source_t src);

/*
 * Part 9 (explainability): a small, factual, stable reason-code
 * vocabulary for the TOP-LEVEL "why" a plan looks the way it does --
 * deliberately narrower than (and mapped FROM, never duplicating) the
 * many detailed reason codes gpu_policy.h/kv_residency_policy.h/
 * joint_planner.h/host_memory_guard.h/context_recommender.h already
 * emit; those detailed codes remain available verbatim in
 * membrane_plan_t's own echoed sub-results for a consumer that wants
 * them. Never change the meaning of an already-shipped code, only add
 * new ones (same convention as every other reason-code set in this
 * project).
 */
# define MEMBRANE_PLAN_REASON_CONTEXT_CAPPED_BY_HOST_MEMORY	"CONTEXT_CAPPED_BY_HOST_MEMORY"
# define MEMBRANE_PLAN_REASON_CONTEXT_CAPPED_BY_VRAM			"CONTEXT_CAPPED_BY_VRAM"
# define MEMBRANE_PLAN_REASON_GPU_LAYERS_CAPPED_BY_VRAM		"GPU_LAYERS_CAPPED_BY_VRAM"
# define MEMBRANE_PLAN_REASON_KV_Q8_SELECTED_FOR_HEADROOM		"KV_Q8_SELECTED_FOR_HEADROOM"
# define MEMBRANE_PLAN_REASON_KV_Q5_SELECTED_FOR_HEADROOM		"KV_Q5_SELECTED_FOR_HEADROOM"
# define MEMBRANE_PLAN_REASON_USER_OVERRIDE_PRESERVED			"USER_OVERRIDE_PRESERVED"
# define MEMBRANE_PLAN_REASON_MODEL_VARIANT_ESTIMATE_ONLY		"MODEL_VARIANT_ESTIMATE_ONLY"
# define MEMBRANE_PLAN_REASON_CONTEXT_REDUCED_FOR_FIT			"CONTEXT_REDUCED_FOR_FIT"
# define MEMBRANE_PLAN_REASON_KV_PRECISION_REDUCED_FOR_FIT		"KV_PRECISION_REDUCED_FOR_FIT"
# define MEMBRANE_PLAN_REASON_GPU_LAYERS_REDUCED_FOR_FIT		"GPU_LAYERS_REDUCED_FOR_FIT"
# define MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT				"HOST_MEMORY_LIMIT"
# define MEMBRANE_PLAN_REASON_VRAM_LIMIT						"VRAM_LIMIT"
# define MEMBRANE_PLAN_REASON_USER_FORCED_VARIANT				"USER_FORCED_VARIANT"
# define MEMBRANE_PLAN_REASON_NO_FEASIBLE_PLAN					"NO_FEASIBLE_PLAN"
# define MEMBRANE_PLAN_REASON_FULL_GPU_OFFLOAD_FEASIBLE		"FULL_GPU_OFFLOAD_FEASIBLE"
# define MEMBRANE_PLAN_REASON_CPU_ONLY_PLAN					"CPU_ONLY_PLAN"
# define MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE		"MODEL_METADATA_UNAVAILABLE"

# define MEMBRANE_PLAN_MAX_REASONS	12
# define MEMBRANE_PLAN_REASON_DETAIL_MAX	256

typedef struct s_membrane_plan_reason
{
	char	code[48];					/* one of the MEMBRANE_PLAN_REASON_*
										 * strings above */
	char	detail[MEMBRANE_PLAN_REASON_DETAIL_MAX];	/* human-readable
										 * detail, may echo an underlying
										 * subsystem's own reason string */
}	membrane_plan_reason_t;

typedef struct s_membrane_plan_identity
{
	char	model_name[MEMBRANE_PLAN_NAME_MAX];	/* registry name or
										 * catalog family name, always set */
	char	model_path[MEMBRANE_PLAN_PATH_MAX];	/* empty iff !installed */
	int		installed;					/* 1 = real local GGUF file,
										 * 0 = catalog metadata only */
	char	arch_name[MEMBRANE_GPU_ARCH_NAME_MAX];
	int		arch_known;
	char	display_name[MEMBRANE_PLAN_NAME_MAX];	/* catalog display name,
										 * may equal model_name */
	char	parameter_count[MEMBRANE_PLAN_LABEL_MAX];	/* catalog label,
										 * e.g. "135M" -- empty if unknown */
	char	variant[MEMBRANE_PLAN_QUANT_MAX];	/* quant string, e.g.
										 * "Q4_K_M" -- empty if unknown */
	int		variant_known;
	membrane_plan_source_t	variant_source;
	int		variant_estimate_only;		/* 1 = variant fit was computed
										 * from catalog size_bytes as a
										 * proxy (pre-download), never a
										 * real GGUF measurement -- see
										 * variant_selector.h's own top
										 * comment */
}	membrane_plan_identity_t;

typedef struct s_membrane_plan_workload
{
	uint64_t	requested_context;		/* 0 if auto/unspecified */
	membrane_plan_source_t	context_source;
	int			precision_request;		/* MEMBRANE_JOINT_KV_* or
										 * MEMBRANE_JOINT_PRECISION_
										 * REQUEST_AUTO */
	membrane_plan_source_t	precision_source;
	int32_t		gpu_layers_request;		/* MEMBRANE_JOINT_GPU_LAYERS_
										 * REQUEST_{ALL,AUTO} or explicit N */
	membrane_plan_source_t	gpu_layers_source;
	int			kv_placement_mode;		/* MEMBRANE_JOINT_PLACEMENT_* */
	membrane_plan_source_t	kv_placement_source;
}	membrane_plan_workload_t;

typedef struct s_membrane_plan_hardware
{
	uint64_t	host_total_bytes;
	uint64_t	host_available_bytes;
	int			host_available_known;
	uint64_t	host_reserve_bytes;

	int			device_known;			/* 0 = CPU-only / no GPU backend */
	char		backend[MEMBRANE_GPU_DEVICE_NAME_MAX];
	char		device_name[MEMBRANE_GPU_DEVICE_NAME_MAX];
	uint64_t	device_total_bytes;
	uint64_t	device_free_bytes;
}	membrane_plan_hardware_t;

typedef struct s_membrane_plan_decisions
{
	int			has_decisions;			/* 0 iff no candidate was ever
										 * selected (feasibility.feasible
										 * is then also 0) -- every field
										 * below is meaningful only when
										 * this is 1 */
	uint64_t	context;
	membrane_plan_source_t	context_source;
	int32_t		gpu_layers;
	membrane_plan_source_t	gpu_layers_source;
	int			kv_precision;			/* MEMBRANE_JOINT_KV_* */
	membrane_plan_source_t	kv_precision_source;
	int			kv_placement;			/* MEMBRANE_JOINT_PLACEMENT_* */
	membrane_plan_source_t	kv_placement_source;
}	membrane_plan_decisions_t;

typedef struct s_membrane_plan_feasibility
{
	int			feasible;
	char		limiting_resource[48];	/* MEMBRANE_PLAN_REASON_* or ""
										 * when feasible */
	uint64_t	max_feasible_context;	/* largest context this exact
										 * invocation's candidate search
										 * found feasible -- 0 if unknown/
										 * not established (never invented,
										 * Part 10) */
	int			max_feasible_context_known;
	uint64_t	host_required_bytes;
	uint64_t	host_headroom_bytes;	/* available - reserve - required,
										 * clamped to 0 -- 0 if unknown */
	int			host_headroom_known;
}	membrane_plan_feasibility_t;

typedef struct s_membrane_plan
{
	int		schema_version;				/* MEMBRANE_PLAN_SCHEMA_VERSION */
	membrane_plan_identity_t		identity;
	membrane_plan_workload_t		workload;
	membrane_plan_hardware_t		hardware;
	membrane_plan_decisions_t		decisions;
	membrane_plan_feasibility_t	feasibility;

	membrane_plan_reason_t	reasons[MEMBRANE_PLAN_MAX_REASONS];
	size_t					reason_count;

	char	explanation[384];			/* short, derived (never invented)
										 * human summary -- the same
										 * ctxrec_result_t::explanation
										 * this module's caller supplied,
										 * or a locally-derived equivalent
										 * for the explicit-context path */

	/* Full underlying results, echoed verbatim for a consumer that wants
	 * the detailed, already-established reason-code taxonomy this
	 * common representation deliberately does not replace (Part 3: this
	 * is a container, not a second planner). Meaningful iff
	 * has_ctxrec_result. */
	int							has_ctxrec_result;
	membrane_ctxrec_result_t	ctxrec_result;
}	membrane_plan_t;

/* Caller-supplied identity facts membrane_plan_assemble() has no way to
 * derive itself (model name/path/arch, catalog display metadata) -- see
 * this header's own top comment. */
typedef struct s_membrane_plan_identity_input
{
	const char	*model_name;
	const char	*model_path;			/* NULL/empty if !installed */
	int			installed;
	const char	*arch_name;				/* NULL/empty if unknown */
	int			arch_known;
	const char	*display_name;			/* NULL/empty -> model_name used */
	const char	*parameter_count;		/* NULL/empty if unknown */
}	membrane_plan_identity_input_t;

/* Caller-supplied variant facts (Part 4/9's MODEL_VARIANT_ESTIMATE_ONLY) --
 * see variant_selector.h. has_variant is 0 for an architecture with no
 * separate quant dimension exposed by this planning stage. */
typedef struct s_membrane_plan_variant_input
{
	int			has_variant;
	const char	*quant;
	int			estimate_only;
	membrane_plan_source_t	source;		/* EXPLICIT_USER or HARDWARE_AUTO */
}	membrane_plan_variant_input_t;

/* Part 7: explicit-vs-auto sourcing for the 4 workload dimensions this
 * exact `membrane plan` invocation asked for -- mirrors membrane_ctxrec_
 * request_t's own request fields, forwarded unchanged by the caller
 * (context_recommender.h's own "explicit constraints are forwarded to
 * every candidate" contract; this struct only carries WHERE each value
 * came from, never a second copy of the value itself beyond
 * requested_context, which context_recommender.h has no dedicated field
 * for when it is exactly one caller-built single-candidate request --
 * see plan_cmd.cpp's own explicit-context adapter). */
typedef struct s_membrane_plan_request_meta
{
	uint64_t	requested_context;		/* 0 if auto */
	membrane_plan_source_t	context_source;
	membrane_plan_source_t	precision_source;
	membrane_plan_source_t	gpu_layers_source;
	membrane_plan_source_t	kv_placement_source;
}	membrane_plan_request_meta_t;

/* Pure passthrough display facts (gpu_device.h's own enumeration output,
 * already resolved by the caller) -- this module never calls gpu_device.h
 * itself, kept llama-free. NULL/empty backend/device_name are treated
 * as "no GPU device" (matches device_known below being driven off
 * req->device_total_bytes instead, so an inconsistent caller can never
 * make this module claim a device exists with no name). */
typedef struct s_membrane_plan_hardware_input
{
	const char	*backend;
	const char	*device_name;
}	membrane_plan_hardware_input_t;

/*
 * Assembles the common plan representation from one already-resolved
 * membrane_ctxrec_result_t/request pair (Part 8: exactly one hardware
 * snapshot, already baked into req by the caller) plus caller-supplied
 * identity/variant facts this module cannot compute itself.
 *
 * Never re-derives or second-guesses rec/req -- every planner-owned
 * field (decisions, feasibility, hardware snapshot, host-memory
 * numbers) is copied or trivially mapped from them. out->reasons[] is
 * populated by mapping rec's own status/evaluated[]::reason_code (and,
 * for a variant estimate, variant->estimate_only) onto this header's
 * small MEMBRANE_PLAN_REASON_* vocabulary -- never a fabricated
 * explanation.
 *
 * req may be NULL only when rec is also NULL (no ctxrec pipeline ran at
 * all -- e.g. a catalog-only model with no installed GGUF to plan
 * context/GPU/KV against, Part 4's own disclosure requirement); in that
 * case out->has_ctxrec_result is 0, out->decisions.has_decisions is 0,
 * and out->feasibility.feasible is 0 with limiting_resource
 * MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE -- identity/variant
 * fields are still populated normally.
 *
 * Deterministic: identical inputs always produce an identical result.
 * Returns out->feasibility.feasible (also the function's return value);
 * always returns 1 for a catalog-only/no-ctxrec call, since "no plan
 * was attempted" is not itself an infeasibility (the caller/CLI layer
 * is responsible for disclosing that distinction in its own output).
 */
int	membrane_plan_assemble(const membrane_ctxrec_result_t *rec,
		const membrane_ctxrec_request_t *req,
		const membrane_plan_request_meta_t *meta,
		const membrane_plan_identity_input_t *identity,
		const membrane_plan_variant_input_t *variant,
		const membrane_plan_hardware_input_t *hardware,
		membrane_plan_t *out);

# ifdef __cplusplus
}
# endif

#endif
