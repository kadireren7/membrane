#ifndef MEMBRANE_RUN_CONTEXT_RECOMMENDER_H
# define MEMBRANE_RUN_CONTEXT_RECOMMENDER_H

# include <stddef.h>
# include <stdint.h>

# include "joint_planner.h"
# include "host_memory_guard.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 33: the PURE/TESTABLE CORE of MEMBRANE_JOINT context
 * recommendation ("safe context recommendation", Phase 32's chosen
 * v0.4 primary feature -- docs/v0.4-roadmap.md). No CLI surface yet
 * (--ctx auto ships in Phase 34), no file I/O, no llama model load, no
 * GPU device enumeration, no /proc read -- every fact this module
 * needs is a plain-data caller input, exactly like every other
 * *_policy.h/joint_planner.h module in this project (this header's own
 * top comment convention).
 *
 * This module does NOT reimplement GPU-layer ranking, precision
 * ranking, placement ranking, compatibility gating, or memory-fit
 * arithmetic -- it is a BOUNDED, DETERMINISTIC OUTER LOOP around the
 * existing, unchanged membrane_joint_plan_resolve() (joint_planner.h),
 * called once per candidate context size (Section 12/15 of the Phase
 * 33 task). It never becomes a second planner: it cannot select a
 * different GPU-layer count, precision, or placement than the joint
 * planner itself would have chosen for that exact candidate ctx.
 *
 * Three distinct context values (Phase 32's own required distinction,
 * docs/v0.4-roadmap.md Section 5 / this task's Section 4):
 *   MODEL_MAX_CONTEXT    -- the model's own GGUF metadata ceiling
 *                            ("{arch}.context_length", source-verified
 *                            against llama.cpp's LLM_KV_CONTEXT_LENGTH
 *                            -- see gpu_device.h's model_max_context
 *                            field).
 *   HARDWARE_FIT_CONTEXT -- the largest evaluated candidate whose plan
 *                            the existing joint planner accepted.
 *   RECOMMENDED_CONTEXT  -- what this module would present as its
 *                            conservative default. Phase 33's policy
 *                            (no empirical basis yet exists for a
 *                            smaller headroom-adjusted value -- see
 *                            this header's own POLICY_MAX_ESTIMATED_FIT
 *                            comment below and docs/context-
 *                            recommendation.md's "why recommendation is
 *                            not yet a guarantee" section) sets
 *                            RECOMMENDED_CONTEXT == HARDWARE_FIT_CONTEXT
 *                            exactly, never silently -- the result
 *                            always names which policy produced it.
 *
 * HOST-MEMORY SAFETY (Phase 34): Phase 33 disclosed, and audited
 * directly against source, that the joint planner this module reuses
 * never had a real host-RAM capacity check (joint_planner.h's own
 * candidate field comment: "fits_host... always 1 today (no host-RAM
 * capacity guard exists in the product)"), and main.cpp's only host-
 * memory signal (MEMBRANE_WARNING_HOST_MEMORY_PRESSURE) is explicitly
 * documented as "observability only... never influenced any pass/fail
 * decision." Phase 34 closes this gap FOR THIS MODULE ONLY (no CLI
 * surface exists yet -- explicit runs via main.cpp are entirely
 * unaffected) by validating every candidate's host-resident weight/KV
 * bytes against real host RAM availability via host_memory_guard.h --
 * a candidate can no longer be reported feasible on host-resident
 * memory this module never checked. See host_memory_guard.h's own top
 * comment for the reserve policy's real evidence basis, and this
 * header's host_memory_checked/host_memory_fit/host_required_bytes
 * fields below.
 *
 * RESIDUAL, STILL-DISCLOSED GAP: the new guard validates host-resident
 * WEIGHT and KV bytes only -- it does not (and, without further
 * evidence, cannot yet) validate process/backend BASELINE host
 * overhead that exists even for a fully GPU-resident plan (real
 * evidence: a 100%-GPU Vulkan run with zero host-resident weight/KV
 * bytes still showed ~248 MiB of real host RSS -- see host_memory_
 * guard.h's own top comment). membrane_ctxrec_result_t::host_memory_
 * unvalidated (kept from Phase 33) still means exactly what it always
 * did -- see its own field comment below for how it now relates to
 * the new fields. See docs/context-recommendation.md and docs/host-
 * memory-guard.md for the full disclosure.
 */

/* Section 8: llama.cpp's OWN documented minimum, not an invented
 * number -- third_party/llama.cpp/common/common.h's
 * `fit_params_min_ctx = 4096; // minimum context size to set when
 * trying to reduce memory use`. Reusing upstream's own established
 * floor is real evidence, not a guess. */
# define MEMBRANE_CTXREC_MIN_CANDIDATE	((uint64_t)4096)

/* Section 7/17: strict candidate-count bound -- doubling from
 * MEMBRANE_CTXREC_MIN_CANDIDATE covers any real model_max_context up
 * to 4096 * 2^19 (~2.1 billion tokens) within this many slots, with
 * room for the always-appended model_max_context entry; the generator
 * itself (membrane_ctxrec_generate_candidates()) never writes past
 * max_out regardless of this constant, so it is a documented default
 * bound, not the only enforcement. */
# define MEMBRANE_CTXREC_MAX_CANDIDATES	20

# define MEMBRANE_CTXREC_STATUS_OK							"OK"
# define MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN	"MODEL_MAX_CONTEXT_UNKNOWN"
# define MEMBRANE_CTXREC_STATUS_INVALID_MODEL_MAX_CONTEXT	"INVALID_MODEL_MAX_CONTEXT"
# define MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX	"MINIMUM_EXCEEDS_MODEL_MAX"
# define MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT		"NO_FEASIBLE_CONTEXT"
# define MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL		"PLANNER_REJECTED_ALL"
# define MEMBRANE_CTXREC_STATUS_INVALID_INPUT				"INVALID_INPUT"

/* Section 16: the only recommendation policy Phase 33 implements --
 * "largest evaluated candidate the existing planner accepted," with an
 * explicit, non-silent name so a future policy (B: one candidate of
 * headroom, or C: an evidence-derived reserve) is never confused with
 * this one. Never described as OOM-free (Section 16's own requirement,
 * and Section 5's transient-compute-buffer disclosure). */
# define MEMBRANE_CTXREC_POLICY_MAX_ESTIMATED_FIT	"RECOMMENDATION_POLICY_MAX_ESTIMATED_FIT"

/* Per-candidate reason when a candidate's ctx value itself cannot even
 * be forwarded to the joint planner (Section 18: ctx_size there is
 * uint32_t) -- a pathologically large model_max_context/candidate, not
 * a realistic model. Fails this ONE candidate closed rather than
 * truncating/wrapping it into a smaller, wrong ctx_size. */
# define MEMBRANE_CTXREC_REASON_CTX_EXCEEDS_UINT32	"CTX_EXCEEDS_UINT32"

/* Per-candidate reason when a caller-constructed (not generator-
 * produced) candidate falls outside [minimum_required_context,
 * model_max_context] -- defensive only; membrane_ctxrec_generate_
 * candidates() itself never produces such a value, but this module
 * must never evaluate one as if it were legal even if a caller (e.g. a
 * hand-built unit test) supplies one directly (Section 28's own
 * invariants must hold regardless of how candidates[] was built). */
# define MEMBRANE_CTXREC_REASON_OUT_OF_DECLARED_BOUNDS	"CANDIDATE_OUT_OF_DECLARED_BOUNDS"

/*
 * Pure, deterministic, checked-arithmetic candidate generator (Section
 * 7/8/9/17). Effective floor = max(MEMBRANE_CTXREC_MIN_CANDIDATE,
 * minimum_required_context); geometric doubling from there up to (and
 * always including, as the final entry) model_max_context. Writes
 * nothing and returns 0 if:
 *   - minimum_required_context > model_max_context, or
 *   - the effective floor itself exceeds model_max_context (Section 8:
 *     "model_max_context < minimum candidate").
 * Every written candidate is >= minimum_required_context and <=
 * model_max_context (property-tested, Section 28); the sequence is
 * strictly increasing; never writes past max_out; never overflows
 * (doubling stops -- and model_max_context is appended directly --
 * before any multiplication could wrap a uint64_t).
 *
 * Returns the number of candidates written (<= max_out, <=
 * MEMBRANE_CTXREC_MAX_CANDIDATES).
 */
size_t	membrane_ctxrec_generate_candidates(uint64_t model_max_context,
			uint64_t minimum_required_context, uint64_t *out_candidates,
			size_t max_out);

/* One candidate's ctx size plus the real per-mode KV byte totals for
 * that EXACT ctx -- computed by the caller via the same ggml_row_size()
 * -based arithmetic joint_planner.h's own kv_bytes_native/q8/q5 fields
 * already require of every caller (this module stays llama-free and
 * never computes these itself -- see this header's own top comment).
 * Populate via membrane_ctxrec_generate_candidates() above plus the
 * caller's own kv_bytes_for_mode()-equivalent per candidate. */
typedef struct s_membrane_ctxrec_candidate_input
{
	uint64_t	ctx;
	uint64_t	kv_bytes_native;
	uint64_t	kv_bytes_q8;
	uint64_t	kv_bytes_q5;
}	membrane_ctxrec_candidate_input_t;

typedef struct s_membrane_ctxrec_evaluated
{
	uint64_t	ctx;
	int			feasible;					/* joint planner's out->ok
											 * for this candidate */
	char		reason_code[40];			/* joint planner's own
											 * reason_code (or this
											 * header's own CTX_EXCEEDS_
											 * UINT32 if the candidate
											 * could not even be
											 * evaluated) */
	int32_t		selected_gpu_layers;		/* meaningful iff feasible */
	int			selected_kv_precision;		/* MEMBRANE_JOINT_KV_*,
											 * meaningful iff feasible */
	int			selected_kv_placement;		/* MEMBRANE_JOINT_PLACEMENT_*,
											 * meaningful iff feasible */
	int			host_resident;				/* 1 iff the joint planner
											 * accepted this candidate
											 * AND it leaves any weight
											 * or KV bytes CPU/host-
											 * resident -- independent of
											 * whether that residency
											 * then passed the Phase 34
											 * host-memory guard (see
											 * host_memory_fit below) */

	/* Phase 34: this candidate's own host-memory guard outcome.
	 * host_memory_checked is 1 whenever the guard actually ran
	 * (joint-planner-feasible candidates only -- an infeasible
	 * candidate's host bytes were never computed, matching Section 15:
	 * this module never re-decides a candidate the joint planner
	 * already rejected). host_memory_fit is only meaningful when
	 * host_memory_checked is 1; a candidate with host_memory_checked=1
	 * and host_memory_fit=0 is NEVER counted feasible (ev->feasible is
	 * forced to 0 in that case -- Section 11 of the Phase 34 task's own
	 * minimum requirement). host_required_bytes/host_reserve_bytes
	 * mirror host_memory_guard_result_t's own fields for this exact
	 * candidate. */
	int			host_memory_checked;
	int			host_memory_fit;
	uint64_t	host_required_bytes;
	uint64_t	host_reserve_bytes;
}	membrane_ctxrec_evaluated_t;

typedef struct s_membrane_ctxrec_request
{
	/* Model/device facts -- identical meaning to joint_planner.h's own
	 * request fields of the same name, forwarded verbatim to every
	 * per-candidate call (Section 12: never duplicated). */
	int32_t		n_layer_all;
	uint64_t	bytes_per_layer;
	uint64_t	output_role_bytes;
	const char	*arch_name;
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	uint64_t	device_free_bytes;
	uint64_t	device_total_bytes;

	/* Phase 34: the model's own real total tensor-byte footprint
	 * (gpu_device.h's membrane_gpu_model_estimate_t::total_bytes,
	 * every GGUF tensor summed -- NOT just the blk.N. layers
	 * bytes_per_layer already covers). Needed to derive this
	 * candidate's own host-resident weight bytes: total_weight_bytes
	 * minus a GPU-offload credit that deliberately withholds
	 * output_role_bytes (see context_recommender.c's
	 * host_weight_bytes_for() doc comment for why withholding it is
	 * the conservative, never-undercounts direction, not an invented
	 * number), checked/clamped to >= 0. 0 (the zero-initialized
	 * default) is treated the same as "unknown" would be --
	 * host_weight_bytes then comes out 0 for every candidate, which
	 * only under-counts a REAL run if the caller genuinely has this
	 * real GGUF value and simply forgot to pass it; the dev harness
	 * (context_recommender_dryrun.cpp) always passes the real value. */
	uint64_t	total_weight_bytes;

	/* Phase 34: real host memory facts -- the SAME single source
	 * main.cpp's own read_host_meminfo() already reads (/proc/meminfo,
	 * Section 9 of the Phase 34 task: reuse, never a second path).
	 * host_available_known distinguishes "read and it was genuinely
	 * this value" from "could not be read" -- see host_memory_guard.h's
	 * own fail-closed contract. */
	uint64_t	host_total_bytes;
	uint64_t	host_available_bytes;
	int			host_available_known;

	/* Explicit user constraints (Section 10/6): forwarded UNCHANGED to
	 * every candidate, so an explicit --kv/--gpu-layers/--kv-placement
	 * request stays a hard constraint of the WHOLE recommendation, not
	 * something re-decided per candidate. Explicit --device is already
	 * captured by device_free_bytes/device_total_bytes above (the
	 * caller resolves --device to a concrete device before calling this
	 * module, exactly as joint_planner.h's own existing callers do). */
	int			precision_request;
	int32_t		gpu_layers_request;
	int			kv_placement_mode;
	int			want_kv_budget;
	uint64_t	kv_budget_bytes;

	/* Section 3/4/14: the model metadata ceiling. model_max_context_
	 * known distinguishes "the GGUF key was missing/unreadable"
	 * (MODEL_MAX_CONTEXT_UNKNOWN) from "the key was present" -- a
	 * present-but-zero value is INVALID_MODEL_MAX_CONTEXT, never
	 * silently treated as unknown or as a real ceiling. */
	uint64_t	model_max_context;
	int			model_max_context_known;

	/* Section 9: every returned candidate/evaluation is >= this value.
	 * 0 means "no caller-supplied minimum" (only MEMBRANE_CTXREC_MIN_
	 * CANDIDATE itself is the effective floor). */
	uint64_t	minimum_required_context;

	/* Section 11/12: candidate_count entries, precomputed by the caller
	 * (membrane_ctxrec_generate_candidates() plus its own real
	 * per-candidate KV-byte arithmetic) -- see this header's own top
	 * comment for why KV bytes are caller-supplied. */
	membrane_ctxrec_candidate_input_t
				candidates[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		candidate_count;
}	membrane_ctxrec_request_t;

typedef struct s_membrane_ctxrec_result
{
	int			ok;							/* 1 iff status is OK */
	char		status[40];					/* MEMBRANE_CTXREC_STATUS_* */
	char		reason[384];

	uint64_t	model_max_context;
	uint64_t	minimum_required_context;
	uint64_t	hardware_fit_context;			/* 0 if !ok */
	uint64_t	recommended_context;			/* 0 if !ok */
	char		recommendation_policy[64];		/* MEMBRANE_CTXREC_POLICY_*,
											 * empty if !ok */

	size_t		candidate_count;
	size_t		evaluated_count;
	membrane_ctxrec_evaluated_t
				evaluated[MEMBRANE_CTXREC_MAX_CANDIDATES];

	int			hardware_fit_index;			/* into evaluated[], -1 if
											 * none feasible */
	membrane_joint_plan_result_t	selected_plan;	/* full joint-planner
											 * result for hardware_fit_
											 * context -- meaningful iff
											 * hardware_fit_index >= 0 */

	/* Kept from Phase 33, meaning UNCHANGED: 1 iff the selected plan
	 * leaves any weight or KV bytes CPU/host-resident -- i.e.
	 * evaluated[hardware_fit_index].host_resident, echoed here so a
	 * caller never has to re-derive it. Always 0 when !ok. As of Phase
	 * 34, that residency IS now validated against real host RAM (see
	 * host_memory_checked/host_memory_fit below) -- this field staying
	 * 1 no longer means "unchecked," only "host-resident, and process/
	 * backend baseline overhead beyond weight+KV bytes remains
	 * unvalidated" (this header's own top comment). */
	int			host_memory_unvalidated;

	/* Phase 34: echoed from evaluated[hardware_fit_index] for direct
	 * access without re-indexing -- see membrane_ctxrec_evaluated_t's
	 * own field comments above for exact meaning. All 0/false when
	 * !ok. */
	int			host_memory_checked;
	int			host_memory_fit;
	uint64_t	host_required_bytes;
	uint64_t	host_available_bytes;		/* echoed request input --
											 * identical for every
											 * candidate (host memory
											 * facts don't vary with
											 * ctx), unlike
											 * host_required_bytes */
	uint64_t	host_reserve_bytes;

	/* Section 23: enough structured information to build a human
	 * sentence without embedding prose logic in this module -- e.g.
	 * "ctx=N is the largest of K evaluated candidates with a legal
	 * plan." Not meant to be the ONLY explanation surface (Phase 34
	 * owns user-facing text), just a ready-made, deterministic one. */
	char		explanation[384];
}	membrane_ctxrec_result_t;

/*
 * Evaluates EVERY candidate in req->candidates against the existing,
 * unchanged joint planner (membrane_joint_plan_resolve(), called once
 * per candidate with that candidate's own ctx/kv_bytes_* forwarded
 * verbatim) -- Section 15: feasibility is NOT assumed monotonic across
 * context size (precision/placement/layer selection can each change
 * independently as ctx grows), so there is no early break; the full
 * bounded candidate set is always evaluated. hardware_fit_context is
 * the LARGEST feasible candidate found (ties broken by candidate order,
 * which is itself already strictly increasing). recommended_context
 * follows MEMBRANE_CTXREC_POLICY_MAX_ESTIMATED_FIT (this header's own
 * top comment) -- Phase 33 sets it equal to hardware_fit_context.
 *
 * Fails closed (out->ok = 0, hardware_fit_context/recommended_context
 * left at 0) on: NULL req/out, invalid model/device facts, an unknown
 * or invalid model_max_context, minimum_required_context exceeding
 * model_max_context, an empty candidate set, or every candidate being
 * rejected by the joint planner -- see the MEMBRANE_CTXREC_STATUS_*
 * constants above for which.
 *
 * Phase 34: a candidate the joint planner itself accepted is THEN
 * checked against host_memory_guard.h whenever it leaves any weight or
 * KV bytes host-resident -- a candidate that fails that check is
 * treated exactly like a joint-planner rejection (ineligible, not
 * selected; PLANNER_REJECTED_ALL if every candidate fails this way).
 * Section 11 of the Phase 34 task's own minimum requirement: this
 * function never returns status OK for a candidate requiring host
 * memory unless that host-memory feasibility was actually validated.
 *

 * Deterministic: identical inputs always produce an identical result
 * (no time, randomness, or I/O anywhere in this module).
 *
 * Returns out->ok (also the function's return value).
 */
int	membrane_ctxrec_resolve(const membrane_ctxrec_request_t *req,
		membrane_ctxrec_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
