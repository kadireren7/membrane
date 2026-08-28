#ifndef MEMBRANE_RUN_JOINT_PLANNER_H
# define MEMBRANE_RUN_JOINT_PLANNER_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 20: llama-free (no ggml/llama header) pure arithmetic for a
 * bounded, deterministic JOINT plan across GPU layer count + KV
 * precision, evaluated together instead of the pre-Phase-20 sequence
 * (main.cpp's resolve_gpu_config(): precision/layers resolved
 * pre-load; kv_residency_policy.h's own header comment on the
 * pre-Phase-20 flow: "Section 27: precision is decided first,
 * placement second, strictly sequential, no joint search").
 *
 * This module does NOT reimplement architecture compatibility, memory
 * budget arithmetic, Q8-vs-Q5 selection, or KV placement -- it calls
 * the existing pure modules that already own each of those
 * (compat_check.h, gpu_policy.h, adaptive_kv_policy.h,
 * kv_residency_policy.h) once per candidate, exactly as
 * resolve_gpu_config()'s pre-Phase-20 adaptive branch already did for
 * its own two (Q8/Q5) candidates -- Phase 20 generalizes that same
 * pattern to also vary GPU layer count, and folds KV-placement
 * feasibility into candidate ranking instead of a wholly separate,
 * later, non-joint step.
 *
 * KV placement caveat: this module calls membrane_kv_residency_resolve()
 * a SECOND time per candidate (main.cpp's own resolve_kv_placement()
 * still runs once more, unchanged, after model load, as the
 * AUTHORITATIVE placement decision the run actually uses -- Section 21
 * of kv-residency-policy.h: weight placement must stay identical
 * regardless of --kv-placement, and this module never touches weight
 * placement either). This module's own kv_residency_resolve() call
 * uses the PRE-LOAD estimate (n_layer/kv-bytes from gpu_device.h's
 * membrane_gpu_estimate_model(), not the post-load real model shape),
 * for candidate ranking/fit purposes only -- see docs/joint-planner.md
 * for why this is a documented, evidence-backed limitation rather than
 * a claim that pre-load and post-load placement resolution can never
 * differ.
 *
 * Weight-byte estimate correction (Phase 19 finding, Phase 20 fix):
 * the pre-Phase-20 estimate (selected_layers * bytes_per_layer) is a
 * measured, source-verified under-count -- llama.cpp treats
 * n_gpu_layers as counting the model's n_layer_all blk.N. layers PLUS
 * ONE extra "output" layer (llama_model::n_gpu_layers()'s own comment:
 * "plus 1 for the 'output' layer"), assigning GPU residency to the
 * output-role tensor(s) FIRST (before any blk.N. layer), while the
 * input token-embedding tensor is ALWAYS CPU-resident regardless of
 * n_gpu_layers (llama-model.cpp's dev_input assignment comment: "there
 * is very little benefit to offloading the input layer, so always
 * keep it on the CPU"). So for any requested V >= 1 GPU layers,
 * llama.cpp actually offloads (V-1) blk.N. layers + the output-role
 * tensor(s) -- see membrane_joint_estimate_gpu_weight_bytes() below,
 * and gpu_device.h's output_role_bytes field for exactly what "the
 * output-role tensor(s)" means for a tied- vs untied-embedding model.
 * Verified against real Phase 19 measurements
 * (results/planner-accuracy/measurements.json): this correction
 * reduces the observed-vs-estimated GPU weight-byte error from
 * +21.3%/+11.6%/+12.7% (SmolLM2-135M/360M, Qwen2.5-1.5B) down to
 * +3.0%/+1.1%/+0.2% -- see docs/joint-planner.md.
 */

/* Physical KV precision candidates. NATIVE is a legitimate candidate
 * value in this module's general API (see MEMBRANE_JOINT_PRECISION_
 * REQUEST_AUTO below for why the current CLI never actually asks this
 * module to choose between it and Q8/Q5 in the same run), matching
 * this header's own kv_store_telemetry.h-compatible numbering. */
# define MEMBRANE_JOINT_KV_NATIVE	0
# define MEMBRANE_JOINT_KV_Q8		1
# define MEMBRANE_JOINT_KV_Q5		2

/* Physical KV placement outcomes a candidate can land on -- echoes
 * membrane_kv_residency_resolve()'s own DEFAULT/GPU/CPU/AUTO constants
 * (kv_residency_policy.h), never redefines their meaning. */
# define MEMBRANE_JOINT_PLACEMENT_DEFAULT	0
# define MEMBRANE_JOINT_PLACEMENT_GPU		1
# define MEMBRANE_JOINT_PLACEMENT_CPU		2
# define MEMBRANE_JOINT_PLACEMENT_AUTO		3

/* Explicit-constraint sentinel: this dimension is NOT a hard user
 * constraint, so the planner may search it. Any other value for
 * requested_precision/requested_gpu_layers is a HARD constraint --
 * Section 8 of the Phase 20 task: explicit flags override only their
 * own dimension, never get silently overridden by the planner. */
# define MEMBRANE_JOINT_PRECISION_REQUEST_AUTO		(-1)
# define MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL		(-1)
# define MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO	(-2)

# define MEMBRANE_JOINT_MAX_CANDIDATES	8

/* Stable, machine-readable reason codes (never rename once shipped --
 * same convention as every other *_policy.h module in this project).
 * Reused where an existing module's own code already applies verbatim
 * (e.g. a candidate's fit failure reuses gpu_policy.h's own
 * MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT via reason_code,
 * these constants are for candidate-selection-level outcomes this
 * module itself is responsible for). */
# define MEMBRANE_JOINT_REASON_SELECTED				"CANDIDATE_SELECTED"
# define MEMBRANE_JOINT_REASON_INCOMPATIBLE_ARCH		"INCOMPATIBLE_KV_ARCH"
# define MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT	"GPU_MEMORY_INSUFFICIENT"
# define MEMBRANE_JOINT_REASON_EXPLICIT_PRECISION		"EXPLICIT_PRECISION_CONSTRAINT"
# define MEMBRANE_JOINT_REASON_EXPLICIT_GPU_LAYERS		"EXPLICIT_GPU_LAYER_CONSTRAINT"
# define MEMBRANE_JOINT_REASON_NO_FEASIBLE_CANDIDATE	"NO_FEASIBLE_CANDIDATE"
# define MEMBRANE_JOINT_REASON_INVALID_CONFIG			"JOINT_PLANNER_INVALID_CONFIG"

/* joint-v1: the exact lexicographic order this module ranks eligible
 * candidates in (see membrane_joint_plan_resolve()'s doc comment
 * below for the full policy). Reported in JSON as planner.
 * policy_version -- bump only if the ORDER changes, not for
 * unrelated additions. */
# define MEMBRANE_JOINT_POLICY_VERSION	"joint-auto-v1"

typedef struct s_membrane_joint_candidate
{
	int32_t		gpu_layers;					/* concrete resolved layer
											 * count for this candidate
											 * (0 == CPU-only weights) */
	int			kv_precision;				/* MEMBRANE_JOINT_KV_* */
	int			kv_placement;				/* MEMBRANE_JOINT_PLACEMENT_*
											 * -- this candidate's own
											 * placement OUTCOME (from
											 * membrane_kv_residency_
											 * resolve(), given the
											 * request's kv_placement_mode),
											 * not a separately-searched
											 * dimension -- see this
											 * header's own top comment */

	uint64_t	estimated_weight_gpu_bytes;	/* corrected formula --
											 * see membrane_joint_
											 * estimate_gpu_weight_
											 * bytes() */
	uint64_t	estimated_kv_gpu_bytes;		/* 0 if kv_placement is
											 * CPU or DEFAULT-off-GPU */
	uint64_t	estimated_total_gpu_bytes;	/* weight + kv, both
											 * GPU-resident portions
											 * only */
	uint64_t	estimated_host_kv_bytes;	/* 0 if kv_placement is
											 * GPU (fully GPU-resident) */

	uint64_t	available_gpu_bytes;		/* echoed request input, for
											 * telemetry */
	uint64_t	safety_reserve_bytes;		/* echoed from gpu_policy.h's
											 * membrane_gpu_reserve_
											 * bytes() */

	int			compatible;					/* compat_check.h result --
											 * always 1 for NATIVE */
	int			fits_gpu;					/* gpu_policy.h's own fit
											 * check passed for this
											 * exact (gpu_layers,
											 * precision) pair */
	int			fits_host;					/* kv_residency_resolve()
											 * succeeded for this
											 * candidate's placement --
											 * always 1 today (no host-
											 * RAM capacity guard exists
											 * in the product; see
											 * docs/joint-planner.md) */
	int			eligible;					/* compatible && fits_gpu &&
											 * fits_host -- only eligible
											 * candidates are ranked */

	char		reason_code[40];			/* MEMBRANE_JOINT_REASON_* on
											 * ineligibility, or the
											 * winning candidate's own
											 * outcome code (may be one
											 * of gpu_policy.h's/
											 * adaptive_kv_policy.h's own
											 * codes, echoed verbatim --
											 * Section 19: reuse existing
											 * codes, no redundant
											 * parallel names) */
}	membrane_joint_candidate_t;

typedef struct s_membrane_joint_plan_request
{
	/* Model/device facts -- all pre-load (gpu_device.h's
	 * membrane_gpu_estimate_model()). */
	int32_t		n_layer_all;
	uint64_t	bytes_per_layer;
	uint64_t	output_role_bytes;			/* see gpu_device.h */
	const char	*arch_name;					/* may be NULL/empty if the
											 * pre-load estimate's own
											 * hparams_available was 0
											 * -- every non-NATIVE
											 * candidate is then
											 * ineligible (compat_check
											 * fails closed on an
											 * unknown architecture) */
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	uint32_t	ctx_size;					/* forwarded to compat_check.h
											 * verbatim */
	uint64_t	device_free_bytes;
	uint64_t	device_total_bytes;

	/* Real per-mode KV byte totals for the requested ctx_size, computed
	 * by the CALLER via the exact same kv_bytes_for_mode()/
	 * ggml_row_size() arithmetic every explicit --kv q8/q5/native
	 * request already uses (this module stays ggml-free, like every
	 * other *_policy.h module) -- 0 for a precision this run will
	 * never need to evaluate (harmless: only consulted for candidates
	 * this module actually generates). */
	uint64_t	kv_bytes_native;
	uint64_t	kv_bytes_q8;
	uint64_t	kv_bytes_q5;

	/* Explicit-constraint inputs (Section 8): pass
	 * MEMBRANE_JOINT_PRECISION_REQUEST_AUTO / MEMBRANE_JOINT_GPU_
	 * LAYERS_REQUEST_{ALL,AUTO} for a flexible dimension the planner
	 * may search; any other value is a HARD single-candidate
	 * constraint. gpu_layers_request is never 0 here -- main.cpp's
	 * existing CPU-only short-circuit (o.gpu_layers == 0) never reaches
	 * this module at all, exactly as it never reached resolve_gpu_
	 * config()'s pre-Phase-20 GPU-estimate branches either. */
	int			precision_request;			/* MEMBRANE_JOINT_KV_* or
											 * MEMBRANE_JOINT_PRECISION_
											 * REQUEST_AUTO */
	int32_t		gpu_layers_request;			/* MEMBRANE_JOINT_GPU_LAYERS_
											 * REQUEST_{ALL,AUTO} or an
											 * explicit N >= 1 */

	int			kv_placement_mode;			/* MEMBRANE_JOINT_PLACEMENT_*
											 * -- passed to membrane_kv_
											 * residency_resolve() for
											 * every candidate unchanged
											 * (never itself searched --
											 * see this header's top
											 * comment) */

	int			want_kv_budget;				/* --kv-budget-mib was given
											 * (adaptive-only; matches
											 * adaptive_kv_policy.h's own
											 * candidate.valid input) */
	uint64_t	kv_budget_bytes;
}	membrane_joint_plan_request_t;

typedef struct s_membrane_joint_plan_result
{
	int			ok;							/* 1 iff a candidate was
											 * selected */
	membrane_joint_candidate_t	candidates[MEMBRANE_JOINT_MAX_CANDIDATES];
	int			candidate_count;			/* generated candidates,
											 * eligible or not -- kept
											 * for Phase 21's own reuse
											 * (Section 18) */
	int			selected_index;			/* into candidates[], -1 if
											 * !ok */
	char		reason_code[40];			/* MEMBRANE_JOINT_REASON_* --
											 * the winning candidate's
											 * own code if ok, or
											 * MEMBRANE_JOINT_REASON_NO_
											 * FEASIBLE_CANDIDATE
											 * otherwise */
	char		reason[384];				/* free-text detail, same
											 * "CODE: details" convention
											 * as gpu_policy.h's own
											 * result type */

	/* Populated only when precision_request was MEMBRANE_JOINT_
	 * PRECISION_REQUEST_AUTO -- the individual Q8/Q5 candidates
	 * membrane_adaptive_kv_resolve() chose between internally, echoed
	 * back for callers that report both sides of that decision (e.g.
	 * JSON diagnostics predating Phase 20), not just the winner.
	 * adaptive_evaluated is 0 (all other adaptive_* fields left at 0)
	 * for an explicit-precision request. */
	int			adaptive_evaluated;
	uint64_t	adaptive_q8_kv_bytes;
	int32_t		adaptive_q8_layers;
	int			adaptive_q8_valid;
	uint64_t	adaptive_q5_kv_bytes;
	int32_t		adaptive_q5_layers;
	int			adaptive_q5_valid;
}	membrane_joint_plan_result_t;

/* Corrected GPU weight-byte estimate for offloading `v` layers (0 <=
 * v <= n_layer_all) -- see this header's own top comment for the
 * source-verified mechanism. v == 0 is always 0 (no GPU weights at
 * all). Saturates rather than overflowing on pathological inputs
 * (checked arithmetic); callers passing v > n_layer_all get the same
 * result as v == n_layer_all (this function does not itself validate
 * v against n_layer_all -- membrane_joint_plan_resolve() never
 * generates such a candidate). */
uint64_t	membrane_joint_estimate_gpu_weight_bytes(int32_t v,
				uint64_t bytes_per_layer, uint64_t output_role_bytes);

/*
 * Generates a small, bounded candidate set (Section 7: meaningful
 * transition points, never a full cross product) and ranks the
 * eligible ones by this fixed lexicographic policy (joint-auto-v1):
 *
 *   1. Must satisfy every explicit constraint in the request (enforced
 *      by candidate GENERATION itself -- an explicit dimension never
 *      produces more than one candidate value for that dimension).
 *   2. Must be compat_check-compatible (native always is; q8/q5
 *      require the real architecture gate to pass).
 *   3. Must conservatively fit (gpu_policy.h's own budget check, AND
 *      this candidate's own kv_residency_resolve() call, both pass).
 *   4. Prefer NATIVE precision over Q8 over Q5 (quality-first) --
 *      Section 13: NEVER a performance-based preference (no product
 *      evidence supports GPU-KV/more-GPU-layers/any precision being
 *      generally faster; Phase 12G found no such advantage).
 *   5. Within the same precision, prefer MORE GPU-resident weight
 *      layers (more of the model benefits from GPU compute, still not
 *      a speed claim -- purely "more capacity used, not wasted").
 *   6. Deterministic tie-break: candidate generation order itself is
 *      already deterministic (see membrane_joint_estimate_gpu_weight_
 *      bytes()'s doc comment) -- no additional random/unstable
 *      criterion is ever consulted.
 *
 * Candidate generation, per dimension:
 *   - Precision: precision_request itself if explicit (single
 *     candidate). If MEMBRANE_JOINT_PRECISION_REQUEST_AUTO: {Q8, Q5}
 *     ONLY -- matches adaptive_kv_policy.h's own, already-shipped,
 *     already-tested "never native" contract for adaptive requests
 *     exactly (Section 15 of adaptive_kv_policy.h); this module does
 *     not change that decision, it generalizes WHICH gpu_layers count
 *     each of Q8/Q5 gets evaluated against.
 *   - GPU layers: gpu_layers_request itself if explicit ALL or an
 *     explicit N (single candidate, must fit or the whole candidate is
 *     ineligible -- never silently shrunk, matching gpu_policy.h's own
 *     existing ALL/N contract). If MEMBRANE_JOINT_GPU_LAYERS_REQUEST_
 *     AUTO: up to two candidates per precision -- gpu_policy.h's own
 *     AUTO-resolved max-fit layer count (if any layer fits) using the
 *     CORRECTED weight formula for the budget arithmetic, and 0
 *     (CPU-only weights) as an explicit fallback candidate, so a
 *     constrained-VRAM case (Section 14's Qwen2.5 regression) has a
 *     legal candidate to fall back to within the SAME precision,
 *     rather than only ever considering one GPU-layer count.
 *   - Placement: NEVER independently generated -- see this header's
 *     top comment. Each (precision, gpu_layers) candidate's placement
 *     is resolved once via membrane_kv_residency_resolve(request's
 *     kv_placement_mode, ...), using this candidate's own estimated_
 *     weight_gpu_bytes as weight_bytes_selected (Section 21 of
 *     kv_residency_policy.h: weight placement is never re-decided by
 *     that call).
 *
 * candidate_count is always <= MEMBRANE_JOINT_MAX_CANDIDATES (2
 * precisions x 2 layer counts = 4, well within the 8-slot bound).
 *
 * Deterministic: identical inputs always produce identical
 * candidates[]/selected_index -- no randomness, no I/O, no clock.
 * Returns out->ok (also the function's return value); 0 means no
 * eligible candidate existed (out->reason_code is MEMBRANE_JOINT_
 * REASON_NO_FEASIBLE_CANDIDATE, candidates[] and candidate_count are
 * still populated with every generated -- ineligible -- candidate and
 * its own reason, for diagnostics).
 */
int	membrane_joint_plan_resolve(const membrane_joint_plan_request_t *req,
		membrane_joint_plan_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
