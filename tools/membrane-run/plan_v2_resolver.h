#ifndef MEMBRANE_RUN_PLAN_V2_RESOLVER_H
# define MEMBRANE_RUN_PLAN_V2_RESOLVER_H

# include <stddef.h>
# include <stdint.h>

# include "membrane_plan.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone G2 (Planner v2 joint variant/quant optimizer). This module
 * is the ONE missing orchestration layer G1 deliberately left out (see
 * membrane_plan.h's own top comment and docs/planner-v2-foundation.md's
 * "Explicitly out of scope for G1"): reasoning about MODEL VARIANT/QUANT
 * jointly with context x GPU-layers x KV-precision x KV-placement,
 * instead of picking a variant first (variant_selector.h's pre-G2 role)
 * and only THEN handing one fixed variant to the context/joint-planner
 * pipeline.
 *
 * Still NOT a second planner: for every variant this module is given
 * real (or scaled-estimate, see plan_v2_variant_t::hparams_known below)
 * hparams for, it calls the exact same, unchanged
 * membrane_ctxrec_resolve() (context_recommender.h) + membrane_plan_
 * assemble() (membrane_plan.h) pipeline G1 already uses for ONE model --
 * this module's only new logic is running that pipeline once per
 * candidate variant and then choosing among the resulting per-variant
 * membrane_plan_t results by one small, fixed, documented policy (see
 * membrane_plan_v2_resolve()'s own doc comment below). No GPU-layer,
 * KV-precision, KV-placement, or memory-fit arithmetic is reimplemented
 * here.
 *
 * llama-free, no ggml/GGUF/device access, no I/O -- exactly like every
 * other *_policy.h/joint_planner.h/context_recommender.h module in this
 * project. Real GGUF metadata reading, catalog lookup, and the variant-
 * quality ordering this module's caller is responsible for supplying
 * (see membrane_plan_v2_variant_t's own comment on `variants[]` order)
 * all live in the C++ orchestration layer (tools/membrane/plan_cmd.cpp),
 * which already has ggml.h/model_catalog.h available -- this module
 * never touches either.
 *
 * WHY hparams are not always real (Part 5 of the G2 task): quantized
 * GGUF variants of the same model family share identical layer count/
 * embedding/head-count hparams (quantization changes per-tensor byte
 * width, never tensor shapes or counts) -- so when the caller has ONE
 * real installed sibling variant, every other catalog variant of that
 * SAME family can be evaluated through the real joint pipeline too,
 * using the installed sibling's real hparams plus that OTHER variant's
 * own real, catalog-recorded size_bytes to scale the byte-dependent
 * fields (bytes_per_layer/output_role_bytes/total_weight_bytes) -- see
 * plan_cmd.cpp's own scale_variant_hparams() for the exact, disclosed-
 * estimate formula. When no real sibling install exists at all, there
 * is no real hparams basis for ANY variant of that family, and this
 * module's caller must set hparams_known=0 for every one of them --
 * this module then falls back to the same coarse, catalog-size-based
 * host-memory-only check variant_selector.h's own membrane_select_
 * variant() already performs (never re-derived here, see coarse_fits/
 * coarse_reason_code/coarse_reason below), with NO context/GPU-layer/
 * KV-precision decision at all (disclosed via the automatic
 * MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE reason membrane_plan_
 * assemble() already pushes for a NULL ctxrec result).
 */

# define MEMBRANE_PLAN_V2_SCHEMA_VERSION	2

/* Part 13 (complexity bounds): a real catalog family in this project
 * never lists more than a handful of quants (4 for smollm2-135m-
 * instruct, the largest today) -- 8 is a generous, still-bounded
 * ceiling, never a claim that unbounded variant counts are supported.
 * Worst-case pure-arithmetic cost: MEMBRANE_PLAN_V2_MAX_VARIANTS *
 * MEMBRANE_CTXREC_MAX_CANDIDATES (20) ctxrec evaluations, each an
 * internal bounded (<= MEMBRANE_JOINT_MAX_CANDIDATES, 8) joint-planner
 * candidate search -- see docs/planner-v2-joint-variant.md's own
 * "search bound" section and test_plan_v2_resolver.c's own search-bound
 * test (Part 14, test J). */
# define MEMBRANE_PLAN_V2_MAX_VARIANTS			8

/* Part 7: a bounded shortlist, never every evaluated variant/candidate
 * dumped verbatim -- see membrane_plan_v2_resolve()'s own doc comment. */
# define MEMBRANE_PLAN_V2_MAX_ALTERNATIVES		5

/* plan-v2-variant-joint-v1: the exact deterministic selection policy
 * membrane_plan_v2_resolve() implements -- see its own doc comment.
 * Bump only if the ORDER/POLICY changes, not for unrelated additions
 * (same convention as joint_planner.h's own MEMBRANE_JOINT_POLICY_
 * VERSION). */
# define MEMBRANE_PLAN_V2_POLICY_VERSION	"plan-v2-variant-joint-v1"

typedef struct s_membrane_plan_v2_variant
{
	/* Identical contract to membrane_plan_assemble()'s own `identity`/
	 * `variant` parameters -- forwarded verbatim, one pair per
	 * candidate variant. `variant.has_variant` should be 1 with
	 * quant/estimate_only/source set (every G2 candidate IS a named
	 * quant, unlike G1's single-model path where a variant could be
	 * entirely unknown). */
	membrane_plan_identity_input_t	identity;
	membrane_plan_variant_input_t	variant;

	/* 1: run the real ctxrec/joint pipeline via ctxrec_req/ctxrec_meta
	 * below (real installed hparams, or a real-sibling-scaled
	 * estimate -- variant.estimate_only distinguishes the two).
	 * 0: no real/scaled hparams exist for this variant at all --
	 * evaluate only the coarse catalog-size host-memory proxy via
	 * coarse_fits/coarse_reason_code/coarse_reason below (Part 5's own
	 * "otherwise mark unavailable"). */
	int				hparams_known;

	/* Meaningful iff hparams_known. This module overwrites
	 * ctxrec_req.host_total_bytes/host_available_bytes/
	 * host_available_known/device_free_bytes/device_total_bytes from
	 * membrane_plan_v2_request_t's own single hardware snapshot before
	 * calling membrane_ctxrec_resolve() -- Part 9's own "identical
	 * supplied hardware snapshot" requirement is enforced HERE, not
	 * merely assumed of the caller (see test_plan_v2_resolver.c's own
	 * snapshot-consistency test, Part 14 test I). Every other field
	 * (model/candidate-context facts, explicit-override request
	 * fields) is used exactly as supplied. */
	membrane_ctxrec_request_t		ctxrec_req;
	membrane_plan_request_meta_t	ctxrec_meta;

	/* Meaningful iff !hparams_known -- already resolved by the caller
	 * via variant_selector.h's own membrane_select_variant()/
	 * host_memory_guard.h machinery (never re-derived here; Part 5/12's
	 * own "reuse existing reason codes, never a second, independently-
	 * drifting fit check" convention, identical to membrane_select_
	 * variant()'s own doc comment). coarse_reason_code/coarse_reason
	 * are only used when !coarse_fits. */
	int				coarse_fits;
	char			coarse_reason_code[40];
	char			coarse_reason[256];
}	membrane_plan_v2_variant_t;

typedef struct s_membrane_plan_v2_request
{
	/* Part 9: ONE real hardware/host-memory snapshot, applied to every
	 * variant's hparams_known ctxrec_req by this module itself (see
	 * membrane_plan_v2_variant_t's own comment above) -- the caller
	 * reads this exactly once (membrane_read_host_meminfo() +
	 * membrane_gpu_list_devices(), same as plan_cmd.cpp's existing
	 * single-model path) and never re-reads it per variant. */
	uint64_t		host_total_bytes;
	uint64_t		host_available_bytes;
	int				host_available_known;
	uint64_t		device_free_bytes;
	uint64_t		device_total_bytes;

	/* Part 8 (deterministic selection policy): candidate QUALITY order
	 * is this array's own order, highest-quality-first -- the caller is
	 * responsible for sorting variants[] by real, recorded quality
	 * (this project's own catalog `size_bytes`, descending -- a real,
	 * verified download-size figure, monotonically larger for a higher-
	 * bit-width quant of the SAME family in every quant scheme this
	 * catalog uses; never an invented/weighted score, see plan_cmd.cpp's
	 * own sort). This module treats candidate order as the ONLY quality
	 * signal it is given -- it never re-derives or second-guesses it,
	 * and ties break by this same stable input order (deterministic,
	 * no additional criterion consulted -- same convention as
	 * joint_planner.h's own tie-break). */
	membrane_plan_v2_variant_t	variants[MEMBRANE_PLAN_V2_MAX_VARIANTS];
	size_t			variant_count;
}	membrane_plan_v2_request_t;

typedef struct s_membrane_plan_v2_result
{
	int		schema_version;			/* MEMBRANE_PLAN_V2_SCHEMA_VERSION */
	char	policy_version[32];		/* MEMBRANE_PLAN_V2_POLICY_VERSION */

	/* One full membrane_plan_t per input variant, SAME order as
	 * req->variants[] (bounded to MEMBRANE_PLAN_V2_MAX_VARIANTS) --
	 * Part 7: kept for alternatives/diagnostics, never discarded. Every
	 * entry's own identity.variant/feasibility/reasons/decisions are
	 * exactly what plan_cmd.cpp already renders for a single model
	 * (Part 12: real, existing reason codes, per variant). */
	size_t			candidate_count;
	membrane_plan_t	candidates[MEMBRANE_PLAN_V2_MAX_VARIANTS];

	int		has_selected;
	int		selected_index;			/* into candidates[], -1 if !has_selected */

	/* Part 7: a bounded shortlist of OTHER feasible candidates (never
	 * selected_index itself), same tier/order policy as selection --
	 * see membrane_plan_v2_resolve()'s own doc comment. */
	size_t	alternative_count;
	int		alternative_indices[MEMBRANE_PLAN_V2_MAX_ALTERNATIVES];

	int		feasible;				/* candidates[selected_index].feasibility.
									 * feasible, or 0 if !has_selected */
	char	explanation[384];		/* derived, never invented -- see
									 * membrane_plan_v2_resolve()'s own
									 * doc comment */
}	membrane_plan_v2_result_t;

/*
 * Runs every candidate variant in req->variants[] (bounded to
 * MEMBRANE_PLAN_V2_MAX_VARIANTS, Part 13) through its designated path
 * (real/scaled joint pipeline, or coarse host-memory-only proxy -- see
 * membrane_plan_v2_variant_t's own comment) and selects ONE by this
 * fixed, deterministic policy (plan-v2-variant-joint-v1):
 *
 *   1. Hard constraints first (unchanged from every reused module):
 *      compat/architecture gating, VRAM/device capacity, host-RAM
 *      capacity, and any explicit --ctx/--kv/--gpu-layers/--quant the
 *      CALLER already baked into req->variants[]::ctxrec_req/
 *      ctxrec_meta (Part 4: this module has no override semantics of
 *      its own -- an explicit --quant is enforced simply by the caller
 *      only ever including that ONE variant in variants[]; an explicit
 *      --ctx/--kv/--gpu-layers is enforced by every variant's own
 *      ctxrec_req carrying that exact same hard constraint, never
 *      loosened per variant).
 *   2. Among variants left standing, a candidate with an ACTUAL
 *      evaluated context/GPU-layer/KV-precision/KV-placement decision
 *      (decisions.has_decisions == 1, i.e. hparams_known == 1 and the
 *      joint pipeline found it feasible) is always preferred over one
 *      with only a coarse catalog-size memory-fit estimate and no real
 *      decision (decisions.has_decisions == 0) -- Part 5's own
 *      "preserve uncertainty" principle: a plan this module can
 *      actually stand behind beats a merely-estimated-to-fit one, cost
 *      never enters this comparison.
 *   3. Within the same tier, prefer the higher-quality variant -- i.e.
 *      the FIRST feasible one in req->variants[]'s own caller-supplied
 *      quality order (see membrane_plan_v2_request_t's own comment).
 *      This module never computes or compares a quality score itself.
 *   4. Within one variant, context/GPU-layers/KV-precision/KV-placement
 *      are entirely the existing, unchanged membrane_ctxrec_resolve()/
 *      membrane_joint_plan_resolve() policy's own choice (largest
 *      feasible context, then joint-auto-v1's own GPU/KV/placement
 *      order) -- never re-decided here.
 *   5. Deterministic tie-break: input order (already the quality order,
 *      itself already deterministic) -- no random/unstable criterion is
 *      ever consulted.
 *
 * Never claims the selected candidate is "optimal", "best", or
 * "fastest" -- no timing/throughput evidence exists yet (deferred to
 * Milestone G3, see docs/planner-v2-joint-variant.md). "Selected"/
 * "preferred by policy"/"recommended"/"feasible" are the only framings
 * this module (and its callers) use.
 *
 * Deterministic: identical inputs always produce an identical result --
 * no randomness, no I/O, no clock, no re-reading any hardware/model
 * fact this module was not explicitly handed.
 *
 * Returns out->feasible (also the function's return value); out is
 * fully zeroed and out->selected_index is set to -1 even when req is
 * NULL or req->variant_count is 0 (out->feasible is then 0).
 */
int	membrane_plan_v2_resolve(const membrane_plan_v2_request_t *req,
		membrane_plan_v2_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
