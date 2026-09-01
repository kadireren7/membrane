#include <stdint.h>
#include <string.h>

#include "context_recommender.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

/* Synthetic (Section 27: "no model download") per-token KV byte costs,
 * illustrative only -- native > q8 > q5, matching every real KV mode's
 * actual ordering, same convention as test_joint_planner.c's own
 * fill_base_request(). Scales with ctx so larger candidates genuinely
 * cost more memory, letting tests exercise real fit/no-fit transitions
 * across the candidate set instead of every candidate being trivially
 * identical. */
static uint64_t	kv_native(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 256);
}

static uint64_t	kv_q8(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 128);
}

static uint64_t	kv_q5(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 96);
}

/* n_layer_all=10, bytes_per_layer=10 MiB, output_role_bytes=20 MiB,
 * real llama-compatible hparams (head_dim 64, divisible by
 * compat_check.h's 32-byte block size) -- same shape as
 * test_joint_planner.c's own fill_base_request() so this module's
 * tests can be sanity-checked against that file's already-verified
 * hand derivations. model_max_context=32768 with the real llama.cpp
 * fit_params_min_ctx=4096 floor gives candidates [4096, 8192, 16384,
 * 32768] (hand-verified below). */
static void	fill_base_request(membrane_ctxrec_request_t *req)
{
	memset(req, 0, sizeof(*req));
	req->n_layer_all = 10;
	req->bytes_per_layer = 10 * MIB;
	req->output_role_bytes = 20 * MIB;
	req->arch_name = "llama";
	req->n_embd = 512;
	req->n_head = 8;
	req->n_head_kv = 8;
	req->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	req->model_max_context = 32768;
	req->model_max_context_known = 1;
}

/* Generates the real candidate set via membrane_ctxrec_generate_
 * candidates() (never hand-listed -- Section 12: the generator itself
 * is under test elsewhere, this helper just wires its real output into
 * a request) and fills each candidate's KV bytes via the synthetic
 * formulas above. */
static size_t	populate_candidates(membrane_ctxrec_request_t *req)
{
	uint64_t	ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n;
	size_t		i;

	n = membrane_ctxrec_generate_candidates(req->model_max_context,
			req->minimum_required_context, ctxs,
			MEMBRANE_CTXREC_MAX_CANDIDATES);
	for (i = 0; i < n; ++i)
	{
		req->candidates[i].ctx = ctxs[i];
		req->candidates[i].kv_bytes_native = kv_native(ctxs[i],
				req->n_layer_all);
		req->candidates[i].kv_bytes_q8 = kv_q8(ctxs[i], req->n_layer_all);
		req->candidates[i].kv_bytes_q5 = kv_q5(ctxs[i], req->n_layer_all);
	}
	req->candidate_count = n;
	return (n);
}

/* ------------------------------------------------------------------ */
/* membrane_ctxrec_generate_candidates()                              */
/* ------------------------------------------------------------------ */

static void	test_gen_basic_doubling_matches_hand_derivation(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(32768, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 4, "4096/8192/16384/32768 -- 4 candidates");
	TEST_ASSERT(out[0] == 4096 && out[1] == 8192 && out[2] == 16384
		&& out[3] == 32768, "exact doubling sequence, model max last");
}

static void	test_gen_real_smollm2_qwen25_value(void)
{
	/* Real GGUF metadata value (this phase's own Section 29 smoke,
	 * verified directly against SmolLM2-135M/360M-Instruct-f16.gguf and
	 * qwen2.5-1.5b-instruct-fp16.gguf's own llama.context_length /
	 * qwen2.context_length keys). */
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(8192, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 2, "floor 4096 doubles once to reach model max 8192");
	TEST_ASSERT(out[0] == 4096 && out[1] == 8192, "[4096, 8192]");
}

static void	test_gen_floor_equals_model_max_single_candidate(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(4096, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 1 && out[0] == 4096, "single candidate == model max");
}

static void	test_gen_non_power_of_two_model_max(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(6000, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 2 && out[0] == 4096 && out[1] == 6000,
		"model max appended verbatim even when not a power of two");
}

/* Section 8/17: model_max_context below the minimum candidate -- the
 * real stories15M.gguf fixture's own llama.context_length (128) is
 * exactly this case (checked directly this phase, not synthesized). */
static void	test_gen_model_max_below_floor_returns_zero(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(128, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 0, "real stories15M-shaped model max (128) < floor "
		"(4096) yields zero candidates, not a fabricated one");
}

static void	test_gen_minimum_exceeds_model_max_returns_zero(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(4096, 8192, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 0, "minimum_required_context > model_max_context "
		"yields zero candidates");
}

static void	test_gen_minimum_raises_floor(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(32768, 5000, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);

	TEST_ASSERT(n == 4, "floor raised to 5000 (> llama.cpp's own 4096) "
		"-- [5000, 10000, 20000, 32768]");
	TEST_ASSERT(out[0] == 5000, "effective floor is the caller minimum, "
		"not MEMBRANE_CTXREC_MIN_CANDIDATE, when it is larger");
	TEST_ASSERT(out[1] == 10000 && out[2] == 20000,
		"doubling continues from the raised floor");
	TEST_ASSERT(out[n - 1] == 32768, "model max always the final entry");
}

static void	test_gen_respects_small_max_out(void)
{
	uint64_t	out[1];
	size_t		n = membrane_ctxrec_generate_candidates(32768, 0, out, 1);

	TEST_ASSERT(n == 1, "a 1-slot buffer yields exactly 1 candidate");
	TEST_ASSERT(out[0] == 32768, "the single slot holds model max, the "
		"most informative single value");
}

static void	test_gen_huge_model_max_bounded_and_no_overflow(void)
{
	uint64_t	out[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n = membrane_ctxrec_generate_candidates(UINT64_MAX, 0, out,
			MEMBRANE_CTXREC_MAX_CANDIDATES);
	size_t		i;

	TEST_ASSERT(n > 0 && n <= MEMBRANE_CTXREC_MAX_CANDIDATES,
		"pathological model max still yields a bounded candidate count");
	TEST_ASSERT(out[n - 1] == UINT64_MAX, "model max is always the last "
		"entry, even at UINT64_MAX");
	for (i = 1; i < n; ++i)
		TEST_ASSERT(out[i] > out[i - 1], "strictly increasing throughout");
}

static void	test_gen_null_out_is_safe(void)
{
	TEST_ASSERT(membrane_ctxrec_generate_candidates(32768, 0, NULL, 10)
		== 0, "NULL output buffer never crashes, just yields 0");
}

/* ------------------------------------------------------------------ */
/* membrane_ctxrec_resolve() -- Section 27 test matrix                */
/* ------------------------------------------------------------------ */

/* 1/2/3. Ample memory: every candidate feasible, largest wins, model
 * max itself is the winning candidate. */
static void	test_ample_memory_multiple_feasible_picks_largest(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(strcmp(res.status, MEMBRANE_CTXREC_STATUS_OK) == 0,
		"status OK");
	TEST_ASSERT(res.evaluated_count == 4, "all 4 candidates evaluated");
	TEST_ASSERT(res.hardware_fit_context == 32768,
		"largest candidate (== model max) is the hardware fit");
	TEST_ASSERT(res.hardware_fit_context == req.model_max_context,
		"largest model context is included exactly and wins here");
	TEST_ASSERT(res.recommended_context == res.hardware_fit_context,
		"Phase 33 policy: recommended == hardware fit exactly");
	TEST_ASSERT(strcmp(res.recommendation_policy,
			MEMBRANE_CTXREC_POLICY_MAX_ESTIMATED_FIT) == 0,
		"policy is named explicitly, never silent");
	TEST_ASSERT(res.selected_plan.ok == 1, "selected plan is feasible");
}

/* 4. Non-power-of-two model max, real end-to-end resolve. */
static void	test_non_power_of_two_model_max_resolves(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context = 6000;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(res.hardware_fit_context == 6000,
		"non-power-of-two model max itself is reachable and wins");
}

/* 5a. No candidates at all (model max below the floor) -->
 * NO_FEASIBLE_CONTEXT, not a fabricated recommendation. */
static void	test_no_candidates_at_all_is_no_feasible_context(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context = 128;	/* real stories15M-shaped value */
	req.device_free_bytes = 4 * GIB;
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(populate_candidates(&req) == 0, "generator itself yields "
		"zero candidates for this model max");
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT) == 0,
		"NO_FEASIBLE_CONTEXT, no fabricated recommendation");
	TEST_ASSERT(res.recommended_context == 0 && res.hardware_fit_context == 0,
		"never a nonzero recommendation when status is not OK");
}

/* 5b/16. Every candidate evaluated but every one rejected (unsupported
 * architecture -- compat_check fails closed for every candidate
 * regardless of memory) --> PLANNER_REJECTED_ALL, distinct from
 * NO_FEASIBLE_CONTEXT above (candidates existed and were evaluated). */
static void	test_unsupported_architecture_is_planner_rejected_all(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.arch_name = "not_a_real_architecture";
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(populate_candidates(&req) == 4, "candidates were generated");
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL) == 0,
		"PLANNER_REJECTED_ALL -- candidates existed but none passed "
		"compatibility, never bypassed (Section 21)");
	TEST_ASSERT(res.evaluated_count == 4, "every candidate was still "
		"evaluated, not stopped early");
}

/* 6. minimum_required_context filters out smaller candidates. */
static void	test_minimum_required_context_filters_smaller_candidates(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	size_t						i;

	fill_base_request(&req);
	req.minimum_required_context = 5000;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	for (i = 0; i < req.candidate_count; ++i)
		TEST_ASSERT(req.candidates[i].ctx >= 5000,
			"generator never produces a candidate below the minimum");
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(res.recommended_context >= req.minimum_required_context,
		"recommendation respects the caller's minimum");
}

/* 7. minimum_required_context > model_max_context fails clearly. */
static void	test_minimum_exceeds_model_max(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.minimum_required_context = 65536;	/* > model_max 32768 */
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX) == 0,
		"MINIMUM_EXCEEDS_MODEL_MAX, checked before any candidate work");
}

/* 8. Missing model_max_context --> MODEL_MAX_CONTEXT_UNKNOWN, never a
 * guessed ceiling. */
static void	test_missing_model_max_is_unknown(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context_known = 0;
	req.model_max_context = 0;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN) == 0,
		"MODEL_MAX_CONTEXT_UNKNOWN when the key was never read");
}

/* 8b. Present-but-zero value is a DIFFERENT status than missing --
 * never conflated (Section 14/17). */
static void	test_zero_model_max_is_invalid_not_unknown(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context_known = 1;
	req.model_max_context = 0;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_INVALID_MODEL_MAX_CONTEXT) == 0,
		"INVALID_MODEL_MAX_CONTEXT, distinct from UNKNOWN");
}

/* 9. CPU-only (no GPU device at all), EXPLICIT precision: the joint
 * planner's resolve_single_precision() AUTO-gpu-layers path always
 * includes a gpu_layers=0 fallback candidate (joint_planner.c), so an
 * explicit --kv request genuinely resolves to a legal CPU-only plan --
 * this module forwards that honestly, and flags host_memory_unvalidated
 * because nothing in the existing planner stack has ever checked real
 * host RAM capacity (a pre-existing, disclosed product gap -- see
 * context_recommender.h's own top comment and docs/context-
 * recommendation.md).
 *
 * Real, source-verified finding this phase (documented in docs/
 * context-recommendation.md, not silently worked around): an ADAPTIVE/
 * bare-`--auto`-precision request combined with AUTO gpu-layers at
 * zero GPU budget does NOT reach a CPU-only fallback in the existing,
 * unchanged joint planner -- joint_planner.c always calls
 * membrane_adaptive_kv_resolve() with is_gpu_backend hardcoded to 1,
 * and its own resolve_gpu()/resolve_gpu_partial() return ar.ok=0 (no
 * candidate at all, GPU_MEMORY_INSUFFICIENT) rather than falling back
 * to CPU-only, when neither Q8 nor Q5 reaches even one GPU layer. This
 * module reuses the joint planner exactly as it exists (Section 12 of
 * the Phase 33 task forbids changing its ranking) -- it does not, and
 * must not, paper over this with its own fallback logic. A genuinely
 * GPU-less host is expected to reach this module with an EXPLICIT
 * precision request instead (matching main.cpp's own real gpu_layers==0
 * short-circuit, which never invokes the joint planner with an implicit
 * adaptive AUTO request at all). */
static void	test_cpu_only_explicit_precision_resolves(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.device_free_bytes = 0;
	req.device_total_bytes = 0;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1,
		"CPU-only with explicit precision still resolves -- no fake "
		"GPU plan is fabricated");
	TEST_ASSERT(res.selected_plan.candidates[res.selected_plan.selected_index]
			.gpu_layers == 0, "resolved plan is genuinely CPU-only");
	TEST_ASSERT(res.host_memory_unvalidated == 1, "honestly flagged: real "
		"host RAM was never checked for this plan");
}

/* 9b. The real, disclosed gap above, made explicit as its own test:
 * ADAPTIVE precision + AUTO gpu-layers + zero GPU budget does NOT
 * resolve today (existing joint planner behavior, unchanged). */
static void	test_cpu_only_adaptive_precision_does_not_resolve_today(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = 0;
	req.device_total_bytes = 0;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0,
		"documented existing-planner gap: adaptive precision has no "
		"CPU-only fallback at zero GPU budget");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL) == 0,
		"fails closed via PLANNER_REJECTED_ALL, never a fabricated plan");
}

/* 10. GPU full residency (ample VRAM) -- already covered by
 * test_ample_memory_multiple_feasible_picks_largest above; this test
 * asserts the specific selected-plan shape (Section 27 item 10). */
static void	test_gpu_full_residency_selected_shape(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	const membrane_joint_candidate_t	*win;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	win = &res.selected_plan.candidates[res.selected_plan.selected_index];
	TEST_ASSERT(win->gpu_layers == req.n_layer_all,
		"full residency: every layer selected");
	TEST_ASSERT(res.host_memory_unvalidated == 0, "a fully GPU-resident "
		"plan has no host-resident bytes to flag");
}

/* 11. CPU KV spill: constrained VRAM forces weights onto the GPU but KV
 * onto the CPU (kv_placement_mode explicit CPU) -- host_memory_
 * unvalidated must still be honestly set (KV bytes are host-resident
 * even though weights are not). */
static void	test_cpu_kv_spill_flags_host_memory_unvalidated(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_CPU;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(res.host_memory_unvalidated == 1,
		"explicit CPU KV placement is host-resident -- flagged honestly");
}

/* 12. Adaptive precision transitions across the candidate set (larger
 * ctx needs more KV bytes, which can change the Q8-vs-Q5 tradeoff) --
 * this test only asserts the recommender forwards whatever the
 * (unchanged) joint planner decided, never a decision of its own. */
static void	test_adaptive_precision_forwarded_from_joint_planner(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	const membrane_joint_candidate_t	*win;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	win = &res.selected_plan.candidates[res.selected_plan.selected_index];
	TEST_ASSERT(res.evaluated[res.hardware_fit_index].selected_kv_precision
			== win->kv_precision, "recommender's own record matches the "
		"joint planner's real winning candidate exactly");
}

/* 13/14. Explicit --kv q8 / --kv q5 stay hard constraints across every
 * candidate (Section 10) -- never silently promoted to adaptive. */
static void	test_explicit_q8_is_hard_constraint_every_candidate(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	size_t						i;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	for (i = 0; i < res.evaluated_count; ++i)
		if (res.evaluated[i].feasible)
			TEST_ASSERT(res.evaluated[i].selected_kv_precision
					== MEMBRANE_JOINT_KV_Q8,
				"explicit q8 never varies across candidates");
}

static void	test_explicit_q5_is_hard_constraint_every_candidate(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	size_t						i;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_Q5;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	for (i = 0; i < res.evaluated_count; ++i)
		if (res.evaluated[i].feasible)
			TEST_ASSERT(res.evaluated[i].selected_kv_placement >= 0
					&& res.evaluated[i].selected_kv_precision
						== MEMBRANE_JOINT_KV_Q5,
				"explicit q5 never varies across candidates");
}

/* 15. Explicit GPU placement stays a hard constraint too. */
static void	test_explicit_gpu_placement_is_hard_constraint(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_GPU;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(res.evaluated[res.hardware_fit_index].selected_kv_placement
			== MEMBRANE_JOINT_PLACEMENT_GPU,
		"explicit GPU placement request is honored, never silently "
		"overridden");
}

/* 17. Malformed shape: n_embd not evenly divisible by n_head --
 * compat_check.c's own real rejection, never bypassed. */
static void	test_malformed_head_shape_rejected(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.n_embd = 100;
	req.n_head = 8;	/* 100 % 8 != 0 */
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0,
		"malformed shape fails closed, never bypassed by this module");
	TEST_ASSERT(strcmp(res.status,
			MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL) == 0,
		"the existing compat_check rejection surfaces as PLANNER_"
		"REJECTED_ALL, not a silently-different outcome");
}

/* 18. Deterministic repeat: identical inputs, byte-for-byte identical
 * logical result (compared field-by-field, not memcmp -- the struct
 * has no padding-sensitive comparison requirement, but this is more
 * robust regardless). */
static void	test_deterministic_repeat(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res1;
	membrane_ctxrec_result_t	res2;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res1) == 1, "first resolve");
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res2) == 1, "second resolve");
	TEST_ASSERT(res1.hardware_fit_context == res2.hardware_fit_context,
		"deterministic hardware fit");
	TEST_ASSERT(res1.recommended_context == res2.recommended_context,
		"deterministic recommendation");
	TEST_ASSERT(res1.evaluated_count == res2.evaluated_count
		&& memcmp(res1.evaluated, res2.evaluated,
			sizeof(res1.evaluated[0]) * res1.evaluated_count) == 0,
		"per-candidate evaluation is byte-for-byte identical");
}

/* 19. A hand-built candidate ctx that overflows joint_planner's own
 * uint32_t ctx_size -- fails that ONE candidate closed, never wraps/
 * truncates into a smaller, wrong value, and never crashes. */
static void	test_ctx_overflow_candidate_fails_closed(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context = (uint64_t)UINT32_MAX + 4096;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	req.candidate_count = 1;
	req.candidates[0].ctx = (uint64_t)UINT32_MAX + 1;
	req.candidates[0].kv_bytes_native = 1 * MIB;
	req.candidates[0].kv_bytes_q8 = 1 * MIB;
	req.candidates[0].kv_bytes_q5 = 1 * MIB;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0,
		"the only candidate overflows uint32_t -- no crash, fails closed");
	TEST_ASSERT(res.evaluated[0].feasible == 0, "marked infeasible");
	TEST_ASSERT(strcmp(res.evaluated[0].reason_code,
			MEMBRANE_CTXREC_REASON_CTX_EXCEEDS_UINT32) == 0,
		"honest, specific reason -- never silently truncated");
}

/* 20. Candidate-count bound: even a maximal, real doubling sequence
 * never exceeds MEMBRANE_CTXREC_MAX_CANDIDATES (already exercised by
 * test_gen_huge_model_max_bounded_and_no_overflow above for the
 * generator itself; this asserts resolve() propagates the same bound
 * into evaluated_count/candidate_count). */
static void	test_candidate_count_bound_propagates_to_result(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.model_max_context = UINT64_MAX;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(req.candidate_count <= MEMBRANE_CTXREC_MAX_CANDIDATES,
		"generator bound honored");
	membrane_ctxrec_resolve(&req, &res);
	TEST_ASSERT(res.evaluated_count <= MEMBRANE_CTXREC_MAX_CANDIDATES,
		"result never exceeds the same bound");
}

/* Explicit GPU-layers hard constraint (Section 10), forwarded
 * unchanged across every candidate ctx. */
static void	test_explicit_gpu_layers_hard_constraint(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	size_t						i;

	fill_base_request(&req);
	req.gpu_layers_request = 5;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	for (i = 0; i < res.evaluated_count; ++i)
		if (res.evaluated[i].feasible)
			TEST_ASSERT(res.evaluated[i].selected_gpu_layers == 5,
				"explicit gpu-layers=5 never re-decided per candidate");
}

/* Section 15: no unsafe early break -- a smaller candidate that is
 * infeasible followed by a larger, feasible one (hand-built, since a
 * real monotonic-in-ctx KV formula makes this hard to reproduce
 * naturally, exactly as this task's own Section 15 anticipates) must
 * still be found; a "stop at first infeasible" implementation would
 * incorrectly report NO/PLANNER_REJECTED here. */
static void	test_no_monotonicity_assumption(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	/* Explicit native precision + explicit gpu_layers=ALL (10): a
	 * single, hard fit check per candidate (weight_bytes + full KV
	 * reservation <= budget, DEFAULT placement reserves the whole KV
	 * estimate -- kv_residency_policy.c's own membrane_kv_policy_
	 * preflight_reservation()) with NO CPU-only fallback candidate to
	 * mask the result -- exactly what's needed to probe monotonicity
	 * cleanly. budget = 1 GiB free - max(15%*2GiB, 512 MiB) reserve =
	 * 1024 MiB - 512 MiB = 512 MiB; weight_bytes for all 10 layers =
	 * 9*10 MiB + 20 MiB = 110 MiB (hand-derived, same formula
	 * test_joint_planner.c's own tests already verify). */
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.gpu_layers_request = 10;
	req.device_free_bytes = 1 * GIB;
	req.device_total_bytes = 2 * GIB;
	req.candidate_count = 2;
	/* Candidate 0 (smaller ctx=4096): 110 MiB weight + 900 MiB KV =
	 * 1010 MiB > 512 MiB budget -- does NOT fit. Deliberately oversized
	 * KV relative to a real formula's output at this ctx -- unrealistic
	 * as physics, valid as an algorithm-behavior probe (this module
	 * only trusts caller-supplied bytes, never enforces physical
	 * realism -- that is the caller's job, exactly like every other
	 * joint_planner.h caller). */
	req.candidates[0].ctx = 4096;
	req.candidates[0].kv_bytes_native = 900 * MIB;
	req.candidates[0].kv_bytes_q8 = 900 * MIB;
	req.candidates[0].kv_bytes_q5 = 900 * MIB;
	/* Candidate 1 (larger ctx=8192): 110 MiB weight + 1 MiB KV = 111
	 * MiB <= 512 MiB budget -- fits easily. */
	req.candidates[1].ctx = 8192;
	req.candidates[1].kv_bytes_native = 1 * MIB;
	req.candidates[1].kv_bytes_q8 = 1 * MIB;
	req.candidates[1].kv_bytes_q5 = 1 * MIB;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1,
		"the larger, later candidate is still found -- no early break");
	TEST_ASSERT(res.hardware_fit_context == 8192,
		"correctly selects the only feasible candidate, ctx=8192");
	TEST_ASSERT(res.evaluated_count == 2, "both candidates were evaluated");
	TEST_ASSERT(res.evaluated[0].feasible == 0
		&& res.evaluated[1].feasible == 1,
		"candidate 0 (smaller ctx) correctly infeasible, candidate 1 "
		"(larger ctx) correctly feasible -- proves feasibility is not "
		"assumed monotonic in ctx");
}

static void	test_invalid_n_layer_all_fails_closed(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	req.n_layer_all = 0;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 0, "fails closed");
	TEST_ASSERT(strcmp(res.status, MEMBRANE_CTXREC_STATUS_INVALID_INPUT)
		== 0, "INVALID_INPUT");
}

static void	test_null_req_and_out_are_safe(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;

	fill_base_request(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(NULL, &res) == 0,
		"NULL request never crashes, just fails");
	TEST_ASSERT(membrane_ctxrec_resolve(&req, NULL) == 0,
		"NULL out pointer never crashes, just fails");
}

/* ------------------------------------------------------------------ */
/* Section 28: property/invariant tests                               */
/* ------------------------------------------------------------------ */

static void	test_property_invariants_hold(void)
{
	membrane_ctxrec_request_t	req;
	membrane_ctxrec_result_t	res;
	size_t						i;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &res) == 1, "resolves ok");
	TEST_ASSERT(res.recommended_context <= res.hardware_fit_context,
		"recommended_context <= hardware_fit_context");
	TEST_ASSERT(res.hardware_fit_context <= res.model_max_context,
		"hardware_fit_context <= model_max_context");
	TEST_ASSERT(res.recommended_context >= res.minimum_required_context,
		"recommended_context >= minimum_required_context when status OK");
	TEST_ASSERT(res.selected_plan.ok == 1, "selected plan is feasible");
	TEST_ASSERT(res.candidate_count <= MEMBRANE_CTXREC_MAX_CANDIDATES,
		"candidate count <= fixed maximum");
	for (i = 0; i < res.evaluated_count; ++i)
	{
		TEST_ASSERT(res.evaluated[i].ctx <= res.model_max_context,
			"no candidate > model_max_context");
		if (i > 0)
			TEST_ASSERT(res.evaluated[i].ctx > res.evaluated[i - 1].ctx,
				"candidate order strictly increasing");
	}
}

int	main(void)
{
	test_gen_basic_doubling_matches_hand_derivation();
	test_gen_real_smollm2_qwen25_value();
	test_gen_floor_equals_model_max_single_candidate();
	test_gen_non_power_of_two_model_max();
	test_gen_model_max_below_floor_returns_zero();
	test_gen_minimum_exceeds_model_max_returns_zero();
	test_gen_minimum_raises_floor();
	test_gen_respects_small_max_out();
	test_gen_huge_model_max_bounded_and_no_overflow();
	test_gen_null_out_is_safe();
	test_ample_memory_multiple_feasible_picks_largest();
	test_non_power_of_two_model_max_resolves();
	test_no_candidates_at_all_is_no_feasible_context();
	test_unsupported_architecture_is_planner_rejected_all();
	test_minimum_required_context_filters_smaller_candidates();
	test_minimum_exceeds_model_max();
	test_missing_model_max_is_unknown();
	test_zero_model_max_is_invalid_not_unknown();
	test_cpu_only_explicit_precision_resolves();
	test_cpu_only_adaptive_precision_does_not_resolve_today();
	test_gpu_full_residency_selected_shape();
	test_cpu_kv_spill_flags_host_memory_unvalidated();
	test_adaptive_precision_forwarded_from_joint_planner();
	test_explicit_q8_is_hard_constraint_every_candidate();
	test_explicit_q5_is_hard_constraint_every_candidate();
	test_explicit_gpu_placement_is_hard_constraint();
	test_malformed_head_shape_rejected();
	test_deterministic_repeat();
	test_ctx_overflow_candidate_fails_closed();
	test_candidate_count_bound_propagates_to_result();
	test_explicit_gpu_layers_hard_constraint();
	test_no_monotonicity_assumption();
	test_invalid_n_layer_all_fails_closed();
	test_null_req_and_out_are_safe();
	test_property_invariants_hold();
	printf("test_context_recommender: all tests passed\n");
	return (0);
}
