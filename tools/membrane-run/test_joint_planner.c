#include <stdint.h>
#include <string.h>

#include "joint_planner.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

/* Shared synthetic model shape reused across most tests below (hand
 * verified against the exact integer arithmetic
 * membrane_joint_plan_resolve() uses -- see the Phase 20 PR
 * description for the derivation): n_layer_all=10, bytes_per_layer=
 * 10 MiB, output_role_bytes=20 MiB, real llama-compatible hparams
 * (head_dim 64, divisible by compat_check.h's 32-byte block size).
 * kv_bytes_native/q8/q5 are illustrative (not tied to any real ctx),
 * chosen so native > q8 > q5, matching every real KV mode's actual
 * ordering. */
static void	fill_base_request(membrane_joint_plan_request_t *req)
{
	memset(req, 0, sizeof(*req));
	req->n_layer_all = 10;
	req->bytes_per_layer = 10 * MIB;
	req->output_role_bytes = 20 * MIB;
	req->arch_name = "llama";
	req->n_embd = 512;
	req->n_head = 8;
	req->n_head_kv = 8;
	req->ctx_size = 2048;
	req->kv_bytes_native = 40 * MIB;
	req->kv_bytes_q8 = 20 * MIB;
	req->kv_bytes_q5 = 14 * MIB;
	req->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
}

/* ------------------------------------------------------------------ */
/* membrane_joint_estimate_gpu_weight_bytes() -- Section 25            */
/* ------------------------------------------------------------------ */

static void	test_estimate_zero_layers_is_zero(void)
{
	TEST_ASSERT(membrane_joint_estimate_gpu_weight_bytes(0, 10 * MIB,
			20 * MIB) == 0, "0 GPU layers is always 0 bytes");
	TEST_ASSERT(membrane_joint_estimate_gpu_weight_bytes(-1, 10 * MIB,
			20 * MIB) == 0, "negative v is treated as 0, not underflowed");
}

static void	test_estimate_one_layer_is_output_role_only(void)
{
	/* Source-verified mechanism (joint_planner.h's own top comment):
	 * V=1 offloads the output-role tensor(s) FIRST, zero blk.N. layers
	 * -- (1-1)*bytes_per_layer + output_role_bytes == output_role_bytes
	 * exactly, never bytes_per_layer itself (the pre-Phase-20 bug this
	 * corrects). */
	TEST_ASSERT(membrane_joint_estimate_gpu_weight_bytes(1, 10 * MIB,
			20 * MIB) == 20 * MIB,
		"V=1 is exactly output_role_bytes, not bytes_per_layer");
}

static void	test_estimate_all_layers_matches_hand_derivation(void)
{
	/* Real Phase 19 SmolLM2-135M shape: n_layer_all=30,
	 * bytes_per_layer=7082496, output_role_bytes=56625408 (tied
	 * embeddings). Hand-derived corrected estimate for "all" (V=30):
	 * 29*7082496 + 56625408 = 262017792 -- verified in this PR's
	 * description against real observed GPU memory (2.98% residual
	 * error, down from 21.3%). */
	TEST_ASSERT(membrane_joint_estimate_gpu_weight_bytes(30, 7082496,
			56625408) == 262017792,
		"corrected estimate matches the real SmolLM2-135M derivation");
}

static void	test_estimate_no_overflow_on_large_inputs(void)
{
	uint64_t	huge = membrane_joint_estimate_gpu_weight_bytes(1000,
			500 * MIB, 4 * GIB);

	/* 999 * 500MiB + 4GiB, comfortably within uint64_t range -- this is
	 * a sanity check that checked/plain uint64 arithmetic doesn't wrap
	 * for realistic-scale-but-large inputs, not a claim about any real
	 * model's actual layer count. */
	TEST_ASSERT(huge == (uint64_t)999 * 500 * MIB + 4 * GIB,
		"large inputs do not overflow/wrap");
}

/* ------------------------------------------------------------------ */
/* Section 23: core candidate generation / ranking scenarios          */
/* ------------------------------------------------------------------ */

/* 1. Large VRAM -> adaptive resolves to Q8 (preferred over Q5 when
 * both reach full residency -- Section 24's tie-break) at every
 * layer. */
static void	test_ample_memory_selects_q8_full_residency(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"ample memory resolves");
	TEST_ASSERT(res.ok == 1, "out->ok reflects success");
	TEST_ASSERT(res.candidates[res.selected_index].kv_precision
			== MEMBRANE_JOINT_KV_Q8,
		"Q8 preferred over Q5 when both reach full residency");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 10,
		"full residency: every layer selected");
	TEST_ASSERT(res.candidates[res.selected_index].kv_placement
			== MEMBRANE_JOINT_PLACEMENT_DEFAULT,
		"DEFAULT placement mode never becomes a GPU/CPU candidate value");
}

/* 2. Explicit --kv native at the SAME constrained budget as test 3/4
 * below reaches only PARTIAL residency (8/10), while --kv q8/q5 (same
 * budget) reach full residency -- proves the joint planner's fit
 * arithmetic genuinely differs by precision (native's real 40 MiB KV
 * vs q8's 20 MiB matters), not just a placeholder. */
static void	test_explicit_native_partial_vs_q8_full(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res_native;
	membrane_joint_plan_result_t	res_q8;

	fill_base_request(&req);
	req.device_free_bytes = 650 * MIB;
	req.device_total_bytes = 1 * GIB;
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res_native) == 1,
		"explicit native resolves (partially)");
	TEST_ASSERT(res_native.candidates[res_native.selected_index].gpu_layers
			== 8, "native reaches only 8/10 layers at this budget");

	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res_q8) == 1,
		"explicit q8 resolves");
	TEST_ASSERT(res_q8.candidates[res_q8.selected_index].gpu_layers == 10,
		"q8's smaller KV footprint reaches full residency at the SAME "
		"budget native could not");
}

/* 3. Q5 required for full residency (Q8 only reaches 9/10, Q5 reaches
 * 10/10) -- MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_FULL_RESIDENCY,
 * reused verbatim from adaptive_kv_policy.h. */
static void	test_constrained_q5_required_for_full_residency(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = 640 * MIB;
	req.device_total_bytes = 1 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"constrained-for-q8 memory still resolves via q5");
	TEST_ASSERT(res.candidates[res.selected_index].kv_precision
			== MEMBRANE_JOINT_KV_Q5, "q5 selected over q8");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 10,
		"q5 reaches full residency where q8 could not");
}

/* CodeRabbit review (PR #30): membrane_adaptive_kv_resolve() ranks Q8
 * vs Q5 using GPU-FIT alone -- it has no visibility into PLACEMENT
 * feasibility. Constructed so BOTH q8 and q5 reach full GPU-fit
 * residency (v=10/10), so adaptive_kv_resolve's own rule 1 picks Q8
 * ("Q8 fits AND reaches full residency -> Q8") -- but with
 * kv_placement_mode=GPU forced, Q8's real 2000 MiB KV fails
 * kv_residency_resolve()'s own runtime margin (5% of 2000 MiB = 100
 * MiB, on top of the same weight bytes both precisions share) while
 * Q5's smaller 1400 MiB KV (margin 70 MiB) still fits. The joint
 * planner must promote Q5 to the selected candidate instead of
 * reporting NO_FEASIBLE_CANDIDATE or silently keeping the
 * placement-ineligible Q8 "winner". Hand-derived and cross-checked
 * against kv_residency_policy.c's exact arithmetic. */
static void	test_adaptive_placement_promotes_runner_up_precision(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.bytes_per_layer = 100 * MIB;
	req.output_role_bytes = 200 * MIB;
	req.kv_bytes_q8 = 2000 * MIB;
	req.kv_bytes_q5 = 1400 * MIB;
	req.device_free_bytes = 4400 * MIB;
	req.device_total_bytes = 8 * GIB;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_GPU;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"resolves via the placement-eligible runner-up, not a hard failure");
	TEST_ASSERT(res.candidates[res.selected_index].kv_precision
			== MEMBRANE_JOINT_KV_Q5,
		"q5 promoted: q8 (adaptive's own GPU-fit winner) failed placement");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 10,
		"full residency preserved for the promoted candidate");
	TEST_ASSERT(res.candidates[res.selected_index].kv_placement
			== MEMBRANE_JOINT_PLACEMENT_GPU,
		"placement genuinely resolved to gpu, not silently left default");
}

/* 4. Neither q8 nor q5 reaches full residency, but q5 fits MORE
 * layers -- MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_MEMORY_GUARD. */
static void	test_constrained_q5_required_for_memory_guard(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = 610 * MIB;
	req.device_total_bytes = 1 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"still resolves, more constrained again");
	TEST_ASSERT(res.candidates[res.selected_index].kv_precision
			== MEMBRANE_JOINT_KV_Q5, "q5 selected: more layers than q8");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 7,
		"q5's own real max-fit layer count at this budget");
}

/* 5. qwen2 (incompatible architecture): adaptive request finds NO
 * eligible candidate at all -- never falls back to native (adaptive
 * never proposes native, matching adaptive_kv_policy.h's own
 * contract), fails closed with a stable, informative reason. */
static void	test_incompatible_architecture_adaptive_fails_closed(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.arch_name = "qwen2";
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 0,
		"qwen2 + adaptive (q8/q5 only) has no eligible candidate");
	TEST_ASSERT(res.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(res.selected_index == -1, "no candidate selected");
	TEST_ASSERT(strcmp(res.candidates[0].reason_code,
			MEMBRANE_JOINT_REASON_INCOMPATIBLE_ARCH) == 0,
		"the one recorded candidate names the real reason");
}

/* 6. Explicit --kv q8 on an incompatible architecture: a single
 * ineligible candidate, never silently substituting native or q5. */
static void	test_explicit_q8_incompatible_architecture(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.arch_name = "qwen2";
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 0,
		"explicit q8 on qwen2 has no eligible candidate");
	TEST_ASSERT(res.candidate_count == 1,
		"exactly one (ineligible) candidate recorded, no substitution");
	TEST_ASSERT(res.candidates[0].kv_precision == MEMBRANE_JOINT_KV_Q8,
		"the recorded candidate is still q8, never silently native/q5");
}

/* 7. Explicit --kv-placement cpu never produces a GPU placement
 * candidate value, regardless of how much memory is free. */
static void	test_explicit_cpu_placement_never_gpu(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_CPU;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"resolves with ample memory");
	TEST_ASSERT(res.candidates[res.selected_index].kv_placement
			== MEMBRANE_JOINT_PLACEMENT_CPU,
		"explicit cpu placement is honored, never overridden to gpu");
	TEST_ASSERT(res.candidates[res.selected_index].estimated_kv_gpu_bytes
			== 0, "no KV bytes counted as GPU-resident under cpu placement");
}

/* 8. Explicit --gpu-layers N: the winning candidate's gpu_layers is
 * EXACTLY N, never silently shrunk (matching gpu_policy.h's own
 * ALL/N contract). */
static void	test_explicit_gpu_layers_never_changed(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.gpu_layers_request = 5;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"explicit N=5 resolves");
	TEST_ASSERT(res.candidate_count == 1,
		"exactly one candidate for an explicit N -- no alternative "
		"layer count is ever considered");
	TEST_ASSERT(res.candidates[0].gpu_layers == 5,
		"selected layer count is exactly the explicit request");
}

/* 9. Explicit --gpu-layers N that does not fit: no feasible candidate
 * (never silently clamped to a smaller N). */
static void	test_explicit_gpu_layers_impossible_fails_closed(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.gpu_layers_request = 10;
	req.device_free_bytes = 530 * MIB;
	req.device_total_bytes = 1 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 0,
		"explicit N=10 does not fit in this tiny budget");
	TEST_ASSERT(res.candidate_count == 1,
		"still exactly one candidate, at the explicit N -- never a "
		"silently smaller alternative");
	TEST_ASSERT(res.candidates[0].gpu_layers == 10,
		"the (ineligible) candidate still reports the real requested N");
}

/* CodeRabbit review (PR #30): gpu_device.h's output_role_bytes == 0
 * means "the output-role tensor(s) could not be identified" (an
 * unavailable estimate), NOT "the real output-role footprint is zero
 * bytes" -- llama.cpp always offloads *something* for the output role
 * once gpu_layers >= 1. A GPU candidate must never be accepted on the
 * strength of a silently-zero fixed cost; only the 0-layer (CPU-only
 * weights) candidate remains legitimate in that case. */
static void	test_unavailable_output_role_bytes_rejects_gpu_candidates(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.output_role_bytes = 0;
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"still resolves, via the 0-layer fallback only");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 0,
		"no GPU-layer candidate is accepted when output_role_bytes is "
		"unavailable, even with ample memory");
	for (int i = 0; i < res.candidate_count; ++i)
		if (res.candidates[i].gpu_layers > 0)
			TEST_ASSERT(res.candidates[i].eligible == 0,
				"every generated GPU-layer candidate is ineligible "
				"when output_role_bytes == 0");
}

/* 10. Degenerate near-zero GPU budget with AUTO layers: the 0-layer
 * CPU-fallback candidate is the only eligible one -- exercises the
 * same "graceful CPU landing" this module must provide when a GPU
 * was requested but essentially no budget exists (main.cpp's own
 * separate, unconditional gpu_layers==0 short-circuit handles the
 * "no GPU backend/device at all" case entirely outside this module --
 * see joint_planner.h's own top comment). */
static void	test_near_zero_budget_falls_back_to_cpu_layers(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.device_free_bytes = 530 * MIB;
	req.device_total_bytes = 1 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"resolves via the 0-layer fallback, not a hard failure");
	TEST_ASSERT(res.candidates[res.selected_index].gpu_layers == 0,
		"selected candidate is the CPU-only fallback");
	TEST_ASSERT(res.candidates[res.selected_index].estimated_weight_gpu_bytes
			== 0, "0 GPU layers means 0 GPU weight bytes");
}

/* 11. Determinism: identical inputs called twice produce byte-
 * identical results (candidate order, selected_index, every field). */
static void	test_deterministic_repeated_calls(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res1;
	membrane_joint_plan_result_t	res2;

	fill_base_request(&req);
	req.device_free_bytes = 610 * MIB;
	req.device_total_bytes = 1 * GIB;
	membrane_joint_plan_resolve(&req, &res1);
	membrane_joint_plan_resolve(&req, &res2);
	TEST_ASSERT(memcmp(&res1, &res2, sizeof(res1)) == 0,
		"identical inputs produce a byte-identical result struct");
}

/* 12. All candidates infeasible (budget below even the fixed
 * output-role cost): deterministic NO_FEASIBLE_CANDIDATE, not a
 * crash or an arbitrarily-chosen "least bad" candidate. */
static void	test_all_candidates_infeasible_is_deterministic(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.device_free_bytes = 512 * MIB;	/* == the fixed reserve floor
										 * exactly -- budget is 0 */
	req.device_total_bytes = 1 * GIB;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_GPU;	/* forces the
										 * 0-layer fallback ineligible
										 * too: GPU placement with 0 GPU
										 * weight layers has nowhere to
										 * put GPU-resident KV either */
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 0,
		"zero budget with GPU-forced placement has no feasible candidate");
	TEST_ASSERT(res.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(res.selected_index == -1, "no candidate selected");
	TEST_ASSERT(strcmp(res.reason_code,
			MEMBRANE_JOINT_REASON_NO_FEASIBLE_CANDIDATE) == 0,
		"stable, deterministic top-level reason code");
}

/* ------------------------------------------------------------------ */
/* Invalid input handling                                              */
/* ------------------------------------------------------------------ */

static void	test_invalid_n_layer_all_fails_closed(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;

	fill_base_request(&req);
	req.n_layer_all = 0;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 0,
		"n_layer_all <= 0 is rejected, not a crash");
	TEST_ASSERT(strcmp(res.reason_code,
			MEMBRANE_JOINT_REASON_INVALID_CONFIG) == 0,
		"invalid-config reason code");
}

static void	test_null_out_is_safe(void)
{
	membrane_joint_plan_request_t	req;

	fill_base_request(&req);
	TEST_ASSERT(membrane_joint_plan_resolve(&req, NULL) == 0,
		"NULL out pointer never crashes, just fails");
}

int	main(void)
{
	test_estimate_zero_layers_is_zero();
	test_estimate_one_layer_is_output_role_only();
	test_estimate_all_layers_matches_hand_derivation();
	test_estimate_no_overflow_on_large_inputs();
	test_ample_memory_selects_q8_full_residency();
	test_explicit_native_partial_vs_q8_full();
	test_constrained_q5_required_for_full_residency();
	test_adaptive_placement_promotes_runner_up_precision();
	test_constrained_q5_required_for_memory_guard();
	test_incompatible_architecture_adaptive_fails_closed();
	test_explicit_q8_incompatible_architecture();
	test_explicit_cpu_placement_never_gpu();
	test_explicit_gpu_layers_never_changed();
	test_explicit_gpu_layers_impossible_fails_closed();
	test_unavailable_output_role_bytes_rejects_gpu_candidates();
	test_near_zero_budget_falls_back_to_cpu_layers();
	test_deterministic_repeated_calls();
	test_all_candidates_infeasible_is_deterministic();
	test_invalid_n_layer_all_fails_closed();
	test_null_out_is_safe();
	printf("test_joint_planner: all tests passed\n");
	return (0);
}
