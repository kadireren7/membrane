#include <stdint.h>
#include <string.h>

#include "gpu_policy.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

static void	test_auto_sufficient_memory_selects_all_layers(void)
{
	membrane_gpu_policy_result_t	r;

	/* 4 GiB device, 3.9 GiB free, ~9 MiB/layer, 30 layers, tiny KV --
	 * simulated inputs in the rough shape of this project's real
	 * SmolLM2-135M/GTX 1650 sizing (Phase 9A/9B), not the measurements
	 * themselves: everything comfortably fits. */
	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO,
			30, (uint64_t)(3.9 * GIB), 4 * GIB, 9 * MIB, 6 * MIB,
			&r) == 1, "auto resolves when the model comfortably fits");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
	TEST_ASSERT(r.selected_layers == 30,
		"auto selects every layer when the budget easily covers it");
}

static void	test_auto_constrained_memory_selects_partial_layers(void)
{
	membrane_gpu_policy_result_t	r;

	/* Same model shape (30 layers, 9 MiB/layer, 6 MiB KV), but on a 2
	 * GiB device with only 700 MiB free: reserve is the 512 MiB fixed
	 * floor (15% of 2 GiB is smaller), leaving 700-512-6=182 MiB for
	 * weights -> floor(182/9)=20 of 30 layers. This is the "partial
	 * offload" case Section 10 asks for -- worked out by hand first so
	 * a wrong policy change trips this test, not just a >0/<30 range
	 * check that would tolerate a differently-broken formula. */
	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO,
			30, 700 * MIB, 2 * GIB, 9 * MIB, 6 * MIB,
			&r) == 1, "auto resolves a smaller device to a partial count");
	TEST_ASSERT(r.ok == 1, "out->ok reflects success");
	TEST_ASSERT(r.safety_reserve_bytes == MEMBRANE_GPU_RESERVE_FIXED_BYTES,
		"512 MiB fixed floor applies (15% of 2 GiB would be smaller)");
	TEST_ASSERT(r.selected_layers == 20,
		"floor(182 MiB / 9 MiB) = 20 layers, worked out by hand");
}

static void	test_auto_reserve_calculation(void)
{
	membrane_gpu_policy_result_t	r;

	/* 4 GiB total: 15% = ~629 MiB, well above the 512 MiB fixed floor
	 * -- the percentage reserve should win here, exactly matching the
	 * documented "larger of the two" policy. */
	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO, 30, 4 * GIB,
		4 * GIB, 1 * MIB, 0, &r);
	TEST_ASSERT(r.safety_reserve_bytes == (4 * GIB / 100 * 15),
		"15% of a 4 GiB device wins over the 512 MiB fixed floor");

	/* 1 GiB total: 15% = ~157 MiB, below the 512 MiB fixed floor --
	 * the fixed floor should win instead. */
	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO, 30, 1 * GIB,
		1 * GIB, 1 * MIB, 0, &r);
	TEST_ASSERT(r.safety_reserve_bytes == MEMBRANE_GPU_RESERVE_FIXED_BYTES,
		"the 512 MiB fixed floor wins over 15% of a 1 GiB device");
}

static void	test_auto_layer_estimate_rounding(void)
{
	membrane_gpu_policy_result_t	r;

	/* Budget for weights after reserve/KV is exactly 2.5x one layer's
	 * size -- integer division must floor to 2, never round up to 3
	 * (which would silently exceed the budget). */
	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO, 10,
		10 * MIB + 512 * MIB /* free */, 512 * MIB /* total, tiny reserve */,
		4 * MIB /* bytes/layer */, 0 /* kv */, &r);
	TEST_ASSERT(r.ok == 1, "resolves with a real budget");
	TEST_ASSERT((uint64_t)r.selected_layers * 4 * MIB <= r.budget_bytes,
		"selected layers never exceed the budget -- floor, not round");
}

static void	test_auto_zero_bytes_per_layer_fails_clearly(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO,
			30, 4 * GIB, 4 * GIB, 0 /* unknown */, 0, &r) == 0,
		"auto fails closed when the per-layer estimate is unavailable");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(strlen(r.reason) > 0, "a reason is given");
}

static void	test_auto_zero_free_memory_fails_no_layers_fit(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO,
			30, 0 /* no free memory reported */, 4 * GIB, 9 * MIB, 0,
			&r) == 0, "auto fails closed when no memory is free");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
}

static void	test_explicit_all_fails_closed_when_insufficient(void)
{
	membrane_gpu_policy_result_t	r;

	/* 30 layers at 9 MiB each need ~270 MiB; only 100 MiB is free on a
	 * 512 MiB device -- explicit "all" must be REJECTED outright, not
	 * silently downgraded to however many layers actually fit (that's
	 * what auto is for). */
	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_ALL,
			30, 100 * MIB, 512 * MIB, 9 * MIB, 0, &r) == 0,
		"explicit --gpu-layers all fails closed when it doesn't fit");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(strlen(r.reason) > 0, "a reason is given");
}

static void	test_explicit_all_succeeds_when_sufficient(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_ALL,
			30, (uint64_t)(3.9 * GIB), 4 * GIB, 9 * MIB, 6 * MIB,
			&r) == 1, "explicit --gpu-layers all succeeds when it fits");
	TEST_ASSERT(r.selected_layers == 30,
		"all resolves to the model's real total layer count");
}

static void	test_explicit_n_fails_closed_when_insufficient(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(20, 30, 100 * MIB, 512 * MIB,
			9 * MIB, 0, &r) == 0,
		"explicit --gpu-layers N fails closed when N doesn't fit");
}

static void	test_explicit_n_clamped_to_model_layer_count(void)
{
	membrane_gpu_policy_result_t	r;

	/* Requesting more layers than the model has (a common real-world
	 * llama.cpp CLI idiom, e.g. "-ngl 99" to mean "all") clamps to the
	 * model's actual count rather than computing a budget for 99
	 * nonexistent layers. */
	TEST_ASSERT(membrane_gpu_policy_resolve(99, 30, (uint64_t)(3.9 * GIB),
			4 * GIB, 9 * MIB, 6 * MIB, &r) == 1,
		"an over-large explicit N clamps to the model's layer count");
	TEST_ASSERT(r.selected_layers == 30,
		"selected_layers is the model's real count, not the requested 99");
}

static void	test_explicit_missing_estimate_does_not_block(void)
{
	membrane_gpu_policy_result_t	r;

	/* Unlike auto, an explicit request proceeds when the estimate is
	 * unavailable -- Section 6: "Do not block configurations merely
	 * because estimates are conservative unless there is a clear
	 * reason"; with no estimate at all, there is no evidence to
	 * reject on, so the user's explicit choice is honored. */
	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_ALL,
			30, 4 * GIB, 4 * GIB, 0 /* unknown */, 0, &r) == 1,
		"explicit --gpu-layers all proceeds when no estimate is available");
	TEST_ASSERT(r.selected_layers == 30, "selected_layers is the full count");
}

static void	test_zero_layers_always_ok(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(0, 30, 0, 0, 0, 0, &r) == 1,
		"requesting zero layers is always satisfiable");
	TEST_ASSERT(r.selected_layers == 0, "selected_layers is 0");
}

static void	test_invalid_layer_count_rejected(void)
{
	membrane_gpu_policy_result_t	r;

	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO,
			0 /* model reports no layers */, 4 * GIB, 4 * GIB, 9 * MIB,
			0, &r) == 0, "a non-positive model layer count fails closed");
}

static void	test_unsupported_negative_layers_rejected(void)
{
	membrane_gpu_policy_result_t	r;

	/* -3 is neither MEMBRANE_GPU_LAYERS_ALL (-1) nor _AUTO (-2) --
	 * with bytes_per_layer_estimate==0 this previously fell through
	 * to the explicit-N branch's "needed=0, no estimate to check
	 * against" path and returned ok=1 with a negative selected_layers,
	 * a genuine bug caught during CodeRabbit review of this file. */
	TEST_ASSERT(membrane_gpu_policy_resolve(-3, 30, 4 * GIB, 4 * GIB, 0, 0,
			&r) == 0, "an unsupported negative layer count with no "
		"estimate fails closed rather than returning a negative "
		"selected_layers");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(strlen(r.reason) > 0, "a reason is given");
	/* With a nonzero estimate, the old code cast -3 to uint64_t before
	 * multiplying -- an unsigned-wraparound path that could produce
	 * an unpredictable "needed" byte count instead of a clean
	 * rejection. */
	TEST_ASSERT(membrane_gpu_policy_resolve(-3, 30, 4 * GIB, 4 * GIB,
			9 * MIB, 0, &r) == 0,
		"an unsupported negative layer count with a real estimate "
		"still fails closed, not via integer-overflow arithmetic");
	TEST_ASSERT(membrane_gpu_policy_resolve(INT32_MIN, 30, 4 * GIB,
			4 * GIB, 9 * MIB, 0, &r) == 0,
		"the most negative int32 value also fails closed");
}

static void	test_null_out_pointer_is_safe(void)
{
	TEST_ASSERT(membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO, 30,
			4 * GIB, 4 * GIB, 9 * MIB, 0, NULL) == 0,
		"a NULL out pointer returns failure rather than crashing");
}

/* Phase 13.1, Section 5: stable reason_code coverage -- one case per
 * code, checked with strcmp (not just strlen(reason) > 0 as the
 * pre-existing tests above already do for the free-text reason). */
static void	test_reason_code_gpu_full_fit(void)
{
	membrane_gpu_policy_result_t	r;

	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_ALL, 30,
		(uint64_t)(3.9 * GIB), 4 * GIB, 9 * MIB, 6 * MIB, &r);
	TEST_ASSERT(strcmp(r.reason_code, MEMBRANE_GPU_POLICY_REASON_GPU_FULL_FIT)
		== 0, "explicit all, fully satisfied, is GPU_FULL_FIT");
}

static void	test_reason_code_gpu_layers_clamped(void)
{
	membrane_gpu_policy_result_t	r;

	membrane_gpu_policy_resolve(99, 30, (uint64_t)(3.9 * GIB), 4 * GIB,
		9 * MIB, 6 * MIB, &r);
	TEST_ASSERT(strcmp(r.reason_code,
			MEMBRANE_GPU_POLICY_REASON_GPU_LAYERS_CLAMPED) == 0,
		"an over-large explicit N, clamped to the model's real count, "
		"is GPU_LAYERS_CLAMPED");
}

static void	test_reason_code_cpu_only_requested(void)
{
	membrane_gpu_policy_result_t	r;

	membrane_gpu_policy_resolve(0, 30, 4 * GIB, 4 * GIB, 9 * MIB, 0, &r);
	TEST_ASSERT(strcmp(r.reason_code,
			MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED) == 0,
		"requested_layers == 0 is CPU_ONLY_REQUESTED");
}

static void	test_reason_code_memory_insufficient(void)
{
	membrane_gpu_policy_result_t	r;

	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_ALL, 30, 100 * MIB,
		512 * MIB, 9 * MIB, 0, &r);
	TEST_ASSERT(strcmp(r.reason_code,
			MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT) == 0,
		"an explicit request that doesn't fit is GPU_MEMORY_INSUFFICIENT");
	TEST_ASSERT(strncmp(r.reason, r.reason_code, strlen(r.reason_code)) == 0,
		"the free-text reason is prefixed with the same stable code");
}

static void	test_reason_code_metadata_unavailable(void)
{
	membrane_gpu_policy_result_t	r;

	membrane_gpu_policy_resolve(MEMBRANE_GPU_LAYERS_AUTO, 30, 4 * GIB,
		4 * GIB, 0 /* no per-layer estimate */, 0, &r);
	TEST_ASSERT(strcmp(r.reason_code,
			MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE) == 0,
		"auto with no per-layer estimate is MODEL_METADATA_UNAVAILABLE");
}

int	main(void)
{
	test_auto_sufficient_memory_selects_all_layers();
	test_auto_constrained_memory_selects_partial_layers();
	test_auto_reserve_calculation();
	test_auto_layer_estimate_rounding();
	test_auto_zero_bytes_per_layer_fails_clearly();
	test_auto_zero_free_memory_fails_no_layers_fit();
	test_explicit_all_fails_closed_when_insufficient();
	test_explicit_all_succeeds_when_sufficient();
	test_explicit_n_fails_closed_when_insufficient();
	test_explicit_n_clamped_to_model_layer_count();
	test_explicit_missing_estimate_does_not_block();
	test_zero_layers_always_ok();
	test_invalid_layer_count_rejected();
	test_unsupported_negative_layers_rejected();
	test_null_out_pointer_is_safe();
	test_reason_code_gpu_full_fit();
	test_reason_code_gpu_layers_clamped();
	test_reason_code_cpu_only_requested();
	test_reason_code_memory_insufficient();
	test_reason_code_metadata_unavailable();
	printf("test_gpu_policy: all tests passed\n");
	return (0);
}
