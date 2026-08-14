#include <stdio.h>
#include <string.h>

#include "adaptive_kv_policy.h"
#include "test_helpers.h"

#define GPU	1
#define CPU	0

/* Real Q5_1/Q8_0/F16 KV bytes-per-token ratios for a 64-wide head
 * (tools/membrane-llama-runtime/test_kv_store_telemetry.c: F16=128,
 * Q8_0=68, Q5_1=48 bytes/token) -- used here only to ground candidate
 * kv_bytes fields in real numbers, never recomputed by this module. */
#define KV_BYTES_Q8	68000ull
#define KV_BYTES_Q5	48000ull

static void	test_gpu_q8_full_residency_wins(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){1, 1, 30, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 1, 30, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"Q8 fits with full residency -> resolves");
	TEST_ASSERT(r.ok == 1, "ok reflects success");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"Q8 chosen when it safely reaches full residency, even though "
		"Q5 also fully fits and uses fewer bytes -- never choose Q5 "
		"merely because it is cheaper");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_ADAPTIVE_REASON_Q8_FULL_RESIDENCY)
		== 0, "reason is Q8_FULL_RESIDENCY");
	TEST_ASSERT(r.selected_layers == 30, "selected_layers echoes Q8's 30");
	TEST_ASSERT(r.selected_kv_bytes == KV_BYTES_Q8,
		"selected_kv_bytes echoes Q8's real byte estimate");
}

static void	test_gpu_q5_required_for_full_residency(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Q8 only reaches 22 of 30 layers (partial); Q5's smaller footprint
	 * reaches all 30 -- this is the ZONE B transition case. */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 22, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 1, 30, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"Q5 resolves when it is the only mode reaching full residency");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q5,
		"Q5 chosen over a partial-residency Q8");
	TEST_ASSERT(strcmp(r.reason,
			MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_FULL_RESIDENCY) == 0,
		"reason is Q5_REQUIRED_FOR_FULL_RESIDENCY");
	TEST_ASSERT(r.selected_layers == 30, "selected_layers echoes Q5's 30");
}

static void	test_gpu_partial_same_layer_count_prefers_q8(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Neither reaches full residency (30), and both land on the exact
	 * same partial layer count -- Q5's smaller KV footprint buys zero
	 * practical residency benefit here, so Q8 must still win. */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves when both fit the same partial layer count");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"Q8 wins a tie -- do not sacrifice quality for zero practical "
		"memory benefit");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_ADAPTIVE_REASON_Q8_FITS) == 0,
		"reason is Q8_FITS");
	TEST_ASSERT(r.selected_layers == 18, "selected_layers echoes 18");
}

static void	test_gpu_partial_q5_meaningfully_more_layers(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Neither reaches full residency (30), but Q5 achieves materially
	 * more layers (25 vs 18) -- a real residency improvement, not a
	 * cosmetic byte-count win. */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 25, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves when Q5 offers more partial layers than Q8");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q5,
		"Q5 chosen for a meaningful residency improvement");
	TEST_ASSERT(strcmp(r.reason,
			MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_MEMORY_GUARD) == 0,
		"reason is Q5_REQUIRED_FOR_MEMORY_GUARD");
	TEST_ASSERT(r.selected_layers == 25, "selected_layers echoes Q5's 25");
}

/* Pins the exact margin the policy treats as "meaningfully more
 * layers" (Section 7): any nonzero improvement in achievable layer
 * count is a real GPU-offload benefit (one more real layer placed),
 * so even a 1-layer difference must select Q5 -- while a 0-layer
 * difference (a true tie, test_gpu_partial_same_layer_count_prefers_
 * q8 above) must still select Q8. Both boundary directions pinned
 * here so a future policy change that shifts this threshold either
 * way trips a test, not just the wide 18-vs-25 gap already covered by
 * test_gpu_partial_q5_meaningfully_more_layers. */
static void	test_gpu_partial_smallest_margin_boundary(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Smallest possible nonzero improvement (19 vs 18) -> Q5. */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 19, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves at the smallest possible nonzero layer-count margin");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q5,
		"a 1-layer improvement is still a real GPU-offload benefit -- "
		"Q5 wins even at the smallest nonzero margin");
	TEST_ASSERT(strcmp(r.reason,
			MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_MEMORY_GUARD) == 0,
		"reason is Q5_REQUIRED_FOR_MEMORY_GUARD");

	/* One layer short of the same margin, but Q8 now has the larger
	 * count (19 vs 18) -- Q8 must win, confirming the comparison is
	 * strictly "more layers", not merely "not equal". */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 19, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves when Q8 has the larger partial layer count");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"Q8 wins when IT has more layers than Q5, not just on a tie");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_ADAPTIVE_REASON_Q8_FITS) == 0,
		"reason is Q8_FITS");
}

static void	test_gpu_only_q5_fits_at_all(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 10, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves when Q8 is invalid outright but Q5 fits");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q5,
		"Q5 chosen as the only mode that fits at all");
	TEST_ASSERT(strcmp(r.reason,
		MEMBRANE_ADAPTIVE_REASON_Q5_ONLY_COMPRESSED_MODE_THAT_FITS) == 0,
		"reason is Q5_ONLY_COMPRESSED_MODE_THAT_FITS");
}

static void	test_gpu_only_q8_fits_at_all(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 5, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves when only Q8 fits");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"Q8 chosen as the only valid mode");
	TEST_ASSERT(strcmp(r.reason, MEMBRANE_ADAPTIVE_REASON_Q8_FITS) == 0,
		"reason is Q8_FITS");
}

static void	test_gpu_neither_fits_fails_closed(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 0,
		"neither mode fitting is reported as a failure");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(r.selected_mode == 0,
		"selected_mode stays 0/unset on failure -- never a silent "
		"native fallback value");
	TEST_ASSERT(strcmp(r.reason,
		MEMBRANE_ADAPTIVE_REASON_NO_COMPRESSED_MODE_FITS) == 0,
		"reason is NO_COMPRESSED_MODE_FITS");
}

static void	test_reserve_boundary_near_full_residency(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Both one layer short of the model's real 30 -- neither is "full
	 * residency" even though they are right at the edge; the decision
	 * must still fall through to the partial-offload tie-break (Q8),
	 * not be nudged toward Q5 just because the boundary is close. */
	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 29, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 29, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r) == 1,
		"resolves right at the full-residency boundary");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"Q8 still wins one layer short of full residency, tied with Q5");
}

static void	test_cpu_q8_default(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){1, 1, 0, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 1, 0, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, CPU, &r) == 1,
		"CPU adaptive resolves");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8,
		"CPU adaptive defaults to Q8 when no budget rules it out");
	TEST_ASSERT(strcmp(r.reason,
		MEMBRANE_ADAPTIVE_REASON_CPU_ADAPTIVE_Q8_DEFAULT) == 0,
		"reason is CPU_ADAPTIVE_Q8_DEFAULT");
	TEST_ASSERT(r.selected_layers == 0,
		"CPU candidates never report a meaningful layer count");
}

static void	test_cpu_budget_forces_q5(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	/* Q8 invalid = caller already determined it exceeds --kv-budget-mib
	 * (this module never computes the budget comparison itself). */
	q8 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 1, 0, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, CPU, &r) == 1,
		"CPU adaptive resolves when the budget rules out Q8");
	TEST_ASSERT(r.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q5,
		"Q5 chosen under CPU memory pressure");
	TEST_ASSERT(strcmp(r.reason,
		MEMBRANE_ADAPTIVE_REASON_CPU_MEMORY_PRESSURE_Q5) == 0,
		"reason is CPU_MEMORY_PRESSURE_Q5");
}

static void	test_cpu_neither_fits_fails_closed(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){0, 0, 0, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, CPU, &r) == 0,
		"CPU adaptive fails closed when the budget rules out both");
	TEST_ASSERT(r.ok == 0, "out->ok reflects failure");
	TEST_ASSERT(strcmp(r.reason,
		MEMBRANE_ADAPTIVE_REASON_NO_COMPRESSED_MODE_FITS) == 0,
		"reason is NO_COMPRESSED_MODE_FITS");
}

static void	test_deterministic_repeated_calls(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r1;
	membrane_adaptive_kv_result_t		r2;

	q8 = (membrane_adaptive_kv_candidate_t){1, 0, 18, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 0, 25, KV_BYTES_Q5};
	membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r1);
	membrane_adaptive_kv_resolve(&q8, &q5, GPU, &r2);
	TEST_ASSERT(memcmp(&r1, &r2, sizeof(r1)) == 0,
		"identical inputs produce a byte-identical result across calls");
}

static void	test_null_inputs_are_safe(void)
{
	membrane_adaptive_kv_candidate_t	q8;
	membrane_adaptive_kv_candidate_t	q5;
	membrane_adaptive_kv_result_t		r;

	q8 = (membrane_adaptive_kv_candidate_t){1, 1, 30, KV_BYTES_Q8};
	q5 = (membrane_adaptive_kv_candidate_t){1, 1, 30, KV_BYTES_Q5};
	TEST_ASSERT(membrane_adaptive_kv_resolve(NULL, &q5, GPU, &r) == 0,
		"a NULL q8 candidate fails cleanly, not a crash");
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, NULL, GPU, &r) == 0,
		"a NULL q5 candidate fails cleanly, not a crash");
	TEST_ASSERT(membrane_adaptive_kv_resolve(&q8, &q5, GPU, NULL) == 0,
		"a NULL out pointer fails cleanly, not a crash");
}

int	main(void)
{
	test_gpu_q8_full_residency_wins();
	test_gpu_q5_required_for_full_residency();
	test_gpu_partial_same_layer_count_prefers_q8();
	test_gpu_partial_q5_meaningfully_more_layers();
	test_gpu_partial_smallest_margin_boundary();
	test_gpu_only_q5_fits_at_all();
	test_gpu_only_q8_fits_at_all();
	test_gpu_neither_fits_fails_closed();
	test_reserve_boundary_near_full_residency();
	test_cpu_q8_default();
	test_cpu_budget_forces_q5();
	test_cpu_neither_fits_fails_closed();
	test_deterministic_repeated_calls();
	test_null_inputs_are_safe();
	printf("test_adaptive_kv_policy: all tests passed\n");
	return (0);
}
