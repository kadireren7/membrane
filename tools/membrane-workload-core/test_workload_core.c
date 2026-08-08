#include <stdint.h>
#include <string.h>

#include "membrane/quant_select.h"
#include "precision_policy.h"
#include "test_helpers.h"
#include "workload_gen.h"

enum { ELEMS = 128 };

static void	test_kind_name_roundtrip(void)
{
	membrane_workload_kind_t	k;
	membrane_workload_kind_t	back;

	k = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	while (k < MEMBRANE_WORKLOAD_COUNT)
	{
		TEST_ASSERT(membrane_workload_kind_from_name(
				membrane_workload_kind_name(k), &back) == 1,
			"kind name resolves back to a kind");
		TEST_ASSERT(back == k, "roundtrip yields the same kind");
		k++;
	}
	TEST_ASSERT(membrane_workload_kind_from_name("not-a-kind", &back) == 0,
		"unknown name rejected");
	TEST_ASSERT(strcmp(MEMBRANE_WORKLOAD_KIND_LABEL, "synthetic") == 0,
		"the shared category label is exactly 'synthetic'");
}

static void	test_policy_name_roundtrip(void)
{
	membrane_bench_policy_t	p;
	membrane_bench_policy_t	back;

	p = MEMBRANE_BENCH_POLICY_Q4_ONLY;
	while (p < MEMBRANE_BENCH_POLICY_COUNT)
	{
		TEST_ASSERT(membrane_bench_policy_from_name(
				membrane_bench_policy_name(p), &back) == 1,
			"policy name resolves back to a policy");
		TEST_ASSERT(back == p, "roundtrip yields the same policy");
		p++;
	}
	TEST_ASSERT(membrane_bench_policy_from_name("bogus", &back) == 0,
		"unknown policy name rejected");
}

static void	test_generation_deterministic(void)
{
	uint16_t	a[ELEMS];
	uint16_t	b[ELEMS];

	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_MIXED,
		777, 42, a, ELEMS);
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_MIXED,
		777, 42, b, ELEMS);
	TEST_ASSERT(memcmp(a, b, sizeof(a)) == 0,
		"same kind/seed/block index yields bit-identical output");
}

static void	test_generation_varies_with_kind_and_index(void)
{
	uint16_t	default_kind[ELEMS];
	uint16_t	low_variance[ELEMS];
	uint16_t	block0[ELEMS];
	uint16_t	block1[ELEMS];

	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT,
		1234, 5, default_kind, ELEMS);
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE,
		1234, 5, low_variance, ELEMS);
	TEST_ASSERT(memcmp(default_kind, low_variance, sizeof(default_kind)) != 0,
		"different kinds produce different data for the same seed/index");
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT,
		1234, 0, block0, ELEMS);
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT,
		1234, 1, block1, ELEMS);
	TEST_ASSERT(memcmp(block0, block1, sizeof(block0)) != 0,
		"different block indices produce different data");
}

/* The four kinds are calibrated (see workload_gen.c) to produce
 * clearly different Q4 acceptance rates under the real engine -- this
 * is an end-to-end check that the calibration still holds, not a
 * hand-picked ratio. */
static void	test_kinds_produce_distinct_q4_rates(void)
{
	membrane_workload_kind_t		kinds[4];
	double							rates[4];
	membrane_quant_select_cfg_t	sel_cfg;
	membrane_quant_select_result_t	sel;
	uint16_t						x[ELEMS];
	uint32_t						b;
	uint64_t						q4;
	size_t							ki;

	kinds[0] = MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE;
	kinds[1] = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	kinds[2] = MEMBRANE_WORKLOAD_SYNTHETIC_MIXED;
	kinds[3] = MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE;
	sel_cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	ki = 0;
	while (ki < 4)
	{
		q4 = 0;
		b = 0;
		while (b < 300)
		{
			membrane_workload_generate_block(kinds[ki], 1234, b, x, ELEMS);
			TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR,
					x, ELEMS, &sel_cfg, &sel) == MEMBRANE_OK,
				"select succeeds");
			if (sel.precision == MEMBRANE_PRECISION_Q4)
				q4++;
			b++;
		}
		rates[ki] = (double)q4 / 300.0;
		ki++;
	}
	TEST_ASSERT(rates[0] > rates[1] && rates[1] > rates[2]
		&& rates[2] > rates[3],
		"low-variance > default > mixed > high-variance in Q4 rate");
}

static void	test_process_block_q4_only_forces_q4(void)
{
	membrane_workload_accum_t	acc;
	uint16_t					x[ELEMS];

	memset(&acc, 0, sizeof(acc));
	/* A noisy block that would normally fail the adaptive bound. */
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE,
		1, 0, x, ELEMS);
	TEST_ASSERT(membrane_bench_process_block(MEMBRANE_SIMD_SCALAR,
			MEMBRANE_BENCH_POLICY_Q4_ONLY, x, ELEMS,
			MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, &acc)
			== MEMBRANE_OK, "process succeeds");
	TEST_ASSERT(acc.q4_blocks == 1 && acc.q8_blocks == 0,
		"q4-only always forces Q4, even above the adaptive bound");
	TEST_ASSERT(acc.blocks_decoded == 1, "block decoded");
}

static void	test_process_block_q8_only_forces_q8(void)
{
	membrane_workload_accum_t	acc;
	uint16_t					x[ELEMS];

	memset(&acc, 0, sizeof(acc));
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE,
		1, 0, x, ELEMS);
	TEST_ASSERT(membrane_bench_process_block(MEMBRANE_SIMD_SCALAR,
			MEMBRANE_BENCH_POLICY_Q8_ONLY, x, ELEMS,
			MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, &acc)
			== MEMBRANE_OK, "process succeeds");
	TEST_ASSERT(acc.q4_blocks == 0 && acc.q8_blocks == 1,
		"q8-only always forces Q8, even when Q4 would have qualified");
}

static void	test_process_block_adaptive_matches_quant_select(void)
{
	membrane_workload_accum_t		acc;
	membrane_quant_select_cfg_t	sel_cfg;
	membrane_quant_select_result_t	sel;
	uint16_t						x[ELEMS];

	memset(&acc, 0, sizeof(acc));
	membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_MIXED,
		9, 3, x, ELEMS);
	sel_cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, ELEMS,
			&sel_cfg, &sel) == MEMBRANE_OK, "select succeeds");
	TEST_ASSERT(membrane_bench_process_block(MEMBRANE_SIMD_SCALAR,
			MEMBRANE_BENCH_POLICY_ADAPTIVE, x, ELEMS,
			MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, &acc)
			== MEMBRANE_OK, "process succeeds");
	if (sel.precision == MEMBRANE_PRECISION_Q4)
		TEST_ASSERT(acc.q4_blocks == 1, "adaptive agrees with quant_select");
	else
		TEST_ASSERT(acc.q8_blocks == 1, "adaptive agrees with quant_select");
}

static void	test_validation_pass_semantics(void)
{
	membrane_workload_accum_t	acc;

	memset(&acc, 0, sizeof(acc));
	acc.blocks_decoded = 10;
	TEST_ASSERT(membrane_workload_validation_pass(&acc, 10) == 1,
		"all decoded, no failures -> pass");
	acc.decode_failures = 1;
	TEST_ASSERT(membrane_workload_validation_pass(&acc, 10) == 0,
		"a decode failure -> fail");
	acc.decode_failures = 0;
	acc.encode_nondeterminism = 1;
	TEST_ASSERT(membrane_workload_validation_pass(&acc, 10) == 0,
		"nondeterministic encoding -> fail");
	acc.encode_nondeterminism = 0;
	TEST_ASSERT(membrane_workload_validation_pass(&acc, 11) == 0,
		"fewer decoded than expected -> fail");
}

static void	test_checked_mul_overflow(void)
{
	uint64_t	out;

	TEST_ASSERT(membrane_checked_mul_u64(1000, 1000, &out) == 1
		&& out == 1000000, "small product computes correctly");
	TEST_ASSERT(membrane_checked_mul_u64(UINT64_MAX, 2, &out) == 0,
		"overflowing product rejected");
	TEST_ASSERT(membrane_checked_mul_u64(0, UINT64_MAX, &out) == 1
		&& out == 0, "zero multiplicand never overflows");
}

/* Regression for a real bug: membrane_bench_packed_bytes's Q8_0 path
 * used to compute (elems / 32) * 34 with `elems` left as uint32_t, so
 * the multiplication happened in 32-bit unsigned arithmetic and wrapped
 * for elems near UINT32_MAX -- before the result ever reached its
 * size_t return type. Checked here against hand-computed 64-bit values
 * (no allocation of that size is ever attempted; this tests only the
 * arithmetic). The Q4_0 path (multiplier 18, not 34) can never
 * overflow uint32_t arithmetic for any valid uint32_t elems -- checked
 * too, so a regression in either direction would be caught. */
static void	test_packed_bytes_does_not_wrap_for_large_elems(void)
{
	uint32_t	huge_elems;

	/* Largest multiple of 32 <= UINT32_MAX. */
	huge_elems = (UINT32_MAX / MEMBRANE_QSIMD_BLOCK_ELEMS)
		* MEMBRANE_QSIMD_BLOCK_ELEMS;
	TEST_ASSERT(huge_elems == 4294967264u, "sanity: known constant");
	TEST_ASSERT(membrane_bench_packed_bytes(huge_elems, 0) == 4563402718ull,
		"Q8_0 packed_bytes is the true 64-bit product, not wrapped to "
		"268435422 as 32-bit arithmetic would give");
	TEST_ASSERT(membrane_bench_packed_bytes(huge_elems, 1) == 2415919086ull,
		"Q4_0 packed_bytes is correct (this path never overflows "
		"uint32_t arithmetic for any valid elems, unlike Q8_0)");
	TEST_ASSERT(membrane_bench_packed_bytes(128, 0) == 136,
		"ordinary small elems still computes correctly (4 blocks * 34)");
	TEST_ASSERT(membrane_bench_packed_bytes(128, 1) == 72,
		"ordinary small elems still computes correctly (4 blocks * 18)");
}

int	main(void)
{
	test_kind_name_roundtrip();
	test_policy_name_roundtrip();
	test_generation_deterministic();
	test_generation_varies_with_kind_and_index();
	test_kinds_produce_distinct_q4_rates();
	test_process_block_q4_only_forces_q4();
	test_process_block_q8_only_forces_q8();
	test_process_block_adaptive_matches_quant_select();
	test_validation_pass_semantics();
	test_checked_mul_overflow();
	test_packed_bytes_does_not_wrap_for_large_elems();
	return (0);
}
