/*
 * Phase 4.4 bit-parity tests: membrane_ggml_quant's output must be
 * byte-for-byte identical to calling ggml's own quantize_row_q8_0 /
 * quantize_row_q4_0 / dequantize_row_q8_0 / dequantize_row_q4_0
 * directly (see src/quant/ggml_quant.c for exactly which ggml symbols
 * and why). This file links against ggml itself specifically so it can
 * make that direct call independently and compare -- it is NOT testing
 * ggml's correctness (ggml is the trusted reference here), it is
 * testing that membrane_ggml_quant's adapter plumbing (F16<->F32
 * boundary conversion, block-size bookkeeping) introduces zero
 * deviation from calling ggml directly.
 */
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "quants.h"

#include "membrane/ggml_quant.h"

#include "test_helpers.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static uint16_t	f16_of(float f)
{
	return (ggml_fp32_to_fp16(f));
}

static void	report_mismatch(const char *what, size_t block_idx,
				size_t elem_idx, const float *input, size_t n,
				uint32_t expected, uint32_t actual)
{
	size_t	lo;
	size_t	hi;
	size_t	i;

	fprintf(stderr, "FAIL: %s mismatch at block %zu elem %zu\n", what,
		block_idx, elem_idx);
	fprintf(stderr, "  expected: 0x%08x  actual: 0x%08x\n", expected,
		actual);
	lo = elem_idx > 4 ? elem_idx - 4 : 0;
	hi = elem_idx + 4 < n ? elem_idx + 4 : n - 1;
	fprintf(stderr, "  input window [%zu..%zu]:", lo, hi);
	i = lo;
	while (i <= hi)
	{
		fprintf(stderr, " %.6g", (double)input[i]);
		i++;
	}
	fprintf(stderr, "\n");
	abort();
}

/* Quantizes `x` (n elements, n a multiple of 32) via membrane_ggml_quant
 * and, independently, via a direct ggml call, and asserts the packed
 * bytes are identical, block by block.
 *
 * `x` is first rounded to F16 (membrane_ggml_quant's real input type,
 * matching what a captured KV cache blob actually contains) and BOTH
 * paths are fed that SAME rounded value -- not the original `x` -- so
 * an `x` that is not exactly F16-representable (almost all of them)
 * cannot manufacture a spurious mismatch purely from the adapter's own,
 * expected, F16->F32 boundary conversion. That boundary conversion is
 * real (every KV cache value already went through it before this
 * function is ever called in production), so it is deliberately
 * included on both sides rather than avoided. */
static void	check_quantize_parity(const char *label, const float *x,
				size_t n, int bits)
{
	uint16_t	*x_f16;
	float		*x_f32;
	size_t		block_bytes;
	size_t		nblocks;
	uint8_t		*via_adapter;
	uint8_t		*via_direct;
	size_t		bi;
	size_t		i;

	x_f16 = malloc(n * sizeof(uint16_t));
	x_f32 = malloc(n * sizeof(float));
	TEST_ASSERT(x_f16 != NULL && x_f32 != NULL, "x_f16/x_f32 allocate");
	i = 0;
	while (i < n)
	{
		x_f16[i] = f16_of(x[i]);
		i++;
	}
	ggml_fp16_to_fp32_row((const ggml_fp16_t *)x_f16, x_f32, (int64_t)n);
	block_bytes = bits == 8 ? MEMBRANE_GGML_Q8_0_BLOCK_BYTES
		: MEMBRANE_GGML_Q4_0_BLOCK_BYTES;
	nblocks = n / MEMBRANE_GGML_QBLOCK_ELEMS;
	via_adapter = malloc(nblocks * block_bytes);
	via_direct = malloc(nblocks * block_bytes);
	TEST_ASSERT(via_adapter != NULL && via_direct != NULL,
		"quantize output buffers allocate");
	if (bits == 8)
	{
		TEST_ASSERT(membrane_ggml_q8_0_quantize(x_f16, n, via_adapter)
			== MEMBRANE_OK, label);
		quantize_row_q8_0(x_f32, via_direct, (int64_t)n);
	}
	else
	{
		TEST_ASSERT(membrane_ggml_q4_0_quantize(x_f16, n, via_adapter)
			== MEMBRANE_OK, label);
		quantize_row_q4_0(x_f32, via_direct, (int64_t)n);
	}
	bi = 0;
	while (bi < nblocks * block_bytes)
	{
		if (via_adapter[bi] != via_direct[bi])
			report_mismatch(label, bi / block_bytes, bi % block_bytes,
				x_f32, n, via_direct[bi], via_adapter[bi]);
		bi++;
	}
	free(x_f16);
	free(x_f32);
	free(via_adapter);
	free(via_direct);
}

/* Same F16-first fairness fix as check_quantize_parity: dequantizes a
 * blob (produced from the F16-rounded `x`) via membrane_ggml_quant and
 * compares against a direct ggml dequantize + re-encode of the SAME
 * blob -- must match exactly. */
static void	check_dequantize_parity(const char *label, const float *x,
				size_t n, int bits)
{
	uint16_t	*x_f16;
	float		*x_f32;
	size_t		block_bytes;
	size_t		nblocks;
	uint8_t		*packed;
	uint16_t	*via_adapter;
	float		*via_direct_f32;
	size_t		i;

	x_f16 = malloc(n * sizeof(uint16_t));
	x_f32 = malloc(n * sizeof(float));
	TEST_ASSERT(x_f16 != NULL && x_f32 != NULL, "x_f16/x_f32 allocate");
	i = 0;
	while (i < n)
	{
		x_f16[i] = f16_of(x[i]);
		i++;
	}
	ggml_fp16_to_fp32_row((const ggml_fp16_t *)x_f16, x_f32, (int64_t)n);
	block_bytes = bits == 8 ? MEMBRANE_GGML_Q8_0_BLOCK_BYTES
		: MEMBRANE_GGML_Q4_0_BLOCK_BYTES;
	nblocks = n / MEMBRANE_GGML_QBLOCK_ELEMS;
	packed = malloc(nblocks * block_bytes);
	via_adapter = malloc(n * sizeof(uint16_t));
	via_direct_f32 = malloc(n * sizeof(float));
	TEST_ASSERT(packed && via_adapter && via_direct_f32,
		"dequantize buffers allocate");
	if (bits == 8)
	{
		quantize_row_q8_0(x_f32, packed, (int64_t)n);
		TEST_ASSERT(membrane_ggml_q8_0_dequantize(packed, n, via_adapter)
			== MEMBRANE_OK, label);
		dequantize_row_q8_0((const block_q8_0 *)packed, via_direct_f32,
			(int64_t)n);
	}
	else
	{
		quantize_row_q4_0(x_f32, packed, (int64_t)n);
		TEST_ASSERT(membrane_ggml_q4_0_dequantize(packed, n, via_adapter)
			== MEMBRANE_OK, label);
		dequantize_row_q4_0((const block_q4_0 *)packed, via_direct_f32,
			(int64_t)n);
	}
	i = 0;
	while (i < n)
	{
		uint16_t	expected;

		expected = f16_of(via_direct_f32[i]);
		if (via_adapter[i] != expected)
			report_mismatch(label, i / MEMBRANE_GGML_QBLOCK_ELEMS,
				i % MEMBRANE_GGML_QBLOCK_ELEMS, via_direct_f32, n, expected,
				via_adapter[i]);
		i++;
	}
	free(x_f16);
	free(x_f32);
	free(packed);
	free(via_adapter);
	free(via_direct_f32);
}

static void	check_both(const char *label, const float *x, size_t n)
{
	check_quantize_parity(label, x, n, 8);
	check_quantize_parity(label, x, n, 4);
	check_dequantize_parity(label, x, n, 8);
	check_dequantize_parity(label, x, n, 4);
}

static void	test_all_zero(void)
{
	float	x[64];

	memset(x, 0, sizeof(x));
	check_both("all-zero", x, 64);
}

static void	test_constant(void)
{
	float	x[64];
	size_t	i;

	i = 0;
	while (i < 64)
	{
		x[i] = 3.5f;
		i++;
	}
	check_both("constant", x, 64);
}

static void	test_extrema(void)
{
	float	x[32];
	size_t	i;

	i = 0;
	while (i < 32)
	{
		x[i] = (i % 2 == 0) ? 65504.0f : -65504.0f;
		i++;
	}
	check_both("extrema-max-f16", x, 32);
	i = 0;
	while (i < 32)
	{
		x[i] = (i % 2 == 0) ? 1e-6f : -1e-6f;
		i++;
	}
	check_both("extrema-near-zero", x, 32);
}

/* Constructs a block with amax=127.0 exactly (so d=id=1.0 and every
 * scaled value equals the original value exactly) with several other
 * elements set to exact x.5 half-integers -- the one case where
 * roundf() (round half away from zero, ggml's scalar
 * quantize_row_q8_0_ref) and IEEE round-half-to-even (ggml's AVX2
 * quantize_row_q8_0, used on this build) can disagree. This test
 * proves membrane_ggml_quant's adapter matches whichever one it calls
 * (quantize_row_q8_0, the real CPU-backend path); a SEPARATE, explicit
 * comparison against quantize_row_q8_0_ref follows to measure (not
 * assume) whether the two ggml-internal paths actually diverge on this
 * build, for docs/phase4-ggml-quant-parity.md. */
static void	test_halfway_rounding(void)
{
	float	x[32];
	size_t	i;

	i = 0;
	while (i < 32)
	{
		x[i] = 0.0f;
		i++;
	}
	x[0] = 127.0f;
	x[1] = 2.5f;
	x[2] = -2.5f;
	x[3] = 3.5f;
	x[4] = -3.5f;
	x[5] = 0.5f;
	x[6] = -0.5f;
	x[7] = 60.5f;
	check_both("halfway-rounding", x, 32);
}

/* Not an assertion -- a measurement. Reports, for the same halfway-
 * boundary block, whether quantize_row_q8_0 (real CPU-backend path,
 * what membrane_ggml_quant calls) and quantize_row_q8_0_ref (ggml's
 * separate "deterministic model file" scalar path) actually produce
 * different bytes on this build. This is the concrete, measured answer
 * to Phase 4.4 item 3's "halfway rounding boundary" requirement. */
static void	report_ref_vs_cpu_backend_rounding(void)
{
	float	x[32];
	uint8_t	via_cpu_backend[MEMBRANE_GGML_Q8_0_BLOCK_BYTES];
	uint8_t	via_ref[MEMBRANE_GGML_Q8_0_BLOCK_BYTES];
	size_t	i;
	bool	differs;

	i = 0;
	while (i < 32)
	{
		x[i] = 0.0f;
		i++;
	}
	x[0] = 127.0f;
	x[1] = 2.5f;
	x[2] = -2.5f;
	x[3] = 3.5f;
	x[4] = -3.5f;
	x[5] = 0.5f;
	x[6] = -0.5f;
	x[7] = 60.5f;
	quantize_row_q8_0(x, via_cpu_backend, 32);
	quantize_row_q8_0_ref(x, (block_q8_0 *)(void *)via_ref, 32);
	differs = memcmp(via_cpu_backend, via_ref,
			MEMBRANE_GGML_Q8_0_BLOCK_BYTES) != 0;
	fprintf(stderr, "MEASURED: quantize_row_q8_0 (CPU backend, used by "
		"membrane_ggml_quant) vs quantize_row_q8_0_ref (GGUF-file "
		"reference) on a halfway-rounding block: %s\n",
		differs ? "DIFFER (as expected -- round-to-even vs "
			"round-half-away-from-zero)" : "IDENTICAL on this build");
	if (differs)
	{
		i = 0;
		while (i < 32)
		{
			if (via_cpu_backend[2 + i] != (uint8_t)via_ref[2 + i])
				fprintf(stderr, "  qs[%zu]: cpu-backend=%d ref=%d "
					"(input %.2f)\n", i, (int8_t)via_cpu_backend[2 + i],
					(int8_t)via_ref[2 + i], (double)x[i]);
			i++;
		}
	}
}

static void	test_nan_inf(void)
{
	float	x[32];
	size_t	i;

	i = 0;
	while (i < 32)
	{
		x[i] = 1.0f;
		i++;
	}
	x[5] = (float)NAN;
	check_quantize_parity("nan-in-block", x, 32, 8);
	check_quantize_parity("nan-in-block", x, 32, 4);
	i = 0;
	while (i < 32)
	{
		x[i] = 1.0f;
		i++;
	}
	x[7] = (float)INFINITY;
	x[9] = (float)-INFINITY;
	check_quantize_parity("inf-in-block", x, 32, 8);
	check_quantize_parity("inf-in-block", x, 32, 4);
}

static void	test_denormal(void)
{
	float	x[32];
	size_t	i;

	i = 0;
	while (i < 32)
	{
		x[i] = 5.9604645e-8f * (float)(i + 1);
		i++;
	}
	check_both("denormal-f16-subnormal", x, 32);
}

static void	test_random_blocks(uint32_t seed, size_t nblocks)
{
	float		*x;
	uint32_t	state;
	size_t		i;

	x = malloc(nblocks * MEMBRANE_GGML_QBLOCK_ELEMS * sizeof(float));
	TEST_ASSERT(x != NULL, "random buffer allocates");
	state = seed;
	i = 0;
	while (i < nblocks * MEMBRANE_GGML_QBLOCK_ELEMS)
	{
		uint32_t	r;
		int32_t		mantissa;
		float		v;

		r = test_rand_next(&state);
		mantissa = (int32_t)(r % 2000001) - 1000000;
		v = (float)mantissa / 1000.0f;
		if ((r >> 24) % 16 == 0)
			v *= 1000.0f;
		x[i] = v;
		i++;
	}
	check_quantize_parity("random-blocks", x, nblocks * 32, 8);
	check_quantize_parity("random-blocks", x, nblocks * 32, 4);
	check_dequantize_parity("random-blocks", x, nblocks * 32, 8);
	check_dequantize_parity("random-blocks", x, nblocks * 32, 4);
	free(x);
}

int	main(void)
{
	test_all_zero();
	test_constant();
	test_extrema();
	test_halfway_rounding();
	report_ref_vs_cpu_backend_rounding();
	test_nan_inf();
	test_denormal();
	test_random_blocks(0xC0FFEEu, 100000);
	fprintf(stderr, "all ggml quant parity tests passed (100000+ random "
		"blocks, all-zero, constant, extrema, NaN/Inf, denormal)\n");
	return (0);
}
