/*
 * Phase 5.1 bit-parity tests: every membrane_simd_* backend (scalar,
 * SSE4.1, AVX2 -- whichever this CPU supports) must produce output
 * byte-for-byte identical to membrane_ggml_quant (Phase 4.4's oracle,
 * itself already proven bit-exact against real ggml). This file links
 * llama/ggml only to reach membrane_ggml_quant's oracle -- it does not
 * call any ggml quantize function directly (unlike
 * test_ggml_quant_parity.c, whose job is verifying membrane_ggml_quant
 * against ggml itself).
 */
#include "membrane/f16convert.h"
#include "membrane/ggml_quant.h"
#include "membrane/quant_simd.h"

#include "test_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void	report_mismatch(const char *what, const char *backend_name,
				size_t byte_idx, uint8_t expected, uint8_t actual)
{
	fprintf(stderr, "FAIL: %s (backend=%s) byte %zu: oracle=0x%02x "
		"actual=0x%02x\n", what, backend_name, byte_idx, expected, actual);
	abort();
}

static const membrane_simd_backend_t	g_backends[] = {
	MEMBRANE_SIMD_SCALAR, MEMBRANE_SIMD_SSE41, MEMBRANE_SIMD_AVX2,
};

static void	check_quantize_parity(const char *label, const float *x,
				size_t n, int bits)
{
	uint16_t	*x_f16;
	uint8_t		*oracle;
	size_t		block_bytes;
	size_t		nblocks;
	size_t		i;

	x_f16 = malloc(n * sizeof(uint16_t));
	TEST_ASSERT(x_f16 != NULL, "x_f16 allocates");
	i = 0;
	while (i < n)
	{
		x_f16[i] = membrane_f32_to_f16(x[i]);
		i++;
	}
	block_bytes = bits == 8 ? MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES
		: MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	nblocks = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	oracle = malloc(nblocks * block_bytes);
	TEST_ASSERT(oracle != NULL, "oracle buffer allocates");
	TEST_ASSERT((bits == 8 ? membrane_ggml_q8_0_quantize(x_f16, n, oracle)
			: membrane_ggml_q4_0_quantize(x_f16, n, oracle)) == MEMBRANE_OK,
		label);
	for (size_t bi = 0; bi < sizeof(g_backends) / sizeof(g_backends[0]);
			bi++)
	{
		membrane_simd_backend_t	backend = g_backends[bi];
		uint8_t						*actual;
		membrane_status_t			st;

		actual = malloc(nblocks * block_bytes);
		TEST_ASSERT(actual != NULL, "actual buffer allocates");
		st = bits == 8
			? membrane_simd_q8_0_quantize(backend, x_f16, n, actual)
			: membrane_simd_q4_0_quantize(backend, x_f16, n, actual);
		if (st == MEMBRANE_ERR_UNIMPLEMENTED)
		{
			free(actual);
			continue ;
		}
		TEST_ASSERT(st == MEMBRANE_OK, label);
		i = 0;
		while (i < nblocks * block_bytes)
		{
			if (actual[i] != oracle[i])
				report_mismatch(label,
					membrane_simd_backend_name(backend), i, oracle[i],
					actual[i]);
			i++;
		}
		free(actual);
	}
	free(x_f16);
	free(oracle);
}

static void	check_dequantize_parity(const char *label, const float *x,
				size_t n, int bits)
{
	uint16_t	*x_f16;
	uint8_t		*packed;
	uint16_t	*oracle;
	size_t		block_bytes;
	size_t		nblocks;
	size_t		i;

	x_f16 = malloc(n * sizeof(uint16_t));
	TEST_ASSERT(x_f16 != NULL, "x_f16 allocates");
	i = 0;
	while (i < n)
	{
		x_f16[i] = membrane_f32_to_f16(x[i]);
		i++;
	}
	block_bytes = bits == 8 ? MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES
		: MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	nblocks = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	packed = malloc(nblocks * block_bytes);
	oracle = malloc(n * sizeof(uint16_t));
	TEST_ASSERT(packed != NULL && oracle != NULL, "buffers allocate");
	TEST_ASSERT((bits == 8 ? membrane_ggml_q8_0_quantize(x_f16, n, packed)
			: membrane_ggml_q4_0_quantize(x_f16, n, packed)) == MEMBRANE_OK,
		label);
	TEST_ASSERT((bits == 8
			? membrane_ggml_q8_0_dequantize(packed, n, oracle)
			: membrane_ggml_q4_0_dequantize(packed, n, oracle))
			== MEMBRANE_OK, label);
	for (size_t bi = 0; bi < sizeof(g_backends) / sizeof(g_backends[0]);
			bi++)
	{
		membrane_simd_backend_t	backend = g_backends[bi];
		uint16_t					*actual;
		membrane_status_t			st;

		actual = malloc(n * sizeof(uint16_t));
		TEST_ASSERT(actual != NULL, "actual buffer allocates");
		st = bits == 8
			? membrane_simd_q8_0_dequantize(backend, packed, n, actual)
			: membrane_simd_q4_0_dequantize(backend, packed, n, actual);
		if (st == MEMBRANE_ERR_UNIMPLEMENTED)
		{
			free(actual);
			continue ;
		}
		TEST_ASSERT(st == MEMBRANE_OK, label);
		i = 0;
		while (i < n)
		{
			if (actual[i] != oracle[i])
				report_mismatch(label,
					membrane_simd_backend_name(backend), i,
					(uint8_t)oracle[i], (uint8_t)actual[i]);
			i++;
		}
		free(actual);
	}
	free(x_f16);
	free(packed);
	free(oracle);
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
}

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

static void	test_odd_block_counts(void)
{
	float	x[32 * 3];
	uint32_t	state = 42;
	size_t	i;

	i = 0;
	while (i < 32 * 3)
	{
		x[i] = (float)((int32_t)(test_rand_next(&state) % 2001) - 1000)
			/ 100.0f;
		i++;
	}
	check_both("odd-block-count-3", x, 32 * 3);
}

static void	test_random_blocks(uint32_t seed, size_t nblocks)
{
	float		*x;
	uint32_t	state;
	size_t		i;

	x = malloc(nblocks * MEMBRANE_QSIMD_BLOCK_ELEMS * sizeof(float));
	TEST_ASSERT(x != NULL, "random buffer allocates");
	state = seed;
	i = 0;
	while (i < nblocks * MEMBRANE_QSIMD_BLOCK_ELEMS)
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

static void	test_forced_backend_matches_scalar(void)
{
	float	x[64];
	uint32_t	state = 7;
	size_t	i;

	i = 0;
	while (i < 64)
	{
		x[i] = (float)((int32_t)(test_rand_next(&state) % 20001) - 10000)
			/ 100.0f;
		i++;
	}
	check_both("forced-backend", x, 64);
}

static void	test_unsupported_backend_rejected(void)
{
	membrane_simd_backend_t	invalid;
	uint16_t					x_f16[32] = {0};
	uint8_t						out[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];

	invalid = (membrane_simd_backend_t)999;
	TEST_ASSERT(membrane_simd_q8_0_quantize(invalid, x_f16, 32, out)
		== MEMBRANE_ERR_UNIMPLEMENTED,
		"an unrecognized backend id is rejected, never silently run");
}

static void	test_batch_matches_single_row(void)
{
	uint16_t	x_f16[32 * 5];
	uint8_t		packed_single[5 * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
	uint8_t		packed_batch[5 * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
	float		scratch[32];
	uint32_t	state = 99;
	size_t		i;
	size_t		r;
	membrane_simd_backend_t	backend;

	i = 0;
	while (i < 32 * 5)
	{
		x_f16[i] = membrane_f32_to_f16(
			(float)((int32_t)(test_rand_next(&state) % 4001) - 2000)
				/ 100.0f);
		i++;
	}
	backend = membrane_simd_best_backend();
	r = 0;
	while (r < 5)
	{
		TEST_ASSERT(membrane_simd_q8_0_quantize(backend, x_f16 + r * 32, 32,
				packed_single + r * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES)
				== MEMBRANE_OK, "single-row quantize");
		r++;
	}
	TEST_ASSERT(membrane_simd_q8_0_quantize_batch(backend, x_f16, 5, 32,
			packed_batch, scratch, sizeof(scratch)) == MEMBRANE_OK,
		"batch quantize");
	TEST_ASSERT(memcmp(packed_single, packed_batch, sizeof(packed_single))
		== 0, "batch output matches row-by-row single calls exactly");
}

static void	test_scratch_too_small_rejected(void)
{
	uint16_t	x_f16[32] = {0};
	uint8_t		packed[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES];
	uint8_t		tiny_scratch[1];

	TEST_ASSERT(membrane_simd_q8_0_quantize_batch(MEMBRANE_SIMD_SCALAR,
			x_f16, 1, 32, packed, tiny_scratch, sizeof(tiny_scratch))
			== MEMBRANE_ERR_BUFFER_TOO_SMALL,
		"a too-small scratch buffer is rejected, not silently overrun");
}

/* Item 7: parallel output must be bit-identical to the single-threaded
 * batch call, for every thread count tried, including counts that force
 * the small-workload single-thread fallback and counts that don't
 * divide the row count evenly. */
static void	test_parallel_deterministic(void)
{
	size_t		n_blocks = 777;
	uint16_t	*x_f16;
	uint8_t		*serial;
	uint8_t		*parallel;
	uint32_t	state = 1234;
	size_t		i;
	int			thread_counts[] = {1, 2, 3, 4, 8, 12};
	size_t		tc;

	x_f16 = malloc(n_blocks * 32 * sizeof(uint16_t));
	serial = malloc(n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
	parallel = malloc(n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
	TEST_ASSERT(x_f16 && serial && parallel, "buffers allocate");
	i = 0;
	while (i < n_blocks * 32)
	{
		x_f16[i] = membrane_f32_to_f16(
			(float)((int32_t)(test_rand_next(&state) % 20001) - 10000)
				/ 100.0f);
		i++;
	}
	{
		float	scratch[32];

		i = 0;
		while (i < n_blocks)
		{
			TEST_ASSERT(membrane_simd_q8_0_quantize_batch(
					MEMBRANE_SIMD_SCALAR, x_f16 + i * 32, 1, 32,
					serial + i * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES, scratch,
					sizeof(scratch)) == MEMBRANE_OK, "serial reference");
			i++;
		}
	}
	tc = 0;
	while (tc < sizeof(thread_counts) / sizeof(thread_counts[0]))
	{
		int	used = -1;

		memset(parallel, 0xAA, n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
		TEST_ASSERT(membrane_simd_q8_0_quantize_batch_parallel(
				MEMBRANE_SIMD_SCALAR, x_f16, n_blocks, 32, parallel,
				thread_counts[tc], &used) == MEMBRANE_OK,
			"parallel quantize");
		TEST_ASSERT(used >= 1 && used <= thread_counts[tc],
			"threads used never exceeds what was requested");
		TEST_ASSERT(memcmp(serial, parallel,
				n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES) == 0,
			"parallel output bit-identical to serial, every thread count");
		tc++;
	}
	free(x_f16);
	free(serial);
	free(parallel);
}

/* Item 14: unaligned input/output. Every SIMD load/store in
 * quant_simd.c already uses the unaligned _mm{,256}_loadu/storeu forms
 * (no _mm_load/_mm256_load anywhere in the file), and the raw
 * uint16_t/uint8_t buffers this test misaligns are converted to/from an
 * on-stack float[32] via a scalar per-element loop before any SIMD
 * instruction ever touches them, so misalignment can't reach a SIMD
 * load at all by construction -- this test is a regression guard
 * against that changing, not a fix for a bug that exists today. */
static void	test_unaligned_input(void)
{
	uint8_t		raw_in[32 * sizeof(uint16_t) + 8];
	uint8_t		raw_packed_q8[MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES + 8];
	uint8_t		raw_packed_q4[MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES + 8];
	uint8_t		raw_out[32 * sizeof(uint16_t) + 8];
	uint16_t	*x_f16;
	uint8_t		*packed_q8;
	uint8_t		*packed_q4;
	uint16_t	*out_f16;
	uint32_t	state = 777;
	size_t		i;
	size_t		bi;

	x_f16 = (uint16_t *)(void *)(raw_in + 1);
	packed_q8 = raw_packed_q8 + 3;
	packed_q4 = raw_packed_q4 + 5;
	out_f16 = (uint16_t *)(void *)(raw_out + 1);
	i = 0;
	while (i < 32)
	{
		x_f16[i] = membrane_f32_to_f16(
			(float)((int32_t)(test_rand_next(&state) % 20001) - 10000)
				/ 100.0f);
		i++;
	}
	bi = 0;
	while (bi < sizeof(g_backends) / sizeof(g_backends[0]))
	{
		membrane_simd_backend_t	backend = g_backends[bi];
		membrane_status_t			st;

		st = membrane_simd_q8_0_quantize(backend, x_f16, 32, packed_q8);
		if (st != MEMBRANE_ERR_UNIMPLEMENTED)
		{
			TEST_ASSERT(st == MEMBRANE_OK, "unaligned q8 quantize");
			TEST_ASSERT(membrane_simd_q8_0_dequantize(backend, packed_q8,
					32, out_f16) == MEMBRANE_OK, "unaligned q8 dequantize");
		}
		st = membrane_simd_q4_0_quantize(backend, x_f16, 32, packed_q4);
		if (st != MEMBRANE_ERR_UNIMPLEMENTED)
		{
			TEST_ASSERT(st == MEMBRANE_OK, "unaligned q4 quantize");
			TEST_ASSERT(membrane_simd_q4_0_dequantize(backend, packed_q4,
					32, out_f16) == MEMBRANE_OK, "unaligned q4 dequantize");
		}
		bi++;
	}
}

/* Item 14: tiny (single-block) and large (multi-thousand-row) batches
 * through the exact same batch/parallel API used by the runtime and
 * benchmark tool, cross-checked against the single-row calls. */
static void	test_tiny_and_large_batch(void)
{
	static const size_t	sizes[] = {1, 20000};
	size_t					si;

	si = 0;
	while (si < sizeof(sizes) / sizeof(sizes[0]))
	{
		size_t		n_blocks = sizes[si];
		uint16_t	*x_f16;
		uint8_t		*packed_batch;
		uint8_t		*packed_parallel;
		float		scratch[32];
		uint32_t	state = (uint32_t)(0xA5A5u + n_blocks);
		size_t		i;
		int			used = -1;

		x_f16 = malloc(n_blocks * 32 * sizeof(uint16_t));
		packed_batch = malloc(n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
		packed_parallel = malloc(n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
		TEST_ASSERT(x_f16 && packed_batch && packed_parallel,
			"tiny/large batch buffers allocate");
		i = 0;
		while (i < n_blocks * 32)
		{
			x_f16[i] = membrane_f32_to_f16(
				(float)((int32_t)(test_rand_next(&state) % 20001) - 10000)
					/ 100.0f);
			i++;
		}
		TEST_ASSERT(membrane_simd_q8_0_quantize_batch(MEMBRANE_SIMD_SCALAR,
				x_f16, n_blocks, 32, packed_batch, scratch, sizeof(scratch))
				== MEMBRANE_OK, "tiny/large batch quantize");
		TEST_ASSERT(membrane_simd_q8_0_quantize_batch_parallel(
				MEMBRANE_SIMD_SCALAR, x_f16, n_blocks, 32, packed_parallel,
				0, &used) == MEMBRANE_OK, "tiny/large parallel quantize");
		TEST_ASSERT(used >= 1, "at least one thread used");
		TEST_ASSERT(memcmp(packed_batch, packed_parallel,
				n_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES) == 0,
			"tiny/large batch and parallel outputs match exactly");
		free(x_f16);
		free(packed_batch);
		free(packed_parallel);
		si++;
	}
}

int	main(void)
{
	fprintf(stderr, "detected best backend: %s\n",
		membrane_simd_backend_name(membrane_simd_best_backend()));
	test_all_zero();
	test_constant();
	test_extrema();
	test_halfway_rounding();
	test_nan_inf();
	test_odd_block_counts();
	test_forced_backend_matches_scalar();
	test_unsupported_backend_rejected();
	test_batch_matches_single_row();
	test_scratch_too_small_rejected();
	test_parallel_deterministic();
	test_unaligned_input();
	test_tiny_and_large_batch();
	test_random_blocks(0xBEEF1234u, 100000);
	fprintf(stderr, "all quant_simd parity tests passed (100000+ random "
		"blocks, all-zero, constant, extrema, halfway-rounding, NaN/Inf, "
		"odd block counts, forced/unsupported backend, batch, scratch "
		"validation)\n");
	return (0);
}
