#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "membrane/f16convert.h"
#include "membrane/quant_select.h"
#include "membrane/quant_simd.h"
#include "demo_core.h"

/* ------------------------------------------------------------------ */
/* Deterministic synthetic KV-like workload generator.                 */
/*                                                                      */
/* Each block gets a per-block DC bias that sweeps smoothly across      */
/* [0, 8) via a Weyl low-discrepancy sequence (phase-shifted by seed),  */
/* plus a fixed-amplitude smooth+noise ripple. A block whose bias       */
/* dominates its ripple quantizes to a small relative error under      */
/* Q4_0 (the scale is set by a value the block barely deviates from);   */
/* a near-zero-bias block does not, and needs Q8_0. This is SYNTHETIC   */
/* data chosen to exercise both branches of real Q4_0/Q8_0 quantization */
/* honestly -- it is not a captured or modeled real KV-cache trace.     */
/* ------------------------------------------------------------------ */

static uint32_t	xorshift32(uint32_t *state)
{
	uint32_t	x;

	x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return (x);
}

static float	block_bias(uint32_t seed, uint32_t blk_idx)
{
	double	t;

	t = fmod((double)blk_idx * 0.6180339887498949
			+ (double)(seed % 997u) / 997.0, 1.0);
	return ((float)(8.0 * t));
}

static void	generate_block(uint32_t seed, uint32_t blk_idx,
					uint16_t *out, uint32_t elems)
{
	uint32_t	rng;
	float		bias;
	float		noise;
	float		smooth;
	uint32_t	k;

	rng = (seed * 2654435761u) ^ (blk_idx * 0x9E3779B9u);
	if (rng == 0)
		rng = 1;
	bias = block_bias(seed, blk_idx);
	k = 0;
	while (k < elems)
	{
		noise = 2.0f * ((float)(xorshift32(&rng) & 0xFFFFu) / 65535.0f)
			- 1.0f;
		smooth = sinf((float)k * 0.15f + (float)blk_idx * 0.01f);
		out[k] = membrane_f32_to_f16(bias + 0.5f * smooth + 0.4f * noise);
		k++;
	}
}

/* ------------------------------------------------------------------ */
/* Overflow-checked size arithmetic.                                   */
/* ------------------------------------------------------------------ */

static int	checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (a != 0 && b > UINT64_MAX / a)
		return (0);
	*out = a * b;
	return (1);
}

/* ------------------------------------------------------------------ */
/* Per-precision running error stats.                                  */
/* ------------------------------------------------------------------ */

typedef struct s_err_stats
{
	double		sum;
	double		max;
	uint64_t	count;
}	err_stats_t;

static void	err_stats_add(err_stats_t *s, double e)
{
	s->sum += e;
	if (e > s->max)
		s->max = e;
	s->count++;
}

static double	err_stats_mean(const err_stats_t *s)
{
	if (s->count == 0)
		return (0.0);
	return (s->sum / (double)s->count);
}

typedef struct s_demo_accum
{
	uint64_t		q4_blocks;
	uint64_t		q8_blocks;
	uint64_t		membrane_bytes;
	uint64_t		blocks_decoded;
	uint64_t		decode_failures;
	uint64_t		encode_nondeterminism;
	err_stats_t		q4_err;
	err_stats_t		q8_err;
}	demo_accum_t;

/* One block's full encode -> account -> decode -> verify cycle for a
 * chosen precision. `packed_bytes` is the exact size the maintained
 * quant_simd block format writes for `elems` elements at this
 * precision (MEMBRANE_QSIMD_Q{4,8}_0_BLOCK_BYTES per 32-element group).
 */
/* Quantizes `x_f16` twice under the chosen precision (into packed_a/b,
 * for the encode-determinism check below) and accounts the result.
 * Returns MEMBRANE_ERR_ALLOC_FAILED or whatever the maintained quantize
 * engine itself reported on failure -- an infrastructure failure, hard
 * to recover from mid-workload, unlike a per-block decode failure
 * (see decode_block below, which is intentionally soft: reported via
 * acc->decode_failures, not a return-code failure).
 */
static membrane_status_t	encode_block(membrane_simd_backend_t backend,
					const uint16_t *x_f16, uint32_t elems, int is_q4,
					uint8_t *packed_a, uint8_t *packed_b,
					size_t packed_bytes, demo_accum_t *acc)
{
	membrane_status_t	st;

	if (is_q4)
		st = membrane_simd_q4_0_quantize(backend, x_f16, elems, packed_a);
	else
		st = membrane_simd_q8_0_quantize(backend, x_f16, elems, packed_a);
	if (st != MEMBRANE_OK)
		return (st);
	if (is_q4)
		st = membrane_simd_q4_0_quantize(backend, x_f16, elems, packed_b);
	else
		st = membrane_simd_q8_0_quantize(backend, x_f16, elems, packed_b);
	if (st != MEMBRANE_OK)
		return (st);
	if (memcmp(packed_a, packed_b, packed_bytes) != 0)
		acc->encode_nondeterminism++;
	acc->membrane_bytes += packed_bytes;
	if (is_q4)
		acc->q4_blocks++;
	else
		acc->q8_blocks++;
	return (MEMBRANE_OK);
}

/* Decode is intentionally soft-fail: per membrane_demo_run's contract,
 * a block that fails to decode is reported via acc->decode_failures
 * (and, in turn, out->validation_pass), not propagated as a hard
 * error -- the workload still completes and the result says why it
 * didn't validate. */
static void	decode_block(membrane_simd_backend_t backend,
					const uint16_t *x_f16, uint32_t elems, int is_q4,
					const uint8_t *packed_a, uint16_t *dec,
					demo_accum_t *acc)
{
	membrane_status_t	st;
	double				err;

	if (is_q4)
		st = membrane_simd_q4_0_dequantize(backend, packed_a, elems, dec);
	else
		st = membrane_simd_q8_0_dequantize(backend, packed_a, elems, dec);
	if (st != MEMBRANE_OK)
	{
		acc->decode_failures++;
		return ;
	}
	acc->blocks_decoded++;
	err = membrane_quant_rel_l2_error(x_f16, dec, elems);
	if (is_q4)
		err_stats_add(&acc->q4_err, err);
	else
		err_stats_add(&acc->q8_err, err);
}

static membrane_status_t	process_precision(membrane_simd_backend_t backend,
					const uint16_t *x_f16, uint32_t elems, int is_q4,
					size_t packed_bytes, demo_accum_t *acc)
{
	uint8_t				*packed_a;
	uint8_t				*packed_b;
	uint16_t			*dec;
	membrane_status_t	st;

	packed_a = malloc(packed_bytes);
	packed_b = malloc(packed_bytes);
	dec = malloc((size_t)elems * sizeof(uint16_t));
	if (packed_a == NULL || packed_b == NULL || dec == NULL)
	{
		free(packed_a);
		free(packed_b);
		free(dec);
		return (MEMBRANE_ERR_ALLOC_FAILED);
	}
	st = encode_block(backend, x_f16, elems, is_q4, packed_a, packed_b,
			packed_bytes, acc);
	if (st == MEMBRANE_OK)
		decode_block(backend, x_f16, elems, is_q4, packed_a, dec, acc);
	free(packed_a);
	free(packed_b);
	free(dec);
	return (st);
}

static membrane_status_t	run_workload(const membrane_demo_config_t *cfg,
					membrane_simd_backend_t backend, demo_accum_t *acc)
{
	membrane_quant_select_cfg_t		sel_cfg;
	membrane_quant_select_result_t		sel;
	uint16_t							x_f16[MEMBRANE_DEMO_ELEMS_PER_BLOCK];
	size_t								q4_bytes;
	size_t								q8_bytes;
	membrane_status_t					st;
	uint32_t							i;

	sel_cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	q4_bytes = (MEMBRANE_DEMO_ELEMS_PER_BLOCK / MEMBRANE_QSIMD_BLOCK_ELEMS)
		* MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	q8_bytes = (MEMBRANE_DEMO_ELEMS_PER_BLOCK / MEMBRANE_QSIMD_BLOCK_ELEMS)
		* MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES;
	i = 0;
	while (i < cfg->blocks)
	{
		generate_block(cfg->seed, i, x_f16, MEMBRANE_DEMO_ELEMS_PER_BLOCK);
		st = membrane_quant_select_precision(backend, x_f16,
			MEMBRANE_DEMO_ELEMS_PER_BLOCK, &sel_cfg, &sel);
		if (st != MEMBRANE_OK)
			return (st);
		if (sel.precision == MEMBRANE_PRECISION_Q4)
			st = process_precision(backend, x_f16, MEMBRANE_DEMO_ELEMS_PER_BLOCK,
				1, q4_bytes, acc);
		else
			st = process_precision(backend, x_f16, MEMBRANE_DEMO_ELEMS_PER_BLOCK,
				0, q8_bytes, acc);
		if (st != MEMBRANE_OK)
			return (st);
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_demo_run(const membrane_demo_config_t *cfg,
							membrane_demo_result_t *out)
{
	demo_accum_t				acc;
	membrane_simd_backend_t	backend;
	uint64_t					total_elems;
	struct timespec				t0;
	struct timespec				t1;
	membrane_status_t			st;

	if (cfg == NULL || out == NULL || cfg->blocks == 0
			|| cfg->blocks > MEMBRANE_DEMO_MAX_BLOCKS)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!checked_mul_u64(cfg->blocks, MEMBRANE_DEMO_ELEMS_PER_BLOCK,
			&total_elems))
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(&acc, 0, sizeof(acc));
	backend = membrane_simd_best_backend();
	clock_gettime(CLOCK_MONOTONIC, &t0);
	st = run_workload(cfg, backend, &acc);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	if (st != MEMBRANE_OK)
		return (st);
	memset(out, 0, sizeof(*out));
	out->config = *cfg;
	out->elems_per_block = MEMBRANE_DEMO_ELEMS_PER_BLOCK;
	out->simd_backend_name = membrane_simd_backend_name(backend);
	out->q4_blocks = acc.q4_blocks;
	out->q8_blocks = acc.q8_blocks;
	checked_mul_u64(total_elems, sizeof(float), &out->baseline_bytes);
	out->membrane_bytes = acc.membrane_bytes;
	out->saved_bytes = (int64_t)out->baseline_bytes
		- (int64_t)out->membrane_bytes;
	if (out->baseline_bytes > 0)
		out->reduction_ratio = (double)out->saved_bytes
			/ (double)out->baseline_bytes;
	out->blocks_decoded = acc.blocks_decoded;
	out->decode_failures = acc.decode_failures;
	out->encode_nondeterminism = acc.encode_nondeterminism;
	out->q4_mean_rel_l2_error = err_stats_mean(&acc.q4_err);
	out->q4_max_rel_l2_error = acc.q4_err.max;
	out->q8_mean_rel_l2_error = err_stats_mean(&acc.q8_err);
	out->q8_max_rel_l2_error = acc.q8_err.max;
	out->elapsed_seconds = (double)(t1.tv_sec - t0.tv_sec)
		+ (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
	out->validation_pass = (out->decode_failures == 0
			&& out->encode_nondeterminism == 0
			&& out->blocks_decoded == cfg->blocks);
	return (MEMBRANE_OK);
}
