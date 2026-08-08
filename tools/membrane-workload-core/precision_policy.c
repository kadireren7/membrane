#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/quant_select.h"
#include "precision_policy.h"

const char	*membrane_bench_policy_name(membrane_bench_policy_t p)
{
	if (p == MEMBRANE_BENCH_POLICY_Q4_ONLY)
		return ("q4-only");
	if (p == MEMBRANE_BENCH_POLICY_Q8_ONLY)
		return ("q8-only");
	return ("adaptive");
}

int	membrane_bench_policy_from_name(const char *name,
		membrane_bench_policy_t *out)
{
	if (name == NULL || out == NULL)
		return (0);
	if (strcmp(name, "q4-only") == 0)
		return (*out = MEMBRANE_BENCH_POLICY_Q4_ONLY, 1);
	if (strcmp(name, "q8-only") == 0)
		return (*out = MEMBRANE_BENCH_POLICY_Q8_ONLY, 1);
	if (strcmp(name, "adaptive") == 0)
		return (*out = MEMBRANE_BENCH_POLICY_ADAPTIVE, 1);
	return (0);
}

static void	err_stats_add(membrane_err_stats_t *s, double e)
{
	s->sum += e;
	if (e > s->max)
		s->max = e;
	s->count++;
}

double	membrane_err_stats_mean(const membrane_err_stats_t *s)
{
	if (s->count == 0)
		return (0.0);
	return (s->sum / (double)s->count);
}

int	membrane_workload_validation_pass(const membrane_workload_accum_t *acc,
		uint64_t blocks_expected)
{
	return (acc->decode_failures == 0 && acc->encode_nondeterminism == 0
		&& acc->blocks_decoded == blocks_expected);
}

/* Quantizes `x_f16` twice under the chosen precision (into packed_a/b,
 * for the encode-determinism check) and accounts the result. Returns
 * MEMBRANE_ERR_ALLOC_FAILED or whatever the maintained quantize engine
 * itself reported -- an infrastructure failure. */
static membrane_status_t	encode_block(membrane_simd_backend_t backend,
					const uint16_t *x_f16, uint32_t elems, int is_q4,
					uint8_t *packed_a, uint8_t *packed_b,
					size_t packed_bytes, membrane_workload_accum_t *acc)
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
	acc->encoded_bytes += packed_bytes;
	if (is_q4)
		acc->q4_blocks++;
	else
		acc->q8_blocks++;
	return (MEMBRANE_OK);
}

/* Soft-fail by design: a decode failure is reported via
 * acc->decode_failures, not propagated as a hard error. */
static void	decode_block(membrane_simd_backend_t backend,
					const uint16_t *x_f16, uint32_t elems, int is_q4,
					const uint8_t *packed_a, uint16_t *dec,
					membrane_workload_accum_t *acc)
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

static membrane_status_t	select_is_q4(membrane_simd_backend_t backend,
					membrane_bench_policy_t policy, const uint16_t *x_f16,
					uint32_t elems, double adaptive_max_q4_rel_l2_error,
					int *is_q4)
{
	membrane_quant_select_cfg_t		sel_cfg;
	membrane_quant_select_result_t		sel;
	membrane_status_t					st;

	if (policy == MEMBRANE_BENCH_POLICY_Q4_ONLY)
		return (*is_q4 = 1, MEMBRANE_OK);
	if (policy == MEMBRANE_BENCH_POLICY_Q8_ONLY)
		return (*is_q4 = 0, MEMBRANE_OK);
	sel_cfg.max_q4_rel_l2_error = adaptive_max_q4_rel_l2_error;
	st = membrane_quant_select_precision(backend, x_f16, elems, &sel_cfg,
			&sel);
	if (st != MEMBRANE_OK)
		return (st);
	*is_q4 = (sel.precision == MEMBRANE_PRECISION_Q4);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_bench_process_block(
						membrane_simd_backend_t backend,
						membrane_bench_policy_t policy,
						const uint16_t *x_f16, uint32_t elems,
						double adaptive_max_q4_rel_l2_error,
						membrane_workload_accum_t *acc)
{
	uint8_t				*packed_a;
	uint8_t				*packed_b;
	uint16_t			*dec;
	size_t				packed_bytes;
	int					is_q4;
	membrane_status_t	st;

	st = select_is_q4(backend, policy, x_f16, elems,
			adaptive_max_q4_rel_l2_error, &is_q4);
	if (st != MEMBRANE_OK)
		return (st);
	/* No overflow guard needed here (unlike membrane_quant_select_precision
	 * in src/quant/quant_select.c, whose `n` is a size_t taken directly
	 * from an external caller): `elems` is uint32_t, so elems/32*34 and
	 * elems*sizeof(uint16_t) are both bounded well under 2^33 -- many
	 * orders of magnitude short of overflowing a 64-bit size_t on every
	 * platform this project builds for. Confirmed: adding a SIZE_MAX
	 * check here is flagged -Wtype-limits "always false". */
	if (is_q4)
		packed_bytes = (elems / MEMBRANE_QSIMD_BLOCK_ELEMS)
			* MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	else
		packed_bytes = (elems / MEMBRANE_QSIMD_BLOCK_ELEMS)
			* MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES;
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
