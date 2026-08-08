#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "membrane/quant_select.h"
#include "bench_core.h"

static uint64_t	now_ns(void)
{
	struct timespec	ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec);
}

static int	cmp_double(const void *a, const void *b)
{
	double	da;
	double	db;

	da = *(const double *)a;
	db = *(const double *)b;
	if (da < db)
		return (-1);
	if (da > db)
		return (1);
	return (0);
}

/* n >= 1, samples sorted ascending on return. */
static double	median_of(double *samples, size_t n)
{
	qsort(samples, n, sizeof(*samples), cmp_double);
	if (n % 2 == 1)
		return (samples[n / 2]);
	return ((samples[n / 2 - 1] + samples[n / 2]) / 2.0);
}

/* membrane_workload_kind_t/membrane_bench_policy_t have only
 * non-negative enumerators, so this compiler (verified: adding an
 * explicit lower-bound check here is flagged -Wtype-limits "always
 * true", i.e. the compiler itself confirms it) represents them as
 * unsigned int -- a bit pattern written via an out-of-range cast such
 * as (membrane_workload_kind_t)-1 reads back as a large positive value
 * here, which this single upper-bound check already rejects. */
static int	cfg_valid(const membrane_bench_config_t *cfg)
{
	return (cfg->blocks != 0 && cfg->blocks <= MEMBRANE_BENCH_MAX_BLOCKS
		&& cfg->iterations != 0
		&& cfg->iterations <= MEMBRANE_BENCH_MAX_ITERATIONS
		&& cfg->warmup <= MEMBRANE_BENCH_MAX_WARMUP
		&& cfg->workload < MEMBRANE_WORKLOAD_COUNT
		&& cfg->policy < MEMBRANE_BENCH_POLICY_COUNT);
}

/* One full pass over every pre-generated block, timed. Fills *acc (any
 * prior contents are overwritten via memset). */
static uint64_t	timed_pass(const membrane_bench_config_t *cfg,
					membrane_simd_backend_t backend, const uint16_t *blocks,
					membrane_workload_accum_t *acc, membrane_status_t *st)
{
	uint64_t	t0;
	uint64_t	t1;
	uint32_t	i;

	memset(acc, 0, sizeof(*acc));
	*st = MEMBRANE_OK;
	t0 = now_ns();
	i = 0;
	while (i < cfg->blocks)
	{
		*st = membrane_bench_process_block(backend, cfg->policy,
				blocks + (size_t)i * MEMBRANE_BENCH_ELEMS_PER_BLOCK,
				MEMBRANE_BENCH_ELEMS_PER_BLOCK,
				MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, acc);
		if (*st != MEMBRANE_OK)
			return (now_ns() - t0);
		i++;
	}
	t1 = now_ns();
	return (t1 - t0);
}

static void	fill_result(const membrane_bench_config_t *cfg,
					membrane_simd_backend_t backend,
					const membrane_workload_accum_t *acc,
					uint64_t total_elems, double *samples, size_t n_samples,
					membrane_bench_result_t *out)
{
	memset(out, 0, sizeof(*out));
	out->config = *cfg;
	out->elems_per_block = MEMBRANE_BENCH_ELEMS_PER_BLOCK;
	out->simd_backend_name = membrane_simd_backend_name(backend);
	out->q4_blocks = acc->q4_blocks;
	out->q8_blocks = acc->q8_blocks;
	membrane_checked_mul_u64(total_elems, sizeof(float), &out->baseline_bytes);
	out->encoded_bytes = acc->encoded_bytes;
	out->saved_bytes = (int64_t)out->baseline_bytes
		- (int64_t)out->encoded_bytes;
	if (out->baseline_bytes > 0)
		out->reduction_ratio = (double)out->saved_bytes
			/ (double)out->baseline_bytes;
	out->blocks_decoded = acc->blocks_decoded;
	out->decode_failures = acc->decode_failures;
	out->encode_nondeterminism = acc->encode_nondeterminism;
	out->q4_mean_rel_l2_error = membrane_err_stats_mean(&acc->q4_err);
	out->q4_max_rel_l2_error = acc->q4_err.max;
	out->q8_mean_rel_l2_error = membrane_err_stats_mean(&acc->q8_err);
	out->q8_max_rel_l2_error = acc->q8_err.max;
	out->validation_pass = membrane_workload_validation_pass(acc,
			cfg->blocks);
	out->median_seconds = median_of(samples, n_samples);
	out->min_seconds = samples[0];
	out->max_seconds = samples[n_samples - 1];
	if (out->median_seconds > 0.0)
	{
		out->blocks_per_second = (double)cfg->blocks / out->median_seconds;
		out->elements_per_second = (double)total_elems / out->median_seconds;
	}
}

membrane_status_t	membrane_bench_run_one(const membrane_bench_config_t *cfg,
							membrane_bench_result_t *out)
{
	uint16_t					*blocks;
	double						*samples;
	membrane_workload_accum_t	acc;
	membrane_workload_accum_t	final_acc;
	membrane_simd_backend_t	backend;
	uint64_t					total_elems;
	uint64_t					blocks_bytes;
	uint64_t					ns;
	membrane_status_t			st;
	uint32_t					total_passes;
	uint32_t	i;

	if (cfg == NULL || out == NULL || !cfg_valid(cfg))
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!membrane_checked_mul_u64(cfg->blocks, MEMBRANE_BENCH_ELEMS_PER_BLOCK,
			&total_elems)
		|| !membrane_checked_mul_u64(total_elems, sizeof(uint16_t),
			&blocks_bytes))
		return (MEMBRANE_ERR_INVALID_ARG);
	blocks = malloc(blocks_bytes);
	total_passes = cfg->warmup + cfg->iterations;
	samples = malloc((size_t)cfg->iterations * sizeof(double));
	if (blocks == NULL || samples == NULL)
		return (free(blocks), free(samples), MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < cfg->blocks)
	{
		membrane_workload_generate_block(cfg->workload, cfg->seed, i,
			blocks + (size_t)i * MEMBRANE_BENCH_ELEMS_PER_BLOCK,
			MEMBRANE_BENCH_ELEMS_PER_BLOCK);
		i++;
	}
	backend = membrane_simd_best_backend();
	memset(&final_acc, 0, sizeof(final_acc));
	i = 0;
	while (i < total_passes)
	{
		ns = timed_pass(cfg, backend, blocks, &acc, &st);
		if (st != MEMBRANE_OK)
			return (free(blocks), free(samples), st);
		if (i == cfg->warmup)
			final_acc = acc;
		if (i >= cfg->warmup)
			samples[i - cfg->warmup] = (double)ns / 1e9;
		i++;
	}
	fill_result(cfg, backend, &final_acc, total_elems, samples,
		cfg->iterations, out);
	free(blocks);
	free(samples);
	return (MEMBRANE_OK);
}

double	membrane_bench_weighted_mean_rel_l2_error(
			const membrane_bench_result_t *r)
{
	uint64_t	total;

	total = r->q4_blocks + r->q8_blocks;
	if (total == 0)
		return (0.0);
	return ((r->q4_mean_rel_l2_error * (double)r->q4_blocks
			+ r->q8_mean_rel_l2_error * (double)r->q8_blocks)
		/ (double)total);
}

membrane_status_t	membrane_bench_run_matrix(
							const membrane_bench_config_t *base,
							membrane_bench_result_t *out, size_t out_cap,
							size_t *out_count)
{
	membrane_bench_config_t	cfg;
	membrane_status_t			st;
	membrane_workload_kind_t	w;
	membrane_bench_policy_t	p;
	size_t						n;

	if (base == NULL || out == NULL || out_count == NULL
			|| out_cap < MEMBRANE_BENCH_MATRIX_CELLS)
		return (MEMBRANE_ERR_INVALID_ARG);
	n = 0;
	w = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	while (w < MEMBRANE_WORKLOAD_COUNT)
	{
		p = MEMBRANE_BENCH_POLICY_Q4_ONLY;
		while (p < MEMBRANE_BENCH_POLICY_COUNT)
		{
			cfg = *base;
			cfg.workload = w;
			cfg.policy = p;
			st = membrane_bench_run_one(&cfg, &out[n]);
			if (st != MEMBRANE_OK)
				return (st);
			n++;
			p++;
		}
		w++;
	}
	*out_count = n;
	return (MEMBRANE_OK);
}
