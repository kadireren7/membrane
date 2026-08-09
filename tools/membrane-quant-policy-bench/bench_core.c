#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
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

/* Explicit unsigned comparisons rather than relying on the enum's
 * natural comparison: C leaves the underlying integer type of an enum
 * with only non-negative enumerators implementation-defined, so an
 * out-of-range value written via a cast like
 * (membrane_workload_kind_t)-1 is only guaranteed to be rejected by
 * `< COUNT` if every supported compiler picks an unsigned
 * representation. Casting both sides to `unsigned int` makes the
 * comparison correct regardless of that choice, without relying on it. */
static int	cfg_valid(const membrane_bench_config_t *cfg)
{
	if (cfg->iterations == 0 || cfg->iterations > MEMBRANE_BENCH_MAX_ITERATIONS
			|| cfg->warmup > MEMBRANE_BENCH_MAX_WARMUP
			|| cfg->blocks > MEMBRANE_BENCH_MAX_BLOCKS
			|| (unsigned int)cfg->policy
				>= (unsigned int)MEMBRANE_BENCH_POLICY_COUNT)
		return (0);
	if (cfg->trace_path != NULL)
		return (1);
	return (cfg->blocks != 0
		&& (unsigned int)cfg->workload < (unsigned int)MEMBRANE_WORKLOAD_COUNT);
}

/* One full pass over every pre-acquired block, timed. Fills *acc (any
 * prior contents are overwritten via memset). */
static uint64_t	timed_pass(const membrane_bench_config_t *cfg,
					membrane_simd_backend_t backend, const uint16_t *blocks,
					uint64_t block_count, uint32_t elems_per_block,
					membrane_workload_accum_t *acc, membrane_status_t *st)
{
	uint64_t	t0;
	uint64_t	t1;
	uint64_t	i;

	memset(acc, 0, sizeof(*acc));
	*st = MEMBRANE_OK;
	t0 = now_ns();
	i = 0;
	while (i < block_count)
	{
		*st = membrane_bench_process_block(backend, cfg->policy,
				blocks + i * elems_per_block, elems_per_block,
				MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, acc);
		if (*st != MEMBRANE_OK)
			return (now_ns() - t0);
		i++;
	}
	t1 = now_ns();
	return (t1 - t0);
}

static membrane_status_t	generate_synthetic_blocks(
								const membrane_bench_config_t *cfg,
								uint16_t **out_blocks,
								uint64_t *out_block_count,
								uint32_t *out_elems_per_block)
{
	uint16_t	*blocks;
	uint64_t	total_elems;
	uint64_t	blocks_bytes;
	uint32_t	i;

	if (!membrane_checked_mul_u64(cfg->blocks, MEMBRANE_BENCH_ELEMS_PER_BLOCK,
			&total_elems)
		|| !membrane_checked_mul_u64(total_elems, sizeof(uint16_t),
			&blocks_bytes))
		return (MEMBRANE_ERR_INVALID_ARG);
	blocks = malloc((size_t)blocks_bytes);
	if (blocks == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < cfg->blocks)
	{
		membrane_workload_generate_block(cfg->workload, cfg->seed, i,
			blocks + (size_t)i * MEMBRANE_BENCH_ELEMS_PER_BLOCK,
			MEMBRANE_BENCH_ELEMS_PER_BLOCK);
		i++;
	}
	*out_blocks = blocks;
	*out_block_count = cfg->blocks;
	*out_elems_per_block = MEMBRANE_BENCH_ELEMS_PER_BLOCK;
	return (MEMBRANE_OK);
}

/*
 * Fully reads a trace (validating its payload checksum end to end, even
 * when cfg->blocks caps how many of those blocks are actually kept) and
 * returns at most cfg->blocks of them (all of them if cfg->blocks == 0).
 */
static membrane_status_t	load_trace_blocks(
								const membrane_bench_config_t *cfg,
								uint16_t **out_blocks,
								uint64_t *out_block_count,
								uint32_t *out_elems_per_block,
								membrane_trace_info_t *out_info)
{
	membrane_trace_reader_t	*r;
	uint16_t					*blocks;
	uint16_t					*scratch;
	uint64_t					use_count;
	uint64_t					total_elems;
	uint64_t					blocks_bytes;
	uint64_t					i;
	membrane_status_t			st;

	st = membrane_trace_open(cfg->trace_path, &r);
	if (st != MEMBRANE_OK)
		return (st);
	membrane_trace_info(r, out_info);
	if (out_info->elements_per_block % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (membrane_trace_close(r), MEMBRANE_ERR_UNIMPLEMENTED);
	use_count = out_info->block_count;
	if (cfg->blocks != 0 && (uint64_t)cfg->blocks < use_count)
		use_count = cfg->blocks;
	if (!membrane_checked_mul_u64(use_count, out_info->elements_per_block,
			&total_elems)
		|| !membrane_checked_mul_u64(total_elems, sizeof(uint16_t),
			&blocks_bytes))
		return (membrane_trace_close(r), MEMBRANE_ERR_INVALID_ARG);
	blocks = malloc((size_t)blocks_bytes);
	scratch = malloc((size_t)out_info->elements_per_block * sizeof(uint16_t));
	if (blocks == NULL || scratch == NULL)
	{
		free(blocks);
		free(scratch);
		return (membrane_trace_close(r), MEMBRANE_ERR_ALLOC_FAILED);
	}
	i = 0;
	st = MEMBRANE_OK;
	while (i < out_info->block_count)
	{
		if (i < use_count)
			st = membrane_trace_read_block(r,
					blocks + i * out_info->elements_per_block);
		else
			st = membrane_trace_read_block(r, scratch);
		if (st != MEMBRANE_OK)
			break ;
		i++;
	}
	free(scratch);
	membrane_trace_close(r);
	if (st != MEMBRANE_OK)
		return (free(blocks), st);
	*out_blocks = blocks;
	*out_block_count = use_count;
	*out_elems_per_block = out_info->elements_per_block;
	return (MEMBRANE_OK);
}

static const char	*safe_basename(const char *path)
{
	const char	*slash;

	slash = strrchr(path, '/');
	if (slash != NULL)
		return (slash + 1);
	return (path);
}

/* Copies `src` into `out` and drops a trailing "-style extension (the
 * last '.' onward), if any, for a shorter display name -- never
 * touches `src` itself. */
static void	copy_display_name(char *out, size_t out_cap, const char *src)
{
	char	*dot;

	snprintf(out, out_cap, "%s", src);
	dot = strrchr(out, '.');
	if (dot != NULL && dot != out)
		*dot = '\0';
}

static void	fill_result(const membrane_bench_config_t *cfg,
					membrane_simd_backend_t backend,
					const membrane_workload_accum_t *acc,
					uint64_t block_count, uint32_t elems_per_block,
					uint64_t total_elems, double *samples, size_t n_samples,
					const membrane_trace_info_t *trace_info,
					membrane_bench_result_t *out)
{
	memset(out, 0, sizeof(*out));
	out->config = *cfg;
	/* Reflect the actual count processed -- for trace mode with
	 * cfg->blocks == 0 ("uncapped"), that's not the same as cfg->blocks. */
	out->config.blocks = (uint32_t)block_count;
	out->elems_per_block = elems_per_block;
	out->simd_backend_name = membrane_simd_backend_name(backend);
	out->is_trace = (cfg->trace_path != NULL);
	if (out->is_trace)
	{
		out->workload_kind_label = "trace";
		out->trace_format_version = trace_info->format_version;
		copy_display_name(out->workload_name, sizeof(out->workload_name),
			safe_basename(cfg->trace_path));
		if (trace_info->metadata[0] != '\0')
			snprintf(out->trace_source_label, sizeof(out->trace_source_label),
				"%s", trace_info->metadata);
		else
			snprintf(out->trace_source_label, sizeof(out->trace_source_label),
				"%s", safe_basename(cfg->trace_path));
	}
	else
	{
		out->workload_kind_label = MEMBRANE_WORKLOAD_KIND_LABEL;
		snprintf(out->workload_name, sizeof(out->workload_name), "%s",
			membrane_workload_kind_name(cfg->workload));
	}
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
	out->validation_pass = membrane_workload_validation_pass(acc, block_count);
	out->median_seconds = median_of(samples, n_samples);
	out->min_seconds = samples[0];
	out->max_seconds = samples[n_samples - 1];
	if (out->median_seconds > 0.0)
	{
		out->blocks_per_second = (double)block_count / out->median_seconds;
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
	membrane_trace_info_t		trace_info;
	uint64_t					block_count;
	uint32_t					elems_per_block;
	uint64_t					total_elems;
	uint64_t					ns;
	membrane_status_t			st;
	uint32_t					total_passes;
	uint32_t					i;

	if (cfg == NULL || out == NULL || !cfg_valid(cfg))
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(&trace_info, 0, sizeof(trace_info));
	if (cfg->trace_path != NULL)
		st = load_trace_blocks(cfg, &blocks, &block_count, &elems_per_block,
				&trace_info);
	else
		st = generate_synthetic_blocks(cfg, &blocks, &block_count,
				&elems_per_block);
	if (st != MEMBRANE_OK)
		return (st);
	/* Cannot overflow: block_count*elems_per_block*sizeof(uint16_t)
	 * was already validated not to overflow while acquiring the
	 * blocks above, and this product is smaller than that one. */
	total_elems = block_count * elems_per_block;
	total_passes = cfg->warmup + cfg->iterations;
	samples = malloc((size_t)cfg->iterations * sizeof(double));
	if (samples == NULL)
		return (free(blocks), MEMBRANE_ERR_ALLOC_FAILED);
	backend = membrane_simd_best_backend();
	memset(&final_acc, 0, sizeof(final_acc));
	i = 0;
	while (i < total_passes)
	{
		ns = timed_pass(cfg, backend, blocks, block_count, elems_per_block,
				&acc, &st);
		if (st != MEMBRANE_OK)
			return (free(blocks), free(samples), st);
		if (i == cfg->warmup)
			final_acc = acc;
		if (i >= cfg->warmup)
			samples[i - cfg->warmup] = (double)ns / 1e9;
		i++;
	}
	fill_result(cfg, backend, &final_acc, block_count, elems_per_block,
		total_elems, samples, cfg->iterations, &trace_info, out);
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

static membrane_status_t	run_trace_matrix(
								const membrane_bench_config_t *base,
								membrane_bench_result_t *out,
								size_t *out_count)
{
	membrane_bench_config_t	cfg;
	membrane_status_t			st;
	membrane_bench_policy_t	p;
	size_t						n;

	n = 0;
	p = MEMBRANE_BENCH_POLICY_Q4_ONLY;
	while (p < MEMBRANE_BENCH_POLICY_COUNT)
	{
		cfg = *base;
		cfg.policy = p;
		st = membrane_bench_run_one(&cfg, &out[n]);
		if (st != MEMBRANE_OK)
			return (st);
		n++;
		p++;
	}
	*out_count = n;
	return (MEMBRANE_OK);
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
	if (base->trace_path != NULL)
		return (run_trace_matrix(base, out, out_count));
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
