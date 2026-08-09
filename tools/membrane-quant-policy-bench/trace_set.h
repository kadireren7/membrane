#ifndef MEMBRANE_TRACE_SET_H
# define MEMBRANE_TRACE_SET_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "bench_core.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMBRANE Product Phase 4: benchmarks a whole DIRECTORY of .memkv
 * traces (e.g. every layer/tensor exported in one
 * membrane-kv-trace-export --output-dir batch) against the 3 precision
 * policies, one real capture's cross-layer/cross-tensor variation in a
 * single invocation. Still fully offline -- no llama.cpp dependency,
 * no model weights, only membrane_bench_run_matrix() (bench_core.h)
 * called once per discovered file, unchanged. Answers "does Q4/Q8
 * suitability vary across layers/tensors on one real model execution,"
 * never "how does this generalize across models/prompts."
 *
 * Discovery is a single, non-recursive readdir() of the given
 * directory: only regular files (symlinks rejected outright -- this is
 * a security boundary, not a convenience skip) whose name ends in
 * ".memkv" are candidates; everything else is silently ignored (a
 * README, a .gitkeep, a subdirectory). Candidates are sorted by
 * filename (deterministic order) before anything is opened.
 *
 * A candidate whose name matches "layer-NNN-k.memkv" / "layer-NNN-
 * v.memkv" (membrane-kv-trace-export's own batch naming) is tagged
 * with a known layer/tensor for reporting; any other name is still
 * benchmarked, just reported with layer/tensor "unknown" (JSON null).
 * Two candidates that both resolve to the same known (layer, tensor)
 * are a conflicting/duplicate set -- rejected with a clear error
 * before any benchmark runs, not silently overwritten or dropped.
 *
 * Every candidate's trace header is opened and validated (see
 * trace_format.h's membrane_trace_open) in a pre-flight pass, and the
 * pre-flight sums every trace's block_count against
 * MEMBRANE_TRACE_SET_MAX_TOTAL_BLOCKS, before any benchmark actually
 * runs -- one corrupt/oversized trace in a large set fails the whole
 * run clearly, rather than silently vanishing from the results or
 * exhausting memory partway through.
 */

# define MEMBRANE_TRACE_SET_MAX_FILES		512u
# define MEMBRANE_TRACE_SET_PATH_CAP		4096u
# define MEMBRANE_TRACE_SET_MAX_TOTAL_BLOCKS	8000000ull
# define MEMBRANE_TRACE_SET_SCHEMA_VERSION	1

typedef struct s_membrane_trace_set_file
{
	char		path[MEMBRANE_TRACE_SET_PATH_CAP];	/* for opening only --
													 * never copied into any
													 * printed output */
	int			has_layer;
	uint32_t	layer;
	int			has_tensor;
	char		tensor;		/* 'K' or 'V', valid only if has_tensor */
}	membrane_trace_set_file_t;

/*
 * Discovers and validates every .memkv candidate directly inside `dir`
 * (not recursive). On success, *out is a newly malloc'd array of
 * *out_count entries (caller frees it) in deterministic filename
 * order; *out_total_blocks is the sum of every candidate's
 * trace_format.h block_count (each trace opened+closed once here,
 * purely to validate and read that count -- no block payload is read
 * yet). Returns MEMBRANE_ERR_IO if `dir` cannot be opened,
 * MEMBRANE_ERR_INVALID_ARG for a symlink entry, more than
 * MEMBRANE_TRACE_SET_MAX_FILES candidates, a duplicate/conflicting
 * (layer, tensor) pair, or a total block count exceeding
 * MEMBRANE_TRACE_SET_MAX_TOTAL_BLOCKS, MEMBRANE_ERR_CORRUPT_DATA if any
 * candidate fails membrane_trace_open. A one-line reason is written to
 * err_buf (err_cap bytes) on any non-OK return.
 */
membrane_status_t	membrane_trace_set_discover(const char *dir,
						membrane_trace_set_file_t **out, size_t *out_count,
						uint64_t *out_total_blocks, char *err_buf,
						size_t err_cap);

typedef struct s_membrane_trace_set_item
{
	membrane_bench_result_t	result;
	int							has_layer;
	uint32_t					layer;
	int							has_tensor;
	char						tensor;
}	membrane_trace_set_item_t;

/*
 * Block-count-weighted / pooled-sum aggregate across every ADAPTIVE-
 * policy item in a trace set (see bench_core.h's
 * membrane_bench_weighted_mean_rel_l2_error for why naive per-trace
 * averaging would be wrong at this scale too): storage bytes are
 * summed then divided once; q4/q8 mean errors are each the exact
 * pooled mean over every underlying block across every trace (sum =
 * per-trace mean * per-trace count, then divided by the pooled count
 * -- exact, not an approximation); max errors are the max of per-trace
 * maxes; blocks_per_second is total processed blocks divided by total
 * MEASURED processing time (sum of every trace's own median_seconds),
 * never a sum of per-trace throughputs (bench_core.h's timing
 * contract: trace loading/I/O stays outside every timed region, so
 * this total is comparable across traces).
 */
typedef struct s_membrane_trace_set_aggregate
{
	uint32_t	traces;
	uint64_t	total_blocks;
	uint64_t	q4_blocks;
	uint64_t	q8_blocks;
	double		q4_ratio;
	uint64_t	baseline_bytes;
	uint64_t	encoded_bytes;
	double		storage_reduction_ratio;
	double		q4_mean_rel_l2_error;
	double		q4_max_rel_l2_error;
	double		q8_mean_rel_l2_error;
	double		q8_max_rel_l2_error;
	double		weighted_mean_rel_l2_error;
	uint64_t	decode_failures;
	double		total_measured_seconds;
	double		blocks_per_second;

	/* Neutral facts (section 9): no "important/easy/hard" labeling. */
	double		min_q4_ratio;
	char		min_q4_ratio_trace[MEMBRANE_BENCH_WORKLOAD_NAME_CAP];
	double		max_q4_ratio;
	char		max_q4_ratio_trace[MEMBRANE_BENCH_WORKLOAD_NAME_CAP];
	double		median_q4_ratio;
	uint64_t	traces_any_q4;
	uint64_t	traces_all_q8;
}	membrane_trace_set_aggregate_t;

/*
 * Runs membrane_bench_run_matrix() (exactly 3 policy results: q4-only,
 * q8-only, adaptive) against every file in `files` (n_files entries,
 * as produced by membrane_trace_set_discover), using
 * base->blocks/iterations/warmup as the per-trace cap/timing
 * parameters (base->trace_path, workload, and seed are ignored). On
 * success *out is a newly malloc'd array of n_files*3 items (caller
 * frees it) and *out_count == n_files*3; *out_agg is filled from
 * exactly the MEMBRANE_BENCH_POLICY_ADAPTIVE subset. Returns whatever
 * status the first failing membrane_bench_run_matrix() call returned.
 */
membrane_status_t	membrane_trace_set_run(
						const membrane_bench_config_t *base,
						const membrane_trace_set_file_t *files,
						size_t n_files, membrane_trace_set_item_t **out,
						size_t *out_count,
						membrane_trace_set_aggregate_t *out_agg);

void	membrane_trace_set_print_human(const membrane_trace_set_item_t *items,
			size_t n, const membrane_trace_set_aggregate_t *agg, FILE *f);
void	membrane_trace_set_print_json(const membrane_trace_set_item_t *items,
			size_t n, const membrane_trace_set_aggregate_t *agg, FILE *f);
void	membrane_trace_set_print_csv(const membrane_trace_set_item_t *items,
			size_t n, FILE *f);

# ifdef __cplusplus
}
# endif

#endif
