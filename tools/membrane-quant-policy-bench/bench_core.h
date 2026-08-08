#ifndef MEMBRANE_BENCH_CORE_H
# define MEMBRANE_BENCH_CORE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"
# include "workload_gen.h"
# include "precision_policy.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMBRANE Product Phase 2 benchmark: measures real storage/accuracy/
 * runtime trade-offs across the 4 synthetic workload kinds (see
 * workload_gen.h -- all synthetic, never real captured KV traces) and
 * the 3 precision policies (see precision_policy.h) that
 * tools/membrane-workload-core already implements, offline and
 * deterministically (given the same workload/policy/blocks/seed).
 *
 * Timed region (see membrane_bench_run_one): exactly
 * membrane_bench_process_block() per block -- selection, encode,
 * decode, validate. Workload generation and result formatting/printing
 * are excluded. Multiple timed iterations are run after a discarded
 * warmup; the reported primary statistic is the MEDIAN wall time
 * (CLOCK_MONOTONIC) across the timed iterations, with min/max also
 * reported. This is local CPU benchmark runtime only -- it is not an
 * LLM inference speedup claim and not a hardware performance claim.
 */

# define MEMBRANE_BENCH_ELEMS_PER_BLOCK		128u	/* multiple of 32 */
# define MEMBRANE_BENCH_DEFAULT_BLOCKS		2048u
# define MEMBRANE_BENCH_DEFAULT_SEED		1234u
# define MEMBRANE_BENCH_DEFAULT_ITERATIONS	5u
# define MEMBRANE_BENCH_DEFAULT_WARMUP		1u
# define MEMBRANE_BENCH_MAX_BLOCKS			1000000u
# define MEMBRANE_BENCH_MAX_ITERATIONS		100000u
# define MEMBRANE_BENCH_MAX_WARMUP			100000u
# define MEMBRANE_BENCH_SCHEMA_VERSION		1

typedef struct s_membrane_bench_config
{
	membrane_workload_kind_t	workload;
	membrane_bench_policy_t	policy;
	uint32_t					blocks;
	uint32_t					seed;
	uint32_t					iterations;
	uint32_t					warmup;
}	membrane_bench_config_t;

typedef struct s_membrane_bench_result
{
	membrane_bench_config_t	config;
	uint32_t					elems_per_block;
	const char					*simd_backend_name;

	uint64_t	q4_blocks;
	uint64_t	q8_blocks;
	uint64_t	baseline_bytes;
	uint64_t	encoded_bytes;
	int64_t		saved_bytes;
	double		reduction_ratio;

	uint64_t	blocks_decoded;
	uint64_t	decode_failures;
	uint64_t	encode_nondeterminism;
	double		q4_mean_rel_l2_error;
	double		q4_max_rel_l2_error;
	double		q8_mean_rel_l2_error;
	double		q8_max_rel_l2_error;
	int			validation_pass;

	double		median_seconds;
	double		min_seconds;
	double		max_seconds;
	double		blocks_per_second;
	double		elements_per_second;
}	membrane_bench_result_t;

/* Exit code the CLI should use for a parse/usage failure. */
# define MEMBRANE_BENCH_EXIT_USAGE_ERROR	2

typedef struct s_membrane_bench_args
{
	membrane_bench_config_t	cfg;
	int							want_json;
	int							want_csv;
	int							want_matrix;
	int							want_help;
}	membrane_bench_args_t;

/*
 * Parses argv[1..argc). Recognizes --workload NAME, --policy
 * q4-only|q8-only|adaptive, --blocks N, --seed N, --iterations N,
 * --warmup N, --json, --csv, --matrix, --help. Rejects unknown
 * options, non-numeric/negative/zero --blocks, zero --iterations,
 * --blocks/--iterations/--warmup above their MAX, and an unrecognized
 * --workload/--policy name. On success returns 0 and fills *args
 * (defaults applied for any flag not given; --workload/--policy are
 * ignored under --matrix, still parsed/validated if present). On
 * failure returns MEMBRANE_BENCH_EXIT_USAGE_ERROR and writes a
 * one-line, NUL-terminated reason to err_buf (err_cap bytes).
 */
int	membrane_bench_parse_args(int argc, char **argv,
		membrane_bench_args_t *args, char *err_buf, size_t err_cap);

/*
 * Runs `cfg` (workload x policy) end to end -- warmup + timed
 * iterations over a once-generated, deterministic block set -- and
 * fills *out. Storage/accuracy/precision-count fields are captured
 * from one representative timed iteration (all timed iterations are
 * numerically identical by construction: membrane_bench_process_block
 * is a pure function of its inputs). Returns MEMBRANE_ERR_INVALID_ARG
 * for a NULL cfg/out or any out-of-range field in *cfg;
 * MEMBRANE_ERR_ALLOC_FAILED if a working buffer cannot be allocated;
 * otherwise propagates an infrastructure failure from
 * membrane_bench_process_block (see precision_policy.h).
 */
membrane_status_t	membrane_bench_run_one(const membrane_bench_config_t *cfg,
							membrane_bench_result_t *out);

/* The full workload x policy matrix. `out` must have space for
 * MEMBRANE_WORKLOAD_COUNT * MEMBRANE_BENCH_POLICY_COUNT results;
 * *out_count is set to that number on success. blocks/seed/iterations/
 * warmup come from `base` (its workload/policy fields are ignored). */
membrane_status_t	membrane_bench_run_matrix(
							const membrane_bench_config_t *base,
							membrane_bench_result_t *out, size_t out_cap,
							size_t *out_count);

# define MEMBRANE_BENCH_MATRIX_CELLS \
	(MEMBRANE_WORKLOAD_COUNT * MEMBRANE_BENCH_POLICY_COUNT)

void	membrane_bench_print_human(const membrane_bench_result_t *r, FILE *f);
void	membrane_bench_print_json(const membrane_bench_result_t *r, FILE *f);
void	membrane_bench_print_csv_header(FILE *f);
void	membrane_bench_print_csv_row(const membrane_bench_result_t *r, FILE *f);

void	membrane_bench_print_matrix_human(const membrane_bench_result_t *rs,
			size_t n, FILE *f);
void	membrane_bench_print_matrix_json(const membrane_bench_result_t *rs,
			size_t n, FILE *f);
void	membrane_bench_print_matrix_csv(const membrane_bench_result_t *rs,
			size_t n, FILE *f);

# ifdef __cplusplus
}
# endif

#endif
