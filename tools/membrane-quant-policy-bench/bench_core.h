#ifndef MEMBRANE_BENCH_CORE_H
# define MEMBRANE_BENCH_CORE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"
# include "workload_gen.h"
# include "precision_policy.h"
# include "trace_format.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMBRANE Product Phase 2/3 benchmark: measures real storage/accuracy/
 * runtime trade-offs across either the 4 synthetic workload kinds (see
 * workload_gen.h -- all synthetic, never real captured KV traces) or,
 * as of Phase 3, one real offline KV trace (see trace_format.h) loaded
 * via --trace, against the 3 precision policies (see precision_policy.h)
 * that tools/membrane-workload-core already implements. Offline: the
 * benchmark process never depends on llama.cpp or model weights,
 * whether it's running the synthetic generator or reading a trace file
 * someone else already captured and converted (see
 * tools/membrane-kv-trace-export).
 *
 * Timed region (see membrane_bench_run_one): exactly
 * membrane_bench_process_block() per block -- selection, encode,
 * decode, validate. Workload/trace generation-or-loading and result
 * formatting/printing are excluded from the timer -- a trace's blocks
 * are fully read into memory once, untimed, before the timed passes
 * begin (same pattern the synthetic generator already used). Multiple
 * timed iterations are run after a discarded warmup; the reported
 * primary statistic is the MEDIAN wall time (CLOCK_MONOTONIC) across
 * the timed iterations, with min/max also reported. This is local CPU
 * benchmark runtime only -- it is not an LLM inference speedup claim
 * and not a hardware performance claim, in either mode.
 *
 * Trace mode (cfg->trace_path != NULL): the synthetic generator is
 * never invoked; cfg->workload/cfg->seed are ignored (kept at whatever
 * default the caller left them at); cfg->blocks == 0 means "use every
 * block in the trace," a nonzero value caps it (never pads beyond what
 * the trace actually has). The result's workload_kind_label is
 * "trace", never "synthetic", and workload_name/trace_source_label are
 * derived only from the trace's own safe metadata or its filename --
 * membrane_bench_result_t never carries prompt/token content, because
 * the trace format itself (trace_format.h) cannot contain any.
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
	/* NULL (default) = synthetic mode. Non-NULL = trace mode: open and
	 * benchmark this .memkv file instead of generating data; see the
	 * "Trace mode" note above for exactly what changes. Points into
	 * caller-owned storage (e.g. an argv element) valid for the whole
	 * call -- not copied or freed by this module. */
	const char					*trace_path;
}	membrane_bench_config_t;

# define MEMBRANE_BENCH_WORKLOAD_NAME_CAP	64u

typedef struct s_membrane_bench_result
{
	membrane_bench_config_t	config;	/* .blocks is the ACTUAL count used */
	uint32_t					elems_per_block;
	const char					*simd_backend_name;

	/* "synthetic" or "trace" -- points to a static string constant,
	 * always non-NULL. */
	const char	*workload_kind_label;
	/* Always populated: the synthetic kind's name, or (trace mode) a
	 * filesystem-safe name derived from the trace file's own basename
	 * -- never a full/absolute path, never trace content. */
	char		workload_name[MEMBRANE_BENCH_WORKLOAD_NAME_CAP];
	/* Trace mode only (empty string otherwise): the trace's own safe
	 * metadata label if it set one, else the same basename as
	 * workload_name. */
	int			is_trace;
	uint32_t	trace_format_version;
	char		trace_source_label[MEMBRANE_TRACE_METADATA_CAP];

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
 * --warmup N, --trace PATH, --json, --csv, --matrix, --help. Rejects
 * unknown options, non-numeric/negative/zero --blocks (unless --trace
 * is also given, where 0 is the valid default meaning "uncapped"),
 * zero --iterations, --blocks/--iterations/--warmup above their MAX,
 * an unrecognized --workload/--policy name, and --workload or --seed
 * combined with --trace (trace mode never uses the synthetic
 * generator, so those flags would silently do nothing -- rejected
 * instead of ignored). On success returns 0 and fills *args (defaults
 * applied for any flag not given; --workload/--policy are ignored
 * under --matrix, still parsed/validated if present). On failure
 * returns MEMBRANE_BENCH_EXIT_USAGE_ERROR and writes a one-line,
 * NUL-terminated reason to err_buf (err_cap bytes).
 */
int	membrane_bench_parse_args(int argc, char **argv,
		membrane_bench_args_t *args, char *err_buf, size_t err_cap);

/*
 * Runs `cfg` end to end -- warmup + timed iterations over a
 * once-acquired (generated or, in trace mode, fully read from disk),
 * deterministic block set -- and fills *out. Storage/accuracy/
 * precision-count fields are captured from one representative timed
 * iteration (all timed iterations are numerically identical by
 * construction: membrane_bench_process_block is a pure function of its
 * inputs). Returns MEMBRANE_ERR_INVALID_ARG for a NULL cfg/out or any
 * out-of-range field in *cfg; MEMBRANE_ERR_ALLOC_FAILED if a working
 * buffer cannot be allocated; MEMBRANE_ERR_UNIMPLEMENTED if a trace's
 * elements_per_block is not a multiple of 32 (quant_simd's fixed
 * requirement); otherwise propagates a trace_format.h read failure or
 * an infrastructure failure from membrane_bench_process_block (see
 * precision_policy.h).
 */
membrane_status_t	membrane_bench_run_one(const membrane_bench_config_t *cfg,
							membrane_bench_result_t *out);

/*
 * base->trace_path == NULL: the full 4-workload x 3-policy synthetic
 * matrix (MEMBRANE_WORKLOAD_COUNT * MEMBRANE_BENCH_POLICY_COUNT
 * results; base->workload/policy are ignored). base->trace_path !=
 * NULL: exactly the 3 policies against that one trace (base->workload
 * is ignored; NOT the 4-workload synthetic sweep). `out` must have
 * space for MEMBRANE_BENCH_MATRIX_CELLS results either way;
 * *out_count is set to the number actually filled (12 or 3).
 * blocks/seed/iterations/warmup/trace_path come from `base`.
 */
membrane_status_t	membrane_bench_run_matrix(
							const membrane_bench_config_t *base,
							membrane_bench_result_t *out, size_t out_cap,
							size_t *out_count);

# define MEMBRANE_BENCH_MATRIX_CELLS \
	(MEMBRANE_WORKLOAD_COUNT * MEMBRANE_BENCH_POLICY_COUNT)

/*
 * Block-count-weighted mean rel-L2 error across whichever mix of Q4/Q8
 * blocks this result actually used. r->q4_mean_rel_l2_error and
 * r->q8_mean_rel_l2_error are means of two different populations, so
 * simply adding them is not a mean of anything and can exceed 1.0 --
 * this is the correct combined figure the compact matrix table prints.
 * Returns 0.0 if the result has no blocks of either precision.
 */
double	membrane_bench_weighted_mean_rel_l2_error(
			const membrane_bench_result_t *r);

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
