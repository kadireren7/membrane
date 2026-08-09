#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bench_core.h"
#include "test_helpers.h"

static membrane_bench_config_t	default_cfg(void)
{
	membrane_bench_config_t	cfg;

	cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	cfg.blocks = 256;
	cfg.seed = 1234;
	cfg.iterations = 2;
	cfg.warmup = 1;
	cfg.trace_path = NULL;
	return (cfg);
}

/* ------------------------------------------------------------------ */
/* Trace-mode fixture (Product Phase 3)                                */
/* ------------------------------------------------------------------ */

static char	g_trace_path[] = "/tmp/membrane-bench-trace-XXXXXX";

/* A deterministic, non-synthetic-generator pattern: if trace mode ever
 * accidentally fell back to the synthetic generator, the resulting
 * blocks (and therefore the benchmark's precision/accuracy numbers)
 * would not match what this exact linear pattern is known to produce
 * -- see test_trace_ignores_workload_and_seed_fields, which relies on
 * this being independent of any workload_gen.h kind/seed. */
static void	write_trace_fixture(uint64_t block_count, uint32_t epb)
{
	uint16_t	*blocks;
	FILE		*f;
	uint64_t	i;

	blocks = malloc((size_t)(block_count * epb) * sizeof(uint16_t));
	i = 0;
	while (i < block_count * epb)
	{
		blocks[i] = (uint16_t)(i * 13u + 5u);
		i++;
	}
	f = fopen(g_trace_path, "wb");
	TEST_ASSERT(f != NULL, "open trace fixture for write");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, epb,
			block_count, blocks, "test fixture, not a real capture", 0)
			== MEMBRANE_OK, "write trace fixture");
	fclose(f);
	free(blocks);
}

static membrane_bench_config_t	trace_cfg(void)
{
	membrane_bench_config_t	cfg;

	cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	cfg.blocks = 0;
	cfg.seed = 0;
	cfg.iterations = 2;
	cfg.warmup = 1;
	cfg.trace_path = g_trace_path;
	return (cfg);
}

static void	test_each_policy_runs_and_validates(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	membrane_bench_policy_t	p;

	p = MEMBRANE_BENCH_POLICY_Q4_ONLY;
	while (p < MEMBRANE_BENCH_POLICY_COUNT)
	{
		cfg = default_cfg();
		cfg.policy = p;
		TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK,
			"run succeeds");
		TEST_ASSERT(r.validation_pass, "this policy's run validates");
		TEST_ASSERT(r.blocks_decoded == cfg.blocks, "every block decoded");
		p++;
	}
}

static void	test_q4_only_forces_precision(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_Q4_ONLY;
	cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK,
		"run succeeds");
	TEST_ASSERT(r.q4_blocks == cfg.blocks && r.q8_blocks == 0,
		"q4-only assigns every block to Q4 regardless of workload");
	TEST_ASSERT(r.validation_pass,
		"still validates even though Q4 error exceeds the adaptive bound "
		"(execution integrity, not adaptive policy quality)");
}

static void	test_q8_only_forces_precision(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_Q8_ONLY;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK,
		"run succeeds");
	TEST_ASSERT(r.q4_blocks == 0 && r.q8_blocks == cfg.blocks,
		"q8-only assigns every block to Q8");
}

static void	test_all_workloads_run_and_validate(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	membrane_workload_kind_t	w;

	w = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	while (w < MEMBRANE_WORKLOAD_COUNT)
	{
		cfg = default_cfg();
		cfg.workload = w;
		TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK,
			"run succeeds");
		TEST_ASSERT(r.validation_pass, "this workload's run validates");
		w++;
	}
}

static void	test_same_config_deterministic_non_timing(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r1;
	membrane_bench_result_t	r2;

	cfg = default_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r1) == MEMBRANE_OK, "run 1");
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r2) == MEMBRANE_OK, "run 2");
	TEST_ASSERT(r1.q4_blocks == r2.q4_blocks && r1.q8_blocks == r2.q8_blocks,
		"precision counts reproducible");
	TEST_ASSERT(r1.baseline_bytes == r2.baseline_bytes
		&& r1.encoded_bytes == r2.encoded_bytes,
		"storage numbers reproducible");
	TEST_ASSERT(r1.q4_mean_rel_l2_error == r2.q4_mean_rel_l2_error
		&& r1.q8_mean_rel_l2_error == r2.q8_mean_rel_l2_error,
		"accuracy numbers bit-identical across runs");
	/* Timing is explicitly NOT required to be deterministic. */
}

static void	test_different_seed_still_validates(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	uint32_t					seeds[3];
	size_t						i;

	seeds[0] = 0;
	seeds[1] = 5;
	seeds[2] = 999999;
	i = 0;
	while (i < 3)
	{
		cfg = default_cfg();
		cfg.seed = seeds[i];
		TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK,
			"run succeeds for this seed");
		TEST_ASSERT(r.validation_pass, "this seed's run validates");
		i++;
	}
}

static void	test_matrix_cell_count(void)
{
	membrane_bench_config_t	base;
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n;

	base = default_cfg();
	TEST_ASSERT(membrane_bench_run_matrix(&base, rs,
			MEMBRANE_BENCH_MATRIX_CELLS, &n) == MEMBRANE_OK,
		"matrix run succeeds");
	TEST_ASSERT(n == MEMBRANE_WORKLOAD_COUNT * MEMBRANE_BENCH_POLICY_COUNT,
		"matrix produces exactly workloads * policies results");
	TEST_ASSERT(n == 12, "currently 4 workloads x 3 policies = 12");
}

static void	test_timing_metrics_are_sane(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(isfinite(r.median_seconds) && r.median_seconds >= 0.0,
		"median_seconds finite and non-negative");
	TEST_ASSERT(isfinite(r.min_seconds) && r.min_seconds >= 0.0,
		"min_seconds finite and non-negative");
	TEST_ASSERT(isfinite(r.max_seconds) && r.max_seconds >= 0.0,
		"max_seconds finite and non-negative");
	TEST_ASSERT(r.min_seconds <= r.median_seconds
		&& r.median_seconds <= r.max_seconds, "min <= median <= max");
	TEST_ASSERT(isfinite(r.blocks_per_second) && r.blocks_per_second >= 0.0,
		"blocks_per_second finite and non-negative");
	TEST_ASSERT(isfinite(r.elements_per_second)
		&& r.elements_per_second >= 0.0,
		"elements_per_second finite and non-negative");
}

static void	test_zero_blocks_rejected(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.blocks = 0;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"zero blocks rejected");
}

static void	test_zero_iterations_rejected(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.iterations = 0;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"zero iterations rejected");
}

static void	test_oversized_blocks_rejected(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.blocks = MEMBRANE_BENCH_MAX_BLOCKS + 1;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"over-the-limit block count rejected, not attempted");
}

static void	test_invalid_enum_values_rejected(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	cfg = default_cfg();
	cfg.workload = (membrane_workload_kind_t)-1;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"an out-of-range workload enum is rejected, not silently defaulted");
	cfg = default_cfg();
	cfg.policy = (membrane_bench_policy_t)-1;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"an out-of-range policy enum is rejected, not silently defaulted");
}

static void	test_weighted_mean_is_between_component_means(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	double						w;

	cfg = default_cfg();
	cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_MIXED;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.q4_blocks > 0 && r.q8_blocks > 0,
		"this workload/policy actually mixes both precisions "
		"(otherwise the weighting has nothing to test)");
	w = membrane_bench_weighted_mean_rel_l2_error(&r);
	TEST_ASSERT(w <= 1.0, "a weighted mean of two <=1.0 means cannot "
		"exceed 1.0 (the un-weighted sum used before this fix could)");
	TEST_ASSERT(w >= (r.q4_mean_rel_l2_error < r.q8_mean_rel_l2_error
			? r.q4_mean_rel_l2_error : r.q8_mean_rel_l2_error)
		&& w <= (r.q4_mean_rel_l2_error > r.q8_mean_rel_l2_error
			? r.q4_mean_rel_l2_error : r.q8_mean_rel_l2_error),
		"a weighted mean of two values lies between them");
}

static void	test_arg_parsing(void)
{
	membrane_bench_args_t	args;
	char					err[160];
	char					*argv_ok[] = {"membrane-quant-policy-bench",
								"--workload", "synthetic-mixed", "--policy",
								"q8-only", "--blocks", "64", "--json"};
	char					*argv_bad_policy[] = {"x", "--policy", "bogus"};
	char					*argv_bad_workload[] = {"x", "--workload",
								"bogus"};
	char					*argv_zero_blocks[] = {"x", "--blocks", "0"};
	char					*argv_zero_iter[] = {"x", "--iterations", "0"};
	char					*argv_huge_blocks[] = {"x", "--blocks",
								"999999999999"};
	char					*argv_unknown[] = {"x", "--nope"};
	char					*argv_matrix[] = {"x", "--matrix", "--csv"};

	TEST_ASSERT(membrane_bench_parse_args(8, argv_ok, &args, err,
			sizeof(err)) == 0, "valid args parse");
	TEST_ASSERT(args.cfg.workload == MEMBRANE_WORKLOAD_SYNTHETIC_MIXED
		&& args.cfg.policy == MEMBRANE_BENCH_POLICY_Q8_ONLY
		&& args.cfg.blocks == 64 && args.want_json == 1,
		"parsed values match the given flags");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_bad_policy, &args, err,
			sizeof(err)) != 0, "unknown policy rejected");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_bad_workload, &args, err,
			sizeof(err)) != 0, "unknown workload rejected");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_zero_blocks, &args, err,
			sizeof(err)) != 0, "--blocks 0 rejected");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_zero_iter, &args, err,
			sizeof(err)) != 0, "--iterations 0 rejected");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_huge_blocks, &args, err,
			sizeof(err)) != 0, "oversized --blocks rejected");
	TEST_ASSERT(membrane_bench_parse_args(2, argv_unknown, &args, err,
			sizeof(err)) != 0, "unknown option rejected");
	TEST_ASSERT(membrane_bench_parse_args(3, argv_matrix, &args, err,
			sizeof(err)) == 0, "--matrix --csv parses");
	TEST_ASSERT(args.want_matrix == 1 && args.want_csv == 1,
		"matrix/csv flags recorded");
	{
		char	*argv_trace[] = {"x", "--trace", "some/path.memkv"};
		char	*argv_trace_workload[] = {"x", "--trace", "p.memkv",
							"--workload", "synthetic-mixed"};
		char	*argv_trace_seed[] = {"x", "--trace", "p.memkv", "--seed",
							"1"};
		char	*argv_trace_blocks[] = {"x", "--trace", "p.memkv",
							"--blocks", "10"};

		TEST_ASSERT(membrane_bench_parse_args(3, argv_trace, &args, err,
				sizeof(err)) == 0, "--trace alone parses");
		TEST_ASSERT(args.cfg.trace_path != NULL
			&& strcmp(args.cfg.trace_path, "some/path.memkv") == 0,
			"trace path recorded");
		TEST_ASSERT(args.cfg.blocks == 0,
			"blocks defaults to 0 (uncapped) when --trace given without "
			"--blocks");
		TEST_ASSERT(membrane_bench_parse_args(4, argv_trace_workload, &args,
				err, sizeof(err)) != 0,
			"--workload combined with --trace rejected");
		TEST_ASSERT(membrane_bench_parse_args(5, argv_trace_seed, &args, err,
				sizeof(err)) != 0, "--seed combined with --trace rejected");
		TEST_ASSERT(membrane_bench_parse_args(5, argv_trace_blocks, &args,
				err, sizeof(err)) == 0,
			"--blocks combined with --trace is allowed (used as a cap)");
		TEST_ASSERT(args.cfg.blocks == 10,
			"explicit --blocks with --trace is kept, not reset to 0");
	}
}

static int	has(const char *s, const char *key)
{
	return (strstr(s, key) != NULL);
}

static void	test_json_has_required_fields(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	char						buf[4096];
	FILE						*f;

	cfg = default_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen succeeds");
	membrane_bench_print_json(&r, f);
	fclose(f);
	TEST_ASSERT(has(buf, "\"schema_version\""), "schema_version present");
	TEST_ASSERT(has(buf, "\"workload\""), "workload present");
	TEST_ASSERT(has(buf, "\"workload_kind\":\"synthetic\""),
		"workload_kind is synthetic");
	TEST_ASSERT(has(buf, "\"policy\""), "policy present");
	TEST_ASSERT(has(buf, "\"storage\""), "storage present");
	TEST_ASSERT(has(buf, "\"precision_counts\""), "precision_counts present");
	TEST_ASSERT(has(buf, "\"accuracy\""), "accuracy present");
	TEST_ASSERT(has(buf, "\"timing\""), "timing present");
	TEST_ASSERT(has(buf, "\"median_seconds\""), "median_seconds present");
	TEST_ASSERT(has(buf, "\"blocks_per_second\""),
		"blocks_per_second present");
	TEST_ASSERT(buf[0] == '{', "starts with an object");
}

static void	test_matrix_json_is_an_array(void)
{
	membrane_bench_config_t	base;
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n;
	char						buf[16384];
	FILE						*f;
	size_t						len;
	size_t						i;
	int							depth;
	int							balanced;

	base = default_cfg();
	TEST_ASSERT(membrane_bench_run_matrix(&base, rs,
			MEMBRANE_BENCH_MATRIX_CELLS, &n) == MEMBRANE_OK, "matrix run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen succeeds");
	membrane_bench_print_matrix_json(rs, n, f);
	fclose(f);
	TEST_ASSERT(buf[0] == '[', "matrix JSON is an array, not concatenated "
		"objects");
	len = strlen(buf);
	depth = 0;
	balanced = 1;
	i = 0;
	while (i < len)
	{
		if (buf[i] == '{' || buf[i] == '[')
			depth++;
		else if (buf[i] == '}' || buf[i] == ']')
			depth--;
		if (depth < 0)
			balanced = 0;
		i++;
	}
	TEST_ASSERT(balanced && depth == 0, "brackets/braces are balanced");
}

static void	test_csv_header_and_row_count(void)
{
	membrane_bench_config_t	base;
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n;
	char						buf[16384];
	FILE						*f;
	size_t						lines;
	size_t						i;

	base = default_cfg();
	TEST_ASSERT(membrane_bench_run_matrix(&base, rs,
			MEMBRANE_BENCH_MATRIX_CELLS, &n) == MEMBRANE_OK, "matrix run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen succeeds");
	membrane_bench_print_matrix_csv(rs, n, f);
	fclose(f);
	lines = 0;
	i = 0;
	while (buf[i] != '\0')
	{
		if (buf[i] == '\n')
			lines++;
		i++;
	}
	TEST_ASSERT(lines == n + 1, "one header line plus one line per result");
	TEST_ASSERT(strncmp(buf, "workload,workload_kind,policy,", 30) == 0,
		"header starts with the expected columns");
}

/* ------------------------------------------------------------------ */
/* Trace-mode tests (Product Phase 3)                                  */
/* ------------------------------------------------------------------ */

static void	test_trace_q4_only(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	write_trace_fixture(40, 32);
	cfg = trace_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_Q4_ONLY;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.is_trace, "result is marked as trace mode");
	TEST_ASSERT(strcmp(r.workload_kind_label, "trace") == 0, "kind label");
	TEST_ASSERT(r.q4_blocks == 40 && r.q8_blocks == 0, "q4-only forces Q4");
	TEST_ASSERT(r.validation_pass, "validates");
}

static void	test_trace_q8_only(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	write_trace_fixture(40, 32);
	cfg = trace_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_Q8_ONLY;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.q4_blocks == 0 && r.q8_blocks == 40, "q8-only forces Q8");
	TEST_ASSERT(r.validation_pass, "validates");
}

static void	test_trace_adaptive(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	write_trace_fixture(40, 32);
	cfg = trace_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.q4_blocks + r.q8_blocks == 40, "every block assigned once");
	TEST_ASSERT(r.validation_pass, "validates");
}

static void	test_trace_matrix_gives_3_rows(void)
{
	membrane_bench_config_t	base;
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n;
	size_t						i;

	write_trace_fixture(20, 32);
	base = trace_cfg();
	TEST_ASSERT(membrane_bench_run_matrix(&base, rs,
			MEMBRANE_BENCH_MATRIX_CELLS, &n) == MEMBRANE_OK, "matrix run");
	TEST_ASSERT(n == 3, "exactly 3 policy rows, not the 12-cell synthetic "
		"matrix");
	i = 0;
	while (i < n)
	{
		TEST_ASSERT(rs[i].is_trace, "every row is trace mode");
		i++;
	}
	TEST_ASSERT(rs[0].config.policy == MEMBRANE_BENCH_POLICY_Q4_ONLY
		&& rs[1].config.policy == MEMBRANE_BENCH_POLICY_Q8_ONLY
		&& rs[2].config.policy == MEMBRANE_BENCH_POLICY_ADAPTIVE,
		"the 3 rows are exactly the 3 policies");
}

static void	test_trace_blocks_cap(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	write_trace_fixture(100, 32);
	cfg = trace_cfg();
	cfg.blocks = 10;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.config.blocks == 10, "reported blocks reflects the cap");
	TEST_ASSERT(r.q4_blocks + r.q8_blocks == 10, "only the capped count "
		"was actually processed");
}

static void	test_trace_uncapped_uses_all_blocks(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;

	write_trace_fixture(77, 32);
	cfg = trace_cfg();
	TEST_ASSERT(cfg.blocks == 0, "trace_cfg default is uncapped");
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	TEST_ASSERT(r.config.blocks == 77, "uncapped run uses every block");
}

/* If trace mode ever silently fell back to (or mixed in) the synthetic
 * generator, these fields would matter and change the result; in real
 * trace mode they must be completely inert. */
static void	test_trace_ignores_workload_and_seed_fields(void)
{
	membrane_bench_config_t	cfg_a;
	membrane_bench_config_t	cfg_b;
	membrane_bench_result_t	r_a;
	membrane_bench_result_t	r_b;

	write_trace_fixture(30, 32);
	cfg_a = trace_cfg();
	cfg_a.workload = MEMBRANE_WORKLOAD_SYNTHETIC_MIXED;
	cfg_a.seed = 111;
	cfg_b = trace_cfg();
	cfg_b.workload = MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE;
	cfg_b.seed = 222;
	TEST_ASSERT(membrane_bench_run_one(&cfg_a, &r_a) == MEMBRANE_OK, "run a");
	TEST_ASSERT(membrane_bench_run_one(&cfg_b, &r_b) == MEMBRANE_OK, "run b");
	TEST_ASSERT(r_a.q4_blocks == r_b.q4_blocks
		&& r_a.q8_blocks == r_b.q8_blocks
		&& r_a.encoded_bytes == r_b.encoded_bytes
		&& r_a.q4_mean_rel_l2_error == r_b.q4_mean_rel_l2_error,
		"workload/seed have zero effect in trace mode -- the synthetic "
		"generator was never invoked");
}

static void	test_trace_non_timing_deterministic(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r1;
	membrane_bench_result_t	r2;

	write_trace_fixture(25, 32);
	cfg = trace_cfg();
	cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r1) == MEMBRANE_OK, "run 1");
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r2) == MEMBRANE_OK, "run 2");
	TEST_ASSERT(r1.q4_blocks == r2.q4_blocks && r1.q8_blocks == r2.q8_blocks
		&& r1.encoded_bytes == r2.encoded_bytes
		&& r1.reduction_ratio == r2.reduction_ratio
		&& r1.q4_mean_rel_l2_error == r2.q4_mean_rel_l2_error,
		"non-timing fields reproducible across separate trace runs");
}

static void	test_trace_json_has_required_fields(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	char						buf[4096];
	FILE						*f;

	write_trace_fixture(20, 32);
	cfg = trace_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_bench_print_json(&r, f);
	fclose(f);
	TEST_ASSERT(has(buf, "\"workload_kind\":\"trace\""), "workload_kind");
	TEST_ASSERT(has(buf, "\"seed\":null"), "seed is JSON null in trace mode");
	TEST_ASSERT(has(buf, "\"trace\":{"), "trace object present");
	TEST_ASSERT(has(buf, "\"dtype\":\"f16\""), "trace dtype");
	TEST_ASSERT(has(buf, "\"source_label\""), "trace source_label");
}

static void	test_trace_csv_has_trace_columns(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	char						buf[4096];
	FILE						*f;

	write_trace_fixture(20, 32);
	cfg = trace_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_bench_print_csv_header(f);
	membrane_bench_print_csv_row(&r, f);
	fclose(f);
	TEST_ASSERT(has(buf, "trace_format_version,trace_source_label"),
		"header carries the trace columns");
	TEST_ASSERT(has(buf, "trace,"), "row carries the trace workload_kind");
	TEST_ASSERT(has(buf, ",,"), "seed field is empty (CSV's N/A) in trace "
		"mode");
}

/* Locks in Phase 2 backward compatibility: adding trace mode must not
 * change one byte of the synthetic-mode JSON shape. */
static void	test_synthetic_json_unaffected_by_trace_mode(void)
{
	membrane_bench_config_t	cfg;
	membrane_bench_result_t	r;
	char						buf[4096];
	FILE						*f;

	cfg = default_cfg();
	TEST_ASSERT(membrane_bench_run_one(&cfg, &r) == MEMBRANE_OK, "run");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_bench_print_json(&r, f);
	fclose(f);
	TEST_ASSERT(!has(buf, "\"trace\":{"), "no trace object for synthetic "
		"results");
	TEST_ASSERT(!has(buf, "\"seed\":null"), "seed stays numeric for "
		"synthetic results");
	TEST_ASSERT(has(buf, "\"workload_kind\":\"synthetic\""),
		"workload_kind is still synthetic");
}

int	main(void)
{
	int	fd;

	test_each_policy_runs_and_validates();
	test_q4_only_forces_precision();
	test_q8_only_forces_precision();
	test_all_workloads_run_and_validate();
	test_same_config_deterministic_non_timing();
	test_different_seed_still_validates();
	test_matrix_cell_count();
	test_timing_metrics_are_sane();
	test_zero_blocks_rejected();
	test_zero_iterations_rejected();
	test_oversized_blocks_rejected();
	test_invalid_enum_values_rejected();
	test_weighted_mean_is_between_component_means();
	test_arg_parsing();
	test_json_has_required_fields();
	test_matrix_json_is_an_array();
	test_csv_header_and_row_count();
	test_synthetic_json_unaffected_by_trace_mode();
	fd = mkstemp(g_trace_path);
	TEST_ASSERT(fd >= 0, "trace fixture temp file");
	close(fd);
	test_trace_q4_only();
	test_trace_q8_only();
	test_trace_adaptive();
	test_trace_matrix_gives_3_rows();
	test_trace_blocks_cap();
	test_trace_uncapped_uses_all_blocks();
	test_trace_ignores_workload_and_seed_fields();
	test_trace_non_timing_deterministic();
	test_trace_json_has_required_fields();
	test_trace_csv_has_trace_columns();
	unlink(g_trace_path);
	return (0);
}
