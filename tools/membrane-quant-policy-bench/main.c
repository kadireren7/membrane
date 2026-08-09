#include <stdio.h>
#include <stdlib.h>

#include "bench_core.h"
#include "trace_set.h"

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-quant-policy-bench [options]\n"
		"\n"
		"Offline benchmark of MEMBRANE's maintained Q4_0/Q8_0 quantization\n"
		"path across synthetic workloads, or one real captured KV trace\n"
		"(--trace), against 3 precision policies: no model download, no\n"
		"network access, no llama.cpp dependency, in either mode.\n"
		"\n"
		"Timed region: exactly one block's precision selection + encode +\n"
		"decode + validation (membrane_bench_process_block). Workload/\n"
		"trace generation-or-loading and result formatting are NOT timed\n"
		"-- a trace's blocks are fully read into memory once, untimed,\n"
		"before timing starts. --warmup passes run the identical timed\n"
		"region and are discarded; the reported primary statistic is the\n"
		"MEDIAN wall time (CLOCK_MONOTONIC) across --iterations timed\n"
		"passes; min/max are also reported. This is local CPU benchmark\n"
		"runtime only -- not an LLM inference speedup claim and not a\n"
		"hardware performance claim, in either mode.\n"
		"\n"
		"  --workload NAME   synthetic-default | synthetic-low-variance |\n"
		"                    synthetic-high-variance | synthetic-mixed\n"
		"                    (default: synthetic-default; ignored under\n"
		"                    --matrix, which runs all four; rejected\n"
		"                    together with --trace)\n"
		"  --policy NAME     q4-only | q8-only | adaptive\n"
		"                    (default: adaptive; ignored under --matrix,\n"
		"                    which runs all three)\n"
		"  --blocks N        synthetic mode: number of generated blocks\n"
		"                    (default %u). Trace mode: caps how many of\n"
		"                    the trace's blocks are used (default: all\n"
		"                    of them)\n"
		"  --seed N          PRNG seed for the synthetic workload\n"
		"                    (default %u; rejected together with --trace)\n"
		"  --trace PATH      benchmark a real captured KV trace instead\n"
		"                    of a synthetic workload (see\n"
		"                    tools/membrane-kv-trace-export and\n"
		"                    docs/kv-trace-format.md). --matrix then runs\n"
		"                    exactly the 3 policies against this one\n"
		"                    trace, not the 4-workload synthetic sweep\n"
		"  --trace-dir DIR   benchmark every .memkv trace directly\n"
		"                    inside DIR (e.g. a membrane-kv-trace-export\n"
		"                    --output-dir batch): all 3 policies per\n"
		"                    trace plus a cross-trace adaptive summary.\n"
		"                    Requires --matrix; rejected together with\n"
		"                    --trace/--workload/--seed\n"
		"  --iterations N    timed iterations, median reported "
		"(default %u)\n"
		"  --warmup N        discarded warmup iterations (default %u)\n"
		"  --matrix          run the full policy matrix (synthetic: %d\n"
		"                    cells; --trace: 3 cells)\n"
		"  --json            print machine-readable JSON\n"
		"  --csv             print CSV (one header + one row per result)\n"
		"  --help            print this message and exit\n",
		MEMBRANE_BENCH_DEFAULT_BLOCKS, MEMBRANE_BENCH_DEFAULT_SEED,
		MEMBRANE_BENCH_DEFAULT_ITERATIONS, MEMBRANE_BENCH_DEFAULT_WARMUP,
		(int)MEMBRANE_BENCH_MATRIX_CELLS);
}

static const char	*run_failure_reason(membrane_status_t st, int is_trace_set)
{
	if (st == MEMBRANE_ERR_ALLOC_FAILED)
		return ("allocation failure");
	if (st == MEMBRANE_ERR_IO)
	{
		if (is_trace_set)
			return ("cannot read one of the discovered trace files "
				"(--trace-dir contents changed, or a file became "
				"unreadable, between discovery and benchmarking?)");
		return ("cannot read the trace file (--trace path wrong or "
			"unreadable?)");
	}
	if (st == MEMBRANE_ERR_CORRUPT_DATA)
		return ("the trace file is corrupt or truncated");
	if (st == MEMBRANE_ERR_UNIMPLEMENTED)
		return ("the trace's dtype or elements_per_block is not "
			"supported (elements_per_block must be a multiple of 32)");
	return ("invalid configuration");
}

static int	run_single(const membrane_bench_args_t *args)
{
	membrane_bench_result_t	r;
	membrane_status_t			st;

	st = membrane_bench_run_one(&args->cfg, &r);
	if (st != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: benchmark run "
			"failed: %s\n", run_failure_reason(st, 0));
		return (1);
	}
	if (args->want_csv)
	{
		membrane_bench_print_csv_header(stdout);
		membrane_bench_print_csv_row(&r, stdout);
	}
	else if (args->want_json)
		membrane_bench_print_json(&r, stdout);
	else
		membrane_bench_print_human(&r, stdout);
	return (r.validation_pass ? 0 : 1);
}

static int	run_matrix(const membrane_bench_args_t *args)
{
	membrane_bench_result_t	rs[MEMBRANE_BENCH_MATRIX_CELLS];
	size_t						n;
	size_t						i;
	int							all_pass;
	membrane_status_t			st;

	st = membrane_bench_run_matrix(&args->cfg, rs, MEMBRANE_BENCH_MATRIX_CELLS,
			&n);
	if (st != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: matrix run failed: "
			"%s\n", run_failure_reason(st, 0));
		return (1);
	}
	if (args->want_csv)
		membrane_bench_print_matrix_csv(rs, n, stdout);
	else if (args->want_json)
		membrane_bench_print_matrix_json(rs, n, stdout);
	else
		membrane_bench_print_matrix_human(rs, n, stdout);
	all_pass = 1;
	i = 0;
	while (i < n)
	{
		if (!rs[i].validation_pass)
			all_pass = 0;
		i++;
	}
	return (all_pass ? 0 : 1);
}

static int	run_trace_set(const membrane_bench_args_t *args)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total_blocks;
	char							err_buf[256];
	membrane_status_t				st;
	int								all_pass;
	size_t							i;

	st = membrane_trace_set_discover(args->trace_dir, &files, &n_files,
			&total_blocks, err_buf, sizeof(err_buf));
	if (st != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: %s\n", err_buf);
		return (1);
	}
	st = membrane_trace_set_run(&args->cfg, files, n_files, &items, &n_items,
			&agg);
	free(files);
	if (st != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: trace-set run "
			"failed: %s\n", run_failure_reason(st, 1));
		return (1);
	}
	if (args->want_csv)
		membrane_trace_set_print_csv(items, n_items, stdout);
	else if (args->want_json)
		membrane_trace_set_print_json(items, n_items, &agg, stdout);
	else
		membrane_trace_set_print_human(items, n_items, &agg, stdout);
	all_pass = 1;
	i = 0;
	while (i < n_items)
	{
		if (!items[i].result.validation_pass)
			all_pass = 0;
		i++;
	}
	free(items);
	return (all_pass ? 0 : 1);
}

int	main(int argc, char **argv)
{
	membrane_bench_args_t	args;
	char					err_buf[160];
	int						rc;

	rc = membrane_bench_parse_args(argc, argv, &args, err_buf,
			sizeof(err_buf));
	if (args.want_help)
		return (usage(stdout), 0);
	if (rc != 0)
	{
		fprintf(stderr, "membrane-quant-policy-bench: %s\n", err_buf);
		usage(stderr);
		return (rc);
	}
	if (args.trace_dir != NULL)
		return (run_trace_set(&args));
	if (args.want_matrix)
		return (run_matrix(&args));
	return (run_single(&args));
}
