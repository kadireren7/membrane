#include <stdio.h>

#include "bench_core.h"

static void	usage(FILE *out)
{
	fprintf(out,
		"Usage: membrane-quant-policy-bench [options]\n"
		"\n"
		"Offline benchmark of MEMBRANE's maintained Q4_0/Q8_0 quantization\n"
		"path across synthetic workloads and precision policies: no model\n"
		"download, no network access.\n"
		"\n"
		"Timed region: exactly one block's precision selection + encode +\n"
		"decode + validation (membrane_bench_process_block). Workload\n"
		"generation and result formatting are NOT timed. --warmup passes\n"
		"run the identical timed region and are discarded; the reported\n"
		"primary statistic is the MEDIAN wall time (CLOCK_MONOTONIC) across\n"
		"--iterations timed passes; min/max are also reported. This is\n"
		"local CPU benchmark runtime only -- not an LLM inference speedup\n"
		"claim and not a hardware performance claim.\n"
		"\n"
		"  --workload NAME   synthetic-default | synthetic-low-variance |\n"
		"                    synthetic-high-variance | synthetic-mixed\n"
		"                    (default: synthetic-default; ignored under\n"
		"                    --matrix, which runs all four)\n"
		"  --policy NAME     q4-only | q8-only | adaptive\n"
		"                    (default: adaptive; ignored under --matrix,\n"
		"                    which runs all three)\n"
		"  --blocks N        number of synthetic blocks (default %u)\n"
		"  --seed N          PRNG seed for the synthetic workload "
		"(default %u)\n"
		"  --iterations N    timed iterations, median reported "
		"(default %u)\n"
		"  --warmup N        discarded warmup iterations (default %u)\n"
		"  --matrix          run the full workload x policy matrix "
		"(%d cells)\n"
		"  --json            print machine-readable JSON\n"
		"  --csv             print CSV (one header + one row per result)\n"
		"  --help            print this message and exit\n",
		MEMBRANE_BENCH_DEFAULT_BLOCKS, MEMBRANE_BENCH_DEFAULT_SEED,
		MEMBRANE_BENCH_DEFAULT_ITERATIONS, MEMBRANE_BENCH_DEFAULT_WARMUP,
		(int)MEMBRANE_BENCH_MATRIX_CELLS);
}

static int	run_single(const membrane_bench_args_t *args)
{
	membrane_bench_result_t	r;

	if (membrane_bench_run_one(&args->cfg, &r) != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: benchmark run "
			"failed (allocation failure?)\n");
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

	if (membrane_bench_run_matrix(&args->cfg, rs, MEMBRANE_BENCH_MATRIX_CELLS,
			&n) != MEMBRANE_OK)
	{
		fprintf(stderr, "membrane-quant-policy-bench: matrix run failed "
			"(allocation failure?)\n");
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
	if (args.want_matrix)
		return (run_matrix(&args));
	return (run_single(&args));
}
