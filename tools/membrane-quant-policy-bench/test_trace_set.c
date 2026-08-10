#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test_helpers.h"
#include "trace_set.h"

static char	g_dir[] = "/tmp/membrane-trace-set-XXXXXX";

static void	write_fixture(const char *name, uint64_t block_count,
				uint32_t epb, uint32_t magnitude)
{
	char		path[512];
	uint16_t	*blocks;
	FILE		*f;
	uint64_t	i;

	snprintf(path, sizeof(path), "%s/%s", g_dir, name);
	blocks = malloc((size_t)(block_count * epb) * sizeof(uint16_t));
	TEST_ASSERT(blocks != NULL, "allocate fixture blocks");
	i = 0;
	while (i < block_count * epb)
	{
		/* An f16 bit pattern that varies with `magnitude` so different
		 * fixtures plausibly land on different sides of the adaptive
		 * Q4 threshold -- the aggregation tests below never assume a
		 * specific split, only that the aggregate is the correct
		 * pooled function of whatever split actually happens. */
		blocks[i] = (uint16_t)((i * 13u + 5u) ^ (magnitude * 977u));
		i++;
	}
	f = fopen(path, "wb");
	TEST_ASSERT(f != NULL, "open trace fixture for write");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, epb,
			block_count, blocks, "test fixture, not a real capture", 0)
			== MEMBRANE_OK, "write trace fixture");
	fclose(f);
	free(blocks);
}

static void	clean_dir(void)
{
	DIR				*d;
	struct dirent	*ent;
	char			path[512];

	d = opendir(g_dir);
	TEST_ASSERT(d != NULL, "reopen trace-set dir fixture for cleanup");
	while ((ent = readdir(d)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
			continue ;
		snprintf(path, sizeof(path), "%s/%s", g_dir, ent->d_name);
		unlink(path);
	}
	closedir(d);
}

static membrane_bench_config_t	base_cfg(void)
{
	membrane_bench_config_t	cfg;

	cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	cfg.blocks = 0;
	cfg.seed = 0;
	cfg.iterations = 2;
	cfg.warmup = 1;
	cfg.trace_path = NULL;
	return (cfg);
}

/* ------------------------------------------------------------------ */
/* Discovery                                                           */
/* ------------------------------------------------------------------ */

static void	test_discover_parses_known_names_and_sorts(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];

	clean_dir();
	write_fixture("layer-001-k.memkv", 4, 32, 1);
	write_fixture("layer-000-v.memkv", 4, 32, 2);
	write_fixture("layer-000-k.memkv", 4, 32, 3);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_OK, err);
	TEST_ASSERT(n == 3, "3 candidates discovered");
	TEST_ASSERT(total == 12, "3 files x 4 blocks each");
	TEST_ASSERT(strstr(files[0].path, "layer-000-k.memkv") != NULL,
		"deterministic filename order: layer-000-k first");
	TEST_ASSERT(strstr(files[1].path, "layer-000-v.memkv") != NULL,
		"then layer-000-v");
	TEST_ASSERT(strstr(files[2].path, "layer-001-k.memkv") != NULL,
		"then layer-001-k");
	TEST_ASSERT(files[0].has_layer && files[0].layer == 0
		&& files[0].has_tensor && files[0].tensor == 'K',
		"layer-000-k.memkv parses to layer=0 tensor=K");
	TEST_ASSERT(files[1].has_tensor && files[1].tensor == 'V',
		"layer-000-v.memkv parses tensor=V");
	free(files);
}

static void	test_discover_unrecognized_name_kept_as_unknown(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];

	clean_dir();
	write_fixture("custom_capture.memkv", 4, 32, 7);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_OK, err);
	TEST_ASSERT(n == 1, "one candidate");
	TEST_ASSERT(!files[0].has_layer && !files[0].has_tensor,
		"a non-conforming filename is still benchmarked, layer/tensor "
		"just unknown");
	free(files);
}

static void	test_discover_ignores_non_memkv_files(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];
	char						path[512];
	FILE						*f;

	clean_dir();
	write_fixture("layer-000-k.memkv", 4, 32, 1);
	snprintf(path, sizeof(path), "%s/README.md", g_dir);
	f = fopen(path, "wb");
	TEST_ASSERT(f != NULL, "open non-memkv fixture");
	fprintf(f, "not a trace");
	fclose(f);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_OK, err);
	TEST_ASSERT(n == 1, "the non-.memkv file is silently ignored");
	free(files);
}

static void	test_discover_empty_dir_fails(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];

	clean_dir();
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_ERR_NOT_FOUND,
		"an empty directory fails clearly, not an empty success");
}

static void	test_discover_missing_dir_fails(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];

	TEST_ASSERT(membrane_trace_set_discover("/no/such/directory", &files, &n,
			&total, err, sizeof(err)) == MEMBRANE_ERR_IO,
		"a missing directory fails clearly");
}

/* membrane-kv-trace-export's own batch naming always zero-pads to
 * exactly 3 digits ("layer-000-k.memkv"), but the trace-set parser
 * accepts any digit string strtol can read -- so "layer-0-k.memkv" and
 * "layer-00-k.memkv" are two distinct filenames that both parse to the
 * same (layer=0, tensor=K), a real conflict a hand-assembled directory
 * could produce even though membrane-kv-trace-export itself never
 * emits it. */
static void	test_discover_duplicate_layer_tensor_rejected(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];

	clean_dir();
	write_fixture("layer-0-k.memkv", 4, 32, 1);
	write_fixture("layer-00-k.memkv", 4, 32, 2);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_ERR_INVALID_ARG,
		"two filenames resolving to the same (layer, tensor) are "
		"rejected, not silently overwritten or merged");
	TEST_ASSERT(err[0] != '\0', "an error message is provided");
}

static void	test_discover_symlink_rejected(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];
	char						target[512];
	char						link[512];

	clean_dir();
	write_fixture("layer-000-k.memkv", 4, 32, 1);
	snprintf(target, sizeof(target), "%s/layer-000-k.memkv", g_dir);
	snprintf(link, sizeof(link), "%s/layer-001-k.memkv", g_dir);
	TEST_ASSERT(symlink(target, link) == 0, "create symlink fixture");
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) == MEMBRANE_ERR_INVALID_ARG,
		"a symlink inside the trace directory is rejected outright");
}

static void	test_discover_malformed_trace_fails(void)
{
	membrane_trace_set_file_t	*files;
	size_t						n;
	uint64_t					total;
	char						err[256];
	char						path[512];
	FILE						*f;

	clean_dir();
	snprintf(path, sizeof(path), "%s/layer-000-k.memkv", g_dir);
	f = fopen(path, "wb");
	TEST_ASSERT(f != NULL, "open malformed fixture");
	fprintf(f, "this is not a valid MEMKV01 trace file");
	fclose(f);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n, &total, err,
			sizeof(err)) != MEMBRANE_OK,
		"a malformed trace fails the whole discovery, not silently "
		"dropped");
}

/* ------------------------------------------------------------------ */
/* Run + aggregation                                                   */
/* ------------------------------------------------------------------ */

static void	test_run_gives_3n_items(void)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	membrane_bench_config_t			base;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total;
	char							err[256];

	clean_dir();
	write_fixture("layer-000-k.memkv", 4, 32, 1);
	write_fixture("layer-000-v.memkv", 6, 32, 2);
	write_fixture("layer-001-k.memkv", 8, 32, 3);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n_files, &total,
			err, sizeof(err)) == MEMBRANE_OK, err);
	TEST_ASSERT(n_files == 3, "3 files discovered");
	base = base_cfg();
	TEST_ASSERT(membrane_trace_set_run(&base, files, n_files, &items,
			&n_items, &agg) == MEMBRANE_OK, "trace-set run succeeds");
	TEST_ASSERT(n_items == 9, "3 files x 3 policies = 9 items");
	free(files);
	free(items);
}

static void	test_run_reports_layer_tensor_per_item(void)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	membrane_bench_config_t			base;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total;
	char							err[256];
	size_t							i;

	clean_dir();
	write_fixture("layer-002-v.memkv", 4, 32, 5);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n_files, &total,
			err, sizeof(err)) == MEMBRANE_OK, err);
	base = base_cfg();
	TEST_ASSERT(membrane_trace_set_run(&base, files, n_files, &items,
			&n_items, &agg) == MEMBRANE_OK, "run succeeds");
	i = 0;
	while (i < n_items)
	{
		TEST_ASSERT(items[i].has_layer && items[i].layer == 2
			&& items[i].has_tensor && items[i].tensor == 'V',
			"every policy row for this file carries layer=2 tensor=V");
		TEST_ASSERT(strcmp(items[i].result.workload_kind_label, "trace")
			== 0, "workload_kind is trace");
		i++;
	}
	free(files);
	free(items);
}

static void	test_aggregate_matches_pooled_formula_over_actual_results(void)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	membrane_bench_config_t			base;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total;
	char							err[256];
	size_t							i;
	uint64_t						exp_traces;
	uint64_t						exp_blocks;
	uint64_t						exp_q4;
	uint64_t						exp_q8;
	uint64_t						exp_baseline;
	uint64_t						exp_encoded;
	double							exp_q4_num;
	double							exp_q8_num;
	double							exp_q4_max;
	double							exp_q8_max;
	double							exp_seconds;
	uint64_t						exp_any_q4;
	uint64_t						exp_all_q8;

	clean_dir();
	write_fixture("layer-000-k.memkv", 20, 32, 11);
	write_fixture("layer-000-v.memkv", 30, 32, 97);
	write_fixture("layer-001-k.memkv", 25, 32, 250);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n_files, &total,
			err, sizeof(err)) == MEMBRANE_OK, err);
	base = base_cfg();
	TEST_ASSERT(membrane_trace_set_run(&base, files, n_files, &items,
			&n_items, &agg) == MEMBRANE_OK, "run succeeds");
	/* Recompute the aggregate independently from the actual per-item
	 * results the run produced -- this locks in the pooling FORMULA
	 * (does not assume any particular adaptive q4/q8 split, which
	 * depends on real quant-error content this test does not control). */
	exp_traces = 0;
	exp_blocks = 0;
	exp_q4 = 0;
	exp_q8 = 0;
	exp_baseline = 0;
	exp_encoded = 0;
	exp_q4_num = 0.0;
	exp_q8_num = 0.0;
	exp_q4_max = 0.0;
	exp_q8_max = 0.0;
	exp_seconds = 0.0;
	exp_any_q4 = 0;
	exp_all_q8 = 0;
	i = 0;
	while (i < n_items)
	{
		if (items[i].result.config.policy == MEMBRANE_BENCH_POLICY_ADAPTIVE)
		{
			const membrane_bench_result_t	*r = &items[i].result;

			exp_traces++;
			exp_blocks += r->config.blocks;
			exp_q4 += r->q4_blocks;
			exp_q8 += r->q8_blocks;
			exp_baseline += r->baseline_bytes;
			exp_encoded += r->encoded_bytes;
			exp_q4_num += r->q4_mean_rel_l2_error * (double)r->q4_blocks;
			exp_q8_num += r->q8_mean_rel_l2_error * (double)r->q8_blocks;
			if (r->q4_max_rel_l2_error > exp_q4_max)
				exp_q4_max = r->q4_max_rel_l2_error;
			if (r->q8_max_rel_l2_error > exp_q8_max)
				exp_q8_max = r->q8_max_rel_l2_error;
			exp_seconds += r->median_seconds;
			if (r->q4_blocks > 0)
				exp_any_q4++;
			else
				exp_all_q8++;
		}
		i++;
	}
	TEST_ASSERT(agg.traces == exp_traces, "aggregate trace count");
	TEST_ASSERT(agg.total_blocks == exp_blocks, "aggregate block count");
	TEST_ASSERT(agg.q4_blocks == exp_q4 && agg.q8_blocks == exp_q8,
		"aggregate q4/q8 block counts are exact sums, not averages");
	TEST_ASSERT(agg.baseline_bytes == exp_baseline
		&& agg.encoded_bytes == exp_encoded,
		"aggregate storage bytes are exact sums");
	TEST_ASSERT(fabs(agg.storage_reduction_ratio
			- (double)(exp_baseline - exp_encoded) / (double)exp_baseline)
			< 1e-9,
		"storage_reduction_ratio computed from summed bytes, not "
		"averaged per-trace ratios");
	if (exp_q4 > 0)
		TEST_ASSERT(fabs(agg.q4_mean_rel_l2_error - exp_q4_num / (double)exp_q4)
				< 1e-9,
			"q4 pooled mean == sum(mean_i*count_i)/sum(count_i)");
	if (exp_q8 > 0)
		TEST_ASSERT(fabs(agg.q8_mean_rel_l2_error - exp_q8_num / (double)exp_q8)
				< 1e-9,
			"q8 pooled mean == sum(mean_i*count_i)/sum(count_i)");
	TEST_ASSERT(fabs(agg.q4_max_rel_l2_error - exp_q4_max) < 1e-12
		&& fabs(agg.q8_max_rel_l2_error - exp_q8_max) < 1e-12,
		"aggregate max is the max of per-trace maxes");
	TEST_ASSERT(fabs(agg.weighted_mean_rel_l2_error
			- (exp_q4_num + exp_q8_num) / (double)(exp_q4 + exp_q8)) < 1e-9,
		"combined weighted mean is the pooled mean over every block");
	TEST_ASSERT(fabs(agg.total_measured_seconds - exp_seconds) < 1e-9,
		"total measured seconds is the sum of per-trace median_seconds, "
		"never a sum of throughputs");
	if (agg.total_measured_seconds > 0.0)
		TEST_ASSERT(fabs(agg.blocks_per_second
				- (double)exp_blocks / exp_seconds) < 1e-6,
			"aggregate blocks_per_second == total blocks / total "
			"measured time, not summed per-trace throughput");
	TEST_ASSERT(agg.traces_any_q4 == exp_any_q4
		&& agg.traces_all_q8 == exp_all_q8
		&& agg.traces_any_q4 + agg.traces_all_q8 == agg.traces,
		"any-q4/all-q8 counts partition every adaptive trace");
	free(files);
	free(items);
}

/* ------------------------------------------------------------------ */
/* JSON / CSV / privacy                                                */
/* ------------------------------------------------------------------ */

static void	test_json_row_count_and_shape(void)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	membrane_bench_config_t			base;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total;
	char							err[256];
	char							buf[16384];
	FILE							*f;

	clean_dir();
	memset(buf, 0, sizeof(buf));
	write_fixture("layer-000-k.memkv", 4, 32, 1);
	write_fixture("layer-000-v.memkv", 4, 32, 2);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n_files, &total,
			err, sizeof(err)) == MEMBRANE_OK, err);
	base = base_cfg();
	TEST_ASSERT(membrane_trace_set_run(&base, files, n_files, &items,
			&n_items, &agg) == MEMBRANE_OK, "run succeeds");
	TEST_ASSERT(n_items == 6, "2 traces x 3 policies");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_trace_set_print_json(items, n_items, &agg, f);
	fclose(f);
	TEST_ASSERT(strstr(buf, "\"mode\":\"trace_set\"") != NULL, "mode field");
	TEST_ASSERT(strstr(buf, "\"results\":[") != NULL, "results array");
	TEST_ASSERT(strstr(buf, "\"aggregate\":{") != NULL, "aggregate object");
	TEST_ASSERT(strstr(buf, "\"workload_kind\":\"trace\"") != NULL,
		"per-item workload_kind is trace");
	TEST_ASSERT(strstr(buf, "\"layer\":0") != NULL, "known layer surfaced");
	TEST_ASSERT(strstr(buf, "\"tensor\":\"K\"") != NULL,
		"known tensor surfaced");
	TEST_ASSERT(strstr(buf, g_dir) == NULL,
		"the absolute trace directory path never appears in JSON output");
	free(files);
	free(items);
}

static void	test_csv_row_count_is_3n_plus_header(void)
{
	membrane_trace_set_file_t		*files;
	membrane_trace_set_item_t		*items;
	membrane_trace_set_aggregate_t	agg;
	membrane_bench_config_t			base;
	size_t							n_files;
	size_t							n_items;
	uint64_t						total;
	char							err[256];
	char							buf[32768];
	FILE							*f;
	size_t							lines;
	size_t							i;

	clean_dir();
	memset(buf, 0, sizeof(buf));
	write_fixture("layer-000-k.memkv", 4, 32, 1);
	write_fixture("layer-000-v.memkv", 4, 32, 2);
	write_fixture("layer-001-k.memkv", 4, 32, 3);
	write_fixture("layer-001-v.memkv", 4, 32, 4);
	TEST_ASSERT(membrane_trace_set_discover(g_dir, &files, &n_files, &total,
			err, sizeof(err)) == MEMBRANE_OK, err);
	TEST_ASSERT(n_files == 4, "4 traces discovered");
	base = base_cfg();
	TEST_ASSERT(membrane_trace_set_run(&base, files, n_files, &items,
			&n_items, &agg) == MEMBRANE_OK, "run succeeds");
	TEST_ASSERT(n_items == 12, "4 traces x 3 policies = 12 (3N)");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_trace_set_print_csv(items, n_items, f);
	fclose(f);
	lines = 0;
	i = 0;
	while (buf[i] != '\0')
	{
		if (buf[i] == '\n')
			lines++;
		i++;
	}
	TEST_ASSERT(lines == n_items + 1,
		"exactly one header line plus 3N data rows");
	TEST_ASSERT(strstr(buf, g_dir) == NULL,
		"the absolute trace directory path never appears in CSV output");
	free(files);
	free(items);
}

/* ------------------------------------------------------------------ */
/* CLI arg parsing (bench_io.c's --trace-dir handling)                 */
/* ------------------------------------------------------------------ */

static void	test_arg_parsing_trace_dir(void)
{
	membrane_bench_args_t	args;
	char					err[160];
	char	*argv_bare[] = {"x", "--trace-dir", "/tmp/traces"};
	char	*argv_matrix[] = {"x", "--trace-dir", "/tmp/traces", "--matrix"};
	char	*argv_both[] = {"x", "--trace", "p.memkv", "--trace-dir",
						"/tmp/traces", "--matrix"};
	char	*argv_workload[] = {"x", "--trace-dir", "/tmp/traces", "--matrix",
						"--workload", "synthetic-mixed"};

	TEST_ASSERT(membrane_bench_parse_args(3, argv_bare, &args, err,
			sizeof(err)) != 0, "--trace-dir without --matrix rejected");
	TEST_ASSERT(membrane_bench_parse_args(4, argv_matrix, &args, err,
			sizeof(err)) == 0, "--trace-dir --matrix parses");
	TEST_ASSERT(args.trace_dir != NULL
		&& strcmp(args.trace_dir, "/tmp/traces") == 0,
		"trace_dir recorded");
	TEST_ASSERT(args.cfg.trace_path == NULL,
		"cfg.trace_path stays NULL for trace-set mode");
	TEST_ASSERT(membrane_bench_parse_args(6, argv_both, &args, err,
			sizeof(err)) != 0, "--trace combined with --trace-dir rejected");
	TEST_ASSERT(membrane_bench_parse_args(6, argv_workload, &args, err,
			sizeof(err)) != 0,
		"--workload combined with --trace-dir rejected");
}

int	main(void)
{
	TEST_ASSERT(mkdtemp(g_dir) != NULL, "trace-set dir fixture");
	test_discover_parses_known_names_and_sorts();
	test_discover_unrecognized_name_kept_as_unknown();
	test_discover_ignores_non_memkv_files();
	test_discover_empty_dir_fails();
	test_discover_missing_dir_fails();
	test_discover_duplicate_layer_tensor_rejected();
	test_discover_symlink_rejected();
	test_discover_malformed_trace_fails();
	test_run_gives_3n_items();
	test_run_reports_layer_tensor_per_item();
	test_aggregate_matches_pooled_formula_over_actual_results();
	test_json_row_count_and_shape();
	test_csv_row_count_is_3n_plus_header();
	test_arg_parsing_trace_dir();
	clean_dir();
	return (0);
}
