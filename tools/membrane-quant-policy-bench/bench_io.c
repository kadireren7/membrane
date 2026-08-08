#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/quant_select.h"
#include "bench_core.h"

static int	parse_u32(const char *s, uint32_t *out)
{
	unsigned long	v;
	char			*end;

	if (s == NULL || s[0] == '\0' || s[0] == '-' || !isdigit((unsigned char)s[0]))
		return (0);
	errno = 0;
	v = strtoul(s, &end, 10);
	if (*end != '\0' || errno == ERANGE || v > UINT32_MAX)
		return (0);
	*out = (uint32_t)v;
	return (1);
}

static int	usage_error(char *err_buf, size_t err_cap, const char *msg)
{
	if (err_buf != NULL && err_cap > 0)
		snprintf(err_buf, err_cap, "%s", msg);
	return (MEMBRANE_BENCH_EXIT_USAGE_ERROR);
}

static int	take_next(int argc, char **argv, int *i, const char **val,
					char *err_buf, size_t err_cap, const char *flag)
{
	char	msg[128];

	if (*i + 1 >= argc)
	{
		snprintf(msg, sizeof(msg), "%s requires a value", flag);
		return (usage_error(err_buf, err_cap, msg));
	}
	*i += 1;
	*val = argv[*i];
	return (0);
}

static int	apply_u32_flag(const char *flag, const char *val, uint32_t max,
					int reject_zero, uint32_t *out, char *err_buf,
					size_t err_cap)
{
	char	msg[160];

	if (!parse_u32(val, out) || (reject_zero && *out == 0))
	{
		snprintf(msg, sizeof(msg), "%s must be a %s integer, got '%s'",
			flag, reject_zero ? "positive" : "non-negative", val);
		return (usage_error(err_buf, err_cap, msg));
	}
	if (*out > max)
	{
		snprintf(msg, sizeof(msg), "%s exceeds the maximum supported (%u)",
			flag, max);
		return (usage_error(err_buf, err_cap, msg));
	}
	return (0);
}

int	membrane_bench_parse_args(int argc, char **argv,
		membrane_bench_args_t *args, char *err_buf, size_t err_cap)
{
	const char	*val;
	char		msg[160];
	int			rc;
	int			i;

	args->cfg.workload = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	args->cfg.policy = MEMBRANE_BENCH_POLICY_ADAPTIVE;
	args->cfg.blocks = MEMBRANE_BENCH_DEFAULT_BLOCKS;
	args->cfg.seed = MEMBRANE_BENCH_DEFAULT_SEED;
	args->cfg.iterations = MEMBRANE_BENCH_DEFAULT_ITERATIONS;
	args->cfg.warmup = MEMBRANE_BENCH_DEFAULT_WARMUP;
	args->want_json = 0;
	args->want_csv = 0;
	args->want_matrix = 0;
	args->want_help = 0;
	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
			return (args->want_help = 1, 0);
		else if (strcmp(argv[i], "--json") == 0)
			args->want_json = 1;
		else if (strcmp(argv[i], "--csv") == 0)
			args->want_csv = 1;
		else if (strcmp(argv[i], "--matrix") == 0)
			args->want_matrix = 1;
		else if (strcmp(argv[i], "--workload") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap,
					"--workload");
			if (rc != 0)
				return (rc);
			if (!membrane_workload_kind_from_name(val, &args->cfg.workload))
			{
				snprintf(msg, sizeof(msg), "unknown --workload '%s'", val);
				return (usage_error(err_buf, err_cap, msg));
			}
		}
		else if (strcmp(argv[i], "--policy") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap,
					"--policy");
			if (rc != 0)
				return (rc);
			if (!membrane_bench_policy_from_name(val, &args->cfg.policy))
			{
				snprintf(msg, sizeof(msg), "unknown --policy '%s'", val);
				return (usage_error(err_buf, err_cap, msg));
			}
		}
		else if (strcmp(argv[i], "--blocks") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap,
					"--blocks");
			if (rc != 0)
				return (rc);
			rc = apply_u32_flag("--blocks", val, MEMBRANE_BENCH_MAX_BLOCKS, 1,
					&args->cfg.blocks, err_buf, err_cap);
			if (rc != 0)
				return (rc);
		}
		else if (strcmp(argv[i], "--seed") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap, "--seed");
			if (rc != 0)
				return (rc);
			rc = apply_u32_flag("--seed", val, UINT32_MAX, 0,
					&args->cfg.seed, err_buf, err_cap);
			if (rc != 0)
				return (rc);
		}
		else if (strcmp(argv[i], "--iterations") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap,
					"--iterations");
			if (rc != 0)
				return (rc);
			rc = apply_u32_flag("--iterations", val,
					MEMBRANE_BENCH_MAX_ITERATIONS, 1, &args->cfg.iterations,
					err_buf, err_cap);
			if (rc != 0)
				return (rc);
		}
		else if (strcmp(argv[i], "--warmup") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap,
					"--warmup");
			if (rc != 0)
				return (rc);
			rc = apply_u32_flag("--warmup", val, MEMBRANE_BENCH_MAX_WARMUP, 0,
					&args->cfg.warmup, err_buf, err_cap);
			if (rc != 0)
				return (rc);
		}
		else
		{
			snprintf(msg, sizeof(msg), "unknown option: %s", argv[i]);
			return (usage_error(err_buf, err_cap, msg));
		}
		i++;
	}
	return (0);
}

/* ------------------------------------------------------------------ */
/* Human-readable output                                               */
/* ------------------------------------------------------------------ */

static void	print_human_one(const membrane_bench_result_t *r, FILE *f)
{
	fprintf(f, "Workload           %s (%s)\n",
		membrane_workload_kind_name(r->config.workload),
		MEMBRANE_WORKLOAD_KIND_LABEL);
	fprintf(f, "Policy             %s\n",
		membrane_bench_policy_name(r->config.policy));
	fprintf(f, "Blocks             %u (elems/block %u, seed %u)\n",
		r->config.blocks, r->elems_per_block, r->config.seed);
	fprintf(f, "Precision          q4=%llu q8=%llu\n",
		(unsigned long long)r->q4_blocks, (unsigned long long)r->q8_blocks);
	fprintf(f, "Storage            baseline=%llu encoded=%llu "
		"reduction=%.2f%%\n",
		(unsigned long long)r->baseline_bytes,
		(unsigned long long)r->encoded_bytes, r->reduction_ratio * 100.0);
	fprintf(f, "Accuracy           q4 rel-L2 mean=%.4f max=%.4f | "
		"q8 rel-L2 mean=%.4f max=%.4f\n",
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error,
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error);
	fprintf(f, "Validation         decoded=%llu/%llu failures=%llu "
		"nondeterminism=%llu -> %s\n",
		(unsigned long long)r->blocks_decoded,
		(unsigned long long)r->config.blocks,
		(unsigned long long)r->decode_failures,
		(unsigned long long)r->encode_nondeterminism,
		r->validation_pass ? "PASS" : "FAIL");
	fprintf(f, "Timing (local CPU) median=%.6fs min=%.6fs max=%.6fs "
		"over %u iteration(s), %u warmup -- not an LLM/hardware "
		"benchmark\n",
		r->median_seconds, r->min_seconds, r->max_seconds,
		r->config.iterations, r->config.warmup);
	fprintf(f, "Throughput         %.0f blocks/s, %.0f elements/s "
		"(backend: %s)\n",
		r->blocks_per_second, r->elements_per_second, r->simd_backend_name);
}

void	membrane_bench_print_human(const membrane_bench_result_t *r, FILE *f)
{
	fprintf(f, "MEMBRANE quant policy benchmark (Product Phase 2)\n\n");
	print_human_one(r, f);
}

void	membrane_bench_print_matrix_human(const membrane_bench_result_t *rs,
			size_t n, FILE *f)
{
	size_t	i;

	fprintf(f, "MEMBRANE quant policy benchmark -- matrix "
		"(all workloads are SYNTHETIC; timing is local CPU only)\n\n");
	fprintf(f, "%-24s %-10s %10s %10s %12s %12s\n",
		"Workload", "Policy", "Storage%", "MeanErr", "MedianMs",
		"Blocks/s");
	i = 0;
	while (i < n)
	{
		fprintf(f, "%-24s %-10s %9.2f%% %10.4f %12.3f %12.0f\n",
			membrane_workload_kind_name(rs[i].config.workload),
			membrane_bench_policy_name(rs[i].config.policy),
			rs[i].reduction_ratio * 100.0,
			rs[i].q4_mean_rel_l2_error + rs[i].q8_mean_rel_l2_error,
			rs[i].median_seconds * 1000.0, rs[i].blocks_per_second);
		i++;
	}
}

/* ------------------------------------------------------------------ */
/* JSON output -- every string value here is a compile-time-controlled */
/* constant (enum names, PASS/FAIL, backend names), never user/data-    */
/* derived text, so no runtime escaping is required (matches the        */
/* existing convention in tools/membrane-store-bench and                */
/* tools/membrane-demo).                                                */
/* ------------------------------------------------------------------ */

static void	print_json_body(const membrane_bench_result_t *r, FILE *f)
{
	fprintf(f, "{\"schema_version\":%d,"
		"\"benchmark\":{\"workload\":\"%s\",\"workload_kind\":\"%s\","
		"\"policy\":\"%s\",\"blocks\":%u,\"elements_per_block\":%u,"
		"\"seed\":%u,\"iterations\":%u,\"warmup\":%u},"
		"\"storage\":{\"baseline_bytes\":%llu,\"encoded_bytes\":%llu,"
		"\"absolute_saved_bytes\":%lld,\"storage_reduction_ratio\":%.6f},"
		"\"precision_counts\":{\"q4\":%llu,\"q8\":%llu},"
		"\"accuracy\":{\"q4_mean_rel_l2_error\":%.6f,"
		"\"q4_max_rel_l2_error\":%.6f,\"q8_mean_rel_l2_error\":%.6f,"
		"\"q8_max_rel_l2_error\":%.6f,\"decode_failures\":%llu,"
		"\"encode_nondeterminism\":%llu,\"validation_pass\":%s},"
		"\"timing\":{\"median_seconds\":%.6f,\"min_seconds\":%.6f,"
		"\"max_seconds\":%.6f,\"blocks_per_second\":%.3f,"
		"\"elements_per_second\":%.3f},"
		"\"backend\":{\"simd\":\"%s\"}}",
		MEMBRANE_BENCH_SCHEMA_VERSION,
		membrane_workload_kind_name(r->config.workload),
		MEMBRANE_WORKLOAD_KIND_LABEL,
		membrane_bench_policy_name(r->config.policy),
		r->config.blocks, r->elems_per_block, r->config.seed,
		r->config.iterations, r->config.warmup,
		(unsigned long long)r->baseline_bytes,
		(unsigned long long)r->encoded_bytes, (long long)r->saved_bytes,
		r->reduction_ratio,
		(unsigned long long)r->q4_blocks, (unsigned long long)r->q8_blocks,
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error,
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error,
		(unsigned long long)r->decode_failures,
		(unsigned long long)r->encode_nondeterminism,
		r->validation_pass ? "true" : "false",
		r->median_seconds, r->min_seconds, r->max_seconds,
		r->blocks_per_second, r->elements_per_second,
		r->simd_backend_name);
}

void	membrane_bench_print_json(const membrane_bench_result_t *r, FILE *f)
{
	print_json_body(r, f);
	fprintf(f, "\n");
}

void	membrane_bench_print_matrix_json(const membrane_bench_result_t *rs,
			size_t n, FILE *f)
{
	size_t	i;

	fprintf(f, "[");
	i = 0;
	while (i < n)
	{
		if (i > 0)
			fprintf(f, ",");
		print_json_body(&rs[i], f);
		i++;
	}
	fprintf(f, "]\n");
}

/* ------------------------------------------------------------------ */
/* CSV output -- same escaping argument as the JSON printer above:     */
/* every field is a number or one of our own fixed enum-name strings,  */
/* none of which can contain a comma, quote, or newline, so no CSV     */
/* field-quoting is needed for correctness here.                       */
/* ------------------------------------------------------------------ */

void	membrane_bench_print_csv_header(FILE *f)
{
	fprintf(f, "workload,workload_kind,policy,blocks,elements_per_block,"
		"seed,iterations,warmup,baseline_bytes,encoded_bytes,"
		"absolute_saved_bytes,storage_reduction_ratio,q4_blocks,q8_blocks,"
		"q4_mean_rel_l2_error,q4_max_rel_l2_error,q8_mean_rel_l2_error,"
		"q8_max_rel_l2_error,decode_failures,encode_nondeterminism,"
		"validation_pass,median_seconds,min_seconds,max_seconds,"
		"blocks_per_second,elements_per_second,simd_backend\n");
}

void	membrane_bench_print_csv_row(const membrane_bench_result_t *r, FILE *f)
{
	fprintf(f, "%s,%s,%s,%u,%u,%u,%u,%u,%llu,%llu,%lld,%.6f,%llu,%llu,"
		"%.6f,%.6f,%.6f,%.6f,%llu,%llu,%s,%.6f,%.6f,%.6f,%.3f,%.3f,%s\n",
		membrane_workload_kind_name(r->config.workload),
		MEMBRANE_WORKLOAD_KIND_LABEL,
		membrane_bench_policy_name(r->config.policy),
		r->config.blocks, r->elems_per_block, r->config.seed,
		r->config.iterations, r->config.warmup,
		(unsigned long long)r->baseline_bytes,
		(unsigned long long)r->encoded_bytes, (long long)r->saved_bytes,
		r->reduction_ratio,
		(unsigned long long)r->q4_blocks, (unsigned long long)r->q8_blocks,
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error,
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error,
		(unsigned long long)r->decode_failures,
		(unsigned long long)r->encode_nondeterminism,
		r->validation_pass ? "PASS" : "FAIL",
		r->median_seconds, r->min_seconds, r->max_seconds,
		r->blocks_per_second, r->elements_per_second,
		r->simd_backend_name);
}

void	membrane_bench_print_matrix_csv(const membrane_bench_result_t *rs,
			size_t n, FILE *f)
{
	size_t	i;

	membrane_bench_print_csv_header(f);
	i = 0;
	while (i < n)
	{
		membrane_bench_print_csv_row(&rs[i], f);
		i++;
	}
}
