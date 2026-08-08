#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/quant_select.h"

#include "demo_core.h"

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
	return (MEMBRANE_DEMO_EXIT_USAGE_ERROR);
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

int	membrane_demo_parse_args(int argc, char **argv,
		membrane_demo_config_t *cfg, int *want_json, int *want_help,
		char *err_buf, size_t err_cap)
{
	const char	*val;
	char		msg[160];
	int			rc;
	int			i;

	cfg->blocks = MEMBRANE_DEMO_DEFAULT_BLOCKS;
	cfg->seed = MEMBRANE_DEMO_DEFAULT_SEED;
	*want_json = 0;
	*want_help = 0;
	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
			return (*want_help = 1, 0);
		else if (strcmp(argv[i], "--json") == 0)
			*want_json = 1;
		else if (strcmp(argv[i], "--blocks") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap, "--blocks");
			if (rc != 0)
				return (rc);
			if (!parse_u32(val, &cfg->blocks) || cfg->blocks == 0)
			{
				snprintf(msg, sizeof(msg),
					"--blocks must be a positive integer, got '%s'", val);
				return (usage_error(err_buf, err_cap, msg));
			}
			if (cfg->blocks > MEMBRANE_DEMO_MAX_BLOCKS)
			{
				snprintf(msg, sizeof(msg),
					"--blocks exceeds the maximum supported (%u)",
					MEMBRANE_DEMO_MAX_BLOCKS);
				return (usage_error(err_buf, err_cap, msg));
			}
		}
		else if (strcmp(argv[i], "--seed") == 0)
		{
			rc = take_next(argc, argv, &i, &val, err_buf, err_cap, "--seed");
			if (rc != 0)
				return (rc);
			if (!parse_u32(val, &cfg->seed))
			{
				snprintf(msg, sizeof(msg),
					"--seed must be a non-negative integer, got '%s'", val);
				return (usage_error(err_buf, err_cap, msg));
			}
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

void	membrane_demo_print_human(const membrane_demo_result_t *r, FILE *f)
{
	fprintf(f, "MEMBRANE\n");
	fprintf(f, "Adaptive mixed-precision KV-cache demonstration\n\n");
	fprintf(f, "Workload\n");
	fprintf(f, "  Blocks processed        %u\n", r->config.blocks);
	fprintf(f, "  Elements per block      %u\n", r->elems_per_block);
	fprintf(f, "  Seed                    %u\n\n", r->config.seed);
	fprintf(f, "Precision policy (Q4 accepted if rel-L2 error <= %.2f)\n",
		MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR);
	fprintf(f, "  Q4 blocks               %llu\n",
		(unsigned long long)r->q4_blocks);
	fprintf(f, "  Q8 blocks               %llu\n\n",
		(unsigned long long)r->q8_blocks);
	fprintf(f, "Storage (baseline: FP32 bytes, same logical element count)\n");
	fprintf(f, "  Baseline bytes          %llu\n",
		(unsigned long long)r->baseline_bytes);
	fprintf(f, "  MEMBRANE bytes          %llu\n",
		(unsigned long long)r->membrane_bytes);
	fprintf(f, "  Reduction               %.2f%%\n\n",
		r->reduction_ratio * 100.0);
	fprintf(f, "Validation (lossy reconstruction -- lower is better, "
		"not bit-exact)\n");
	fprintf(f, "  Blocks decoded          %llu / %llu\n",
		(unsigned long long)r->blocks_decoded,
		(unsigned long long)r->config.blocks);
	fprintf(f, "  Decode failures         %llu\n",
		(unsigned long long)r->decode_failures);
	fprintf(f, "  Encode nondeterminism   %llu\n",
		(unsigned long long)r->encode_nondeterminism);
	fprintf(f, "  Q4 rel-L2 error         mean %.4f  max %.4f\n",
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error);
	fprintf(f, "  Q8 rel-L2 error         mean %.4f  max %.4f\n",
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error);
	fprintf(f, "  Result                  %s\n\n",
		r->validation_pass ? "PASS" : "FAIL");
	fprintf(f, "Backend\n");
	fprintf(f, "  CPU quantization        active (%s)\n",
		r->simd_backend_name);
	fprintf(f, "  Maintained RTL          available, not invoked "
		"(tools/membrane-hw-sim)\n");
	fprintf(f, "  Research-only features  see membrane-research\n\n");
	fprintf(f, "Local demo runtime: %.4fs (not a benchmark)\n",
		r->elapsed_seconds);
}

void	membrane_demo_print_json(const membrane_demo_result_t *r, FILE *f)
{
	fprintf(f, "{\"schema_version\":%d,"
		"\"configuration\":{\"blocks\":%u,\"seed\":%u,"
		"\"elems_per_block\":%u},"
		"\"blocks_processed\":%u,\"seed\":%u,"
		"\"precision_counts\":{\"q4\":%llu,\"q8\":%llu},"
		"\"baseline_bytes\":%llu,\"membrane_bytes\":%llu,"
		"\"absolute_saved_bytes\":%lld,\"storage_reduction_ratio\":%.6f,"
		"\"validation\":{\"blocks_decoded\":%llu,\"decode_failures\":%llu,"
		"\"encode_nondeterminism\":%llu,\"q4_mean_rel_l2_error\":%.6f,"
		"\"q4_max_rel_l2_error\":%.6f,\"q8_mean_rel_l2_error\":%.6f,"
		"\"q8_max_rel_l2_error\":%.6f,\"pass\":%s},"
		"\"backend\":{\"simd\":\"%s\"},"
		"\"elapsed_seconds\":%.6f}\n",
		MEMBRANE_DEMO_SCHEMA_VERSION,
		r->config.blocks, r->config.seed, r->elems_per_block,
		r->config.blocks, r->config.seed,
		(unsigned long long)r->q4_blocks, (unsigned long long)r->q8_blocks,
		(unsigned long long)r->baseline_bytes,
		(unsigned long long)r->membrane_bytes,
		(long long)r->saved_bytes, r->reduction_ratio,
		(unsigned long long)r->blocks_decoded,
		(unsigned long long)r->decode_failures,
		(unsigned long long)r->encode_nondeterminism,
		r->q4_mean_rel_l2_error, r->q4_max_rel_l2_error,
		r->q8_mean_rel_l2_error, r->q8_max_rel_l2_error,
		r->validation_pass ? "true" : "false",
		r->simd_backend_name, r->elapsed_seconds);
}
