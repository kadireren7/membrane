#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>

#include "demo_core.h"
#include "test_helpers.h"

static void	test_default_config_validates(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;

	cfg.blocks = 512;
	cfg.seed = MEMBRANE_DEMO_DEFAULT_SEED;
	TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_OK, "run succeeds");
	TEST_ASSERT(r.validation_pass, "default workload validates");
	TEST_ASSERT(r.blocks_decoded == cfg.blocks, "every block decoded");
	TEST_ASSERT(r.decode_failures == 0, "no decode failures");
	TEST_ASSERT(r.encode_nondeterminism == 0, "encoding is reproducible");
	TEST_ASSERT(r.q4_blocks + r.q8_blocks == cfg.blocks,
		"every block was assigned exactly one precision");
}

static void	test_deterministic_for_same_config(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r1;
	membrane_demo_result_t	r2;

	cfg.blocks = 300;
	cfg.seed = 99;
	TEST_ASSERT(membrane_demo_run(&cfg, &r1) == MEMBRANE_OK, "first run");
	TEST_ASSERT(membrane_demo_run(&cfg, &r2) == MEMBRANE_OK, "second run");
	TEST_ASSERT(r1.q4_blocks == r2.q4_blocks, "q4 count reproducible");
	TEST_ASSERT(r1.q8_blocks == r2.q8_blocks, "q8 count reproducible");
	TEST_ASSERT(r1.membrane_bytes == r2.membrane_bytes,
		"storage bytes reproducible");
	TEST_ASSERT(r1.baseline_bytes == r2.baseline_bytes,
		"baseline bytes reproducible");
	TEST_ASSERT(r1.q4_mean_rel_l2_error == r2.q4_mean_rel_l2_error,
		"error stats bit-identical across runs");
}

static void	test_different_seeds_still_validate(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;
	uint32_t				seeds[3];
	size_t					i;

	seeds[0] = 0;
	seeds[1] = 1;
	seeds[2] = 4242;
	i = 0;
	while (i < 3)
	{
		cfg.blocks = 256;
		cfg.seed = seeds[i];
		TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_OK,
			"run succeeds for this seed");
		TEST_ASSERT(r.validation_pass, "this seed's workload validates");
		i++;
	}
}

static void	test_storage_accounting_is_real(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;
	uint64_t				expect_baseline;

	cfg.blocks = 100;
	cfg.seed = 5;
	TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_OK, "run succeeds");
	expect_baseline = (uint64_t)cfg.blocks * MEMBRANE_DEMO_ELEMS_PER_BLOCK
		* sizeof(float);
	TEST_ASSERT(r.baseline_bytes == expect_baseline,
		"baseline is exactly blocks * elems_per_block * sizeof(float)");
	TEST_ASSERT(r.membrane_bytes < r.baseline_bytes,
		"quantized storage is smaller than the FP32 baseline");
	TEST_ASSERT(r.saved_bytes == (int64_t)r.baseline_bytes
			- (int64_t)r.membrane_bytes,
		"saved_bytes is exactly baseline minus membrane bytes");
}

static void	test_zero_blocks_rejected(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;

	cfg.blocks = 0;
	cfg.seed = 1;
	TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"zero blocks rejected by the core run function");
}

static void	test_oversized_blocks_rejected_safely(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;

	cfg.blocks = MEMBRANE_DEMO_MAX_BLOCKS + 1;
	cfg.seed = 1;
	TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"over-the-limit block count rejected, not attempted");
}

static void	test_arg_parsing(void)
{
	membrane_demo_config_t	cfg;
	int						want_json;
	int						want_help;
	char					err[160];
	char					*argv_ok[] = {"membrane-demo", "--blocks", "64",
								"--seed", "9", "--json"};
	char					*argv_zero[] = {"membrane-demo", "--blocks", "0"};
	char					*argv_bad[] = {"membrane-demo", "--blocks", "xx"};
	char					*argv_huge[] = {"membrane-demo", "--blocks",
								"999999999999"};
	char					*argv_unknown[] = {"membrane-demo", "--bogus"};
	char					*argv_help[] = {"membrane-demo", "--help"};

	TEST_ASSERT(membrane_demo_parse_args(6, argv_ok, &cfg, &want_json,
			&want_help, err, sizeof(err)) == 0, "valid args parse");
	TEST_ASSERT(cfg.blocks == 64 && cfg.seed == 9 && want_json == 1
		&& want_help == 0, "parsed values match the given flags");
	TEST_ASSERT(membrane_demo_parse_args(3, argv_zero, &cfg, &want_json,
			&want_help, err, sizeof(err)) != 0, "--blocks 0 rejected");
	TEST_ASSERT(membrane_demo_parse_args(3, argv_bad, &cfg, &want_json,
			&want_help, err, sizeof(err)) != 0,
		"non-numeric --blocks rejected");
	TEST_ASSERT(membrane_demo_parse_args(3, argv_huge, &cfg, &want_json,
			&want_help, err, sizeof(err)) != 0,
		"--blocks above uint32 range rejected");
	TEST_ASSERT(membrane_demo_parse_args(2, argv_unknown, &cfg, &want_json,
			&want_help, err, sizeof(err)) != 0, "unknown option rejected");
	TEST_ASSERT(err[0] != '\0', "an error message is always produced");
	want_help = 0;
	TEST_ASSERT(membrane_demo_parse_args(2, argv_help, &cfg, &want_json,
			&want_help, err, sizeof(err)) == 0, "--help itself is not an error");
	TEST_ASSERT(want_help == 1, "--help sets want_help");
}

static int	json_has_key(const char *json, const char *key)
{
	return (strstr(json, key) != NULL);
}

static void	test_json_output_has_required_fields(void)
{
	membrane_demo_config_t	cfg;
	membrane_demo_result_t	r;
	char					buf[4096];
	FILE					*f;
	size_t					n;

	cfg.blocks = 64;
	cfg.seed = 3;
	TEST_ASSERT(membrane_demo_run(&cfg, &r) == MEMBRANE_OK, "run succeeds");
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen for JSON capture succeeds");
	membrane_demo_print_json(&r, f);
	fclose(f);
	TEST_ASSERT(json_has_key(buf, "\"schema_version\""), "has schema_version");
	TEST_ASSERT(json_has_key(buf, "\"blocks_processed\""),
		"has blocks_processed");
	TEST_ASSERT(json_has_key(buf, "\"baseline_bytes\""), "has baseline_bytes");
	TEST_ASSERT(json_has_key(buf, "\"membrane_bytes\""), "has membrane_bytes");
	TEST_ASSERT(json_has_key(buf, "\"storage_reduction_ratio\""),
		"has storage_reduction_ratio");
	TEST_ASSERT(json_has_key(buf, "\"precision_counts\""),
		"has precision_counts");
	TEST_ASSERT(json_has_key(buf, "\"validation\""), "has validation");
	TEST_ASSERT(json_has_key(buf, "\"seed\""), "has seed");
	TEST_ASSERT(json_has_key(buf, "\"configuration\""), "has configuration");
	/* Minimal syntax sanity check: balanced braces, starts/ends right. */
	n = strlen(buf);
	TEST_ASSERT(n > 2 && buf[0] == '{', "JSON starts with '{'");
	TEST_ASSERT(buf[n - 2] == '}' || buf[n - 1] == '}',
		"JSON ends with '}' (trailing newline allowed)");
}

int	main(void)
{
	test_default_config_validates();
	test_deterministic_for_same_config();
	test_different_seeds_still_validate();
	test_storage_accounting_is_real();
	test_zero_blocks_rejected();
	test_oversized_blocks_rejected_safely();
	test_arg_parsing();
	test_json_output_has_required_fields();
	return (0);
}
