#ifndef MEMBRANE_DEMO_CORE_H
# define MEMBRANE_DEMO_CORE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMBRANE Product Phase 1 demo: a self-contained, no-network,
 * no-model-download walk through the maintained CPU quantization path
 * -- generate deterministic synthetic KV-like F16 blocks, pick Q4_0 vs
 * Q8_0 per block via membrane/quant_select.h, encode with the
 * maintained bit-exact engine (membrane/quant_simd.h), account real
 * storage bytes against an explicit FP32 baseline, decode, and verify.
 *
 * This module holds the reusable logic; tools/membrane-demo/main.c is
 * a thin CLI wrapper, and tests/unit call these functions directly
 * (see tools/membrane-kv-workingset-sim's membrane_wssim_core for the
 * same split used elsewhere in this project).
 */

# define MEMBRANE_DEMO_ELEMS_PER_BLOCK	128u	/* multiple of 32 */
# define MEMBRANE_DEMO_DEFAULT_BLOCKS	8192u
# define MEMBRANE_DEMO_DEFAULT_SEED	1234u
# define MEMBRANE_DEMO_MAX_BLOCKS	1000000u
# define MEMBRANE_DEMO_SCHEMA_VERSION	1

typedef struct s_membrane_demo_config
{
	uint32_t	blocks;
	uint32_t	seed;
}	membrane_demo_config_t;

typedef struct s_membrane_demo_result
{
	membrane_demo_config_t	config;
	uint32_t				elems_per_block;
	const char				*simd_backend_name;
	uint64_t				q4_blocks;
	uint64_t				q8_blocks;
	uint64_t				baseline_bytes;
	uint64_t				membrane_bytes;
	int64_t					saved_bytes;
	double					reduction_ratio;
	uint64_t				blocks_decoded;
	uint64_t				decode_failures;
	uint64_t				encode_nondeterminism;
	double					q4_mean_rel_l2_error;
	double					q4_max_rel_l2_error;
	double					q8_mean_rel_l2_error;
	double					q8_max_rel_l2_error;
	double					elapsed_seconds;
	int						validation_pass;
}	membrane_demo_result_t;

/* Exit code the CLI should use for a parse/usage failure. */
# define MEMBRANE_DEMO_EXIT_USAGE_ERROR	2

/*
 * Parses argv[1..argc) into *cfg. Recognizes --blocks N, --seed N,
 * --json, --help; rejects unknown options, non-numeric or negative
 * --blocks/--seed, --blocks 0, and --blocks above
 * MEMBRANE_DEMO_MAX_BLOCKS. On success returns 0 and fills *cfg
 * (defaults applied for any flag not given); *want_help is set to 1 if
 * --help was seen (cfg is not meaningful in that case). On failure
 * returns MEMBRANE_DEMO_EXIT_USAGE_ERROR and writes a one-line,
 * NUL-terminated reason to err_buf (err_cap bytes).
 */
int	membrane_demo_parse_args(int argc, char **argv,
		membrane_demo_config_t *cfg, int *want_json, int *want_help,
		char *err_buf, size_t err_cap);

/*
 * Runs the full deterministic pipeline and fills *out. Returns
 * MEMBRANE_OK on success -- including when some blocks fail to decode,
 * which is reported via out->decode_failures/out->validation_pass
 * rather than as a return-code failure. Returns MEMBRANE_ERR_INVALID_ARG
 * for a NULL cfg/out, cfg->blocks == 0, or cfg->blocks over
 * MEMBRANE_DEMO_MAX_BLOCKS; MEMBRANE_ERR_ALLOC_FAILED if a working
 * buffer cannot be allocated.
 */
membrane_status_t	membrane_demo_run(const membrane_demo_config_t *cfg,
							membrane_demo_result_t *out);

void	membrane_demo_print_human(const membrane_demo_result_t *r, FILE *f);
void	membrane_demo_print_json(const membrane_demo_result_t *r, FILE *f);

# ifdef __cplusplus
}
# endif

#endif
