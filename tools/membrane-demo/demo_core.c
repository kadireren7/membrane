#define _POSIX_C_SOURCE 200809L

#include <string.h>
#include <time.h>

#include "membrane/quant_select.h"
#include "precision_policy.h"
#include "workload_gen.h"
#include "demo_core.h"

/*
 * Product Phase 2 note: the generation, precision-policy, and
 * encode/decode/account/validate logic that used to live entirely in
 * this file now lives in tools/membrane-workload-core (shared with
 * tools/membrane-quant-policy-bench). This file is now a thin adapter:
 * MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT + MEMBRANE_BENCH_POLICY_ADAPTIVE
 * with membrane-demo's original fixed elems-per-block/threshold,
 * mapped onto the same public membrane_demo_result_t this module has
 * always returned. The default demo's output is unchanged by this
 * refactor (verified byte-for-byte against the pre-refactor binary).
 */

static membrane_status_t	run_workload(const membrane_demo_config_t *cfg,
					membrane_simd_backend_t backend,
					membrane_workload_accum_t *acc)
{
	uint16_t			x_f16[MEMBRANE_DEMO_ELEMS_PER_BLOCK];
	membrane_status_t	st;
	uint32_t			i;

	i = 0;
	while (i < cfg->blocks)
	{
		membrane_workload_generate_block(MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT,
			cfg->seed, i, x_f16, MEMBRANE_DEMO_ELEMS_PER_BLOCK);
		st = membrane_bench_process_block(backend,
				MEMBRANE_BENCH_POLICY_ADAPTIVE, x_f16,
				MEMBRANE_DEMO_ELEMS_PER_BLOCK,
				MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR, acc);
		if (st != MEMBRANE_OK)
			return (st);
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_demo_run(const membrane_demo_config_t *cfg,
							membrane_demo_result_t *out)
{
	membrane_workload_accum_t	acc;
	membrane_simd_backend_t	backend;
	uint64_t					total_elems;
	struct timespec				t0;
	struct timespec				t1;
	membrane_status_t			st;

	if (cfg == NULL || out == NULL || cfg->blocks == 0
			|| cfg->blocks > MEMBRANE_DEMO_MAX_BLOCKS)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!membrane_checked_mul_u64(cfg->blocks, MEMBRANE_DEMO_ELEMS_PER_BLOCK,
			&total_elems))
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(&acc, 0, sizeof(acc));
	backend = membrane_simd_best_backend();
	clock_gettime(CLOCK_MONOTONIC, &t0);
	st = run_workload(cfg, backend, &acc);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	if (st != MEMBRANE_OK)
		return (st);
	memset(out, 0, sizeof(*out));
	out->config = *cfg;
	out->elems_per_block = MEMBRANE_DEMO_ELEMS_PER_BLOCK;
	out->simd_backend_name = membrane_simd_backend_name(backend);
	out->q4_blocks = acc.q4_blocks;
	out->q8_blocks = acc.q8_blocks;
	membrane_checked_mul_u64(total_elems, sizeof(float), &out->baseline_bytes);
	out->membrane_bytes = acc.encoded_bytes;
	out->saved_bytes = (int64_t)out->baseline_bytes
		- (int64_t)out->membrane_bytes;
	if (out->baseline_bytes > 0)
		out->reduction_ratio = (double)out->saved_bytes
			/ (double)out->baseline_bytes;
	out->blocks_decoded = acc.blocks_decoded;
	out->decode_failures = acc.decode_failures;
	out->encode_nondeterminism = acc.encode_nondeterminism;
	out->q4_mean_rel_l2_error = membrane_err_stats_mean(&acc.q4_err);
	out->q4_max_rel_l2_error = acc.q4_err.max;
	out->q8_mean_rel_l2_error = membrane_err_stats_mean(&acc.q8_err);
	out->q8_max_rel_l2_error = acc.q8_err.max;
	out->elapsed_seconds = (double)(t1.tv_sec - t0.tv_sec)
		+ (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
	out->validation_pass = membrane_workload_validation_pass(&acc,
			cfg->blocks);
	return (MEMBRANE_OK);
}
