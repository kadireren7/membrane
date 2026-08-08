#ifndef MEMBRANE_BENCH_PRECISION_POLICY_H
# define MEMBRANE_BENCH_PRECISION_POLICY_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"
# include "membrane/quant_simd.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Shared per-block precision policy + encode/decode/account/validate,
 * used by both membrane-demo and membrane-bench. All three policies use
 * the same maintained bit-exact Q4_0/Q8_0 engine (membrane/quant_simd.h)
 * -- Q4_ONLY/Q8_ONLY simply force the precision without a trial;
 * ADAPTIVE reuses membrane/quant_select.h's real content-driven
 * selection unchanged.
 */
typedef enum e_membrane_bench_policy
{
	MEMBRANE_BENCH_POLICY_Q4_ONLY = 0,
	MEMBRANE_BENCH_POLICY_Q8_ONLY,
	MEMBRANE_BENCH_POLICY_ADAPTIVE,
	MEMBRANE_BENCH_POLICY_COUNT
}	membrane_bench_policy_t;

const char	*membrane_bench_policy_name(membrane_bench_policy_t p);
int			membrane_bench_policy_from_name(const char *name,
				membrane_bench_policy_t *out);

typedef struct s_membrane_err_stats
{
	double		sum;
	double		max;
	uint64_t	count;
}	membrane_err_stats_t;

double	membrane_err_stats_mean(const membrane_err_stats_t *s);

/*
 * Running totals for one workload x policy run. blocks_decoded +
 * decode_failures == the number of blocks processed so far.
 * decode_failures is an execution-integrity signal (the maintained
 * decoder itself reported failure), never a policy-quality judgement:
 * q4_err/q8_err (the real relative-L2 reconstruction error for each
 * precision actually used) are the policy-quality signal, reported
 * separately and never converted into a pass/fail on their own -- see
 * membrane_workload_validation_pass below.
 */
typedef struct s_membrane_workload_accum
{
	uint64_t				q4_blocks;
	uint64_t				q8_blocks;
	uint64_t				encoded_bytes;
	uint64_t				blocks_decoded;
	uint64_t				decode_failures;
	uint64_t				encode_nondeterminism;
	membrane_err_stats_t	q4_err;
	membrane_err_stats_t	q8_err;
}	membrane_workload_accum_t;

/*
 * Processes exactly one block under `policy`, updating *acc, and is the
 * single timed region tools/membrane-bench measures (selection +
 * encode + decode + validate; NOT block generation or output
 * formatting). Q4_ONLY/Q8_ONLY force the precision directly (no trial
 * call); ADAPTIVE calls membrane_quant_select_precision with
 * `adaptive_max_q4_rel_l2_error` unchanged from membrane/quant_select.h's
 * documented contract.
 *
 * Returns MEMBRANE_OK, or an infrastructure failure (allocation, or the
 * quant_simd engine itself reporting non-OK on encode/select) that the
 * caller should treat as fatal for the whole run. A per-block DECODE
 * failure is intentionally NOT one of these -- it is soft-reported via
 * acc->decode_failures, matching membrane-demo's existing documented
 * contract (see tools/membrane-demo/demo_core.h).
 */
membrane_status_t	membrane_bench_process_block(
						membrane_simd_backend_t backend,
						membrane_bench_policy_t policy,
						const uint16_t *x_f16, uint32_t elems,
						double adaptive_max_q4_rel_l2_error,
						membrane_workload_accum_t *acc);

/*
 * Execution-integrity validation, as distinct from policy quality
 * (rel-L2 error, reported separately): true iff every block presented
 * was decoded, none was corrupted/missing, and encoding was
 * reproducible. A q4-only run whose Q4 error exceeds the adaptive
 * threshold still validation_passes -- that threshold is an ADAPTIVE
 * policy parameter, not an execution-integrity requirement.
 */
int	membrane_workload_validation_pass(const membrane_workload_accum_t *acc,
		uint64_t blocks_expected);

# ifdef __cplusplus
}
# endif

#endif
