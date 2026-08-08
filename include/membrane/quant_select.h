#ifndef MEMBRANE_QUANT_SELECT_H
# define MEMBRANE_QUANT_SELECT_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"
# include "membrane/policy.h"
# include "membrane/quant_simd.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Product Phase 1 adapter: the maintained project has no runtime API
 * that picks Q4_0 vs Q8_0 for one block from its actual content --
 * membrane/policy.h's membrane_policy_t is a per-layer file format
 * produced by the (llama-gated) offline optimizer, not a content-driven
 * per-block decision. This is the smallest addition that drives a real
 * one: it quantizes+dequantizes the block under Q4_0 with the existing
 * bit-exact engine (membrane/quant_simd.h) and measures the actual
 * relative-L2 reconstruction error. Q4_0 is selected only if that error
 * clears cfg->max_q4_rel_l2_error; otherwise Q8_0. No ratio is assumed
 * or hardcoded -- the split falls out of real quantization behaviour on
 * whatever data is passed in.
 */

/*
 * 0.05 matches the "high fidelity" relative-L2 acceptance bound already
 * established for lossy KV reconstruction in
 * tests/unit/test_kvquant.c's test_smooth_signal_high_fidelity (Q8
 * group quantizer) -- reused here rather than inventing a new number.
 */
# define MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR	0.05

typedef struct s_membrane_quant_select_cfg
{
	double	max_q4_rel_l2_error;
}	membrane_quant_select_cfg_t;

typedef struct s_membrane_quant_select_result
{
	membrane_precision_t	precision;	/* MEMBRANE_PRECISION_Q4 or _Q8 */
	double					q4_rel_l2_error;
}	membrane_quant_select_result_t;

/*
 * Relative-L2 reconstruction error between two same-length F16 arrays:
 * ||b - a||_2 / ||a||_2 (0.0 if a is all-zero and b matches, 1.0 if a is
 * all-zero and b does not). Non-finite elements of `a` are excluded, the
 * same convention membrane/kvquant.h's compute_error uses.
 */
double	membrane_quant_rel_l2_error(const uint16_t *a_f16,
			const uint16_t *b_f16, size_t n);

/*
 * `n` must be a positive multiple of MEMBRANE_QSIMD_BLOCK_ELEMS (32).
 * Returns MEMBRANE_ERR_INVALID_ARG for a NULL x_f16/cfg/out, n == 0, or
 * n not a multiple of 32; MEMBRANE_ERR_ALLOC_FAILED if the Q4_0 trial
 * buffer cannot be allocated; MEMBRANE_ERR_UNIMPLEMENTED if `backend`
 * isn't supported by this CPU.
 */
membrane_status_t	membrane_quant_select_precision(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n,
						const membrane_quant_select_cfg_t *cfg,
						membrane_quant_select_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
