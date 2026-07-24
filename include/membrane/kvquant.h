#ifndef MEMBRANE_KVQUANT_H
# define MEMBRANE_KVQUANT_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"
# include "membrane/q8block.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Numerical-error metrics for one F16 tensor payload under one Q8
 * quantization config (Phase 3.1). Unlike every earlier phase, the
 * transform is lossy, so there is no "integrity PASS/FAIL" -- the
 * questions are how big the error is and whether it is safe, which is
 * what these fields answer. Deterministic for a given input and cfg.
 *
 * MSE/RMSE/MAE/max_abs_err/cosine_similarity/rel_l2_error are computed
 * only over elements whose ORIGINAL value was finite (NaN/Inf inputs have
 * no well-defined "error"; they are counted separately in nan_input /
 * inf_input, sourced from the encoder's own stats).
 */
typedef struct s_membrane_kv_quant_metrics
{
	uint64_t	raw_bytes;
	uint64_t	encoded_bytes;
	uint64_t	elements;
	uint64_t	groups;
	uint64_t	saturated;
	uint64_t	nan_input;
	uint64_t	inf_input;
	uint64_t	degenerate_groups;
	uint64_t	metadata_bytes;		/* encoded_bytes beyond 1 byte/element */
	double		mse;
	double		rmse;
	double		mae;
	double		max_abs_err;
	double		cosine_similarity;
	double		rel_l2_error;		/* ||decoded-original||_2 / ||original||_2 */
	int			decode_ok;
}	membrane_kv_quant_metrics_t;

/*
 * Encodes `payload` (len F16 bytes) under `cfg`, decodes it back, and
 * fills `out` with the size and numerical-error metrics above. Returns
 * MEMBRANE_ERR_INVALID_ARG for an odd length or an invalid cfg (mirrors
 * membrane_q8_encode); a decode failure is reported via decode_ok rather
 * than propagated, since a corrupted stream is not this function's own
 * error to raise -- callers that need to distinguish "OK" from "the codec
 * itself is broken" should check decode_ok.
 */
membrane_status_t	membrane_kv_quant_compute(const uint8_t *payload,
						size_t len, const membrane_q8_cfg_t *cfg,
						membrane_kv_quant_metrics_t *out);

# ifdef __cplusplus
}
# endif

#endif
