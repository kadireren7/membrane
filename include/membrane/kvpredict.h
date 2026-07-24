#ifndef MEMBRANE_KVPREDICT_H
# define MEMBRANE_KVPREDICT_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"
# include "membrane/f16xor.h"

# ifdef __cplusplus
extern "C" {
# endif

# define MEMBRANE_PRED_QUARTILES 4

/*
 * Residual compressibility metrics for one F16 tensor payload under one
 * predictor (Phase 2.3). Everything is measured on the lossless residual,
 * never on a lossy approximation. Deterministic for a given input.
 *
 * ideal_bytes is an information-theoretic estimate only: the byte count a
 * perfect order-0 entropy coder would need for the two residual byte
 * planes (low_bytes*H_low/8 + high_bytes*H_high/8), rounded up. It is a
 * ceiling, not an achieved size -- no entropy coder is run.
 */
typedef struct s_membrane_residual_metrics
{
	uint64_t	raw_bytes;
	uint64_t	total_u16;
	uint64_t	zero_u16;			/* residual uint16 values equal to 0 */
	uint64_t	zero_bytes;			/* residual bytes equal to 0 */
	uint64_t	longest_zero_run;	/* longest run of 0x00 residual bytes */
	uint64_t	ideal_bytes;
	uint64_t	stride_elems;		/* element stride the predictor resolved to */
	double		entropy;			/* residual, bits/byte */
	double		low_entropy;
	double		high_entropy;
	double		quartile_entropy[MEMBRANE_PRED_QUARTILES];
	int			applicable;			/* 0 if predictor needed a row width it lacked */
}	membrane_residual_metrics_t;

typedef struct s_membrane_residual_cfg
{
	membrane_predictor_t	predictor;
	size_t					row_elems;	/* elements per token row (TOKEN/ROW) */
	size_t					n_rows;		/* token count; 0 skips quartile split */
}	membrane_residual_cfg_t;

/*
 * Applies cfg's predictor to buf (len bytes, must be even) and fills out
 * with the residual metrics. A TOKEN/ROW predictor with row_elems == 0
 * yields applicable = 0 and metrics equal to the NONE baseline instead of
 * failing. Returns MEMBRANE_ERR_INVALID_ARG for a NULL/odd input,
 * MEMBRANE_ERR_ALLOC_FAILED if the residual buffer cannot be allocated.
 */
membrane_status_t	membrane_kv_residual_metrics(const uint8_t *buf,
						size_t len, const membrane_residual_cfg_t *cfg,
						membrane_residual_metrics_t *out);

# ifdef __cplusplus
}
# endif

#endif
