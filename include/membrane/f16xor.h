#ifndef MEMBRANE_F16XOR_H
# define MEMBRANE_F16XOR_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Lossless predictive transform for F16 tensor data (Phase 2.3).
 *
 * Each F16 value is treated as its uint16 bit pattern. A predictor
 * replaces every value with its XOR against an earlier value chosen by a
 * fixed element stride; the first `stride` values are kept raw as seeds.
 * The transform is exactly invertible (XOR is its own inverse), so it is
 * lossless regardless of the data. Element stride comes from the predictor
 * mode and the tensor's row width:
 *
 *   NONE               stride 0  (identity, the baseline)
 *   XOR_PREVIOUS_ELEMENT   stride 1  (neighbour in the linear stream)
 *   XOR_PREVIOUS_TOKEN     stride row_elems (same slot, previous token)
 *   XOR_PREVIOUS_ROW       stride row_elems (same slot, previous row)
 *
 * For a row-major KV tensor (dims = [row_bytes, n_tokens]) a row is one
 * token, so TOKEN and ROW resolve to the same stride; they are kept
 * distinct for layouts (e.g. transposed V) where they would differ.
 */
typedef enum e_membrane_predictor
{
	MEMBRANE_PRED_NONE = 0,
	MEMBRANE_PRED_XOR_PREV_ELEMENT,
	MEMBRANE_PRED_XOR_PREV_TOKEN,
	MEMBRANE_PRED_XOR_PREV_ROW,
	MEMBRANE_PRED_COUNT
}	membrane_predictor_t;

typedef struct s_membrane_f16xor_cfg
{
	membrane_predictor_t	predictor;
	size_t					row_elems;	/* elements per token row */
}	membrane_f16xor_cfg_t;

/* Short lowercase name for a predictor, "invalid" if out of range. */
const char	*membrane_predictor_name(membrane_predictor_t p);

/*
 * Byte-level XOR residual with an element stride (stride_elems uint16
 * elements = 2*stride_elems bytes). `out` must not alias `in` and holds
 * `len` bytes; stride_elems 0 copies. Undoing it (`inverse`, in place)
 * reproduces the original bytes exactly.
 */
void		membrane_f16xor_forward(const uint8_t *in, size_t len,
				size_t stride_elems, uint8_t *out);
void		membrane_f16xor_inverse(uint8_t *buf, size_t len,
				size_t stride_elems);

/*
 * Resolves cfg to an element stride, applies the residual transform, then
 * splits the residual into low/high byte planes stored RAW behind an
 * 18-byte versioned header. in_len must be even. Returns
 * MEMBRANE_ERR_INVALID_ARG for an odd length or a TOKEN/ROW predictor with
 * row_elems == 0 (shape unknown), MEMBRANE_ERR_BUFFER_TOO_SMALL if out_cap
 * cannot hold the header.
 */
membrane_status_t	membrane_f16xor_encode(const membrane_f16xor_cfg_t *cfg,
						const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len);

/*
 * Inverse of membrane_f16xor_encode: self-describing (reads predictor and
 * stride from the header), so it does not need cfg. Rejects malformed,
 * truncated, or inconsistent streams as MEMBRANE_ERR_CORRUPT_DATA.
 */
membrane_status_t	membrane_f16xor_decode(const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len);

# ifdef __cplusplus
}
# endif

#endif
