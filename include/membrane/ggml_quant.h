#ifndef MEMBRANE_GGML_QUANT_H
# define MEMBRANE_GGML_QUANT_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 4.4: bit-exact adapters around the pinned llama.cpp/ggml
 * commit's OWN Q8_0/Q4_0 quantize and dequantize implementations. No
 * math is reimplemented here -- every function below calls straight
 * into ggml's real, linked symbols, so every byte produced is exactly
 * what ggml itself computed for the identical input. See
 * docs/phase4-ggml-quant-parity.md for exactly which ggml symbols each
 * of these calls, why those specific ones (not e.g. the GGUF-file
 * "_ref" reference forms) were chosen, and the commit hash they were
 * read from.
 *
 * Only built when MEMBRANE_ENABLE_LLAMA is ON (this module links
 * directly against ggml, unlike the rest of membrane_core, which stays
 * dependency-free for the portable Release/ASan test suites).
 */

# define MEMBRANE_GGML_QBLOCK_ELEMS		32u
# define MEMBRANE_GGML_Q8_0_BLOCK_BYTES		34u
# define MEMBRANE_GGML_Q4_0_BLOCK_BYTES		18u

/*
 * Quantizes `n` F16 values (`n` must be a multiple of
 * MEMBRANE_GGML_QBLOCK_ELEMS) into packed ggml Q8_0/Q4_0 blocks,
 * written to `out` (must be at least
 * (n/32)*MEMBRANE_GGML_Q{8,4}_0_BLOCK_BYTES bytes). Returns
 * MEMBRANE_ERR_INVALID_ARG if n is not a multiple of the block size or
 * any pointer is NULL.
 */
membrane_status_t	membrane_ggml_q8_0_quantize(const uint16_t *x_f16,
						size_t n, uint8_t *out);
membrane_status_t	membrane_ggml_q4_0_quantize(const uint16_t *x_f16,
						size_t n, uint8_t *out);

/*
 * Dequantizes `n` elements worth of packed ggml Q8_0/Q4_0 blocks (`in`
 * must hold (n/32)*MEMBRANE_GGML_Q{8,4}_0_BLOCK_BYTES bytes) back to
 * F16, written to `out_f16` (n elements).
 */
membrane_status_t	membrane_ggml_q8_0_dequantize(const uint8_t *in,
						size_t n, uint16_t *out_f16);
membrane_status_t	membrane_ggml_q4_0_dequantize(const uint8_t *in,
						size_t n, uint16_t *out_f16);

/*
 * Round-trips `n` F16 values in place through the given storage bit
 * width (4 or 8; 16 is a no-op) using the functions above -- the
 * ggml-exact replacement for the retired quant_roundtrip_inplace
 * (Phase 3.3-4.3's own linear per-32-element quantizer, which did not
 * match ggml's block format or rounding; see
 * docs/phase4-ggml-quant-parity.md).
 */
membrane_status_t	membrane_ggml_quant_roundtrip(uint16_t *x_f16, size_t n,
						int bits);

# ifdef __cplusplus
}
# endif

#endif
