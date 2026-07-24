#ifndef MEMBRANE_Q8BLOCK_H
# define MEMBRANE_Q8BLOCK_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Block-wise F16 -> int8 quantization (Phase 3.1), the first genuinely
 * lossy transform in MEMBRANE. It is deliberately NOT registered in
 * membrane_codec_t / the block-layer codec table: that table's contract
 * (see membrane/block.h and src/block/block.c) verifies every block
 * round-trips to the ORIGINAL bytes via a checksum computed before
 * compression, which only lossless codecs can satisfy. Q8 quantization is
 * exposed here as a standalone module with its own corruption-detection
 * checksum (over the quantized bytes, not the original ones) instead, and
 * is measured directly by the analyzer and membrane_kv_quant_compute()
 * (membrane/kvquant.h), not through membrane_block_write/read.
 *
 * Layout: F16 elements are grouped into fixed-size groups of
 * `group_elems` elements; each group gets its own float32 scale
 * (symmetric) or scale+bias (affine, bias = the group's minimum value).
 * A short versioned header carries the group size, element count, and a
 * CRC32 over everything after the header (metadata + int8 payload) so
 * bitstream corruption is caught before any dequantization happens.
 *
 * NaN/Inf policy (the transform must never crash or emit undefined
 * bytes on non-finite input): scale/bias are computed from the FINITE
 * elements in a group only. A NaN element quantizes to the code that
 * decodes to 0 (affine: the group minimum, via the -128 code) and never
 * takes part in choosing the scale. +Inf / -Inf saturate to the group's
 * extreme code (so they decode to the group's finite max/min, not to
 * literal infinity -- a fixed-width code cannot represent an unbounded
 * value). A group with no finite elements at all falls back to scale 0
 * (symmetric) or scale 0 / bias 0 (affine), which decodes every element
 * in that group to exactly 0.0 -- a safe, well-defined default rather
 * than a NaN propagating further. All of this is exercised in
 * tests/unit/test_q8block.c.
 */

typedef enum e_membrane_q8_mode
{
	MEMBRANE_Q8_SYMMETRIC = 0,
	MEMBRANE_Q8_AFFINE = 1
}	membrane_q8_mode_t;

typedef struct s_membrane_q8_cfg
{
	membrane_q8_mode_t	mode;
	size_t				group_elems;	/* elements per quantization group */
}	membrane_q8_cfg_t;

/* Populated by membrane_q8_encode (all fields optional to inspect). */
typedef struct s_membrane_q8_stats
{
	uint64_t	elements;
	uint64_t	groups;
	uint64_t	saturated;		/* quantized codes that hit an extreme value */
	uint64_t	nan_input;
	uint64_t	inf_input;
	uint64_t	degenerate_groups;	/* groups with zero finite elements */
}	membrane_q8_stats_t;

# define MEMBRANE_Q8_HEADER 20

/* Exact encoded size for `in_len` F16 bytes under `cfg`. group_elems must
 * be > 0; mode must be a valid membrane_q8_mode_t. Returns SIZE_MAX on
 * overflow or an invalid cfg. */
size_t				membrane_q8_bound(size_t in_len,
						const membrane_q8_cfg_t *cfg);

/*
 * Quantizes `in` (in_len bytes of F16 elements, must be even) per cfg.
 * `stats` may be NULL. Returns MEMBRANE_ERR_INVALID_ARG for an odd
 * length, a zero group size, or an unrecognised mode;
 * MEMBRANE_ERR_BUFFER_TOO_SMALL if out_cap is under membrane_q8_bound().
 */
membrane_status_t	membrane_q8_encode(const membrane_q8_cfg_t *cfg,
						const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len,
						membrane_q8_stats_t *stats);

/*
 * Dequantizes a stream produced by membrane_q8_encode back into F16 bytes
 * (out_len == the original in_len). Self-describing: mode and group size
 * come from the header. Validates the header, the payload CRC32, and
 * every length field before touching the payload; malformed, truncated,
 * or tampered streams return MEMBRANE_ERR_CORRUPT_DATA.
 */
membrane_status_t	membrane_q8_decode(const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len);

# ifdef __cplusplus
}
# endif

#endif
