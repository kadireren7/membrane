#ifndef MEMBRANE_HUFFMAN_H
# define MEMBRANE_HUFFMAN_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Length-limited canonical Huffman coder over 8-bit symbols (Phase 2.4).
 *
 * A self-contained stream is produced: a uint32 symbol count, then -- when
 * the count is non-zero -- a 128-byte table of 256 nibble-packed code
 * lengths (0 = unused, 1..15 otherwise), then the MSB-first bitstream.
 * Code lengths are capped at 15 bits (zlib-style Kraft repair), so the
 * table is fixed-size and the decoder needs no large lookup tables. The
 * coder is order-0: it models only the symbol distribution, which is the
 * point of the experiment -- it is meant to cash in the entropy the F16
 * high byte plane has (~5.6 bits/byte) and nothing more.
 *
 * membrane_huffman_compress never errors on incompressible data; it may
 * simply produce output larger than the input (the caller decides whether
 * to keep it). Decompress rejects malformed tables (over-subscribed code
 * lengths) and truncated bitstreams as MEMBRANE_ERR_CORRUPT_DATA.
 */

# define MEMBRANE_HUFFMAN_MAXLEN 15
# define MEMBRANE_HUFFMAN_TABLE_BYTES 128
# define MEMBRANE_HUFFMAN_HEADER (4 + MEMBRANE_HUFFMAN_TABLE_BYTES)

/* Worst-case compressed size for `len` input bytes. */
size_t				membrane_huffman_bound(size_t len);

membrane_status_t	membrane_huffman_compress(const uint8_t *src, size_t len,
						uint8_t *out, size_t out_cap, size_t *out_len);

membrane_status_t	membrane_huffman_decompress(const uint8_t *src, size_t len,
						uint8_t *out, size_t out_cap, size_t *out_len);

# ifdef __cplusplus
}
# endif

#endif
