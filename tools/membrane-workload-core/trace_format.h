#ifndef MEMBRANE_TRACE_FORMAT_H
# define MEMBRANE_TRACE_FORMAT_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMKV: a minimal, versioned, offline trace format for feeding real
 * captured KV-cache block data into tools/membrane-quant-policy-bench,
 * without that benchmark ever depending on llama.cpp or model weights at
 * run time. A trace holds ONLY the numerical block payload (raw F16
 * values, block-major) plus a short optional metadata label -- no
 * prompt/token text, no model weights. See docs/kv-trace-format.md for
 * the full wire layout and provenance/privacy rationale.
 *
 * Produced by tools/membrane-kv-trace-export (converts an existing
 * membrane-kv-capture .kvdump record; itself llama-free) or by any other
 * writer of this format; consumed here by a genuinely streaming reader
 * (membrane_trace_read_block reads one block per call via sequential
 * fread, never the whole payload at once) so a caller controls its own
 * memory footprint.
 *
 * Wire layout (little-endian throughout), MEMBRANE_TRACE_HEADER_SIZE
 * (64) bytes:
 *
 *   offset  size  field
 *   0       8     magic       "MEMKV01" + 1 NUL byte
 *   8       4     format_version (u32)
 *   12      4     dtype          (u32, membrane_trace_dtype_t)
 *   16      4     elements_per_block (u32, > 0)
 *   20      4     metadata_length (u32, bytes immediately following header)
 *   24      8     block_count    (u64, > 0)
 *   32      8     payload_bytes  (u64, == block_count * elements_per_block
 *                                 * dtype element size)
 *   40      8     created_unix_time (u64, informational only)
 *   48      4     payload_checksum (u32, CRC32 of the raw block payload)
 *   52      4     header_checksum  (u32, CRC32 of bytes [0, 52))
 *   56      8     reserved       (u64, must be 0)
 *
 * Followed by metadata_length bytes of UTF-8 metadata (a short safe
 * label -- never prompt/token text), then payload_bytes bytes of raw
 * little-endian block data.
 */
# define MEMBRANE_TRACE_MAGIC			"MEMKV01"
# define MEMBRANE_TRACE_MAGIC_SIZE		8u
# define MEMBRANE_TRACE_VERSION			1u
# define MEMBRANE_TRACE_HEADER_SIZE		64u

/* Defensive bounds against a hostile/corrupt file -- chosen generously
 * for the trace sizes this format actually targets (one flattened
 * kvdump tensor record: realistically tens of MB at most), not as a
 * claim about what a "reasonable" trace must look like. */
# define MEMBRANE_TRACE_MAX_METADATA		4096u
# define MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK	65536u
# define MEMBRANE_TRACE_MAX_BLOCK_COUNT		1000000ull
# define MEMBRANE_TRACE_MAX_PAYLOAD_BYTES	(512ull * 1024 * 1024)

/* Internal safe copy of the metadata string; the file's own
 * metadata_length may be smaller (never larger, rejected by
 * membrane_trace_open) -- this just bounds membrane_trace_info_t's own
 * stack footprint and truncates a legitimately-short label safely. */
# define MEMBRANE_TRACE_METADATA_CAP		256u

typedef enum e_membrane_trace_dtype
{
	MEMBRANE_TRACE_DTYPE_F16 = 0
}	membrane_trace_dtype_t;

typedef struct s_membrane_trace_info
{
	uint32_t	format_version;
	uint32_t	dtype;
	uint32_t	elements_per_block;
	uint64_t	block_count;
	uint64_t	payload_bytes;
	uint64_t	created_unix_time;
	char		metadata[MEMBRANE_TRACE_METADATA_CAP];
}	membrane_trace_info_t;

typedef struct s_membrane_trace_reader	membrane_trace_reader_t;

/*
 * Opens `path`, reads and fully validates the fixed header and metadata
 * (magic, version, header checksum, dtype, elements_per_block,
 * block_count, and payload_bytes -- checked with overflow-safe
 * arithmetic against block_count*elements_per_block*element_size --
 * all before any block is read). On success returns MEMBRANE_OK and
 * *out owns the open file; the caller must call membrane_trace_close.
 * Returns MEMBRANE_ERR_IO if the file cannot be opened/read,
 * MEMBRANE_ERR_CORRUPT_DATA for a bad magic/version/header checksum, a
 * truncated header, or any length field failing validation (including
 * one that would overflow), MEMBRANE_ERR_UNIMPLEMENTED for a dtype this
 * reader does not support.
 */
membrane_status_t	membrane_trace_open(const char *path,
						membrane_trace_reader_t **out);

/* Copies the validated header/metadata into *info. */
void	membrane_trace_info(const membrane_trace_reader_t *r,
			membrane_trace_info_t *info);

/*
 * Reads exactly the next block (info->elements_per_block little-endian
 * values of info->dtype, currently always F16 -- out_f16 must hold at
 * least that many uint16_t) via a bounded sequential read; never reads
 * more than one block's worth of the file per call. Returns
 * MEMBRANE_ERR_NOT_FOUND once every block has been read (out_f16
 * untouched); MEMBRANE_ERR_CORRUPT_DATA on a short/truncated read, or,
 * on the call that completes the last block, if the running payload
 * checksum computed across every block read so far does not match the
 * header's payload_checksum.
 */
membrane_status_t	membrane_trace_read_block(membrane_trace_reader_t *r,
						uint16_t *out_f16);

/* NULL is a no-op. */
void	membrane_trace_close(membrane_trace_reader_t *r);

/*
 * Writes a complete trace file in one call from an in-memory,
 * block-major `blocks_f16` buffer (block_count * elements_per_block
 * contiguous little-endian values) -- used by
 * tools/membrane-kv-trace-export, which already holds a whole
 * (bounded-size) flattened kvdump payload in memory. Returns
 * MEMBRANE_ERR_INVALID_ARG for a zero/oversized block_count or
 * elements_per_block, an oversized metadata string, or a
 * block_count*elements_per_block product that would overflow;
 * MEMBRANE_ERR_IO on a write failure. `metadata` may be NULL (treated
 * as empty).
 */
membrane_status_t	membrane_trace_write(FILE *f,
						membrane_trace_dtype_t dtype,
						uint32_t elements_per_block, uint64_t block_count,
						const uint16_t *blocks_f16, const char *metadata,
						uint64_t created_unix_time);

# ifdef __cplusplus
}
# endif

#endif
