#ifndef MEMBRANE_ATTNTRACE2_H
# define MEMBRANE_ATTNTRACE2_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/attntrace.h"
# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 6.4: a compact, optionally DEFLATE-compressed sibling of
 * attntrace.h -- built specifically because Phase 6.3's raw format
 * (uint32 block_id + float32 score, 8 bytes/entry) produced 34-121MB
 * committed trace files that GitHub's push size limit rejected once,
 * forcing an emergency top-k reduction (docs/phase6-exact-sparse-
 * retrieval.md section 2/17). This format fixes the root cause
 * instead of trading away resolution again:
 *
 * - block_id: uint16 instead of uint32 (2 bytes). Lossless for any
 *   block id < 65,535 -- true at every context length this project
 *   has swept (128K tokens / 16-token blocks = 8,192 blocks max).
 * - score: uint8 fixed-point in [0,255] representing [0.0,1.0]
 *   instead of float32 (1 byte). A REAL, disclosed lossy step
 *   (~0.4% quantization error) -- but the score was already a lossy
 *   top-k-truncated approximation of the real softmax output
 *   (attntrace.h's own header comment), and every downstream
 *   consumer (working-set policies, hit/miss decisions) only uses
 *   relative ranking and coarse magnitude, never exact values.
 * - optional whole-payload DEFLATE compression (zlib), on top of the
 *   already-3-bytes/entry compact encoding.
 *
 * Combined: 8 bytes/entry -> 3 bytes/entry before compression (2.67x),
 * typically another ~1.5-2x from DEFLATE on real captured data (block
 * ids cluster heavily around sink/recency positions, compressing
 * well) -- real, measured per-file, not assumed (see the capture
 * tool's own reported compression ratio).
 *
 * `membrane_attntrace2_read_entries()` decodes directly back into
 * `membrane_attntrace_entry_t` (attntrace.h's own type), so every
 * existing consumer of the v1 format works unchanged against v2 data
 * -- only the loader differs (attn_workload.cpp's `load_attn_trace()`
 * auto-detects the magic number and dispatches to the right reader).
 */
# define MEMBRANE_ATTNTRACE2_MAGIC 0x32544D41u	/* "AMT2" little-endian */
# define MEMBRANE_ATTNTRACE2_VERSION 1U
# define MEMBRANE_ATTNTRACE2_HEADER_SIZE 128U
# define MEMBRANE_ATTNTRACE2_MODEL_CAP 64U
# define MEMBRANE_ATTNTRACE2_MAX_STEPS (1u << 20)
# define MEMBRANE_ATTNTRACE2_MAX_LAYER 256U
# define MEMBRANE_ATTNTRACE2_MAX_HEAD 256U
# define MEMBRANE_ATTNTRACE2_MAX_TOPK 32U
# define MEMBRANE_ATTNTRACE2_MAX_BLOCK_ID 0xFFFEu	/* 0xFFFF reserved
							 * as the "no entry"
							 * sentinel */

typedef struct s_membrane_attntrace2_header
{
	char		model[MEMBRANE_ATTNTRACE2_MODEL_CAP];
	uint32_t	source;
	uint32_t	n_layer;
	uint32_t	n_head;
	uint32_t	block_size_tokens;
	uint32_t	prompt_len;
	uint32_t	step_count;
	uint32_t	top_k;
	uint64_t	created_unix_time;
	uint32_t	compressed;			/* 0 or 1 */
	uint32_t	uncompressed_payload_size;	/* bytes, compact encoding */
	uint32_t	stored_payload_size;		/* bytes actually on disk */
	uint32_t	payload_checksum;		/* CRC32 of the UNCOMPRESSED
							 * compact payload, always
							 * (even when stored
							 * compressed) */
}	membrane_attntrace2_header_t;

/*
 * Writes a full v2 trace: header, then the compact-encoded (and, if
 * `compress` is nonzero, DEFLATE-compressed) entry payload. `entries`
 * uses the SAME `membrane_attntrace_entry_t` type as attntrace.h --
 * this function does the quantization internally. Any input
 * block_id > MEMBRANE_ATTNTRACE2_MAX_BLOCK_ID is rejected
 * (MEMBRANE_ERR_INVALID_ARG) rather than silently truncated.
 */
membrane_status_t	membrane_attntrace2_write(FILE *f,
						const membrane_attntrace2_header_t *h,
						const membrane_attntrace_entry_t *entries,
						int compress);

membrane_status_t	membrane_attntrace2_read_header(FILE *f,
						membrane_attntrace2_header_t *h);

/* Decodes back into the same membrane_attntrace_entry_t
 * representation attntrace.h's v1 reader produces -- decompresses
 * first if the header says compressed, then verifies
 * payload_checksum against the UNCOMPRESSED bytes, then decodes each
 * 3-byte compact entry (u16 block_id, u8 score) into the wider
 * (u32, float) representation. */
membrane_status_t	membrane_attntrace2_read_entries(FILE *f,
						const membrane_attntrace2_header_t *h,
						membrane_attntrace_entry_t *out);

size_t	membrane_attntrace2_entry_count(
			const membrane_attntrace2_header_t *h);

# ifdef __cplusplus
}
# endif

#endif
