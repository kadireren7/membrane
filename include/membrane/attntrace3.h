#ifndef MEMBRANE_ATTNTRACE3_H
# define MEMBRANE_ATTNTRACE3_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/attntrace.h"
# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 6.5: chunked, out-of-core sibling of attntrace2.h. v2 fixed the
 * "file too big to commit" problem (3 bytes/entry, whole-payload
 * DEFLATE); it did not fix "file too big to hold entirely in RAM at
 * once" -- v2's own reader still does one malloc(n*3) + one fread for
 * the WHOLE payload (attntrace2.c's membrane_attntrace2_read_entries),
 * which is exactly what makes the unified sweep's ~2-4 GiB synthetic-
 * extended trace OOM this machine. v3 keeps v2's entry encoding
 * (uint16 block_id + uint8 fixed-point score) but splits the payload
 * into independently-checksummed, independently-(de)compressible
 * chunks of `chunk_steps` decode-steps each, indexed right after the
 * header so a reader can seek straight to any chunk without scanning
 * or materializing the rest -- see attn_trace_reader.h (C++, built on
 * top of this) for the streaming/mmap backends that consume this.
 *
 * File layout:
 *   [0, HEADER_SIZE)                       fixed header
 *   [index_offset, index_offset+index_size) chunk_count fixed-size
 *                                            index entries, in
 *                                            increasing step order
 *                                            (deterministic)
 *   [payload_offset, EOF)                   chunk payloads, in the
 *                                            same increasing order,
 *                                            each independently
 *                                            optionally-DEFLATE'd and
 *                                            CRC32'd
 *
 * `header.file_sha256` is SHA-256 over [index_offset, EOF) -- the
 * index and every chunk payload, i.e. everything a reader actually
 * trusts. It excludes the header itself (which would be
 * self-referential) the same way v2's payload_checksum excludes its
 * own header. Writers compute it with one bounded streaming pass
 * (membrane_sha256_update in fixed-size reads, see hash.h) rather than
 * hashing the whole file in memory -- the entire point of this format
 * is to never require that.
 */
# define MEMBRANE_ATTNTRACE3_MAGIC 0x33544D41u	/* "AMT3" little-endian */
# define MEMBRANE_ATTNTRACE3_VERSION 1U
# define MEMBRANE_ATTNTRACE3_HEADER_SIZE 256U
# define MEMBRANE_ATTNTRACE3_MODEL_CAP 64U
# define MEMBRANE_ATTNTRACE3_MAX_STEPS (1u << 24)
# define MEMBRANE_ATTNTRACE3_MAX_LAYER 256U
# define MEMBRANE_ATTNTRACE3_MAX_HEAD 256U
# define MEMBRANE_ATTNTRACE3_MAX_TOPK 32U
# define MEMBRANE_ATTNTRACE3_MAX_BLOCK_ID 0xFFFEu
# define MEMBRANE_ATTNTRACE3_MAX_CHUNK_STEPS (1u << 20)
# define MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE 32U

typedef struct s_membrane_attntrace3_header
{
	char		model[MEMBRANE_ATTNTRACE3_MODEL_CAP];
	uint32_t	source;
	uint32_t	n_layer;
	uint32_t	n_head;
	uint32_t	block_size_tokens;
	uint32_t	prompt_len;
	uint32_t	step_count;
	uint32_t	top_k;
	uint64_t	created_unix_time;
	uint32_t	chunk_steps;	/* steps per chunk; last chunk may hold
						 * fewer (step_count % chunk_steps) */
	uint32_t	chunk_count;
	uint64_t	index_offset;
	uint64_t	index_size;
	uint64_t	payload_offset;
	uint8_t		file_sha256[32];
}	membrane_attntrace3_header_t;

typedef struct s_membrane_attntrace3_chunk_index_entry
{
	uint32_t	step_lo;
	uint32_t	step_count_in_chunk;
	uint64_t	payload_offset;		/* absolute file offset */
	uint32_t	stored_size;		/* bytes on disk */
	uint32_t	uncompressed_size;	/* bytes, compact encoding */
	uint32_t	crc32;				/* CRC32 of the UNCOMPRESSED
							 * compact chunk payload, always
							 * (even when stored compressed) */
	uint8_t		compressed;			/* 0 or 1 */
}	membrane_attntrace3_chunk_index_entry_t;

/*
 * Streaming writer. Total `step_count` must be known up front (every
 * real caller in this codebase already knows its target step count),
 * so the chunk index can be reserved and written before any payload,
 * letting a reader seek to any chunk without a prior scan. Chunks
 * MUST be written in order 0..chunk_count-1 -- out-of-order writes
 * are rejected (deterministic ordering, not just a convention).
 *
 * `f` must be opened "w+b" (read+write), not "wb": writer_close()
 * reads back what was just written to compute the streaming
 * file_sha256 over the same FILE* rather than reopening the path.
 */
typedef struct s_membrane_attntrace3_writer
{
	membrane_attntrace3_header_t	h;
	uint64_t						next_payload_offset;
	uint32_t						next_chunk_id;
}	membrane_attntrace3_writer_t;

membrane_status_t	membrane_attntrace3_writer_open(FILE *f,
						membrane_attntrace3_writer_t *w,
						const char *model, uint32_t source,
						uint32_t n_layer, uint32_t n_head,
						uint32_t block_size_tokens, uint32_t prompt_len,
						uint32_t step_count, uint32_t top_k,
						uint32_t chunk_steps);

/*
 * Encodes and writes chunk `chunk_id` (the writer's own next expected
 * id -- see above). `entries` holds exactly
 * membrane_attntrace3_writer_chunk_len(w, chunk_id) *
 * n_layer * n_head * top_k entries, step-major/layer/head/top_k
 * nested, same as v1/v2. Never holds more than one chunk's worth of
 * entries at a time -- the caller is expected to generate one chunk,
 * call this, then discard it before generating the next.
 */
membrane_status_t	membrane_attntrace3_writer_put_chunk(FILE *f,
						membrane_attntrace3_writer_t *w, uint32_t chunk_id,
						const membrane_attntrace_entry_t *entries,
						int compress);

uint32_t	membrane_attntrace3_writer_chunk_len(
				const membrane_attntrace3_writer_t *w, uint32_t chunk_id);

/* Finalizes: computes file_sha256 over [index_offset, EOF) with one
 * bounded streaming pass and patches it (plus the header CRC32) into
 * the already-written header. Fails if not every chunk was written. */
membrane_status_t	membrane_attntrace3_writer_close(FILE *f,
						membrane_attntrace3_writer_t *w);

membrane_status_t	membrane_attntrace3_read_header(FILE *f,
						membrane_attntrace3_header_t *h);

/* Reads the full chunk index (h->chunk_count entries, caller-owned
 * buffer) -- small and constant-ish size (chunk_count *
 * MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE bytes; at chunk_steps=512 a
 * 130560-step trace is 256 entries / 8 KiB), safe to hold in full even
 * under a tight memory budget. */
membrane_status_t	membrane_attntrace3_read_index(FILE *f,
						const membrane_attntrace3_header_t *h,
						membrane_attntrace3_chunk_index_entry_t *out);

/* FILE*-based single-chunk read (buffered/streaming backend path):
 * seeks, reads the stored bytes, decodes via
 * membrane_attntrace3_decode_chunk_buf below. */
membrane_status_t	membrane_attntrace3_read_chunk(FILE *f,
						const membrane_attntrace3_header_t *h,
						const membrane_attntrace3_chunk_index_entry_t *entry,
						membrane_attntrace_entry_t *out);

/* Pure in-memory decode of one chunk's already-read (or mmap'd)
 * stored bytes -- the mmap backend calls this directly on the mapped
 * pointer, no intermediate copy. Decompresses if `entry->compressed`,
 * verifies CRC32 against `entry->crc32`, decodes compact entries into
 * the wide (u32, float) representation. */
membrane_status_t	membrane_attntrace3_decode_chunk_buf(
						const uint8_t *stored,
						const membrane_attntrace3_chunk_index_entry_t *entry,
						membrane_attntrace_entry_t *out);

/* Re-derives file_sha256 over [h->index_offset, EOF) and compares
 * against the header's stored digest -- lets a caller (tests, the
 * integrity tool) validate a whole file without loading any chunk's
 * entries. */
membrane_status_t	membrane_attntrace3_verify_file_sha256(FILE *f,
						const membrane_attntrace3_header_t *h);

uint32_t	membrane_attntrace3_step_to_chunk(
				const membrane_attntrace3_header_t *h, uint32_t step);

size_t	membrane_attntrace3_chunk_entry_count(
			const membrane_attntrace3_header_t *h,
			const membrane_attntrace3_chunk_index_entry_t *entry);

# ifdef __cplusplus
}
# endif

#endif
