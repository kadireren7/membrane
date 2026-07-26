#ifndef MEMBRANE_ATTNTRACE_H
# define MEMBRANE_ATTNTRACE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 6.2: a versioned, replayable attention-access trace format,
 * extending Phase 6.1's kvtrace.h (which records only per-step KV
 * BYTE GROWTH) with per-step, per-layer, per-head block-level
 * attention distributions -- what membrane-kv-workingset-sim's
 * working-set policies and prefetch predictor are evaluated against.
 *
 * A trace records, for one real (or synthetic) decode, per decode
 * step / layer / query head: the top-`top_k` KV BLOCKS (not
 * individual tokens -- see block_size_tokens) by attention mass,
 * each with a normalized [0,1] score (that head's softmax weight
 * summed over the block's tokens). `tools/membrane-kv-attn-trace-
 * capture` produces MEMBRANE_ATTNTRACE_SOURCE_REAL_CAPTURE traces by
 * hooking llama.cpp's public ggml_backend_sched_eval_callback
 * (llama_context_params.cb_eval) to read the REAL "kq_soft_max"
 * tensor computed during an actual decode (flash attention disabled
 * for the capture run, since the fused flash-attention kernel never
 * materializes this tensor) -- genuine attention weights, not
 * modeled. membrane-kv-workingset-sim also generates
 * MEMBRANE_ATTNTRACE_SOURCE_SYNTHETIC traces (same format) for
 * sweeps at context lengths no single real capture run covers.
 *
 * Deliberately NOT stored per-token or as a dense n_kv-length vector:
 * at up to 128K context that would be gigabytes per trace. Only the
 * top `top_k` blocks per (step, layer, head) are kept -- this is a
 * real, disclosed reduction of the raw softmax output, not a claim
 * that attention mass outside the top-k is zero. See
 * docs/phase6-attention-working-set.md for the full disclosure.
 */
# define MEMBRANE_ATTNTRACE_MAGIC 0x54544D41u	/* "AMTT" little-endian */
# define MEMBRANE_ATTNTRACE_VERSION 1U
# define MEMBRANE_ATTNTRACE_HEADER_SIZE 128U
# define MEMBRANE_ATTNTRACE_MODEL_CAP 64U
# define MEMBRANE_ATTNTRACE_MAX_STEPS (1u << 20)
# define MEMBRANE_ATTNTRACE_MAX_LAYER 256U
# define MEMBRANE_ATTNTRACE_MAX_HEAD 256U
# define MEMBRANE_ATTNTRACE_MAX_TOPK 32U

typedef enum e_membrane_attntrace_source
{
	MEMBRANE_ATTNTRACE_SOURCE_SYNTHETIC = 0,
	MEMBRANE_ATTNTRACE_SOURCE_REAL_CAPTURE = 1
}	membrane_attntrace_source_t;

typedef struct s_membrane_attntrace_header
{
	char		model[MEMBRANE_ATTNTRACE_MODEL_CAP];
	uint32_t	source;			/* membrane_attntrace_source_t */
	uint32_t	n_layer;
	uint32_t	n_head;			/* query heads (post-GQA-expansion) */
	uint32_t	block_size_tokens;	/* KV tokens aggregated per block */
	uint32_t	prompt_len;		/* tokens resident before step 0 */
	uint32_t	step_count;		/* decode steps recorded */
	uint32_t	top_k;			/* entries per (step, layer, head) */
	uint64_t	created_unix_time;
	uint32_t	payload_checksum;	/* CRC32 of the whole record array */
}	membrane_attntrace_header_t;

/* One (block_id, normalized attention mass) entry. */
typedef struct s_membrane_attntrace_entry
{
	uint32_t	block_id;
	float		score;
}	membrane_attntrace_entry_t;

/*
 * Writes a full trace file: header, then
 * step_count * n_layer * n_head * top_k entries, in that nesting
 * order (step outermost, entry innermost). Returns
 * MEMBRANE_ERR_INVALID_ARG for a zero/over-cap step_count, n_layer,
 * n_head, or top_k, or a model name too long to fit;
 * MEMBRANE_ERR_IO on a write failure.
 */
membrane_status_t	membrane_attntrace_write(FILE *f,
						const membrane_attntrace_header_t *h,
						const membrane_attntrace_entry_t *entries);

/*
 * Reads and validates the header (magic, version, bound checks).
 * Does not read the entry array -- call membrane_attntrace_read_entries
 * next. Returns MEMBRANE_ERR_IO if the file cannot be read,
 * MEMBRANE_ERR_CORRUPT_DATA for a bad magic/version/truncated header
 * or an out-of-bound field.
 */
membrane_status_t	membrane_attntrace_read_header(FILE *f,
						membrane_attntrace_header_t *h);

/*
 * Reads h->step_count * h->n_layer * h->n_head * h->top_k entries
 * into `out` (caller-owned, must hold that many entries) and
 * verifies payload_checksum. Returns MEMBRANE_ERR_CORRUPT_DATA on
 * truncation or checksum mismatch.
 */
membrane_status_t	membrane_attntrace_read_entries(FILE *f,
						const membrane_attntrace_header_t *h,
						membrane_attntrace_entry_t *out);

/* Total entry count for a header -- bounds-checked against
 * MEMBRANE_ATTNTRACE_MAX_STEPS-scale overflow before multiplying. */
size_t	membrane_attntrace_entry_count(const membrane_attntrace_header_t *h);

# ifdef __cplusplus
}
# endif

#endif
