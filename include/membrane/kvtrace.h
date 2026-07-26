#ifndef MEMBRANE_KVTRACE_H
# define MEMBRANE_KVTRACE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 6.1: a versioned, replayable KV decode-access trace format.
 *
 * A trace records, for one sequence's decode phase, the REAL per-step
 * growth of that sequence's KV cache: one uint32_t byte count per decode
 * step, each the actual number of bytes the FP16 KV cache grew by when
 * that one token's K/V vectors were committed across every layer/head.
 * `membrane-kv-trace-capture` (tools/membrane-kv-trace-capture) produces
 * these by measuring real llama_state_seq_get_size() deltas step-by-step
 * during an actual decode -- MEMBRANE_KVTRACE_SOURCE_REAL_CAPTURE traces
 * are genuine measurements, not modeled.
 *
 * `membrane-cxl-sim` also generates MEMBRANE_KVTRACE_SOURCE_SYNTHETIC
 * traces (same file format, same reader) for parametric sweeps at
 * context lengths/concurrency no single real capture run covers --
 * synthetic traces replay a real-measured per-step byte rate (typically
 * copied from a real capture's own steady-state rate) rather than
 * inventing one, but are extrapolated in LENGTH, which is why the
 * `source` field exists: so nothing downstream can silently mistake an
 * extrapolated trace for a fully-real one.
 *
 * Deliberately NOT modeled per-block/per-layer: for tractable discrete-
 * event simulation at up to 512 concurrent sequences x 128K context
 * (potentially billions of individual 32-element block accesses), each
 * step is one aggregated write event sized at the real total per-token
 * KV footprint (summed across all layers/heads) -- see
 * docs/phase6-cxl-near-memory.md's "simulation granularity" section for
 * this scope decision stated in full.
 */
# define MEMBRANE_KVTRACE_MAGIC 0x5254564Du	/* "MVTR" little-endian */
# define MEMBRANE_KVTRACE_VERSION 1U
# define MEMBRANE_KVTRACE_HEADER_SIZE 128U
# define MEMBRANE_KVTRACE_MODEL_CAP 64U
# define MEMBRANE_KVTRACE_MAX_STEPS (1u << 24)

typedef enum e_membrane_kvtrace_source
{
	MEMBRANE_KVTRACE_SOURCE_SYNTHETIC = 0,
	MEMBRANE_KVTRACE_SOURCE_REAL_CAPTURE = 1
}	membrane_kvtrace_source_t;

typedef struct s_membrane_kvtrace_header
{
	char		model[MEMBRANE_KVTRACE_MODEL_CAP];
	uint32_t	source;			/* membrane_kvtrace_source_t */
	uint32_t	n_layer;		/* real model geometry when known, else 0 */
	uint32_t	n_head_kv;		/* real model geometry when known, else 0 */
	uint32_t	prompt_len;		/* tokens already resident before step 0 */
	uint32_t	step_count;		/* number of per-step byte deltas that follow */
	uint64_t	created_unix_time;
	uint32_t	payload_checksum;	/* CRC32 of the step_count x u32 array */
}	membrane_kvtrace_header_t;

/*
 * Writes a full trace file: header, then step_count little-endian
 * uint32_t byte deltas. Returns MEMBRANE_ERR_INVALID_ARG for a zero
 * step_count, a step_count over MEMBRANE_KVTRACE_MAX_STEPS, or a model
 * name too long to fit; MEMBRANE_ERR_IO on a write failure.
 */
membrane_status_t	membrane_kvtrace_write(FILE *f,
						const membrane_kvtrace_header_t *h,
						const uint32_t *step_bytes);

/*
 * Reads and validates the header (magic, version, step_count bound).
 * Does not read the step array -- call membrane_kvtrace_read_steps next.
 * Returns MEMBRANE_ERR_IO if the file cannot be read,
 * MEMBRANE_ERR_CORRUPT_DATA for a bad magic/version/truncated header.
 */
membrane_status_t	membrane_kvtrace_read_header(FILE *f,
						membrane_kvtrace_header_t *h);

/*
 * Reads h->step_count uint32_t deltas into `out` (caller-owned, must
 * hold at least h->step_count entries) and verifies payload_checksum.
 * Returns MEMBRANE_ERR_CORRUPT_DATA on truncation or checksum mismatch.
 */
membrane_status_t	membrane_kvtrace_read_steps(FILE *f,
						const membrane_kvtrace_header_t *h, uint32_t *out);

# ifdef __cplusplus
}
# endif

#endif
