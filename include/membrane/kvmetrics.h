#ifndef MEMBRANE_KVMETRICS_H
# define MEMBRANE_KVMETRICS_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Compressibility metrics for one buffer split into fixed-size blocks.
 * Deterministic: the same buffer and block size always yield identical
 * results. adaptive_* mirrors the store's per-block decision (RLE with
 * RAW fallback via membrane_block_write); integrity_ok confirms every
 * adaptive block decoded back bit-identically.
 */
typedef struct s_membrane_kv_metrics
{
	uint64_t	raw_bytes;
	uint64_t	rle_bytes;
	uint64_t	adaptive_bytes;
	uint64_t	blocks;
	uint64_t	adaptive_raw_blocks;
	uint64_t	adaptive_rle_blocks;
	uint64_t	zero_bytes;
	uint64_t	total_runs;
	uint64_t	max_run;
	double		entropy;	/* size-weighted mean Shannon entropy, bits/byte */
	int			integrity_ok;
}	membrane_kv_metrics_t;

membrane_status_t	membrane_kv_metrics_compute(const uint8_t *buf,
						size_t len, size_t block_size,
						membrane_kv_metrics_t *out);

# ifdef __cplusplus
}
# endif

#endif
