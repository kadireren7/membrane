#ifndef MEMBRANE_BLOCK_H
# define MEMBRANE_BLOCK_H

# include <stddef.h>
# include <stdint.h>

# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

typedef enum e_membrane_state
{
	MEMBRANE_BLOCK_HOT = 0,
	MEMBRANE_BLOCK_WARM,
	MEMBRANE_BLOCK_COLD,
	MEMBRANE_BLOCK_EVICTED
}	membrane_state_t;

/*
 * data points to block-owned storage holding the (possibly compressed)
 * bytes; checksum is the CRC32 of the original, uncompressed bytes.
 * requested_codec is the caller's preference and is retried on every
 * write; stored_codec records how the current data is actually stored
 * (it becomes MEMBRANE_CODEC_RAW when the requested codec did not pay).
 */
typedef struct s_membrane_block
{
	uint64_t			id;
	void				*data;
	size_t				original_size;
	size_t				stored_size;
	uint64_t			access_count;
	uint64_t			last_access_ns;
	/* DEPRECATED: tier state moves to the Phase 1 store's entries;
	 * the block layer never reads or updates this field. */
	membrane_state_t	state;
	membrane_codec_t	requested_codec;
	membrane_codec_t	stored_codec;
	uint32_t			checksum;
}	membrane_block_t;

/*
 * Allocates a block struct for the given id/codec. The block owns no
 * storage until membrane_block_write() is called. Returns NULL on
 * allocation failure.
 */
membrane_block_t	*membrane_block_create(uint64_t id,
						membrane_codec_t codec);

/* Frees a block's owned storage and the struct itself. NULL is a no-op. */
void				membrane_block_destroy(membrane_block_t *block);

/*
 * Compresses `in` (in_len bytes) via the block's codec into
 * block-owned storage, updating original_size, stored_size, checksum,
 * and access metadata. Compression is attempted with requested_codec
 * on every write; if it does not actually shrink the data, the block
 * falls back to RAW storage (recorded in stored_codec), so
 * incompressible data never expands. The fallback never sticks: a
 * later write with compressible data compresses again automatically.
 */
membrane_status_t	membrane_block_write(membrane_block_t *block,
						const uint8_t *in, size_t in_len);

/*
 * Decompresses the block's stored data into caller-provided `out`
 * (out_cap bytes) and verifies the result against the stored checksum.
 * Pure with respect to the block: no metadata is modified, so
 * concurrent decodes of the same block are safe. Returns
 * MEMBRANE_ERR_CORRUPT_DATA if the checksum does not match.
 */
membrane_status_t	membrane_block_decode(const membrane_block_t *block,
						uint8_t *out, size_t out_cap, size_t *out_len);

/*
 * Records an access: increments access_count and refreshes
 * last_access_ns. Callers that share a block across threads must
 * serialize this themselves (the Phase 1 store does it under its lock).
 */
void				membrane_block_touch(membrane_block_t *block);

/* Convenience wrapper: decode, then touch on success. Not thread-safe. */
membrane_status_t	membrane_block_read(membrane_block_t *block,
						uint8_t *out, size_t out_cap, size_t *out_len);

/* CRC32 (poly 0xEDB88320, reflected, table-based) over buf[0..len). */
uint32_t			membrane_block_checksum(const uint8_t *buf, size_t len);

# ifdef __cplusplus
}
# endif

#endif
