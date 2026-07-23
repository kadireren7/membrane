#ifndef MEMBRANE_BLOCK_H
#define MEMBRANE_BLOCK_H

#include <stddef.h>
#include <stdint.h>

#include "membrane/codec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum e_membrane_state
{
    MEMBRANE_BLOCK_HOT = 0,
    MEMBRANE_BLOCK_WARM,
    MEMBRANE_BLOCK_COLD,
    MEMBRANE_BLOCK_EVICTED
}   membrane_state_t;

typedef struct s_membrane_block
{
    uint64_t            id;
    void                *data;           /* owned storage, possibly compressed */
    size_t              original_size;
    size_t              stored_size;
    uint64_t            access_count;
    uint64_t            last_access_ns;
    membrane_state_t    state;
    membrane_codec_t    codec;
    uint32_t            checksum;        /* CRC32 of the original, uncompressed bytes */
}   membrane_block_t;

/*
 * Allocates a block struct for the given id/codec. The block owns no
 * storage until membrane_block_write() is called. Returns NULL on
 * allocation failure.
 */
membrane_block_t *membrane_block_create(uint64_t id, membrane_codec_t codec);

/* Frees a block's owned storage and the struct itself. Safe to call with NULL. */
void membrane_block_destroy(membrane_block_t *block);

/*
 * Compresses `in` (in_len bytes) via the block's codec into
 * block-owned storage, updating original_size, stored_size, checksum,
 * and access metadata.
 */
membrane_status_t membrane_block_write(membrane_block_t *block,
                                        const uint8_t *in, size_t in_len);

/*
 * Decompresses the block's stored data into caller-provided `out`
 * (out_cap bytes), verifies the result against the stored checksum,
 * and updates access metadata. Returns MEMBRANE_ERR_CORRUPT_DATA if
 * the checksum does not match.
 */
membrane_status_t membrane_block_read(membrane_block_t *block,
                                       uint8_t *out, size_t out_cap,
                                       size_t *out_len);

/* CRC32 (poly 0xEDB88320, reflected, table-based) over buf[0..len). */
uint32_t membrane_block_checksum(const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* MEMBRANE_BLOCK_H */
