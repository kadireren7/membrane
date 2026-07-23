/* Stub: implemented separately. */

#include "membrane/block.h"

membrane_block_t *membrane_block_create(uint64_t id, membrane_codec_t codec)
{
    (void)id;
    (void)codec;
    return NULL;
}

void membrane_block_destroy(membrane_block_t *block)
{
    (void)block;
}

membrane_status_t membrane_block_write(membrane_block_t *block,
                                        const uint8_t *in, size_t in_len)
{
    (void)block;
    (void)in;
    (void)in_len;
    return MEMBRANE_ERR_UNIMPLEMENTED;
}

membrane_status_t membrane_block_read(membrane_block_t *block,
                                       uint8_t *out, size_t out_cap,
                                       size_t *out_len)
{
    (void)block;
    (void)out;
    (void)out_cap;
    (void)out_len;
    return MEMBRANE_ERR_UNIMPLEMENTED;
}

uint32_t membrane_block_checksum(const uint8_t *buf, size_t len)
{
    (void)buf;
    (void)len;
    return 0;
}
