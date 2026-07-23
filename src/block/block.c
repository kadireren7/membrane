#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/stats.h"

/* CRC32, poly 0xEDB88320 (reflected), table built lazily on first use. */
static uint32_t g_crc_table[256];
static int      g_crc_table_ready = 0;

static void crc32_init_table(void)
{
    for (uint32_t i = 0; i < 256; i++)
    {
        uint32_t c = i;

        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crc_table[i] = c;
    }
    g_crc_table_ready = 1;
}

uint32_t membrane_block_checksum(const uint8_t *buf, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;

    if (!g_crc_table_ready)
        crc32_init_table();
    for (size_t i = 0; i < len; i++)
        crc = g_crc_table[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

membrane_block_t *membrane_block_create(uint64_t id, membrane_codec_t codec)
{
    membrane_block_t *block;

    if (membrane_codec_get(codec) == NULL)
        return NULL;
    block = calloc(1, sizeof(*block));
    if (block == NULL)
        return NULL;
    block->id = id;
    block->codec = codec;
    block->state = MEMBRANE_BLOCK_HOT;
    return block;
}

void membrane_block_destroy(membrane_block_t *block)
{
    if (block == NULL)
        return;
    free(block->data);
    free(block);
}

membrane_status_t membrane_block_write(membrane_block_t *block,
                                        const uint8_t *in, size_t in_len)
{
    const membrane_codec_vtable_t   *codec;
    uint8_t                         *buf;
    size_t                          bound;
    size_t                          stored;
    membrane_status_t               status;

    if (block == NULL || (in == NULL && in_len > 0))
        return MEMBRANE_ERR_INVALID_ARG;
    codec = membrane_codec_get(block->codec);
    if (codec == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    if (in_len == 0)
    {
        free(block->data);
        block->data = NULL;
        block->original_size = 0;
        block->stored_size = 0;
        block->checksum = membrane_block_checksum(in, 0);
        block->access_count++;
        block->last_access_ns = membrane_now_ns();
        return MEMBRANE_OK;
    }
    bound = codec->bound(in_len);
    if (bound < in_len)
        return MEMBRANE_ERR_INVALID_ARG;
    buf = malloc(bound);
    if (buf == NULL)
        return MEMBRANE_ERR_ALLOC_FAILED;
    status = codec->compress(in, in_len, buf, bound, &stored);
    if (status != MEMBRANE_OK)
    {
        free(buf);
        return status;
    }
    if (stored < bound)
    {
        uint8_t *shrunk = realloc(buf, stored);

        if (shrunk != NULL)
            buf = shrunk;
    }
    free(block->data);
    block->data = buf;
    block->original_size = in_len;
    block->stored_size = stored;
    block->checksum = membrane_block_checksum(in, in_len);
    block->access_count++;
    block->last_access_ns = membrane_now_ns();
    return MEMBRANE_OK;
}

membrane_status_t membrane_block_read(membrane_block_t *block,
                                       uint8_t *out, size_t out_cap,
                                       size_t *out_len)
{
    const membrane_codec_vtable_t   *codec;
    size_t                          produced;
    membrane_status_t               status;

    if (block == NULL || out_len == NULL || (out == NULL && out_cap > 0))
        return MEMBRANE_ERR_INVALID_ARG;
    codec = membrane_codec_get(block->codec);
    if (codec == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    if (out_cap < block->original_size)
        return MEMBRANE_ERR_BUFFER_TOO_SMALL;
    if (block->original_size == 0)
    {
        *out_len = 0;
        block->access_count++;
        block->last_access_ns = membrane_now_ns();
        return MEMBRANE_OK;
    }
    status = codec->decompress(block->data, block->stored_size,
                               out, out_cap, &produced);
    if (status != MEMBRANE_OK)
        return status;
    if (produced != block->original_size
        || membrane_block_checksum(out, produced) != block->checksum)
        return MEMBRANE_ERR_CORRUPT_DATA;
    *out_len = produced;
    block->access_count++;
    block->last_access_ns = membrane_now_ns();
    return MEMBRANE_OK;
}
