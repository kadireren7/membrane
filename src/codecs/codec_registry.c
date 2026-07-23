#include <string.h>

#include "membrane/codec.h"

extern const membrane_codec_vtable_t g_membrane_codec_raw;
extern const membrane_codec_vtable_t g_membrane_codec_rle;

static membrane_status_t unimplemented_compress(const uint8_t *in, size_t in_len,
                                                 uint8_t *out, size_t out_cap,
                                                 size_t *out_len)
{
    (void)in;
    (void)in_len;
    (void)out;
    (void)out_cap;
    (void)out_len;
    return MEMBRANE_ERR_UNIMPLEMENTED;
}

static membrane_status_t unimplemented_decompress(const uint8_t *in, size_t in_len,
                                                   uint8_t *out, size_t out_cap,
                                                   size_t *out_len)
{
    (void)in;
    (void)in_len;
    (void)out;
    (void)out_cap;
    (void)out_len;
    return MEMBRANE_ERR_UNIMPLEMENTED;
}

static size_t unimplemented_bound(size_t in_len)
{
    (void)in_len;
    return 0;
}

static const membrane_codec_vtable_t g_membrane_codec_lz4 = {
    .name = "lz4",
    .id = MEMBRANE_CODEC_LZ4,
    .compress = unimplemented_compress,
    .decompress = unimplemented_decompress,
    .bound = unimplemented_bound,
};

static const membrane_codec_vtable_t g_membrane_codec_bitpack = {
    .name = "bitpack",
    .id = MEMBRANE_CODEC_BITPACK,
    .compress = unimplemented_compress,
    .decompress = unimplemented_decompress,
    .bound = unimplemented_bound,
};

const membrane_codec_vtable_t *membrane_codec_get(membrane_codec_t id)
{
    switch (id)
    {
        case MEMBRANE_CODEC_RAW:
            return &g_membrane_codec_raw;
        case MEMBRANE_CODEC_RLE:
            return &g_membrane_codec_rle;
        case MEMBRANE_CODEC_LZ4:
            return &g_membrane_codec_lz4;
        case MEMBRANE_CODEC_BITPACK:
            return &g_membrane_codec_bitpack;
        default:
            return NULL;
    }
}

int membrane_codec_from_name(const char *name, membrane_codec_t *out_id)
{
    if (name == NULL || out_id == NULL)
        return 0;
    for (int id = 0; id < MEMBRANE_CODEC_COUNT; id++)
    {
        const membrane_codec_vtable_t *codec = membrane_codec_get((membrane_codec_t)id);

        if (codec != NULL && strcmp(codec->name, name) == 0)
        {
            *out_id = (membrane_codec_t)id;
            return 1;
        }
    }
    return 0;
}
