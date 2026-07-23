/* Stub: implemented separately. */

#include "membrane/codec.h"

static membrane_status_t rle_compress(const uint8_t *in, size_t in_len,
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

static membrane_status_t rle_decompress(const uint8_t *in, size_t in_len,
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

static size_t rle_bound(size_t in_len)
{
    /* Worst case: every byte becomes its own (run_length, value) pair. */
    return in_len * 2;
}

const membrane_codec_vtable_t g_membrane_codec_rle = {
    .name = "rle",
    .id = MEMBRANE_CODEC_RLE,
    .compress = rle_compress,
    .decompress = rle_decompress,
    .bound = rle_bound,
};
