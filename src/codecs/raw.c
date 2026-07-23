/* Stub: implemented separately. */

#include "membrane/codec.h"

static membrane_status_t raw_compress(const uint8_t *in, size_t in_len,
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

static membrane_status_t raw_decompress(const uint8_t *in, size_t in_len,
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

static size_t raw_bound(size_t in_len)
{
    return in_len;
}

const membrane_codec_vtable_t g_membrane_codec_raw = {
    .name = "raw",
    .id = MEMBRANE_CODEC_RAW,
    .compress = raw_compress,
    .decompress = raw_decompress,
    .bound = raw_bound,
};
