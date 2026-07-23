#include <string.h>

#include "membrane/codec.h"

static membrane_status_t raw_compress(const uint8_t *in, size_t in_len,
                                       uint8_t *out, size_t out_cap,
                                       size_t *out_len)
{
    if ((in == NULL && in_len > 0) || out_len == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    if (out_cap < in_len)
        return MEMBRANE_ERR_BUFFER_TOO_SMALL;
    if (in_len > 0)
        memcpy(out, in, in_len);
    *out_len = in_len;
    return MEMBRANE_OK;
}

static membrane_status_t raw_decompress(const uint8_t *in, size_t in_len,
                                         uint8_t *out, size_t out_cap,
                                         size_t *out_len)
{
    if ((in == NULL && in_len > 0) || out_len == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    if (out_cap < in_len)
        return MEMBRANE_ERR_BUFFER_TOO_SMALL;
    if (in_len > 0)
        memcpy(out, in, in_len);
    *out_len = in_len;
    return MEMBRANE_OK;
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
