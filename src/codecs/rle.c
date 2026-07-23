/*
 * Byte-oriented run-length encoding.
 *
 * Stream format: a sequence of (run_length_minus_1, value) byte pairs,
 * so a single pair encodes a run of 1..256 identical bytes. An empty
 * input encodes to an empty output. Any odd-length stream is malformed.
 */

#include "membrane/codec.h"

static membrane_status_t rle_compress(const uint8_t *in, size_t in_len,
                                       uint8_t *out, size_t out_cap,
                                       size_t *out_len)
{
    size_t  ipos;
    size_t  opos;

    if ((in == NULL && in_len > 0) || out_len == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    ipos = 0;
    opos = 0;
    while (ipos < in_len)
    {
        uint8_t value = in[ipos];
        size_t  run = 1;

        while (ipos + run < in_len && in[ipos + run] == value && run < 256)
            run++;
        if (opos + 2 > out_cap)
            return MEMBRANE_ERR_BUFFER_TOO_SMALL;
        out[opos] = (uint8_t)(run - 1);
        out[opos + 1] = value;
        opos += 2;
        ipos += run;
    }
    *out_len = opos;
    return MEMBRANE_OK;
}

static membrane_status_t rle_decompress(const uint8_t *in, size_t in_len,
                                         uint8_t *out, size_t out_cap,
                                         size_t *out_len)
{
    size_t  ipos;
    size_t  opos;

    if ((in == NULL && in_len > 0) || out_len == NULL)
        return MEMBRANE_ERR_INVALID_ARG;
    if (in_len % 2 != 0)
        return MEMBRANE_ERR_CORRUPT_DATA;
    ipos = 0;
    opos = 0;
    while (ipos < in_len)
    {
        size_t  run = (size_t)in[ipos] + 1;
        uint8_t value = in[ipos + 1];

        if (opos + run > out_cap)
            return MEMBRANE_ERR_BUFFER_TOO_SMALL;
        for (size_t i = 0; i < run; i++)
            out[opos + i] = value;
        opos += run;
        ipos += 2;
    }
    *out_len = opos;
    return MEMBRANE_OK;
}

static size_t rle_bound(size_t in_len)
{
    /* Worst case: every byte becomes its own (run_length, value) pair. */
    if (in_len > SIZE_MAX / 2)
        return SIZE_MAX;
    return in_len * 2;
}

const membrane_codec_vtable_t g_membrane_codec_rle = {
    .name = "rle",
    .id = MEMBRANE_CODEC_RLE,
    .compress = rle_compress,
    .decompress = rle_decompress,
    .bound = rle_bound,
};
