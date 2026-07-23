#include <string.h>

#include "membrane/codec.h"
#include "test_helpers.h"

static const membrane_codec_vtable_t *g_raw;

static void test_empty_input(void)
{
    uint8_t out[8];
    uint8_t back[8];
    size_t  out_len = 99;
    size_t  back_len = 99;

    TEST_ASSERT(g_raw->compress(NULL, 0, out, sizeof(out), &out_len) == MEMBRANE_OK,
                "raw compress of empty input");
    TEST_ASSERT(out_len == 0, "empty input compresses to zero bytes");
    TEST_ASSERT(g_raw->decompress(NULL, 0, back, sizeof(back), &back_len) == MEMBRANE_OK,
                "raw decompress of empty input");
    TEST_ASSERT(back_len == 0, "empty input decompresses to zero bytes");
}

static void test_one_byte(void)
{
    uint8_t in = 0x5A;
    uint8_t out[4];
    uint8_t back[4];
    size_t  out_len;
    size_t  back_len;

    TEST_ASSERT(g_raw->compress(&in, 1, out, sizeof(out), &out_len) == MEMBRANE_OK,
                "raw compress one byte");
    TEST_ASSERT(out_len == 1 && out[0] == 0x5A, "raw stores the byte verbatim");
    TEST_ASSERT(g_raw->decompress(out, out_len, back, sizeof(back), &back_len) == MEMBRANE_OK,
                "raw decompress one byte");
    TEST_ASSERT(back_len == 1 && back[0] == 0x5A, "raw round-trips one byte");
}

static void test_random_data(void)
{
    enum { N = 64 * 1024 };
    uint8_t *in = malloc(N);
    uint8_t *out = malloc(g_raw->bound(N));
    uint8_t *back = malloc(N);
    size_t  out_len;
    size_t  back_len;

    TEST_ASSERT(in && out && back, "test buffers allocate");
    test_fill_random(in, N, 1234);
    TEST_ASSERT(g_raw->compress(in, N, out, g_raw->bound(N), &out_len) == MEMBRANE_OK,
                "raw compress random data");
    TEST_ASSERT(out_len == N, "raw output size equals input size");
    TEST_ASSERT(g_raw->decompress(out, out_len, back, N, &back_len) == MEMBRANE_OK,
                "raw decompress random data");
    TEST_ASSERT(back_len == N && memcmp(in, back, N) == 0,
                "raw round-trips random data bit-identically");
    free(in);
    free(out);
    free(back);
}

static void test_buffer_too_small(void)
{
    uint8_t in[16] = {0};
    uint8_t out[8];
    size_t  out_len;

    TEST_ASSERT(g_raw->compress(in, sizeof(in), out, sizeof(out), &out_len)
                    == MEMBRANE_ERR_BUFFER_TOO_SMALL,
                "raw compress rejects undersized output buffer");
    TEST_ASSERT(g_raw->decompress(in, sizeof(in), out, sizeof(out), &out_len)
                    == MEMBRANE_ERR_BUFFER_TOO_SMALL,
                "raw decompress rejects undersized output buffer");
}

int main(void)
{
    g_raw = membrane_codec_get(MEMBRANE_CODEC_RAW);
    TEST_ASSERT(g_raw != NULL, "raw codec is registered");
    TEST_ASSERT(strcmp(g_raw->name, "raw") == 0, "raw codec name");

    test_empty_input();
    test_one_byte();
    test_random_data();
    test_buffer_too_small();
    return 0;
}
