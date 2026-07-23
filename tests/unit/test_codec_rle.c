#include <string.h>

#include "membrane/codec.h"
#include "test_helpers.h"

static const membrane_codec_vtable_t *g_rle;

static void roundtrip(const uint8_t *in, size_t in_len, const char *label)
{
    size_t  bound = g_rle->bound(in_len);
    uint8_t *out = malloc(bound ? bound : 1);
    uint8_t *back = malloc(in_len ? in_len : 1);
    size_t  out_len;
    size_t  back_len;

    TEST_ASSERT(out && back, "roundtrip buffers allocate");
    TEST_ASSERT(g_rle->compress(in, in_len, out, bound, &out_len) == MEMBRANE_OK, label);
    TEST_ASSERT(out_len <= bound, "compressed size within bound");
    TEST_ASSERT(g_rle->decompress(out, out_len, back, in_len, &back_len) == MEMBRANE_OK, label);
    TEST_ASSERT(back_len == in_len, "decompressed size matches original");
    TEST_ASSERT(in_len == 0 || memcmp(in, back, in_len) == 0,
                "round-trip is bit-identical");
    free(out);
    free(back);
}

static void test_empty_input(void)
{
    roundtrip(NULL, 0, "rle empty input");
}

static void test_one_byte(void)
{
    uint8_t in = 0xA5;

    roundtrip(&in, 1, "rle one byte");
}

static void test_random_data(void)
{
    enum { N = 64 * 1024 };
    uint8_t *in = malloc(N);

    TEST_ASSERT(in != NULL, "random buffer allocates");
    test_fill_random(in, N, 42);
    roundtrip(in, N, "rle random data");
    free(in);
}

static void test_all_zero(void)
{
    enum { N = 64 * 1024 };
    uint8_t *in = calloc(1, N);
    uint8_t out[1024];
    size_t  out_len;

    TEST_ASSERT(in != NULL, "zero buffer allocates");
    roundtrip(in, N, "rle all-zero data");
    /* 64 KiB of zeros = 256 max-length runs = 512 bytes: must compress hard. */
    TEST_ASSERT(g_rle->compress(in, N, out, sizeof(out), &out_len) == MEMBRANE_OK,
                "rle compress all-zero into small buffer");
    TEST_ASSERT(out_len == 512, "all-zero compresses to 2 bytes per 256-run");
    free(in);
}

static void test_repeated_pattern(void)
{
    enum { N = 4096 };
    uint8_t in[N];

    for (size_t i = 0; i < N; i++)
        in[i] = (i % 2 == 0) ? 'A' : 'B';
    roundtrip(in, N, "rle alternating pattern (worst case)");

    /* A run longer than 256 must split into multiple pairs correctly. */
    memset(in, 'X', 300);
    roundtrip(in, 300, "rle run longer than 256");
}

static void test_corrupt_stream(void)
{
    uint8_t bad[3] = {0xFF, 0x00, 0xFF};
    uint8_t out[4096];
    size_t  out_len;

    /* Odd-length stream is structurally invalid. */
    TEST_ASSERT(g_rle->decompress(bad, 3, out, sizeof(out), &out_len)
                    == MEMBRANE_ERR_CORRUPT_DATA,
                "rle rejects odd-length stream");

    /* A stream expanding past out_cap must fail cleanly, not overflow. */
    uint8_t big[2] = {0xFF, 0x42};
    uint8_t tiny[16];

    TEST_ASSERT(g_rle->decompress(big, 2, tiny, sizeof(tiny), &out_len)
                    == MEMBRANE_ERR_BUFFER_TOO_SMALL,
                "rle rejects output overflow");
}

int main(void)
{
    g_rle = membrane_codec_get(MEMBRANE_CODEC_RLE);
    TEST_ASSERT(g_rle != NULL, "rle codec is registered");
    TEST_ASSERT(strcmp(g_rle->name, "rle") == 0, "rle codec name");

    test_empty_input();
    test_one_byte();
    test_random_data();
    test_all_zero();
    test_repeated_pattern();
    test_corrupt_stream();
    return 0;
}
