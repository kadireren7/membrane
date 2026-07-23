#include <string.h>

#include "membrane/codec.h"
#include "test_helpers.h"

static const membrane_codec_vtable_t	*g_rle;

static void	roundtrip(const uint8_t *in, size_t in_len, const char *label)
{
	uint8_t	*out;
	uint8_t	*back;
	size_t	bound;
	size_t	out_len;
	size_t	back_len;

	bound = g_rle->bound(in_len);
	out = malloc(bound + !bound);
	back = malloc(in_len + !in_len);
	TEST_ASSERT(out && back, "roundtrip buffers allocate");
	TEST_ASSERT(g_rle->compress(in, in_len, out, bound, &out_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(out_len <= bound, "compressed size within bound");
	TEST_ASSERT(g_rle->decompress(out, out_len, back, in_len, &back_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(back_len == in_len, "decompressed size matches original");
	TEST_ASSERT(in_len == 0 || memcmp(in, back, in_len) == 0,
		"round-trip is bit-identical");
	free(out);
	free(back);
}

static void	test_empty_input(void)
{
	roundtrip(NULL, 0, "rle empty input");
}

static void	test_one_byte(void)
{
	uint8_t	in;

	in = 0xA5;
	roundtrip(&in, 1, "rle one byte");
}

static void	test_random_data(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;

	in = malloc(N);
	TEST_ASSERT(in != NULL, "random buffer allocates");
	test_fill_random(in, N, 42);
	roundtrip(in, N, "rle random data");
	free(in);
}

static void	test_all_zero(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;
	uint8_t	out[1024];
	size_t	out_len;

	in = calloc(1, N);
	TEST_ASSERT(in != NULL, "zero buffer allocates");
	roundtrip(in, N, "rle all-zero data");
	TEST_ASSERT(g_rle->compress(in, N, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "rle compress all-zero into small buffer");
	TEST_ASSERT(out_len == 512, "all-zero compresses to 2 bytes per 256-run");
	free(in);
}

static void	test_repeated_pattern(void)
{
	enum { N = 4096 };
	uint8_t	in[N];
	size_t	i;

	i = 0;
	while (i < N)
	{
		if (i % 2 == 0)
			in[i] = 'A';
		else
			in[i] = 'B';
		i++;
	}
	roundtrip(in, N, "rle alternating pattern (worst case)");
	memset(in, 'X', 300);
	roundtrip(in, 300, "rle run longer than 256");
}

static void	test_corrupt_stream(void)
{
	uint8_t	bad[3];
	uint8_t	big[2];
	uint8_t	out[16];
	size_t	out_len;

	memset(bad, 0xFF, sizeof(bad));
	TEST_ASSERT(g_rle->decompress(bad, 3, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "rle rejects odd-length stream");
	big[0] = 0xFF;
	big[1] = 0x42;
	TEST_ASSERT(g_rle->decompress(big, 2, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "rle rejects output overflow");
}

int	main(void)
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
	return (0);
}
