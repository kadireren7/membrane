#include <stdlib.h>
#include <string.h>

#include "membrane/huffman.h"
#include "test_helpers.h"

/* Compress then decompress, assert a lossless round-trip. Returns the
 * compressed length so callers can check the ratio. */
static size_t	roundtrip(const uint8_t *in, size_t len, const char *label)
{
	uint8_t	*comp;
	uint8_t	*back;
	size_t	bound;
	size_t	comp_len;
	size_t	back_len;

	bound = membrane_huffman_bound(len);
	comp = malloc(bound + 1);
	back = malloc(len + 1);
	TEST_ASSERT(comp && back, "huffman buffers allocate");
	TEST_ASSERT(membrane_huffman_compress(in, len, comp, bound, &comp_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(comp_len <= bound, "compressed size within bound");
	TEST_ASSERT(membrane_huffman_decompress(comp, comp_len, back, len,
		&back_len) == MEMBRANE_OK, label);
	TEST_ASSERT(back_len == len, "decoded size matches original");
	TEST_ASSERT(len == 0 || memcmp(in, back, len) == 0,
		"round-trip is bit-identical");
	free(comp);
	free(back);
	return (comp_len);
}

static void	test_empty(void)
{
	uint8_t	out[8];
	size_t	out_len;

	roundtrip(NULL, 0, "huffman empty");
	TEST_ASSERT(membrane_huffman_compress(NULL, 0, out, sizeof(out), &out_len)
		== MEMBRANE_OK && out_len == 4, "empty encodes to a 4-byte count");
}

static void	test_single_symbol(void)
{
	enum { N = 8192 };
	uint8_t	in[N];
	size_t	comp;

	memset(in, 0x7E, sizeof(in));
	comp = roundtrip(in, N, "huffman single symbol");
	TEST_ASSERT(comp < N / 4, "single symbol compresses hard (1 bit/symbol)");
}

static void	test_full_alphabet(void)
{
	enum { N = 256 * 32 };
	uint8_t	in[N];
	size_t	i;

	i = 0;
	while (i < N)
	{
		in[i] = (uint8_t)(i & 0xFF);
		i++;
	}
	roundtrip(in, N, "huffman full uniform alphabet");
}

static void	test_random(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;

	in = malloc(N);
	TEST_ASSERT(in != NULL, "random buffer allocates");
	test_fill_random(in, N, 8181);
	roundtrip(in, N, "huffman random");
	free(in);
}

/* Geometric frequencies drive the natural Huffman depth past 15 bits,
 * exercising the length-limiting repair; the round-trip must still hold. */
static void	test_skewed_triggers_limiter(void)
{
	uint8_t	*in;
	size_t	total;
	size_t	off;
	int		k;
	size_t	c;

	total = 0;
	k = 0;
	while (k <= 16)
		total += ((size_t)1 << (16 - k++));
	in = malloc(total);
	TEST_ASSERT(in != NULL, "skewed buffer allocates");
	off = 0;
	k = 0;
	while (k <= 16)
	{
		c = (size_t)1 << (16 - k);
		memset(in + off, k, c);
		off += c;
		k++;
	}
	roundtrip(in, total, "huffman skewed (length limiter)");
	free(in);
}

/* An over-subscribed length table (every symbol length 1) is malformed. */
static void	test_malformed_table(void)
{
	enum { N = 200 };
	uint8_t	in[N];
	uint8_t	comp[512];
	uint8_t	back[N];
	size_t	comp_len;
	size_t	back_len;

	test_fill_random(in, N, 4);
	TEST_ASSERT(membrane_huffman_compress(in, N, comp, sizeof(comp), &comp_len)
		== MEMBRANE_OK, "compress for tamper");
	memset(comp + 4, 0x11, MEMBRANE_HUFFMAN_TABLE_BYTES);
	TEST_ASSERT(membrane_huffman_decompress(comp, comp_len, back, N, &back_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "over-subscribed table rejected");
}

static void	test_truncated_and_header(void)
{
	enum { N = 4096 };
	uint8_t	in[N];
	uint8_t	comp[8192];
	uint8_t	back[N];
	size_t	comp_len;
	size_t	back_len;

	memset(in, 0, sizeof(in));
	test_fill_random(in, N, 21);
	TEST_ASSERT(membrane_huffman_compress(in, N, comp, sizeof(comp), &comp_len)
		== MEMBRANE_OK, "compress for truncation");
	TEST_ASSERT(membrane_huffman_decompress(comp, comp_len - 1, back, N,
		&back_len) == MEMBRANE_ERR_CORRUPT_DATA, "truncated bitstream rejected");
	TEST_ASSERT(membrane_huffman_decompress(comp, 3, back, N, &back_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "sub-count length rejected");
	TEST_ASSERT(membrane_huffman_decompress(comp, MEMBRANE_HUFFMAN_HEADER - 1,
		back, N, &back_len) == MEMBRANE_ERR_CORRUPT_DATA,
		"non-empty stream missing table rejected");
}

static void	test_overflow(void)
{
	enum { N = 1024 };
	uint8_t	in[N];
	uint8_t	comp[4096];
	uint8_t	back[N];
	size_t	comp_len;
	size_t	back_len;

	test_fill_random(in, N, 33);
	TEST_ASSERT(membrane_huffman_compress(in, N, comp, 4, &comp_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "compress rejects tiny out buffer");
	TEST_ASSERT(membrane_huffman_compress(in, N, comp, sizeof(comp), &comp_len)
		== MEMBRANE_OK, "compress ok");
	TEST_ASSERT(membrane_huffman_decompress(comp, comp_len, back, N / 2,
		&back_len) == MEMBRANE_ERR_BUFFER_TOO_SMALL,
		"decompress rejects small out buffer");
}

int	main(void)
{
	test_empty();
	test_single_symbol();
	test_full_alphabet();
	test_random();
	test_skewed_triggers_limiter();
	test_malformed_table();
	test_truncated_and_header();
	test_overflow();
	return (0);
}
