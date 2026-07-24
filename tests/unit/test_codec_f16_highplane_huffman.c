#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "test_helpers.h"

# define HP_HEADER 18

static const membrane_codec_vtable_t	*g_hp;

/* Fill a buffer of F16 elements with a constant high byte and random low
 * byte: the high plane collapses to one symbol (very compressible), the
 * low plane stays noise. */
static void	fill_compressible(uint8_t *buf, size_t n, uint32_t seed)
{
	size_t	i;

	test_fill_random(buf, n, seed);
	i = 1;
	while (i < n)
	{
		buf[i] = 0x3C;
		i += 2;
	}
}

static void	roundtrip(const uint8_t *in, size_t len, const char *label)
{
	uint8_t	*out;
	uint8_t	*back;
	size_t	bound;
	size_t	out_len;
	size_t	back_len;

	bound = g_hp->bound(len);
	out = malloc(bound + 1);
	back = malloc(len + 1);
	TEST_ASSERT(out && back, "roundtrip buffers allocate");
	TEST_ASSERT(g_hp->compress(in, len, out, bound, &out_len) == MEMBRANE_OK,
		label);
	TEST_ASSERT(out_len <= bound, "compressed size within bound");
	TEST_ASSERT(g_hp->decompress(out, out_len, back, len, &back_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(back_len == len && (len == 0 || memcmp(in, back, len) == 0),
		"round-trip is bit-identical");
	free(out);
	free(back);
}

static void	test_roundtrips(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;

	roundtrip(NULL, 0, "highplane empty");
	in = malloc(N);
	TEST_ASSERT(in != NULL, "buffer allocates");
	test_fill_random(in, N, 707);
	roundtrip(in, N, "highplane random F16");
	fill_compressible(in, N, 909);
	roundtrip(in, N, "highplane compressible F16");
	free(in);
}

static void	test_compressible_shrinks(void)
{
	enum { N = 64 * 1024 };
	uint8_t	in[N];
	uint8_t	out[HP_HEADER + N + 512];
	size_t	out_len;

	fill_compressible(in, N, 111);
	TEST_ASSERT(g_hp->compress(in, N, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "compress compressible");
	TEST_ASSERT(out_len < N, "constant high plane compresses below raw");
}

static void	test_odd_length(void)
{
	uint8_t	in[9];
	uint8_t	out[64];
	size_t	out_len;

	memset(in, 0x5A, sizeof(in));
	TEST_ASSERT(g_hp->compress(in, 9, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_INVALID_ARG, "odd length rejected");
}

static void	test_corrupted_header(void)
{
	enum { N = 512 };
	uint8_t	in[N];
	uint8_t	out[HP_HEADER + N + 512];
	uint8_t	tmp[HP_HEADER + N + 512];
	uint8_t	back[N];
	size_t	out_len;
	size_t	n;

	fill_compressible(in, N, 5);
	TEST_ASSERT(g_hp->compress(in, N, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "compress valid");
	memcpy(tmp, out, out_len);
	tmp[0] = 9;
	TEST_ASSERT(g_hp->decompress(tmp, out_len, back, N, &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "bad version rejected");
	memcpy(tmp, out, out_len);
	tmp[6] ^= 0xFF;
	TEST_ASSERT(g_hp->decompress(tmp, out_len, back, N, &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "low_len != element_count rejected");
	TEST_ASSERT(g_hp->decompress(out, out_len - 1, back, N, &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "truncated stream rejected");
	TEST_ASSERT(g_hp->decompress(out, out_len, back, N / 2, &n)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "small out buffer rejected");
}

static void	test_checksum_tamper(void)
{
	enum { N = 512 };
	uint8_t	in[N];
	uint8_t	out[HP_HEADER + N + 512];
	uint8_t	back[N];
	size_t	out_len;
	size_t	n;

	fill_compressible(in, N, 6);
	TEST_ASSERT(g_hp->compress(in, N, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "compress valid");
	out[HP_HEADER] ^= 0xFF;	/* flip a low-plane byte; header CRC must trip */
	TEST_ASSERT(g_hp->decompress(out, out_len, back, N, &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "tampered low plane fails codec CRC");
}

static void	test_block_paths(void)
{
	enum { N = 4096 };
	membrane_block_t	*blk;
	uint8_t				in[N];
	uint8_t				back[N];
	size_t				got;

	fill_compressible(in, N, 1234);
	blk = membrane_block_create(0, MEMBRANE_CODEC_F16_HIGHPLANE_HUFFMAN);
	TEST_ASSERT(blk != NULL, "block create");
	TEST_ASSERT(membrane_block_write(blk, in, N) == MEMBRANE_OK, "block write");
	TEST_ASSERT(blk->stored_codec == MEMBRANE_CODEC_F16_HIGHPLANE_HUFFMAN,
		"compressible F16 uses the huffman codec");
	TEST_ASSERT(blk->stored_size < N, "huffman shrinks the block");
	TEST_ASSERT(membrane_block_read(blk, back, N, &got) == MEMBRANE_OK
		&& got == N && memcmp(in, back, N) == 0, "huffman block round-trips");
	((uint8_t *)blk->data)[blk->stored_size - 1] ^= 0xFF;
	TEST_ASSERT(membrane_block_decode(blk, back, N, &got)
		== MEMBRANE_ERR_CORRUPT_DATA, "tampered stored bytes fail checksum");
	membrane_block_destroy(blk);
	test_fill_random(in, N, 4321);
	blk = membrane_block_create(0, MEMBRANE_CODEC_F16_HIGHPLANE_HUFFMAN);
	TEST_ASSERT(blk != NULL, "block create 2");
	TEST_ASSERT(membrane_block_write(blk, in, N) == MEMBRANE_OK, "block write 2");
	TEST_ASSERT(blk->stored_codec == MEMBRANE_CODEC_RAW,
		"incompressible F16 falls back to RAW");
	membrane_block_destroy(blk);
}

int	main(void)
{
	g_hp = membrane_codec_get(MEMBRANE_CODEC_F16_HIGHPLANE_HUFFMAN);
	TEST_ASSERT(g_hp != NULL, "highplane huffman codec is registered");
	TEST_ASSERT(strcmp(g_hp->name, "f16hp") == 0, "codec name");
	test_roundtrips();
	test_compressible_shrinks();
	test_odd_length();
	test_corrupted_header();
	test_checksum_tamper();
	test_block_paths();
	return (0);
}
