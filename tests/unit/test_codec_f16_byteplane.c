#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "test_helpers.h"

# define BP_HEADER 14

static const membrane_codec_vtable_t	*g_bp;

/* Compress `in`, decompress the result, and assert a bit-identical
 * round-trip. Returns the compressed length via *comp_len for callers
 * that want to inspect the stream. */
static void	roundtrip(const uint8_t *in, size_t in_len, const char *label)
{
	uint8_t	*out;
	uint8_t	*back;
	size_t	bound;
	size_t	out_len;
	size_t	back_len;

	bound = g_bp->bound(in_len);
	out = malloc(bound + 1);
	back = malloc(in_len + 1);
	TEST_ASSERT(out && back, "roundtrip buffers allocate");
	TEST_ASSERT(g_bp->compress(in, in_len, out, bound, &out_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(out_len <= bound, "compressed size within bound");
	TEST_ASSERT(g_bp->decompress(out, out_len, back, in_len, &back_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(back_len == in_len, "decompressed size matches original");
	TEST_ASSERT(in_len == 0 || memcmp(in, back, in_len) == 0,
		"round-trip is bit-identical");
	free(out);
	free(back);
}

/* A hand-built 4-element F16 stream and the exact header it must produce. */
static void	test_known_sequence(void)
{
	static const uint8_t	known[8] = {
		0x00, 0x3C, 0x00, 0x40, 0x66, 0x3E, 0xCD, 0xBC};
	uint8_t					out[64];
	size_t					out_len;
	uint32_t				plane_len;

	roundtrip(known, sizeof(known), "byteplane known sequence");
	TEST_ASSERT(g_bp->compress(known, sizeof(known), out, sizeof(out),
		&out_len) == MEMBRANE_OK, "byteplane compress known");
	TEST_ASSERT(out[0] == 1 && out[1] == 0, "header version/reserved");
	plane_len = (uint32_t)out[2] | ((uint32_t)out[3] << 8)
		| ((uint32_t)out[4] << 16) | ((uint32_t)out[5] << 24);
	TEST_ASSERT(plane_len == 4, "header plane_len is in_len/2");
}

static void	test_empty(void)
{
	uint8_t	out[BP_HEADER];
	size_t	out_len;
	size_t	back_len;

	TEST_ASSERT(g_bp->compress(NULL, 0, out, sizeof(out), &out_len)
		== MEMBRANE_OK && out_len == BP_HEADER, "empty compresses to header");
	TEST_ASSERT(g_bp->decompress(out, out_len, NULL, 0, &back_len)
		== MEMBRANE_OK && back_len == 0, "empty decompresses to nothing");
}

static void	test_random(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;

	in = malloc(N);
	TEST_ASSERT(in != NULL, "random buffer allocates");
	test_fill_random(in, N, 4242);
	roundtrip(in, N, "byteplane random F16");
	free(in);
}

/* High byte constant, low byte with runs -- both planes compress well. */
static void	test_repeated(void)
{
	enum { N = 8192 };
	uint8_t	in[N];
	size_t	i;

	i = 0;
	while (i < N)
	{
		in[i] = (uint8_t)((i / 2) / 64);
		in[i + 1] = 0x3C;
		i += 2;
	}
	roundtrip(in, N, "byteplane repeated F16");
}

static void	test_odd_length_rejected(void)
{
	uint8_t	in[5];
	uint8_t	out[64];
	size_t	out_len;

	memset(in, 0x7F, sizeof(in));
	TEST_ASSERT(g_bp->compress(in, 5, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_INVALID_ARG, "odd length rejected");
	TEST_ASSERT(g_bp->compress(in, 1, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_INVALID_ARG, "single byte rejected");
}

static void	test_compress_out_too_small(void)
{
	uint8_t	in[8];
	uint8_t	out[10];
	size_t	out_len;

	memset(in, 0xAB, sizeof(in));
	TEST_ASSERT(g_bp->compress(in, sizeof(in), out, sizeof(out), &out_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "compress rejects tiny out buffer");
}

/* Builds a valid stream, then mutates it to exercise every decode guard. */
static void	test_corrupted_streams(void)
{
	static const uint8_t	in[8] = {
		0x00, 0x3C, 0x00, 0x40, 0x66, 0x3E, 0xCD, 0xBC};
	uint8_t					out[64];
	uint8_t					back[8];
	uint8_t					tmp[64];
	size_t					out_len;
	size_t					n;

	TEST_ASSERT(g_bp->compress(in, sizeof(in), out, sizeof(out), &out_len)
		== MEMBRANE_OK, "build valid stream");
	memcpy(tmp, out, out_len);
	tmp[0] = 2;
	TEST_ASSERT(g_bp->decompress(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "bad version rejected");
	memcpy(tmp, out, out_len);
	tmp[1] = 9;
	TEST_ASSERT(g_bp->decompress(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "nonzero reserved rejected");
	TEST_ASSERT(g_bp->decompress(out, BP_HEADER - 1, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "sub-header length rejected");
	TEST_ASSERT(g_bp->decompress(out, out_len - 1, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "truncated stream rejected");
}

/* Header lengths that are internally inconsistent must be caught. */
static void	test_plane_length_mismatch(void)
{
	static const uint8_t	in[8] = {
		0x00, 0x3C, 0x00, 0x40, 0x66, 0x3E, 0xCD, 0xBC};
	uint8_t					out[64];
	uint8_t					back[64];
	size_t					out_len;
	size_t					n;

	TEST_ASSERT(g_bp->compress(in, sizeof(in), out, sizeof(out), &out_len)
		== MEMBRANE_OK, "build valid stream");
	out[2] = 8;
	TEST_ASSERT(g_bp->decompress(out, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "overstated plane_len rejected");
	out[2] = 2;
	TEST_ASSERT(g_bp->decompress(out, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "understated plane_len rejected");
}

static void	test_decompress_out_too_small(void)
{
	static const uint8_t	in[8] = {
		0x00, 0x3C, 0x00, 0x40, 0x66, 0x3E, 0xCD, 0xBC};
	uint8_t					out[64];
	uint8_t					back[4];
	size_t					out_len;
	size_t					n;

	TEST_ASSERT(g_bp->compress(in, sizeof(in), out, sizeof(out), &out_len)
		== MEMBRANE_OK, "build valid stream");
	TEST_ASSERT(g_bp->decompress(out, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "decompress rejects small out");
}

/* Through the block layer: incompressible F16 must fall back to RAW. */
static void	test_raw_fallback(void)
{
	enum { N = 4096 };
	membrane_block_t	*blk;
	uint8_t				in[N];
	uint8_t				back[N];
	size_t				got;

	test_fill_random(in, N, 31337);
	blk = membrane_block_create(0, MEMBRANE_CODEC_F16_BYTEPLANE_RLE);
	TEST_ASSERT(blk != NULL, "block create");
	TEST_ASSERT(membrane_block_write(blk, in, N) == MEMBRANE_OK, "block write");
	TEST_ASSERT(blk->stored_codec == MEMBRANE_CODEC_RAW,
		"random F16 falls back to RAW");
	TEST_ASSERT(blk->stored_size == N, "RAW fallback keeps original size");
	TEST_ASSERT(membrane_block_read(blk, back, N, &got) == MEMBRANE_OK
		&& got == N && memcmp(in, back, N) == 0, "RAW fallback round-trips");
	membrane_block_destroy(blk);
}

/* Through the block layer: compressible F16 stays byteplane-coded and the
 * checksum trips if the stored bytes are tampered with. */
static void	test_block_checksum(void)
{
	enum { N = 4096 };
	membrane_block_t	*blk;
	uint8_t				in[N];
	uint8_t				back[N];
	size_t				got;
	size_t				i;

	i = 0;
	while (i < N)
	{
		in[i] = 0x00;
		in[i + 1] = 0x3C;
		i += 2;
	}
	blk = membrane_block_create(0, MEMBRANE_CODEC_F16_BYTEPLANE_RLE);
	TEST_ASSERT(blk != NULL, "block create");
	TEST_ASSERT(membrane_block_write(blk, in, N) == MEMBRANE_OK, "block write");
	TEST_ASSERT(blk->stored_codec == MEMBRANE_CODEC_F16_BYTEPLANE_RLE,
		"compressible F16 uses the byteplane codec");
	TEST_ASSERT(blk->stored_size < N, "byteplane actually shrinks the block");
	TEST_ASSERT(membrane_block_read(blk, back, N, &got) == MEMBRANE_OK
		&& got == N && memcmp(in, back, N) == 0, "byteplane block round-trips");
	((uint8_t *)blk->data)[blk->stored_size - 1] ^= 0xFF;
	TEST_ASSERT(membrane_block_decode(blk, back, N, &got)
		== MEMBRANE_ERR_CORRUPT_DATA, "tampered stored bytes fail checksum");
	membrane_block_destroy(blk);
}

int	main(void)
{
	g_bp = membrane_codec_get(MEMBRANE_CODEC_F16_BYTEPLANE_RLE);
	TEST_ASSERT(g_bp != NULL, "byteplane codec is registered");
	TEST_ASSERT(strcmp(g_bp->name, "f16bp") == 0, "byteplane codec name");
	test_known_sequence();
	test_empty();
	test_random();
	test_repeated();
	test_odd_length_rejected();
	test_compress_out_too_small();
	test_corrupted_streams();
	test_plane_length_mismatch();
	test_decompress_out_too_small();
	test_raw_fallback();
	test_block_checksum();
	return (0);
}
