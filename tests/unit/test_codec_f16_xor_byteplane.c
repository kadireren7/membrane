#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "membrane/f16xor.h"
#include "test_helpers.h"

# define XBP_HEADER 18

static const membrane_codec_vtable_t	*g_xbp;

static void	put_u16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

/* Encode with the given predictor, decode, assert a lossless round-trip. */
static void	roundtrip(membrane_predictor_t pred, size_t row_elems,
				const uint8_t *in, size_t len, const char *label)
{
	membrane_f16xor_cfg_t	cfg;
	uint8_t					*out;
	uint8_t					*back;
	size_t					out_len;
	size_t					back_len;

	cfg.predictor = pred;
	cfg.row_elems = row_elems;
	out = malloc(XBP_HEADER + len + 1);
	back = malloc(len + 1);
	TEST_ASSERT(out && back, "roundtrip buffers allocate");
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, len, out,
		XBP_HEADER + len, &out_len) == MEMBRANE_OK, label);
	TEST_ASSERT(out_len == XBP_HEADER + len, "encoded size is header + planes");
	TEST_ASSERT(membrane_f16xor_decode(out, out_len, back, len, &back_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(back_len == len, "decoded size matches original");
	TEST_ASSERT(len == 0 || memcmp(in, back, len) == 0,
		"round-trip is bit-identical");
	free(out);
	free(back);
}

/* Every predictor mode must round-trip, on a buffer that is a whole number
 * of rows so TOKEN/ROW are exercised on real strides. */
static void	test_all_predictors(void)
{
	enum { ROWS = 32, ELEMS = 12, N = ROWS * ELEMS * 2 };
	uint8_t	in[N];

	test_fill_random(in, N, 909);
	roundtrip(MEMBRANE_PRED_NONE, 0, in, N, "predictor none");
	roundtrip(MEMBRANE_PRED_XOR_PREV_ELEMENT, 0, in, N, "predictor element");
	roundtrip(MEMBRANE_PRED_XOR_PREV_TOKEN, ELEMS, in, N, "predictor token");
	roundtrip(MEMBRANE_PRED_XOR_PREV_ROW, ELEMS, in, N, "predictor row");
}

/* A hand-built uint16 stream: check the element-XOR residual is exactly
 * v0, v1^v0, v2^v1, ... and that the first element is kept raw. */
static void	test_known_sequence(void)
{
	uint8_t					in[8];
	uint8_t					out[64];
	uint8_t					back[8];
	membrane_f16xor_cfg_t	cfg;
	size_t					out_len;
	size_t					back_len;

	put_u16(in + 0, 0x1234);
	put_u16(in + 2, 0x12FF);
	put_u16(in + 4, 0x0001);
	put_u16(in + 6, 0xFFFF);
	cfg.predictor = MEMBRANE_PRED_XOR_PREV_ELEMENT;
	cfg.row_elems = 0;
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, sizeof(in), out, sizeof(out),
		&out_len) == MEMBRANE_OK, "encode known");
	TEST_ASSERT(out[0] == 1 && out[1] == MEMBRANE_PRED_XOR_PREV_ELEMENT,
		"header version + predictor");
	/* residual r1 = 0x12FF ^ 0x1234 = 0x00CB; low plane holds low bytes. */
	TEST_ASSERT(out[XBP_HEADER + 1] == 0xCB, "residual low byte of element 1");
	TEST_ASSERT(out[XBP_HEADER + 4 + 1] == 0x00, "residual high byte elem 1");
	TEST_ASSERT(membrane_f16xor_decode(out, out_len, back, sizeof(back),
		&back_len) == MEMBRANE_OK && memcmp(back, in, sizeof(in)) == 0,
		"known sequence round-trips");
}

static void	test_all_zero(void)
{
	enum { N = 4096 };
	uint8_t	in[N];

	memset(in, 0, sizeof(in));
	roundtrip(MEMBRANE_PRED_NONE, 0, in, N, "all-zero none");
	roundtrip(MEMBRANE_PRED_XOR_PREV_ELEMENT, 0, in, N, "all-zero element");
	roundtrip(MEMBRANE_PRED_XOR_PREV_TOKEN, 64, in, N, "all-zero token");
}

/* Monotonic uint16 bit patterns (v[i] = i): a stress case for the XOR
 * chain where residuals are small but non-trivial. */
static void	test_monotonic(void)
{
	enum { COUNT = 2048, N = COUNT * 2 };
	uint8_t	in[N];
	size_t	i;

	i = 0;
	while (i < COUNT)
	{
		put_u16(in + 2 * i, (uint16_t)i);
		i++;
	}
	roundtrip(MEMBRANE_PRED_XOR_PREV_ELEMENT, 0, in, N, "monotonic element");
	roundtrip(MEMBRANE_PRED_XOR_PREV_TOKEN, 128, in, N, "monotonic token");
}

static void	test_random_and_empty(void)
{
	enum { N = 64 * 1024 };
	uint8_t	*in;

	roundtrip(MEMBRANE_PRED_NONE, 0, NULL, 0, "empty none");
	roundtrip(MEMBRANE_PRED_XOR_PREV_ELEMENT, 0, NULL, 0, "empty element");
	in = malloc(N);
	TEST_ASSERT(in != NULL, "random buffer allocates");
	test_fill_random(in, N, 12321);
	roundtrip(MEMBRANE_PRED_XOR_PREV_ELEMENT, 0, in, N, "random element");
	roundtrip(MEMBRANE_PRED_XOR_PREV_TOKEN, 288, in, N, "random token");
	free(in);
}

static void	test_invalid_args(void)
{
	membrane_f16xor_cfg_t	cfg;
	uint8_t					in[8];
	uint8_t					out[64];
	size_t					out_len;

	memset(in, 0x5A, sizeof(in));
	cfg.predictor = MEMBRANE_PRED_XOR_PREV_TOKEN;
	cfg.row_elems = 0;
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, sizeof(in), out, sizeof(out),
		&out_len) == MEMBRANE_ERR_INVALID_ARG, "token without shape rejected");
	cfg.predictor = MEMBRANE_PRED_XOR_PREV_ELEMENT;
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, 7, out, sizeof(out), &out_len)
		== MEMBRANE_ERR_INVALID_ARG, "odd length rejected");
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, sizeof(in), out, 10, &out_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "tiny out buffer rejected");
}

static void	test_corrupted_header(void)
{
	membrane_f16xor_cfg_t	cfg;
	uint8_t					in[16];
	uint8_t					out[64];
	uint8_t					tmp[64];
	uint8_t					back[16];
	size_t					out_len;
	size_t					n;

	memset(in, 0x3C, sizeof(in));
	cfg.predictor = MEMBRANE_PRED_XOR_PREV_ELEMENT;
	cfg.row_elems = 0;
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, sizeof(in), out, sizeof(out),
		&out_len) == MEMBRANE_OK, "encode valid");
	memcpy(tmp, out, out_len);
	tmp[0] = 2;
	TEST_ASSERT(membrane_f16xor_decode(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "bad version rejected");
	memcpy(tmp, out, out_len);
	tmp[1] = (uint8_t)MEMBRANE_PRED_COUNT;
	TEST_ASSERT(membrane_f16xor_decode(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "out-of-range predictor rejected");
	memcpy(tmp, out, out_len);
	tmp[10] ^= 0xFF;
	TEST_ASSERT(membrane_f16xor_decode(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "inconsistent low_len rejected");
	memcpy(tmp, out, out_len);
	tmp[6] = 0xFF;
	tmp[7] = 0xFF;
	TEST_ASSERT(membrane_f16xor_decode(tmp, out_len, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "inconsistent plane_len rejected");
}

static void	test_truncated(void)
{
	membrane_f16xor_cfg_t	cfg;
	uint8_t					in[16];
	uint8_t					out[64];
	uint8_t					back[16];
	size_t					out_len;
	size_t					n;

	memset(in, 0x11, sizeof(in));
	cfg.predictor = MEMBRANE_PRED_XOR_PREV_ELEMENT;
	cfg.row_elems = 0;
	TEST_ASSERT(membrane_f16xor_encode(&cfg, in, sizeof(in), out, sizeof(out),
		&out_len) == MEMBRANE_OK, "encode valid");
	TEST_ASSERT(membrane_f16xor_decode(out, out_len - 1, back, sizeof(back), &n)
		== MEMBRANE_ERR_CORRUPT_DATA, "truncated stream rejected");
	TEST_ASSERT(membrane_f16xor_decode(out, XBP_HEADER - 1, back, sizeof(back),
		&n) == MEMBRANE_ERR_CORRUPT_DATA, "sub-header length rejected");
	TEST_ASSERT(membrane_f16xor_decode(out, out_len, back, 4, &n)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "small out buffer rejected");
}

/* Through the block layer: the RAW-plane codec always expands, so the
 * block must fall back to RAW and still round-trip and checksum-verify. */
static void	test_block_raw_fallback(void)
{
	enum { N = 4096 };
	membrane_block_t	*blk;
	uint8_t				in[N];
	uint8_t				back[N];
	size_t				got;

	test_fill_random(in, N, 55555);
	blk = membrane_block_create(0, MEMBRANE_CODEC_F16_XOR_BYTEPLANE);
	TEST_ASSERT(blk != NULL, "block create");
	TEST_ASSERT(membrane_block_write(blk, in, N) == MEMBRANE_OK, "block write");
	TEST_ASSERT(blk->stored_codec == MEMBRANE_CODEC_RAW,
		"RAW-plane codec falls back to RAW");
	TEST_ASSERT(membrane_block_read(blk, back, N, &got) == MEMBRANE_OK
		&& got == N && memcmp(in, back, N) == 0, "fallback round-trips");
	((uint8_t *)blk->data)[0] ^= 0xFF;
	TEST_ASSERT(membrane_block_decode(blk, back, N, &got)
		== MEMBRANE_ERR_CORRUPT_DATA, "tampered bytes fail checksum");
	membrane_block_destroy(blk);
}

/* The registered codec's own compress/decompress path (element predictor). */
static void	test_vtable_roundtrip(void)
{
	enum { N = 8192 };
	uint8_t	in[N];
	uint8_t	out[XBP_HEADER + N];
	uint8_t	back[N];
	size_t	out_len;
	size_t	back_len;

	test_fill_random(in, N, 4711);
	TEST_ASSERT(g_xbp->compress(in, N, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "vtable compress");
	TEST_ASSERT(g_xbp->bound(N) == XBP_HEADER + N, "vtable bound");
	TEST_ASSERT(g_xbp->decompress(out, out_len, back, N, &back_len)
		== MEMBRANE_OK && back_len == N && memcmp(in, back, N) == 0,
		"vtable round-trips");
}

int	main(void)
{
	g_xbp = membrane_codec_get(MEMBRANE_CODEC_F16_XOR_BYTEPLANE);
	TEST_ASSERT(g_xbp != NULL, "xor byteplane codec is registered");
	TEST_ASSERT(strcmp(g_xbp->name, "f16xor") == 0, "codec name");
	test_all_predictors();
	test_known_sequence();
	test_all_zero();
	test_monotonic();
	test_random_and_empty();
	test_invalid_args();
	test_corrupted_header();
	test_truncated();
	test_block_raw_fallback();
	test_vtable_roundtrip();
	return (0);
}
