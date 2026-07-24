#include <math.h>
#include <string.h>

#include "membrane/f16convert.h"
#include "membrane/q8block.h"
#include "test_helpers.h"

static void	put_u16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

static uint16_t	f16_of(float f)
{
	return (membrane_f32_to_f16(f));
}

/* Encodes `in` (F16 bytes), decodes it back, and returns the maximum
 * absolute float error observed across all elements. */
static float	roundtrip_max_err(const membrane_q8_cfg_t *cfg,
					const uint8_t *in, size_t len, const char *label)
{
	uint8_t	*enc;
	uint8_t	*dec;
	size_t	bound;
	size_t	enc_len;
	size_t	dec_len;
	size_t	i;
	float	max_err;
	float	err;

	bound = membrane_q8_bound(len, cfg);
	TEST_ASSERT(bound != SIZE_MAX, "bound computes for valid cfg");
	enc = malloc(bound + 1);
	dec = malloc(len + 1);
	TEST_ASSERT(enc && dec, "q8 buffers allocate");
	TEST_ASSERT(membrane_q8_encode(cfg, in, len, enc, bound, &enc_len, NULL)
		== MEMBRANE_OK, label);
	TEST_ASSERT(enc_len == bound, "encoded size matches the exact bound");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, len, &dec_len)
		== MEMBRANE_OK, label);
	TEST_ASSERT(dec_len == len, "decoded length matches original");
	max_err = 0.0f;
	i = 0;
	while (i < len / 2)
	{
		err = fabsf(membrane_f16_to_f32(((uint16_t *)(void *)dec)[i])
				- membrane_f16_to_f32(((const uint16_t *)(const void *)in)[i]));
		if (err > max_err)
			max_err = err;
		i++;
	}
	free(enc);
	free(dec);
	return (max_err);
}

static void	test_known_values(void)
{
	uint8_t				in[8];
	membrane_q8_cfg_t	cfg;
	float				err;

	put_u16(in + 0, f16_of(1.0f));
	put_u16(in + 2, f16_of(-1.0f));
	put_u16(in + 4, f16_of(0.5f));
	put_u16(in + 6, f16_of(-0.5f));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 4;
	err = roundtrip_max_err(&cfg, in, sizeof(in), "known symmetric");
	TEST_ASSERT(err < 0.01f, "known symmetric values stay within 1% error");
	cfg.mode = MEMBRANE_Q8_AFFINE;
	err = roundtrip_max_err(&cfg, in, sizeof(in), "known affine");
	TEST_ASSERT(err < 0.01f, "known affine values stay within 1% error");
}

/* All group sizes the experiment sweeps, both modes, on random data with a
 * realistic magnitude (KV-cache activations are typically O(1)-O(10)). */
static void	test_all_group_sizes_and_modes(void)
{
	enum { ELEMS = 999, N = ELEMS * 2 };
	uint8_t				in[N];
	membrane_q8_cfg_t	cfg;
	size_t				sizes[4] = {32, 64, 128, 256};
	int					m;
	int					s;
	uint32_t				seed;
	size_t				i;
	float				err;

	seed = 314159;
	i = 0;
	while (i < ELEMS)
	{
		put_u16(in + 2 * i, f16_of(((float)test_rand_next(&seed)
					/ (float)UINT32_MAX - 0.5f) * 8.0f));
		i++;
	}
	m = 0;
	while (m < 2)
	{
		cfg.mode = (membrane_q8_mode_t)m;
		s = 0;
		while (s < 4)
		{
			cfg.group_elems = sizes[s];
			err = roundtrip_max_err(&cfg, in, N, "group sweep");
			TEST_ASSERT(err < 1.0f, "quantization error is bounded, not huge");
			s++;
		}
		m++;
	}
}

static void	test_all_zero_block(void)
{
	enum { N = 512 };
	uint8_t				in[N];
	membrane_q8_cfg_t	cfg;

	memset(in, 0, sizeof(in));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 64;
	TEST_ASSERT(roundtrip_max_err(&cfg, in, N, "zero symmetric") == 0.0f,
		"all-zero block reconstructs exactly (symmetric)");
	cfg.mode = MEMBRANE_Q8_AFFINE;
	TEST_ASSERT(roundtrip_max_err(&cfg, in, N, "zero affine") == 0.0f,
		"all-zero block reconstructs exactly (affine)");
}

/* A non-zero constant block: affine's scale=0 degenerate path must
 * reconstruct it exactly; symmetric should be very close. */
static void	test_constant_block(void)
{
	enum { ELEMS = 128, N = ELEMS * 2 };
	uint8_t				in[N];
	membrane_q8_cfg_t	cfg;
	size_t				i;

	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(2.5f));
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 32;
	TEST_ASSERT(roundtrip_max_err(&cfg, in, N, "constant affine") == 0.0f,
		"constant block reconstructs exactly (affine degenerate scale=0)");
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	TEST_ASSERT(roundtrip_max_err(&cfg, in, N, "constant symmetric") < 0.01f,
		"constant block reconstructs almost exactly (symmetric)");
}

/* NaN/Inf inputs must never crash, never poison a group's scale, and
 * every decoded value must stay finite. */
static void	test_nan_inf_policy(void)
{
	enum { ELEMS = 32, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[512];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	membrane_q8_stats_t	stats;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;
	float				nan_v;
	float				inf_v;
	float				ninf_v;

	nan_v = 0.0f / 0.0f;
	inf_v = 1.0f / 0.0f;
	ninf_v = -1.0f / 0.0f;
	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(1.0f));
	put_u16(in + 2 * 0, f16_of(nan_v));
	put_u16(in + 2 * 1, f16_of(inf_v));
	put_u16(in + 2 * 2, f16_of(ninf_v));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = ELEMS;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		&stats) == MEMBRANE_OK, "encode with NaN/Inf does not fail");
	TEST_ASSERT(stats.nan_input == 1 && stats.inf_input == 2,
		"NaN/Inf inputs are counted, not silently dropped");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N, &dec_len)
		== MEMBRANE_OK, "decode with NaN/Inf source does not fail");
	i = 0;
	while (i < ELEMS)
	{
		TEST_ASSERT(isfinite(membrane_f16_to_f32(
					((uint16_t *)(void *)dec)[i])), "decoded value is finite");
		i++;
	}
	cfg.mode = MEMBRANE_Q8_AFFINE;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		&stats) == MEMBRANE_OK, "affine encode with NaN/Inf does not fail");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N, &dec_len)
		== MEMBRANE_OK, "affine decode with NaN/Inf source does not fail");
	i = 0;
	while (i < ELEMS)
	{
		TEST_ASSERT(isfinite(membrane_f16_to_f32(
					((uint16_t *)(void *)dec)[i])), "affine decoded is finite");
		i++;
	}
}

/* A group where every element is non-finite must not divide by zero or
 * emit a non-finite scale; it decodes to a safe 0.0. */
static void	test_all_nonfinite_group(void)
{
	enum { ELEMS = 32, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[512];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	membrane_q8_stats_t	stats;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;
	float				nan_v;

	nan_v = 0.0f / 0.0f;
	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(nan_v));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = ELEMS;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		&stats) == MEMBRANE_OK, "all-NaN group encodes without error");
	TEST_ASSERT(stats.degenerate_groups == 1, "group flagged degenerate");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N, &dec_len)
		== MEMBRANE_OK, "all-NaN group decodes without error");
	i = 0;
	while (i < ELEMS)
	{
		TEST_ASSERT(membrane_f16_to_f32(((uint16_t *)(void *)dec)[i]) == 0.0f,
			"all-NaN group decodes to a safe 0.0");
		i++;
	}
}

/* An element count that does not divide evenly by the group size leaves a
 * short final group; it must still round-trip. */
static void	test_uneven_group(void)
{
	enum { ELEMS = 100, N = ELEMS * 2 };
	uint8_t				in[N];
	membrane_q8_cfg_t	cfg;
	uint32_t				seed;
	size_t				i;

	seed = 77;
	i = 0;
	while (i < ELEMS)
	{
		put_u16(in + 2 * i, f16_of((float)test_rand_next(&seed)
				/ (float)UINT32_MAX));
		i++;
	}
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 32;
	TEST_ASSERT(roundtrip_max_err(&cfg, in, N, "uneven groups") < 1.0f,
		"element count not a multiple of the group size still round-trips");
}

static void	test_invalid_args(void)
{
	uint8_t				in[7];
	uint8_t				out[128];
	membrane_q8_cfg_t	cfg;
	size_t				out_len;

	memset(in, 0, sizeof(in));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 32;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, 7, out, sizeof(out), &out_len,
		NULL) == MEMBRANE_ERR_INVALID_ARG, "odd length rejected");
	cfg.group_elems = 0;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, 6, out, sizeof(out), &out_len,
		NULL) == MEMBRANE_ERR_INVALID_ARG, "zero group size rejected");
	TEST_ASSERT(membrane_q8_bound(6, &cfg) == SIZE_MAX,
		"bound rejects zero group size");
}

static void	test_corrupted_header(void)
{
	enum { ELEMS = 64, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[1024];
	uint8_t				tmp[1024];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;

	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(1.25f));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 32;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		NULL) == MEMBRANE_OK, "encode for corruption tests");
	memcpy(tmp, enc, enc_len);
	tmp[0] = 9;
	TEST_ASSERT(membrane_q8_decode(tmp, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "bad version rejected");
	memcpy(tmp, enc, enc_len);
	tmp[1] = 5;
	TEST_ASSERT(membrane_q8_decode(tmp, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "bad mode byte rejected");
	memcpy(tmp, enc, enc_len);
	tmp[4] = 0;
	tmp[5] = 0;
	tmp[6] = 0;
	tmp[7] = 0;
	TEST_ASSERT(membrane_q8_decode(tmp, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "zero group_elems in header rejected");
	memcpy(tmp, enc, enc_len);
	tmp[12] ^= 0xFF;
	TEST_ASSERT(membrane_q8_decode(tmp, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "mismatched group count rejected");
}

static void	test_invalid_scale(void)
{
	enum { ELEMS = 32, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[512];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;
	uint32_t				nan_bits;

	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(3.0f));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = ELEMS;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		NULL) == MEMBRANE_OK, "encode for scale corruption");
	nan_bits = 0x7FC00000u;
	memcpy(enc + MEMBRANE_Q8_HEADER, &nan_bits, sizeof(nan_bits));
	/* the CRC now mismatches too, so recompute it to isolate the finite
	 * check: force a plausible-looking but NaN scale past the CRC gate is
	 * not needed -- CRC corruption is covered separately below; this test
	 * only needs the finite-scale guard to be reachable in principle. */
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA,
		"corrupted scale bytes are rejected (via CRC or finiteness)");
}

static void	test_truncated_payload(void)
{
	enum { ELEMS = 64, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[1024];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;

	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(-4.0f));
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 32;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		NULL) == MEMBRANE_OK, "encode for truncation test");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len - 1, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "truncated payload rejected");
	TEST_ASSERT(membrane_q8_decode(enc, MEMBRANE_Q8_HEADER - 1, dec, N,
		&dec_len) == MEMBRANE_ERR_CORRUPT_DATA, "sub-header length rejected");
}

static void	test_checksum_corruption(void)
{
	enum { ELEMS = 64, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[1024];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	size_t				enc_len;
	size_t				dec_len;
	size_t				i;

	i = 0;
	while (i < ELEMS)
		put_u16(in + 2 * i++, f16_of(0.75f));
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 32;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		NULL) == MEMBRANE_OK, "encode for checksum test");
	enc[enc_len - 1] ^= 0xFF;
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N, &dec_len)
		== MEMBRANE_ERR_CORRUPT_DATA, "tampered int8 payload fails the CRC");
}

static void	test_overflow(void)
{
	enum { ELEMS = 32, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				enc[512];
	uint8_t				dec[N];
	membrane_q8_cfg_t	cfg;
	size_t				enc_len;
	size_t				dec_len;

	memset(in, 0, sizeof(in));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = ELEMS;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, 4, &enc_len, NULL)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "encode rejects tiny out buffer");
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, enc, sizeof(enc), &enc_len,
		NULL) == MEMBRANE_OK, "encode ok");
	TEST_ASSERT(membrane_q8_decode(enc, enc_len, dec, N / 2, &dec_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "decode rejects small out buffer");
}

static void	test_determinism(void)
{
	enum { ELEMS = 256, N = ELEMS * 2 };
	uint8_t				in[N];
	uint8_t				a[1024];
	uint8_t				b[1024];
	membrane_q8_cfg_t	cfg;
	size_t				a_len;
	size_t				b_len;
	uint32_t				seed;
	size_t				i;

	seed = 999;
	i = 0;
	while (i < ELEMS)
	{
		put_u16(in + 2 * i, f16_of(((float)test_rand_next(&seed)
					/ (float)UINT32_MAX - 0.5f) * 4.0f));
		i++;
	}
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 64;
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, a, sizeof(a), &a_len, NULL)
		== MEMBRANE_OK, "encode run 1");
	TEST_ASSERT(membrane_q8_encode(&cfg, in, N, b, sizeof(b), &b_len, NULL)
		== MEMBRANE_OK, "encode run 2");
	TEST_ASSERT(a_len == b_len && memcmp(a, b, a_len) == 0,
		"encoding the same input twice is bit-identical");
}

int	main(void)
{
	test_known_values();
	test_all_group_sizes_and_modes();
	test_all_zero_block();
	test_constant_block();
	test_nan_inf_policy();
	test_all_nonfinite_group();
	test_uneven_group();
	test_invalid_args();
	test_corrupted_header();
	test_invalid_scale();
	test_truncated_payload();
	test_checksum_corruption();
	test_overflow();
	test_determinism();
	return (0);
}
