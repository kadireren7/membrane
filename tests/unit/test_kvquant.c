#include <math.h>
#include <string.h>

#include "membrane/f16convert.h"
#include "membrane/kvquant.h"
#include "test_helpers.h"

static void	put_u16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

static void	test_all_zero_is_exact(void)
{
	enum { N = 512 };
	uint8_t						in[N];
	membrane_q8_cfg_t			cfg;
	membrane_kv_quant_metrics_t	m;

	memset(in, 0, sizeof(in));
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 64;
	TEST_ASSERT(membrane_kv_quant_compute(in, N, &cfg, &m) == MEMBRANE_OK,
		"compute succeeds");
	TEST_ASSERT(m.decode_ok, "decode succeeds");
	TEST_ASSERT(m.mse == 0.0 && m.mae == 0.0 && m.max_abs_err == 0.0,
		"all-zero block has zero error");
	TEST_ASSERT(m.cosine_similarity == 1.0, "zero vectors are cosine 1.0");
	TEST_ASSERT(m.elements == N / 2 && m.saturated == 0, "sane element/sat count");
}

/* A smooth, moderate-magnitude signal (typical of KV activations) should
 * quantize with high fidelity: cosine close to 1, small relative error. */
static void	test_smooth_signal_high_fidelity(void)
{
	enum { ELEMS = 2048, N = ELEMS * 2 };
	uint8_t						in[N];
	membrane_q8_cfg_t			cfg;
	membrane_kv_quant_metrics_t	m;
	size_t						i;

	i = 0;
	while (i < ELEMS)
	{
		put_u16(in + 2 * i, membrane_f32_to_f16(
				sinf((float)i * 0.01f) * 2.0f));
		i++;
	}
	cfg.mode = MEMBRANE_Q8_SYMMETRIC;
	cfg.group_elems = 32;
	TEST_ASSERT(membrane_kv_quant_compute(in, N, &cfg, &m) == MEMBRANE_OK,
		"compute succeeds");
	TEST_ASSERT(m.decode_ok, "decode succeeds");
	TEST_ASSERT(m.cosine_similarity > 0.999, "smooth signal keeps cosine > 0.999");
	TEST_ASSERT(m.rel_l2_error < 0.05, "smooth signal keeps rel L2 error small");
	TEST_ASSERT(m.encoded_bytes < m.raw_bytes, "quantized output is smaller");
}

static void	test_deterministic(void)
{
	enum { ELEMS = 300, N = ELEMS * 2 };
	uint8_t						in[N];
	membrane_q8_cfg_t			cfg;
	membrane_kv_quant_metrics_t	a;
	membrane_kv_quant_metrics_t	b;
	uint32_t						seed;
	size_t						i;

	seed = 42;
	i = 0;
	while (i < ELEMS)
	{
		put_u16(in + 2 * i, membrane_f32_to_f16(((float)test_rand_next(&seed)
					/ (float)UINT32_MAX - 0.5f) * 6.0f));
		i++;
	}
	cfg.mode = MEMBRANE_Q8_AFFINE;
	cfg.group_elems = 128;
	TEST_ASSERT(membrane_kv_quant_compute(in, N, &cfg, &a) == MEMBRANE_OK,
		"compute run 1");
	TEST_ASSERT(membrane_kv_quant_compute(in, N, &cfg, &b) == MEMBRANE_OK,
		"compute run 2");
	TEST_ASSERT(memcmp(&a, &b, sizeof(a)) == 0,
		"metrics are bit-identical across runs");
}

int	main(void)
{
	test_all_zero_is_exact();
	test_smooth_signal_high_fidelity();
	test_deterministic();
	return (0);
}
