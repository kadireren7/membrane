#include <math.h>
#include <stdlib.h>

#include "membrane/f16convert.h"
#include "membrane/quant_select.h"
#include "test_helpers.h"

enum { N = 128 };

/* Strong constant bias with a tiny ripple: the Q4_0 scale is dominated
 * by the (perfectly-representable-in-relative-terms) bias, so measured
 * rel_l2_error is ~0.0015 -- safely under the default 0.05 bound. */
static void	fill_bias_dominated(uint16_t *x, size_t n)
{
	size_t	i;

	i = 0;
	while (i < n)
	{
		x[i] = membrane_f32_to_f16(50.0f + 0.05f * sinf((float)i * 0.2f));
		i++;
	}
}

/* Zero-mean wide-range pseudo-random values: no bias for the Q4_0 scale
 * to exploit, so measured rel_l2_error is ~0.061 -- safely over the
 * default 0.05 bound. */
static void	fill_zero_mean_wide_range(uint16_t *x, size_t n)
{
	uint32_t	state;
	size_t		i;

	state = 12345;
	i = 0;
	while (i < n)
	{
		x[i] = membrane_f32_to_f16(20.0f * (2.0f
				* ((float)(test_rand_next(&state) & 0xFFFF) / 65535.0f)
				- 1.0f));
		i++;
	}
}

static void	test_bias_dominated_signal_picks_q4(void)
{
	uint16_t						x[N];
	membrane_quant_select_cfg_t	cfg;
	membrane_quant_select_result_t	r;

	fill_bias_dominated(x, N);
	cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, &r) == MEMBRANE_OK, "select succeeds");
	TEST_ASSERT(r.precision == MEMBRANE_PRECISION_Q4,
		"a bias-dominated block fits inside the Q4 error bound");
	TEST_ASSERT(r.q4_rel_l2_error <= cfg.max_q4_rel_l2_error,
		"reported error is actually within the bound it satisfied");
}

static void	test_wide_range_noise_picks_q8(void)
{
	uint16_t						x[N];
	membrane_quant_select_cfg_t	cfg;
	membrane_quant_select_result_t	r;

	fill_zero_mean_wide_range(x, N);
	cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, &r) == MEMBRANE_OK, "select succeeds");
	TEST_ASSERT(r.precision == MEMBRANE_PRECISION_Q8,
		"wide-range near-random values exceed the Q4 error bound");
	TEST_ASSERT(r.q4_rel_l2_error > cfg.max_q4_rel_l2_error,
		"reported error is actually the reason Q8 was chosen");
}

static void	test_threshold_is_honoured(void)
{
	uint16_t						x[N];
	membrane_quant_select_cfg_t	cfg;
	membrane_quant_select_result_t	r;

	fill_zero_mean_wide_range(x, N);
	cfg.max_q4_rel_l2_error = 1.0;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, &r) == MEMBRANE_OK, "select succeeds");
	TEST_ASSERT(r.precision == MEMBRANE_PRECISION_Q4,
		"a wide-open bound accepts even the noisy signal under Q4");
}

static void	test_deterministic(void)
{
	uint16_t						x[N];
	membrane_quant_select_cfg_t	cfg;
	membrane_quant_select_result_t	r1;
	membrane_quant_select_result_t	r2;

	fill_bias_dominated(x, N);
	cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, &r1) == MEMBRANE_OK, "first select succeeds");
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, &r2) == MEMBRANE_OK, "second select succeeds");
	TEST_ASSERT(r1.precision == r2.precision, "same precision chosen twice");
	TEST_ASSERT(r1.q4_rel_l2_error == r2.q4_rel_l2_error,
		"identical input yields bit-identical error twice");
}

static void	test_invalid_args(void)
{
	uint16_t						x[N];
	membrane_quant_select_cfg_t	cfg;
	membrane_quant_select_result_t	r;

	cfg.max_q4_rel_l2_error = MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR;
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, NULL, N,
			&cfg, &r) == MEMBRANE_ERR_INVALID_ARG, "NULL input rejected");
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, 0,
			&cfg, &r) == MEMBRANE_ERR_INVALID_ARG, "zero length rejected");
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, 31,
			&cfg, &r) == MEMBRANE_ERR_INVALID_ARG,
		"length not a multiple of 32 rejected");
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			NULL, &r) == MEMBRANE_ERR_INVALID_ARG, "NULL cfg rejected");
	TEST_ASSERT(membrane_quant_select_precision(MEMBRANE_SIMD_SCALAR, x, N,
			&cfg, NULL) == MEMBRANE_ERR_INVALID_ARG, "NULL out rejected");
}

static void	test_rel_l2_error_helper(void)
{
	uint16_t	a[32];
	uint16_t	b[32];
	size_t		i;

	i = 0;
	while (i < 32)
	{
		a[i] = membrane_f32_to_f16(0.0f);
		b[i] = membrane_f32_to_f16(0.0f);
		i++;
	}
	TEST_ASSERT(membrane_quant_rel_l2_error(a, b, 32) == 0.0,
		"all-zero vs all-zero is zero error");
	a[0] = membrane_f32_to_f16(1.0f);
	b[0] = membrane_f32_to_f16(1.0f);
	TEST_ASSERT(membrane_quant_rel_l2_error(a, a, 32) == 0.0,
		"identical arrays are zero error");
	TEST_ASSERT(membrane_quant_rel_l2_error(a, b, 32) == 0.0,
		"a matches b is zero error");
}

int	main(void)
{
	test_bias_dominated_signal_picks_q4();
	test_wide_range_noise_picks_q8();
	test_threshold_is_honoured();
	test_deterministic();
	test_invalid_args();
	test_rel_l2_error_helper();
	return (0);
}
