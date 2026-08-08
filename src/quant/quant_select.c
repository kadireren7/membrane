#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "membrane/f16convert.h"
#include "membrane/quant_select.h"

double	membrane_quant_rel_l2_error(const uint16_t *a_f16,
			const uint16_t *b_f16, size_t n)
{
	double	sum_sq_diff;
	double	sum_sq_a;
	size_t	i;
	float	a;
	float	b;

	sum_sq_diff = 0.0;
	sum_sq_a = 0.0;
	i = 0;
	while (i < n)
	{
		a = membrane_f16_to_f32(a_f16[i]);
		if (isfinite(a))
		{
			b = membrane_f16_to_f32(b_f16[i]);
			sum_sq_diff += ((double)b - (double)a) * ((double)b - (double)a);
			sum_sq_a += (double)a * (double)a;
		}
		i++;
	}
	if (sum_sq_a == 0.0)
		return (sum_sq_diff == 0.0 ? 0.0 : 1.0);
	return (sqrt(sum_sq_diff) / sqrt(sum_sq_a));
}

membrane_status_t	membrane_quant_select_precision(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n,
						const membrane_quant_select_cfg_t *cfg,
						membrane_quant_select_result_t *out)
{
	uint8_t				*packed;
	uint16_t			*dec;
	size_t				packed_bytes;
	membrane_status_t	st;

	if (x_f16 == NULL || cfg == NULL || out == NULL || n == 0
			|| n % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (n > SIZE_MAX / sizeof(*dec))
		return (MEMBRANE_ERR_INVALID_ARG);
	packed_bytes = (n / MEMBRANE_QSIMD_BLOCK_ELEMS)
		* MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	packed = malloc(packed_bytes);
	dec = malloc(n * sizeof(uint16_t));
	if (packed == NULL || dec == NULL)
		return (free(packed), free(dec), MEMBRANE_ERR_ALLOC_FAILED);
	st = membrane_simd_q4_0_quantize(backend, x_f16, n, packed);
	if (st != MEMBRANE_OK)
		return (free(packed), free(dec), st);
	st = membrane_simd_q4_0_dequantize(backend, packed, n, dec);
	if (st != MEMBRANE_OK)
		return (free(packed), free(dec), st);
	out->q4_rel_l2_error = membrane_quant_rel_l2_error(x_f16, dec, n);
	if (out->q4_rel_l2_error <= cfg->max_q4_rel_l2_error)
		out->precision = MEMBRANE_PRECISION_Q4;
	else
		out->precision = MEMBRANE_PRECISION_Q8;
	return (free(packed), free(dec), MEMBRANE_OK);
}
