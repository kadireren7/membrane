#include <math.h>
#include <string.h>

#include "membrane/f16convert.h"
#include "workload_gen.h"

/*
 * noise_amp per kind, calibrated (see membrane/quant_select.h's default
 * 0.05 rel-L2 bound) against the real Q4_0 quantize/dequantize engine
 * over a 2000-block sweep at seed 1234, elems=128:
 *
 *   DEFAULT        (1.0):  ~78.5% of blocks clear the Q4 bound
 *   LOW_VARIANCE   (0.1):  ~97.8% clear it (easily Q4-compressible)
 *   HIGH_VARIANCE  (4.0):  ~16.5% clear it (mostly need Q8)
 *   MIXED          (bimodal 0.1/4.0 via a second, decorrelated Weyl
 *                   sequence, not a continuous sweep): ~65.4% clear it
 *
 * These are the actual measured splits this generator produces, not
 * chosen ratios -- rerunning tools/membrane-bench reproduces them.
 */
static const float	NOISE_AMP_DEFAULT = 1.0f;
static const float	NOISE_AMP_LOW_VARIANCE = 0.1f;
static const float	NOISE_AMP_HIGH_VARIANCE = 4.0f;
static const float	BIAS_SCALE = 8.0f;

static uint32_t	xorshift32(uint32_t *state)
{
	uint32_t	x;

	x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return (x);
}

static float	weyl_frac(uint32_t seed, uint32_t blk_idx, uint32_t salt)
{
	double	t;

	t = fmod((double)blk_idx * 0.6180339887498949
			+ (double)((seed + salt) % 997u) / 997.0, 1.0);
	return ((float)t);
}

static float	block_noise_amp(membrane_workload_kind_t kind, uint32_t seed,
					uint32_t blk_idx)
{
	if (kind == MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE)
		return (NOISE_AMP_LOW_VARIANCE);
	if (kind == MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE)
		return (NOISE_AMP_HIGH_VARIANCE);
	if (kind == MEMBRANE_WORKLOAD_SYNTHETIC_MIXED)
	{
		/* A second, decorrelated Weyl sequence (distinct salt) picks
		 * the per-block regime, so MIXED is a genuine bimodal split
		 * -- not the same continuous bias-only sweep as DEFAULT. */
		if (weyl_frac(seed, blk_idx, 97u) < 0.5f)
			return (NOISE_AMP_LOW_VARIANCE);
		return (NOISE_AMP_HIGH_VARIANCE);
	}
	return (NOISE_AMP_DEFAULT);
}

void	membrane_workload_generate_block(membrane_workload_kind_t kind,
			uint32_t seed, uint32_t blk_idx, uint16_t *out, uint32_t elems)
{
	uint32_t	rng;
	float		bias;
	float		noise_amp;
	float		noise;
	float		smooth;
	uint32_t	k;

	rng = (seed * 2654435761u) ^ (blk_idx * 0x9E3779B9u);
	if (rng == 0)
		rng = 1;
	bias = BIAS_SCALE * weyl_frac(seed, blk_idx, 0u);
	noise_amp = block_noise_amp(kind, seed, blk_idx);
	k = 0;
	while (k < elems)
	{
		noise = 2.0f * ((float)(xorshift32(&rng) & 0xFFFFu) / 65535.0f)
			- 1.0f;
		smooth = sinf((float)k * 0.15f + (float)blk_idx * 0.01f);
		out[k] = membrane_f32_to_f16(bias
				+ noise_amp * (0.5f * smooth + 0.4f * noise));
		k++;
	}
}

const char	*membrane_workload_kind_name(membrane_workload_kind_t k)
{
	if (k == MEMBRANE_WORKLOAD_SYNTHETIC_LOW_VARIANCE)
		return ("synthetic-low-variance");
	if (k == MEMBRANE_WORKLOAD_SYNTHETIC_HIGH_VARIANCE)
		return ("synthetic-high-variance");
	if (k == MEMBRANE_WORKLOAD_SYNTHETIC_MIXED)
		return ("synthetic-mixed");
	return ("synthetic-default");
}

int	membrane_workload_kind_from_name(const char *name,
		membrane_workload_kind_t *out)
{
	membrane_workload_kind_t	k;

	if (name == NULL || out == NULL)
		return (0);
	k = MEMBRANE_WORKLOAD_SYNTHETIC_DEFAULT;
	while (k < MEMBRANE_WORKLOAD_COUNT)
	{
		if (strcmp(name, membrane_workload_kind_name(k)) == 0)
			return (*out = k, 1);
		k++;
	}
	return (0);
}

int	membrane_checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (a != 0 && b > UINT64_MAX / a)
		return (0);
	*out = a * b;
	return (1);
}
