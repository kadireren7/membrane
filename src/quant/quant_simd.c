/*
 * Phase 5.1: portable, MEMBRANE-owned scalar+SIMD Q8_0/Q4_0
 * quantize/dequantize engine. Every backend implements the EXACT same
 * formula ggml itself uses (read from the pinned llama.cpp commit,
 * documented in docs/phase5-quant-engine.md and cross-referenced
 * against docs/phase4-ggml-quant-parity.md):
 *
 *   Q8_0 quantize: amax = max(|x|) over the 32-element block;
 *     d = amax/127; id = amax!=0 ? 127/amax : 0 (NOT 1/d -- ggml's own
 *     AVX2 CPU-backend path computes the reciprocal directly from amax,
 *     a different floating-point operation than 1/d that can round
 *     differently in the last bit; replicated exactly here);
 *     qs[j] = (int8_t)rint(x[j]*id) -- round-to-nearest-even (matches
 *     ggml's _mm256_round_ps(..., _MM_ROUND_NEAREST) exactly; C's
 *     roundf() would be WRONG here, it rounds half-away-from-zero).
 *   Q8_0 dequantize: x[j] = qs[j]*d. Trivial, no rounding, safe to
 *     vectorize with zero parity risk.
 *   Q4_0 quantize: amax/max found via a SEQUENTIAL scalar scan that
 *     keeps the SIGNED value at the first-seen maximal-|.| position
 *     (ggml's `if (amax < fabsf(v))` is a strict less-than, so ties
 *     keep whichever element came first) -- kept scalar in EVERY
 *     backend here, per item 4's explicit allowance ("if not possible,
 *     leave that step scalar"): a differently-ordered SIMD reduction
 *     could pick a different tied element and diverge the sign of `d`.
 *     d = max/-8; id = d!=0 ? 1/d : 0; per pair:
 *     qi = (int8_t)(x*id + 8.5f) (a TRUNCATING cast, not a round --
 *     this is how ggml's own scalar formula implements rounding here),
 *     xi = (uint8_t)MIN(15, qi) (note: no lower clamp -- a value that
 *     legitimately wraps in ggml wraps here too, replicated exactly).
 *   Q4_0 dequantize: trivial, safe to vectorize.
 *
 * SSE4.1/AVX2 accelerate the safe-to-vectorize steps (amax reduction,
 * element-wise scale+round, dequantize) via GCC/Clang function target
 * attributes (one translation unit, ISA-tagged functions, no separate
 * compile flags needed) and runtime dispatch via __builtin_cpu_supports
 * -- never executes an instruction the running CPU doesn't have.
 */
#include "membrane/quant_simd.h"
#include "membrane/f16convert.h"

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
# include <immintrin.h>
# define MEMBRANE_QSIMD_X86 1
#endif

/* ------------------------------------------------------------------ */
/* Backend detection (item 3): cached in a function-local static --    */
/* not a mutable global the caller can observe/race on, and idempotent */
/* regardless of which thread computes it first.                       */
/* ------------------------------------------------------------------ */

membrane_simd_backend_t	membrane_simd_best_backend(void)
{
	static membrane_simd_backend_t	cached = MEMBRANE_SIMD_COUNT;

	if (cached != MEMBRANE_SIMD_COUNT)
		return (cached);
#if defined(MEMBRANE_QSIMD_X86)
	__builtin_cpu_init();
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
		cached = MEMBRANE_SIMD_AVX2;
	else if (__builtin_cpu_supports("sse4.1"))
		cached = MEMBRANE_SIMD_SSE41;
	else
		cached = MEMBRANE_SIMD_SCALAR;
#else
	cached = MEMBRANE_SIMD_SCALAR;
#endif
	return (cached);
}

const char	*membrane_simd_backend_name(membrane_simd_backend_t b)
{
	if (b == MEMBRANE_SIMD_AVX2)
		return ("avx2");
	if (b == MEMBRANE_SIMD_SSE41)
		return ("sse4.1");
	return ("scalar");
}

static bool	backend_supported(membrane_simd_backend_t b)
{
	if (b == MEMBRANE_SIMD_SCALAR)
		return (true);
#if defined(MEMBRANE_QSIMD_X86)
	__builtin_cpu_init();
	if (b == MEMBRANE_SIMD_SSE41)
		return (__builtin_cpu_supports("sse4.1"));
	if (b == MEMBRANE_SIMD_AVX2)
		return (__builtin_cpu_supports("avx2")
			&& __builtin_cpu_supports("fma"));
#endif
	return (false);
}

/* ------------------------------------------------------------------ */
/* Scalar core -- canonical, always-correct, matches ggml exactly.      */
/* ------------------------------------------------------------------ */

/* x86's CVTPS2DQ (what every backend here ultimately uses to turn a
 * rounded float into an integer, whether via an explicit intrinsic or
 * via the C compiler's own float-to-int cast) maps a NaN or
 * out-of-int32-range input to the "integer indefinite" sentinel
 * (INT32_MIN, 0x80000000), NOT to whatever the low bits of the float
 * happen to be. ggml's own AVX2 Q8_0 quantizer then feeds that through
 * two SATURATING packs (int32->int16->int8), so the sentinel always
 * lands on INT8_MIN (-128) -- verified empirically against
 * membrane_ggml_quant for NaN, +Inf, and -Inf, all three producing -128
 * (see docs/phase5-quant-engine.md). A plain C truncating cast
 * ((int8_t)some_int32) does NOT reproduce this -- it takes the low
 * byte of whatever the conversion produced, which is a real,
 * discovered-by-testing bit-parity bug this function exists to close,
 * not a hypothetical. */
static int8_t	sat_i8_from_rounded_f32(float r)
{
	int32_t	iv;

	if (!isfinite(r) || r >= 2147483648.0f || r < -2147483648.0f)
		return (INT8_MIN);
	iv = (int32_t)r;
	if (iv > 127)
		return (127);
	if (iv < -128)
		return (-128);
	return ((int8_t)iv);
}

static void	q8_0_quant_block_scalar(const float *x, uint8_t *out)
{
	float	amax;
	float	d;
	float	id;
	uint16_t	dh;
	int8_t	*qs;
	int		j;

	amax = 0.0f;
	j = 0;
	while (j < 32)
	{
		float	v = fabsf(x[j]);

		if (v > amax)
			amax = v;
		j++;
	}
	d = amax / 127.0f;
	id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
	dh = membrane_f32_to_f16(d);
	memcpy(out, &dh, 2);
	qs = (int8_t *)(void *)(out + 2);
	j = 0;
	while (j < 32)
	{
		qs[j] = sat_i8_from_rounded_f32(rintf(x[j] * id));
		j++;
	}
}

static void	q8_0_dequant_block_scalar(const uint8_t *in, float *out)
{
	uint16_t		dh;
	float			d;
	const int8_t	*qs;
	int				j;

	memcpy(&dh, in, 2);
	d = membrane_f16_to_f32(dh);
	qs = (const int8_t *)(const void *)(in + 2);
	j = 0;
	while (j < 32)
	{
		out[j] = (float)qs[j] * d;
		j++;
	}
}

static void	q4_0_quant_block_scalar(const float *x, uint8_t *out)
{
	float	amax;
	float	mx;
	float	d;
	float	id;
	uint16_t	dh;
	uint8_t	*qs;
	int		j;

	amax = 0.0f;
	mx = 0.0f;
	j = 0;
	while (j < 32)
	{
		float	v = x[j];

		if (amax < fabsf(v))
		{
			amax = fabsf(v);
			mx = v;
		}
		j++;
	}
	d = mx / -8.0f;
	id = (d != 0.0f) ? 1.0f / d : 0.0f;
	dh = membrane_f32_to_f16(d);
	memcpy(out, &dh, 2);
	qs = out + 2;
	j = 0;
	while (j < 16)
	{
		float	x0 = x[j] * id;
		float	x1 = x[j + 16] * id;
		int8_t	qi0 = (int8_t)(x0 + 8.5f);
		int8_t	qi1 = (int8_t)(x1 + 8.5f);
		int		m0 = (15 < (int)qi0) ? 15 : (int)qi0;
		int		m1 = (15 < (int)qi1) ? 15 : (int)qi1;
		uint8_t	xi0 = (uint8_t)m0;
		uint8_t	xi1 = (uint8_t)m1;

		qs[j] = xi0 | (uint8_t)(xi1 << 4);
		j++;
	}
}

static void	q4_0_dequant_block_scalar(const uint8_t *in, float *out)
{
	uint16_t		dh;
	float			d;
	const uint8_t	*qs;
	int				j;

	memcpy(&dh, in, 2);
	d = membrane_f16_to_f32(dh);
	qs = in + 2;
	j = 0;
	while (j < 16)
	{
		int	x0 = (qs[j] & 0x0F) - 8;
		int	x1 = (qs[j] >> 4) - 8;

		out[j] = (float)x0 * d;
		out[j + 16] = (float)x1 * d;
		j++;
	}
}

/* ------------------------------------------------------------------ */
/* SSE4.1 -- accelerates amax reduction, element scale+round, and both  */
/* dequantize paths. Q4_0's signed-argmax scan stays scalar (see file   */
/* header comment).                                                     */
/* ------------------------------------------------------------------ */

#if defined(MEMBRANE_QSIMD_X86)

/* cvtps_epi32 itself already produces the INT32_MIN "integer
 * indefinite" sentinel for NaN/out-of-range inputs (that part of the
 * hardware behavior needs no help); only the final int32->int8
 * saturating narrow needs to be done explicitly here, matching
 * sat_i8_from_rounded_f32's float-side handling above. */
static int8_t	sat_i8_from_i32(int32_t iv)
{
	if (iv > 127)
		return (127);
	if (iv < -128)
		return (-128);
	return ((int8_t)iv);
}

__attribute__((target("sse4.1")))
static float	hmax_sse41(__m128 v)
{
	__m128	shuf;
	__m128	mx;

	shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
	mx = _mm_max_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, mx);
	mx = _mm_max_ss(mx, shuf);
	return (_mm_cvtss_f32(mx));
}

__attribute__((target("sse4.1")))
static void	q8_0_quant_block_sse41(const float *x, uint8_t *out)
{
	__m128			absmask;
	__m128			v0;
	__m128			v1;
	__m128			v2;
	__m128			v3;
	__m128			mx;
	float			amax;
	float			d;
	float			id;
	uint16_t		dh;
	int8_t			*qs;
	int				j;

	absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	v0 = _mm_and_ps(_mm_loadu_ps(x), absmask);
	v1 = _mm_and_ps(_mm_loadu_ps(x + 4), absmask);
	v2 = _mm_and_ps(_mm_loadu_ps(x + 8), absmask);
	v3 = _mm_and_ps(_mm_loadu_ps(x + 12), absmask);
	mx = _mm_max_ps(_mm_max_ps(v0, v1), _mm_max_ps(v2, v3));
	v0 = _mm_and_ps(_mm_loadu_ps(x + 16), absmask);
	v1 = _mm_and_ps(_mm_loadu_ps(x + 20), absmask);
	v2 = _mm_and_ps(_mm_loadu_ps(x + 24), absmask);
	v3 = _mm_and_ps(_mm_loadu_ps(x + 28), absmask);
	mx = _mm_max_ps(mx, _mm_max_ps(_mm_max_ps(v0, v1), _mm_max_ps(v2, v3)));
	amax = hmax_sse41(mx);
	d = amax / 127.0f;
	id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
	dh = membrane_f32_to_f16(d);
	memcpy(out, &dh, 2);
	qs = (int8_t *)(void *)(out + 2);
	{
		__m128	idv = _mm_set1_ps(id);

		j = 0;
		while (j < 32)
		{
			__m128	xv = _mm_loadu_ps(x + j);
			__m128	scaled = _mm_mul_ps(xv, idv);
			__m128	rounded = _mm_round_ps(scaled,
					_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			__m128i	iv = _mm_cvtps_epi32(rounded);
			int32_t	tmp[4];

			_mm_storeu_si128((__m128i *)(void *)tmp, iv);
			qs[j] = sat_i8_from_i32(tmp[0]);
			qs[j + 1] = sat_i8_from_i32(tmp[1]);
			qs[j + 2] = sat_i8_from_i32(tmp[2]);
			qs[j + 3] = sat_i8_from_i32(tmp[3]);
			j += 4;
		}
	}
}

__attribute__((target("sse4.1")))
static void	q8_0_dequant_block_sse41(const uint8_t *in, float *out)
{
	uint16_t		dh;
	float			d;
	const int8_t	*qs;
	__m128			dv;
	int				j;

	memcpy(&dh, in, 2);
	d = membrane_f16_to_f32(dh);
	dv = _mm_set1_ps(d);
	qs = (const int8_t *)(const void *)(in + 2);
	j = 0;
	while (j < 32)
	{
		__m128i	bytes = _mm_cvtsi32_si128(
				(int32_t)((uint8_t)qs[j] | ((uint8_t)qs[j + 1] << 8)
					| ((uint32_t)(uint8_t)qs[j + 2] << 16)
					| ((uint32_t)(uint8_t)qs[j + 3] << 24)));
		__m128i	i32 = _mm_cvtepi8_epi32(bytes);
		__m128	f32 = _mm_cvtepi32_ps(i32);
		__m128	res = _mm_mul_ps(f32, dv);

		_mm_storeu_ps(out + j, res);
		j += 4;
	}
}

__attribute__((target("sse4.1")))
static void	q4_0_dequant_block_sse41(const uint8_t *in, float *out)
{
	q4_0_dequant_block_scalar(in, out);
}

__attribute__((target("sse4.1")))
static void	q4_0_quant_block_sse41(const float *x, uint8_t *out)
{
	q4_0_quant_block_scalar(x, out);
}

/* ------------------------------------------------------------------ */
/* AVX2 -- same steps, wider vectors.                                    */
/* ------------------------------------------------------------------ */

__attribute__((target("avx2,fma")))
static float	hmax_avx2(__m256 v)
{
	__m128	lo = _mm256_castps256_ps128(v);
	__m128	hi = _mm256_extractf128_ps(v, 1);
	__m128	mx = _mm_max_ps(lo, hi);
	__m128	shuf = _mm_shuffle_ps(mx, mx, _MM_SHUFFLE(2, 3, 0, 1));
	__m128	mx2 = _mm_max_ps(mx, shuf);
	__m128	shuf2 = _mm_movehl_ps(shuf, mx2);
	__m128	mx3 = _mm_max_ss(mx2, shuf2);

	return (_mm_cvtss_f32(mx3));
}

__attribute__((target("avx2,fma")))
static void	q8_0_quant_block_avx2(const float *x, uint8_t *out)
{
	__m256			absmask;
	__m256			v0;
	__m256			v1;
	__m256			v2;
	__m256			v3;
	__m256			mx;
	float			amax;
	float			d;
	float			id;
	uint16_t		dh;
	int8_t			*qs;

	absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	v0 = _mm256_and_ps(_mm256_loadu_ps(x), absmask);
	v1 = _mm256_and_ps(_mm256_loadu_ps(x + 8), absmask);
	v2 = _mm256_and_ps(_mm256_loadu_ps(x + 16), absmask);
	v3 = _mm256_and_ps(_mm256_loadu_ps(x + 24), absmask);
	mx = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
	amax = hmax_avx2(mx);
	d = amax / 127.0f;
	id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
	dh = membrane_f32_to_f16(d);
	memcpy(out, &dh, 2);
	qs = (int8_t *)(void *)(out + 2);
	{
		__m256	idv = _mm256_set1_ps(id);
		int		j = 0;

		while (j < 32)
		{
			__m256	xv = _mm256_loadu_ps(x + j);
			__m256	scaled = _mm256_mul_ps(xv, idv);
			__m256	rounded = _mm256_round_ps(scaled,
					_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			__m256i	iv = _mm256_cvtps_epi32(rounded);
			int32_t	tmp[8];
			int		k;

			_mm256_storeu_si256((__m256i *)(void *)tmp, iv);
			k = 0;
			while (k < 8)
			{
				qs[j + k] = sat_i8_from_i32(tmp[k]);
				k++;
			}
			j += 8;
		}
	}
}

__attribute__((target("avx2,fma")))
static void	q8_0_dequant_block_avx2(const uint8_t *in, float *out)
{
	uint16_t		dh;
	float			d;
	const int8_t	*qs;
	__m256			dv;

	memcpy(&dh, in, 2);
	d = membrane_f16_to_f32(dh);
	dv = _mm256_set1_ps(d);
	qs = (const int8_t *)(const void *)(in + 2);
	{
		__m128i	b0 = _mm_loadl_epi64((const __m128i *)(const void *)qs);
		__m256i	i32lo = _mm256_cvtepi8_epi32(b0);
		__m256	f32lo = _mm256_cvtepi32_ps(i32lo);
		__m256	reslo = _mm256_mul_ps(f32lo, dv);
		__m128i	b1 = _mm_loadl_epi64(
				(const __m128i *)(const void *)(qs + 8));
		__m256i	i32hi = _mm256_cvtepi8_epi32(b1);
		__m256	f32hi = _mm256_cvtepi32_ps(i32hi);
		__m256	reshi = _mm256_mul_ps(f32hi, dv);
		__m128i	b2 = _mm_loadl_epi64(
				(const __m128i *)(const void *)(qs + 16));
		__m256i	i32lo2 = _mm256_cvtepi8_epi32(b2);
		__m256	f32lo2 = _mm256_cvtepi32_ps(i32lo2);
		__m256	reslo2 = _mm256_mul_ps(f32lo2, dv);
		__m128i	b3 = _mm_loadl_epi64(
				(const __m128i *)(const void *)(qs + 24));
		__m256i	i32hi2 = _mm256_cvtepi8_epi32(b3);
		__m256	f32hi2 = _mm256_cvtepi32_ps(i32hi2);
		__m256	reshi2 = _mm256_mul_ps(f32hi2, dv);

		_mm256_storeu_ps(out, reslo);
		_mm256_storeu_ps(out + 8, reshi);
		_mm256_storeu_ps(out + 16, reslo2);
		_mm256_storeu_ps(out + 24, reshi2);
	}
}

__attribute__((target("avx2,fma")))
static void	q4_0_dequant_block_avx2(const uint8_t *in, float *out)
{
	q4_0_dequant_block_scalar(in, out);
}

__attribute__((target("avx2,fma")))
static void	q4_0_quant_block_avx2(const float *x, uint8_t *out)
{
	q4_0_quant_block_scalar(x, out);
}

#endif /* MEMBRANE_QSIMD_X86 */

/* ------------------------------------------------------------------ */
/* Dispatch + whole-array driving loop.                                 */
/* ------------------------------------------------------------------ */

typedef void (*qblock_quant_fn)(const float *, uint8_t *);
typedef void (*qblock_dequant_fn)(const uint8_t *, float *);

static qblock_quant_fn	pick_q8_quant(membrane_simd_backend_t b)
{
#if defined(MEMBRANE_QSIMD_X86)
	if (b == MEMBRANE_SIMD_AVX2)
		return (q8_0_quant_block_avx2);
	if (b == MEMBRANE_SIMD_SSE41)
		return (q8_0_quant_block_sse41);
#endif
	(void)b;
	return (q8_0_quant_block_scalar);
}

static qblock_dequant_fn	pick_q8_dequant(membrane_simd_backend_t b)
{
#if defined(MEMBRANE_QSIMD_X86)
	if (b == MEMBRANE_SIMD_AVX2)
		return (q8_0_dequant_block_avx2);
	if (b == MEMBRANE_SIMD_SSE41)
		return (q8_0_dequant_block_sse41);
#endif
	(void)b;
	return (q8_0_dequant_block_scalar);
}

static qblock_quant_fn	pick_q4_quant(membrane_simd_backend_t b)
{
#if defined(MEMBRANE_QSIMD_X86)
	if (b == MEMBRANE_SIMD_AVX2)
		return (q4_0_quant_block_avx2);
	if (b == MEMBRANE_SIMD_SSE41)
		return (q4_0_quant_block_sse41);
#endif
	(void)b;
	return (q4_0_quant_block_scalar);
}

static qblock_dequant_fn	pick_q4_dequant(membrane_simd_backend_t b)
{
#if defined(MEMBRANE_QSIMD_X86)
	if (b == MEMBRANE_SIMD_AVX2)
		return (q4_0_dequant_block_avx2);
	if (b == MEMBRANE_SIMD_SSE41)
		return (q4_0_dequant_block_sse41);
#endif
	(void)b;
	return (q4_0_dequant_block_scalar);
}

static membrane_status_t	check_common(membrane_simd_backend_t backend,
								const void *a, const void *b, size_t n)
{
	if (!backend_supported(backend))
		return (MEMBRANE_ERR_UNIMPLEMENTED);
	if (a == NULL || b == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (n == 0 || n % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q8_0_quantize(membrane_simd_backend_t
						backend, const uint16_t *x_f16, size_t n,
						uint8_t *out)
{
	membrane_status_t	st;
	qblock_quant_fn		fn;
	size_t				nb;
	size_t				i;
	float				block[32];

	st = check_common(backend, x_f16, out, n);
	if (st != MEMBRANE_OK)
		return (st);
	fn = pick_q8_quant(backend);
	nb = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	i = 0;
	while (i < nb)
	{
		int	j;

		j = 0;
		while (j < 32)
		{
			block[j] = membrane_f16_to_f32(x_f16[i * 32 + (size_t)j]);
			j++;
		}
		fn(block, out + i * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q8_0_dequantize(membrane_simd_backend_t
						backend, const uint8_t *in, size_t n,
						uint16_t *out_f16)
{
	membrane_status_t	st;
	qblock_dequant_fn	fn;
	size_t				nb;
	size_t				i;
	float				block[32];

	st = check_common(backend, in, out_f16, n);
	if (st != MEMBRANE_OK)
		return (st);
	fn = pick_q8_dequant(backend);
	nb = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	i = 0;
	while (i < nb)
	{
		int	j;

		fn(in + i * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES, block);
		j = 0;
		while (j < 32)
		{
			out_f16[i * 32 + (size_t)j] = membrane_f32_to_f16(block[j]);
			j++;
		}
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q4_0_quantize(membrane_simd_backend_t
						backend, const uint16_t *x_f16, size_t n,
						uint8_t *out)
{
	membrane_status_t	st;
	qblock_quant_fn		fn;
	size_t				nb;
	size_t				i;
	float				block[32];

	st = check_common(backend, x_f16, out, n);
	if (st != MEMBRANE_OK)
		return (st);
	fn = pick_q4_quant(backend);
	nb = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	i = 0;
	while (i < nb)
	{
		int	j;

		j = 0;
		while (j < 32)
		{
			block[j] = membrane_f16_to_f32(x_f16[i * 32 + (size_t)j]);
			j++;
		}
		fn(block, out + i * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES);
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q4_0_dequantize(membrane_simd_backend_t
						backend, const uint8_t *in, size_t n,
						uint16_t *out_f16)
{
	membrane_status_t	st;
	qblock_dequant_fn	fn;
	size_t				nb;
	size_t				i;
	float				block[32];

	st = check_common(backend, in, out_f16, n);
	if (st != MEMBRANE_OK)
		return (st);
	fn = pick_q4_dequant(backend);
	nb = n / MEMBRANE_QSIMD_BLOCK_ELEMS;
	i = 0;
	while (i < nb)
	{
		int	j;

		fn(in + i * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES, block);
		j = 0;
		while (j < 32)
		{
			out_f16[i * 32 + (size_t)j] = membrane_f32_to_f16(block[j]);
			j++;
		}
		i++;
	}
	return (MEMBRANE_OK);
}

/* ------------------------------------------------------------------ */
/* Batch API (item 5): loops rows, using the caller's scratch buffer as */
/* the per-row F32 intermediate instead of allocating one per call.     */
/* ------------------------------------------------------------------ */

size_t	membrane_simd_batch_scratch_bytes(size_t elems_per_row)
{
	return (elems_per_row * sizeof(float));
}

membrane_status_t	membrane_simd_q8_0_quantize_batch(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						void *scratch, size_t scratch_bytes)
{
	size_t	row_blocks;
	size_t	packed_row_bytes;
	size_t	r;

	if (elems_per_row == 0 || elems_per_row % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (scratch_bytes < membrane_simd_batch_scratch_bytes(elems_per_row))
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (x_f16 == NULL || packed == NULL || scratch == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	(void)scratch;
	row_blocks = elems_per_row / MEMBRANE_QSIMD_BLOCK_ELEMS;
	packed_row_bytes = row_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES;
	r = 0;
	while (r < n_blocks)
	{
		membrane_status_t	st;

		st = membrane_simd_q8_0_quantize(backend, x_f16 + r * elems_per_row,
				elems_per_row, packed + r * packed_row_bytes);
		if (st != MEMBRANE_OK)
			return (st);
		r++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q8_0_dequantize_batch(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						void *scratch, size_t scratch_bytes)
{
	size_t	row_blocks;
	size_t	packed_row_bytes;
	size_t	r;

	if (elems_per_row == 0 || elems_per_row % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (scratch_bytes < membrane_simd_batch_scratch_bytes(elems_per_row))
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (packed == NULL || out_f16 == NULL || scratch == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	(void)scratch;
	row_blocks = elems_per_row / MEMBRANE_QSIMD_BLOCK_ELEMS;
	packed_row_bytes = row_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES;
	r = 0;
	while (r < n_blocks)
	{
		membrane_status_t	st;

		st = membrane_simd_q8_0_dequantize(backend,
				packed + r * packed_row_bytes, elems_per_row,
				out_f16 + r * elems_per_row);
		if (st != MEMBRANE_OK)
			return (st);
		r++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q4_0_quantize_batch(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						void *scratch, size_t scratch_bytes)
{
	size_t	row_blocks;
	size_t	packed_row_bytes;
	size_t	r;

	if (elems_per_row == 0 || elems_per_row % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (scratch_bytes < membrane_simd_batch_scratch_bytes(elems_per_row))
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (x_f16 == NULL || packed == NULL || scratch == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	(void)scratch;
	row_blocks = elems_per_row / MEMBRANE_QSIMD_BLOCK_ELEMS;
	packed_row_bytes = row_blocks * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	r = 0;
	while (r < n_blocks)
	{
		membrane_status_t	st;

		st = membrane_simd_q4_0_quantize(backend, x_f16 + r * elems_per_row,
				elems_per_row, packed + r * packed_row_bytes);
		if (st != MEMBRANE_OK)
			return (st);
		r++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_simd_q4_0_dequantize_batch(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						void *scratch, size_t scratch_bytes)
{
	size_t	row_blocks;
	size_t	packed_row_bytes;
	size_t	r;

	if (elems_per_row == 0 || elems_per_row % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (scratch_bytes < membrane_simd_batch_scratch_bytes(elems_per_row))
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (packed == NULL || out_f16 == NULL || scratch == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	(void)scratch;
	row_blocks = elems_per_row / MEMBRANE_QSIMD_BLOCK_ELEMS;
	packed_row_bytes = row_blocks * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES;
	r = 0;
	while (r < n_blocks)
	{
		membrane_status_t	st;

		st = membrane_simd_q4_0_dequantize(backend,
				packed + r * packed_row_bytes, elems_per_row,
				out_f16 + r * elems_per_row);
		if (st != MEMBRANE_OK)
			return (st);
		r++;
	}
	return (MEMBRANE_OK);
}

/* ------------------------------------------------------------------ */
/* Parallel worker pool (item 7). Rows are fully independent, so each   */
/* worker just runs the existing serial batch call over its own,        */
/* disjoint, contiguous row range -- no shared mutable state, output    */
/* is bit-identical regardless of how many threads ran it.              */
/* ------------------------------------------------------------------ */

unsigned int	membrane_simd_default_threads(void)
{
	long	n;

	n = sysconf(_SC_NPROCESSORS_ONLN);
	if (n < 1)
		return (1);
	return ((unsigned int)n);
}

typedef enum e_qpool_op
{
	QPOOL_Q8_QUANT,
	QPOOL_Q8_DEQUANT,
	QPOOL_Q4_QUANT,
	QPOOL_Q4_DEQUANT
}	qpool_op_t;

typedef struct s_qpool_job
{
	qpool_op_t					op;
	membrane_simd_backend_t	backend;
	const void					*src;
	void						*dst;
	size_t						row_start;
	size_t						row_count;
	size_t						elems_per_row;
	membrane_status_t			result;
}	qpool_job_t;

static size_t	qpool_row_bytes_packed(qpool_op_t op, size_t elems_per_row)
{
	size_t	row_blocks;

	row_blocks = elems_per_row / MEMBRANE_QSIMD_BLOCK_ELEMS;
	if (op == QPOOL_Q8_QUANT || op == QPOOL_Q8_DEQUANT)
		return (row_blocks * MEMBRANE_QSIMD_Q8_0_BLOCK_BYTES);
	return (row_blocks * MEMBRANE_QSIMD_Q4_0_BLOCK_BYTES);
}

/* Each worker's per-row F32 scratch, sized to cover every row width
 * this project's benchmark matrix uses (up to 4096 elements = 16384
 * bytes); membrane_simd_batch_scratch_bytes() is still the source of
 * truth callers should query for their own elems_per_row. */
# define MEMBRANE_QPOOL_WORKER_SCRATCH_BYTES	(4096u * sizeof(float))

static void	*qpool_worker(void *arg)
{
	qpool_job_t	*j;
	uint8_t		scratch[MEMBRANE_QPOOL_WORKER_SCRATCH_BYTES];
	size_t		packed_row_bytes;
	size_t		f16_off;
	size_t		packed_off;

	j = (qpool_job_t *)arg;
	packed_row_bytes = qpool_row_bytes_packed(j->op, j->elems_per_row);
	f16_off = j->row_start * j->elems_per_row;
	packed_off = j->row_start * packed_row_bytes;
	if (j->op == QPOOL_Q8_QUANT)
		j->result = membrane_simd_q8_0_quantize_batch(j->backend,
				(const uint16_t *)j->src + f16_off, j->row_count,
				j->elems_per_row, (uint8_t *)j->dst + packed_off, scratch,
				sizeof(scratch));
	else if (j->op == QPOOL_Q8_DEQUANT)
		j->result = membrane_simd_q8_0_dequantize_batch(j->backend,
				(const uint8_t *)j->src + packed_off, j->row_count,
				j->elems_per_row, (uint16_t *)j->dst + f16_off, scratch,
				sizeof(scratch));
	else if (j->op == QPOOL_Q4_QUANT)
		j->result = membrane_simd_q4_0_quantize_batch(j->backend,
				(const uint16_t *)j->src + f16_off, j->row_count,
				j->elems_per_row, (uint8_t *)j->dst + packed_off, scratch,
				sizeof(scratch));
	else
		j->result = membrane_simd_q4_0_dequantize_batch(j->backend,
				(const uint8_t *)j->src + packed_off, j->row_count,
				j->elems_per_row, (uint16_t *)j->dst + f16_off, scratch,
				sizeof(scratch));
	return (NULL);
}

static membrane_status_t	qpool_run(qpool_op_t op,
								membrane_simd_backend_t backend,
								const void *src, size_t n_blocks,
								size_t elems_per_row, void *dst,
								int n_threads_requested,
								int *out_threads_used)
{
	int			nt;
	qpool_job_t	jobs[256];
	pthread_t	tids[256];
	size_t		rows_per_thread;
	size_t		assigned;
	int			i;
	membrane_status_t	st;

	if (elems_per_row == 0 || elems_per_row % MEMBRANE_QSIMD_BLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (membrane_simd_batch_scratch_bytes(elems_per_row)
			> MEMBRANE_QPOOL_WORKER_SCRATCH_BYTES)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	if (src == NULL || dst == NULL || n_blocks == 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	nt = n_threads_requested > 0 ? n_threads_requested
		: (int)membrane_simd_default_threads();
	if (nt > 256)
		nt = 256;
	if (nt < 1)
		nt = 1;
	while (nt > 1 && n_blocks / (size_t)nt < MEMBRANE_QSIMD_MIN_ROWS_PER_THREAD)
		nt--;
	if (out_threads_used != NULL)
		*out_threads_used = nt;
	if (nt <= 1)
	{
		qpool_job_t	solo;

		solo.op = op;
		solo.backend = backend;
		solo.src = src;
		solo.dst = dst;
		solo.row_start = 0;
		solo.row_count = n_blocks;
		solo.elems_per_row = elems_per_row;
		solo.result = MEMBRANE_OK;
		qpool_worker(&solo);
		return (solo.result);
	}
	rows_per_thread = n_blocks / (size_t)nt;
	assigned = 0;
	i = 0;
	while (i < nt)
	{
		jobs[i].op = op;
		jobs[i].backend = backend;
		jobs[i].src = src;
		jobs[i].dst = dst;
		jobs[i].row_start = assigned;
		jobs[i].row_count = (i == nt - 1) ? (n_blocks - assigned)
			: rows_per_thread;
		jobs[i].elems_per_row = elems_per_row;
		jobs[i].result = MEMBRANE_OK;
		assigned += jobs[i].row_count;
		pthread_create(&tids[i], NULL, qpool_worker, &jobs[i]);
		i++;
	}
	st = MEMBRANE_OK;
	i = 0;
	while (i < nt)
	{
		pthread_join(tids[i], NULL);
		if (jobs[i].result != MEMBRANE_OK)
			st = jobs[i].result;
		i++;
	}
	return (st);
}

membrane_status_t	membrane_simd_q8_0_quantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						int n_threads_requested, int *out_threads_used)
{
	return (qpool_run(QPOOL_Q8_QUANT, backend, x_f16, n_blocks,
			elems_per_row, packed, n_threads_requested, out_threads_used));
}

membrane_status_t	membrane_simd_q8_0_dequantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						int n_threads_requested, int *out_threads_used)
{
	return (qpool_run(QPOOL_Q8_DEQUANT, backend, packed, n_blocks,
			elems_per_row, out_f16, n_threads_requested, out_threads_used));
}

membrane_status_t	membrane_simd_q4_0_quantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint16_t *x_f16, size_t n_blocks,
						size_t elems_per_row, uint8_t *packed,
						int n_threads_requested, int *out_threads_used)
{
	return (qpool_run(QPOOL_Q4_QUANT, backend, x_f16, n_blocks,
			elems_per_row, packed, n_threads_requested, out_threads_used));
}

membrane_status_t	membrane_simd_q4_0_dequantize_batch_parallel(
						membrane_simd_backend_t backend,
						const uint8_t *packed, size_t n_blocks,
						size_t elems_per_row, uint16_t *out_f16,
						int n_threads_requested, int *out_threads_used)
{
	return (qpool_run(QPOOL_Q4_DEQUANT, backend, packed, n_blocks,
			elems_per_row, out_f16, n_threads_requested, out_threads_used));
}
