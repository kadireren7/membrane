/*
 * Phase 4.4: thin adapter over ggml's OWN Q8_0/Q4_0 quantize/dequantize
 * implementations, from the pinned llama.cpp commit
 * c0bc8591e8815c63cb01dd3f051a8b0df02501c9
 * (third_party/llama.cpp, MIT license -- see third_party/llama.cpp/LICENSE).
 * No quantization math is reimplemented in this file; every call below
 * goes straight into a real, linked ggml symbol.
 *
 * Two families of quantize function exist for Q8_0 in this ggml
 * version, and they are NOT bit-identical to each other:
 *
 *   - quantize_row_q8_0_ref (ggml-quants.c) -- the scalar "reference
 *     implementation for deterministic creation of model files" (ggml's
 *     own comment). Used by ggml_quantize_chunk for GGUF file
 *     conversion. Rounds with roundf() (half away from zero).
 *   - quantize_row_q8_0 (ggml-cpu/arch/x86/quants.c on this build,
 *     AVX2) -- what the CPU backend actually calls when converting a
 *     live tensor's values into a Q8_0-typed tensor during inference
 *     (e.g. writing attention output into a Q8_0 KV cache via
 *     MEMBRANE's kv_type_override). Rounds via _mm256_round_ps(...,
 *     _MM_ROUND_NEAREST) -- IEEE-754 round-half-to-even -- which
 *     differs from roundf() exactly at .5 boundaries.
 *
 * MEMBRANE's whole purpose here is to predict/measure real KV cache
 * quantization, not GGUF model-file conversion, so this module calls
 * the CPU-backend forms (quantize_row_q8_0, quantize_row_q4_0 -- the
 * latter has no arch-specific override in this ggml version, so it is
 * already identical to quantize_row_q4_0_ref) -- i.e. the same
 * functions ggml's own CPU backend dispatches to for live tensor
 * quantization. This choice is verified, not assumed: Phase 4.4's
 * cross-tool parity check (docs/phase4-ggml-quant-parity.md) compares
 * this module's output against real KV cache bytes captured from an
 * actual llama.cpp decode.
 *
 * Dequantization has only one implementation each for Q8_0/Q4_0 in this
 * ggml version (ggml-quants.c, no arch-specific override), so there is
 * no such choice to make there.
 */
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "quants.h"

#include "membrane/ggml_quant.h"

#include <stdlib.h>
#include <string.h>

static membrane_status_t	check_args(const void *a, const void *b,
								size_t n)
{
	if (a == NULL || b == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (n == 0 || n % MEMBRANE_GGML_QBLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_ggml_q8_0_quantize(const uint16_t *x_f16,
						size_t n, uint8_t *out)
{
	membrane_status_t	st;
	float				*x_f32;

	st = check_args(x_f16, out, n);
	if (st != MEMBRANE_OK)
		return (st);
	x_f32 = (float *)malloc(n * sizeof(float));
	if (x_f32 == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	ggml_fp16_to_fp32_row((const ggml_fp16_t *)x_f16, x_f32, (int64_t)n);
	quantize_row_q8_0(x_f32, out, (int64_t)n);
	free(x_f32);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_ggml_q4_0_quantize(const uint16_t *x_f16,
						size_t n, uint8_t *out)
{
	membrane_status_t	st;
	float				*x_f32;

	st = check_args(x_f16, out, n);
	if (st != MEMBRANE_OK)
		return (st);
	x_f32 = (float *)malloc(n * sizeof(float));
	if (x_f32 == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	ggml_fp16_to_fp32_row((const ggml_fp16_t *)x_f16, x_f32, (int64_t)n);
	quantize_row_q4_0(x_f32, out, (int64_t)n);
	free(x_f32);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_ggml_q8_0_dequantize(const uint8_t *in,
						size_t n, uint16_t *out_f16)
{
	membrane_status_t	st;
	float				*y_f32;

	st = check_args(in, out_f16, n);
	if (st != MEMBRANE_OK)
		return (st);
	y_f32 = (float *)malloc(n * sizeof(float));
	if (y_f32 == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	dequantize_row_q8_0((const block_q8_0 *)in, y_f32, (int64_t)n);
	ggml_fp32_to_fp16_row(y_f32, (ggml_fp16_t *)out_f16, (int64_t)n);
	free(y_f32);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_ggml_q4_0_dequantize(const uint8_t *in,
						size_t n, uint16_t *out_f16)
{
	membrane_status_t	st;
	float				*y_f32;

	st = check_args(in, out_f16, n);
	if (st != MEMBRANE_OK)
		return (st);
	y_f32 = (float *)malloc(n * sizeof(float));
	if (y_f32 == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	dequantize_row_q4_0((const block_q4_0 *)in, y_f32, (int64_t)n);
	ggml_fp32_to_fp16_row(y_f32, (ggml_fp16_t *)out_f16, (int64_t)n);
	free(y_f32);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_ggml_quant_roundtrip(uint16_t *x_f16, size_t n,
						int bits)
{
	membrane_status_t	st;
	size_t				nblocks;
	uint8_t				*packed;

	if (bits == 16)
		return (MEMBRANE_OK);
	if (bits != 8 && bits != 4)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (x_f16 == NULL || n == 0 || n % MEMBRANE_GGML_QBLOCK_ELEMS != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	nblocks = n / MEMBRANE_GGML_QBLOCK_ELEMS;
	packed = (uint8_t *)malloc(nblocks * (bits == 8
				? MEMBRANE_GGML_Q8_0_BLOCK_BYTES
				: MEMBRANE_GGML_Q4_0_BLOCK_BYTES));
	if (packed == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (bits == 8)
	{
		st = membrane_ggml_q8_0_quantize(x_f16, n, packed);
		if (st == MEMBRANE_OK)
			st = membrane_ggml_q8_0_dequantize(packed, n, x_f16);
	}
	else
	{
		st = membrane_ggml_q4_0_quantize(x_f16, n, packed);
		if (st == MEMBRANE_OK)
			st = membrane_ggml_q4_0_dequantize(packed, n, x_f16);
	}
	free(packed);
	return (st);
}
