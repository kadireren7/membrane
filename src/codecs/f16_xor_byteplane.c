/*
 * F16 XOR byte-plane codec (Phase 2.3, experimental).
 *
 * This codec is a measurement vehicle, not a compressor yet. It applies a
 * lossless predictive XOR transform (see membrane/f16xor.h), splits the
 * residual into its low and high byte planes, and stores both planes RAW.
 * The output is therefore always larger than the input (header + full
 * planes), so through the block layer it always falls back to RAW storage
 * -- exactly as intended: the goal here is to prove, losslessly, how much
 * a predictor lowers residual entropy before an entropy coder is added.
 *
 * Stream: an 18-byte little-endian header
 *   [0]      version   (= 1)
 *   [1]      predictor (membrane_predictor_t)
 *   [2..5]   stride_elems  (uint16 elements XORed against)
 *   [6..9]   plane_len     (bytes per plane = in_len / 2)
 *   [10..13] low_comp_len  (= plane_len; planes are RAW)
 *   [14..17] high_comp_len (= plane_len)
 * followed by the low residual plane then the high residual plane.
 *
 * The registered codec (MEMBRANE_CODEC_F16_XOR_BYTEPLANE) uses the
 * shape-free XOR_PREVIOUS_ELEMENT predictor so it needs no external shape;
 * membrane_f16xor_encode exposes the other predictors to the analyzer and
 * tests. decode is self-describing and inverts any of them.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/codec.h"
#include "membrane/f16xor.h"

# define XBP_HEADER 18
# define XBP_VERSION 1
# define XBP_U32_MAX 0xFFFFFFFFu

const char	*membrane_predictor_name(membrane_predictor_t p)
{
	static const char	*const names[MEMBRANE_PRED_COUNT] = {
		"none", "xor_prev_element", "xor_prev_token", "xor_prev_row"};

	if ((int)p < 0 || p >= MEMBRANE_PRED_COUNT)
		return ("invalid");
	return (names[p]);
}

void	membrane_f16xor_forward(const uint8_t *in, size_t len,
			size_t stride_elems, uint8_t *out)
{
	size_t	bs;
	size_t	i;

	bs = stride_elems * 2;
	if (bs == 0)
	{
		if (len > 0)
			memcpy(out, in, len);
		return ;
	}
	i = 0;
	while (i < len)
	{
		if (i < bs)
			out[i] = in[i];
		else
			out[i] = (uint8_t)(in[i] ^ in[i - bs]);
		i++;
	}
}

void	membrane_f16xor_inverse(uint8_t *buf, size_t len, size_t stride_elems)
{
	size_t	bs;
	size_t	i;

	bs = stride_elems * 2;
	if (bs == 0)
		return ;
	i = bs;
	while (i < len)
	{
		buf[i] = (uint8_t)(buf[i] ^ buf[i - bs]);
		i++;
	}
}

static void	xbp_put_u32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static uint32_t	xbp_get_u32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* Maps a predictor + row width to the element stride it XORs against.
 * Returns MEMBRANE_ERR_INVALID_ARG when TOKEN/ROW lack a row width. */
static membrane_status_t	xbp_resolve_stride(const membrane_f16xor_cfg_t *cfg,
								size_t *stride)
{
	if (cfg->predictor == MEMBRANE_PRED_NONE)
		*stride = 0;
	else if (cfg->predictor == MEMBRANE_PRED_XOR_PREV_ELEMENT)
		*stride = 1;
	else if (cfg->predictor == MEMBRANE_PRED_XOR_PREV_TOKEN
		|| cfg->predictor == MEMBRANE_PRED_XOR_PREV_ROW)
	{
		if (cfg->row_elems == 0)
			return (MEMBRANE_ERR_INVALID_ARG);
		*stride = cfg->row_elems;
	}
	else
		return (MEMBRANE_ERR_INVALID_ARG);
	return (MEMBRANE_OK);
}

static void	xbp_write_header(uint8_t *out, membrane_predictor_t pred,
				size_t stride, size_t plane_len)
{
	out[0] = XBP_VERSION;
	out[1] = (uint8_t)pred;
	xbp_put_u32(out + 2, (uint32_t)stride);
	xbp_put_u32(out + 6, (uint32_t)plane_len);
	xbp_put_u32(out + 10, (uint32_t)plane_len);
	xbp_put_u32(out + 14, (uint32_t)plane_len);
}

/* Deinterleaves the residual into low bytes then high bytes. */
static void	xbp_split_planes(const uint8_t *res, size_t plane_len, uint8_t *out)
{
	size_t	i;

	i = 0;
	while (i < plane_len)
	{
		out[i] = res[2 * i];
		out[plane_len + i] = res[2 * i + 1];
		i++;
	}
}

membrane_status_t	membrane_f16xor_encode(const membrane_f16xor_cfg_t *cfg,
						const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	uint8_t				*res;
	size_t				plane_len;
	size_t				stride;
	membrane_status_t	st;

	if (cfg == NULL || (in == NULL && in_len > 0) || out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (in_len % 2 != 0 || in_len / 2 > XBP_U32_MAX)
		return (MEMBRANE_ERR_INVALID_ARG);
	st = xbp_resolve_stride(cfg, &stride);
	if (st != MEMBRANE_OK)
		return (st);
	plane_len = in_len / 2;
	if (out_cap < XBP_HEADER + in_len)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	res = malloc(in_len + 1);
	if (res == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	membrane_f16xor_forward(in, in_len, stride, res);
	xbp_split_planes(res, plane_len, out + XBP_HEADER);
	free(res);
	xbp_write_header(out, cfg->predictor, stride, plane_len);
	*out_len = XBP_HEADER + in_len;
	return (MEMBRANE_OK);
}

static membrane_status_t	xbp_read_header(const uint8_t *in, size_t in_len,
								size_t *stride, size_t *plane_len)
{
	size_t	low_len;
	size_t	high_len;

	if (in_len < XBP_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (in[0] != XBP_VERSION || in[1] >= MEMBRANE_PRED_COUNT)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*stride = xbp_get_u32(in + 2);
	*plane_len = xbp_get_u32(in + 6);
	low_len = xbp_get_u32(in + 10);
	high_len = xbp_get_u32(in + 14);
	if (low_len != *plane_len || high_len != *plane_len)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (*plane_len > (in_len - XBP_HEADER) / 2
		|| XBP_HEADER + 2 * *plane_len != in_len)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_f16xor_decode(const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	const uint8_t		*low;
	size_t				plane_len;
	size_t				stride;
	size_t				i;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL
		|| (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	st = xbp_read_header(in, in_len, &stride, &plane_len);
	if (st != MEMBRANE_OK)
		return (st);
	if (plane_len > SIZE_MAX / 2 || out_cap < plane_len * 2)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	low = in + XBP_HEADER;
	i = 0;
	while (i < plane_len)
	{
		out[2 * i] = low[i];
		out[2 * i + 1] = low[plane_len + i];
		i++;
	}
	membrane_f16xor_inverse(out, plane_len * 2, stride);
	*out_len = plane_len * 2;
	return (MEMBRANE_OK);
}

static membrane_status_t	xbp_compress(const uint8_t *in, size_t in_len,
					uint8_t *out, size_t out_cap, size_t *out_len)
{
	membrane_f16xor_cfg_t	cfg;

	cfg.predictor = MEMBRANE_PRED_XOR_PREV_ELEMENT;
	cfg.row_elems = 0;
	return (membrane_f16xor_encode(&cfg, in, in_len, out, out_cap, out_len));
}

static size_t	xbp_bound(size_t in_len)
{
	if (in_len > SIZE_MAX - XBP_HEADER)
		return (SIZE_MAX);
	return (XBP_HEADER + in_len);
}

const membrane_codec_vtable_t	g_membrane_codec_f16_xor_byteplane = {
	.name = "f16xor",
	.id = MEMBRANE_CODEC_F16_XOR_BYTEPLANE,
	.compress = xbp_compress,
	.decompress = membrane_f16xor_decode,
	.bound = xbp_bound,
};
