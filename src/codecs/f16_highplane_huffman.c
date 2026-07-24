/*
 * F16 high-plane Huffman codec (Phase 2.4).
 *
 * The lossless lever Phase 2.2 identified: the F16 high byte plane
 * (sign+exponent) sits at ~5.6 bits/byte while the low (mantissa) plane is
 * ~8. So this codec keeps the low plane RAW and entropy-codes only the high
 * plane with canonical Huffman (see membrane/huffman.h). On real KV data
 * that is the whole realisable lossless gain; the codec exists to measure
 * the actual ratio and speed of cashing it in.
 *
 * Stream: an 18-byte little-endian header
 *   [0]      version   (= 1)
 *   [1]      flags     (= 0)
 *   [2..5]   element_count  (F16 elements = in_len / 2 = bytes per plane)
 *   [6..9]   low_len        (= element_count; the RAW low plane length)
 *   [10..13] high_comp_len  (Huffman stream length for the high plane)
 *   [14..17] crc            (CRC32 of the original F16 payload)
 * followed by the RAW low plane (low_len bytes) then the high-plane Huffman
 * stream. When the total is not smaller than the input the block layer
 * stores the block RAW instead, so the cache never expands.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/codec.h"
#include "membrane/huffman.h"

# define HP_HEADER 18
# define HP_VERSION 1
# define HP_U32_MAX 0xFFFFFFFFu

static void	hp_put_u32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static uint32_t	hp_get_u32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* Deinterleaves the low plane into out_low and the high plane into
 * out_high, each plane_len bytes. */
static void	hp_split(const uint8_t *in, size_t plane_len, uint8_t *out_low,
				uint8_t *out_high)
{
	size_t	i;

	i = 0;
	while (i < plane_len)
	{
		out_low[i] = in[2 * i];
		out_high[i] = in[2 * i + 1];
		i++;
	}
}

static membrane_status_t	hp_compress(const uint8_t *in, size_t in_len,
					uint8_t *out, size_t out_cap, size_t *out_len)
{
	uint8_t				*high;
	size_t				plane_len;
	size_t				high_len;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (in_len % 2 != 0 || in_len / 2 > HP_U32_MAX)
		return (MEMBRANE_ERR_INVALID_ARG);
	plane_len = in_len / 2;
	if (out_cap < HP_HEADER + plane_len)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	high = malloc(plane_len + 1);
	if (high == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	hp_split(in, plane_len, out + HP_HEADER, high);
	st = membrane_huffman_compress(high, plane_len,
			out + HP_HEADER + plane_len, out_cap - HP_HEADER - plane_len,
			&high_len);
	free(high);
	if (st != MEMBRANE_OK)
		return (st);
	out[0] = HP_VERSION;
	out[1] = 0;
	hp_put_u32(out + 2, (uint32_t)plane_len);
	hp_put_u32(out + 6, (uint32_t)plane_len);
	hp_put_u32(out + 10, (uint32_t)high_len);
	hp_put_u32(out + 14, membrane_block_checksum(in, in_len));
	*out_len = HP_HEADER + plane_len + high_len;
	return (MEMBRANE_OK);
}

static membrane_status_t	hp_read_header(const uint8_t *in, size_t in_len,
					size_t *plane_len, size_t *high_len, uint32_t *crc)
{
	size_t	low_len;

	if (in_len < HP_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (in[0] != HP_VERSION || in[1] != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*plane_len = hp_get_u32(in + 2);
	low_len = hp_get_u32(in + 6);
	*high_len = hp_get_u32(in + 10);
	*crc = hp_get_u32(in + 14);
	if (low_len != *plane_len)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (*plane_len > in_len - HP_HEADER
		|| HP_HEADER + *plane_len + *high_len != in_len)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

static membrane_status_t	hp_decompress(const uint8_t *in, size_t in_len,
					uint8_t *out, size_t out_cap, size_t *out_len)
{
	uint8_t				*high;
	size_t				plane_len;
	size_t				high_len;
	size_t				got;
	uint32_t			crc;
	size_t				i;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL
		|| (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	st = hp_read_header(in, in_len, &plane_len, &high_len, &crc);
	if (st != MEMBRANE_OK)
		return (st);
	if (plane_len > SIZE_MAX / 2 || out_cap < plane_len * 2)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	high = malloc(plane_len + 1);
	if (high == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	st = membrane_huffman_decompress(in + HP_HEADER + plane_len, high_len,
			high, plane_len, &got);
	if (st == MEMBRANE_OK && got != plane_len)
		st = MEMBRANE_ERR_CORRUPT_DATA;
	i = 0;
	while (st == MEMBRANE_OK && i < plane_len)
	{
		out[2 * i] = in[HP_HEADER + i];
		out[2 * i + 1] = high[i];
		i++;
	}
	free(high);
	if (st != MEMBRANE_OK)
		return (st);
	if (membrane_block_checksum(out, plane_len * 2) != crc)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*out_len = plane_len * 2;
	return (MEMBRANE_OK);
}

/* Worst case: header + RAW low plane + Huffman-bounded high plane. */
static size_t	hp_bound(size_t in_len)
{
	size_t	plane_len;

	plane_len = in_len / 2;
	if (plane_len > SIZE_MAX / 4)
		return (SIZE_MAX);
	return (HP_HEADER + plane_len + membrane_huffman_bound(plane_len));
}

const membrane_codec_vtable_t	g_membrane_codec_f16_highplane_huffman = {
	.name = "f16hp",
	.id = MEMBRANE_CODEC_F16_HIGHPLANE_HUFFMAN,
	.compress = hp_compress,
	.decompress = hp_decompress,
	.bound = hp_bound,
};
