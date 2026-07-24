/*
 * F16 byte-plane RLE codec.
 *
 * F16 tensor data interleaves a low byte and a high byte for every 16-bit
 * element. This codec splits the two byte planes apart -- all low bytes
 * into one plane, all high bytes into another -- then run-length-encodes
 * each plane independently. The hypothesis (measured, not assumed, in
 * docs/phase2-kv-byteplane.md) is that the split planes are more
 * compressible than the interleaved stream. Only even-length inputs
 * (whole F16 elements) are accepted; any other length is rejected.
 *
 * Stream layout: a 14-byte little-endian header
 *   [0]      version  (= 1)
 *   [1]      reserved (= 0)
 *   [2..5]   plane_len      (bytes per plane = in_len / 2)
 *   [6..9]   low_comp_len   (RLE size of the low-byte plane)
 *   [10..13] high_comp_len  (RLE size of the high-byte plane)
 * followed by the low plane's RLE stream, then the high plane's RLE
 * stream. The two byte-level RLE streams reuse MEMBRANE_CODEC_RLE, so
 * this codec is a pure tensor-aware transform in front of it.
 */

#include <stdint.h>
#include <stdlib.h>

#include "membrane/codec.h"

# define BP_HEADER 14
# define BP_VERSION 1
# define BP_U32_MAX 0xFFFFFFFFu

static void	bp_put_u32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static uint32_t	bp_get_u32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* Deinterleaves one byte plane out of `in` (offset 0 = low, 1 = high) and
 * run-length-encodes it into `out`. */
static membrane_status_t	bp_encode_plane(const uint8_t *in, size_t plane_len,
					int offset, uint8_t *out, size_t out_cap, size_t *out_len)
{
	const membrane_codec_vtable_t	*rle;
	uint8_t							*plane;
	size_t							i;
	membrane_status_t				st;

	plane = malloc(plane_len + 1);
	if (plane == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < plane_len)
	{
		plane[i] = in[2 * i + (size_t)offset];
		i++;
	}
	rle = membrane_codec_get(MEMBRANE_CODEC_RLE);
	st = rle->compress(plane, plane_len, out, out_cap, out_len);
	free(plane);
	return (st);
}

static void	bp_write_header(uint8_t *out, size_t plane_len,
				size_t low_len, size_t high_len)
{
	out[0] = BP_VERSION;
	out[1] = 0;
	bp_put_u32(out + 2, (uint32_t)plane_len);
	bp_put_u32(out + 6, (uint32_t)low_len);
	bp_put_u32(out + 10, (uint32_t)high_len);
}

static membrane_status_t	f16bp_compress(const uint8_t *in, size_t in_len,
					uint8_t *out, size_t out_cap, size_t *out_len)
{
	size_t				plane_len;
	size_t				low_len;
	size_t				high_len;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (in_len % 2 != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	plane_len = in_len / 2;
	if (plane_len > BP_U32_MAX)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (out_cap < BP_HEADER)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	st = bp_encode_plane(in, plane_len, 0, out + BP_HEADER,
			out_cap - BP_HEADER, &low_len);
	if (st != MEMBRANE_OK)
		return (st);
	st = bp_encode_plane(in, plane_len, 1, out + BP_HEADER + low_len,
			out_cap - BP_HEADER - low_len, &high_len);
	if (st != MEMBRANE_OK)
		return (st);
	bp_write_header(out, plane_len, low_len, high_len);
	*out_len = BP_HEADER + low_len + high_len;
	return (MEMBRANE_OK);
}

static membrane_status_t	bp_read_header(const uint8_t *in, size_t in_len,
					size_t *plane_len, size_t *low_len, size_t *high_len)
{
	if (in_len < BP_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (in[0] != BP_VERSION || in[1] != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*plane_len = bp_get_u32(in + 2);
	*low_len = bp_get_u32(in + 6);
	*high_len = bp_get_u32(in + 10);
	if (*low_len + *high_len != in_len - BP_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

/* RLE-decodes one plane into a scratch buffer of exactly plane_len bytes,
 * then scatters it back into `out` at the interleaved positions. */
static membrane_status_t	bp_decode_plane(const uint8_t *comp, size_t comp_len,
					size_t plane_len, int offset, uint8_t *out)
{
	const membrane_codec_vtable_t	*rle;
	uint8_t							*plane;
	size_t							got;
	size_t							i;
	membrane_status_t				st;

	plane = malloc(plane_len + 1);
	if (plane == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	rle = membrane_codec_get(MEMBRANE_CODEC_RLE);
	st = rle->decompress(comp, comp_len, plane, plane_len, &got);
	if (st == MEMBRANE_ERR_BUFFER_TOO_SMALL)
		st = MEMBRANE_ERR_CORRUPT_DATA;
	if (st == MEMBRANE_OK && got != plane_len)
		st = MEMBRANE_ERR_CORRUPT_DATA;
	i = 0;
	while (st == MEMBRANE_OK && i < plane_len)
	{
		out[2 * i + (size_t)offset] = plane[i];
		i++;
	}
	free(plane);
	return (st);
}

static membrane_status_t	f16bp_decompress(const uint8_t *in, size_t in_len,
					uint8_t *out, size_t out_cap, size_t *out_len)
{
	size_t				plane_len;
	size_t				low_len;
	size_t				high_len;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL || (out == NULL
			&& out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	st = bp_read_header(in, in_len, &plane_len, &low_len, &high_len);
	if (st != MEMBRANE_OK)
		return (st);
	if (plane_len > SIZE_MAX / 2 || out_cap < plane_len * 2)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	st = bp_decode_plane(in + BP_HEADER, low_len, plane_len, 0, out);
	if (st != MEMBRANE_OK)
		return (st);
	st = bp_decode_plane(in + BP_HEADER + low_len, high_len, plane_len, 1, out);
	if (st != MEMBRANE_OK)
		return (st);
	*out_len = plane_len * 2;
	return (MEMBRANE_OK);
}

/* Worst case: header + two RLE streams, each at most twice its plane. */
static size_t	f16bp_bound(size_t in_len)
{
	size_t	plane_len;

	plane_len = in_len / 2;
	if (plane_len > (SIZE_MAX - BP_HEADER) / 4)
		return (SIZE_MAX);
	return (BP_HEADER + 4 * plane_len);
}

const membrane_codec_vtable_t	g_membrane_codec_f16_byteplane_rle = {
	.name = "f16bp",
	.id = MEMBRANE_CODEC_F16_BYTEPLANE_RLE,
	.compress = f16bp_compress,
	.decompress = f16bp_decompress,
	.bound = f16bp_bound,
};
