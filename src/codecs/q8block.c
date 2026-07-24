#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/f16convert.h"
#include "membrane/q8block.h"

static void	put_u32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static uint32_t	get_u32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static void	put_f32(uint8_t *p, float v)
{
	uint32_t	bits;

	memcpy(&bits, &v, sizeof(bits));
	put_u32(p, bits);
}

static float	get_f32(const uint8_t *p)
{
	uint32_t	bits;
	float		v;

	bits = get_u32(p);
	memcpy(&v, &bits, sizeof(v));
	return (v);
}

static size_t	meta_per_group(membrane_q8_mode_t mode)
{
	if (mode == MEMBRANE_Q8_AFFINE)
		return (8);
	return (4);
}

static size_t	group_count(size_t elements, size_t group_elems)
{
	return ((elements + group_elems - 1) / group_elems);
}

static int	valid_cfg(const membrane_q8_cfg_t *cfg)
{
	if (cfg == NULL || cfg->group_elems == 0)
		return (0);
	if (cfg->mode != MEMBRANE_Q8_SYMMETRIC && cfg->mode != MEMBRANE_Q8_AFFINE)
		return (0);
	return (1);
}

size_t	membrane_q8_bound(size_t in_len, const membrane_q8_cfg_t *cfg)
{
	size_t	elements;
	size_t	groups;
	size_t	meta;

	if (!valid_cfg(cfg) || in_len % 2 != 0)
		return (SIZE_MAX);
	elements = in_len / 2;
	groups = group_count(elements, cfg->group_elems);
	meta = meta_per_group(cfg->mode);
	if (groups > (SIZE_MAX - MEMBRANE_Q8_HEADER - elements) / meta)
		return (SIZE_MAX);
	return (MEMBRANE_Q8_HEADER + groups * meta + elements);
}

typedef struct s_group_range
{
	size_t	start;
	size_t	count;
}	group_range_t;

static group_range_t	group_at(size_t g, size_t group_elems, size_t elements)
{
	group_range_t	r;

	r.start = g * group_elems;
	r.count = group_elems;
	if (r.start + r.count > elements)
		r.count = elements - r.start;
	return (r);
}

/* Scans a group's float values, tracking min/max/max_abs over the finite
 * elements only and counting non-finite inputs. */
static void	scan_group(const float *vals, group_range_t rg, float *min,
				float *max, float *max_abs, membrane_q8_stats_t *st)
{
	size_t	i;
	int		any;

	any = 0;
	*min = 0.0f;
	*max = 0.0f;
	*max_abs = 0.0f;
	i = 0;
	while (i < rg.count)
	{
		if (isnan(vals[i]))
		{
			if (st != NULL)
				st->nan_input += 1;
		}
		else if (isinf(vals[i]))
		{
			if (st != NULL)
				st->inf_input += 1;
		}
		else
		{
			if (!any || vals[i] < *min)
				*min = vals[i];
			if (!any || vals[i] > *max)
				*max = vals[i];
			if (fabsf(vals[i]) > *max_abs)
				*max_abs = fabsf(vals[i]);
			any = 1;
		}
		i++;
	}
	if (!any && st != NULL)
		st->degenerate_groups += 1;
}

static int8_t	clamp_i32(long v, int lo, int hi)
{
	if (v < lo)
		return ((int8_t)lo);
	if (v > hi)
		return ((int8_t)hi);
	return ((int8_t)v);
}

static void	encode_symmetric(const float *vals, group_range_t rg,
				uint8_t *meta, int8_t *codes, membrane_q8_stats_t *st)
{
	float	min;
	float	max;
	float	scale;
	size_t	i;
	long	q;

	scan_group(vals, rg, &min, &max, &scale, st);
	if (scale > 0.0f)
		scale /= 127.0f;
	put_f32(meta, scale);
	i = 0;
	while (i < rg.count)
	{
		if (isnan(vals[i]))
			q = 0;
		else if (isinf(vals[i]))
			q = (vals[i] > 0.0f) ? 127 : -127;
		else if (scale > 0.0f)
			q = lroundf(vals[i] / scale);
		else
			q = 0;
		codes[i] = clamp_i32(q, -127, 127);
		if (codes[i] == 127 || codes[i] == -127)
			if (st != NULL)
				st->saturated += 1;
		i++;
	}
}

static void	encode_affine(const float *vals, group_range_t rg,
				uint8_t *meta, int8_t *codes, membrane_q8_stats_t *st)
{
	float	min;
	float	max;
	float	max_abs;
	float	scale;
	size_t	i;
	long	q;

	scan_group(vals, rg, &min, &max, &max_abs, st);
	scale = (max > min) ? (max - min) / 255.0f : 0.0f;
	put_f32(meta, scale);
	put_f32(meta + 4, min);
	i = 0;
	while (i < rg.count)
	{
		if (isnan(vals[i]))
			q = -128;
		else if (isinf(vals[i]))
			q = (vals[i] > 0.0f) ? 127 : -128;
		else if (scale > 0.0f)
			q = lroundf((vals[i] - min) / scale) - 128;
		else
			q = 0;
		codes[i] = clamp_i32(q, -128, 127);
		if (codes[i] == 127 || codes[i] == -128)
			if (st != NULL)
				st->saturated += 1;
		i++;
	}
}

membrane_status_t	membrane_q8_encode(const membrane_q8_cfg_t *cfg,
						const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len,
						membrane_q8_stats_t *stats)
{
	size_t			elements;
	size_t			groups;
	size_t			meta_sz;
	size_t			needed;
	float			*vals;
	uint8_t			*meta_ptr;
	int8_t			*code_ptr;
	size_t			g;
	group_range_t	rg;
	size_t			i;

	if (!valid_cfg(cfg) || (in == NULL && in_len > 0) || out_len == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (in_len % 2 != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	elements = in_len / 2;
	needed = membrane_q8_bound(in_len, cfg);
	if (needed == SIZE_MAX)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (out_cap < needed)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	groups = group_count(elements, cfg->group_elems);
	meta_sz = meta_per_group(cfg->mode);
	if (stats != NULL)
		memset(stats, 0, sizeof(*stats));
	vals = malloc(elements * sizeof(*vals) + sizeof(*vals));
	if (vals == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < elements)
	{
		vals[i] = membrane_f16_to_f32(((const uint16_t *)(const void *)in)[i]);
		i++;
	}
	meta_ptr = out + MEMBRANE_Q8_HEADER;
	code_ptr = (int8_t *)(out + MEMBRANE_Q8_HEADER + groups * meta_sz);
	g = 0;
	while (g < groups)
	{
		rg = group_at(g, cfg->group_elems, elements);
		if (cfg->mode == MEMBRANE_Q8_AFFINE)
			encode_affine(vals + rg.start, rg, meta_ptr, code_ptr + rg.start,
				stats);
		else
			encode_symmetric(vals + rg.start, rg, meta_ptr,
				code_ptr + rg.start, stats);
		meta_ptr += meta_sz;
		g++;
	}
	free(vals);
	out[0] = 1;
	out[1] = (uint8_t)cfg->mode;
	out[2] = 0;
	out[3] = 0;
	put_u32(out + 4, (uint32_t)cfg->group_elems);
	put_u32(out + 8, (uint32_t)elements);
	put_u32(out + 12, (uint32_t)groups);
	put_u32(out + 16, membrane_block_checksum(out + MEMBRANE_Q8_HEADER,
			needed - MEMBRANE_Q8_HEADER));
	if (stats != NULL)
	{
		stats->elements = elements;
		stats->groups = groups;
	}
	*out_len = needed;
	return (MEMBRANE_OK);
}

static membrane_status_t	read_header(const uint8_t *in, size_t in_len,
								membrane_q8_mode_t *mode, size_t *group_elems,
								size_t *elements, size_t *groups)
{
	uint32_t	ge;
	uint32_t	el;
	uint32_t	gr;
	size_t		meta_sz;

	if (in_len < MEMBRANE_Q8_HEADER)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (in[0] != 1 || in[2] != 0 || in[3] != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (in[1] != MEMBRANE_Q8_SYMMETRIC && in[1] != MEMBRANE_Q8_AFFINE)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*mode = (membrane_q8_mode_t)in[1];
	ge = get_u32(in + 4);
	el = get_u32(in + 8);
	gr = get_u32(in + 12);
	if (ge == 0 || gr != group_count(el, ge))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	meta_sz = meta_per_group(*mode);
	if ((size_t)gr > (SIZE_MAX - MEMBRANE_Q8_HEADER - (size_t)el) / meta_sz)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (MEMBRANE_Q8_HEADER + (size_t)gr * meta_sz + (size_t)el != in_len)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (get_u32(in + 16) != membrane_block_checksum(in + MEMBRANE_Q8_HEADER,
			in_len - MEMBRANE_Q8_HEADER))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*group_elems = ge;
	*elements = el;
	*groups = gr;
	return (MEMBRANE_OK);
}

/* The encoder never emits a non-finite scale or bias (they are derived
 * only from finite input values); a NaN/Inf here means the stream was
 * corrupted after encoding, so it is rejected before any dequantization
 * happens. */
static int	meta_all_finite(membrane_q8_mode_t mode, const uint8_t *meta_ptr,
				size_t groups, size_t meta_sz)
{
	size_t	g;

	g = 0;
	while (g < groups)
	{
		if (!isfinite(get_f32(meta_ptr + g * meta_sz)))
			return (0);
		if (mode == MEMBRANE_Q8_AFFINE
			&& !isfinite(get_f32(meta_ptr + g * meta_sz + 4)))
			return (0);
		g++;
	}
	return (1);
}

static void	decode_group(membrane_q8_mode_t mode, const uint8_t *meta,
				const int8_t *codes, group_range_t rg, uint16_t *out)
{
	float	scale;
	float	bias;
	size_t	i;
	float	v;

	scale = get_f32(meta);
	bias = (mode == MEMBRANE_Q8_AFFINE) ? get_f32(meta + 4) : 0.0f;
	i = 0;
	while (i < rg.count)
	{
		if (mode == MEMBRANE_Q8_AFFINE)
			v = ((float)codes[i] + 128.0f) * scale + bias;
		else
			v = (float)codes[i] * scale;
		out[i] = membrane_f32_to_f16(v);
		i++;
	}
}

membrane_status_t	membrane_q8_decode(const uint8_t *in, size_t in_len,
						uint8_t *out, size_t out_cap, size_t *out_len)
{
	membrane_q8_mode_t	mode;
	size_t				group_elems;
	size_t				elements;
	size_t				groups;
	size_t				meta_sz;
	const uint8_t		*meta_ptr;
	const int8_t		*code_ptr;
	group_range_t		rg;
	size_t				g;
	membrane_status_t	st;

	if ((in == NULL && in_len > 0) || out_len == NULL
		|| (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	st = read_header(in, in_len, &mode, &group_elems, &elements, &groups);
	if (st != MEMBRANE_OK)
		return (st);
	if (elements > SIZE_MAX / 2 || out_cap < elements * 2)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	meta_sz = meta_per_group(mode);
	meta_ptr = in + MEMBRANE_Q8_HEADER;
	code_ptr = (const int8_t *)(in + MEMBRANE_Q8_HEADER + groups * meta_sz);
	if (!meta_all_finite(mode, meta_ptr, groups, meta_sz))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	g = 0;
	while (g < groups)
	{
		rg = group_at(g, group_elems, elements);
		decode_group(mode, meta_ptr, code_ptr + rg.start, rg,
			(uint16_t *)(void *)out + rg.start);
		meta_ptr += meta_sz;
		g++;
	}
	*out_len = elements * 2;
	return (MEMBRANE_OK);
}
