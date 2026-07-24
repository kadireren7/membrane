#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/f16xor.h"
#include "membrane/kvpredict.h"

/* Shannon entropy (bits/byte) over a contiguous byte range. */
static double	range_entropy(const uint8_t *b, size_t len)
{
	uint64_t	hist[256];
	double		h;
	double		p;
	size_t		i;

	if (len == 0)
		return (0.0);
	memset(hist, 0, sizeof(hist));
	i = 0;
	while (i < len)
		hist[b[i++]] += 1;
	h = 0.0;
	i = 0;
	while (i < 256)
	{
		if (hist[i] > 0)
		{
			p = (double)hist[i] / (double)len;
			h -= p * log2(p);
		}
		i++;
	}
	return (h);
}

/* Shannon entropy over every other byte, starting at `off` (0 = low
 * plane, 1 = high plane of the interleaved residual). */
static double	plane_entropy(const uint8_t *b, size_t len, size_t off)
{
	uint64_t	hist[256];
	double		h;
	double		p;
	size_t		i;
	size_t		n;

	memset(hist, 0, sizeof(hist));
	i = off;
	n = 0;
	while (i < len)
	{
		hist[b[i]] += 1;
		n += 1;
		i += 2;
	}
	if (n == 0)
		return (0.0);
	h = 0.0;
	i = 0;
	while (i < 256)
	{
		if (hist[i] > 0)
		{
			p = (double)hist[i] / (double)n;
			h -= p * log2(p);
		}
		i++;
	}
	return (h);
}

static void	zero_stats(const uint8_t *b, size_t len,
				membrane_residual_metrics_t *m)
{
	size_t	i;
	size_t	run;

	i = 0;
	run = 0;
	while (i < len)
	{
		if (b[i] == 0)
		{
			m->zero_bytes += 1;
			run += 1;
			if (run > m->longest_zero_run)
				m->longest_zero_run = run;
		}
		else
			run = 0;
		i++;
	}
	i = 0;
	while (i + 1 < len)
	{
		if (b[i] == 0 && b[i + 1] == 0)
			m->zero_u16 += 1;
		i += 2;
	}
	m->total_u16 = len / 2;
}

static uint64_t	ideal_plane_bytes(size_t plane_bytes, double entropy_lo,
					double entropy_hi)
{
	double	lo;
	double	hi;

	lo = ceil((double)plane_bytes * entropy_lo / 8.0);
	hi = ceil((double)plane_bytes * entropy_hi / 8.0);
	return ((uint64_t)lo + (uint64_t)hi);
}

/* Fills quartile_entropy with the residual entropy of four equal token
 * bands. When the payload cannot be split cleanly into whole-row bands it
 * falls back to the overall entropy in every slot. */
static void	quartile_entropy(const uint8_t *res, size_t len, size_t n_rows,
				membrane_residual_metrics_t *m)
{
	size_t	row_bytes;
	size_t	q;
	size_t	r0;
	size_t	r1;

	q = 0;
	if (n_rows < MEMBRANE_PRED_QUARTILES || len % n_rows != 0)
	{
		while (q < MEMBRANE_PRED_QUARTILES)
			m->quartile_entropy[q++] = m->entropy;
		return ;
	}
	row_bytes = len / n_rows;
	while (q < MEMBRANE_PRED_QUARTILES)
	{
		r0 = n_rows * q / MEMBRANE_PRED_QUARTILES;
		r1 = n_rows * (q + 1) / MEMBRANE_PRED_QUARTILES;
		m->quartile_entropy[q] = range_entropy(res + r0 * row_bytes,
				(r1 - r0) * row_bytes);
		q++;
	}
}

/* Resolves cfg to an element stride, marking TOKEN/ROW inapplicable (and
 * falling back to stride 0) when no row width is available. */
static size_t	resolve_stride(const membrane_residual_cfg_t *cfg,
					int *applicable)
{
	*applicable = 1;
	if (cfg->predictor == MEMBRANE_PRED_XOR_PREV_ELEMENT)
		return (1);
	if (cfg->predictor == MEMBRANE_PRED_XOR_PREV_TOKEN
		|| cfg->predictor == MEMBRANE_PRED_XOR_PREV_ROW)
	{
		if (cfg->row_elems == 0)
		{
			*applicable = 0;
			return (0);
		}
		return (cfg->row_elems);
	}
	return (0);
}

static void	fill_metrics(const uint8_t *res, size_t len, size_t n_rows,
				membrane_residual_metrics_t *m)
{
	size_t	plane_bytes;

	plane_bytes = len / 2;
	m->raw_bytes = len;
	m->entropy = range_entropy(res, len);
	m->low_entropy = plane_entropy(res, len, 0);
	m->high_entropy = plane_entropy(res, len, 1);
	zero_stats(res, len, m);
	m->ideal_bytes = ideal_plane_bytes(plane_bytes, m->low_entropy,
			m->high_entropy);
	quartile_entropy(res, len, n_rows, m);
}

membrane_status_t	membrane_kv_residual_metrics(const uint8_t *buf,
						size_t len, const membrane_residual_cfg_t *cfg,
						membrane_residual_metrics_t *out)
{
	uint8_t	*res;
	size_t	stride;

	if (out == NULL || cfg == NULL || (buf == NULL && len > 0) || len % 2 != 0)
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(out, 0, sizeof(*out));
	stride = resolve_stride(cfg, &out->applicable);
	out->stride_elems = stride;
	if (len == 0)
		return (MEMBRANE_OK);
	res = malloc(len);
	if (res == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	membrane_f16xor_forward(buf, len, stride, res);
	fill_metrics(res, len, cfg->n_rows, out);
	free(res);
	return (MEMBRANE_OK);
}
