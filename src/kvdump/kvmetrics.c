#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/kvmetrics.h"

static double	block_entropy(const uint8_t *b, size_t len, uint64_t *zeros)
{
	uint64_t	hist[256];
	double		h;
	double		p;
	size_t		i;

	memset(hist, 0, sizeof(hist));
	i = 0;
	while (i < len)
		hist[b[i++]] += 1;
	*zeros += hist[0];
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

static void	block_runs(const uint8_t *b, size_t len, membrane_kv_metrics_t *m)
{
	size_t	i;
	size_t	run;

	i = 0;
	while (i < len)
	{
		run = 1;
		while (i + run < len && b[i + run] == b[i])
			run++;
		m->total_runs += 1;
		if (run > m->max_run)
			m->max_run = run;
		i += run;
	}
}

static membrane_status_t	block_rle_size(const uint8_t *b, size_t len,
								uint64_t *out)
{
	const membrane_codec_vtable_t	*rle;
	uint8_t							*tmp;
	size_t							got;
	membrane_status_t				st;

	rle = membrane_codec_get(MEMBRANE_CODEC_RLE);
	tmp = malloc(rle->bound(len));
	if (tmp == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	st = rle->compress(b, len, tmp, rle->bound(len), &got);
	free(tmp);
	if (st != MEMBRANE_OK)
		return (st);
	*out += got;
	return (MEMBRANE_OK);
}

static membrane_status_t	block_adaptive(const uint8_t *b, size_t len,
								uint8_t *scratch, membrane_kv_metrics_t *m)
{
	membrane_block_t	*blk;
	size_t				got;
	membrane_status_t	st;

	blk = membrane_block_create(0, MEMBRANE_CODEC_RLE);
	if (blk == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	st = membrane_block_write(blk, b, len);
	if (st == MEMBRANE_OK)
	{
		m->adaptive_bytes += blk->stored_size;
		if (blk->stored_codec == MEMBRANE_CODEC_RAW)
			m->adaptive_raw_blocks += 1;
		else
			m->adaptive_rle_blocks += 1;
		st = membrane_block_read(blk, scratch, len, &got);
		if (st != MEMBRANE_OK || got != len || memcmp(scratch, b, len) != 0)
			m->integrity_ok = 0;
	}
	membrane_block_destroy(blk);
	return (st);
}

static membrane_status_t	metrics_one_block(const uint8_t *b, size_t len,
								uint8_t *scratch, membrane_kv_metrics_t *m)
{
	membrane_status_t	st;

	m->blocks += 1;
	m->raw_bytes += len;
	m->entropy += block_entropy(b, len, &m->zero_bytes) * (double)len;
	block_runs(b, len, m);
	st = block_rle_size(b, len, &m->rle_bytes);
	if (st != MEMBRANE_OK)
		return (st);
	return (block_adaptive(b, len, scratch, m));
}

static membrane_status_t	metrics_loop(const uint8_t *buf, size_t len,
								size_t block_size, uint8_t *scratch,
								membrane_kv_metrics_t *out)
{
	size_t				off;
	size_t				n;
	membrane_status_t	st;

	off = 0;
	st = MEMBRANE_OK;
	while (off < len && st == MEMBRANE_OK)
	{
		n = len - off;
		if (n > block_size)
			n = block_size;
		st = metrics_one_block(buf + off, n, scratch, out);
		off += n;
	}
	return (st);
}

membrane_status_t	membrane_kv_metrics_compute(const uint8_t *buf,
						size_t len, size_t block_size,
						membrane_kv_metrics_t *out)
{
	uint8_t				*scratch;
	membrane_status_t	st;

	if (out == NULL || block_size == 0 || (buf == NULL && len > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(out, 0, sizeof(*out));
	out->integrity_ok = 1;
	if (len == 0)
		return (MEMBRANE_OK);
	scratch = malloc(block_size);
	if (scratch == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	st = metrics_loop(buf, len, block_size, scratch, out);
	free(scratch);
	if (st == MEMBRANE_OK)
		out->entropy /= (double)len;
	return (st);
}
