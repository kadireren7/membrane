#define _DEFAULT_SOURCE

#include <stdlib.h>
#include <string.h>

#include "membrane/attntrace.h"
#include "membrane/block.h"

static void	put_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

static void	put_le64(uint8_t *p, uint64_t v)
{
	put_le32(p, (uint32_t)v);
	put_le32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t	get_le32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static uint64_t	get_le64(const uint8_t *p)
{
	return ((uint64_t)get_le32(p) | ((uint64_t)get_le32(p + 4) << 32));
}

static void	put_f32(uint8_t *p, float v)
{
	uint32_t	bits;

	memcpy(&bits, &v, sizeof(bits));
	put_le32(p, bits);
}

static float	get_f32(const uint8_t *p)
{
	uint32_t	bits;
	float		v;

	bits = get_le32(p);
	memcpy(&v, &bits, sizeof(v));
	return (v);
}

static int	header_bounds_ok(const membrane_attntrace_header_t *h)
{
	if (h->step_count == 0 || h->step_count > MEMBRANE_ATTNTRACE_MAX_STEPS)
		return (0);
	if (h->n_layer == 0 || h->n_layer > MEMBRANE_ATTNTRACE_MAX_LAYER)
		return (0);
	if (h->n_head == 0 || h->n_head > MEMBRANE_ATTNTRACE_MAX_HEAD)
		return (0);
	if (h->top_k == 0 || h->top_k > MEMBRANE_ATTNTRACE_MAX_TOPK)
		return (0);
	return (1);
}

size_t	membrane_attntrace_entry_count(const membrane_attntrace_header_t *h)
{
	size_t	n;

	if (h == NULL || !header_bounds_ok(h))
		return (0);
	n = (size_t)h->step_count;
	n *= (size_t)h->n_layer;
	n *= (size_t)h->n_head;
	n *= (size_t)h->top_k;
	return (n);
}

static void	serialize_header(uint8_t *b,
				const membrane_attntrace_header_t *h)
{
	put_le32(b + 0, MEMBRANE_ATTNTRACE_MAGIC);
	put_le32(b + 4, MEMBRANE_ATTNTRACE_VERSION);
	memcpy(b + 8, h->model, MEMBRANE_ATTNTRACE_MODEL_CAP);
	put_le32(b + 72, h->source);
	put_le32(b + 76, h->n_layer);
	put_le32(b + 80, h->n_head);
	put_le32(b + 84, h->block_size_tokens);
	put_le32(b + 88, h->prompt_len);
	put_le32(b + 92, h->step_count);
	put_le32(b + 96, h->top_k);
	put_le64(b + 100, h->created_unix_time);
	put_le32(b + 108, h->payload_checksum);
	put_le32(b + 112, membrane_block_checksum(b, 112));
}

static void	deserialize_header(const uint8_t *b,
				membrane_attntrace_header_t *h)
{
	memcpy(h->model, b + 8, MEMBRANE_ATTNTRACE_MODEL_CAP);
	h->model[MEMBRANE_ATTNTRACE_MODEL_CAP - 1] = '\0';
	h->source = get_le32(b + 72);
	h->n_layer = get_le32(b + 76);
	h->n_head = get_le32(b + 80);
	h->block_size_tokens = get_le32(b + 84);
	h->prompt_len = get_le32(b + 88);
	h->step_count = get_le32(b + 92);
	h->top_k = get_le32(b + 96);
	h->created_unix_time = get_le64(b + 100);
	h->payload_checksum = get_le32(b + 108);
}

membrane_status_t	membrane_attntrace_write(FILE *f,
						const membrane_attntrace_header_t *h,
						const membrane_attntrace_entry_t *entries)
{
	uint8_t				hbuf[MEMBRANE_ATTNTRACE_HEADER_SIZE];
	uint8_t				*payload;
	size_t				n;
	size_t				i;
	membrane_attntrace_header_t	hcopy;

	if (f == NULL || h == NULL || entries == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!header_bounds_ok(h))
		return (MEMBRANE_ERR_INVALID_ARG);
	if (strnlen(h->model, MEMBRANE_ATTNTRACE_MODEL_CAP)
			>= (size_t)MEMBRANE_ATTNTRACE_MODEL_CAP)
		return (MEMBRANE_ERR_INVALID_ARG);
	n = membrane_attntrace_entry_count(h);
	payload = malloc(n * 8);
	if (payload == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < n)
	{
		put_le32(payload + 8 * i, entries[i].block_id);
		put_f32(payload + 8 * i + 4, entries[i].score);
		i++;
	}
	hcopy = *h;
	hcopy.payload_checksum = membrane_block_checksum(payload, n * 8);
	serialize_header(hbuf, &hcopy);
	if (fwrite(hbuf, 1, sizeof(hbuf), f) != sizeof(hbuf)
			|| fwrite(payload, 1, n * 8, f) != n * 8)
	{
		free(payload);
		return (MEMBRANE_ERR_IO);
	}
	free(payload);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace_read_header(FILE *f,
						membrane_attntrace_header_t *h)
{
	uint8_t		buf[MEMBRANE_ATTNTRACE_HEADER_SIZE];
	uint32_t	magic;
	uint32_t	version;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_IO);
	magic = get_le32(buf + 0);
	version = get_le32(buf + 4);
	if (magic != MEMBRANE_ATTNTRACE_MAGIC || version != MEMBRANE_ATTNTRACE_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (membrane_block_checksum(buf, 112) != get_le32(buf + 112))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	deserialize_header(buf, h);
	if (!header_bounds_ok(h))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace_read_entries(FILE *f,
						const membrane_attntrace_header_t *h,
						membrane_attntrace_entry_t *out)
{
	uint8_t		*payload;
	size_t		n;
	size_t		i;

	if (f == NULL || h == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	n = membrane_attntrace_entry_count(h);
	if (n == 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	payload = malloc(n * 8);
	if (payload == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(payload, 1, n * 8, f) != n * 8)
	{
		free(payload);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	if (membrane_block_checksum(payload, n * 8) != h->payload_checksum)
	{
		free(payload);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	i = 0;
	while (i < n)
	{
		out[i].block_id = get_le32(payload + 8 * i);
		out[i].score = get_f32(payload + 8 * i + 4);
		i++;
	}
	free(payload);
	return (MEMBRANE_OK);
}
