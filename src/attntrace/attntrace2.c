#define _DEFAULT_SOURCE

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include "membrane/attntrace2.h"
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

static void	put_le16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

static uint16_t	get_le16(const uint8_t *p)
{
	return ((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static int	header_bounds_ok(const membrane_attntrace2_header_t *h)
{
	if (h->step_count == 0 || h->step_count > MEMBRANE_ATTNTRACE2_MAX_STEPS)
		return (0);
	if (h->n_layer == 0 || h->n_layer > MEMBRANE_ATTNTRACE2_MAX_LAYER)
		return (0);
	if (h->n_head == 0 || h->n_head > MEMBRANE_ATTNTRACE2_MAX_HEAD)
		return (0);
	if (h->top_k == 0 || h->top_k > MEMBRANE_ATTNTRACE2_MAX_TOPK)
		return (0);
	return (1);
}

size_t	membrane_attntrace2_entry_count(const membrane_attntrace2_header_t *h)
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
				const membrane_attntrace2_header_t *h)
{
	put_le32(b + 0, MEMBRANE_ATTNTRACE2_MAGIC);
	put_le32(b + 4, MEMBRANE_ATTNTRACE2_VERSION);
	memcpy(b + 8, h->model, MEMBRANE_ATTNTRACE2_MODEL_CAP);
	put_le32(b + 72, h->source);
	put_le32(b + 76, h->n_layer);
	put_le32(b + 80, h->n_head);
	put_le32(b + 84, h->block_size_tokens);
	put_le32(b + 88, h->prompt_len);
	put_le32(b + 92, h->step_count);
	put_le32(b + 96, h->top_k);
	put_le64(b + 100, h->created_unix_time);
	put_le32(b + 108, h->compressed);
	put_le32(b + 112, h->uncompressed_payload_size);
	put_le32(b + 116, h->stored_payload_size);
	put_le32(b + 120, h->payload_checksum);
	put_le32(b + 124, membrane_block_checksum(b, 124));
}

static void	deserialize_header(const uint8_t *b,
				membrane_attntrace2_header_t *h)
{
	memcpy(h->model, b + 8, MEMBRANE_ATTNTRACE2_MODEL_CAP);
	h->model[MEMBRANE_ATTNTRACE2_MODEL_CAP - 1] = '\0';
	h->source = get_le32(b + 72);
	h->n_layer = get_le32(b + 76);
	h->n_head = get_le32(b + 80);
	h->block_size_tokens = get_le32(b + 84);
	h->prompt_len = get_le32(b + 88);
	h->step_count = get_le32(b + 92);
	h->top_k = get_le32(b + 96);
	h->created_unix_time = get_le64(b + 100);
	h->compressed = get_le32(b + 108);
	h->uncompressed_payload_size = get_le32(b + 112);
	h->stored_payload_size = get_le32(b + 116);
	h->payload_checksum = get_le32(b + 120);
}

membrane_status_t	membrane_attntrace2_write(FILE *f,
						const membrane_attntrace2_header_t *h,
						const membrane_attntrace_entry_t *entries,
						int compress)
{
	uint8_t							hbuf[MEMBRANE_ATTNTRACE2_HEADER_SIZE];
	uint8_t							*compact;
	uint8_t							*stored;
	size_t							n;
	size_t							i;
	uLongf							stored_len;
	membrane_attntrace2_header_t	hcopy;

	if (f == NULL || h == NULL || entries == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!header_bounds_ok(h))
		return (MEMBRANE_ERR_INVALID_ARG);
	if (strnlen(h->model, MEMBRANE_ATTNTRACE2_MODEL_CAP)
			>= (size_t)MEMBRANE_ATTNTRACE2_MODEL_CAP)
		return (MEMBRANE_ERR_INVALID_ARG);
	n = membrane_attntrace2_entry_count(h);
	compact = malloc(n * 3);
	if (compact == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < n)
	{
		uint32_t	bid = entries[i].block_id;
		uint8_t		q8;

		if (bid == UINT32_MAX)
			bid = 0xFFFFu;
		else if (bid > MEMBRANE_ATTNTRACE2_MAX_BLOCK_ID)
		{
			free(compact);
			return (MEMBRANE_ERR_INVALID_ARG);
		}
		put_le16(compact + 3 * i, (uint16_t)bid);
		if (bid == 0xFFFFu)
			q8 = 0;
		else
		{
			float	s = entries[i].score;

			if (s < 0.0f)
				s = 0.0f;
			else if (s > 1.0f)
				s = 1.0f;
			q8 = (uint8_t)lroundf(s * 255.0f);
		}
		compact[3 * i + 2] = q8;
		i++;
	}
	hcopy = *h;
	hcopy.uncompressed_payload_size = (uint32_t)(n * 3);
	hcopy.payload_checksum = membrane_block_checksum(compact, n * 3);
	if (compress)
	{
		stored_len = compressBound((uLong)(n * 3));
		stored = malloc(stored_len);
		if (stored == NULL || compress2(stored, &stored_len, compact,
				(uLong)(n * 3), Z_BEST_COMPRESSION) != Z_OK)
		{
			free(compact);
			free(stored);
			return (MEMBRANE_ERR_ALLOC_FAILED);
		}
		hcopy.compressed = 1;
		hcopy.stored_payload_size = (uint32_t)stored_len;
	}
	else
	{
		stored = compact;
		stored_len = (uLong)(n * 3);
		hcopy.compressed = 0;
		hcopy.stored_payload_size = (uint32_t)stored_len;
	}
	serialize_header(hbuf, &hcopy);
	if (fwrite(hbuf, 1, sizeof(hbuf), f) != sizeof(hbuf)
			|| fwrite(stored, 1, stored_len, f) != stored_len)
	{
		free(compact);
		if (stored != compact)
			free(stored);
		return (MEMBRANE_ERR_IO);
	}
	free(compact);
	if (stored != compact)
		free(stored);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace2_read_header(FILE *f,
						membrane_attntrace2_header_t *h)
{
	uint8_t		buf[MEMBRANE_ATTNTRACE2_HEADER_SIZE];
	uint32_t	magic;
	uint32_t	version;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_IO);
	magic = get_le32(buf + 0);
	version = get_le32(buf + 4);
	if (magic != MEMBRANE_ATTNTRACE2_MAGIC || version != MEMBRANE_ATTNTRACE2_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (membrane_block_checksum(buf, 124) != get_le32(buf + 124))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	deserialize_header(buf, h);
	if (!header_bounds_ok(h))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace2_read_entries(FILE *f,
						const membrane_attntrace2_header_t *h,
						membrane_attntrace_entry_t *out)
{
	uint8_t		*stored;
	uint8_t		*compact;
	size_t		n;
	size_t		i;

	if (f == NULL || h == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	n = membrane_attntrace2_entry_count(h);
	if (n == 0 || (size_t)h->uncompressed_payload_size != n * 3)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	stored = malloc(h->stored_payload_size ? h->stored_payload_size : 1);
	if (stored == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(stored, 1, h->stored_payload_size, f) != h->stored_payload_size)
	{
		free(stored);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	if (h->compressed)
	{
		uLongf	compact_len = (uLongf)(n * 3);

		compact = malloc(n * 3);
		if (compact == NULL)
		{
			free(stored);
			return (MEMBRANE_ERR_ALLOC_FAILED);
		}
		if (uncompress(compact, &compact_len, stored,
				(uLong)h->stored_payload_size) != Z_OK
				|| compact_len != n * 3)
		{
			free(stored);
			free(compact);
			return (MEMBRANE_ERR_CORRUPT_DATA);
		}
		free(stored);
	}
	else
		compact = stored;
	if (membrane_block_checksum(compact, n * 3) != h->payload_checksum)
	{
		free(compact);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	i = 0;
	while (i < n)
	{
		uint16_t	bid16 = get_le16(compact + 3 * i);
		uint8_t		q8 = compact[3 * i + 2];

		if (bid16 == 0xFFFFu)
		{
			out[i].block_id = UINT32_MAX;
			out[i].score = 0.0f;
		}
		else
		{
			out[i].block_id = bid16;
			out[i].score = (float)q8 / 255.0f;
		}
		i++;
	}
	free(compact);
	return (MEMBRANE_OK);
}
