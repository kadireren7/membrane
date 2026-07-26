#define _DEFAULT_SOURCE

#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/kvtrace.h"

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

static void	serialize_header(uint8_t *b, const membrane_kvtrace_header_t *h)
{
	put_le32(b + 0, MEMBRANE_KVTRACE_MAGIC);
	put_le32(b + 4, MEMBRANE_KVTRACE_VERSION);
	memcpy(b + 8, h->model, MEMBRANE_KVTRACE_MODEL_CAP);
	put_le32(b + 72, h->source);
	put_le32(b + 76, h->n_layer);
	put_le32(b + 80, h->n_head_kv);
	put_le32(b + 84, h->prompt_len);
	put_le32(b + 88, h->step_count);
	put_le64(b + 92, h->created_unix_time);
	put_le32(b + 100, h->payload_checksum);
	put_le32(b + 104, membrane_block_checksum(b, 104));
}

static void	deserialize_header(const uint8_t *b, membrane_kvtrace_header_t *h)
{
	memcpy(h->model, b + 8, MEMBRANE_KVTRACE_MODEL_CAP);
	h->model[MEMBRANE_KVTRACE_MODEL_CAP - 1] = '\0';
	h->source = get_le32(b + 72);
	h->n_layer = get_le32(b + 76);
	h->n_head_kv = get_le32(b + 80);
	h->prompt_len = get_le32(b + 84);
	h->step_count = get_le32(b + 88);
	h->created_unix_time = get_le64(b + 92);
	h->payload_checksum = get_le32(b + 100);
}

membrane_status_t	membrane_kvtrace_write(FILE *f,
						const membrane_kvtrace_header_t *h,
						const uint32_t *step_bytes)
{
	uint8_t				hbuf[MEMBRANE_KVTRACE_HEADER_SIZE];
	uint8_t				*payload;
	size_t				payload_len;
	membrane_kvtrace_header_t	hcopy;
	uint32_t			i;

	if (f == NULL || h == NULL || step_bytes == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (h->step_count == 0 || h->step_count > MEMBRANE_KVTRACE_MAX_STEPS)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (strnlen(h->model, MEMBRANE_KVTRACE_MODEL_CAP)
			>= (size_t)MEMBRANE_KVTRACE_MODEL_CAP)
		return (MEMBRANE_ERR_INVALID_ARG);
	payload_len = (size_t)h->step_count * 4;
	payload = malloc(payload_len);
	if (payload == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < h->step_count)
	{
		put_le32(payload + 4 * i, step_bytes[i]);
		i++;
	}
	hcopy = *h;
	hcopy.payload_checksum = membrane_block_checksum(payload, payload_len);
	serialize_header(hbuf, &hcopy);
	if (fwrite(hbuf, 1, sizeof(hbuf), f) != sizeof(hbuf)
			|| fwrite(payload, 1, payload_len, f) != payload_len)
	{
		free(payload);
		return (MEMBRANE_ERR_IO);
	}
	free(payload);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_kvtrace_read_header(FILE *f,
						membrane_kvtrace_header_t *h)
{
	uint8_t		buf[MEMBRANE_KVTRACE_HEADER_SIZE];
	uint32_t	magic;
	uint32_t	version;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_IO);
	magic = get_le32(buf + 0);
	version = get_le32(buf + 4);
	if (magic != MEMBRANE_KVTRACE_MAGIC || version != MEMBRANE_KVTRACE_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (membrane_block_checksum(buf, 104) != get_le32(buf + 104))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	deserialize_header(buf, h);
	if (h->step_count == 0 || h->step_count > MEMBRANE_KVTRACE_MAX_STEPS)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_kvtrace_read_steps(FILE *f,
						const membrane_kvtrace_header_t *h, uint32_t *out)
{
	uint8_t		*payload;
	size_t		payload_len;
	uint32_t	i;

	if (f == NULL || h == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	payload_len = (size_t)h->step_count * 4;
	payload = malloc(payload_len);
	if (payload == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(payload, 1, payload_len, f) != payload_len)
	{
		free(payload);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	if (membrane_block_checksum(payload, payload_len) != h->payload_checksum)
	{
		free(payload);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	i = 0;
	while (i < h->step_count)
	{
		out[i] = get_le32(payload + 4 * i);
		i++;
	}
	free(payload);
	return (MEMBRANE_OK);
}
