#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/kvdump.h"

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

static void	serialize_header(uint8_t *b, const membrane_kv_header_t *h)
{
	int	i;

	put_le32(b + 0, MEMBRANE_KV_MAGIC);
	put_le32(b + 4, MEMBRANE_KV_VERSION);
	memcpy(b + 8, h->model, MEMBRANE_KV_MODEL_CAP);
	put_le32(b + 72, h->layer);
	put_le32(b + 76, h->tensor_type);
	put_le32(b + 80, h->token_start);
	put_le32(b + 84, h->token_end);
	put_le32(b + 88, h->dtype);
	put_le32(b + 92, h->n_dims);
	i = 0;
	while (i < MEMBRANE_KV_MAX_DIMS)
	{
		put_le64(b + 96 + 8 * i, h->dims[i]);
		i++;
	}
	put_le64(b + 128, h->payload_size);
	put_le32(b + 136, h->checksum);
	put_le32(b + 140, membrane_block_checksum(b, 140));
}

static void	deserialize_header(const uint8_t *b, membrane_kv_header_t *h)
{
	int	i;

	memcpy(h->model, b + 8, MEMBRANE_KV_MODEL_CAP);
	h->model[MEMBRANE_KV_MODEL_CAP - 1] = '\0';
	h->layer = get_le32(b + 72);
	h->tensor_type = get_le32(b + 76);
	h->token_start = get_le32(b + 80);
	h->token_end = get_le32(b + 84);
	h->dtype = get_le32(b + 88);
	h->n_dims = get_le32(b + 92);
	i = 0;
	while (i < MEMBRANE_KV_MAX_DIMS)
	{
		h->dims[i] = get_le64(b + 96 + 8 * i);
		i++;
	}
	h->payload_size = get_le64(b + 128);
	h->checksum = get_le32(b + 136);
}

membrane_status_t	membrane_kvdump_write(FILE *f,
						const membrane_kv_header_t *h, const void *payload)
{
	uint8_t	buf[MEMBRANE_KV_HEADER_SIZE];

	if (f == NULL || h == NULL || (payload == NULL && h->payload_size > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	if (h->n_dims > MEMBRANE_KV_MAX_DIMS
		|| h->payload_size > MEMBRANE_KV_MAX_PAYLOAD)
		return (MEMBRANE_ERR_INVALID_ARG);
	serialize_header(buf, h);
	if (fwrite(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_IO);
	if (h->payload_size > 0
		&& fwrite(payload, 1, h->payload_size, f) != h->payload_size)
		return (MEMBRANE_ERR_IO);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_kvdump_read_header(FILE *f,
						membrane_kv_header_t *h)
{
	uint8_t	buf[MEMBRANE_KV_HEADER_SIZE];
	size_t	got;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	got = fread(buf, 1, sizeof(buf), f);
	if (got == 0 && feof(f))
		return (MEMBRANE_ERR_NOT_FOUND);
	if (got != sizeof(buf))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (get_le32(buf + 0) != MEMBRANE_KV_MAGIC
		|| get_le32(buf + 4) != MEMBRANE_KV_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (get_le32(buf + 140) != membrane_block_checksum(buf, 140))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	deserialize_header(buf, h);
	if (h->n_dims > MEMBRANE_KV_MAX_DIMS
		|| h->payload_size > MEMBRANE_KV_MAX_PAYLOAD)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_kvdump_read_payload(FILE *f,
						const membrane_kv_header_t *h, uint8_t **out)
{
	uint8_t	*buf;

	if (f == NULL || h == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	*out = NULL;
	if (h->payload_size == 0)
	{
		if (h->checksum != membrane_block_checksum(NULL, 0))
			return (MEMBRANE_ERR_CORRUPT_DATA);
		return (MEMBRANE_OK);
	}
	buf = malloc(h->payload_size);
	if (buf == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(buf, 1, h->payload_size, f) != h->payload_size)
		return (free(buf), MEMBRANE_ERR_CORRUPT_DATA);
	if (membrane_block_checksum(buf, h->payload_size) != h->checksum)
		return (free(buf), MEMBRANE_ERR_CORRUPT_DATA);
	*out = buf;
	return (MEMBRANE_OK);
}
