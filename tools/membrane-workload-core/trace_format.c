#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>

#include "crc32_table.h"
#include "trace_format.h"

static void	put_le16(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

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

static uint16_t	get_le16(const uint8_t *p)
{
	return ((uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8));
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

/* Same algorithm as membrane_block_checksum (src/block/block.c), split
 * into init/update/final so a streaming reader can compute it across
 * many small reads instead of one whole-payload buffer -- verified
 * bit-identical to membrane_block_checksum by
 * test_trace_format_matches_membrane_block_checksum. */
static uint32_t	crc32_init(void)
{
	return (0xFFFFFFFFu);
}

static uint32_t	crc32_update(uint32_t crc, const uint8_t *buf, size_t len)
{
	size_t	i;

	i = 0;
	while (i < len)
	{
		crc = g_crc_table[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
		i++;
	}
	return (crc);
}

static uint32_t	crc32_final(uint32_t crc)
{
	return (crc ^ 0xFFFFFFFFu);
}

static size_t	dtype_element_size(uint32_t dtype)
{
	if (dtype == MEMBRANE_TRACE_DTYPE_F16)
		return (sizeof(uint16_t));
	return (0);
}

struct s_membrane_trace_reader
{
	FILE				*f;
	membrane_trace_info_t	info;
	uint64_t			blocks_remaining;
	uint32_t			running_crc;
	uint32_t			expected_payload_checksum;
};

/* Overflow-checked a*b: 0 (and *out untouched) if the product would
 * overflow uint64_t, 1 and *out set otherwise. */
static int	checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
	if (a != 0 && b > UINT64_MAX / a)
		return (0);
	*out = a * b;
	return (1);
}

static membrane_status_t	validate_header_fields(
								const membrane_trace_info_t *info)
{
	uint64_t	expect_payload;

	if (info->format_version != MEMBRANE_TRACE_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (dtype_element_size(info->dtype) == 0)
		return (MEMBRANE_ERR_UNIMPLEMENTED);
	if (info->elements_per_block == 0
			|| info->elements_per_block > MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (info->block_count == 0
			|| info->block_count > MEMBRANE_TRACE_MAX_BLOCK_COUNT)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (!checked_mul_u64(info->block_count, info->elements_per_block,
			&expect_payload)
			|| !checked_mul_u64(expect_payload,
				dtype_element_size(info->dtype), &expect_payload))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (expect_payload != info->payload_bytes
			|| expect_payload > MEMBRANE_TRACE_MAX_PAYLOAD_BYTES)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

static membrane_status_t	read_and_validate_header(FILE *f,
								membrane_trace_info_t *info,
								uint32_t *metadata_length,
								uint32_t *payload_checksum)
{
	uint8_t				buf[MEMBRANE_TRACE_HEADER_SIZE];
	uint32_t			header_crc;
	uint64_t			reserved;

	if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (memcmp(buf, MEMBRANE_TRACE_MAGIC, 7) != 0 || buf[7] != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	header_crc = crc32_final(crc32_update(crc32_init(), buf, 52));
	if (header_crc != get_le32(buf + 52))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	reserved = get_le64(buf + 56);
	if (reserved != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	memset(info, 0, sizeof(*info));
	info->format_version = get_le32(buf + 8);
	info->dtype = get_le32(buf + 12);
	info->elements_per_block = get_le32(buf + 16);
	*metadata_length = get_le32(buf + 20);
	info->block_count = get_le64(buf + 24);
	info->payload_bytes = get_le64(buf + 32);
	info->created_unix_time = get_le64(buf + 40);
	*payload_checksum = get_le32(buf + 48);
	if (*metadata_length > MEMBRANE_TRACE_MAX_METADATA)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (validate_header_fields(info));
}

static membrane_status_t	read_metadata(FILE *f,
								membrane_trace_info_t *info,
								uint32_t metadata_length)
{
	uint8_t		*raw;
	uint32_t	keep;

	if (metadata_length == 0)
		return (MEMBRANE_OK);
	raw = malloc(metadata_length);
	if (raw == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(raw, 1, metadata_length, f) != metadata_length)
		return (free(raw), MEMBRANE_ERR_CORRUPT_DATA);
	keep = metadata_length;
	if (keep > MEMBRANE_TRACE_METADATA_CAP - 1)
		keep = MEMBRANE_TRACE_METADATA_CAP - 1;
	memcpy(info->metadata, raw, keep);
	info->metadata[keep] = '\0';
	free(raw);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_trace_open(const char *path,
						membrane_trace_reader_t **out)
{
	FILE					*f;
	membrane_trace_reader_t	*r;
	uint32_t				metadata_length;
	uint32_t				payload_checksum;
	membrane_status_t		st;

	if (path == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	f = fopen(path, "rb");
	if (f == NULL)
		return (MEMBRANE_ERR_IO);
	r = calloc(1, sizeof(*r));
	if (r == NULL)
		return (fclose(f), MEMBRANE_ERR_ALLOC_FAILED);
	st = read_and_validate_header(f, &r->info, &metadata_length,
			&payload_checksum);
	if (st == MEMBRANE_OK)
		st = read_metadata(f, &r->info, metadata_length);
	if (st != MEMBRANE_OK)
		return (fclose(f), free(r), st);
	r->f = f;
	r->blocks_remaining = r->info.block_count;
	r->running_crc = crc32_init();
	r->expected_payload_checksum = payload_checksum;
	*out = r;
	return (MEMBRANE_OK);
}

void	membrane_trace_info(const membrane_trace_reader_t *r,
			membrane_trace_info_t *info)
{
	*info = r->info;
}

membrane_status_t	membrane_trace_read_block(membrane_trace_reader_t *r,
						uint16_t *out_f16)
{
	uint8_t		*raw;
	size_t		raw_len;
	uint32_t	i;

	if (r == NULL || out_f16 == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (r->blocks_remaining == 0)
		return (MEMBRANE_ERR_NOT_FOUND);
	raw_len = (size_t)r->info.elements_per_block * sizeof(uint16_t);
	raw = malloc(raw_len);
	if (raw == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fread(raw, 1, raw_len, r->f) != raw_len)
		return (free(raw), MEMBRANE_ERR_CORRUPT_DATA);
	i = 0;
	while (i < r->info.elements_per_block)
	{
		out_f16[i] = get_le16(raw + 2 * i);
		i++;
	}
	r->running_crc = crc32_update(r->running_crc, raw, raw_len);
	free(raw);
	r->blocks_remaining--;
	if (r->blocks_remaining == 0
			&& crc32_final(r->running_crc) != r->expected_payload_checksum)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

void	membrane_trace_close(membrane_trace_reader_t *r)
{
	if (r == NULL)
		return ;
	if (r->f != NULL)
		fclose(r->f);
	free(r);
}

membrane_status_t	membrane_trace_write(FILE *f,
						membrane_trace_dtype_t dtype,
						uint32_t elements_per_block, uint64_t block_count,
						const uint16_t *blocks_f16, const char *metadata,
						uint64_t created_unix_time)
{
	uint8_t		header[MEMBRANE_TRACE_HEADER_SIZE];
	uint8_t		*payload;
	uint64_t	total_elems;
	uint64_t	payload_bytes;
	uint32_t	metadata_length;
	uint32_t	payload_crc;
	uint64_t	i;

	if (f == NULL || blocks_f16 == NULL || elements_per_block == 0
			|| elements_per_block > MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK
			|| block_count == 0 || block_count > MEMBRANE_TRACE_MAX_BLOCK_COUNT)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (!checked_mul_u64(block_count, elements_per_block, &total_elems)
			|| !checked_mul_u64(total_elems, sizeof(uint16_t), &payload_bytes)
			|| payload_bytes > MEMBRANE_TRACE_MAX_PAYLOAD_BYTES)
		return (MEMBRANE_ERR_INVALID_ARG);
	metadata_length = 0;
	if (metadata != NULL)
		metadata_length = (uint32_t)strnlen(metadata,
				MEMBRANE_TRACE_MAX_METADATA + 1);
	if (metadata_length > MEMBRANE_TRACE_MAX_METADATA)
		return (MEMBRANE_ERR_INVALID_ARG);
	payload = malloc((size_t)payload_bytes);
	if (payload == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	i = 0;
	while (i < total_elems)
	{
		put_le16(payload + 2 * i, blocks_f16[i]);
		i++;
	}
	payload_crc = crc32_final(crc32_update(crc32_init(), payload,
				(size_t)payload_bytes));
	memset(header, 0, sizeof(header));
	memcpy(header, MEMBRANE_TRACE_MAGIC, 7);
	put_le32(header + 8, MEMBRANE_TRACE_VERSION);
	put_le32(header + 12, (uint32_t)dtype);
	put_le32(header + 16, elements_per_block);
	put_le32(header + 20, metadata_length);
	put_le64(header + 24, block_count);
	put_le64(header + 32, payload_bytes);
	put_le64(header + 40, created_unix_time);
	put_le32(header + 48, payload_crc);
	put_le32(header + 52, crc32_final(crc32_update(crc32_init(), header, 52)));
	if (fwrite(header, 1, sizeof(header), f) != sizeof(header)
			|| (metadata_length > 0
				&& fwrite(metadata, 1, metadata_length, f) != metadata_length)
			|| fwrite(payload, 1, (size_t)payload_bytes, f) != payload_bytes)
		return (free(payload), MEMBRANE_ERR_IO);
	free(payload);
	return (MEMBRANE_OK);
}
