#include <stdlib.h>
#include <string.h>

#include "membrane/block.h"
#include "membrane/stats.h"

/* CRC32, poly 0xEDB88320 (reflected), table built lazily on first use. */
static uint32_t	g_crc_table[256];
static int		g_crc_table_ready = 0;

static void	crc32_init_table(void)
{
	uint32_t	c;
	uint32_t	i;
	int			k;

	i = 0;
	while (i < 256)
	{
		c = i;
		k = 0;
		while (k < 8)
		{
			if (c & 1)
				c = 0xEDB88320u ^ (c >> 1);
			else
				c >>= 1;
			k++;
		}
		g_crc_table[i] = c;
		i++;
	}
	g_crc_table_ready = 1;
}

uint32_t	membrane_block_checksum(const uint8_t *buf, size_t len)
{
	uint32_t	crc;
	size_t		i;

	if (!g_crc_table_ready)
		crc32_init_table();
	crc = 0xFFFFFFFFu;
	i = 0;
	while (i < len)
	{
		crc = g_crc_table[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
		i++;
	}
	return (crc ^ 0xFFFFFFFFu);
}

membrane_block_t	*membrane_block_create(uint64_t id, membrane_codec_t codec)
{
	membrane_block_t	*block;

	if (membrane_codec_get(codec) == NULL)
		return (NULL);
	block = calloc(1, sizeof(*block));
	if (block == NULL)
		return (NULL);
	block->id = id;
	block->requested_codec = codec;
	block->stored_codec = codec;
	block->state = MEMBRANE_BLOCK_HOT;
	return (block);
}

void	membrane_block_destroy(membrane_block_t *block)
{
	if (block == NULL)
		return ;
	free(block->data);
	free(block);
}

static void	block_touch(membrane_block_t *block)
{
	block->access_count++;
	block->last_access_ns = membrane_now_ns();
}

static membrane_status_t	block_write_empty(membrane_block_t *block)
{
	free(block->data);
	block->data = NULL;
	block->original_size = 0;
	block->stored_size = 0;
	block->stored_codec = block->requested_codec;
	block->checksum = membrane_block_checksum(NULL, 0);
	block_touch(block);
	return (MEMBRANE_OK);
}

static void	block_adopt(membrane_block_t *block, uint8_t *buf, size_t stored,
				membrane_codec_t stored_codec)
{
	uint8_t	*shrunk;

	shrunk = realloc(buf, stored);
	if (shrunk != NULL)
		buf = shrunk;
	free(block->data);
	block->data = buf;
	block->stored_size = stored;
	block->stored_codec = stored_codec;
}

static membrane_status_t	block_compress(membrane_block_t *block,
				const membrane_codec_vtable_t *codec,
				const uint8_t *in, size_t in_len)
{
	uint8_t				*buf;
	size_t				bound;
	size_t				stored;
	membrane_status_t	status;

	bound = codec->bound(in_len);
	if (bound < in_len)
		return (MEMBRANE_ERR_INVALID_ARG);
	buf = malloc(bound);
	if (buf == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	status = codec->compress(in, in_len, buf, bound, &stored);
	if (status != MEMBRANE_OK)
	{
		free(buf);
		return (status);
	}
	if (stored >= in_len && codec->id != MEMBRANE_CODEC_RAW)
	{
		memcpy(buf, in, in_len);
		block_adopt(block, buf, in_len, MEMBRANE_CODEC_RAW);
	}
	else
		block_adopt(block, buf, stored, codec->id);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_block_write(membrane_block_t *block,
				const uint8_t *in, size_t in_len)
{
	const membrane_codec_vtable_t	*codec;
	membrane_status_t				status;

	if (block == NULL || (in == NULL && in_len > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	codec = membrane_codec_get(block->requested_codec);
	if (codec == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (in_len == 0)
		return (block_write_empty(block));
	status = block_compress(block, codec, in, in_len);
	if (status != MEMBRANE_OK)
		return (status);
	block->original_size = in_len;
	block->checksum = membrane_block_checksum(in, in_len);
	block_touch(block);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_block_read(membrane_block_t *block,
				uint8_t *out, size_t out_cap, size_t *out_len)
{
	const membrane_codec_vtable_t	*codec;
	size_t							produced;
	membrane_status_t				status;

	if (block == NULL || out_len == NULL || (out == NULL && out_cap > 0))
		return (MEMBRANE_ERR_INVALID_ARG);
	codec = membrane_codec_get(block->stored_codec);
	if (codec == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (out_cap < block->original_size)
		return (MEMBRANE_ERR_BUFFER_TOO_SMALL);
	status = codec->decompress(block->data, block->stored_size,
			out, out_cap, &produced);
	if (status != MEMBRANE_OK)
		return (status);
	if (produced != block->original_size
		|| membrane_block_checksum(out, produced) != block->checksum)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	*out_len = produced;
	block_touch(block);
	return (MEMBRANE_OK);
}
