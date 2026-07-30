#define _DEFAULT_SOURCE

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include "membrane/attntrace3.h"
#include "membrane/block.h"
#include "membrane/hash.h"

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

static uint32_t	chunk_count_for(uint32_t step_count, uint32_t chunk_steps)
{
	return ((step_count + chunk_steps - 1) / chunk_steps);
}

static int	header_bounds_ok(const membrane_attntrace3_header_t *h)
{
	uint32_t	expect_chunks;

	if (h->step_count == 0 || h->step_count > MEMBRANE_ATTNTRACE3_MAX_STEPS)
		return (0);
	if (h->n_layer == 0 || h->n_layer > MEMBRANE_ATTNTRACE3_MAX_LAYER)
		return (0);
	if (h->n_head == 0 || h->n_head > MEMBRANE_ATTNTRACE3_MAX_HEAD)
		return (0);
	if (h->top_k == 0 || h->top_k > MEMBRANE_ATTNTRACE3_MAX_TOPK)
		return (0);
	if (h->chunk_steps == 0
			|| h->chunk_steps > MEMBRANE_ATTNTRACE3_MAX_CHUNK_STEPS)
		return (0);
	expect_chunks = chunk_count_for(h->step_count, h->chunk_steps);
	if (h->chunk_count != expect_chunks)
		return (0);
	if (h->index_offset != MEMBRANE_ATTNTRACE3_HEADER_SIZE)
		return (0);
	if (h->index_size != (uint64_t)h->chunk_count
			* MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE)
		return (0);
	if (h->payload_offset != h->index_offset + h->index_size)
		return (0);
	return (1);
}

static void	serialize_header(uint8_t *b,
				const membrane_attntrace3_header_t *h)
{
	memset(b, 0, MEMBRANE_ATTNTRACE3_HEADER_SIZE);
	put_le32(b + 0, MEMBRANE_ATTNTRACE3_MAGIC);
	put_le32(b + 4, MEMBRANE_ATTNTRACE3_VERSION);
	memcpy(b + 8, h->model, MEMBRANE_ATTNTRACE3_MODEL_CAP);
	put_le32(b + 72, h->source);
	put_le32(b + 76, h->n_layer);
	put_le32(b + 80, h->n_head);
	put_le32(b + 84, h->block_size_tokens);
	put_le32(b + 88, h->prompt_len);
	put_le32(b + 92, h->step_count);
	put_le32(b + 96, h->top_k);
	put_le64(b + 100, h->created_unix_time);
	put_le32(b + 108, h->chunk_steps);
	put_le32(b + 112, h->chunk_count);
	put_le64(b + 116, h->index_offset);
	put_le64(b + 124, h->index_size);
	put_le64(b + 132, h->payload_offset);
	memcpy(b + 140, h->file_sha256, 32);
	put_le32(b + 172, membrane_block_checksum(b, 172));
}

static void	deserialize_header(const uint8_t *b,
				membrane_attntrace3_header_t *h)
{
	memcpy(h->model, b + 8, MEMBRANE_ATTNTRACE3_MODEL_CAP);
	h->model[MEMBRANE_ATTNTRACE3_MODEL_CAP - 1] = '\0';
	h->source = get_le32(b + 72);
	h->n_layer = get_le32(b + 76);
	h->n_head = get_le32(b + 80);
	h->block_size_tokens = get_le32(b + 84);
	h->prompt_len = get_le32(b + 88);
	h->step_count = get_le32(b + 92);
	h->top_k = get_le32(b + 96);
	h->created_unix_time = get_le64(b + 100);
	h->chunk_steps = get_le32(b + 108);
	h->chunk_count = get_le32(b + 112);
	h->index_offset = get_le64(b + 116);
	h->index_size = get_le64(b + 124);
	h->payload_offset = get_le64(b + 132);
	memcpy(h->file_sha256, b + 140, 32);
}

static void	serialize_index_entry(uint8_t *b,
				const membrane_attntrace3_chunk_index_entry_t *e)
{
	memset(b, 0, MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE);
	put_le32(b + 0, e->step_lo);
	put_le32(b + 4, e->step_count_in_chunk);
	put_le64(b + 8, e->payload_offset);
	put_le32(b + 16, e->stored_size);
	put_le32(b + 20, e->uncompressed_size);
	put_le32(b + 24, e->crc32);
	b[28] = e->compressed;
}

static void	deserialize_index_entry(const uint8_t *b,
				membrane_attntrace3_chunk_index_entry_t *e)
{
	e->step_lo = get_le32(b + 0);
	e->step_count_in_chunk = get_le32(b + 4);
	e->payload_offset = get_le64(b + 8);
	e->stored_size = get_le32(b + 16);
	e->uncompressed_size = get_le32(b + 20);
	e->crc32 = get_le32(b + 24);
	e->compressed = b[28];
}

uint32_t	membrane_attntrace3_writer_chunk_len(
				const membrane_attntrace3_writer_t *w, uint32_t chunk_id)
{
	uint32_t	lo;

	if (w == NULL || chunk_id >= w->h.chunk_count)
		return (0);
	lo = chunk_id * w->h.chunk_steps;
	if (lo + w->h.chunk_steps > w->h.step_count)
		return (w->h.step_count - lo);
	return (w->h.chunk_steps);
}

size_t	membrane_attntrace3_chunk_entry_count(
			const membrane_attntrace3_header_t *h,
			const membrane_attntrace3_chunk_index_entry_t *entry)
{
	if (h == NULL || entry == NULL)
		return (0);
	return ((size_t)entry->step_count_in_chunk * h->n_layer * h->n_head
		* h->top_k);
}

uint32_t	membrane_attntrace3_step_to_chunk(
				const membrane_attntrace3_header_t *h, uint32_t step)
{
	uint32_t	id;

	if (h == NULL || h->chunk_steps == 0)
		return (0);
	id = step / h->chunk_steps;
	if (id >= h->chunk_count)
		id = h->chunk_count > 0 ? h->chunk_count - 1 : 0;
	return (id);
}

membrane_status_t	membrane_attntrace3_writer_open(FILE *f,
						membrane_attntrace3_writer_t *w,
						const char *model, uint32_t source,
						uint32_t n_layer, uint32_t n_head,
						uint32_t block_size_tokens, uint32_t prompt_len,
						uint32_t step_count, uint32_t top_k,
						uint32_t chunk_steps)
{
	uint8_t		hbuf[MEMBRANE_ATTNTRACE3_HEADER_SIZE];
	uint8_t		ibuf[MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE];
	uint32_t	i;

	if (f == NULL || w == NULL || model == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (strnlen(model, MEMBRANE_ATTNTRACE3_MODEL_CAP)
			>= (size_t)MEMBRANE_ATTNTRACE3_MODEL_CAP)
		return (MEMBRANE_ERR_INVALID_ARG);
	memset(w, 0, sizeof(*w));
	memset(w->h.model, 0, sizeof(w->h.model));
	strncpy(w->h.model, model, MEMBRANE_ATTNTRACE3_MODEL_CAP - 1);
	w->h.source = source;
	w->h.n_layer = n_layer;
	w->h.n_head = n_head;
	w->h.block_size_tokens = block_size_tokens;
	w->h.prompt_len = prompt_len;
	w->h.step_count = step_count;
	w->h.top_k = top_k;
	w->h.chunk_steps = chunk_steps;
	w->h.chunk_count = chunk_count_for(step_count, chunk_steps);
	w->h.index_offset = MEMBRANE_ATTNTRACE3_HEADER_SIZE;
	w->h.index_size = (uint64_t)w->h.chunk_count
		* MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE;
	w->h.payload_offset = w->h.index_offset + w->h.index_size;
	if (!header_bounds_ok(&w->h))
		return (MEMBRANE_ERR_INVALID_ARG);
	serialize_header(hbuf, &w->h);
	if (fseek(f, 0, SEEK_SET) != 0
			|| fwrite(hbuf, 1, sizeof(hbuf), f) != sizeof(hbuf))
		return (MEMBRANE_ERR_IO);
	memset(ibuf, 0, sizeof(ibuf));
	i = 0;
	while (i < w->h.chunk_count)
	{
		if (fwrite(ibuf, 1, sizeof(ibuf), f) != sizeof(ibuf))
			return (MEMBRANE_ERR_IO);
		i++;
	}
	if (fflush(f) != 0)
		return (MEMBRANE_ERR_IO);
	w->next_payload_offset = w->h.payload_offset;
	w->next_chunk_id = 0;
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_writer_put_chunk(FILE *f,
						membrane_attntrace3_writer_t *w, uint32_t chunk_id,
						const membrane_attntrace_entry_t *entries,
						int compress)
{
	uint32_t							chunk_len;
	size_t								n;
	size_t								i;
	uint8_t								*compact;
	uint8_t								*stored;
	uLongf								stored_len;
	membrane_attntrace3_chunk_index_entry_t	e;
	uint8_t								ibuf[MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE];

	if (f == NULL || w == NULL || entries == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (chunk_id != w->next_chunk_id || chunk_id >= w->h.chunk_count)
		return (MEMBRANE_ERR_INVALID_ARG);
	chunk_len = membrane_attntrace3_writer_chunk_len(w, chunk_id);
	n = (size_t)chunk_len * w->h.n_layer * w->h.n_head * w->h.top_k;
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
		else if (bid > MEMBRANE_ATTNTRACE3_MAX_BLOCK_ID)
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
	memset(&e, 0, sizeof(e));
	e.step_lo = chunk_id * w->h.chunk_steps;
	e.step_count_in_chunk = chunk_len;
	e.payload_offset = w->next_payload_offset;
	e.uncompressed_size = (uint32_t)(n * 3);
	e.crc32 = membrane_block_checksum(compact, n * 3);
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
		e.compressed = 1;
		e.stored_size = (uint32_t)stored_len;
	}
	else
	{
		stored = compact;
		stored_len = (uLong)(n * 3);
		e.compressed = 0;
		e.stored_size = (uint32_t)stored_len;
	}
	if (fseek(f, (long)e.payload_offset, SEEK_SET) != 0
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
	serialize_index_entry(ibuf, &e);
	if (fseek(f, (long)(w->h.index_offset
				+ (uint64_t)chunk_id * MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE),
			SEEK_SET) != 0
			|| fwrite(ibuf, 1, sizeof(ibuf), f) != sizeof(ibuf))
		return (MEMBRANE_ERR_IO);
	if (fflush(f) != 0)
		return (MEMBRANE_ERR_IO);
	w->next_payload_offset += stored_len;
	w->next_chunk_id++;
	return (MEMBRANE_OK);
}

static membrane_status_t	stream_hash_range(FILE *f, uint64_t from,
								uint8_t out_digest[MEMBRANE_SHA256_DIGEST_BYTES])
{
	membrane_sha256_ctx_t	ctx;
	uint8_t					buf[65536];
	long					total;
	uint64_t				remaining;
	size_t					want;
	size_t					got;

	if (fseek(f, 0, SEEK_END) != 0)
		return (MEMBRANE_ERR_IO);
	total = ftell(f);
	if (total < 0 || (uint64_t)total < from)
		return (MEMBRANE_ERR_IO);
	remaining = (uint64_t)total - from;
	if (fseek(f, (long)from, SEEK_SET) != 0)
		return (MEMBRANE_ERR_IO);
	membrane_sha256_init(&ctx);
	while (remaining > 0)
	{
		want = sizeof(buf);
		if ((uint64_t)want > remaining)
			want = (size_t)remaining;
		got = fread(buf, 1, want, f);
		if (got == 0)
			return (MEMBRANE_ERR_IO);
		membrane_sha256_update(&ctx, buf, got);
		remaining -= got;
	}
	membrane_sha256_final(&ctx, out_digest);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_writer_close(FILE *f,
						membrane_attntrace3_writer_t *w)
{
	uint8_t				hbuf[MEMBRANE_ATTNTRACE3_HEADER_SIZE];
	membrane_status_t	st;

	if (f == NULL || w == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (w->next_chunk_id != w->h.chunk_count)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fflush(f) != 0)
		return (MEMBRANE_ERR_IO);
	st = stream_hash_range(f, w->h.index_offset, w->h.file_sha256);
	if (st != MEMBRANE_OK)
		return (st);
	serialize_header(hbuf, &w->h);
	if (fseek(f, 0, SEEK_SET) != 0
			|| fwrite(hbuf, 1, sizeof(hbuf), f) != sizeof(hbuf))
		return (MEMBRANE_ERR_IO);
	if (fflush(f) != 0)
		return (MEMBRANE_ERR_IO);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_read_header(FILE *f,
						membrane_attntrace3_header_t *h)
{
	uint8_t		buf[MEMBRANE_ATTNTRACE3_HEADER_SIZE];
	uint32_t	magic;
	uint32_t	version;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
		return (MEMBRANE_ERR_IO);
	magic = get_le32(buf + 0);
	version = get_le32(buf + 4);
	if (magic != MEMBRANE_ATTNTRACE3_MAGIC
			|| version != MEMBRANE_ATTNTRACE3_VERSION)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	if (membrane_block_checksum(buf, 172) != get_le32(buf + 172))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	deserialize_header(buf, h);
	if (!header_bounds_ok(h))
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_read_index(FILE *f,
						const membrane_attntrace3_header_t *h,
						membrane_attntrace3_chunk_index_entry_t *out)
{
	uint8_t		buf[MEMBRANE_ATTNTRACE3_INDEX_ENTRY_SIZE];
	uint32_t	i;
	uint32_t	expect_lo;

	if (f == NULL || h == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (fseek(f, (long)h->index_offset, SEEK_SET) != 0)
		return (MEMBRANE_ERR_IO);
	i = 0;
	while (i < h->chunk_count)
	{
		if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
			return (MEMBRANE_ERR_IO);
		deserialize_index_entry(buf, &out[i]);
		expect_lo = i * h->chunk_steps;
		if (out[i].step_lo != expect_lo || out[i].step_count_in_chunk == 0
				|| out[i].step_lo + out[i].step_count_in_chunk
					> h->step_count)
			return (MEMBRANE_ERR_CORRUPT_DATA);
		i++;
	}
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_decode_chunk_buf(
						const uint8_t *stored,
						const membrane_attntrace3_chunk_index_entry_t *entry,
						membrane_attntrace_entry_t *out)
{
	uint8_t		*compact;
	int			owns_compact;
	size_t		n;
	size_t		i;

	if (stored == NULL || entry == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	if (entry->uncompressed_size == 0 || entry->uncompressed_size % 3 != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	n = (size_t)entry->uncompressed_size / 3;
	owns_compact = 0;
	if (entry->compressed)
	{
		uLongf	compact_len = (uLongf)entry->uncompressed_size;

		compact = malloc(entry->uncompressed_size);
		if (compact == NULL)
			return (MEMBRANE_ERR_ALLOC_FAILED);
		owns_compact = 1;
		if (uncompress(compact, &compact_len, stored,
				(uLong)entry->stored_size) != Z_OK
				|| compact_len != entry->uncompressed_size)
		{
			free(compact);
			return (MEMBRANE_ERR_CORRUPT_DATA);
		}
	}
	else
	{
		if (entry->stored_size != entry->uncompressed_size)
		{
			return (MEMBRANE_ERR_CORRUPT_DATA);
		}
		compact = (uint8_t *)stored;
	}
	if (membrane_block_checksum(compact, n * 3) != entry->crc32)
	{
		if (owns_compact)
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
	if (owns_compact)
		free(compact);
	return (MEMBRANE_OK);
}

membrane_status_t	membrane_attntrace3_read_chunk(FILE *f,
						const membrane_attntrace3_header_t *h,
						const membrane_attntrace3_chunk_index_entry_t *entry,
						membrane_attntrace_entry_t *out)
{
	uint8_t				*stored;
	membrane_status_t	st;

	(void)h;
	if (f == NULL || entry == NULL || out == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	stored = malloc(entry->stored_size ? entry->stored_size : 1);
	if (stored == NULL)
		return (MEMBRANE_ERR_ALLOC_FAILED);
	if (fseek(f, (long)entry->payload_offset, SEEK_SET) != 0
			|| fread(stored, 1, entry->stored_size, f) != entry->stored_size)
	{
		free(stored);
		return (MEMBRANE_ERR_CORRUPT_DATA);
	}
	st = membrane_attntrace3_decode_chunk_buf(stored, entry, out);
	free(stored);
	return (st);
}

membrane_status_t	membrane_attntrace3_verify_file_sha256(FILE *f,
						const membrane_attntrace3_header_t *h)
{
	uint8_t				digest[32];
	membrane_status_t	st;

	if (f == NULL || h == NULL)
		return (MEMBRANE_ERR_INVALID_ARG);
	st = stream_hash_range(f, h->index_offset, digest);
	if (st != MEMBRANE_OK)
		return (st);
	if (memcmp(digest, h->file_sha256, 32) != 0)
		return (MEMBRANE_ERR_CORRUPT_DATA);
	return (MEMBRANE_OK);
}
