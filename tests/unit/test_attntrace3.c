#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/attntrace3.h"
#include "test_helpers.h"

static char	g_path[] = "/tmp/membrane-attntrace3-XXXXXX";

static void	fill_entries(membrane_attntrace_entry_t *e, size_t n,
				size_t base)
{
	size_t	i;

	i = 0;
	while (i < n)
	{
		if ((base + i) % 7 == 0)
		{
			e[i].block_id = UINT32_MAX;
			e[i].score = 0.0f;
		}
		else
		{
			e[i].block_id = (uint32_t)((base + i) % 4000);
			e[i].score = (float)((base + i) % 101) / 100.0f;
		}
		i++;
	}
}

/* Writes a full trace one chunk at a time (never more than one chunk's
 * entries resident at once), same discipline the real streaming
 * synthetic generator will follow. */
static void	write_full_trace(uint32_t step_count, uint32_t n_layer,
				uint32_t n_head, uint32_t top_k, uint32_t chunk_steps,
				int compress)
{
	membrane_attntrace3_writer_t	w;
	FILE							*f;
	uint32_t						chunk_id;
	size_t							base_step;

	f = fopen(g_path, "w+b");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace3_writer_open(f, &w, "SmolLM2-135M-test",
		MEMBRANE_ATTNTRACE_SOURCE_REAL_CAPTURE, n_layer, n_head, 32, 512,
		step_count, top_k, chunk_steps) == MEMBRANE_OK, "writer_open");
	base_step = 0;
	chunk_id = 0;
	while (chunk_id < w.h.chunk_count)
	{
		uint32_t					clen = membrane_attntrace3_writer_chunk_len(
			&w, chunk_id);
		size_t						n = (size_t)clen * n_layer * n_head
			* top_k;
		membrane_attntrace_entry_t	*chunk = malloc(n * sizeof(*chunk));

		TEST_ASSERT(chunk != NULL, "alloc one chunk's entries");
		fill_entries(chunk, n, base_step * n_layer * n_head * top_k);
		TEST_ASSERT(membrane_attntrace3_writer_put_chunk(f, &w, chunk_id,
			chunk, compress) == MEMBRANE_OK, "put_chunk");
		free(chunk);
		base_step += clen;
		chunk_id++;
	}
	TEST_ASSERT(membrane_attntrace3_writer_close(f, &w) == MEMBRANE_OK,
		"writer_close");
	fclose(f);
}

static void	roundtrip_case(int compress)
{
	membrane_attntrace3_header_t				h;
	membrane_attntrace3_chunk_index_entry_t	*idx;
	FILE										*f;
	uint32_t									step_count = 130;
	uint32_t									n_layer = 4;
	uint32_t									n_head = 3;
	uint32_t									top_k = 8;
	uint32_t									chunk_steps = 32;
	uint32_t									c;

	write_full_trace(step_count, n_layer, n_head, top_k, chunk_steps,
		compress);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "open trace for read");
	TEST_ASSERT(membrane_attntrace3_read_header(f, &h) == MEMBRANE_OK,
		"header read succeeds");
	TEST_ASSERT(strcmp(h.model, "SmolLM2-135M-test") == 0,
		"model name round-trips");
	TEST_ASSERT(h.step_count == step_count, "step_count round-trips");
	TEST_ASSERT(h.chunk_count == (step_count + chunk_steps - 1)
		/ chunk_steps, "chunk_count derived correctly");
	TEST_ASSERT(membrane_attntrace3_verify_file_sha256(f, &h) == MEMBRANE_OK,
		"file-level SHA-256 verifies on an untouched file");

	idx = malloc(h.chunk_count * sizeof(*idx));
	TEST_ASSERT(idx != NULL, "alloc index");
	TEST_ASSERT(membrane_attntrace3_read_index(f, &h, idx) == MEMBRANE_OK,
		"index read succeeds");

	c = 0;
	while (c < h.chunk_count)
	{
		size_t						n = membrane_attntrace3_chunk_entry_count(
			&h, &idx[c]);
		membrane_attntrace_entry_t	*expect = malloc(n * sizeof(*expect));
		membrane_attntrace_entry_t	*got = malloc(n * sizeof(*got));
		size_t						base
			= (size_t)idx[c].step_lo * n_layer * n_head * top_k;
		size_t						i;

		TEST_ASSERT(expect != NULL && got != NULL, "alloc compare buffers");
		fill_entries(expect, n, base);
		TEST_ASSERT((int)idx[c].compressed == compress,
			"per-chunk compressed flag matches writer request");
		TEST_ASSERT(membrane_attntrace3_read_chunk(f, &h, &idx[c], got)
			== MEMBRANE_OK, "chunk read succeeds");
		i = 0;
		while (i < n)
		{
			TEST_ASSERT(got[i].block_id == expect[i].block_id,
				"block_id round-trips exactly");
			float	diff = got[i].score - expect[i].score;
			if (diff < 0)
				diff = -diff;
			TEST_ASSERT(diff <= (1.0f / 255.0f) + 1e-6f,
				"score round-trips within one quantization step");
			i++;
		}
		free(expect);
		free(got);
		c++;
	}
	free(idx);
	fclose(f);
	unlink(g_path);
}

static void	test_roundtrip_uncompressed(void)
{
	roundtrip_case(0);
	printf("PASS test_roundtrip_uncompressed\n");
}

static void	test_roundtrip_compressed(void)
{
	roundtrip_case(1);
	printf("PASS test_roundtrip_compressed\n");
}

static void	test_out_of_order_chunk_write_rejected(void)
{
	membrane_attntrace3_writer_t	w;
	FILE							*f;
	/* chunk_steps(4) * n_layer(4) * n_head(3) * top_k(8) */
	membrane_attntrace_entry_t		chunk[4 * 4 * 3 * 8];

	f = fopen(g_path, "w+b");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace3_writer_open(f, &w, "m",
		MEMBRANE_ATTNTRACE_SOURCE_SYNTHETIC, 4, 3, 32, 0, 8, 8, 4)
		== MEMBRANE_OK, "writer_open (2 chunks)");
	fill_entries(chunk, sizeof(chunk) / sizeof(chunk[0]), 0);
	TEST_ASSERT(membrane_attntrace3_writer_put_chunk(f, &w, 1, chunk, 0)
		== MEMBRANE_ERR_INVALID_ARG,
		"writing chunk 1 before chunk 0 is rejected (deterministic order)");
	TEST_ASSERT(membrane_attntrace3_writer_put_chunk(f, &w, 0, chunk, 0)
		== MEMBRANE_OK, "writing chunk 0 first succeeds");
	fclose(f);
	unlink(g_path);
	printf("PASS test_out_of_order_chunk_write_rejected\n");
}

/* One chunk's CRC32 must be independently verifiable: corrupting chunk
 * 1's payload must be rejected when reading chunk 1, but must NOT
 * disturb reading chunk 0 -- the whole point of per-chunk checksums
 * over one whole-file checksum. */
static void	test_independent_chunk_corruption(void)
{
	membrane_attntrace3_header_t				h;
	membrane_attntrace3_chunk_index_entry_t	*idx;
	membrane_attntrace_entry_t					*buf0;
	membrane_attntrace_entry_t					*buf1;
	FILE										*f;
	uint32_t									step_count = 64;
	uint32_t									n_layer = 2;
	uint32_t									n_head = 2;
	uint32_t									top_k = 4;
	uint32_t									chunk_steps = 16;

	write_full_trace(step_count, n_layer, n_head, top_k, chunk_steps, 0);

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen trace for corruption");
	TEST_ASSERT(membrane_attntrace3_read_header(f, &h) == MEMBRANE_OK,
		"header reads before corruption");
	idx = malloc(h.chunk_count * sizeof(*idx));
	TEST_ASSERT(idx != NULL, "alloc index");
	TEST_ASSERT(membrane_attntrace3_read_index(f, &h, idx) == MEMBRANE_OK,
		"index reads before corruption");
	TEST_ASSERT(h.chunk_count >= 2, "fixture has at least 2 chunks");
	TEST_ASSERT(fseek(f, (long)idx[1].payload_offset, SEEK_SET) == 0,
		"seek into chunk 1's payload");
	TEST_ASSERT(fputc(0x77, f) != EOF, "corrupt one byte of chunk 1");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	buf0 = malloc(membrane_attntrace3_chunk_entry_count(&h, &idx[0])
		* sizeof(*buf0));
	buf1 = malloc(membrane_attntrace3_chunk_entry_count(&h, &idx[1])
		* sizeof(*buf1));
	TEST_ASSERT(buf0 != NULL && buf1 != NULL, "alloc chunk buffers");
	TEST_ASSERT(membrane_attntrace3_read_chunk(f, &h, &idx[0], buf0)
		== MEMBRANE_OK, "uncorrupted chunk 0 still reads fine");
	TEST_ASSERT(membrane_attntrace3_read_chunk(f, &h, &idx[1], buf1)
		== MEMBRANE_ERR_CORRUPT_DATA,
		"corrupted chunk 1's own CRC32 rejects it");
	free(buf0);
	free(buf1);
	free(idx);
	fclose(f);
	unlink(g_path);
	printf("PASS test_independent_chunk_corruption\n");
}

static void	test_file_sha256_rejects_index_tamper(void)
{
	membrane_attntrace3_header_t	h;
	FILE							*f;

	write_full_trace(48, 2, 2, 4, 16, 1);

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen trace for corruption");
	TEST_ASSERT(membrane_attntrace3_read_header(f, &h) == MEMBRANE_OK,
		"header reads before corruption");
	/* Flip a byte inside the chunk index itself (not any chunk payload)
	 * -- per-chunk CRC32 would not catch this, only the file-level
	 * SHA-256 (which covers the index too) should. */
	TEST_ASSERT(fseek(f, (long)h.index_offset + 1, SEEK_SET) == 0,
		"seek into chunk index");
	TEST_ASSERT(fputc(0x99, f) != EOF, "corrupt one index byte");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	TEST_ASSERT(membrane_attntrace3_read_header(f, &h) == MEMBRANE_OK,
		"header itself still reads (only the index was corrupted)");
	TEST_ASSERT(membrane_attntrace3_verify_file_sha256(f, &h)
		== MEMBRANE_ERR_CORRUPT_DATA,
		"file-level SHA-256 catches index corruption per-chunk CRC32 "
		"cannot see");
	fclose(f);
	unlink(g_path);
	printf("PASS test_file_sha256_rejects_index_tamper\n");
}

static void	test_oversized_block_id_rejected(void)
{
	membrane_attntrace3_writer_t	w;
	membrane_attntrace_entry_t		entries[1];
	FILE							*f;

	f = fopen(g_path, "w+b");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace3_writer_open(f, &w, "m",
		MEMBRANE_ATTNTRACE_SOURCE_SYNTHETIC, 1, 1, 32, 0, 1, 1, 1)
		== MEMBRANE_OK, "writer_open");
	entries[0].block_id = MEMBRANE_ATTNTRACE3_MAX_BLOCK_ID + 1;
	entries[0].score = 0.5f;
	TEST_ASSERT(membrane_attntrace3_writer_put_chunk(f, &w, 0, entries, 0)
		== MEMBRANE_ERR_INVALID_ARG, "oversized block_id rejected");
	fclose(f);
	unlink(g_path);
	printf("PASS test_oversized_block_id_rejected\n");
}

static void	test_step_to_chunk_and_uneven_last_chunk(void)
{
	membrane_attntrace3_header_t	h;
	FILE							*f;

	/* 20 steps, chunk_steps=8 -> chunks of 8, 8, 4 (uneven last chunk). */
	write_full_trace(20, 1, 1, 2, 8, 0);
	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	TEST_ASSERT(membrane_attntrace3_read_header(f, &h) == MEMBRANE_OK,
		"header reads");
	TEST_ASSERT(h.chunk_count == 3, "uneven last chunk still counted");
	TEST_ASSERT(membrane_attntrace3_step_to_chunk(&h, 0) == 0,
		"step 0 -> chunk 0");
	TEST_ASSERT(membrane_attntrace3_step_to_chunk(&h, 7) == 0,
		"step 7 -> chunk 0");
	TEST_ASSERT(membrane_attntrace3_step_to_chunk(&h, 8) == 1,
		"step 8 -> chunk 1");
	TEST_ASSERT(membrane_attntrace3_step_to_chunk(&h, 19) == 2,
		"step 19 -> chunk 2 (the short last chunk)");
	fclose(f);
	unlink(g_path);
	printf("PASS test_step_to_chunk_and_uneven_last_chunk\n");
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file");
	close(fd);
	test_roundtrip_uncompressed();
	test_roundtrip_compressed();
	test_out_of_order_chunk_write_rejected();
	test_independent_chunk_corruption();
	test_file_sha256_rejects_index_tamper();
	test_oversized_block_id_rejected();
	test_step_to_chunk_and_uneven_last_chunk();
	unlink(g_path);
	return (0);
}
