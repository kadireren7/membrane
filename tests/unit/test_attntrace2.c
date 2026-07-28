#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/attntrace2.h"
#include "test_helpers.h"

static char	g_path[] = "/tmp/membrane-attntrace2-XXXXXX";

static void	sample_header(membrane_attntrace2_header_t *h,
				uint32_t step_count, uint32_t n_layer, uint32_t n_head,
				uint32_t top_k)
{
	memset(h, 0, sizeof(*h));
	snprintf(h->model, sizeof(h->model), "SmolLM2-135M-test");
	h->source = MEMBRANE_ATTNTRACE_SOURCE_REAL_CAPTURE;
	h->n_layer = n_layer;
	h->n_head = n_head;
	h->block_size_tokens = 32;
	h->prompt_len = 512;
	h->step_count = step_count;
	h->top_k = top_k;
	h->created_unix_time = 1234567890ULL;
}

static void	fill_entries(membrane_attntrace_entry_t *e, size_t n)
{
	size_t	i;

	i = 0;
	while (i < n)
	{
		if (i % 7 == 0)
		{
			e[i].block_id = UINT32_MAX;
			e[i].score = 0.0f;
		}
		else
		{
			e[i].block_id = (uint32_t)(i % 4000);
			e[i].score = (float)(i % 101) / 100.0f;
		}
		i++;
	}
}

static void	roundtrip_case(int compress)
{
	membrane_attntrace2_header_t	h;
	membrane_attntrace2_header_t	back;
	membrane_attntrace_entry_t		*entries;
	membrane_attntrace_entry_t		*readback;
	size_t							n;
	size_t							i;
	FILE							*f;

	sample_header(&h, 32, 4, 3, 8);
	n = membrane_attntrace2_entry_count(&h);
	entries = malloc(n * sizeof(*entries));
	readback = malloc(n * sizeof(*readback));
	TEST_ASSERT(entries != NULL && readback != NULL, "alloc entries");
	fill_entries(entries, n);

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace2_write(f, &h, entries, compress)
		== MEMBRANE_OK, "trace write succeeds");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "open trace for read");
	TEST_ASSERT(membrane_attntrace2_read_header(f, &back) == MEMBRANE_OK,
		"header read succeeds");
	TEST_ASSERT(strcmp(back.model, h.model) == 0, "model name round-trips");
	TEST_ASSERT(back.n_layer == h.n_layer, "n_layer round-trips");
	TEST_ASSERT((int)back.compressed == compress, "compressed flag round-trips");
	TEST_ASSERT(membrane_attntrace2_read_entries(f, &back, readback)
		== MEMBRANE_OK, "entries read succeeds");
	i = 0;
	while (i < n)
	{
		TEST_ASSERT(readback[i].block_id == entries[i].block_id,
			"block_id round-trips exactly");
		/* score is lossily quantized to 8 bits -- must be close, not
		 * exact (except the UINT32_MAX sentinel, which is exact 0). */
		float	diff = readback[i].score - entries[i].score;
		if (diff < 0)
			diff = -diff;
		TEST_ASSERT(diff <= (1.0f / 255.0f) + 1e-6f,
			"score round-trips within one quantization step");
		i++;
	}
	fclose(f);
	free(entries);
	free(readback);
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

static void	test_compression_actually_shrinks_real_shaped_data(void)
{
	membrane_attntrace2_header_t	h;
	membrane_attntrace_entry_t		*entries;
	size_t							n;
	size_t							i;
	FILE							*f;
	long							uncompressed_size;
	long							compressed_size;

	/* Real captures cluster heavily on a small set of block ids (sink
	 * + recency) -- simulate that shape, not uniform noise, since
	 * DEFLATE's real benefit depends on it. */
	sample_header(&h, 512, 8, 4, 8);
	n = membrane_attntrace2_entry_count(&h);
	entries = malloc(n * sizeof(*entries));
	TEST_ASSERT(entries != NULL, "alloc entries");
	i = 0;
	while (i < n)
	{
		entries[i].block_id = (i % 8 == 0) ? 0 : (uint32_t)((i / 8) % 5);
		entries[i].score = 0.5f;
		i++;
	}

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open uncompressed for write");
	TEST_ASSERT(membrane_attntrace2_write(f, &h, entries, 0) == MEMBRANE_OK,
		"uncompressed write succeeds");
	fseek(f, 0, SEEK_END);
	uncompressed_size = ftell(f);
	fclose(f);

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open compressed for write");
	TEST_ASSERT(membrane_attntrace2_write(f, &h, entries, 1) == MEMBRANE_OK,
		"compressed write succeeds");
	fseek(f, 0, SEEK_END);
	compressed_size = ftell(f);
	fclose(f);

	TEST_ASSERT(compressed_size < uncompressed_size,
		"compression must actually shrink real-shaped repetitive data");
	free(entries);
	unlink(g_path);
	printf("PASS test_compression_actually_shrinks_real_shaped_data\n");
}

static void	test_corrupt_payload_rejected(void)
{
	membrane_attntrace2_header_t	h;
	membrane_attntrace2_header_t	back;
	membrane_attntrace_entry_t		entries[8];
	membrane_attntrace_entry_t		readback[8];
	FILE							*f;

	sample_header(&h, 1, 1, 1, 8);
	fill_entries(entries, 8);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace2_write(f, &h, entries, 0) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen trace for corruption");
	/* Entry index 1 (i % 7 != 0, so it's a real, non-sentinel block_id
	 * -- offset 0 would be entry 0's block_id LSB, which for this
	 * fixture is the UINT32_MAX sentinel already encoded as 0xFF, so
	 * writing 0xFF there would not actually change anything). */
	TEST_ASSERT(fseek(f, (long)MEMBRANE_ATTNTRACE2_HEADER_SIZE + 3,
		SEEK_SET) == 0, "seek to payload");
	TEST_ASSERT(fputc(0x77, f) != EOF, "corrupt one payload byte");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	TEST_ASSERT(membrane_attntrace2_read_header(f, &back) == MEMBRANE_OK,
		"header still reads (only payload was corrupted)");
	TEST_ASSERT(membrane_attntrace2_read_entries(f, &back, readback)
		== MEMBRANE_ERR_CORRUPT_DATA, "corrupted payload checksum rejected");
	fclose(f);
	unlink(g_path);
	printf("PASS test_corrupt_payload_rejected\n");
}

static void	test_oversized_block_id_rejected(void)
{
	membrane_attntrace2_header_t	h;
	membrane_attntrace_entry_t		entries[1];
	FILE							*f;

	sample_header(&h, 1, 1, 1, 1);
	entries[0].block_id = MEMBRANE_ATTNTRACE2_MAX_BLOCK_ID + 1;
	entries[0].score = 0.5f;
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace2_write(f, &h, entries, 0)
		== MEMBRANE_ERR_INVALID_ARG, "oversized block_id rejected");
	fclose(f);
	unlink(g_path);
	printf("PASS test_oversized_block_id_rejected\n");
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file");
	close(fd);
	test_roundtrip_uncompressed();
	test_roundtrip_compressed();
	test_compression_actually_shrinks_real_shaped_data();
	test_corrupt_payload_rejected();
	test_oversized_block_id_rejected();
	unlink(g_path);
	return (0);
}
