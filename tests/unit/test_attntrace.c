#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/attntrace.h"
#include "test_helpers.h"

static char	g_path[] = "/tmp/membrane-attntrace-XXXXXX";

static void	sample_header(membrane_attntrace_header_t *h,
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
		e[i].block_id = (uint32_t)(i % 17);
		e[i].score = (float)(i % 100) / 100.0f;
		i++;
	}
}

static void	test_roundtrip(void)
{
	membrane_attntrace_header_t	h;
	membrane_attntrace_header_t	back;
	membrane_attntrace_entry_t		*entries;
	membrane_attntrace_entry_t		*readback;
	size_t							n;
	size_t							i;
	FILE							*f;

	sample_header(&h, 4, 3, 2, 8);
	n = membrane_attntrace_entry_count(&h);
	TEST_ASSERT(n == 4 * 3 * 2 * 8, "entry count computed correctly");
	entries = malloc(n * sizeof(*entries));
	readback = malloc(n * sizeof(*readback));
	TEST_ASSERT(entries != NULL && readback != NULL, "alloc entries");
	fill_entries(entries, n);

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace_write(f, &h, entries) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "open trace for read");
	TEST_ASSERT(membrane_attntrace_read_header(f, &back) == MEMBRANE_OK,
		"header read succeeds");
	TEST_ASSERT(strcmp(back.model, h.model) == 0, "model name round-trips");
	TEST_ASSERT(back.n_layer == h.n_layer, "n_layer round-trips");
	TEST_ASSERT(back.n_head == h.n_head, "n_head round-trips");
	TEST_ASSERT(back.top_k == h.top_k, "top_k round-trips");
	TEST_ASSERT(back.block_size_tokens == h.block_size_tokens,
		"block_size_tokens round-trips");
	TEST_ASSERT(membrane_attntrace_read_entries(f, &back, readback)
		== MEMBRANE_OK, "entries read succeeds");
	i = 0;
	while (i < n)
	{
		TEST_ASSERT(readback[i].block_id == entries[i].block_id,
			"block_id round-trips");
		TEST_ASSERT(readback[i].score == entries[i].score,
			"score round-trips");
		i++;
	}
	fclose(f);
	free(entries);
	free(readback);
	unlink(g_path);
}

static void	test_corrupt_payload_rejected(void)
{
	membrane_attntrace_header_t	h;
	membrane_attntrace_header_t	back;
	membrane_attntrace_entry_t		entries[8];
	membrane_attntrace_entry_t		readback[8];
	FILE							*f;

	sample_header(&h, 1, 1, 1, 8);
	fill_entries(entries, 8);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace_write(f, &h, entries) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen trace for corruption");
	TEST_ASSERT(fseek(f, (long)MEMBRANE_ATTNTRACE_HEADER_SIZE, SEEK_SET) == 0,
		"seek to payload");
	TEST_ASSERT(fputc(0xFF, f) != EOF, "corrupt one payload byte");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	TEST_ASSERT(membrane_attntrace_read_header(f, &back) == MEMBRANE_OK,
		"header still reads (only payload was corrupted)");
	TEST_ASSERT(membrane_attntrace_read_entries(f, &back, readback)
		== MEMBRANE_ERR_CORRUPT_DATA, "corrupted payload checksum rejected");
	fclose(f);
	unlink(g_path);
}

static void	test_bounds_rejected(void)
{
	membrane_attntrace_header_t	h;
	membrane_attntrace_entry_t		dummy;
	FILE							*f;

	sample_header(&h, 0, 1, 1, 8);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace_write(f, &h, &dummy)
		== MEMBRANE_ERR_INVALID_ARG, "zero step_count rejected");
	fclose(f);

	sample_header(&h, 1, 1, 1, MEMBRANE_ATTNTRACE_MAX_TOPK + 1);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_attntrace_write(f, &h, &dummy)
		== MEMBRANE_ERR_INVALID_ARG, "over-cap top_k rejected");
	fclose(f);
	unlink(g_path);
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file");
	close(fd);
	test_roundtrip();
	test_corrupt_payload_rejected();
	test_bounds_rejected();
	unlink(g_path);
	return (0);
}
