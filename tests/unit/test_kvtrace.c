#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/kvtrace.h"
#include "test_helpers.h"

static char	g_path[] = "/tmp/membrane-kvtrace-XXXXXX";

static void	sample_header(membrane_kvtrace_header_t *h, uint32_t step_count)
{
	memset(h, 0, sizeof(*h));
	snprintf(h->model, sizeof(h->model), "SmolLM2-135M-test");
	h->source = MEMBRANE_KVTRACE_SOURCE_REAL_CAPTURE;
	h->n_layer = 30;
	h->n_head_kv = 3;
	h->prompt_len = 512;
	h->step_count = step_count;
	h->created_unix_time = 1234567890ULL;
}

static void	test_roundtrip(void)
{
	membrane_kvtrace_header_t	h;
	membrane_kvtrace_header_t	back;
	uint32_t					steps[64];
	uint32_t					readback[64];
	uint32_t					i;
	FILE						*f;

	sample_header(&h, 64);
	i = 0;
	while (i < 64)
	{
		steps[i] = 4000u + i * 3u;
		i++;
	}
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_kvtrace_write(f, &h, steps) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "open trace for read");
	TEST_ASSERT(membrane_kvtrace_read_header(f, &back) == MEMBRANE_OK,
		"header read succeeds");
	TEST_ASSERT(strcmp(back.model, h.model) == 0, "model name round-trips");
	TEST_ASSERT(back.source == h.source, "source round-trips");
	TEST_ASSERT(back.n_layer == h.n_layer, "n_layer round-trips");
	TEST_ASSERT(back.n_head_kv == h.n_head_kv, "n_head_kv round-trips");
	TEST_ASSERT(back.prompt_len == h.prompt_len, "prompt_len round-trips");
	TEST_ASSERT(back.step_count == h.step_count, "step_count round-trips");
	TEST_ASSERT(membrane_kvtrace_read_steps(f, &back, readback) == MEMBRANE_OK,
		"steps read succeeds");
	i = 0;
	while (i < 64)
	{
		TEST_ASSERT(readback[i] == steps[i], "step value round-trips");
		i++;
	}
	fclose(f);
	unlink(g_path);
}

static void	test_corrupt_payload_rejected(void)
{
	membrane_kvtrace_header_t	h;
	membrane_kvtrace_header_t	back;
	uint32_t					steps[8];
	uint32_t					readback[8];
	uint32_t					i;
	FILE						*f;
	long						payload_off;

	sample_header(&h, 8);
	i = 0;
	while (i < 8)
	{
		steps[i] = 1000u + i;
		i++;
	}
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_kvtrace_write(f, &h, steps) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen trace for corruption");
	payload_off = (long)MEMBRANE_KVTRACE_HEADER_SIZE;
	TEST_ASSERT(fseek(f, payload_off, SEEK_SET) == 0, "seek to payload");
	TEST_ASSERT(fputc(0xFF, f) != EOF, "corrupt one payload byte");
	fclose(f);

	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace for read");
	TEST_ASSERT(membrane_kvtrace_read_header(f, &back) == MEMBRANE_OK,
		"header still reads (only payload was corrupted)");
	TEST_ASSERT(membrane_kvtrace_read_steps(f, &back, readback)
		== MEMBRANE_ERR_CORRUPT_DATA, "corrupted payload checksum rejected");
	fclose(f);
	unlink(g_path);
}

static void	test_zero_step_count_rejected(void)
{
	membrane_kvtrace_header_t	h;
	uint32_t					dummy;
	FILE						*f;

	sample_header(&h, 0);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	TEST_ASSERT(membrane_kvtrace_write(f, &h, &dummy) == MEMBRANE_ERR_INVALID_ARG,
		"zero step_count rejected");
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
	test_zero_step_count_rejected();
	unlink(g_path);
	return (0);
}
