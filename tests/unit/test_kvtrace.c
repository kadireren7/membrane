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

/* Overwrites a slab of stack with `v` and returns, so the writer called
 * next builds its own (uninitialized) header buffer on top of a known
 * pattern. This only makes the defect observable -- the assertions
 * below are the real contract, and both hold unconditionally once the
 * reserved tail is zeroed, so neither can go flaky. */
__attribute__((noinline)) static void	dirty_stack(unsigned char v)
{
	volatile unsigned char	slab[16384];

	memset((void *)slab, v, sizeof(slab));
	(void)slab[0];
}

static void	write_sample_trace(unsigned char stack_fill, uint8_t *out)
{
	membrane_kvtrace_header_t	h;
	uint32_t					steps[4];
	FILE						*f;
	size_t						i;

	sample_header(&h, 4);
	i = 0;
	while (i < 4)
	{
		steps[i] = (uint32_t)(1000 + i);
		i++;
	}
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open trace for write");
	/* Called here, after fopen(), so the pattern lands at the same call
	 * depth the writer's own frame will occupy. */
	dirty_stack(stack_fill);
	TEST_ASSERT(membrane_kvtrace_write(f, &h, steps) == MEMBRANE_OK,
		"trace write succeeds");
	fclose(f);
	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen trace to inspect raw header");
	TEST_ASSERT(fread(out, 1, MEMBRANE_KVTRACE_HEADER_SIZE, f)
		== MEMBRANE_KVTRACE_HEADER_SIZE, "read raw header back");
	fclose(f);
}

/* Serialized fields cover [0, 108) of the 128-byte header. The reserved
 * tail must be zero, and identical input must produce an identical
 * header regardless of what happened to be on the stack beforehand. */
static void	test_reserved_header_bytes_are_canonical(void)
{
	uint8_t	first[MEMBRANE_KVTRACE_HEADER_SIZE];
	uint8_t	second[MEMBRANE_KVTRACE_HEADER_SIZE];
	size_t	i;

	write_sample_trace(0xAA, first);
	write_sample_trace(0x55, second);
	i = 108;
	while (i < MEMBRANE_KVTRACE_HEADER_SIZE)
	{
		TEST_ASSERT(first[i] == 0 && second[i] == 0,
			"reserved header bytes [108, 128) are zero");
		i++;
	}
	TEST_ASSERT(memcmp(first, second, MEMBRANE_KVTRACE_HEADER_SIZE) == 0,
		"identical input produces a byte-identical header");
	unlink(g_path);
	printf("PASS test_reserved_header_bytes_are_canonical\n");
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
	test_reserved_header_bytes_are_canonical();
	unlink(g_path);
	return (0);
}
