#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/block.h"
#include "membrane/kvdump.h"
#include "membrane/kvmetrics.h"
#include "test_helpers.h"

# define PAYLOAD 8192

static char	g_path[] = "/tmp/membrane-kvdump-XXXXXX";

static void	sample_header(membrane_kv_header_t *h, size_t payload_size)
{
	memset(h, 0, sizeof(*h));
	snprintf(h->model, sizeof(h->model), "test-model");
	h->layer = 3;
	h->tensor_type = MEMBRANE_KV_TENSOR_V;
	h->token_start = 0;
	h->token_end = 512;
	h->dtype = 1;
	h->n_dims = 2;
	h->dims[0] = 288;
	h->dims[1] = 512;
	h->payload_size = payload_size;
}

static void	write_sample(const uint8_t *payload, size_t len)
{
	membrane_kv_header_t	h;
	FILE					*f;

	sample_header(&h, len);
	h.checksum = membrane_block_checksum(payload, len);
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open dump for write");
	TEST_ASSERT(membrane_kvdump_write(f, &h, payload) == MEMBRANE_OK,
		"dump write succeeds");
	fclose(f);
}

static void	test_roundtrip(void)
{
	membrane_kv_header_t	h;
	uint8_t					payload[PAYLOAD];
	uint8_t					*back;
	FILE					*f;

	test_fill_random(payload, PAYLOAD, 77);
	write_sample(payload, PAYLOAD);
	f = fopen(g_path, "rb");
	TEST_ASSERT(membrane_kvdump_read_header(f, &h) == MEMBRANE_OK,
		"header reads back");
	TEST_ASSERT(strcmp(h.model, "test-model") == 0 && h.layer == 3
		&& h.tensor_type == MEMBRANE_KV_TENSOR_V && h.token_end == 512
		&& h.n_dims == 2 && h.dims[0] == 288 && h.dims[1] == 512
		&& h.payload_size == PAYLOAD, "header fields round-trip");
	TEST_ASSERT(membrane_kvdump_read_payload(f, &h, &back) == MEMBRANE_OK
		&& memcmp(back, payload, PAYLOAD) == 0, "payload round-trips");
	free(back);
	TEST_ASSERT(membrane_kvdump_read_header(f, &h) == MEMBRANE_ERR_NOT_FOUND,
		"clean EOF reports NOT_FOUND");
	fclose(f);
}

static void	flip_byte(long off)
{
	FILE	*f;
	int		c;

	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen dump");
	fseek(f, off, SEEK_SET);
	c = fgetc(f);
	fseek(f, off, SEEK_SET);
	fputc(c ^ 0xFF, f);
	fclose(f);
}

static void	test_corrupted_header(void)
{
	membrane_kv_header_t	h;
	uint8_t					payload[64];
	FILE					*f;

	test_fill_random(payload, sizeof(payload), 5);
	write_sample(payload, sizeof(payload));
	flip_byte(20);	/* inside the model-name field: header CRC must trip */
	f = fopen(g_path, "rb");
	TEST_ASSERT(membrane_kvdump_read_header(f, &h)
		== MEMBRANE_ERR_CORRUPT_DATA, "corrupted header is rejected");
	fclose(f);
}

static void	test_checksum_failure(void)
{
	membrane_kv_header_t	h;
	uint8_t					payload[64];
	uint8_t					*back;
	FILE					*f;

	test_fill_random(payload, sizeof(payload), 6);
	write_sample(payload, sizeof(payload));
	flip_byte(MEMBRANE_KV_HEADER_SIZE + 10);
	f = fopen(g_path, "rb");
	TEST_ASSERT(membrane_kvdump_read_header(f, &h) == MEMBRANE_OK,
		"header still valid");
	TEST_ASSERT(membrane_kvdump_read_payload(f, &h, &back)
		== MEMBRANE_ERR_CORRUPT_DATA, "payload checksum failure detected");
	fclose(f);
}

static void	test_truncated_payload(void)
{
	membrane_kv_header_t	h;
	uint8_t					payload[256];
	uint8_t					*back;
	FILE					*f;

	test_fill_random(payload, sizeof(payload), 7);
	write_sample(payload, sizeof(payload));
	TEST_ASSERT(truncate(g_path, MEMBRANE_KV_HEADER_SIZE + 100) == 0,
		"truncate dump");
	f = fopen(g_path, "rb");
	TEST_ASSERT(membrane_kvdump_read_header(f, &h) == MEMBRANE_OK,
		"header still valid after truncation");
	TEST_ASSERT(membrane_kvdump_read_payload(f, &h, &back)
		== MEMBRANE_ERR_CORRUPT_DATA, "truncated payload detected");
	fclose(f);
}

static void	test_empty_and_shapes(void)
{
	membrane_kv_header_t	h;
	membrane_kv_header_t	r;
	uint8_t					*back;
	FILE					*f;

	sample_header(&h, 0);
	h.n_dims = 4;
	h.dims[2] = 7;
	h.dims[3] = 9;
	h.checksum = membrane_block_checksum(NULL, 0);
	f = fopen(g_path, "wb");
	TEST_ASSERT(membrane_kvdump_write(f, &h, NULL) == MEMBRANE_OK,
		"empty tensor writes");
	fclose(f);
	f = fopen(g_path, "rb");
	TEST_ASSERT(membrane_kvdump_read_header(f, &r) == MEMBRANE_OK
		&& r.n_dims == 4 && r.dims[3] == 9 && r.payload_size == 0,
		"empty tensor and 4-dim shape round-trip");
	TEST_ASSERT(membrane_kvdump_read_payload(f, &r, &back) == MEMBRANE_OK
		&& back == NULL, "empty payload reads as NULL");
	fclose(f);
}

static void	test_overflow_guards(void)
{
	membrane_kv_header_t	h;
	uint8_t					payload[16];
	FILE					*f;

	sample_header(&h, sizeof(payload));
	h.n_dims = 5;
	f = fopen(g_path, "wb");
	TEST_ASSERT(membrane_kvdump_write(f, &h, payload)
		== MEMBRANE_ERR_INVALID_ARG, "n_dims > max rejected on write");
	sample_header(&h, sizeof(payload));
	h.payload_size = MEMBRANE_KV_MAX_PAYLOAD + 1;
	TEST_ASSERT(membrane_kvdump_write(f, &h, payload)
		== MEMBRANE_ERR_INVALID_ARG, "absurd payload_size rejected on write");
	fclose(f);
}

static void	test_metrics_deterministic(void)
{
	membrane_kv_metrics_t	a;
	membrane_kv_metrics_t	b;
	uint8_t					buf[PAYLOAD];

	test_fill_random(buf, PAYLOAD, 99);
	memset(buf, 0, PAYLOAD / 2);
	TEST_ASSERT(membrane_kv_metrics_compute(buf, PAYLOAD, 4096, &a)
		== MEMBRANE_OK, "metrics compute");
	TEST_ASSERT(membrane_kv_metrics_compute(buf, PAYLOAD, 4096, &b)
		== MEMBRANE_OK, "metrics compute again");
	TEST_ASSERT(memcmp(&a, &b, sizeof(a)) == 0,
		"metrics are bit-identical across runs");
	TEST_ASSERT(a.integrity_ok == 1 && a.blocks == 2
		&& a.zero_bytes >= PAYLOAD / 2, "metrics values are sane");
}

static void	test_metrics_known_values(void)
{
	membrane_kv_metrics_t	m;
	uint8_t					zeros[4096];

	memset(zeros, 0, sizeof(zeros));
	TEST_ASSERT(membrane_kv_metrics_compute(zeros, sizeof(zeros), 4096, &m)
		== MEMBRANE_OK, "metrics on zeros");
	TEST_ASSERT(m.entropy == 0.0 && m.zero_bytes == 4096
		&& m.total_runs == 1 && m.max_run == 4096
		&& m.adaptive_rle_blocks == 1 && m.adaptive_bytes == 32,
		"all-zero block: H=0, one run, RLE 128x");
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file");
	close(fd);
	test_roundtrip();
	test_corrupted_header();
	test_checksum_failure();
	test_truncated_payload();
	test_empty_and_shapes();
	test_overflow_guards();
	test_metrics_deterministic();
	test_metrics_known_values();
	unlink(g_path);
	return (0);
}
