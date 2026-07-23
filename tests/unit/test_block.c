#include <string.h>

#include "membrane/block.h"
#include "test_helpers.h"

static void	test_checksum_known_vector(void)
{
	const uint8_t	vec[] = "123456789";

	TEST_ASSERT(membrane_block_checksum(vec, 9) == 0xCBF43926u,
		"CRC32 matches the standard check value");
	TEST_ASSERT(membrane_block_checksum(NULL, 0) == 0,
		"CRC32 of empty input is zero");
}

static void	test_lifecycle(void)
{
	membrane_block_t	*block;

	block = membrane_block_create(7, MEMBRANE_CODEC_RAW);
	TEST_ASSERT(block != NULL, "block allocates");
	TEST_ASSERT(block->id == 7, "block keeps its id");
	TEST_ASSERT(block->codec == MEMBRANE_CODEC_RAW, "block keeps its codec");
	TEST_ASSERT(block->data == NULL && block->stored_size == 0,
		"fresh block owns no storage");
	TEST_ASSERT(block->state == MEMBRANE_BLOCK_HOT, "fresh block starts hot");
	membrane_block_destroy(block);
	membrane_block_destroy(NULL);
}

static void	test_write_read_roundtrip(void)
{
	enum { N = 4096 };
	membrane_block_t	*block;
	uint8_t				in[N];
	uint8_t				out[N];
	size_t				out_len;

	block = membrane_block_create(1, MEMBRANE_CODEC_RLE);
	TEST_ASSERT(block != NULL, "block allocates");
	test_fill_random(in, N, 99);
	TEST_ASSERT(membrane_block_write(block, in, N) == MEMBRANE_OK,
		"block write");
	TEST_ASSERT(block->original_size == N, "original size recorded");
	TEST_ASSERT(block->stored_size > 0, "stored size recorded");
	TEST_ASSERT(block->access_count == 1, "write counts as an access");
	TEST_ASSERT(membrane_block_read(block, out, N, &out_len) == MEMBRANE_OK,
		"block read");
	TEST_ASSERT(out_len == N && memcmp(in, out, N) == 0,
		"block round-trips bit-identically");
	TEST_ASSERT(block->access_count == 2, "read counts as an access");
	membrane_block_destroy(block);
}

static void	test_empty_write(void)
{
	membrane_block_t	*block;
	uint8_t				out[4];
	size_t				out_len;

	block = membrane_block_create(2, MEMBRANE_CODEC_RLE);
	out_len = 99;
	TEST_ASSERT(block != NULL, "block allocates");
	TEST_ASSERT(membrane_block_write(block, NULL, 0) == MEMBRANE_OK,
		"writing empty input succeeds");
	TEST_ASSERT(membrane_block_read(block, out, sizeof(out), &out_len)
		== MEMBRANE_OK, "reading empty block succeeds");
	TEST_ASSERT(out_len == 0, "empty block reads back zero bytes");
	membrane_block_destroy(block);
}

static void	test_corrupt_detection(void)
{
	enum { N = 4096 };
	membrane_block_t	*block;
	uint8_t				in[N];
	uint8_t				out[N];
	size_t				out_len;

	block = membrane_block_create(3, MEMBRANE_CODEC_RLE);
	TEST_ASSERT(block != NULL, "block allocates");
	memset(in, 0x77, N);
	TEST_ASSERT(membrane_block_write(block, in, N) == MEMBRANE_OK,
		"block write");
	((uint8_t *)block->data)[block->stored_size / 2] ^= 0xFF;
	TEST_ASSERT(membrane_block_read(block, out, N, &out_len) != MEMBRANE_OK,
		"corrupted block is detected on read");
	membrane_block_destroy(block);
}

static void	test_alloc_failure(void)
{
	membrane_block_t	*block;

	block = membrane_block_create(4, MEMBRANE_CODEC_RAW);
	TEST_ASSERT(block != NULL, "block allocates");
	TEST_ASSERT(membrane_block_write(block, (const uint8_t *)"", (size_t)-1)
		== MEMBRANE_ERR_ALLOC_FAILED, "absurd allocation fails cleanly");
	membrane_block_destroy(block);
}

static void	test_invalid_args(void)
{
	membrane_block_t	*block;
	uint8_t				buf[16];
	size_t				out_len;

	block = membrane_block_create(5, MEMBRANE_CODEC_RAW);
	TEST_ASSERT(membrane_block_create(0, MEMBRANE_CODEC_COUNT) == NULL,
		"invalid codec id is rejected at create");
	TEST_ASSERT(block != NULL, "block allocates");
	TEST_ASSERT(membrane_block_write(NULL, buf, 16) == MEMBRANE_ERR_INVALID_ARG,
		"NULL block rejected on write");
	TEST_ASSERT(membrane_block_read(block, buf, 16, NULL)
		== MEMBRANE_ERR_INVALID_ARG, "NULL out_len rejected on read");
	TEST_ASSERT(membrane_block_write(block, buf, 16) == MEMBRANE_OK,
		"block write");
	TEST_ASSERT(membrane_block_read(block, buf, 4, &out_len)
		== MEMBRANE_ERR_BUFFER_TOO_SMALL, "undersized read buffer rejected");
	membrane_block_destroy(block);
}

static void	fill_mixed_content(uint8_t *buf, size_t len)
{
	size_t	off;

	memset(buf, 0, len);
	off = 0;
	while (off < len)
	{
		test_fill_random(buf + off, 4096, (uint32_t)(off + 1));
		off += 8192;
	}
}

static void	test_large_buffer(void)
{
	enum { N = 64 * 1024 * 1024 };
	membrane_block_t	*block;
	uint8_t				*in;
	uint8_t				*out;
	size_t				out_len;

	block = membrane_block_create(6, MEMBRANE_CODEC_RLE);
	in = malloc(N);
	out = malloc(N);
	TEST_ASSERT(block && in && out, "large buffers allocate");
	fill_mixed_content(in, N);
	TEST_ASSERT(membrane_block_write(block, in, N) == MEMBRANE_OK,
		"large write");
	TEST_ASSERT(membrane_block_read(block, out, N, &out_len) == MEMBRANE_OK,
		"large read");
	TEST_ASSERT(out_len == (size_t)N && memcmp(in, out, N) == 0,
		"large buffer round-trips bit-identically");
	free(in);
	free(out);
	membrane_block_destroy(block);
}

int	main(void)
{
	test_checksum_known_vector();
	test_lifecycle();
	test_write_read_roundtrip();
	test_empty_write();
	test_corrupt_detection();
	test_alloc_failure();
	test_invalid_args();
	test_large_buffer();
	return (0);
}
