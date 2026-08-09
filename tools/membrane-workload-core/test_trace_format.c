#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/block.h"
#include "test_helpers.h"
#include "trace_format.h"

static char	g_path[] = "/tmp/membrane-trace-format-XXXXXX";

/* Standard bit-wise CRC32 (poly 0xEDB88320, reflected) -- deliberately
 * NOT reusing src/block/crc32_table.h's table or trace_format.c's own
 * incremental helper: this independently verifies the same algorithm
 * (round-trip via membrane_trace_write/_read_block agreeing proves
 * they match) and lets the malformed-header tests below construct a
 * header with a genuinely VALID header_checksum for otherwise-invalid
 * field values -- necessary because membrane_trace_open checks the
 * header checksum before any field-level validation, so a naive byte
 * patch without recomputing it would only ever exercise the checksum
 * check, never the field checks these tests target. */
static uint32_t	bitwise_crc32(const uint8_t *buf, size_t len)
{
	uint32_t	crc;
	size_t		i;
	int			j;

	crc = 0xFFFFFFFFu;
	i = 0;
	while (i < len)
	{
		crc ^= buf[i];
		j = 0;
		while (j < 8)
		{
			if (crc & 1)
				crc = (crc >> 1) ^ 0xEDB88320u;
			else
				crc >>= 1;
			j++;
		}
		i++;
	}
	return (crc ^ 0xFFFFFFFFu);
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

static uint32_t	get_le32(const uint8_t *p)
{
	return ((uint32_t)p[0] | ((uint32_t)p[1] << 8)
		| ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* Builds a syntactically well-formed, checksum-VALID 64-byte header
 * (no metadata, no payload) directly from field values, bypassing
 * membrane_trace_write's own arg validation entirely -- this is how
 * the "reader rejects a malformed but not-corrupted-looking file"
 * tests below construct inputs the public write API would refuse to
 * produce. */
static void	craft_header(uint8_t *h, uint32_t version, uint32_t dtype,
				uint32_t elements_per_block, uint32_t metadata_length,
				uint64_t block_count, uint64_t payload_bytes)
{
	memset(h, 0, MEMBRANE_TRACE_HEADER_SIZE);
	memcpy(h, MEMBRANE_TRACE_MAGIC, 7);
	put_le32(h + 8, version);
	put_le32(h + 12, dtype);
	put_le32(h + 16, elements_per_block);
	put_le32(h + 20, metadata_length);
	put_le64(h + 24, block_count);
	put_le64(h + 32, payload_bytes);
	put_le64(h + 40, 0);
	put_le32(h + 48, 0);
	put_le32(h + 52, bitwise_crc32(h, 52));
}

static void	write_file(const uint8_t *bytes, size_t n)
{
	FILE	*f;

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open temp file for write");
	TEST_ASSERT(fwrite(bytes, 1, n, f) == n, "write temp file");
	fclose(f);
}

static membrane_status_t	try_open_and_drain(uint16_t **out_first_block)
{
	membrane_trace_reader_t	*r;
	membrane_trace_info_t	info;
	uint16_t				*buf;
	membrane_status_t		st;

	st = membrane_trace_open(g_path, &r);
	if (st != MEMBRANE_OK)
		return (st);
	membrane_trace_info(r, &info);
	buf = malloc((size_t)info.elements_per_block * sizeof(uint16_t));
	while (1)
	{
		st = membrane_trace_read_block(r, buf);
		if (st != MEMBRANE_OK)
			break ;
	}
	if (out_first_block != NULL)
		*out_first_block = buf;
	else
		free(buf);
	membrane_trace_close(r);
	if (st == MEMBRANE_ERR_NOT_FOUND)
		return (MEMBRANE_OK);
	return (st);
}

/* Deterministic fixture: `n_blocks` blocks of `epb` elements each. */
static uint16_t	*make_fixture(uint64_t n_blocks, uint32_t epb)
{
	uint16_t	*blocks;
	uint64_t	i;

	blocks = malloc((size_t)(n_blocks * epb) * sizeof(uint16_t));
	i = 0;
	while (i < n_blocks * epb)
	{
		blocks[i] = (uint16_t)(i * 37u + 11u);
		i++;
	}
	return (blocks);
}

static void	test_roundtrip_small(void)
{
	uint16_t				*blocks;
	uint16_t				*readback;
	membrane_trace_reader_t	*r;
	membrane_trace_info_t	info;

	blocks = make_fixture(1, 32);
	{
		FILE	*f;

		f = fopen(g_path, "wb");
		TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 32, 1,
				blocks, "fixture", 42) == MEMBRANE_OK, "write minimal trace");
		fclose(f);
	}
	TEST_ASSERT(membrane_trace_open(g_path, &r) == MEMBRANE_OK, "open");
	membrane_trace_info(r, &info);
	TEST_ASSERT(info.format_version == MEMBRANE_TRACE_VERSION, "version");
	TEST_ASSERT(info.dtype == MEMBRANE_TRACE_DTYPE_F16, "dtype");
	TEST_ASSERT(info.elements_per_block == 32, "elements_per_block");
	TEST_ASSERT(info.block_count == 1, "block_count");
	TEST_ASSERT(info.payload_bytes == 64, "payload_bytes");
	TEST_ASSERT(strcmp(info.metadata, "fixture") == 0, "metadata");
	readback = malloc(32 * sizeof(uint16_t));
	TEST_ASSERT(membrane_trace_read_block(r, readback) == MEMBRANE_OK,
		"read the one block");
	TEST_ASSERT(memcmp(readback, blocks, 32 * sizeof(uint16_t)) == 0,
		"block matches");
	TEST_ASSERT(membrane_trace_read_block(r, readback) == MEMBRANE_ERR_NOT_FOUND,
		"no second block");
	membrane_trace_close(r);
	free(readback);
	free(blocks);
}

static void	test_roundtrip_multi_block(void)
{
	uint16_t				*blocks;
	uint16_t				readback[64];
	membrane_trace_reader_t	*r;
	membrane_trace_info_t	info;
	uint32_t				i;

	blocks = make_fixture(50, 64);
	{
		FILE	*f;

		f = fopen(g_path, "wb");
		TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 64, 50,
				blocks, "", 0) == MEMBRANE_OK, "write multi-block trace");
		fclose(f);
	}
	TEST_ASSERT(membrane_trace_open(g_path, &r) == MEMBRANE_OK, "open");
	membrane_trace_info(r, &info);
	TEST_ASSERT(info.block_count == 50, "block_count");
	TEST_ASSERT(info.metadata[0] == '\0', "empty metadata stays empty");
	i = 0;
	while (i < 50)
	{
		TEST_ASSERT(membrane_trace_read_block(r, readback) == MEMBRANE_OK,
			"read block");
		TEST_ASSERT(memcmp(readback, blocks + (size_t)i * 64,
				64 * sizeof(uint16_t)) == 0, "block content matches");
		i++;
	}
	TEST_ASSERT(membrane_trace_read_block(r, readback) == MEMBRANE_ERR_NOT_FOUND,
		"exactly 50 blocks");
	membrane_trace_close(r);
	free(blocks);
}

static void	test_bad_magic(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 32, 0,
		1, 64);
	h[0] = 'X';
	/* Magic is checked before the header checksum, so this is rejected
	 * even though craft_header still gave it a technically-valid CRC
	 * for the corrupted bytes -- either way it must fail. */
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"bad magic rejected");
}

static void	test_unsupported_version(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION + 1, MEMBRANE_TRACE_DTYPE_F16, 32,
		0, 1, 64);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"unsupported version rejected");
}

static void	test_truncated_header(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 32, 0,
		1, 64);
	write_file(h, 10);
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"truncated header rejected");
}

static void	test_truncated_payload(void)
{
	uint16_t	*blocks;
	FILE		*f;
	long		full_size;

	blocks = make_fixture(4, 32);
	f = fopen(g_path, "wb");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 32, 4,
			blocks, NULL, 0) == MEMBRANE_OK, "write for truncation test");
	fclose(f);
	free(blocks);
	f = fopen(g_path, "rb");
	fseek(f, 0, SEEK_END);
	full_size = ftell(f);
	fclose(f);
	TEST_ASSERT(truncate(g_path, full_size - 10) == 0,
		"truncate trailing payload bytes");
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"truncated payload rejected");
}

static void	test_payload_bytes_field_inconsistent(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	/* block_count=2, elements_per_block=32 implies payload_bytes=128,
	 * but the header declares 999 -- rejected without ever trusting
	 * the mismatched value. */
	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 32, 0,
		2, 999);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"inconsistent payload_bytes field rejected");
}

static void	test_elements_per_block_zero_rejected(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 0, 0,
		1, 0);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"zero elements_per_block rejected");
}

static void	test_unsupported_dtype_rejected(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION, 99, 32, 0, 1, 64);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_UNIMPLEMENTED,
		"unsupported dtype rejected distinctly from generic corruption");
}

static void	test_zero_block_count_rejected(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 32, 0,
		0, 0);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"zero block_count rejected");
}

static void	test_excessive_metadata_length_rejected(void)
{
	uint8_t	h[MEMBRANE_TRACE_HEADER_SIZE];

	/* Declares far more metadata than MEMBRANE_TRACE_MAX_METADATA and
	 * far more than actually follows in the file -- must be rejected
	 * before any allocation sized off this field is attempted (the
	 * file is only MEMBRANE_TRACE_HEADER_SIZE bytes long). */
	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16, 32,
		UINT32_MAX, 1, 64);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"excessive metadata_length rejected without a matching allocation");
}

/* block_count and elements_per_block are each individually bounded
 * (MEMBRANE_TRACE_MAX_BLOCK_COUNT, MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK)
 * before their product is ever computed, so that product cannot
 * actually reach anywhere near UINT64_MAX in practice -- this checks
 * the reachable case instead: block_count at its individual maximum
 * combined with a large elements_per_block produces a payload well
 * over MEMBRANE_TRACE_MAX_PAYLOAD_BYTES, and that must be rejected
 * (by the bound check, not by 64-bit wraparound) without attempting
 * the multi-terabyte allocation such a file would otherwise imply. */
static void	test_excessive_declared_size_rejected_without_allocating(void)
{
	uint8_t		h[MEMBRANE_TRACE_HEADER_SIZE];
	uint64_t	huge_payload;

	huge_payload = (uint64_t)MEMBRANE_TRACE_MAX_BLOCK_COUNT
		* MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK * 2;
	craft_header(h, MEMBRANE_TRACE_VERSION, MEMBRANE_TRACE_DTYPE_F16,
		MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK, 0,
		MEMBRANE_TRACE_MAX_BLOCK_COUNT, huge_payload);
	write_file(h, sizeof(h));
	TEST_ASSERT(try_open_and_drain(NULL) == MEMBRANE_ERR_CORRUPT_DATA,
		"declared size far over the payload cap rejected, not allocated");
}

/*
 * Proves trace_format.c's incremental crc32_init/_update/_final is
 * bit-identical to the independent production checksum
 * (membrane_block_checksum, src/block/block.c) on the exact bytes
 * membrane_trace_write puts on disk -- not just that the writer and
 * reader in this same file agree with each other. Serializes the
 * fixture to little-endian bytes itself (matching membrane_trace_write's
 * own put_le16 loop) rather than reusing any trace_format.c helper, so
 * this check does not depend on the code it is verifying.
 */
static void	test_trace_format_matches_membrane_block_checksum(void)
{
	uint16_t	*blocks;
	uint8_t		*le_payload;
	uint8_t		header[MEMBRANE_TRACE_HEADER_SIZE];
	FILE		*f;
	uint32_t	expected;
	uint32_t	stored;
	uint64_t	i;
	const uint32_t	n_blocks = 5;
	const uint32_t	epb = 64;

	blocks = make_fixture(n_blocks, epb);
	le_payload = malloc((size_t)n_blocks * epb * sizeof(uint16_t));
	i = 0;
	while (i < (uint64_t)n_blocks * epb)
	{
		le_payload[2 * i] = (uint8_t)blocks[i];
		le_payload[2 * i + 1] = (uint8_t)(blocks[i] >> 8);
		i++;
	}
	expected = membrane_block_checksum(le_payload,
			(size_t)n_blocks * epb * sizeof(uint16_t));
	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open temp file for write");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, epb,
			n_blocks, blocks, "checksum fixture", 0) == MEMBRANE_OK,
		"write checksum fixture");
	fclose(f);
	f = fopen(g_path, "rb");
	TEST_ASSERT(f != NULL, "reopen temp file for read");
	TEST_ASSERT(fread(header, 1, sizeof(header), f) == sizeof(header),
		"read header back");
	fclose(f);
	stored = get_le32(header + 48);
	TEST_ASSERT(stored == expected,
		"trace_format.c's incremental CRC32 matches "
		"membrane_block_checksum on the same bytes");
	free(le_payload);
	free(blocks);
}

static void	test_writer_rejects_invalid_args(void)
{
	uint16_t	*blocks;
	FILE		*f;

	blocks = make_fixture(1, 32);
	f = fopen(g_path, "wb");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 0, 1,
			blocks, NULL, 0) == MEMBRANE_ERR_INVALID_ARG,
		"zero elements_per_block rejected by the writer");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 32, 0,
			blocks, NULL, 0) == MEMBRANE_ERR_INVALID_ARG,
		"zero block_count rejected by the writer");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16,
			MEMBRANE_TRACE_MAX_ELEMENTS_PER_BLOCK + 1, 1, blocks, NULL, 0)
			== MEMBRANE_ERR_INVALID_ARG,
		"oversized elements_per_block rejected by the writer");
	TEST_ASSERT(membrane_trace_write(f, MEMBRANE_TRACE_DTYPE_F16, 32,
			MEMBRANE_TRACE_MAX_BLOCK_COUNT + 1, blocks, NULL, 0)
			== MEMBRANE_ERR_INVALID_ARG,
		"oversized block_count rejected by the writer");
	fclose(f);
	free(blocks);
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file");
	close(fd);
	test_roundtrip_small();
	test_roundtrip_multi_block();
	test_bad_magic();
	test_unsupported_version();
	test_truncated_header();
	test_truncated_payload();
	test_payload_bytes_field_inconsistent();
	test_elements_per_block_zero_rejected();
	test_unsupported_dtype_rejected();
	test_zero_block_count_rejected();
	test_excessive_metadata_length_rejected();
	test_excessive_declared_size_rejected_without_allocating();
	test_trace_format_matches_membrane_block_checksum();
	test_writer_rejects_invalid_args();
	unlink(g_path);
	return (0);
}
