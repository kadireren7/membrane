#include <string.h>

#include "membrane/block.h"
#include "membrane/fpga_dma.h"

static void	put_u16le(uint8_t *p, uint16_t v)
{
	p[0] = (uint8_t)(v & 0xFFu);
	p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void	put_u32le(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)(v & 0xFFu);
	p[1] = (uint8_t)((v >> 8) & 0xFFu);
	p[2] = (uint8_t)((v >> 16) & 0xFFu);
	p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void	put_u64le(uint8_t *p, uint64_t v)
{
	put_u32le(p, (uint32_t)(v & 0xFFFFFFFFu));
	put_u32le(p + 4, (uint32_t)((v >> 32) & 0xFFFFFFFFu));
}

static uint16_t	get_u16le(const uint8_t *p)
{
	return ((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t	get_u32le(const uint8_t *p)
{
	uint32_t	v;

	v = (uint32_t)p[0];
	v |= (uint32_t)p[1] << 8;
	v |= (uint32_t)p[2] << 16;
	v |= (uint32_t)p[3] << 24;
	return (v);
}

static uint64_t	get_u64le(const uint8_t *p)
{
	uint64_t	lo;
	uint64_t	hi;

	lo = get_u32le(p);
	hi = get_u32le(p + 4);
	return (lo | (hi << 32));
}

void	membrane_fpga_header_encode(const membrane_fpga_header_t *hdr,
			uint8_t out[MEMBRANE_FPGA_DMA_HEADER_BYTES])
{
	uint32_t	crc;

	memset(out, 0, MEMBRANE_FPGA_DMA_HEADER_BYTES);
	put_u32le(out + 0, hdr->magic);
	put_u16le(out + 4, hdr->version_major);
	put_u16le(out + 6, hdr->version_minor);
	put_u64le(out + 8, hdr->transaction_id);
	out[16] = hdr->operation;
	put_u32le(out + 20, hdr->element_count);
	put_u32le(out + 24, hdr->input_byte_length);
	put_u32le(out + 28, hdr->output_capacity);
	put_u16le(out + 32, hdr->policy_layer_id);
	put_u16le(out + 34, hdr->policy_flags);
	put_u32le(out + 36, hdr->flags);
	crc = membrane_block_checksum(out, 40);
	put_u32le(out + 40, crc);
	put_u32le(out + 44, hdr->payload_checksum);
}

void	membrane_fpga_header_decode(
			const uint8_t in[MEMBRANE_FPGA_DMA_HEADER_BYTES],
			membrane_fpga_header_t *hdr)
{
	memset(hdr, 0, sizeof(*hdr));
	hdr->magic = get_u32le(in + 0);
	hdr->version_major = get_u16le(in + 4);
	hdr->version_minor = get_u16le(in + 6);
	hdr->transaction_id = get_u64le(in + 8);
	hdr->operation = in[16];
	hdr->element_count = get_u32le(in + 20);
	hdr->input_byte_length = get_u32le(in + 24);
	hdr->output_capacity = get_u32le(in + 28);
	hdr->policy_layer_id = get_u16le(in + 32);
	hdr->policy_flags = get_u16le(in + 34);
	hdr->flags = get_u32le(in + 36);
	hdr->header_checksum = get_u32le(in + 40);
	hdr->payload_checksum = get_u32le(in + 44);
}

static int	operation_is_valid(uint8_t op)
{
	return (op == MEMBRANE_FPGA_OP_Q8_ENCODE
		|| op == MEMBRANE_FPGA_OP_Q8_DECODE
		|| op == MEMBRANE_FPGA_OP_Q4_ENCODE
		|| op == MEMBRANE_FPGA_OP_Q4_DECODE);
}

membrane_fpga_validate_result_t	membrane_fpga_header_validate(
			const membrane_fpga_header_t *hdr,
			const uint8_t raw[MEMBRANE_FPGA_DMA_HEADER_BYTES])
{
	uint32_t	crc;

	if (hdr->magic != MEMBRANE_FPGA_DMA_MAGIC)
		return (MEMBRANE_FPGA_VALIDATE_BAD_MAGIC);
	if (hdr->version_major != MEMBRANE_FPGA_DMA_VERSION_MAJOR)
		return (MEMBRANE_FPGA_VALIDATE_BAD_VERSION);
	crc = membrane_block_checksum(raw, 40);
	if (crc != hdr->header_checksum)
		return (MEMBRANE_FPGA_VALIDATE_BAD_HEADER_CHECKSUM);
	if (!operation_is_valid(hdr->operation))
		return (MEMBRANE_FPGA_VALIDATE_BAD_OPERATION);
	if (hdr->element_count == 0)
		return (MEMBRANE_FPGA_VALIDATE_ZERO_ELEMENTS);
	return (MEMBRANE_FPGA_VALIDATE_OK);
}

int	membrane_fpga_payload_checksum_ok(const membrane_fpga_header_t *hdr,
			const uint8_t *payload, size_t len)
{
	uint32_t	crc;

	crc = membrane_block_checksum(payload, len);
	return (crc == hdr->payload_checksum);
}
