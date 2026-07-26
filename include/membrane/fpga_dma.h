#ifndef MEMBRANE_FPGA_DMA_H
# define MEMBRANE_FPGA_DMA_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 5.4: the host<->FPGA DMA packet format. One packet carries one
 * batch (element_count blocks) of a single operation. The header is a
 * fixed 64 bytes (one common DMA-descriptor/cache-line size), always
 * little-endian on the wire regardless of host byte order -- encoded/
 * decoded through membrane_fpga_header_encode/decode below rather than
 * memcpy'd as a C struct, so the wire format never depends on this
 * compiler's struct padding or this host's endianness (both of which
 * are otherwise unspecified/implementation-defined in C, and this is
 * meant to also describe what a real RTL bridge parses, where there is
 * no struct-layout escape hatch at all).
 *
 * Layout (byte offset : field : size):
 *   0  : magic              : 4  (MEMBRANE_FPGA_DMA_MAGIC)
 *   4  : version_major      : 2
 *   6  : version_minor      : 2
 *   8  : transaction_id     : 8  (host-assigned, opaque to the device;
 *                                 the RTL bridge only ever sees the low
 *                                 32 bits, used as its internal command-
 *                                 slot tag -- see fpga_regs.h)
 *   16 : operation          : 1  (e_membrane_fpga_op)
 *   17 : reserved0          : 3  (must be zero; explicit padding, not
 *                                 compiler-inserted)
 *   20 : element_count      : 4  (blocks in this batch)
 *   24 : input_byte_length  : 4  (payload bytes the device will read)
 *   28 : output_capacity    : 4  (payload bytes the host has reserved
 *                                 for the response; a request whose
 *                                 real output would exceed this is
 *                                 rejected with
 *                                 MEMBRANE_FPGA_ERR_SHORT_OUTPUT rather
 *                                 than silently truncated)
 *   32 : policy_layer_id    : 2  (opaque metadata passed through
 *                                 unmodified; used by the host runtime
 *                                 to route completions back to the
 *                                 right caller/tensor, never
 *                                 interpreted by the device)
 *   34 : policy_flags       : 2  (opaque metadata, same as above)
 *   36 : flags              : 4  (e_membrane_fpga_packet_flag bitmask)
 *   40 : header_checksum    : 4  (CRC32, membrane_block_checksum, over
 *                                 wire bytes [0,40) -- i.e. everything
 *                                 above, before this field)
 *   44 : payload_checksum   : 4  (CRC32 over the input_byte_length
 *                                 payload bytes that follow this
 *                                 header on the wire; zero when
 *                                 input_byte_length is zero)
 *   48 : reserved1          : 16 (must be zero; reserved for future
 *                                 fields without breaking the 64-byte
 *                                 size or shifting the checksum
 *                                 offsets)
 */
# define MEMBRANE_FPGA_DMA_MAGIC		0x4650424Du	/* "MBPF" LE */
# define MEMBRANE_FPGA_DMA_VERSION_MAJOR	1u
# define MEMBRANE_FPGA_DMA_VERSION_MINOR	0u
# define MEMBRANE_FPGA_DMA_HEADER_BYTES		64

typedef enum e_membrane_fpga_op
{
	MEMBRANE_FPGA_OP_Q8_ENCODE = 0,
	MEMBRANE_FPGA_OP_Q8_DECODE = 1,
	MEMBRANE_FPGA_OP_Q4_ENCODE = 2,
	MEMBRANE_FPGA_OP_Q4_DECODE = 3
}	membrane_fpga_op_t;

typedef enum e_membrane_fpga_packet_flag
{
	MEMBRANE_FPGA_FLAG_NONE		= 0,
	MEMBRANE_FPGA_FLAG_NO_FALLBACK	= 1u << 0,
	MEMBRANE_FPGA_FLAG_HIGH_PRIORITY	= 1u << 1
}	membrane_fpga_packet_flag_t;

typedef struct s_membrane_fpga_header
{
	uint32_t	magic;
	uint16_t	version_major;
	uint16_t	version_minor;
	uint64_t	transaction_id;
	uint8_t		operation;
	uint32_t	element_count;
	uint32_t	input_byte_length;
	uint32_t	output_capacity;
	uint16_t	policy_layer_id;
	uint16_t	policy_flags;
	uint32_t	flags;
	uint32_t	header_checksum;
	uint32_t	payload_checksum;
}	membrane_fpga_header_t;

/*
 * Encodes hdr into the 64-byte little-endian wire format at
 * out[0..MEMBRANE_FPGA_DMA_HEADER_BYTES), computing and filling in
 * header_checksum itself (any value the caller left in
 * hdr->header_checksum is ignored and overwritten). Caller is
 * responsible for having already set payload_checksum (computed over
 * the actual payload bytes via membrane_block_checksum, see
 * membrane/block.h) before calling this.
 */
void	membrane_fpga_header_encode(const membrane_fpga_header_t *hdr,
			uint8_t out[MEMBRANE_FPGA_DMA_HEADER_BYTES]);

/*
 * Decodes the 64-byte little-endian wire format at
 * in[0..MEMBRANE_FPGA_DMA_HEADER_BYTES) into *hdr. Does not itself
 * validate magic/version/checksum -- see membrane_fpga_header_validate.
 */
void	membrane_fpga_header_decode(
			const uint8_t in[MEMBRANE_FPGA_DMA_HEADER_BYTES],
			membrane_fpga_header_t *hdr);

typedef enum e_membrane_fpga_validate_result
{
	MEMBRANE_FPGA_VALIDATE_OK = 0,
	MEMBRANE_FPGA_VALIDATE_BAD_MAGIC,
	MEMBRANE_FPGA_VALIDATE_BAD_VERSION,
	MEMBRANE_FPGA_VALIDATE_BAD_HEADER_CHECKSUM,
	MEMBRANE_FPGA_VALIDATE_BAD_OPERATION,
	MEMBRANE_FPGA_VALIDATE_ZERO_ELEMENTS
}	membrane_fpga_validate_result_t;

/*
 * Structural validation of a decoded header: magic, major version
 * (minor version differences are tolerated, matching ordinary
 * semantic-versioning-style forward compatibility), header_checksum
 * (recomputed from the raw 64-byte wire bytes passed in `raw`, NOT
 * from hdr's fields -- the checksum must catch corruption of the
 * wire bytes themselves, not just corruption a decode step might
 * silently repair), a recognized operation value, and a nonzero
 * element_count. Does NOT check payload_checksum (the payload has not
 * necessarily arrived yet when a header is first validated) -- see
 * membrane_fpga_payload_checksum_ok for that, called once the payload
 * bytes are actually available.
 */
membrane_fpga_validate_result_t	membrane_fpga_header_validate(
			const membrane_fpga_header_t *hdr,
			const uint8_t raw[MEMBRANE_FPGA_DMA_HEADER_BYTES]);

/*
 * True if the CRC32 of payload[0..len) matches hdr->payload_checksum.
 */
int	membrane_fpga_payload_checksum_ok(const membrane_fpga_header_t *hdr,
			const uint8_t *payload, size_t len);

# ifdef __cplusplus
}
# endif

#endif
