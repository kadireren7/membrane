#ifndef MEMBRANE_FPGA_REGS_H
# define MEMBRANE_FPGA_REGS_H

/*
 * Phase 5.4: the FPGA DMA bridge's MMIO register map. This is the fixed
 * ABI between the host runtime (tools/membrane-fpga-runtime,
 * src/fpga/) and the RTL DMA bridge (rtl/membrane_dma_bridge.sv) --
 * both sides implement this exact byte-offset/bit-field layout, kept
 * in sync by hand (the RTL side cannot #include a C header) with the
 * register offsets/widths/reset-behavior documented in
 * rtl/membrane_dma_bridge.sv's own header comment cross-referencing
 * this file. Every register is 32 bits, at a 4-byte-aligned offset,
 * little-endian on the wire (matching fpga_dma.h's packet format).
 *
 * All registers live in one 64-byte (0x00-0x3C) MMIO block:
 *
 *   0x00  VERSION            RO   [31:16]=major [15:0]=minor
 *   0x04  CAPABILITIES       RO   bit flags, see
 *                                 e_membrane_fpga_cap below
 *   0x08  QUEUE_BASE_LO      RW   command-queue base address, low 32
 *   0x0C  QUEUE_BASE_HI      RW   command-queue base address, high 32
 *   0x10  QUEUE_SIZE         RW   command-queue depth (must be a
 *                                 power of two; matches
 *                                 stream_fifo.sv's own DEPTH
 *                                 constraint, see rtl/stream_fifo.sv)
 *   0x14  DOORBELL           WO   host writes the new command-queue
 *                                 producer index every time it enqueues
 *                                 one or more new commands
 *   0x18  COMPLETION_HEAD    RW   host-managed completion-queue
 *                                 consumer index; host writes this
 *                                 after consuming completions to let
 *                                 the device reclaim the slots
 *   0x1C  COMPLETION_TAIL    RO   device-managed completion-queue
 *                                 producer index; host polls this (or
 *                                 waits on an interrupt/eventfd in a
 *                                 real driver) to discover new
 *                                 completions
 *   0x20  ERROR_FLAGS        RW1C bit flags, see
 *                                 e_membrane_fpga_error below; write a
 *                                 1 to a bit to clear it, write 0 has
 *                                 no effect (standard RW1C so a
 *                                 read-modify-write race between two
 *                                 error sources cannot silently drop
 *                                 one of them)
 *   0x24  PROCESSED_BLOCKS   RO   cumulative count of blocks retired
 *                                 since the last RESET
 *   0x28  STALL_CYCLES       RO   cumulative count of cycles the
 *                                 bridge's input DMA reader was
 *                                 stalled waiting for
 *                                 membrane_quant_stream_top's
 *                                 `in_ready` since the last RESET
 *   0x2C  INPUT_BYTES_LO     RO   cumulative input payload bytes
 *                                 consumed, low 32
 *   0x30  INPUT_BYTES_HI     RO   cumulative input payload bytes
 *                                 consumed, high 32
 *   0x34  OUTPUT_BYTES_LO    RO   cumulative output payload bytes
 *                                 produced, low 32
 *   0x38  OUTPUT_BYTES_HI    RO   cumulative output payload bytes
 *                                 produced, high 32
 *   0x3C  RESET              WO   write 1 to trigger a soft reset
 *                                 (clears all queues and counters,
 *                                 equivalent to the RTL bridge's own
 *                                 `rst_n` deassertion); reads as 0
 *
 * DOORBELL/COMPLETION_HEAD/COMPLETION_TAIL are plain incrementing
 * indices (not wrapped by software), matching the same
 * wraparound-via-MSB-compare convention rtl/stream_fifo.sv already
 * uses internally -- QUEUE_SIZE must be a power of two so
 * `index & (QUEUE_SIZE-1)` recovers the physical slot.
 */

# define MEMBRANE_FPGA_REG_VERSION		0x00u
# define MEMBRANE_FPGA_REG_CAPABILITIES	0x04u
# define MEMBRANE_FPGA_REG_QUEUE_BASE_LO	0x08u
# define MEMBRANE_FPGA_REG_QUEUE_BASE_HI	0x0Cu
# define MEMBRANE_FPGA_REG_QUEUE_SIZE		0x10u
# define MEMBRANE_FPGA_REG_DOORBELL		0x14u
# define MEMBRANE_FPGA_REG_COMPLETION_HEAD	0x18u
# define MEMBRANE_FPGA_REG_COMPLETION_TAIL	0x1Cu
# define MEMBRANE_FPGA_REG_ERROR_FLAGS		0x20u
# define MEMBRANE_FPGA_REG_PROCESSED_BLOCKS	0x24u
# define MEMBRANE_FPGA_REG_STALL_CYCLES	0x28u
# define MEMBRANE_FPGA_REG_INPUT_BYTES_LO	0x2Cu
# define MEMBRANE_FPGA_REG_INPUT_BYTES_HI	0x30u
# define MEMBRANE_FPGA_REG_OUTPUT_BYTES_LO	0x34u
# define MEMBRANE_FPGA_REG_OUTPUT_BYTES_HI	0x38u
# define MEMBRANE_FPGA_REG_RESET		0x3Cu
# define MEMBRANE_FPGA_REG_SPACE_BYTES		0x40u

/* CAPABILITIES bits */
# define MEMBRANE_FPGA_CAP_Q8_ENCODE	(1u << 0)
# define MEMBRANE_FPGA_CAP_Q8_DECODE	(1u << 1)
# define MEMBRANE_FPGA_CAP_Q4_ENCODE	(1u << 2)
# define MEMBRANE_FPGA_CAP_Q4_DECODE	(1u << 3)
# define MEMBRANE_FPGA_CAP_CHECKSUM	(1u << 4)

/*
 * ERROR_FLAGS bits (RW1C). Each corresponds to a specific stress-test
 * condition this phase's DMA stress suite exercises deliberately (see
 * docs/phase5-pcie-hardware-loop.md section 7).
 */
# define MEMBRANE_FPGA_ERR_BAD_HEADER_CHECKSUM		(1u << 0)
# define MEMBRANE_FPGA_ERR_BAD_PAYLOAD_CHECKSUM	(1u << 1)
# define MEMBRANE_FPGA_ERR_MALFORMED_HEADER		(1u << 2)
# define MEMBRANE_FPGA_ERR_QUEUE_FULL			(1u << 3)
# define MEMBRANE_FPGA_ERR_SHORT_OUTPUT		(1u << 4)
# define MEMBRANE_FPGA_ERR_TIMEOUT			(1u << 5)
# define MEMBRANE_FPGA_ERR_COMPLETION_OVERFLOW		(1u << 6)

#endif
