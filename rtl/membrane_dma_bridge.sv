// Phase 5.4: membrane_dma_bridge -- wraps membrane_quant_stream_top with
// a command queue, payload streamer, completion queue, and the MMIO
// register file documented in include/membrane/fpga_regs.h, so the
// whole thing can sit behind a host-side DMA emulation (or, eventually,
// a real PCIe DMA engine) without membrane_quant_stream_top itself
// needing any awareness of packets, batching, or MMIO.
//
// ---- scope, disclosed up front ----
// - Vendor-neutral: this is NOT a real AXI4/AXI4-Lite or PCIe TLP
//   interface, and does not attempt to be. The register port is a
//   plain synchronous address/data/strobe interface; the payload ports
//   are plain 32-bit-per-beat valid/ready streams. A real integration
//   would sit a vendor's AXI4-Lite-to-this-interface and DMA-engine-to-
//   this-interface adapter in front of these ports -- not written here,
//   see docs/phase5-pcie-hardware-loop.md section 4.
// - Command-queue and completion-queue storage is entirely ON-DEVICE
//   (two stream_fifo.sv instances), not a host-memory descriptor ring
//   fetched via DMA reads. QUEUE_BASE_LO/HI and QUEUE_SIZE exist in the
//   register map for ABI forward-compatibility with a future host-
//   memory-ring design but have no effect on this bridge's behavior
//   today (RW, read-back-what-was-written, otherwise inert) -- disclosed
//   as such, not silently ignored.
// - Only ONE command is "in flight" (streaming payload in / collecting
//   output) at the bridge level at a time; a command's OWN
//   `element_count` blocks ARE fully pipelined through
//   membrane_quant_stream_top for real batch throughput (the
//   performance-relevant case), but the bridge does not interleave
//   blocks from two DIFFERENT commands. A production bridge wanting to
//   keep the pipeline busy across command boundaries would need a
//   {command-slot, block-index} tag in `in_id` instead of this design's
//   plain block-index tag -- not done here, disclosed as a scope
//   decision, not an oversight.
// - Header/payload CRC32 checksum verification (fpga_dma.h's
//   header_checksum/payload_checksum) is NOT re-implemented as a
//   hardware CRC engine here -- it is checked by the host-side
//   emulation harness before a command is pushed (using the identical
//   membrane_block_checksum function that computed it), representing
//   where a real system's root-complex-adjacent logic would sit. This
//   bridge DOES validate header STRUCTURE in hardware (magic, version,
//   operation code, non-zero element_count, output_capacity sizing) --
//   see MEMBRANE_FPGA_ERR_MALFORMED_HEADER/_SHORT_OUTPUT below. A real
//   production bridge would likely want defense-in-depth with an
//   on-device CRC engine too, for corruption introduced in the DMA path
//   itself; not built here, disclosed as a known gap.
//
// ---- operation <-> membrane_quant_stream_top mode mapping ----
// Deliberately identical numeric encoding, not a coincidence:
// MEMBRANE_FPGA_OP_Q8_ENCODE=0/Q8_DECODE=1/Q4_ENCODE=2/Q4_DECODE=3
// (include/membrane/fpga_dma.h) equal membrane_quant_stream_top's own
// MODE_Q8_ENC=2'b00/MODE_Q8_DEC=2'b01/MODE_Q4_ENC=2'b10/MODE_Q4_DEC=2'b11
// (rtl/membrane_quant_stream_top.sv), so `in_mode = header.operation[1:0]`
// needs no remapping table.
module membrane_dma_bridge #(
	parameter int ID_WIDTH = 16,
	parameter int CMD_FIFO_DEPTH = 16,
	parameter int COMPLETION_FIFO_DEPTH = 16,
	parameter int RESULT_FIFO_DEPTH = 8
) (
	input	logic			clk,
	input	logic			rst_n,

	// ---- register interface (see membrane/fpga_regs.h) ----
	input	logic			reg_write,
	input	logic			reg_read,
	input	logic	[7:0]	reg_addr,
	input	logic	[31:0]	reg_wdata,
	output	logic	[31:0]	reg_rdata,
	output	logic			reg_ready,

	// ---- command push: one 64-byte (512-bit) header per push, bytes
	// packed LSB-first (header_bits[8*i +: 8] = wire_byte[i]), matching
	// membrane_fpga_header_encode's output byte order exactly ----
	input	logic			cmd_push_valid,
	output	logic			cmd_push_ready,
	input	logic	[511:0]	cmd_push_header,

	// ---- input payload stream, 32 bits/beat, LSB-first ----
	input	logic			payload_in_valid,
	output	logic			payload_in_ready,
	input	logic	[31:0]	payload_in_data,

	// ---- output payload stream, 32 bits/beat, LSB-first ----
	output	logic			payload_out_valid,
	input	logic			payload_out_ready,
	output	logic	[31:0]	payload_out_data,

	// ---- completion pop: {output_byte_length[127:96], status[95:64],
	// transaction_id[63:0]} ----
	output	logic			completion_valid,
	input	logic			completion_ready,
	output	logic	[127:0]	completion_record
);
	// ---- soft reset (MEMBRANE_FPGA_REG_RESET) ----
	logic	soft_reset_pulse;
	logic	bridge_rst_n;

	assign bridge_rst_n = rst_n && !soft_reset_pulse;

	// =====================================================================
	// Command FIFO (host -> bridge)
	// =====================================================================
	logic			cmd_fifo_out_valid;
	logic			cmd_fifo_pop;
	logic	[511:0]	cmd_fifo_out_header;
	logic	[$clog2(CMD_FIFO_DEPTH):0]	cmd_fifo_occ;

	stream_fifo #(.WIDTH(512), .DEPTH(CMD_FIFO_DEPTH)) u_cmd_fifo (
		.clk(clk), .rst_n(bridge_rst_n),
		.in_valid(cmd_push_valid), .in_ready(cmd_push_ready),
		.in_data(cmd_push_header),
		.out_valid(cmd_fifo_out_valid), .out_ready(cmd_fifo_pop),
		.out_data(cmd_fifo_out_header), .occupancy(cmd_fifo_occ));

	// ---- header field extraction (byte offsets match fpga_dma.h) ----
	logic	[31:0]	hdr_magic;
	logic	[15:0]	hdr_version_major;
	logic	[63:0]	hdr_transaction_id;
	logic	[7:0]	hdr_operation;
	logic	[31:0]	hdr_element_count;
	logic	[31:0]	hdr_output_capacity;

	assign hdr_magic = cmd_fifo_out_header[31:0];
	assign hdr_version_major = cmd_fifo_out_header[47:32];
	assign hdr_transaction_id = cmd_fifo_out_header[127:64];
	assign hdr_operation = cmd_fifo_out_header[135:128];
	assign hdr_element_count = cmd_fifo_out_header[191:160];
	assign hdr_output_capacity = cmd_fifo_out_header[255:224];

	// =====================================================================
	// Completion FIFO (bridge -> host)
	// =====================================================================
	logic			completion_push_valid;
	logic			completion_push_ready;
	logic	[127:0]	completion_push_record;

	stream_fifo #(.WIDTH(128), .DEPTH(COMPLETION_FIFO_DEPTH)) u_completion_fifo (
		.clk(clk), .rst_n(bridge_rst_n),
		.in_valid(completion_push_valid), .in_ready(completion_push_ready),
		.in_data(completion_push_record),
		.out_valid(completion_valid), .out_ready(completion_ready),
		.out_data(completion_record), .occupancy());

	// =====================================================================
	// Result FIFO: completed output blocks awaiting host drain. Bundles
	// {byte_count[6:0], block_bytes[511:0]} = 519 bits; out_ready is
	// wired straight to membrane_quant_stream_top's out_ready, so this
	// FIFO filling up naturally backpressures the compute pipeline --
	// no separate flow-control logic needed.
	// =====================================================================
	logic			result_push_valid;
	logic			result_push_ready;
	logic	[518:0]	result_push_data;
	logic			result_pop_valid;
	logic			result_pop_ready;
	logic	[518:0]	result_pop_data;

	stream_fifo #(.WIDTH(519), .DEPTH(RESULT_FIFO_DEPTH)) u_result_fifo (
		.clk(clk), .rst_n(bridge_rst_n),
		.in_valid(result_push_valid), .in_ready(result_push_ready),
		.in_data(result_push_data),
		.out_valid(result_pop_valid), .out_ready(result_pop_ready),
		.out_data(result_pop_data), .occupancy());

	// =====================================================================
	// membrane_quant_stream_top instance
	// =====================================================================
	logic			top_in_valid, top_in_ready;
	logic	[1:0]	top_in_mode;
	logic	[ID_WIDTH-1:0]	top_in_id;
	logic	[511:0]	top_in_data;
	logic			top_out_valid, top_out_ready;
	logic	[1:0]	top_out_mode;
	logic	[ID_WIDTH-1:0]	top_out_id;
	logic	[511:0]	top_out_data;
	logic			top_out_error;

	membrane_quant_stream_top #(.ID_WIDTH(ID_WIDTH)) u_top (
		.clk(clk), .rst_n(bridge_rst_n),
		.in_valid(top_in_valid), .in_ready(top_in_ready),
		.in_mode(top_in_mode), .in_id(top_in_id), .in_data(top_in_data),
		.out_valid(top_out_valid), .out_ready(top_out_ready),
		.out_mode(top_out_mode), .out_id(top_out_id), .out_data(top_out_data),
		.out_error(top_out_error));

	assign top_out_ready = result_push_ready;
	assign result_push_valid = top_out_valid;

	// =====================================================================
	// Per-operation block byte sizes
	// =====================================================================
	// Both functions return the byte count rounded UP to the next 4-byte
	// boundary (the payload ports' own beat width), NOT the true Q8_0
	// (34) / Q4_0 (18) packed-block size. This is a real, deliberate
	// design fix, not an oversight: with multiple blocks streamed
	// back-to-back in one batch, a per-block size that ISN'T a multiple
	// of the 4-byte beat width causes the accumulator/drainer state to
	// straddle a block boundary mid-beat -- the bytes belonging to the
	// NEXT block that arrive bundled into the SAME beat as the CURRENT
	// block's last byte were being silently discarded when the
	// accumulator reset for the next block, corrupting every block
	// after the first misaligned one in a batch. Caught by
	// tools/membrane-fpga-runtime's multi-block batch verification
	// (task 111), not by the single-block smoke test, since a
	// single-block "batch" never exercises a block-to-block boundary at
	// all. Padding each block to a 4-byte-aligned stride on the wire
	// (2 pad bytes after every Q8_0 block, 2 after every Q4_0 block)
	// sidesteps the whole class of bug instead of trying to track
	// sub-beat carry-over state; the host-side padding/unpadding is in
	// tools/membrane-fpga-runtime/main.cpp's block byte-stride helpers.
	// The real (unpadded) per-block size is never needed anywhere else
	// in this module -- SHORT_OUTPUT sizing and the completion record's
	// output_byte_length are consistently in PADDED terms too, so the
	// host's own output_capacity/payload_pull sizing must match.
	function automatic logic [6:0] op_input_bytes(input logic [1:0] op);
		if (op == 2'b00 || op == 2'b10)
			op_input_bytes = 7'd64;
		else if (op == 2'b01)
			op_input_bytes = 7'd36;
		else
			op_input_bytes = 7'd20;
	endfunction

	function automatic logic [6:0] op_output_bytes(input logic [1:0] op);
		if (op == 2'b01 || op == 2'b11)
			op_output_bytes = 7'd64;
		else if (op == 2'b00)
			op_output_bytes = 7'd36;
		else
			op_output_bytes = 7'd20;
	endfunction

	// =====================================================================
	// Command processor FSM
	// =====================================================================
	localparam logic [2:0] ST_IDLE = 3'd0;
	localparam logic [2:0] ST_CHECK = 3'd1;
	localparam logic [2:0] ST_STREAM = 3'd2;
	localparam logic [2:0] ST_ERROR = 3'd3;
	localparam logic [2:0] ST_COMPLETE = 3'd4;

	logic	[2:0]	state;
	logic	[1:0]	cur_op;
	logic	[63:0]	cur_txn_id;
	logic	[31:0]	cur_element_count;
	logic	[6:0]	cur_in_bytes_per_block;
	logic	[6:0]	cur_out_bytes_per_block;
	logic	[31:0]	blocks_issued;
	logic	[31:0]	blocks_completed;
	logic	[31:0]	cur_error_status;

	// ---- input accumulator: gathers payload_in_data beats into one
	// 512-bit block, MSB-padded with zero above the operation's real
	// input byte width (matching how membrane_quant_stream_top's
	// decode-mode inputs already tolerate don't-care upper bits, see
	// its own header comment on the shared 512-bit bus convention). ----
	logic	[511:0]	in_accum;
	logic	[6:0]	in_accum_bytes;
	logic			in_block_ready;

	assign in_block_ready = (in_accum_bytes >= cur_in_bytes_per_block);
	assign payload_in_ready = (state == ST_STREAM)
		&& (blocks_issued < cur_element_count) && !in_block_ready;

	always_ff @(posedge clk or negedge bridge_rst_n) begin
		if (!bridge_rst_n) begin
			in_accum <= '0;
			in_accum_bytes <= 7'd0;
		end else if (payload_in_valid && payload_in_ready) begin
			in_accum[in_accum_bytes * 8 +: 32] <= payload_in_data;
			in_accum_bytes <= in_accum_bytes + 7'd4;
		end else if (top_in_valid && top_in_ready) begin
			in_accum <= '0;
			in_accum_bytes <= 7'd0;
		end
	end

	assign top_in_valid = (state == ST_STREAM) && in_block_ready
		&& (blocks_issued < cur_element_count);
	assign top_in_mode = cur_op;
	assign top_in_id = blocks_issued[ID_WIDTH-1:0];
	assign top_in_data = in_accum;

	// ---- output drain: pops one result-fifo entry at a time, streams
	// its byte_count bytes out 4 bytes/beat. ----
	logic	[511:0]	out_drain_bytes;
	logic	[6:0]	out_drain_total;
	logic	[6:0]	out_drain_offset;
	logic			out_drain_active;

	assign out_drain_active = (out_drain_offset < out_drain_total);
	assign result_pop_ready = !out_drain_active;
	assign payload_out_valid = out_drain_active;
	assign payload_out_data = out_drain_bytes[out_drain_offset * 8 +: 32];

	always_ff @(posedge clk or negedge bridge_rst_n) begin
		if (!bridge_rst_n) begin
			out_drain_bytes <= '0;
			out_drain_total <= 7'd0;
			out_drain_offset <= 7'd0;
			blocks_completed <= 32'd0;
		end else begin
			if (result_pop_valid && result_pop_ready) begin
				out_drain_bytes <= result_pop_data[511:0];
				out_drain_total <= result_pop_data[518:512];
				out_drain_offset <= 7'd0;
			end else if (payload_out_valid && payload_out_ready) begin
				out_drain_offset <= out_drain_offset + 7'd4;
				if (out_drain_offset + 7'd4 >= out_drain_total)
					blocks_completed <= blocks_completed + 32'd1;
			end
			if (state == ST_IDLE || state == ST_CHECK)
				blocks_completed <= 32'd0;
		end
	end

	assign result_push_data = {cur_out_bytes_per_block, top_out_data};

	// ---- counters (MMIO-readable) ----
	logic	[31:0]	reg_processed_blocks;
	logic	[31:0]	reg_stall_cycles;
	logic	[63:0]	reg_input_bytes;
	logic	[63:0]	reg_output_bytes;
	logic	[31:0]	reg_error_flags;
	logic	[31:0]	reg_completion_tail_ctr;
	logic	[31:0]	reg_completion_head;
	logic	[31:0]	reg_doorbell_ctr;
	logic	[63:0]	reg_queue_base;
	logic	[31:0]	reg_queue_size;

	always_ff @(posedge clk or negedge bridge_rst_n) begin
		if (!bridge_rst_n) begin
			reg_processed_blocks <= 32'd0;
			reg_stall_cycles <= 32'd0;
			reg_input_bytes <= 64'd0;
			reg_output_bytes <= 64'd0;
		end else begin
			if (top_in_valid && top_in_ready)
				reg_input_bytes <= reg_input_bytes + {57'd0, cur_in_bytes_per_block};
			if (top_out_valid && top_out_ready) begin
				reg_output_bytes <= reg_output_bytes + {57'd0, cur_out_bytes_per_block};
				reg_processed_blocks <= reg_processed_blocks + 32'd1;
			end
			if ((state == ST_STREAM) && !top_in_ready)
				reg_stall_cycles <= reg_stall_cycles + 32'd1;
		end
	end

	// ---- main FSM ----
	always_ff @(posedge clk or negedge bridge_rst_n) begin
		if (!bridge_rst_n) begin
			state <= ST_IDLE;
			cur_op <= 2'b00;
			cur_txn_id <= 64'd0;
			cur_element_count <= 32'd0;
			cur_in_bytes_per_block <= 7'd0;
			cur_out_bytes_per_block <= 7'd0;
			blocks_issued <= 32'd0;
			cur_error_status <= 32'd0;
			cmd_fifo_pop <= 1'b0;
			completion_push_valid <= 1'b0;
			completion_push_record <= 128'd0;
		end else begin
			cmd_fifo_pop <= 1'b0;
			completion_push_valid <= 1'b0;
			if (state == ST_IDLE) begin
				if (cmd_fifo_out_valid) begin
					cmd_fifo_pop <= 1'b1;
					cur_op <= hdr_operation[1:0];
					cur_txn_id <= hdr_transaction_id;
					cur_element_count <= hdr_element_count;
					cur_in_bytes_per_block <= op_input_bytes(hdr_operation[1:0]);
					cur_out_bytes_per_block <= op_output_bytes(hdr_operation[1:0]);
					blocks_issued <= 32'd0;
					state <= ST_CHECK;
				end
			end else if (state == ST_CHECK) begin
				if (hdr_magic != 32'h4650424D || hdr_version_major != 16'd1
						|| hdr_element_count == 32'd0
						|| hdr_operation[7:2] != 6'd0) begin
					cur_error_status <= 32'h4; // MEMBRANE_FPGA_ERR_MALFORMED_HEADER
					state <= ST_ERROR;
				end else if ({25'd0, cur_element_count} * {57'd0, cur_out_bytes_per_block}
						> {32'd0, hdr_output_capacity}) begin
					cur_error_status <= 32'h10; // MEMBRANE_FPGA_ERR_SHORT_OUTPUT
					state <= ST_ERROR;
				end else begin
					cur_error_status <= 32'd0;
					state <= ST_STREAM;
				end
			end else if (state == ST_STREAM) begin
				if (top_in_valid && top_in_ready)
					blocks_issued <= blocks_issued + 32'd1;
				if (blocks_completed >= cur_element_count)
					state <= ST_COMPLETE;
			end else if (state == ST_ERROR) begin
				completion_push_valid <= 1'b1;
				completion_push_record <= {32'd0, cur_error_status, cur_txn_id};
				state <= ST_IDLE;
			end else if (state == ST_COMPLETE) begin
				completion_push_valid <= 1'b1;
				completion_push_record <= {
					cur_element_count * {25'd0, cur_out_bytes_per_block},
					cur_error_status, cur_txn_id};
				state <= ST_IDLE;
			end
		end
	end

	// =====================================================================
	// Register file (see include/membrane/fpga_regs.h)
	// =====================================================================
	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			reg_error_flags <= 32'd0;
			reg_completion_head <= 32'd0;
			reg_doorbell_ctr <= 32'd0;
			reg_queue_base <= 64'd0;
			reg_queue_size <= 32'd0;
			soft_reset_pulse <= 1'b0;
		end else begin
			soft_reset_pulse <= 1'b0;
			if (state == ST_ERROR)
				reg_error_flags <= reg_error_flags | cur_error_status;
			if (reg_write) begin
				if (reg_addr == 8'h08)
					reg_queue_base[31:0] <= reg_wdata;
				else if (reg_addr == 8'h0C)
					reg_queue_base[63:32] <= reg_wdata;
				else if (reg_addr == 8'h10)
					reg_queue_size <= reg_wdata;
				else if (reg_addr == 8'h14)
					reg_doorbell_ctr <= reg_doorbell_ctr + 32'd1;
				else if (reg_addr == 8'h18)
					reg_completion_head <= reg_wdata;
				else if (reg_addr == 8'h20)
					reg_error_flags <= reg_error_flags & ~reg_wdata;
				else if (reg_addr == 8'h3C && reg_wdata[0])
					soft_reset_pulse <= 1'b1;
			end
		end
	end

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n)
			reg_completion_tail_ctr <= 32'd0;
		else if (completion_valid && completion_ready)
			reg_completion_tail_ctr <= reg_completion_tail_ctr + 32'd1;
	end

	always_comb begin
		reg_ready = reg_write || reg_read;
		if (reg_addr == 8'h00)
			reg_rdata = {16'd1, 16'd0}; // VERSION 1.0
		else if (reg_addr == 8'h04)
			reg_rdata = 32'h1F; // CAPABILITIES: all 4 ops + checksum-aware
		else if (reg_addr == 8'h08)
			reg_rdata = reg_queue_base[31:0];
		else if (reg_addr == 8'h0C)
			reg_rdata = reg_queue_base[63:32];
		else if (reg_addr == 8'h10)
			reg_rdata = reg_queue_size;
		else if (reg_addr == 8'h14)
			reg_rdata = reg_doorbell_ctr;
		else if (reg_addr == 8'h18)
			reg_rdata = reg_completion_head;
		else if (reg_addr == 8'h1C)
			reg_rdata = reg_completion_tail_ctr;
		else if (reg_addr == 8'h20)
			reg_rdata = reg_error_flags;
		else if (reg_addr == 8'h24)
			reg_rdata = reg_processed_blocks;
		else if (reg_addr == 8'h28)
			reg_rdata = reg_stall_cycles;
		else if (reg_addr == 8'h2C)
			reg_rdata = reg_input_bytes[31:0];
		else if (reg_addr == 8'h30)
			reg_rdata = reg_input_bytes[63:32];
		else if (reg_addr == 8'h34)
			reg_rdata = reg_output_bytes[31:0];
		else if (reg_addr == 8'h38)
			reg_rdata = reg_output_bytes[63:32];
		else
			reg_rdata = 32'd0;
	end
endmodule
