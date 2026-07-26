// Phase 5.4: functional smoke test for membrane_dma_bridge -- pushes a
// real Q8_0-encode command header, streams real F16 payload bytes,
// drains the packed output, and checks the completion record, all
// against the same golden vectors used by Phase 5.3's top-level test.
module tb_dma_bridge;
	localparam int N = 200;
	logic	[15:0]	x_flat		[0:N * 32 - 1];
	logic	[7:0]	q8pack_flat	[0:N * 34 - 1];

	logic	clk, rst_n;
	logic	reg_write, reg_read;
	logic	[7:0]	reg_addr;
	logic	[31:0]	reg_wdata, reg_rdata;
	logic	reg_ready;
	logic	cmd_push_valid, cmd_push_ready;
	logic	[511:0]	cmd_push_header;
	logic	payload_in_valid, payload_in_ready;
	logic	[31:0]	payload_in_data;
	logic	payload_out_valid, payload_out_ready;
	logic	[31:0]	payload_out_data;
	logic	completion_valid, completion_ready;
	logic	[127:0]	completion_record;

	membrane_dma_bridge dut (
		.clk(clk), .rst_n(rst_n),
		.reg_write(reg_write), .reg_read(reg_read), .reg_addr(reg_addr),
		.reg_wdata(reg_wdata), .reg_rdata(reg_rdata), .reg_ready(reg_ready),
		.cmd_push_valid(cmd_push_valid), .cmd_push_ready(cmd_push_ready),
		.cmd_push_header(cmd_push_header),
		.payload_in_valid(payload_in_valid), .payload_in_ready(payload_in_ready),
		.payload_in_data(payload_in_data),
		.payload_out_valid(payload_out_valid), .payload_out_ready(payload_out_ready),
		.payload_out_data(payload_out_data),
		.completion_valid(completion_valid), .completion_ready(completion_ready),
		.completion_record(completion_record));

	initial clk = 0;
	always #5 clk = ~clk;

	function automatic logic [511:0] build_header(
			input logic [63:0] txn_id, input logic [7:0] op,
			input logic [31:0] elem_count, input logic [31:0] in_len,
			input logic [31:0] out_cap);
		logic [511:0]	h;
		logic [31:0]	crc_bytes40;

		h = 512'h0;
		h[31:0] = 32'h4650424D;
		h[47:32] = 16'd1;
		h[63:48] = 16'd0;
		h[127:64] = txn_id;
		h[135:128] = op;
		h[191:160] = elem_count;
		h[223:192] = in_len;
		h[255:224] = out_cap;
		// header_checksum (bytes 40-43) intentionally left zero: the
		// bridge does not validate it (see membrane_dma_bridge.sv's
		// header comment on checksum scope), so a real host would fill
		// this via membrane_fpga_header_encode; this testbench only
		// needs the fields the bridge itself actually reads.
		build_header = h;
	endfunction

	int	blk;
	int	byte_idx;
	int	fails;
	logic [31:0] beat;

	initial begin
		rst_n = 0;
		reg_write = 0; reg_read = 0; reg_addr = 0; reg_wdata = 0;
		cmd_push_valid = 0; cmd_push_header = 0;
		payload_in_valid = 0; payload_in_data = 0;
		payload_out_ready = 0;
		completion_ready = 0;
		fails = 0;
		$readmemh("/tmp/top_x_120k.txt", x_flat);
		$readmemh("/tmp/top_q8pack_120k.txt", q8pack_flat);
		@(posedge clk);
		@(posedge clk);
		#2 rst_n = 1;
		#2;

		// push a Q8_0 encode command for N blocks
		cmd_push_header = build_header(64'hCAFEBABEDEADBEEF, 8'd0, N,
			N * 64, N * 34);
		cmd_push_valid = 1;
		@(posedge clk);
		while (!cmd_push_ready) @(posedge clk);
		#1 cmd_push_valid = 0;

		// stream payload in (N*64 bytes = N*16 beats of 4 bytes), while
		// concurrently draining output as it becomes available
		fork
			begin : drive_input
				blk = 0;
				byte_idx = 0;
				while (blk < N) begin
					beat = {x_flat[blk*32 + byte_idx/2 + 1],
						x_flat[blk*32 + byte_idx/2]};
					payload_in_data = beat;
					payload_in_valid = 1;
					@(posedge clk);
					while (!payload_in_ready) @(posedge clk);
					byte_idx += 4;
					if (byte_idx >= 64) begin
						byte_idx = 0;
						blk++;
					end
				end
				payload_in_valid = 0;
			end
			begin : drain_output
				int	got_blk;
				int	got_off;
				logic [7:0] got_byte;
				logic [7:0] exp_byte;

				got_blk = 0;
				got_off = 0;
				payload_out_ready = 1;
				while (got_blk < N) begin
					@(posedge clk);
					#1;
					if (payload_out_valid && payload_out_ready) begin
						for (int b = 0; b < 4; b++) begin
							if (got_off + b < 34) begin
								got_byte = payload_out_data[b*8 +: 8];
								exp_byte = q8pack_flat[got_blk*34 + got_off + b];
								if (got_byte !== exp_byte) begin
									if (fails < 10)
										$display("FAIL blk=%0d byte=%0d expect=%02h got=%02h",
											got_blk, got_off + b, exp_byte, got_byte);
									fails++;
								end
							end
						end
						got_off += 4;
						if (got_off >= 34) begin
							got_off = 0;
							got_blk++;
						end
					end
				end
			end
		join

		completion_ready = 1;
		@(posedge clk);
		while (!completion_valid) @(posedge clk);
		#1;
		if (completion_record[63:0] !== 64'hCAFEBABEDEADBEEF) begin
			$display("FAIL: completion txn_id mismatch, got=%016h", completion_record[63:0]);
			fails++;
		end
		if (completion_record[95:64] !== 32'd0) begin
			$display("FAIL: completion status nonzero: %08h", completion_record[95:64]);
			fails++;
		end
		if (completion_record[127:96] !== N * 34) begin
			$display("FAIL: completion output_byte_length mismatch: expect=%0d got=%0d",
				N * 34, completion_record[127:96]);
			fails++;
		end

		if (fails == 0)
			$display("PASS: dma_bridge Q8 encode, %0d blocks, 0 fails", N);
		else
			$display("FAIL: %0d mismatches", fails);
		$finish;
	end

	initial begin
		#2000000;
		$display("TIMEOUT");
		$finish;
	end
endmodule
