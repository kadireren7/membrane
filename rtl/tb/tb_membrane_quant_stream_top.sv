// Phase 5.3: functional smoke test for membrane_quant_stream_top --
// fast Icarus-based wiring/latency-arithmetic validation on a modest
// vector count (a few hundred blocks per mode, plus a mixed-mode
// interleaved run), with randomized valid/ready backpressure on both
// sides. The large-scale (100,000+ per direction/format) verification
// required by the phase spec runs separately under Verilator (see
// rtl/tb/tb_top_verilator.cpp), which this file's discipline mirrors:
// this test exists to catch pipeline wiring/latency-arithmetic bugs
// cheaply and quickly before committing to a slow, large Verilator run.
module tb_membrane_quant_stream_top;
	localparam int ID_WIDTH = 16;
	localparam int N_PER_MODE = 300;
	localparam int N_MIX = 400;

	logic	clk, rst_n;
	logic	in_valid, in_ready;
	logic	[1:0]	in_mode;
	logic	[ID_WIDTH-1:0]	in_id;
	logic	[511:0]	in_data;
	logic	out_valid, out_ready;
	logic	[1:0]	out_mode;
	logic	[ID_WIDTH-1:0]	out_id;
	logic	[511:0]	out_data;
	logic	out_error;

	membrane_quant_stream_top #(.ID_WIDTH(ID_WIDTH)) dut (
		.clk(clk), .rst_n(rst_n),
		.in_valid(in_valid), .in_ready(in_ready), .in_mode(in_mode),
		.in_id(in_id), .in_data(in_data),
		.out_valid(out_valid), .out_ready(out_ready), .out_mode(out_mode),
		.out_id(out_id), .out_data(out_data), .out_error(out_error));

	initial clk = 0;
	always #5 clk = ~clk;

	// ---- golden vector storage ----
	localparam int NMAX = 20000;
	logic	[15:0]	x_flat		[0:NMAX * 32 - 1];
	logic	[7:0]	q8pack_flat	[0:NMAX * 34 - 1];
	logic	[15:0]	q8dequant_flat	[0:NMAX * 32 - 1];
	logic	[7:0]	q4pack_flat	[0:NMAX * 18 - 1];
	logic	[15:0]	q4unpack_flat	[0:NMAX * 32 - 1];

	initial begin
		$readmemh("/tmp/maxabs_x.txt", x_flat);
		$readmemh("/tmp/pack_expected.txt", q8pack_flat);
		$readmemh("/tmp/dequant_expected.txt", q8dequant_flat);
		$readmemh("/tmp/q4pack_expected.txt", q4pack_flat);
		$readmemh("/tmp/q4unpack_expected.txt", q4unpack_flat);
	end

	// ---- helpers to build in_data / expected out_data for a given
	// (mode, block index) pair ----
	function automatic logic [511:0] build_in_data(input logic [1:0] mode,
			input int blk);
		logic [511:0] d;
		int j;

		d = 512'h0;
		case (mode)
			2'b00, 2'b10: // Q8/Q4 encode: 32 F16 lanes
				for (j = 0; j < 32; j++)
					d[j * 16 +: 16] = x_flat[blk * 32 + j];
			2'b01: // Q8 decode: 34-byte packed block, low 272 bits
				for (j = 0; j < 34; j++)
					d[j * 8 +: 8] = q8pack_flat[blk * 34 + j];
			default: // Q4 decode: 18-byte packed block, low 144 bits
				for (j = 0; j < 18; j++)
					d[j * 8 +: 8] = q4pack_flat[blk * 18 + j];
		endcase
		return (d);
	endfunction

	function automatic logic check_out_data(input logic [1:0] mode,
			input int blk, input logic [511:0] got);
		int j;
		logic ok;
		logic [7:0] gb;

		ok = 1'b1;
		case (mode)
			2'b00: // Q8 encode: compare low 34 bytes
				for (j = 0; j < 34; j++) begin
					gb = got[j * 8 +: 8];
					if (gb !== q8pack_flat[blk * 34 + j])
						ok = 1'b0;
				end
			2'b01: // Q8 decode: compare 32 F16 lanes
				for (j = 0; j < 32; j++)
					if (got[j * 16 +: 16] !== q8dequant_flat[blk * 32 + j])
						ok = 1'b0;
			2'b10: // Q4 encode: compare low 18 bytes
				for (j = 0; j < 18; j++) begin
					gb = got[j * 8 +: 8];
					if (gb !== q4pack_flat[blk * 18 + j])
						ok = 1'b0;
				end
			default: // Q4 decode: compare 32 F16 lanes
				for (j = 0; j < 32; j++)
					if (got[j * 16 +: 16] !== q4unpack_flat[blk * 32 + j])
						ok = 1'b0;
		endcase
		return (ok);
	endfunction

	// ---- driver / checker state ----
	int	total_issued;
	int	total_checked;
	int	total_fails;

	logic	[1:0]	issue_mode_q	[0:4095];
	int		issue_blk_q	[0:4095];
	int		issue_id_q	[0:4095];
	int		issue_wr, issue_rd;

	logic	[1:0]	check_mode_q	[0:4095];
	int		check_blk_q	[0:4095];
	int		check_id_q	[0:4095];
	int		check_wr, check_rd;

	// input driver: for each queued issue, present in_valid/in_data/
	// in_mode/in_id until in_ready, with a pseudo-random stall before
	// each attempt.
	task automatic drive_input(input int count, input logic [1:0] fixed_mode,
			input bit mixed);
		int i;
		logic [1:0] m;
		int stall;

		i = 0;
		while (i < count) begin
			stall = $urandom_range(0, 2);
			repeat (stall) @(posedge clk);
			m = mixed ? $urandom_range(0, 3) : fixed_mode;
			in_mode = m;
			in_id = issue_wr[ID_WIDTH-1:0];
			in_data = build_in_data(m, i % NMAX);
			in_valid = 1'b1;
			@(posedge clk);
			while (!in_ready) @(posedge clk);
			issue_mode_q[issue_wr] = m;
			issue_blk_q[issue_wr] = i % NMAX;
			issue_id_q[issue_wr] = issue_wr;
			issue_wr = issue_wr + 1;
			total_issued = total_issued + 1;
			i = i + 1;
		end
		in_valid = 1'b0;
	endtask

	// output checker: randomized out_ready backpressure, verify each
	// retiring transaction's id matches issue order (FIFO ordering) and
	// its data matches the golden vector for the block/mode it was
	// issued with.
	task automatic check_output(input int count);
		int i;
		int stall;
		logic ok;

		i = 0;
		while (i < count) begin
			stall = $urandom_range(0, 2);
			repeat (stall) @(posedge clk);
			out_ready = 1'b1;
			@(posedge clk);
			while (!out_valid) @(posedge clk);
			if (out_id !== issue_id_q[check_rd][ID_WIDTH-1:0])
				begin
					$display("FAIL: id mismatch at retire %0d: expect=%0d got=%0d",
						i, issue_id_q[check_rd], out_id);
					total_fails = total_fails + 1;
				end
			if (out_mode !== issue_mode_q[check_rd])
				begin
					$display("FAIL: mode mismatch at retire %0d", i);
					total_fails = total_fails + 1;
				end
			ok = check_out_data(issue_mode_q[check_rd], issue_blk_q[check_rd], out_data);
			if (!ok) begin
				$display("FAIL: data mismatch at retire %0d (mode=%0d blk=%0d id=%0d)",
					i, issue_mode_q[check_rd], issue_blk_q[check_rd], issue_id_q[check_rd]);
				total_fails = total_fails + 1;
			end
			check_rd = check_rd + 1;
			total_checked = total_checked + 1;
			i = i + 1;
		end
		out_ready = 1'b0;
	endtask

	initial begin
		rst_n = 0;
		in_valid = 0;
		out_ready = 0;
		issue_wr = 0;
		issue_rd = 0;
		check_rd = 0;
		total_issued = 0;
		total_checked = 0;
		total_fails = 0;
		#20;
		@(posedge clk);
		rst_n = 1;
		@(posedge clk);

		// wait for vectors to load
		#1;

		// per-mode runs
		fork
			drive_input(N_PER_MODE, 2'b00, 1'b0);
			check_output(N_PER_MODE);
		join
		$display("progress: Q8 encode done (%0d checked, %0d fails so far)",
			total_checked, total_fails);

		fork
			drive_input(N_PER_MODE, 2'b01, 1'b0);
			check_output(N_PER_MODE);
		join
		$display("progress: Q8 decode done (%0d checked, %0d fails so far)",
			total_checked, total_fails);

		fork
			drive_input(N_PER_MODE, 2'b10, 1'b0);
			check_output(N_PER_MODE);
		join
		$display("progress: Q4 encode done (%0d checked, %0d fails so far)",
			total_checked, total_fails);

		fork
			drive_input(N_PER_MODE, 2'b11, 1'b0);
			check_output(N_PER_MODE);
		join
		$display("progress: Q4 decode done (%0d checked, %0d fails so far)",
			total_checked, total_fails);

		// mixed-mode interleaved run: exercises the shared-latency
		// ordering guarantee across mode boundaries directly.
		fork
			drive_input(N_MIX, 2'b00, 1'b1);
			check_output(N_MIX);
		join
		$display("progress: mixed-mode interleave done (%0d checked, %0d fails so far)",
			total_checked, total_fails);

		if (total_fails == 0)
			$display("PASS: membrane_quant_stream_top smoke test, %0d transactions, 0 fails",
				total_checked);
		else
			$display("FAIL: membrane_quant_stream_top smoke test, %0d / %0d fails",
				total_fails, total_checked);
		$finish;
	end

	// heartbeat / timeout safety
	initial begin
		#2000000;
		$display("TIMEOUT: only %0d/%0d checked", total_checked,
			4 * N_PER_MODE + N_MIX);
		$finish;
	end
endmodule
