// Since EXP-FPGA-DIV-001 (rtl/membrane_fp_divider_radix4.sv promoted into
// q4_scale.sv's `u_div_id`), q4_scale is a variable-latency, single-
// in-flight module, and the caller must not reassert `valid_in` while a
// transaction is outstanding (q4_scale.sv's own `` `ifndef SYNTHESIS ``
// assertion enforces this). The driver below tracks that with its own
// local `outstanding` flag, set the cycle `valid_in` is issued and
// cleared when `valid_out` finally fires -- NOT gated on the module's own
// `busy` output directly: `busy` is u_div_id's busy bit, which only goes
// high 2 cycles after `valid_in` (1 cycle for u_div_d's own DELAY=1 pipe
// to produce d_valid, 1 more for u_div_id to register accepting it), so a
// driver gating on `!busy` alone can issue a second transaction into that
// 2-cycle window and trip the single in-flight assertion -- caught by
// this test's own first draft failing exactly that assertion, not by
// reasoning about it in advance. `membrane_quant_stream_top.sv`'s own
// `q4enc_inflight` uses the same "set on issue, not on a lagging busy
// pin" discipline for the same reason. See
// experiments/EXP-FPGA-DIV-001/promotion-audit.md.
module tb_q4_scale;
	localparam int N = 20000;
	logic	[31:0]	mx_vec	[0:N - 1];
	logic	[15:0]	d_exp	[0:N - 1];
	logic	[31:0]	id_exp	[0:N - 1];
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[31:0]	mx_in;
	logic			valid_out;
	logic	[15:0]	d_out;
	logic	[31:0]	id_out;
	logic			busy;
	logic			outstanding;
	int				in_idx;
	int				out_idx;
	int				fails;

	q4_scale #(.DIV_DELAY(1)) dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .mx_f32(mx_in),
		.valid_out(valid_out), .d_f16_out(d_out), .id_f32_out(id_out),
		.busy(busy));

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		$readmemh("/tmp/q4mx.txt", mx_vec);
		$readmemh("/tmp/q4scale_d.txt", d_exp);
		$readmemh("/tmp/q4scale_id.txt", id_exp);
		rst_n = 0;
		valid_in = 0;
		outstanding = 0;
		in_idx = 0;
		out_idx = 0;
		fails = 0;
		@(posedge clk);
		@(posedge clk);
		#2;
		rst_n = 1;
		#2;
		while (out_idx < N) begin
			if (in_idx < N && !outstanding) begin
				valid_in = 1;
				mx_in = mx_vec[in_idx];
				in_idx++;
				outstanding = 1;
			end else
				valid_in = 0;
			@(posedge clk);
			#1;
			if (valid_out) begin
				if (d_out !== d_exp[out_idx] || id_out !== id_exp[out_idx]) begin
					if (fails < 20)
						$display("FAIL block %0d: mx=%08h d_expect=%04h d_got=%04h id_expect=%08h id_got=%08h",
							out_idx, mx_vec[out_idx], d_exp[out_idx], d_out,
							id_exp[out_idx], id_out);
					fails++;
				end
				out_idx++;
				outstanding = 0;
			end
		end
		if (fails == 0)
			$display("PASS: q4_scale bit-exact on %0d blocks", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end

	// Heartbeat/timeout: single in-flight discipline means this run now
	// takes far longer in simulated time than the old back-to-back
	// driver did (each block's own divider latency instead of a fixed
	// 2-cycle pipe) -- a generous bound, not a tight one.
	initial begin
		#200000000;
		$display("TIMEOUT: only %0d/%0d blocks checked", out_idx, N);
		$finish;
	end
endmodule
