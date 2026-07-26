module tb_q4_unpack;
	localparam int N = 20000;
	logic	[7:0]	pack_flat	[0:N * 18 - 1];
	logic	[15:0]	exp_flat	[0:N * 32 - 1];
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[143:0]	packed_in;
	logic			valid_out;
	logic	[511:0]	x_out;
	int				in_idx;
	int				out_idx;
	int				fails;

	q4_unpack #(.MUL_DELAY(1)) dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.packed_in(packed_in), .valid_out(valid_out), .x_out(x_out));

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		$readmemh("/tmp/q4pack_expected.txt", pack_flat);
		$readmemh("/tmp/q4unpack_expected.txt", exp_flat);
		rst_n = 0;
		valid_in = 0;
		in_idx = 0;
		out_idx = 0;
		fails = 0;
		@(posedge clk);
		@(posedge clk);
		#2;
		rst_n = 1;
		#2;
		while (out_idx < N) begin
			if (in_idx < N) begin
				valid_in = 1;
				for (int j = 0; j < 18; j++)
					packed_in[j * 8 +: 8] = pack_flat[in_idx * 18 + j];
			end else
				valid_in = 0;
			@(posedge clk);
			#1;
			if (valid_out) begin
				logic	mismatch;
				logic	[15:0]	got_w;

				mismatch = 1'b0;
				for (int j = 0; j < 32; j++) begin
					got_w = x_out[j * 16 +: 16];
					if (got_w !== exp_flat[out_idx * 32 + j])
						mismatch = 1'b1;
				end
				if (mismatch) begin
					if (fails < 10)
						$display("FAIL block %0d", out_idx);
					fails++;
				end
				out_idx++;
			end
			if (in_idx < N)
				in_idx++;
		end
		if (fails == 0)
			$display("PASS: q4_unpack bit-exact on %0d blocks", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
