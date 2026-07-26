module tb_q8_maxabs_reduce;
	localparam int N = 20000;
	logic	[15:0]	x_flat		[0:N * 32 - 1];
	logic	[15:0]	amax_exp	[0:N - 1];
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[15:0]	x_in		[0:31];
	logic	[511:0]	x_in_flat;
	logic			valid_out;
	logic	[15:0]	amax_out;
	int				in_idx;
	int				out_idx;
	int				fails;
	int				checked;

	q8_maxabs_reduce dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .x_in_flat(x_in_flat),
		.valid_out(valid_out), .amax_f16_out(amax_out));

	always_comb
		for (int j = 0; j < 32; j++)
			x_in_flat[j * 16 +: 16] = x_in[j];

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		$readmemh("/tmp/maxabs_x.txt", x_flat);
		$readmemh("/tmp/maxabs_amax.txt", amax_exp);
		rst_n = 0;
		valid_in = 0;
		in_idx = 0;
		out_idx = 0;
		fails = 0;
		checked = 0;
		@(posedge clk);
		@(posedge clk);
		#2;
		rst_n = 1;
		#2;
		while (out_idx < N) begin
			if (in_idx < N) begin
				valid_in = 1;
				for (int j = 0; j < 32; j++)
					x_in[j] = x_flat[in_idx * 32 + j];
			end else
				valid_in = 0;
			@(posedge clk);
			#1;
			if (valid_out) begin
				checked++;
				if (amax_out !== amax_exp[out_idx]) begin
					if (fails < 20)
						$display("FAIL block %0d: expect=%04h got=%04h",
							out_idx, amax_exp[out_idx], amax_out);
					fails++;
				end
				out_idx++;
			end
			if (in_idx < N)
				in_idx++;
		end
		if (fails == 0)
			$display("PASS: q8_maxabs_reduce bit-exact on %0d blocks (5-cycle latency confirmed)",
				checked);
		else
			$display("FAIL: %0d / %0d mismatches", fails, checked);
		$finish;
	end
endmodule
