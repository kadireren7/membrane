module tb_q8_maxabs_debug;
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[15:0]	x_in		[0:31];
	logic			valid_out;
	logic	[15:0]	amax_out;
	int				cyc;

	q8_maxabs_reduce dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .x_in(x_in),
		.valid_out(valid_out), .amax_f16_out(amax_out));

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		rst_n = 0;
		valid_in = 0;
		for (int j = 0; j < 32; j++)
			x_in[j] = 16'h0000;
		@(posedge clk);
		@(posedge clk);
		#2;
		rst_n = 1;
		#2;
		cyc = 0;
		repeat (12) begin
			valid_in = (cyc < 3);
			for (int j = 0; j < 32; j++)
				x_in[j] = 16'(cyc * 100 + j);
			@(posedge clk);
			#1;
			$display("cyc=%0d valid_in=%0d vshift=%b valid_out=%0d amax_out=%h",
				cyc, valid_in, dut.u_valid.shift, valid_out, amax_out);
			cyc++;
		end
		$finish;
	end
endmodule
