module tb_fp_adder;
	parameter int N = 80008;
	logic	[31:0]	a_vec	[0:N - 1];
	logic	[31:0]	b_vec	[0:N - 1];
	logic	[0:0]	s_vec	[0:N - 1];
	logic	[31:0]	r_vec	[0:N - 1];
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[31:0]	a_in;
	logic	[31:0]	b_in;
	logic			subtract_in;
	logic			valid_out;
	logic	[31:0]	result_out;
	int				in_idx;
	int				out_idx;
	int				fails;
	string			a_file, b_file, s_file, r_file;

	membrane_fp_adder #(.DELAY(1)) dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(a_in),
		.b_in(b_in), .subtract_in(subtract_in), .valid_out(valid_out),
		.result_out(result_out));

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		if (!$value$plusargs("afile=%s", a_file))
			a_file = "/tmp/fpadd_a_s.txt";
		if (!$value$plusargs("bfile=%s", b_file))
			b_file = "/tmp/fpadd_b_s.txt";
		if (!$value$plusargs("sfile=%s", s_file))
			s_file = "/tmp/fpadd_s_s.txt";
		if (!$value$plusargs("rfile=%s", r_file))
			r_file = "/tmp/fpadd_r_s.txt";
		$readmemh(a_file, a_vec);
		$readmemh(b_file, b_vec);
		$readmemb(s_file, s_vec);
		$readmemh(r_file, r_vec);
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
				a_in = a_vec[in_idx];
				b_in = b_vec[in_idx];
				subtract_in = s_vec[in_idx][0];
			end else
				valid_in = 0;
			@(posedge clk);
			#1;
			if (valid_out) begin
				if (result_out !== r_vec[out_idx]) begin
					if (fails < 20)
						$display("FAIL %0d: a=%08h b=%08h sub=%0d expect=%08h got=%08h",
							out_idx, a_vec[out_idx], b_vec[out_idx],
							s_vec[out_idx][0], r_vec[out_idx], result_out);
					fails++;
				end
				out_idx++;
			end
			if (in_idx < N)
				in_idx++;
		end
		if (fails == 0)
			$display("PASS: membrane_fp_adder bit-exact on %0d vectors", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
