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
	int				in_idx;
	int				out_idx;
	int				fails;

	q4_scale #(.DIV_DELAY(1)) dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .mx_f32(mx_in),
		.valid_out(valid_out), .d_f16_out(d_out), .id_f32_out(id_out));

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		$readmemh("/tmp/q4mx.txt", mx_vec);
		$readmemh("/tmp/q4scale_d.txt", d_exp);
		$readmemh("/tmp/q4scale_id.txt", id_exp);
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
				mx_in = mx_vec[in_idx];
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
			end
			if (in_idx < N)
				in_idx++;
		end
		if (fails == 0)
			$display("PASS: q4_scale bit-exact on %0d blocks", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
