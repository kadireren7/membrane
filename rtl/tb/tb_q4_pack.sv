module tb_q4_pack;
	localparam int N = 20000;
	logic	[15:0]	x_flat		[0:N * 32 - 1];
	logic	[15:0]	d_vec		[0:N - 1];
	logic	[31:0]	id_vec		[0:N - 1];
	logic	[7:0]	pack_exp	[0:N * 18 - 1];
	logic			clk;
	logic			rst_n;
	logic			valid_in;
	logic	[15:0]	x_in		[0:31];
	logic	[511:0]	x_in_flat;
	logic	[15:0]	d_f16;
	logic	[31:0]	id_f32;
	logic			valid_out;
	logic	[143:0]	packed_out;
	int				in_idx;
	int				out_idx;
	int				fails;

	q4_pack #(.MUL_DELAY(1)) dut (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .x_in_flat(x_in_flat),
		.d_f16(d_f16), .id_f32(id_f32), .valid_out(valid_out),
		.packed_out(packed_out));

	always_comb
		for (int j = 0; j < 32; j++)
			x_in_flat[j * 16 +: 16] = x_in[j];

	initial clk = 0;
	always #5 clk = ~clk;

	initial begin
		$readmemh("/tmp/maxabs_x.txt", x_flat);
		$readmemh("/tmp/q4scale_d.txt", d_vec);
		$readmemh("/tmp/q4scale_id.txt", id_vec);
		$readmemh("/tmp/q4pack_expected.txt", pack_exp);
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
				for (int j = 0; j < 32; j++)
					x_in[j] = x_flat[in_idx * 32 + j];
				d_f16 = d_vec[in_idx];
				id_f32 = id_vec[in_idx];
			end else
				valid_in = 0;
			@(posedge clk);
			#1;
			if (valid_out) begin
				logic	mismatch;
				logic	[7:0]	got_byte;

				mismatch = 1'b0;
				for (int j = 0; j < 18; j++) begin
					got_byte = packed_out[j * 8 +: 8];
					if (got_byte !== pack_exp[out_idx * 18 + j])
						mismatch = 1'b1;
				end
				if (mismatch) begin
					if (fails < 10) begin
						$display("FAIL block %0d:", out_idx);
						$write("  expect:");
						for (int j = 0; j < 18; j++)
							$write(" %02h", pack_exp[out_idx * 18 + j]);
						$write("\n  got:   ");
						for (int j = 0; j < 18; j++)
							$write(" %02h", packed_out[j * 8 +: 8]);
						$write("\n");
					end
					fails++;
				end
				out_idx++;
			end
			if (in_idx < N)
				in_idx++;
		end
		if (fails == 0)
			$display("PASS: q4_pack bit-exact on %0d blocks", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
