module tb_f32_to_f16_vectors;
	import membrane_fp_pkg::*;

	localparam int N = 263538;
	logic [31:0]	f_vec[0:N-1];
	logic [15:0]	exp_vec[0:N-1];
	logic [15:0]	got;
	int		i;
	int		fails;

	initial begin
		$readmemh("/tmp/f32_f16_split_f.txt", f_vec);
		$readmemh("/tmp/f32_f16_split_h.txt", exp_vec);
		fails = 0;
		for (i = 0; i < N; i++) begin
			got = f32_to_f16_bits(f_vec[i]);
			if (got !== exp_vec[i]) begin
				if (fails < 20)
					$display("FAIL f=%08h expect=%04h got=%04h",
						f_vec[i], exp_vec[i], got);
				fails++;
			end
		end
		if (fails == 0)
			$display("PASS: all %0d F32->F16 vectors bit-exact", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
