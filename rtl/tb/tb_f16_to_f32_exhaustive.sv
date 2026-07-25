module tb_f16_to_f32_exhaustive;
	import membrane_fp_pkg::*;

	logic [15:0]	h_vec  [0:65535];
	logic [31:0]	exp_vec[0:65535];
	logic [31:0]	got;
	int		i;
	int		fails;

	initial begin
		$readmemh("/tmp/f16_f32_vectors_split_h.txt", h_vec);
		$readmemh("/tmp/f16_f32_vectors_split_e.txt", exp_vec);
		fails = 0;
		for (i = 0; i < 65536; i++) begin
			got = f16_to_f32_bits(h_vec[i]);
			if (got !== exp_vec[i]) begin
				if (fails < 20)
					$display("FAIL h=%04h expect=%08h got=%08h",
						h_vec[i], exp_vec[i], got);
				fails++;
			end
		end
		if (fails == 0)
			$display("PASS: all 65536 F16->F32 patterns bit-exact");
		else
			$display("FAIL: %0d / 65536 mismatches", fails);
		$finish;
	end
endmodule
