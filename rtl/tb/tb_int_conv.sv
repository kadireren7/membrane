module tb_int_conv;
	import membrane_fp_pkg::*;

	localparam int N9 = 512;
	localparam int NR8 = 20013;
	localparam int NT = 20008;
	logic	[8:0]	i9_in	[0:N9 - 1];
	logic	[31:0]	i9_r	[0:N9 - 1];
	logic	[31:0]	r8_in	[0:NR8 - 1];
	logic	[7:0]	r8_r	[0:NR8 - 1];
	logic	[31:0]	t_in	[0:NT - 1];
	logic	[31:0]	t_r		[0:NT - 1];
	int				fails;
	int				i;
	logic	[31:0]	got32;
	logic	[7:0]	got8;

	initial begin
		$readmemh("/tmp/i9_in.txt", i9_in);
		$readmemh("/tmp/i9_r.txt", i9_r);
		$readmemh("/tmp/r8_in.txt", r8_in);
		$readmemh("/tmp/r8_r.txt", r8_r);
		$readmemh("/tmp/t_in.txt", t_in);
		$readmemh("/tmp/t_r.txt", t_r);
		fails = 0;

		for (i = 0; i < N9; i++) begin
			got32 = int9_to_f32_bits($signed(i9_in[i]));
			if (got32 !== i9_r[i]) begin
				if (fails < 20)
					$display("FAIL int9_to_f32 %0d: v=%0d expect=%08h got=%08h",
						i, $signed(i9_in[i]), i9_r[i], got32);
				fails++;
			end
		end
		$display("int9_to_f32_bits: %0d/%0d checked, %0d fails so far",
			N9, N9, fails);

		for (i = 0; i < NR8; i++) begin
			got8 = f32_round_sat_to_i8(r8_in[i]);
			if (got8 !== r8_r[i]) begin
				if (fails < 20)
					$display("FAIL f32_round_sat_to_i8 %0d: f=%08h expect=%02h got=%02h",
						i, r8_in[i], r8_r[i], got8);
				fails++;
			end
		end
		$display("f32_round_sat_to_i8: %0d checked, %0d fails so far",
			NR8, fails);

		for (i = 0; i < NT; i++) begin
			got32 = f32_trunc_to_i32(t_in[i]);
			if (got32 !== t_r[i]) begin
				if (fails < 20)
					$display("FAIL f32_trunc_to_i32 %0d: f=%08h expect=%08h got=%08h",
						i, t_in[i], t_r[i], got32);
				fails++;
			end
		end
		$display("f32_trunc_to_i32: %0d checked, %0d fails so far", NT,
			fails);

		if (fails == 0)
			$display("PASS: all int conversion functions bit-exact");
		else
			$display("FAIL: %0d total mismatches", fails);
		$finish;
	end
endmodule
