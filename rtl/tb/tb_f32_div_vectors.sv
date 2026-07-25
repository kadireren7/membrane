// Verifies the "widen to F64, divide in `real`, narrow with round-to-
// nearest-even" construction q8_scale.sv relies on actually reproduces
// correctly-rounded IEEE-754 float32 division, by comparing against the
// C compiler/CPU's own float32 division (the ground truth) for a large
// random sample plus the specific d=amax/127, id=127/amax operations this
// datapath performs.
module tb_f32_div_vectors;
	import membrane_fp_pkg::*;

	localparam int N = 100000;
	logic [31:0]	amax_vec[0:N-1];
	logic [31:0]	d_vec[0:N-1];
	logic [31:0]	id_vec[0:N-1];
	int		i;
	int		fails;
	logic [63:0]	amax64, c127_64, r127_64;
	real		amax_r, d_r, id_r;
	logic [31:0]	d_got, id_got;

	initial begin
		$readmemh("/tmp/div_amax.txt", amax_vec);
		$readmemh("/tmp/div_d.txt", d_vec);
		$readmemh("/tmp/div_id.txt", id_vec);
		c127_64 = f32_widen_to_f64(32'h42FE0000); // 127.0
		fails = 0;
		for (i = 0; i < N; i++) begin
			amax64 = f32_widen_to_f64(amax_vec[i]);
			amax_r = $bitstoreal(amax64);
			d_r = amax_r / $bitstoreal(c127_64);
			id_r = (amax_r != 0.0) ? $bitstoreal(c127_64) / amax_r : 0.0;
			d_got = f64_narrow_to_f32_rne($realtobits(d_r));
			id_got = f64_narrow_to_f32_rne($realtobits(id_r));
			if (d_got !== d_vec[i] || id_got !== id_vec[i]) begin
				if (fails < 20)
					$display("FAIL amax=%08h d_expect=%08h d_got=%08h id_expect=%08h id_got=%08h",
						amax_vec[i], d_vec[i], d_got, id_vec[i], id_got);
				fails++;
			end
		end
		if (fails == 0)
			$display("PASS: all %0d F32 divide vectors bit-exact", N);
		else
			$display("FAIL: %0d / %0d mismatches", fails, N);
		$finish;
	end
endmodule
