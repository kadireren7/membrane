// Phase 5.2: Q4_0 scale computation, factored into its own module for the
// same reason as q4_scan.sv -- see its header comment. d = mx/-8.0,
// id = d!=0 ? 1/d : 0, bit-exact with q4_0_quant_block_scalar
// (src/quant/quant_simd.c). Not synthesizable as written (real-arithmetic
// divide via widen/$bitstoreal/narrow), same disclosure as q8_scale.sv.
module q4_scale (
	input	logic	[31:0]	mx_f32,
	output	logic	[15:0]	d_f16_out,
	output	logic	[31:0]	id_f32_out
);
	import membrane_fp_pkg::*;

	logic	[63:0]	mx_f64;
	logic	[63:0]	neg8_f64;
	real			mx_r;
	real			d_r;
	real			id_r;
	logic	[31:0]	d_f32;
	logic	[31:0]	id_f32;

	assign mx_f64 = f32_widen_to_f64(mx_f32);
	assign neg8_f64 = f32_widen_to_f64(32'hC1000000); // -8.0

	always_comb begin
		mx_r = $bitstoreal(mx_f64);
		d_r = mx_r / $bitstoreal(neg8_f64);
		d_f32 = f64_narrow_to_f32_rne($realtobits(d_r));
		id_r = (d_f32 != 32'h0)
			? (1.0 / $bitstoreal(f32_widen_to_f64(d_f32))) : 0.0;
		id_f32 = (d_f32 != 32'h0)
			? f64_narrow_to_f32_rne($realtobits(id_r)) : 32'h0;
	end

	assign d_f16_out = f32_to_f16_bits(d_f32);
	assign id_f32_out = id_f32;
endmodule
