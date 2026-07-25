module tb_fp_functions_smoke;
	import membrane_fp_pkg::*;

	logic [31:0] r32;
	logic [15:0] r16;
	logic [63:0] r64;

	initial begin
		r32 = f16_to_f32_bits(16'h3C00); // 1.0
		$display("f16(1.0)->f32 bits=%h (expect 3f800000)", r32);
		r16 = f32_to_f16_bits(32'h3f800000);
		$display("f32(1.0)->f16 bits=%h (expect 3c00)", r16);
		r64 = f32_widen_to_f64(32'h3f800000);
		$display("f32(1.0)->f64 bits=%h (expect 3ff0000000000000)", r64);
		r32 = f64_narrow_to_f32_rne(64'h3ff0000000000000);
		$display("f64(1.0)->f32 bits=%h (expect 3f800000)", r32);
		$finish;
	end
endmodule
