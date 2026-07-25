// Phase 5.2: Q8_0 dequantize -- unpack the 34-byte block (2-byte F16
// scale + 32 signed int8), multiply each lane by the scale, narrow back
// to F16. Bit-exact with q8_0_dequant_block_scalar
// (src/quant/quant_simd.c): `out[j] = (float)qs[j] * d;` followed by the
// caller's F32->F16 narrow (membrane_simd_q8_0_dequantize's driving
// loop).
//
// Unlike q8_quantize_pack, this direction needs no NaN/Inf special-casing
// in the multiply itself: `qs[j]` is always a finite int8 in [-128,127]
// and `d` is finite-or-Infinity (never NaN, by construction of
// q8_scale/the amax invariant documented in q8_maxabs_reduce.sv) or
// exactly 0.0 -- `finite * Infinity` for a nonzero qs[j] IS a genuine
// Infinity result (correctly propagated), and `0 * Infinity = NaN` for a
// zero qs[j] against an Infinity scale is a real, intentional case this
// module must reproduce exactly (matches the C reference and ggml, see
// docs/phase4-ggml-quant-parity.md) -- so ordinary IEEE-754 multiply
// semantics, with no extra sentinel logic, are correct here; only the
// F32->F16 narrow needs care (reusing f32_to_f16_bits, already verified
// bit-exact including its NaN-canonicalization branch).
//
// The multiply, like q8_scale's divide, is not synthesizable as written
// (widen-to-F64 + `real` multiply + narrow-with-RNE) -- see q8_scale.sv's
// header for the disclosure this implies.
module q8_dequantize #(
	parameter int DELAY = 2
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[271:0]	packed_in,
	output	logic			valid_out,
	output	logic	[511:0]	x_out
);
	import membrane_fp_pkg::*;

	logic	[15:0]	d_f16;
	logic	[31:0]	d_f32;
	logic	[63:0]	d_f64;

	assign d_f16 = packed_in[15:0];
	assign d_f32 = f16_to_f32_bits(d_f16);
	assign d_f64 = f32_widen_to_f64(d_f32);

	function automatic logic [15:0] dequant_lane(input logic [7:0] q8,
			input logic [63:0] d64);
		int				qi;
		real			q_r;
		real			d_r;
		real			prod_r;
		logic	[31:0]	prod_f32;

		qi = $signed(q8);
		q_r = real'(qi);
		d_r = $bitstoreal(d64);
		prod_r = q_r * d_r;
		prod_f32 = f64_narrow_to_f32_rne($realtobits(prod_r));
		return (f32_to_f16_bits(prod_f32));
	endfunction

	logic	[15:0]	x_comb	[0:31];

	always_comb begin
		x_comb[0] = dequant_lane(packed_in[23:16], d_f64);
		x_comb[1] = dequant_lane(packed_in[31:24], d_f64);
		x_comb[2] = dequant_lane(packed_in[39:32], d_f64);
		x_comb[3] = dequant_lane(packed_in[47:40], d_f64);
		x_comb[4] = dequant_lane(packed_in[55:48], d_f64);
		x_comb[5] = dequant_lane(packed_in[63:56], d_f64);
		x_comb[6] = dequant_lane(packed_in[71:64], d_f64);
		x_comb[7] = dequant_lane(packed_in[79:72], d_f64);
		x_comb[8] = dequant_lane(packed_in[87:80], d_f64);
		x_comb[9] = dequant_lane(packed_in[95:88], d_f64);
		x_comb[10] = dequant_lane(packed_in[103:96], d_f64);
		x_comb[11] = dequant_lane(packed_in[111:104], d_f64);
		x_comb[12] = dequant_lane(packed_in[119:112], d_f64);
		x_comb[13] = dequant_lane(packed_in[127:120], d_f64);
		x_comb[14] = dequant_lane(packed_in[135:128], d_f64);
		x_comb[15] = dequant_lane(packed_in[143:136], d_f64);
		x_comb[16] = dequant_lane(packed_in[151:144], d_f64);
		x_comb[17] = dequant_lane(packed_in[159:152], d_f64);
		x_comb[18] = dequant_lane(packed_in[167:160], d_f64);
		x_comb[19] = dequant_lane(packed_in[175:168], d_f64);
		x_comb[20] = dequant_lane(packed_in[183:176], d_f64);
		x_comb[21] = dequant_lane(packed_in[191:184], d_f64);
		x_comb[22] = dequant_lane(packed_in[199:192], d_f64);
		x_comb[23] = dequant_lane(packed_in[207:200], d_f64);
		x_comb[24] = dequant_lane(packed_in[215:208], d_f64);
		x_comb[25] = dequant_lane(packed_in[223:216], d_f64);
		x_comb[26] = dequant_lane(packed_in[231:224], d_f64);
		x_comb[27] = dequant_lane(packed_in[239:232], d_f64);
		x_comb[28] = dequant_lane(packed_in[247:240], d_f64);
		x_comb[29] = dequant_lane(packed_in[255:248], d_f64);
		x_comb[30] = dequant_lane(packed_in[263:256], d_f64);
		x_comb[31] = dequant_lane(packed_in[271:264], d_f64);
	end

	logic	[15:0]	x_pipe	[0:DELAY - 1][0:31];

	always_ff @(posedge clk) begin
		for (int j = 0; j < 32; j++)
			x_pipe[0][j] <= x_comb[j];
		for (int k = DELAY - 1; k > 0; k--)
			for (int j = 0; j < 32; j++)
				x_pipe[k][j] <= x_pipe[k - 1][j];
	end

	valid_delay_line #(.DEPTH(DELAY)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out));

	assign x_out[15:0] = x_pipe[DELAY - 1][0];
	assign x_out[31:16] = x_pipe[DELAY - 1][1];
	assign x_out[47:32] = x_pipe[DELAY - 1][2];
	assign x_out[63:48] = x_pipe[DELAY - 1][3];
	assign x_out[79:64] = x_pipe[DELAY - 1][4];
	assign x_out[95:80] = x_pipe[DELAY - 1][5];
	assign x_out[111:96] = x_pipe[DELAY - 1][6];
	assign x_out[127:112] = x_pipe[DELAY - 1][7];
	assign x_out[143:128] = x_pipe[DELAY - 1][8];
	assign x_out[159:144] = x_pipe[DELAY - 1][9];
	assign x_out[175:160] = x_pipe[DELAY - 1][10];
	assign x_out[191:176] = x_pipe[DELAY - 1][11];
	assign x_out[207:192] = x_pipe[DELAY - 1][12];
	assign x_out[223:208] = x_pipe[DELAY - 1][13];
	assign x_out[239:224] = x_pipe[DELAY - 1][14];
	assign x_out[255:240] = x_pipe[DELAY - 1][15];
	assign x_out[271:256] = x_pipe[DELAY - 1][16];
	assign x_out[287:272] = x_pipe[DELAY - 1][17];
	assign x_out[303:288] = x_pipe[DELAY - 1][18];
	assign x_out[319:304] = x_pipe[DELAY - 1][19];
	assign x_out[335:320] = x_pipe[DELAY - 1][20];
	assign x_out[351:336] = x_pipe[DELAY - 1][21];
	assign x_out[367:352] = x_pipe[DELAY - 1][22];
	assign x_out[383:368] = x_pipe[DELAY - 1][23];
	assign x_out[399:384] = x_pipe[DELAY - 1][24];
	assign x_out[415:400] = x_pipe[DELAY - 1][25];
	assign x_out[431:416] = x_pipe[DELAY - 1][26];
	assign x_out[447:432] = x_pipe[DELAY - 1][27];
	assign x_out[463:448] = x_pipe[DELAY - 1][28];
	assign x_out[479:464] = x_pipe[DELAY - 1][29];
	assign x_out[495:480] = x_pipe[DELAY - 1][30];
	assign x_out[511:496] = x_pipe[DELAY - 1][31];
endmodule
