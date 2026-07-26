// Phase 5.3: Q8_0 dequantize -- unpack the 34-byte block (2-byte F16
// scale + 32 signed int8), multiply each lane by the scale, narrow back
// to F16. Bit-exact with q8_0_dequant_block_scalar
// (src/quant/quant_simd.c): `out[j] = (float)qs[j] * d;` followed by the
// caller's F32->F16 narrow (membrane_simd_q8_0_dequantize's driving
// loop).
//
// Fully synthesizable: qs[j] (int8) -> F32 is membrane_fp_pkg.sv's
// membrane_fp_pkg::int9_to_f32_bits() (exact, sign-extending int8 into the 9-bit input
// that function expects), the multiply is a membrane_fp_multiplier.sv
// instance per lane (32 total, same area-vs-latency tradeoff disclosure
// as q8_quantize_pack.sv), and the multiplier's own NaN-operand and
// Inf*0 handling (see that module's header) already reproduces this
// direction's real, reachable edge case exactly: `0 * Infinity = NaN`
// when a lane's quantized value is 0 and the block's scale is Infinity
// (amax was Infinity) -- verified empirically against this machine's
// native float32 multiply, not assumed from the IEEE-754 standard text.
module q8_dequantize #(
	parameter int MUL_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[271:0]	packed_in,
	output	logic			valid_out,
	output	logic	[511:0]	x_out
);

	logic	[15:0]	d_f16;
	logic	[31:0]	d_f32;
	logic	[7:0]	qs		[0:31];
	logic	[31:0]	qs_f32	[0:31];
	logic	[31:0]	prod_f32	[0:31];
	logic			mul_valid	[0:31];

	assign d_f16 = packed_in[15:0];
	assign d_f32 = membrane_fp_pkg::f16_to_f32_bits(d_f16);

	assign qs[0] = packed_in[23:16];
	assign qs[1] = packed_in[31:24];
	assign qs[2] = packed_in[39:32];
	assign qs[3] = packed_in[47:40];
	assign qs[4] = packed_in[55:48];
	assign qs[5] = packed_in[63:56];
	assign qs[6] = packed_in[71:64];
	assign qs[7] = packed_in[79:72];
	assign qs[8] = packed_in[87:80];
	assign qs[9] = packed_in[95:88];
	assign qs[10] = packed_in[103:96];
	assign qs[11] = packed_in[111:104];
	assign qs[12] = packed_in[119:112];
	assign qs[13] = packed_in[127:120];
	assign qs[14] = packed_in[135:128];
	assign qs[15] = packed_in[143:136];
	assign qs[16] = packed_in[151:144];
	assign qs[17] = packed_in[159:152];
	assign qs[18] = packed_in[167:160];
	assign qs[19] = packed_in[175:168];
	assign qs[20] = packed_in[183:176];
	assign qs[21] = packed_in[191:184];
	assign qs[22] = packed_in[199:192];
	assign qs[23] = packed_in[207:200];
	assign qs[24] = packed_in[215:208];
	assign qs[25] = packed_in[223:216];
	assign qs[26] = packed_in[231:224];
	assign qs[27] = packed_in[239:232];
	assign qs[28] = packed_in[247:240];
	assign qs[29] = packed_in[255:248];
	assign qs[30] = packed_in[263:256];
	assign qs[31] = packed_in[271:264];

	assign qs_f32[0] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[0][7], qs[0]}));
	assign qs_f32[1] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[1][7], qs[1]}));
	assign qs_f32[2] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[2][7], qs[2]}));
	assign qs_f32[3] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[3][7], qs[3]}));
	assign qs_f32[4] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[4][7], qs[4]}));
	assign qs_f32[5] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[5][7], qs[5]}));
	assign qs_f32[6] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[6][7], qs[6]}));
	assign qs_f32[7] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[7][7], qs[7]}));
	assign qs_f32[8] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[8][7], qs[8]}));
	assign qs_f32[9] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[9][7], qs[9]}));
	assign qs_f32[10] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[10][7], qs[10]}));
	assign qs_f32[11] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[11][7], qs[11]}));
	assign qs_f32[12] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[12][7], qs[12]}));
	assign qs_f32[13] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[13][7], qs[13]}));
	assign qs_f32[14] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[14][7], qs[14]}));
	assign qs_f32[15] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[15][7], qs[15]}));
	assign qs_f32[16] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[16][7], qs[16]}));
	assign qs_f32[17] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[17][7], qs[17]}));
	assign qs_f32[18] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[18][7], qs[18]}));
	assign qs_f32[19] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[19][7], qs[19]}));
	assign qs_f32[20] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[20][7], qs[20]}));
	assign qs_f32[21] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[21][7], qs[21]}));
	assign qs_f32[22] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[22][7], qs[22]}));
	assign qs_f32[23] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[23][7], qs[23]}));
	assign qs_f32[24] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[24][7], qs[24]}));
	assign qs_f32[25] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[25][7], qs[25]}));
	assign qs_f32[26] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[26][7], qs[26]}));
	assign qs_f32[27] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[27][7], qs[27]}));
	assign qs_f32[28] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[28][7], qs[28]}));
	assign qs_f32[29] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[29][7], qs[29]}));
	assign qs_f32[30] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[30][7], qs[30]}));
	assign qs_f32[31] = membrane_fp_pkg::int9_to_f32_bits($signed({qs[31][7], qs[31]}));

	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul0 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[0]), .b_in(d_f32), .valid_out(mul_valid[0]), .result_out(prod_f32[0]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[1]), .b_in(d_f32), .valid_out(mul_valid[1]), .result_out(prod_f32[1]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[2]), .b_in(d_f32), .valid_out(mul_valid[2]), .result_out(prod_f32[2]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul3 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[3]), .b_in(d_f32), .valid_out(mul_valid[3]), .result_out(prod_f32[3]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul4 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[4]), .b_in(d_f32), .valid_out(mul_valid[4]), .result_out(prod_f32[4]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul5 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[5]), .b_in(d_f32), .valid_out(mul_valid[5]), .result_out(prod_f32[5]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul6 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[6]), .b_in(d_f32), .valid_out(mul_valid[6]), .result_out(prod_f32[6]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul7 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[7]), .b_in(d_f32), .valid_out(mul_valid[7]), .result_out(prod_f32[7]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul8 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[8]), .b_in(d_f32), .valid_out(mul_valid[8]), .result_out(prod_f32[8]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul9 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[9]), .b_in(d_f32), .valid_out(mul_valid[9]), .result_out(prod_f32[9]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul10 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[10]), .b_in(d_f32), .valid_out(mul_valid[10]), .result_out(prod_f32[10]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul11 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[11]), .b_in(d_f32), .valid_out(mul_valid[11]), .result_out(prod_f32[11]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul12 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[12]), .b_in(d_f32), .valid_out(mul_valid[12]), .result_out(prod_f32[12]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul13 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[13]), .b_in(d_f32), .valid_out(mul_valid[13]), .result_out(prod_f32[13]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul14 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[14]), .b_in(d_f32), .valid_out(mul_valid[14]), .result_out(prod_f32[14]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul15 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[15]), .b_in(d_f32), .valid_out(mul_valid[15]), .result_out(prod_f32[15]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul16 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[16]), .b_in(d_f32), .valid_out(mul_valid[16]), .result_out(prod_f32[16]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul17 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[17]), .b_in(d_f32), .valid_out(mul_valid[17]), .result_out(prod_f32[17]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul18 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[18]), .b_in(d_f32), .valid_out(mul_valid[18]), .result_out(prod_f32[18]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul19 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[19]), .b_in(d_f32), .valid_out(mul_valid[19]), .result_out(prod_f32[19]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul20 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[20]), .b_in(d_f32), .valid_out(mul_valid[20]), .result_out(prod_f32[20]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul21 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[21]), .b_in(d_f32), .valid_out(mul_valid[21]), .result_out(prod_f32[21]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul22 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[22]), .b_in(d_f32), .valid_out(mul_valid[22]), .result_out(prod_f32[22]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul23 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[23]), .b_in(d_f32), .valid_out(mul_valid[23]), .result_out(prod_f32[23]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul24 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[24]), .b_in(d_f32), .valid_out(mul_valid[24]), .result_out(prod_f32[24]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul25 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[25]), .b_in(d_f32), .valid_out(mul_valid[25]), .result_out(prod_f32[25]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul26 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[26]), .b_in(d_f32), .valid_out(mul_valid[26]), .result_out(prod_f32[26]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul27 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[27]), .b_in(d_f32), .valid_out(mul_valid[27]), .result_out(prod_f32[27]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul28 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[28]), .b_in(d_f32), .valid_out(mul_valid[28]), .result_out(prod_f32[28]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul29 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[29]), .b_in(d_f32), .valid_out(mul_valid[29]), .result_out(prod_f32[29]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul30 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[30]), .b_in(d_f32), .valid_out(mul_valid[30]), .result_out(prod_f32[30]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul31 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(qs_f32[31]), .b_in(d_f32), .valid_out(mul_valid[31]), .result_out(prod_f32[31]));

	assign valid_out = mul_valid[0];

	assign x_out[15:0] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[0]);
	assign x_out[31:16] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[1]);
	assign x_out[47:32] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[2]);
	assign x_out[63:48] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[3]);
	assign x_out[79:64] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[4]);
	assign x_out[95:80] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[5]);
	assign x_out[111:96] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[6]);
	assign x_out[127:112] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[7]);
	assign x_out[143:128] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[8]);
	assign x_out[159:144] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[9]);
	assign x_out[175:160] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[10]);
	assign x_out[191:176] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[11]);
	assign x_out[207:192] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[12]);
	assign x_out[223:208] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[13]);
	assign x_out[239:224] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[14]);
	assign x_out[255:240] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[15]);
	assign x_out[271:256] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[16]);
	assign x_out[287:272] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[17]);
	assign x_out[303:288] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[18]);
	assign x_out[319:304] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[19]);
	assign x_out[335:320] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[20]);
	assign x_out[351:336] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[21]);
	assign x_out[367:352] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[22]);
	assign x_out[383:368] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[23]);
	assign x_out[399:384] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[24]);
	assign x_out[415:400] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[25]);
	assign x_out[431:416] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[26]);
	assign x_out[447:432] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[27]);
	assign x_out[463:448] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[28]);
	assign x_out[479:464] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[29]);
	assign x_out[495:480] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[30]);
	assign x_out[511:496] = membrane_fp_pkg::f32_to_f16_bits(prod_f32[31]);
endmodule
