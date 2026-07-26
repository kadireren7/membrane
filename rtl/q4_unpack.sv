// Phase 5.3: Q4_0 dequantize -- unpack the 18-byte block (2-byte F16
// scale + 16 bytes of packed 4-bit pairs), reconstruct each lane, narrow
// to F16. Bit-exact with q4_0_dequant_block_scalar (src/quant/quant_simd.c):
//   lo = qs[j] & 0xF; hi = qs[j] >> 4;
//   out[j]    = ((float)lo - 8.0f) * d;
//   out[j+16] = ((float)hi - 8.0f) * d;
//
// Fully synthesizable: each nibble (0..15, unsigned) minus 8 is an EXACT
// integer in [-8,7] -- no rounding is ever needed for that subtraction,
// so it is folded directly into membrane_fp_pkg::int9_to_f32_bits() (already verified
// exhaustively, see membrane_fp_pkg.sv), mirroring q8_dequantize.sv's
// qs_f32 conversion. The multiply-by-scale step runs through one
// membrane_fp_multiplier.sv instance per lane (32 total, same area-vs-
// latency tradeoff disclosure as q8_dequantize.sv/q8_quantize_pack.sv).
module q4_unpack #(
	parameter int MUL_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[143:0]	packed_in,
	output	logic			valid_out,
	output	logic	[511:0]	x_out
);

	logic	[15:0]	d_f16;
	logic	[31:0]	d_f32;

	assign d_f16 = packed_in[15:0];
	assign d_f32 = membrane_fp_pkg::f16_to_f32_bits(d_f16);

	logic	[7:0]	qb		[0:15];

	assign qb[0] = packed_in[23:16];
	assign qb[1] = packed_in[31:24];
	assign qb[2] = packed_in[39:32];
	assign qb[3] = packed_in[47:40];
	assign qb[4] = packed_in[55:48];
	assign qb[5] = packed_in[63:56];
	assign qb[6] = packed_in[71:64];
	assign qb[7] = packed_in[79:72];
	assign qb[8] = packed_in[87:80];
	assign qb[9] = packed_in[95:88];
	assign qb[10] = packed_in[103:96];
	assign qb[11] = packed_in[111:104];
	assign qb[12] = packed_in[119:112];
	assign qb[13] = packed_in[127:120];
	assign qb[14] = packed_in[135:128];
	assign qb[15] = packed_in[143:136];

	// nib-8: an EXACT signed integer in [-8,7] (no rounding ever needed).
	// Avoids SystemVerilog `'(...)` cast-expression syntax throughout
	// (yosys's read_verilog -sv rejects that form, see
	// membrane_fp_pkg.sv's header for the same finding on `int'(...)`)
	// by declaring `ext` as a plain `logic signed [8:0]` and assigning
	// the unsigned zero-extended nibble into it via ordinary assignment
	// (which reinterprets, not converts, the bits), then subtracting a
	// signed constant -- the same "assign into an already-signed
	// variable, then do signed arithmetic on it" pattern used for
	// `exp_f_signed` in f32_to_f16_bits.
	// Assigns the function's own name instead of using `return` --
	// yosys 0.33 rejects `return` inside module-scoped functions, see
	// membrane_fp_adder.sv's lzc_9 header comment for the full finding.
	function automatic logic signed [8:0] nib_v9(input logic [3:0] nib);
		logic	signed	[8:0]	ext;

		ext = {5'b0, nib};
		nib_v9 = ext - 9'sd8;
	endfunction

	// nib_f32[j] holds lane j (j=0..15, the low nibbles) at index j and
	// lane j+16 (the high nibbles) at index j+16 -- same lane numbering
	// as the retired real-arithmetic version's x_comb array.
	logic	[31:0]	nib_f32		[0:31];

	assign nib_f32[0] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[0][3:0]));
	assign nib_f32[16] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[0][7:4]));
	assign nib_f32[1] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[1][3:0]));
	assign nib_f32[17] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[1][7:4]));
	assign nib_f32[2] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[2][3:0]));
	assign nib_f32[18] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[2][7:4]));
	assign nib_f32[3] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[3][3:0]));
	assign nib_f32[19] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[3][7:4]));
	assign nib_f32[4] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[4][3:0]));
	assign nib_f32[20] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[4][7:4]));
	assign nib_f32[5] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[5][3:0]));
	assign nib_f32[21] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[5][7:4]));
	assign nib_f32[6] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[6][3:0]));
	assign nib_f32[22] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[6][7:4]));
	assign nib_f32[7] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[7][3:0]));
	assign nib_f32[23] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[7][7:4]));
	assign nib_f32[8] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[8][3:0]));
	assign nib_f32[24] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[8][7:4]));
	assign nib_f32[9] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[9][3:0]));
	assign nib_f32[25] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[9][7:4]));
	assign nib_f32[10] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[10][3:0]));
	assign nib_f32[26] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[10][7:4]));
	assign nib_f32[11] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[11][3:0]));
	assign nib_f32[27] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[11][7:4]));
	assign nib_f32[12] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[12][3:0]));
	assign nib_f32[28] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[12][7:4]));
	assign nib_f32[13] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[13][3:0]));
	assign nib_f32[29] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[13][7:4]));
	assign nib_f32[14] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[14][3:0]));
	assign nib_f32[30] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[14][7:4]));
	assign nib_f32[15] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[15][3:0]));
	assign nib_f32[31] = membrane_fp_pkg::int9_to_f32_bits(nib_v9(qb[15][7:4]));

	logic	[31:0]	prod_f32	[0:31];
	logic			mul_valid	[0:31];

	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul0 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[0]), .b_in(d_f32), .valid_out(mul_valid[0]), .result_out(prod_f32[0]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[1]), .b_in(d_f32), .valid_out(mul_valid[1]), .result_out(prod_f32[1]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[2]), .b_in(d_f32), .valid_out(mul_valid[2]), .result_out(prod_f32[2]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul3 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[3]), .b_in(d_f32), .valid_out(mul_valid[3]), .result_out(prod_f32[3]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul4 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[4]), .b_in(d_f32), .valid_out(mul_valid[4]), .result_out(prod_f32[4]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul5 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[5]), .b_in(d_f32), .valid_out(mul_valid[5]), .result_out(prod_f32[5]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul6 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[6]), .b_in(d_f32), .valid_out(mul_valid[6]), .result_out(prod_f32[6]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul7 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[7]), .b_in(d_f32), .valid_out(mul_valid[7]), .result_out(prod_f32[7]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul8 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[8]), .b_in(d_f32), .valid_out(mul_valid[8]), .result_out(prod_f32[8]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul9 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[9]), .b_in(d_f32), .valid_out(mul_valid[9]), .result_out(prod_f32[9]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul10 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[10]), .b_in(d_f32), .valid_out(mul_valid[10]), .result_out(prod_f32[10]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul11 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[11]), .b_in(d_f32), .valid_out(mul_valid[11]), .result_out(prod_f32[11]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul12 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[12]), .b_in(d_f32), .valid_out(mul_valid[12]), .result_out(prod_f32[12]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul13 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[13]), .b_in(d_f32), .valid_out(mul_valid[13]), .result_out(prod_f32[13]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul14 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[14]), .b_in(d_f32), .valid_out(mul_valid[14]), .result_out(prod_f32[14]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul15 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[15]), .b_in(d_f32), .valid_out(mul_valid[15]), .result_out(prod_f32[15]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul16 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[16]), .b_in(d_f32), .valid_out(mul_valid[16]), .result_out(prod_f32[16]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul17 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[17]), .b_in(d_f32), .valid_out(mul_valid[17]), .result_out(prod_f32[17]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul18 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[18]), .b_in(d_f32), .valid_out(mul_valid[18]), .result_out(prod_f32[18]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul19 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[19]), .b_in(d_f32), .valid_out(mul_valid[19]), .result_out(prod_f32[19]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul20 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[20]), .b_in(d_f32), .valid_out(mul_valid[20]), .result_out(prod_f32[20]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul21 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[21]), .b_in(d_f32), .valid_out(mul_valid[21]), .result_out(prod_f32[21]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul22 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[22]), .b_in(d_f32), .valid_out(mul_valid[22]), .result_out(prod_f32[22]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul23 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[23]), .b_in(d_f32), .valid_out(mul_valid[23]), .result_out(prod_f32[23]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul24 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[24]), .b_in(d_f32), .valid_out(mul_valid[24]), .result_out(prod_f32[24]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul25 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[25]), .b_in(d_f32), .valid_out(mul_valid[25]), .result_out(prod_f32[25]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul26 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[26]), .b_in(d_f32), .valid_out(mul_valid[26]), .result_out(prod_f32[26]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul27 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[27]), .b_in(d_f32), .valid_out(mul_valid[27]), .result_out(prod_f32[27]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul28 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[28]), .b_in(d_f32), .valid_out(mul_valid[28]), .result_out(prod_f32[28]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul29 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[29]), .b_in(d_f32), .valid_out(mul_valid[29]), .result_out(prod_f32[29]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul30 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[30]), .b_in(d_f32), .valid_out(mul_valid[30]), .result_out(prod_f32[30]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul31 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(nib_f32[31]), .b_in(d_f32), .valid_out(mul_valid[31]), .result_out(prod_f32[31]));

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
