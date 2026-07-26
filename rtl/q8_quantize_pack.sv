// Phase 5.3: per-lane multiply + round-to-nearest-even + saturate + pack
// into the 34-byte Q8_0 block (2-byte F16 scale + 32 signed int8), bit-
// exact with q8_0_quant_block_scalar's inner loop:
// `qs[j] = sat_i8_from_rounded_f32(rintf(x[j] * id));`
// (src/quant/quant_simd.c).
//
// Fully synthesizable: each lane's multiply runs through its own
// membrane_fp_multiplier.sv instance (32 in total -- a real design
// targeting minimum area would very likely time-multiplex a smaller
// number of multiplier units across the 32 lanes instead of
// instantiating one per lane, not attempted here, disclosed in
// docs/phase5-synthesizable-fpga.md), and the round-to-nearest-even +
// saturate + NaN/Inf-sentinel step is membrane_fp_pkg.sv's
// membrane_fp_pkg::f32_round_sat_to_i8(), a synthesizable function (no
// `real`/`shortreal`/DPI) verified bit-exact against
// sat_i8_from_rounded_f32(rintf(...)) across 20,000+ vectors
// (rtl/tb/tb_int_conv.sv).
module q8_quantize_pack #(
	parameter int MUL_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[511:0]	x_in_flat,
	input	logic	[15:0]	d_f16_in,
	input	logic	[31:0]	id_f32_in,
	output	logic			valid_out,
	// Flattened 34-byte (272-bit) bus -- see git history / earlier phase
	// docs for why unpacked-array OUTPUT ports are avoided throughout
	// this RTL tree (an Icarus Verilog 12.0 limitation, not a style
	// choice).
	output	logic	[271:0]	packed_out
);
	logic	[15:0]	x_in	[0:31];

	assign x_in[0] = x_in_flat[15:0];
	assign x_in[1] = x_in_flat[31:16];
	assign x_in[2] = x_in_flat[47:32];
	assign x_in[3] = x_in_flat[63:48];
	assign x_in[4] = x_in_flat[79:64];
	assign x_in[5] = x_in_flat[95:80];
	assign x_in[6] = x_in_flat[111:96];
	assign x_in[7] = x_in_flat[127:112];
	assign x_in[8] = x_in_flat[143:128];
	assign x_in[9] = x_in_flat[159:144];
	assign x_in[10] = x_in_flat[175:160];
	assign x_in[11] = x_in_flat[191:176];
	assign x_in[12] = x_in_flat[207:192];
	assign x_in[13] = x_in_flat[223:208];
	assign x_in[14] = x_in_flat[239:224];
	assign x_in[15] = x_in_flat[255:240];
	assign x_in[16] = x_in_flat[271:256];
	assign x_in[17] = x_in_flat[287:272];
	assign x_in[18] = x_in_flat[303:288];
	assign x_in[19] = x_in_flat[319:304];
	assign x_in[20] = x_in_flat[335:320];
	assign x_in[21] = x_in_flat[351:336];
	assign x_in[22] = x_in_flat[367:352];
	assign x_in[23] = x_in_flat[383:368];
	assign x_in[24] = x_in_flat[399:384];
	assign x_in[25] = x_in_flat[415:400];
	assign x_in[26] = x_in_flat[431:416];
	assign x_in[27] = x_in_flat[447:432];
	assign x_in[28] = x_in_flat[463:448];
	assign x_in[29] = x_in_flat[479:464];
	assign x_in[30] = x_in_flat[495:480];
	assign x_in[31] = x_in_flat[511:496];


	logic	[31:0]	x_f32	[0:31];
	logic	[31:0]	prod_f32	[0:31];
	logic			mul_valid	[0:31];

	assign x_f32[0] = membrane_fp_pkg::f16_to_f32_bits(x_in[0]);
	assign x_f32[1] = membrane_fp_pkg::f16_to_f32_bits(x_in[1]);
	assign x_f32[2] = membrane_fp_pkg::f16_to_f32_bits(x_in[2]);
	assign x_f32[3] = membrane_fp_pkg::f16_to_f32_bits(x_in[3]);
	assign x_f32[4] = membrane_fp_pkg::f16_to_f32_bits(x_in[4]);
	assign x_f32[5] = membrane_fp_pkg::f16_to_f32_bits(x_in[5]);
	assign x_f32[6] = membrane_fp_pkg::f16_to_f32_bits(x_in[6]);
	assign x_f32[7] = membrane_fp_pkg::f16_to_f32_bits(x_in[7]);
	assign x_f32[8] = membrane_fp_pkg::f16_to_f32_bits(x_in[8]);
	assign x_f32[9] = membrane_fp_pkg::f16_to_f32_bits(x_in[9]);
	assign x_f32[10] = membrane_fp_pkg::f16_to_f32_bits(x_in[10]);
	assign x_f32[11] = membrane_fp_pkg::f16_to_f32_bits(x_in[11]);
	assign x_f32[12] = membrane_fp_pkg::f16_to_f32_bits(x_in[12]);
	assign x_f32[13] = membrane_fp_pkg::f16_to_f32_bits(x_in[13]);
	assign x_f32[14] = membrane_fp_pkg::f16_to_f32_bits(x_in[14]);
	assign x_f32[15] = membrane_fp_pkg::f16_to_f32_bits(x_in[15]);
	assign x_f32[16] = membrane_fp_pkg::f16_to_f32_bits(x_in[16]);
	assign x_f32[17] = membrane_fp_pkg::f16_to_f32_bits(x_in[17]);
	assign x_f32[18] = membrane_fp_pkg::f16_to_f32_bits(x_in[18]);
	assign x_f32[19] = membrane_fp_pkg::f16_to_f32_bits(x_in[19]);
	assign x_f32[20] = membrane_fp_pkg::f16_to_f32_bits(x_in[20]);
	assign x_f32[21] = membrane_fp_pkg::f16_to_f32_bits(x_in[21]);
	assign x_f32[22] = membrane_fp_pkg::f16_to_f32_bits(x_in[22]);
	assign x_f32[23] = membrane_fp_pkg::f16_to_f32_bits(x_in[23]);
	assign x_f32[24] = membrane_fp_pkg::f16_to_f32_bits(x_in[24]);
	assign x_f32[25] = membrane_fp_pkg::f16_to_f32_bits(x_in[25]);
	assign x_f32[26] = membrane_fp_pkg::f16_to_f32_bits(x_in[26]);
	assign x_f32[27] = membrane_fp_pkg::f16_to_f32_bits(x_in[27]);
	assign x_f32[28] = membrane_fp_pkg::f16_to_f32_bits(x_in[28]);
	assign x_f32[29] = membrane_fp_pkg::f16_to_f32_bits(x_in[29]);
	assign x_f32[30] = membrane_fp_pkg::f16_to_f32_bits(x_in[30]);
	assign x_f32[31] = membrane_fp_pkg::f16_to_f32_bits(x_in[31]);

	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul0 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[0]), .b_in(id_f32_in), .valid_out(mul_valid[0]), .result_out(prod_f32[0]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[1]), .b_in(id_f32_in), .valid_out(mul_valid[1]), .result_out(prod_f32[1]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[2]), .b_in(id_f32_in), .valid_out(mul_valid[2]), .result_out(prod_f32[2]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul3 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[3]), .b_in(id_f32_in), .valid_out(mul_valid[3]), .result_out(prod_f32[3]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul4 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[4]), .b_in(id_f32_in), .valid_out(mul_valid[4]), .result_out(prod_f32[4]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul5 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[5]), .b_in(id_f32_in), .valid_out(mul_valid[5]), .result_out(prod_f32[5]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul6 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[6]), .b_in(id_f32_in), .valid_out(mul_valid[6]), .result_out(prod_f32[6]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul7 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[7]), .b_in(id_f32_in), .valid_out(mul_valid[7]), .result_out(prod_f32[7]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul8 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[8]), .b_in(id_f32_in), .valid_out(mul_valid[8]), .result_out(prod_f32[8]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul9 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[9]), .b_in(id_f32_in), .valid_out(mul_valid[9]), .result_out(prod_f32[9]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul10 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[10]), .b_in(id_f32_in), .valid_out(mul_valid[10]), .result_out(prod_f32[10]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul11 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[11]), .b_in(id_f32_in), .valid_out(mul_valid[11]), .result_out(prod_f32[11]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul12 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[12]), .b_in(id_f32_in), .valid_out(mul_valid[12]), .result_out(prod_f32[12]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul13 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[13]), .b_in(id_f32_in), .valid_out(mul_valid[13]), .result_out(prod_f32[13]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul14 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[14]), .b_in(id_f32_in), .valid_out(mul_valid[14]), .result_out(prod_f32[14]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul15 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[15]), .b_in(id_f32_in), .valid_out(mul_valid[15]), .result_out(prod_f32[15]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul16 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[16]), .b_in(id_f32_in), .valid_out(mul_valid[16]), .result_out(prod_f32[16]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul17 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[17]), .b_in(id_f32_in), .valid_out(mul_valid[17]), .result_out(prod_f32[17]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul18 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[18]), .b_in(id_f32_in), .valid_out(mul_valid[18]), .result_out(prod_f32[18]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul19 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[19]), .b_in(id_f32_in), .valid_out(mul_valid[19]), .result_out(prod_f32[19]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul20 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[20]), .b_in(id_f32_in), .valid_out(mul_valid[20]), .result_out(prod_f32[20]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul21 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[21]), .b_in(id_f32_in), .valid_out(mul_valid[21]), .result_out(prod_f32[21]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul22 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[22]), .b_in(id_f32_in), .valid_out(mul_valid[22]), .result_out(prod_f32[22]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul23 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[23]), .b_in(id_f32_in), .valid_out(mul_valid[23]), .result_out(prod_f32[23]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul24 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[24]), .b_in(id_f32_in), .valid_out(mul_valid[24]), .result_out(prod_f32[24]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul25 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[25]), .b_in(id_f32_in), .valid_out(mul_valid[25]), .result_out(prod_f32[25]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul26 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[26]), .b_in(id_f32_in), .valid_out(mul_valid[26]), .result_out(prod_f32[26]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul27 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[27]), .b_in(id_f32_in), .valid_out(mul_valid[27]), .result_out(prod_f32[27]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul28 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[28]), .b_in(id_f32_in), .valid_out(mul_valid[28]), .result_out(prod_f32[28]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul29 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[29]), .b_in(id_f32_in), .valid_out(mul_valid[29]), .result_out(prod_f32[29]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul30 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[30]), .b_in(id_f32_in), .valid_out(mul_valid[30]), .result_out(prod_f32[30]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul31 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(x_f32[31]), .b_in(id_f32_in), .valid_out(mul_valid[31]), .result_out(prod_f32[31]));

	logic	[7:0]	qs	[0:31];

	assign qs[0] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[0]);
	assign qs[1] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[1]);
	assign qs[2] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[2]);
	assign qs[3] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[3]);
	assign qs[4] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[4]);
	assign qs[5] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[5]);
	assign qs[6] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[6]);
	assign qs[7] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[7]);
	assign qs[8] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[8]);
	assign qs[9] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[9]);
	assign qs[10] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[10]);
	assign qs[11] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[11]);
	assign qs[12] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[12]);
	assign qs[13] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[13]);
	assign qs[14] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[14]);
	assign qs[15] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[15]);
	assign qs[16] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[16]);
	assign qs[17] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[17]);
	assign qs[18] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[18]);
	assign qs[19] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[19]);
	assign qs[20] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[20]);
	assign qs[21] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[21]);
	assign qs[22] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[22]);
	assign qs[23] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[23]);
	assign qs[24] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[24]);
	assign qs[25] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[25]);
	assign qs[26] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[26]);
	assign qs[27] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[27]);
	assign qs[28] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[28]);
	assign qs[29] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[29]);
	assign qs[30] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[30]);
	assign qs[31] = membrane_fp_pkg::f32_round_sat_to_i8(prod_f32[31]);

	// d_f16_in must be delayed by the same MUL_DELAY cycles as the
	// multiply results it is packed alongside.
	logic	[15:0]	d_pipe	[0:MUL_DELAY - 1];

	always_ff @(posedge clk) begin
		d_pipe[0] <= d_f16_in;
		for (int k = MUL_DELAY - 1; k > 0; k--)
			d_pipe[k] <= d_pipe[k - 1];
	end

	assign valid_out = mul_valid[0];

	assign packed_out[7:0] = d_pipe[MUL_DELAY - 1][7:0];
	assign packed_out[15:8] = d_pipe[MUL_DELAY - 1][15:8];
	assign packed_out[23:16] = qs[0];
	assign packed_out[31:24] = qs[1];
	assign packed_out[39:32] = qs[2];
	assign packed_out[47:40] = qs[3];
	assign packed_out[55:48] = qs[4];
	assign packed_out[63:56] = qs[5];
	assign packed_out[71:64] = qs[6];
	assign packed_out[79:72] = qs[7];
	assign packed_out[87:80] = qs[8];
	assign packed_out[95:88] = qs[9];
	assign packed_out[103:96] = qs[10];
	assign packed_out[111:104] = qs[11];
	assign packed_out[119:112] = qs[12];
	assign packed_out[127:120] = qs[13];
	assign packed_out[135:128] = qs[14];
	assign packed_out[143:136] = qs[15];
	assign packed_out[151:144] = qs[16];
	assign packed_out[159:152] = qs[17];
	assign packed_out[167:160] = qs[18];
	assign packed_out[175:168] = qs[19];
	assign packed_out[183:176] = qs[20];
	assign packed_out[191:184] = qs[21];
	assign packed_out[199:192] = qs[22];
	assign packed_out[207:200] = qs[23];
	assign packed_out[215:208] = qs[24];
	assign packed_out[223:216] = qs[25];
	assign packed_out[231:224] = qs[26];
	assign packed_out[239:232] = qs[27];
	assign packed_out[247:240] = qs[28];
	assign packed_out[255:248] = qs[29];
	assign packed_out[263:256] = qs[30];
	assign packed_out[271:264] = qs[31];
endmodule
