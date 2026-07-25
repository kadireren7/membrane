// Phase 5.2: Q4_0 dequantize -- unpack the 18-byte block (2-byte F16
// scale + 16 bytes of packed 4-bit pairs), reconstruct each lane, narrow
// to F16. Bit-exact with q4_0_dequant_block_scalar (src/quant/quant_simd.c).
module q4_unpack #(
	parameter int DELAY = 2
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[143:0]	packed_in,
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

	// C reference (q4_0_dequant_block_scalar):
	//   lo = qs[j] & 0xF; hi = qs[j] >> 4;
	//   out[j]    = ((float)lo - 8.0f) * d;
	//   out[j+16] = ((float)hi - 8.0f) * d;
	function automatic logic [15:0] dequant_nibble(input logic [3:0] nib,
			input logic [63:0] d64);
		real			v_r;
		real			d_r;
		real			prod_r;
		logic	[31:0]	prod_f32;

		v_r = real'(int'(nib)) - 8.0;
		d_r = $bitstoreal(d64);
		prod_r = v_r * d_r;
		prod_f32 = f64_narrow_to_f32_rne($realtobits(prod_r));
		return (f32_to_f16_bits(prod_f32));
	endfunction

	logic	[7:0]	qb		[0:15];
	logic	[15:0]	x_comb	[0:31];

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

	always_comb begin
		x_comb[0] = dequant_nibble(qb[0][3:0], d_f64);
		x_comb[16] = dequant_nibble(qb[0][7:4], d_f64);
		x_comb[1] = dequant_nibble(qb[1][3:0], d_f64);
		x_comb[17] = dequant_nibble(qb[1][7:4], d_f64);
		x_comb[2] = dequant_nibble(qb[2][3:0], d_f64);
		x_comb[18] = dequant_nibble(qb[2][7:4], d_f64);
		x_comb[3] = dequant_nibble(qb[3][3:0], d_f64);
		x_comb[19] = dequant_nibble(qb[3][7:4], d_f64);
		x_comb[4] = dequant_nibble(qb[4][3:0], d_f64);
		x_comb[20] = dequant_nibble(qb[4][7:4], d_f64);
		x_comb[5] = dequant_nibble(qb[5][3:0], d_f64);
		x_comb[21] = dequant_nibble(qb[5][7:4], d_f64);
		x_comb[6] = dequant_nibble(qb[6][3:0], d_f64);
		x_comb[22] = dequant_nibble(qb[6][7:4], d_f64);
		x_comb[7] = dequant_nibble(qb[7][3:0], d_f64);
		x_comb[23] = dequant_nibble(qb[7][7:4], d_f64);
		x_comb[8] = dequant_nibble(qb[8][3:0], d_f64);
		x_comb[24] = dequant_nibble(qb[8][7:4], d_f64);
		x_comb[9] = dequant_nibble(qb[9][3:0], d_f64);
		x_comb[25] = dequant_nibble(qb[9][7:4], d_f64);
		x_comb[10] = dequant_nibble(qb[10][3:0], d_f64);
		x_comb[26] = dequant_nibble(qb[10][7:4], d_f64);
		x_comb[11] = dequant_nibble(qb[11][3:0], d_f64);
		x_comb[27] = dequant_nibble(qb[11][7:4], d_f64);
		x_comb[12] = dequant_nibble(qb[12][3:0], d_f64);
		x_comb[28] = dequant_nibble(qb[12][7:4], d_f64);
		x_comb[13] = dequant_nibble(qb[13][3:0], d_f64);
		x_comb[29] = dequant_nibble(qb[13][7:4], d_f64);
		x_comb[14] = dequant_nibble(qb[14][3:0], d_f64);
		x_comb[30] = dequant_nibble(qb[14][7:4], d_f64);
		x_comb[15] = dequant_nibble(qb[15][3:0], d_f64);
		x_comb[31] = dequant_nibble(qb[15][7:4], d_f64);
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
