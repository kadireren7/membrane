// Phase 5.2: the sequential signed-magnitude scan for Q4_0's amax/mx,
// factored into its own module -- see q4_pack.sv's header comment for
// why this must be sequential (order-preserving), not a tree reduction.
// Kept as a separate module instance from q4_scale/q4_pack's own logic
// for clarity, not correctness -- the actual root cause of an earlier
// hang when these stages were combined was Icarus Verilog 12.0 mishandling
// TWO OR MORE modules calling the same `` `include``d automatic function
// (fixed by moving those functions into membrane_fp_pkg.sv, a proper
// SystemVerilog package, imported here rather than textually included --
// see that file's header comment for the full story).
module q4_scan (
	input	logic	[511:0]	x_in_flat,
	output	logic	[31:0]	mx_f32_out
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


	// Assigns the function's own name instead of using `return` --
	// yosys 0.33 rejects `return` inside module-scoped functions, see
	// membrane_fp_adder.sv's lzc_9 header comment for the full finding.
	function automatic logic is_nan_f32(input logic [31:0] f);
		is_nan_f32 = (f[30:23] == 8'hFF) && (f[22:0] != 23'h0);
	endfunction

	function automatic logic [31:0] fabs_f32(input logic [31:0] f);
		fabs_f32 = {1'b0, f[30:0]};
	endfunction

	function automatic logic mag_gt(input logic [31:0] a_abs,
			input logic [31:0] b_abs);
		mag_gt = (a_abs[30:0] > b_abs[30:0]);
	endfunction

	function automatic logic [31:0] scan_amax(input logic [31:0] cur_amax,
			input logic [31:0] v);
		if (!is_nan_f32(v) && mag_gt(fabs_f32(v), cur_amax))
			scan_amax = fabs_f32(v);
		else
			scan_amax = cur_amax;
	endfunction

	function automatic logic [31:0] scan_mx(input logic [31:0] cur_amax,
			input logic [31:0] cur_mx, input logic [31:0] v);
		if (!is_nan_f32(v) && mag_gt(fabs_f32(v), cur_amax))
			scan_mx = v;
		else
			scan_mx = cur_mx;
	endfunction

	logic	[31:0]	xf32	[0:31];
	logic	[31:0]	amax0, amax1, amax2, amax3, amax4, amax5, amax6, amax7;
	logic	[31:0]	amax8, amax9, amax10, amax11, amax12, amax13, amax14;
	logic	[31:0]	amax15, amax16, amax17, amax18, amax19, amax20, amax21;
	logic	[31:0]	amax22, amax23, amax24, amax25, amax26, amax27, amax28;
	logic	[31:0]	amax29, amax30, amax31, amax32;
	logic	[31:0]	mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7, mx8, mx9, mx10;
	logic	[31:0]	mx11, mx12, mx13, mx14, mx15, mx16, mx17, mx18, mx19;
	logic	[31:0]	mx20, mx21, mx22, mx23, mx24, mx25, mx26, mx27, mx28;
	logic	[31:0]	mx29, mx30, mx31, mx32;

	always_comb begin
		xf32[0] = membrane_fp_pkg::f16_to_f32_bits(x_in[0]);
		xf32[1] = membrane_fp_pkg::f16_to_f32_bits(x_in[1]);
		xf32[2] = membrane_fp_pkg::f16_to_f32_bits(x_in[2]);
		xf32[3] = membrane_fp_pkg::f16_to_f32_bits(x_in[3]);
		xf32[4] = membrane_fp_pkg::f16_to_f32_bits(x_in[4]);
		xf32[5] = membrane_fp_pkg::f16_to_f32_bits(x_in[5]);
		xf32[6] = membrane_fp_pkg::f16_to_f32_bits(x_in[6]);
		xf32[7] = membrane_fp_pkg::f16_to_f32_bits(x_in[7]);
		xf32[8] = membrane_fp_pkg::f16_to_f32_bits(x_in[8]);
		xf32[9] = membrane_fp_pkg::f16_to_f32_bits(x_in[9]);
		xf32[10] = membrane_fp_pkg::f16_to_f32_bits(x_in[10]);
		xf32[11] = membrane_fp_pkg::f16_to_f32_bits(x_in[11]);
		xf32[12] = membrane_fp_pkg::f16_to_f32_bits(x_in[12]);
		xf32[13] = membrane_fp_pkg::f16_to_f32_bits(x_in[13]);
		xf32[14] = membrane_fp_pkg::f16_to_f32_bits(x_in[14]);
		xf32[15] = membrane_fp_pkg::f16_to_f32_bits(x_in[15]);
		xf32[16] = membrane_fp_pkg::f16_to_f32_bits(x_in[16]);
		xf32[17] = membrane_fp_pkg::f16_to_f32_bits(x_in[17]);
		xf32[18] = membrane_fp_pkg::f16_to_f32_bits(x_in[18]);
		xf32[19] = membrane_fp_pkg::f16_to_f32_bits(x_in[19]);
		xf32[20] = membrane_fp_pkg::f16_to_f32_bits(x_in[20]);
		xf32[21] = membrane_fp_pkg::f16_to_f32_bits(x_in[21]);
		xf32[22] = membrane_fp_pkg::f16_to_f32_bits(x_in[22]);
		xf32[23] = membrane_fp_pkg::f16_to_f32_bits(x_in[23]);
		xf32[24] = membrane_fp_pkg::f16_to_f32_bits(x_in[24]);
		xf32[25] = membrane_fp_pkg::f16_to_f32_bits(x_in[25]);
		xf32[26] = membrane_fp_pkg::f16_to_f32_bits(x_in[26]);
		xf32[27] = membrane_fp_pkg::f16_to_f32_bits(x_in[27]);
		xf32[28] = membrane_fp_pkg::f16_to_f32_bits(x_in[28]);
		xf32[29] = membrane_fp_pkg::f16_to_f32_bits(x_in[29]);
		xf32[30] = membrane_fp_pkg::f16_to_f32_bits(x_in[30]);
		xf32[31] = membrane_fp_pkg::f16_to_f32_bits(x_in[31]);

		amax0 = 32'h0;
		mx0 = 32'h0;
		amax1 = scan_amax(amax0, xf32[0]);
		mx1 = scan_mx(amax0, mx0, xf32[0]);
		amax2 = scan_amax(amax1, xf32[1]);
		mx2 = scan_mx(amax1, mx1, xf32[1]);
		amax3 = scan_amax(amax2, xf32[2]);
		mx3 = scan_mx(amax2, mx2, xf32[2]);
		amax4 = scan_amax(amax3, xf32[3]);
		mx4 = scan_mx(amax3, mx3, xf32[3]);
		amax5 = scan_amax(amax4, xf32[4]);
		mx5 = scan_mx(amax4, mx4, xf32[4]);
		amax6 = scan_amax(amax5, xf32[5]);
		mx6 = scan_mx(amax5, mx5, xf32[5]);
		amax7 = scan_amax(amax6, xf32[6]);
		mx7 = scan_mx(amax6, mx6, xf32[6]);
		amax8 = scan_amax(amax7, xf32[7]);
		mx8 = scan_mx(amax7, mx7, xf32[7]);
		amax9 = scan_amax(amax8, xf32[8]);
		mx9 = scan_mx(amax8, mx8, xf32[8]);
		amax10 = scan_amax(amax9, xf32[9]);
		mx10 = scan_mx(amax9, mx9, xf32[9]);
		amax11 = scan_amax(amax10, xf32[10]);
		mx11 = scan_mx(amax10, mx10, xf32[10]);
		amax12 = scan_amax(amax11, xf32[11]);
		mx12 = scan_mx(amax11, mx11, xf32[11]);
		amax13 = scan_amax(amax12, xf32[12]);
		mx13 = scan_mx(amax12, mx12, xf32[12]);
		amax14 = scan_amax(amax13, xf32[13]);
		mx14 = scan_mx(amax13, mx13, xf32[13]);
		amax15 = scan_amax(amax14, xf32[14]);
		mx15 = scan_mx(amax14, mx14, xf32[14]);
		amax16 = scan_amax(amax15, xf32[15]);
		mx16 = scan_mx(amax15, mx15, xf32[15]);
		amax17 = scan_amax(amax16, xf32[16]);
		mx17 = scan_mx(amax16, mx16, xf32[16]);
		amax18 = scan_amax(amax17, xf32[17]);
		mx18 = scan_mx(amax17, mx17, xf32[17]);
		amax19 = scan_amax(amax18, xf32[18]);
		mx19 = scan_mx(amax18, mx18, xf32[18]);
		amax20 = scan_amax(amax19, xf32[19]);
		mx20 = scan_mx(amax19, mx19, xf32[19]);
		amax21 = scan_amax(amax20, xf32[20]);
		mx21 = scan_mx(amax20, mx20, xf32[20]);
		amax22 = scan_amax(amax21, xf32[21]);
		mx22 = scan_mx(amax21, mx21, xf32[21]);
		amax23 = scan_amax(amax22, xf32[22]);
		mx23 = scan_mx(amax22, mx22, xf32[22]);
		amax24 = scan_amax(amax23, xf32[23]);
		mx24 = scan_mx(amax23, mx23, xf32[23]);
		amax25 = scan_amax(amax24, xf32[24]);
		mx25 = scan_mx(amax24, mx24, xf32[24]);
		amax26 = scan_amax(amax25, xf32[25]);
		mx26 = scan_mx(amax25, mx25, xf32[25]);
		amax27 = scan_amax(amax26, xf32[26]);
		mx27 = scan_mx(amax26, mx26, xf32[26]);
		amax28 = scan_amax(amax27, xf32[27]);
		mx28 = scan_mx(amax27, mx27, xf32[27]);
		amax29 = scan_amax(amax28, xf32[28]);
		mx29 = scan_mx(amax28, mx28, xf32[28]);
		amax30 = scan_amax(amax29, xf32[29]);
		mx30 = scan_mx(amax29, mx29, xf32[29]);
		amax31 = scan_amax(amax30, xf32[30]);
		mx31 = scan_mx(amax30, mx30, xf32[30]);
		amax32 = scan_amax(amax31, xf32[31]);
		mx32 = scan_mx(amax31, mx31, xf32[31]);
	end

	assign mx_f32_out = mx32;
endmodule
