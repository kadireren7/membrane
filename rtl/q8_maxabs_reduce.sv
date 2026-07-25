// Phase 5.2: 5-cycle pipelined binary-tree magnitude reduction over one
// 32-element F16 block, bit-exact with the amax computation in
// q8_0_quant_block_scalar (src/quant/quant_simd.c): `amax=0.0f; for each
// x: if (fabsf(x) > amax) amax = fabsf(x);`.
//
// Two correctness properties this module must replicate exactly (both
// verified against the C reference in rtl/tb/tb_q8_maxabs_reduce.sv):
//   1. A NaN operand never wins a comparison (IEEE-754 comparison with
//      NaN is always false, so a NaN `x[j]` can never become the running
//      max) -- implemented as an explicit is_nan flag that always loses
//      to a non-NaN opponent, propagating either NaN forward on a
//      NaN-vs-NaN tie (harmless, since a NaN can never reach the final
//      output -- see property 2).
//   2. The C loop's `amax` starts at 0.0 and can only be reassigned via
//      a true `v > amax` comparison, which is never satisfied by NaN --
//      so amax is NEVER NaN, even if every element in the block is NaN
//      (amax stays 0.0 in that case). This module reproduces that by
//      substituting 0.0 for the tree's winner if that winner is NaN.
//
// Magnitude comparison uses the standard IEEE-754 property that, for two
// non-negative, non-NaN finite-or-infinite floats, comparing their raw
// bit patterns as unsigned integers gives the same ordering as comparing
// their numeric values (Infinity's all-ones exponent + zero mantissa
// naturally sorts above every finite value) -- so no floating-point ALU
// is needed for this stage at all, only integer comparison.
//
// Candidates are carried as a plain logic [15:0] {is_nan, 15'0, mag[14:0]}
// (bit 15 = is_nan flag, bits [14:0] = sign-cleared f16 bits) rather than
// a packed struct.
//
// Implementation note: every register update in this file is written as
// an explicit, individually-indexed statement inside a SINGLE always_ff
// block, rather than a `generate...for` block or a for-loop over an
// array inside always_ff. This is a deliberate simulator-portability
// workaround, not a style choice: Icarus Verilog 12.0 was found, via a
// series of minimal reproductions (see rtl/tb/ commit history), to
// silently corrupt the NBA scheduling of OTHER, unrelated always_ff
// blocks in the same module whenever a `generate...for` block containing
// always_ff is present anywhere in that module -- registers that should
// lag each other by one cycle (v1 <= valid_in; v2 <= v1; ...) instead
// updated on the SAME edge, as if the assignments were blocking. This
// reproduced with a `for` loop over an array inside a plain always_ff too
// (both trigger the same "sorry: constant selects..." diagnostic from the
// tool). Fully explicit, literal-indexed statements avoid the bug
// entirely and were verified correct in isolation before being used here.
module q8_maxabs_reduce (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[15:0]	x_in		[0:31],
	output	logic			valid_out,
	output	logic	[15:0]	amax_f16_out
);

	function automatic logic [15:0] mk_cand(input logic [15:0] x);
		logic	is_nan;

		is_nan = (x[14:10] == 5'h1F) && (x[9:0] != 10'h0);
		return ({is_nan, x[14:0]});
	endfunction

	function automatic logic [15:0] pick(input logic [15:0] a,
			input logic [15:0] b);
		logic	a_nan;
		logic	b_nan;

		a_nan = a[15];
		b_nan = b[15];
		if (a_nan && b_nan)
			return (a);
		if (a_nan)
			return (b);
		if (b_nan)
			return (a);
		if (a[14:0] >= b[14:0])
			return (a);
		return (b);
	endfunction

	logic	[15:0]	level0	[0:31];
	logic	[15:0]	level1	[0:15];
	logic	[15:0]	level2	[0:7];
	logic	[15:0]	level3	[0:3];
	logic	[15:0]	level4	[0:1];
	logic	[15:0]	level5;
	logic			valid_out_w;

	// The valid pipeline is a genuinely separate module instance, not a
	// register embedded in this module -- see valid_delay_line.sv's
	// header for the Icarus Verilog 12.0 issue this works around (a
	// register updated in the SAME always_ff/module as this module's
	// array-heavy, many-times-automatic-function-calling data path was
	// found, via a series of isolated reproductions, to update one cycle
	// early). This form was verified correct (see
	// rtl/tb/tb_q8_maxabs_reduce.sv).
	valid_delay_line #(.DEPTH(5)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out_w));

	always_comb begin
		level0[0] = mk_cand(x_in[0]);
		level0[1] = mk_cand(x_in[1]);
		level0[2] = mk_cand(x_in[2]);
		level0[3] = mk_cand(x_in[3]);
		level0[4] = mk_cand(x_in[4]);
		level0[5] = mk_cand(x_in[5]);
		level0[6] = mk_cand(x_in[6]);
		level0[7] = mk_cand(x_in[7]);
		level0[8] = mk_cand(x_in[8]);
		level0[9] = mk_cand(x_in[9]);
		level0[10] = mk_cand(x_in[10]);
		level0[11] = mk_cand(x_in[11]);
		level0[12] = mk_cand(x_in[12]);
		level0[13] = mk_cand(x_in[13]);
		level0[14] = mk_cand(x_in[14]);
		level0[15] = mk_cand(x_in[15]);
		level0[16] = mk_cand(x_in[16]);
		level0[17] = mk_cand(x_in[17]);
		level0[18] = mk_cand(x_in[18]);
		level0[19] = mk_cand(x_in[19]);
		level0[20] = mk_cand(x_in[20]);
		level0[21] = mk_cand(x_in[21]);
		level0[22] = mk_cand(x_in[22]);
		level0[23] = mk_cand(x_in[23]);
		level0[24] = mk_cand(x_in[24]);
		level0[25] = mk_cand(x_in[25]);
		level0[26] = mk_cand(x_in[26]);
		level0[27] = mk_cand(x_in[27]);
		level0[28] = mk_cand(x_in[28]);
		level0[29] = mk_cand(x_in[29]);
		level0[30] = mk_cand(x_in[30]);
		level0[31] = mk_cand(x_in[31]);
	end

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
		end else begin
			level1[0] <= pick(level0[0], level0[1]);
			level1[1] <= pick(level0[2], level0[3]);
			level1[2] <= pick(level0[4], level0[5]);
			level1[3] <= pick(level0[6], level0[7]);
			level1[4] <= pick(level0[8], level0[9]);
			level1[5] <= pick(level0[10], level0[11]);
			level1[6] <= pick(level0[12], level0[13]);
			level1[7] <= pick(level0[14], level0[15]);
			level1[8] <= pick(level0[16], level0[17]);
			level1[9] <= pick(level0[18], level0[19]);
			level1[10] <= pick(level0[20], level0[21]);
			level1[11] <= pick(level0[22], level0[23]);
			level1[12] <= pick(level0[24], level0[25]);
			level1[13] <= pick(level0[26], level0[27]);
			level1[14] <= pick(level0[28], level0[29]);
			level1[15] <= pick(level0[30], level0[31]);

			level2[0] <= pick(level1[0], level1[1]);
			level2[1] <= pick(level1[2], level1[3]);
			level2[2] <= pick(level1[4], level1[5]);
			level2[3] <= pick(level1[6], level1[7]);
			level2[4] <= pick(level1[8], level1[9]);
			level2[5] <= pick(level1[10], level1[11]);
			level2[6] <= pick(level1[12], level1[13]);
			level2[7] <= pick(level1[14], level1[15]);

			level3[0] <= pick(level2[0], level2[1]);
			level3[1] <= pick(level2[2], level2[3]);
			level3[2] <= pick(level2[4], level2[5]);
			level3[3] <= pick(level2[6], level2[7]);

			level4[0] <= pick(level3[0], level3[1]);
			level4[1] <= pick(level3[2], level3[3]);

			level5 <= pick(level4[0], level4[1]);
		end
	end

	always_comb begin
		valid_out = valid_out_w;
		amax_f16_out = level5[15] ? 16'h0000 : {1'b0, level5[14:0]};
	end
endmodule
