// Phase 5.2: per-lane multiply + round-to-nearest-even + saturate + pack
// into the 34-byte Q8_0 block (2-byte F16 scale + 32 signed int8), bit-
// exact with q8_0_quant_block_scalar's inner loop:
// `qs[j] = sat_i8_from_rounded_f32(rintf(x[j] * id));`
// (src/quant/quant_simd.c).
//
// Like q8_scale.sv, the multiply and the round-to-nearest-integer step
// are NOT synthesizable as written -- both go through the simulator's
// native `real` (double) arithmetic via $bitstoreal/$realtobits (see
// membrane_fp_functions.svh's f32_widen_to_f64/f64_narrow_to_f32_rne for
// the multiply's correct rounding, verified in
// rtl/tb/tb_f32_div_vectors.sv's methodology; round_half_to_even_int
// below implements rintf's round-to-nearest-even-integer directly on the
// widened double value, which is exact since F32->F64 widening loses no
// precision). NaN/Inf inputs are special-cased BEFORE the round-to-int
// step (rounding a non-finite value to an integer is undefined), matching
// sat_i8_from_rounded_f32's -128 sentinel for any non-finite input,
// itself replicating x86's CVTPS2DQ "integer indefinite" + saturating-
// pack hardware behavior (docs/phase5-quant-engine.md).
module q8_quantize_pack #(
	parameter int DELAY = 3
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[15:0]	x_in		[0:31],
	input	logic	[15:0]	d_f16_in,
	input	logic	[31:0]	id_f32_in,
	output	logic			valid_out,
	// Flattened 34-byte (272-bit) bus, byte 0 = LSB, rather than an
	// unpacked logic [7:0] packed_out[0:33] array port: Icarus Verilog
	// 12.0 was found, via a minimal reproduction, to simply not
	// propagate unpacked-array OUTPUT ports across a module instance
	// boundary at all (the driving signal was verified correct via
	// hierarchical reference inside the module; the port's value as seen
	// from outside was unconditionally X) -- INPUT unpacked-array ports
	// (x_in above) were separately verified to work correctly, so this
	// is specifically an output-port limitation. Flattened vectors are
	// also the more conventional/portable RTL interface idiom for
	// exactly this reason, so this is the pattern used at every module
	// boundary in this phase from here on, not a one-off patch.
	output	logic	[271:0]	packed_out
);
	import membrane_fp_pkg::*;

	function automatic int round_half_to_even_int(input real v);
		int		base;
		real	frac;
		int		result;

		base = $rtoi(v);
		frac = v - real'(base);
		result = base;
		if (frac >= 0.5) begin
			if (frac > 0.5 || (base % 2) != 0)
				result = base + 1;
		end else if (frac <= -0.5) begin
			if (frac < -0.5 || (base % 2) != 0)
				result = base - 1;
		end
		return (result);
	endfunction

	function automatic logic [7:0] quantize_lane(input logic [15:0] xh,
			input logic [31:0] id_f32);
		logic	[31:0]	x_f32;
		logic	[63:0]	x_f64;
		logic	[63:0]	id_f64;
		real			x_r;
		real			id_r;
		real			prod_r;
		logic	[31:0]	prod_f32;
		logic			prod_is_nan_or_inf;
		int				rounded;
		logic	[7:0]	q;

		x_f32 = f16_to_f32_bits(xh);
		x_f64 = f32_widen_to_f64(x_f32);
		id_f64 = f32_widen_to_f64(id_f32);
		x_r = $bitstoreal(x_f64);
		id_r = $bitstoreal(id_f64);
		prod_r = x_r * id_r;
		prod_f32 = f64_narrow_to_f32_rne($realtobits(prod_r));
		prod_is_nan_or_inf = (prod_f32[30:23] == 8'hFF);
		if (prod_is_nan_or_inf)
			q = 8'h80;
		else begin
			real	prod_exact_r;

			prod_exact_r = $bitstoreal(f32_widen_to_f64(prod_f32));
			if (prod_exact_r > 2147483520.0
					|| prod_exact_r < -2147483520.0)
				q = (prod_exact_r > 0.0) ? 8'h7F : 8'h80;
			else begin
				rounded = round_half_to_even_int(prod_exact_r);
				if (rounded > 127)
					q = 8'h7F;
				else if (rounded < -128)
					q = 8'h80;
				else
					q = 8'(rounded);
			end
		end
		return (q);
	endfunction

	logic	[7:0]	qs_comb		[0:31];
	// DELAY-deep pipeline for qs/d, matching valid_delay_line's DELAY
	// cycles below -- a single register stage here would let qs_pipe be
	// overwritten by later blocks (x_in changes every cycle in a real
	// streaming pipeline) before valid_out actually asserts, silently
	// reading stale/wrong data. This was a real bug in an earlier version
	// of this file (a single-cycle register paired with a DELAY=3
	// valid_delay_line), caught during design review before it reached
	// simulation.
	logic	[7:0]	qs_pipe		[0:DELAY - 1][0:31];
	logic	[15:0]	d_pipe		[0:DELAY - 1];

	always_comb begin
		qs_comb[0] = quantize_lane(x_in[0], id_f32_in);
		qs_comb[1] = quantize_lane(x_in[1], id_f32_in);
		qs_comb[2] = quantize_lane(x_in[2], id_f32_in);
		qs_comb[3] = quantize_lane(x_in[3], id_f32_in);
		qs_comb[4] = quantize_lane(x_in[4], id_f32_in);
		qs_comb[5] = quantize_lane(x_in[5], id_f32_in);
		qs_comb[6] = quantize_lane(x_in[6], id_f32_in);
		qs_comb[7] = quantize_lane(x_in[7], id_f32_in);
		qs_comb[8] = quantize_lane(x_in[8], id_f32_in);
		qs_comb[9] = quantize_lane(x_in[9], id_f32_in);
		qs_comb[10] = quantize_lane(x_in[10], id_f32_in);
		qs_comb[11] = quantize_lane(x_in[11], id_f32_in);
		qs_comb[12] = quantize_lane(x_in[12], id_f32_in);
		qs_comb[13] = quantize_lane(x_in[13], id_f32_in);
		qs_comb[14] = quantize_lane(x_in[14], id_f32_in);
		qs_comb[15] = quantize_lane(x_in[15], id_f32_in);
		qs_comb[16] = quantize_lane(x_in[16], id_f32_in);
		qs_comb[17] = quantize_lane(x_in[17], id_f32_in);
		qs_comb[18] = quantize_lane(x_in[18], id_f32_in);
		qs_comb[19] = quantize_lane(x_in[19], id_f32_in);
		qs_comb[20] = quantize_lane(x_in[20], id_f32_in);
		qs_comb[21] = quantize_lane(x_in[21], id_f32_in);
		qs_comb[22] = quantize_lane(x_in[22], id_f32_in);
		qs_comb[23] = quantize_lane(x_in[23], id_f32_in);
		qs_comb[24] = quantize_lane(x_in[24], id_f32_in);
		qs_comb[25] = quantize_lane(x_in[25], id_f32_in);
		qs_comb[26] = quantize_lane(x_in[26], id_f32_in);
		qs_comb[27] = quantize_lane(x_in[27], id_f32_in);
		qs_comb[28] = quantize_lane(x_in[28], id_f32_in);
		qs_comb[29] = quantize_lane(x_in[29], id_f32_in);
		qs_comb[30] = quantize_lane(x_in[30], id_f32_in);
		qs_comb[31] = quantize_lane(x_in[31], id_f32_in);
	end

	always_ff @(posedge clk) begin
		for (int j = 0; j < 32; j++)
			qs_pipe[0][j] <= qs_comb[j];
		d_pipe[0] <= d_f16_in;
		for (int k = DELAY - 1; k > 0; k--) begin
			for (int j = 0; j < 32; j++)
				qs_pipe[k][j] <= qs_pipe[k - 1][j];
			d_pipe[k] <= d_pipe[k - 1];
		end
	end

	valid_delay_line #(.DEPTH(DELAY)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out));

	// A plain `assign` here (rather than always_comb) is a deliberate
	// workaround: using the parameter expression DELAY-1 to index
	// qs_pipe/d_pipe inside an always_comb block was found, empirically,
	// to read back X regardless of what qs_pipe[DELAY-1] actually held
	// (confirmed correct via hierarchical reference in
	// rtl/tb/tb_q8_quantize_pack.sv's debug trace) -- yet another Icarus
	// Verilog 12.0 quirk around "constant" (here, parameter-derived)
	// array indices in procedural blocks, distinct from (but likely
	// related to) the always_ff issue documented in
	// rtl/q8_maxabs_reduce.sv. `assign` does not exhibit it.
	assign packed_out[7:0] = d_pipe[DELAY - 1][7:0];
	assign packed_out[15:8] = d_pipe[DELAY - 1][15:8];
	assign packed_out[23:16] = qs_pipe[DELAY - 1][0];
	assign packed_out[31:24] = qs_pipe[DELAY - 1][1];
	assign packed_out[39:32] = qs_pipe[DELAY - 1][2];
	assign packed_out[47:40] = qs_pipe[DELAY - 1][3];
	assign packed_out[55:48] = qs_pipe[DELAY - 1][4];
	assign packed_out[63:56] = qs_pipe[DELAY - 1][5];
	assign packed_out[71:64] = qs_pipe[DELAY - 1][6];
	assign packed_out[79:72] = qs_pipe[DELAY - 1][7];
	assign packed_out[87:80] = qs_pipe[DELAY - 1][8];
	assign packed_out[95:88] = qs_pipe[DELAY - 1][9];
	assign packed_out[103:96] = qs_pipe[DELAY - 1][10];
	assign packed_out[111:104] = qs_pipe[DELAY - 1][11];
	assign packed_out[119:112] = qs_pipe[DELAY - 1][12];
	assign packed_out[127:120] = qs_pipe[DELAY - 1][13];
	assign packed_out[135:128] = qs_pipe[DELAY - 1][14];
	assign packed_out[143:136] = qs_pipe[DELAY - 1][15];
	assign packed_out[151:144] = qs_pipe[DELAY - 1][16];
	assign packed_out[159:152] = qs_pipe[DELAY - 1][17];
	assign packed_out[167:160] = qs_pipe[DELAY - 1][18];
	assign packed_out[175:168] = qs_pipe[DELAY - 1][19];
	assign packed_out[183:176] = qs_pipe[DELAY - 1][20];
	assign packed_out[191:184] = qs_pipe[DELAY - 1][21];
	assign packed_out[199:192] = qs_pipe[DELAY - 1][22];
	assign packed_out[207:200] = qs_pipe[DELAY - 1][23];
	assign packed_out[215:208] = qs_pipe[DELAY - 1][24];
	assign packed_out[223:216] = qs_pipe[DELAY - 1][25];
	assign packed_out[231:224] = qs_pipe[DELAY - 1][26];
	assign packed_out[239:232] = qs_pipe[DELAY - 1][27];
	assign packed_out[247:240] = qs_pipe[DELAY - 1][28];
	assign packed_out[255:248] = qs_pipe[DELAY - 1][29];
	assign packed_out[263:256] = qs_pipe[DELAY - 1][30];
	assign packed_out[271:264] = qs_pipe[DELAY - 1][31];
endmodule
