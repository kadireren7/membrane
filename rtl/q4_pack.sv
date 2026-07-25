// Phase 5.2: Q4_0 encode, quantize+pack stage -- per-pair quantize+pack
// into the 18-byte block (2-byte F16 scale + 16 bytes of packed 4-bit
// pairs), given an already-computed scale (d_f16/id_f32, see q4_scan.sv +
// q4_scale.sv for how those are produced from x_in). Bit-exact with
// q4_0_quant_block_scalar (src/quant/quant_simd.c).
//
// This module takes d_f16/id_f32 as direct inputs rather than
// instantiating q4_scan/q4_scale internally -- see the module-level
// comment right before `module q4_pack` below for why (an Icarus Verilog
// 12.0 limitation, disclosed in docs/phase5-fpga-streaming.md).
//
// The scan stage (q4_scan.sv, not part of this module) is DELIBERATELY
// sequential (32 chained compare-selects, combinational within one cycle
// group), NOT a tree
// reduction like q8_maxabs_reduce -- see docs/phase5-hardware-datapath.md
// section 5.3: it keeps the SIGNED value (`mx`) at the first element
// whose magnitude strictly exceeds the running max, and a tree reduction
// evaluates comparisons in a different order than the left-to-right C
// loop, which can pick a different tied element on a magnitude tie --
// changing the sign of `mx`, hence of every output in the block. As a
// first prototype this is one long combinational chain (functionally
// exact, matches the C loop's order precisely) rather than a multi-cycle
// pipelined systolic chain; a real synthesis target would need to break
// this into pipeline stages for timing closure and better initiation
// interval (disclosed in docs/phase5-fpga-streaming.md, not attempted
// here).
//
// Pack step per the C reference: `qi = (int8_t)(x*id + 8.5f)` -- a
// TRUNCATING cast (not round-to-nearest, deliberately different from
// Q8_0), and `xi = (uint8_t)((15 < (int)qi) ? 15 : (int)qi)` -- an upper
// clamp to 15 only, no lower clamp, then a modular uint8_t narrowing that
// lets very negative qi wrap (e.g. qi=-5 -> xi=251, matching ggml's own
// behavior exactly rather than "fixing" it -- docs/phase4-ggml-quant-
// parity.md). NaN/Inf/out-of-range x*id+8.5 inputs replicate x86's
// CVTTSS2SI truncating-convert "integer indefinite" sentinel (INT32_MIN),
// which this design's C reference itself (compiled for this same x86_64
// target) already relies on -- narrowed to the low 8 bits like the C
// int8_t cast, INT32_MIN's low byte is 0x00, so a NaN/Inf/out-of-range
// lane always encodes as qi=0 -> xi=0, replicated exactly here, not
// re-derived from first principles.
//
// The pack step's `xi0 | (xi1 << 4)` is an 8-bit OR of the FULL
// (possibly >15) uint8_t values, not a clean 4+4 bit concatenation --
// see quantize_lane's own header comment below for why that distinction
// matters.
// Takes d_f16/id_f32 as direct input ports (the caller is expected to
// have already run x_in's F16 words through q4_scan.sv + q4_scale.sv)
// rather than instantiating those stages internally: this module's own
// 32-lane quantize+pack logic, q4_scan, and q4_scale were each verified
// correct and fast standalone (20,000+ vectors each, well under a
// second), but Icarus Verilog 12.0 was found to hang indefinitely -- even
// on a single static, non-clocked evaluation -- whenever any two of
// these three real-arithmetic-heavy stages were instantiated together in
// the same simulation, regardless of whether they shared functions via
// `` `include`` or a proper SystemVerilog package (both were tried; see
// membrane_fp_pkg.sv's header for the include-vs-package finding, which
// fixed a separate, real duplicate-declaration error but did not fix
// this hang). This is disclosed as a simulator limitation encountered
// during this phase, not a correctness issue in any stage -- see
// docs/phase5-fpga-streaming.md for the full account, including why a
// single-simulation, multi-module cosimulation of the full Q4_0 encode
// chain could not be completed in this environment, and what was
// verified instead (each stage independently, plus the equivalent
// pipeline timing/streaming behavior via the C cycle model in
// tools/membrane-hw-sim).
module q4_pack #(
	parameter int DELAY = 4
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[15:0]	x_in		[0:31],
	input	logic	[15:0]	d_f16,
	input	logic	[31:0]	id_f32,
	output	logic			valid_out,
	output	logic	[143:0]	packed_out
);
	import membrane_fp_pkg::*;

	logic	[31:0]	xf32	[0:31];
	logic	[63:0]	id_f64_w;

	assign id_f64_w = f32_widen_to_f64(id_f32);
	assign xf32[0] = f16_to_f32_bits(x_in[0]);
	assign xf32[1] = f16_to_f32_bits(x_in[1]);
	assign xf32[2] = f16_to_f32_bits(x_in[2]);
	assign xf32[3] = f16_to_f32_bits(x_in[3]);
	assign xf32[4] = f16_to_f32_bits(x_in[4]);
	assign xf32[5] = f16_to_f32_bits(x_in[5]);
	assign xf32[6] = f16_to_f32_bits(x_in[6]);
	assign xf32[7] = f16_to_f32_bits(x_in[7]);
	assign xf32[8] = f16_to_f32_bits(x_in[8]);
	assign xf32[9] = f16_to_f32_bits(x_in[9]);
	assign xf32[10] = f16_to_f32_bits(x_in[10]);
	assign xf32[11] = f16_to_f32_bits(x_in[11]);
	assign xf32[12] = f16_to_f32_bits(x_in[12]);
	assign xf32[13] = f16_to_f32_bits(x_in[13]);
	assign xf32[14] = f16_to_f32_bits(x_in[14]);
	assign xf32[15] = f16_to_f32_bits(x_in[15]);
	assign xf32[16] = f16_to_f32_bits(x_in[16]);
	assign xf32[17] = f16_to_f32_bits(x_in[17]);
	assign xf32[18] = f16_to_f32_bits(x_in[18]);
	assign xf32[19] = f16_to_f32_bits(x_in[19]);
	assign xf32[20] = f16_to_f32_bits(x_in[20]);
	assign xf32[21] = f16_to_f32_bits(x_in[21]);
	assign xf32[22] = f16_to_f32_bits(x_in[22]);
	assign xf32[23] = f16_to_f32_bits(x_in[23]);
	assign xf32[24] = f16_to_f32_bits(x_in[24]);
	assign xf32[25] = f16_to_f32_bits(x_in[25]);
	assign xf32[26] = f16_to_f32_bits(x_in[26]);
	assign xf32[27] = f16_to_f32_bits(x_in[27]);
	assign xf32[28] = f16_to_f32_bits(x_in[28]);
	assign xf32[29] = f16_to_f32_bits(x_in[29]);
	assign xf32[30] = f16_to_f32_bits(x_in[30]);
	assign xf32[31] = f16_to_f32_bits(x_in[31]);

	// ---- per-lane quantize ------------------------------------------
	// Returns the FULL 8-bit `(uint8_t)m` value from the C reference, not
	// a clamped 4-bit nibble: `m` has an upper clamp to 15 but NO lower
	// clamp, so a very negative m (e.g. -128) becomes a uint8_t >= 16
	// (e.g. 128) whose upper bits then genuinely overlap with the
	// neighboring lane's nibble in the final `xi0 | (xi1 << 4)` pack --
	// ggml's actual, intentional behavior (docs/phase4-ggml-quant-
	// parity.md), replicated exactly rather than "cleaned up" to a safe
	// 4-bit value, which would silently diverge from the C reference and
	// ggml for any block containing an extreme (very negative after
	// scaling) lane.
	function automatic logic [7:0] quantize_lane(input logic [31:0] xv,
			input logic [63:0] id64);
		real			x_r;
		real			id_r_l;
		real			prod_r;
		logic	[31:0]	prod_f32;
		real			prod_exact_r;
		real			sum_r;
		logic	[31:0]	sum_f32;
		real			sum_exact_r;
		logic			out_of_range;
		int				qi32;
		logic	[7:0]	qi8;
		int				qi8_signed;
		int				m;

		x_r = $bitstoreal(f32_widen_to_f64(xv));
		id_r_l = $bitstoreal(id64);
		prod_r = x_r * id_r_l;
		prod_f32 = f64_narrow_to_f32_rne($realtobits(prod_r));
		prod_exact_r = $bitstoreal(f32_widen_to_f64(prod_f32));
		sum_r = prod_exact_r + 8.5;
		sum_f32 = f64_narrow_to_f32_rne($realtobits(sum_r));
		sum_exact_r = $bitstoreal(f32_widen_to_f64(sum_f32));
		out_of_range = (sum_f32[30:23] == 8'hFF)
			|| (sum_exact_r >= 2147483648.0)
			|| (sum_exact_r < -2147483648.0);
		if (out_of_range)
			qi32 = 32'h80000000;
		else
			qi32 = $rtoi(sum_exact_r); // truncates toward zero
		qi8 = qi32[7:0];
		qi8_signed = $signed(qi8);
		m = (qi8_signed > 15) ? 15 : qi8_signed;
		return (m[7:0]);
	endfunction

	logic	[7:0]	q_comb	[0:31];

	always_comb begin
		q_comb[0] = quantize_lane(xf32[0], id_f64_w);
		q_comb[1] = quantize_lane(xf32[1], id_f64_w);
		q_comb[2] = quantize_lane(xf32[2], id_f64_w);
		q_comb[3] = quantize_lane(xf32[3], id_f64_w);
		q_comb[4] = quantize_lane(xf32[4], id_f64_w);
		q_comb[5] = quantize_lane(xf32[5], id_f64_w);
		q_comb[6] = quantize_lane(xf32[6], id_f64_w);
		q_comb[7] = quantize_lane(xf32[7], id_f64_w);
		q_comb[8] = quantize_lane(xf32[8], id_f64_w);
		q_comb[9] = quantize_lane(xf32[9], id_f64_w);
		q_comb[10] = quantize_lane(xf32[10], id_f64_w);
		q_comb[11] = quantize_lane(xf32[11], id_f64_w);
		q_comb[12] = quantize_lane(xf32[12], id_f64_w);
		q_comb[13] = quantize_lane(xf32[13], id_f64_w);
		q_comb[14] = quantize_lane(xf32[14], id_f64_w);
		q_comb[15] = quantize_lane(xf32[15], id_f64_w);
		q_comb[16] = quantize_lane(xf32[16], id_f64_w);
		q_comb[17] = quantize_lane(xf32[17], id_f64_w);
		q_comb[18] = quantize_lane(xf32[18], id_f64_w);
		q_comb[19] = quantize_lane(xf32[19], id_f64_w);
		q_comb[20] = quantize_lane(xf32[20], id_f64_w);
		q_comb[21] = quantize_lane(xf32[21], id_f64_w);
		q_comb[22] = quantize_lane(xf32[22], id_f64_w);
		q_comb[23] = quantize_lane(xf32[23], id_f64_w);
		q_comb[24] = quantize_lane(xf32[24], id_f64_w);
		q_comb[25] = quantize_lane(xf32[25], id_f64_w);
		q_comb[26] = quantize_lane(xf32[26], id_f64_w);
		q_comb[27] = quantize_lane(xf32[27], id_f64_w);
		q_comb[28] = quantize_lane(xf32[28], id_f64_w);
		q_comb[29] = quantize_lane(xf32[29], id_f64_w);
		q_comb[30] = quantize_lane(xf32[30], id_f64_w);
		q_comb[31] = quantize_lane(xf32[31], id_f64_w);
	end

	// Pack: byte k (k=0..15) = q_comb[k] | (q_comb[k+16] << 4), matching
	// the C reference's `xi0 | (xi1 << 4)` exactly.
	logic	[7:0]	packed_bytes_comb	[0:15];

	always_comb
		for (int k = 0; k < 16; k++)
			packed_bytes_comb[k] = q_comb[k] | (q_comb[k + 16] << 4);

	logic	[143:0]	pipe	[0:DELAY - 1];
	logic	[143:0]	stage0_flat;

	// Byte layout matches the C reference exactly: scale FIRST (bytes
	// 0-1), then the 16 packed-nibble bytes (bytes 2-17) -- an earlier
	// version of this put the scale last, a real bug caught by the
	// vector test in rtl/tb/tb_q4_pack.sv (every byte correct, but
	// rotated by 2 positions).
	always_comb begin
		stage0_flat[7:0] = d_f16[7:0];
		stage0_flat[15:8] = d_f16[15:8];
		stage0_flat[23:16] = packed_bytes_comb[0];
		stage0_flat[31:24] = packed_bytes_comb[1];
		stage0_flat[39:32] = packed_bytes_comb[2];
		stage0_flat[47:40] = packed_bytes_comb[3];
		stage0_flat[55:48] = packed_bytes_comb[4];
		stage0_flat[63:56] = packed_bytes_comb[5];
		stage0_flat[71:64] = packed_bytes_comb[6];
		stage0_flat[79:72] = packed_bytes_comb[7];
		stage0_flat[87:80] = packed_bytes_comb[8];
		stage0_flat[95:88] = packed_bytes_comb[9];
		stage0_flat[103:96] = packed_bytes_comb[10];
		stage0_flat[111:104] = packed_bytes_comb[11];
		stage0_flat[119:112] = packed_bytes_comb[12];
		stage0_flat[127:120] = packed_bytes_comb[13];
		stage0_flat[135:128] = packed_bytes_comb[14];
		stage0_flat[143:136] = packed_bytes_comb[15];
	end

	always_ff @(posedge clk) begin
		pipe[0] <= stage0_flat;
		for (int k = DELAY - 1; k > 0; k--)
			pipe[k] <= pipe[k - 1];
	end

	valid_delay_line #(.DEPTH(DELAY)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out));

	assign packed_out = pipe[DELAY - 1];
endmodule
