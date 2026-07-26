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
	parameter int MUL_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[511:0]	x_in_flat,
	input	logic	[15:0]	d_f16,
	input	logic	[31:0]	id_f32,
	output	logic			valid_out,
	output	logic	[143:0]	packed_out
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


	// Total pipeline depth: multiplier (MUL_DELAY) -> adder (MUL_DELAY).
	// The trunc/clamp/pack step downstream of the adder is plain
	// combinational logic (like q8_quantize_pack.sv's round/sat/pack),
	// not a further registered stage.
	localparam int DELAY = 2 * MUL_DELAY;

	logic	[31:0]	xf32	[0:31];

	assign xf32[0] = membrane_fp_pkg::f16_to_f32_bits(x_in[0]);
	assign xf32[1] = membrane_fp_pkg::f16_to_f32_bits(x_in[1]);
	assign xf32[2] = membrane_fp_pkg::f16_to_f32_bits(x_in[2]);
	assign xf32[3] = membrane_fp_pkg::f16_to_f32_bits(x_in[3]);
	assign xf32[4] = membrane_fp_pkg::f16_to_f32_bits(x_in[4]);
	assign xf32[5] = membrane_fp_pkg::f16_to_f32_bits(x_in[5]);
	assign xf32[6] = membrane_fp_pkg::f16_to_f32_bits(x_in[6]);
	assign xf32[7] = membrane_fp_pkg::f16_to_f32_bits(x_in[7]);
	assign xf32[8] = membrane_fp_pkg::f16_to_f32_bits(x_in[8]);
	assign xf32[9] = membrane_fp_pkg::f16_to_f32_bits(x_in[9]);
	assign xf32[10] = membrane_fp_pkg::f16_to_f32_bits(x_in[10]);
	assign xf32[11] = membrane_fp_pkg::f16_to_f32_bits(x_in[11]);
	assign xf32[12] = membrane_fp_pkg::f16_to_f32_bits(x_in[12]);
	assign xf32[13] = membrane_fp_pkg::f16_to_f32_bits(x_in[13]);
	assign xf32[14] = membrane_fp_pkg::f16_to_f32_bits(x_in[14]);
	assign xf32[15] = membrane_fp_pkg::f16_to_f32_bits(x_in[15]);
	assign xf32[16] = membrane_fp_pkg::f16_to_f32_bits(x_in[16]);
	assign xf32[17] = membrane_fp_pkg::f16_to_f32_bits(x_in[17]);
	assign xf32[18] = membrane_fp_pkg::f16_to_f32_bits(x_in[18]);
	assign xf32[19] = membrane_fp_pkg::f16_to_f32_bits(x_in[19]);
	assign xf32[20] = membrane_fp_pkg::f16_to_f32_bits(x_in[20]);
	assign xf32[21] = membrane_fp_pkg::f16_to_f32_bits(x_in[21]);
	assign xf32[22] = membrane_fp_pkg::f16_to_f32_bits(x_in[22]);
	assign xf32[23] = membrane_fp_pkg::f16_to_f32_bits(x_in[23]);
	assign xf32[24] = membrane_fp_pkg::f16_to_f32_bits(x_in[24]);
	assign xf32[25] = membrane_fp_pkg::f16_to_f32_bits(x_in[25]);
	assign xf32[26] = membrane_fp_pkg::f16_to_f32_bits(x_in[26]);
	assign xf32[27] = membrane_fp_pkg::f16_to_f32_bits(x_in[27]);
	assign xf32[28] = membrane_fp_pkg::f16_to_f32_bits(x_in[28]);
	assign xf32[29] = membrane_fp_pkg::f16_to_f32_bits(x_in[29]);
	assign xf32[30] = membrane_fp_pkg::f16_to_f32_bits(x_in[30]);
	assign xf32[31] = membrane_fp_pkg::f16_to_f32_bits(x_in[31]);

	// ---- per-lane quantize ------------------------------------------
	// x*id runs through membrane_fp_multiplier (exact 24x24 mantissa
	// product, single correctly-rounded result -- equivalent to the C
	// reference's "multiply in double, round once to f32" construction,
	// since a double has ample precision to hold the exact product of
	// two f32 values before that single rounding). The `+8.5f` step runs
	// through membrane_fp_adder the same way. f32_trunc_to_i32 (already
	// verified standalone, see membrane_fp_pkg.sv) then replicates the
	// C reference's truncating-toward-zero cast including its NaN/Inf/
	// out-of-range INT32_MIN sentinel.
	logic	[31:0]	prod_f32	[0:31];
	logic			mul_valid	[0:31];
	logic	[31:0]	sum_f32		[0:31];
	logic			add_valid	[0:31];

	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul0 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[0]), .b_in(id_f32), .valid_out(mul_valid[0]), .result_out(prod_f32[0]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[1]), .b_in(id_f32), .valid_out(mul_valid[1]), .result_out(prod_f32[1]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[2]), .b_in(id_f32), .valid_out(mul_valid[2]), .result_out(prod_f32[2]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul3 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[3]), .b_in(id_f32), .valid_out(mul_valid[3]), .result_out(prod_f32[3]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul4 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[4]), .b_in(id_f32), .valid_out(mul_valid[4]), .result_out(prod_f32[4]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul5 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[5]), .b_in(id_f32), .valid_out(mul_valid[5]), .result_out(prod_f32[5]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul6 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[6]), .b_in(id_f32), .valid_out(mul_valid[6]), .result_out(prod_f32[6]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul7 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[7]), .b_in(id_f32), .valid_out(mul_valid[7]), .result_out(prod_f32[7]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul8 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[8]), .b_in(id_f32), .valid_out(mul_valid[8]), .result_out(prod_f32[8]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul9 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[9]), .b_in(id_f32), .valid_out(mul_valid[9]), .result_out(prod_f32[9]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul10 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[10]), .b_in(id_f32), .valid_out(mul_valid[10]), .result_out(prod_f32[10]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul11 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[11]), .b_in(id_f32), .valid_out(mul_valid[11]), .result_out(prod_f32[11]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul12 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[12]), .b_in(id_f32), .valid_out(mul_valid[12]), .result_out(prod_f32[12]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul13 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[13]), .b_in(id_f32), .valid_out(mul_valid[13]), .result_out(prod_f32[13]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul14 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[14]), .b_in(id_f32), .valid_out(mul_valid[14]), .result_out(prod_f32[14]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul15 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[15]), .b_in(id_f32), .valid_out(mul_valid[15]), .result_out(prod_f32[15]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul16 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[16]), .b_in(id_f32), .valid_out(mul_valid[16]), .result_out(prod_f32[16]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul17 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[17]), .b_in(id_f32), .valid_out(mul_valid[17]), .result_out(prod_f32[17]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul18 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[18]), .b_in(id_f32), .valid_out(mul_valid[18]), .result_out(prod_f32[18]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul19 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[19]), .b_in(id_f32), .valid_out(mul_valid[19]), .result_out(prod_f32[19]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul20 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[20]), .b_in(id_f32), .valid_out(mul_valid[20]), .result_out(prod_f32[20]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul21 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[21]), .b_in(id_f32), .valid_out(mul_valid[21]), .result_out(prod_f32[21]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul22 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[22]), .b_in(id_f32), .valid_out(mul_valid[22]), .result_out(prod_f32[22]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul23 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[23]), .b_in(id_f32), .valid_out(mul_valid[23]), .result_out(prod_f32[23]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul24 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[24]), .b_in(id_f32), .valid_out(mul_valid[24]), .result_out(prod_f32[24]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul25 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[25]), .b_in(id_f32), .valid_out(mul_valid[25]), .result_out(prod_f32[25]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul26 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[26]), .b_in(id_f32), .valid_out(mul_valid[26]), .result_out(prod_f32[26]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul27 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[27]), .b_in(id_f32), .valid_out(mul_valid[27]), .result_out(prod_f32[27]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul28 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[28]), .b_in(id_f32), .valid_out(mul_valid[28]), .result_out(prod_f32[28]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul29 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[29]), .b_in(id_f32), .valid_out(mul_valid[29]), .result_out(prod_f32[29]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul30 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[30]), .b_in(id_f32), .valid_out(mul_valid[30]), .result_out(prod_f32[30]));
	membrane_fp_multiplier #(.DELAY(MUL_DELAY)) u_mul31 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(xf32[31]), .b_in(id_f32), .valid_out(mul_valid[31]), .result_out(prod_f32[31]));

	// +8.5f, constant 0x41080000, subtract_in tied low (pure add).
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add0 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[0]), .a_in(prod_f32[0]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[0]), .result_out(sum_f32[0]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add1 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[1]), .a_in(prod_f32[1]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[1]), .result_out(sum_f32[1]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add2 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[2]), .a_in(prod_f32[2]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[2]), .result_out(sum_f32[2]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add3 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[3]), .a_in(prod_f32[3]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[3]), .result_out(sum_f32[3]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add4 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[4]), .a_in(prod_f32[4]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[4]), .result_out(sum_f32[4]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add5 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[5]), .a_in(prod_f32[5]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[5]), .result_out(sum_f32[5]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add6 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[6]), .a_in(prod_f32[6]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[6]), .result_out(sum_f32[6]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add7 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[7]), .a_in(prod_f32[7]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[7]), .result_out(sum_f32[7]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add8 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[8]), .a_in(prod_f32[8]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[8]), .result_out(sum_f32[8]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add9 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[9]), .a_in(prod_f32[9]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[9]), .result_out(sum_f32[9]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add10 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[10]), .a_in(prod_f32[10]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[10]), .result_out(sum_f32[10]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add11 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[11]), .a_in(prod_f32[11]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[11]), .result_out(sum_f32[11]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add12 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[12]), .a_in(prod_f32[12]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[12]), .result_out(sum_f32[12]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add13 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[13]), .a_in(prod_f32[13]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[13]), .result_out(sum_f32[13]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add14 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[14]), .a_in(prod_f32[14]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[14]), .result_out(sum_f32[14]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add15 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[15]), .a_in(prod_f32[15]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[15]), .result_out(sum_f32[15]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add16 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[16]), .a_in(prod_f32[16]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[16]), .result_out(sum_f32[16]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add17 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[17]), .a_in(prod_f32[17]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[17]), .result_out(sum_f32[17]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add18 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[18]), .a_in(prod_f32[18]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[18]), .result_out(sum_f32[18]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add19 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[19]), .a_in(prod_f32[19]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[19]), .result_out(sum_f32[19]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add20 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[20]), .a_in(prod_f32[20]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[20]), .result_out(sum_f32[20]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add21 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[21]), .a_in(prod_f32[21]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[21]), .result_out(sum_f32[21]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add22 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[22]), .a_in(prod_f32[22]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[22]), .result_out(sum_f32[22]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add23 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[23]), .a_in(prod_f32[23]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[23]), .result_out(sum_f32[23]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add24 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[24]), .a_in(prod_f32[24]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[24]), .result_out(sum_f32[24]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add25 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[25]), .a_in(prod_f32[25]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[25]), .result_out(sum_f32[25]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add26 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[26]), .a_in(prod_f32[26]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[26]), .result_out(sum_f32[26]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add27 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[27]), .a_in(prod_f32[27]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[27]), .result_out(sum_f32[27]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add28 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[28]), .a_in(prod_f32[28]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[28]), .result_out(sum_f32[28]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add29 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[29]), .a_in(prod_f32[29]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[29]), .result_out(sum_f32[29]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add30 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[30]), .a_in(prod_f32[30]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[30]), .result_out(sum_f32[30]));
	membrane_fp_adder #(.DELAY(MUL_DELAY)) u_add31 (.clk(clk), .rst_n(rst_n), .valid_in(mul_valid[31]), .a_in(prod_f32[31]), .b_in(32'h41080000), .subtract_in(1'b0), .valid_out(add_valid[31]), .result_out(sum_f32[31]));

	// ---- per-lane truncate + upper-clamp-only ------------------------
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
	// Assigns the function's own name instead of using `return` --
	// yosys 0.33 rejects `return` inside module-scoped functions, see
	// membrane_fp_adder.sv's lzc_9 header comment for the full finding.
	function automatic logic [7:0] trunc_clamp_lane(input logic [31:0] sv);
		logic	[31:0]	qi32;
		logic	[7:0]	qi8;
		int				qi8_signed;
		int				m;

		qi32 = membrane_fp_pkg::f32_trunc_to_i32(sv);
		qi8 = qi32[7:0];
		qi8_signed = $signed(qi8);
		m = (qi8_signed > 15) ? 15 : qi8_signed;
		trunc_clamp_lane = m[7:0];
	endfunction

	logic	[7:0]	q_comb	[0:31];

	always_comb begin
		q_comb[0] = trunc_clamp_lane(sum_f32[0]);
		q_comb[1] = trunc_clamp_lane(sum_f32[1]);
		q_comb[2] = trunc_clamp_lane(sum_f32[2]);
		q_comb[3] = trunc_clamp_lane(sum_f32[3]);
		q_comb[4] = trunc_clamp_lane(sum_f32[4]);
		q_comb[5] = trunc_clamp_lane(sum_f32[5]);
		q_comb[6] = trunc_clamp_lane(sum_f32[6]);
		q_comb[7] = trunc_clamp_lane(sum_f32[7]);
		q_comb[8] = trunc_clamp_lane(sum_f32[8]);
		q_comb[9] = trunc_clamp_lane(sum_f32[9]);
		q_comb[10] = trunc_clamp_lane(sum_f32[10]);
		q_comb[11] = trunc_clamp_lane(sum_f32[11]);
		q_comb[12] = trunc_clamp_lane(sum_f32[12]);
		q_comb[13] = trunc_clamp_lane(sum_f32[13]);
		q_comb[14] = trunc_clamp_lane(sum_f32[14]);
		q_comb[15] = trunc_clamp_lane(sum_f32[15]);
		q_comb[16] = trunc_clamp_lane(sum_f32[16]);
		q_comb[17] = trunc_clamp_lane(sum_f32[17]);
		q_comb[18] = trunc_clamp_lane(sum_f32[18]);
		q_comb[19] = trunc_clamp_lane(sum_f32[19]);
		q_comb[20] = trunc_clamp_lane(sum_f32[20]);
		q_comb[21] = trunc_clamp_lane(sum_f32[21]);
		q_comb[22] = trunc_clamp_lane(sum_f32[22]);
		q_comb[23] = trunc_clamp_lane(sum_f32[23]);
		q_comb[24] = trunc_clamp_lane(sum_f32[24]);
		q_comb[25] = trunc_clamp_lane(sum_f32[25]);
		q_comb[26] = trunc_clamp_lane(sum_f32[26]);
		q_comb[27] = trunc_clamp_lane(sum_f32[27]);
		q_comb[28] = trunc_clamp_lane(sum_f32[28]);
		q_comb[29] = trunc_clamp_lane(sum_f32[29]);
		q_comb[30] = trunc_clamp_lane(sum_f32[30]);
		q_comb[31] = trunc_clamp_lane(sum_f32[31]);
	end

	// Pack: byte k (k=0..15) = q_comb[k] | (q_comb[k+16] << 4), matching
	// the C reference's `xi0 | (xi1 << 4)` exactly.
	logic	[7:0]	packed_bytes_comb	[0:15];

	always_comb
		for (int k = 0; k < 16; k++)
			packed_bytes_comb[k] = q_comb[k] | (q_comb[k + 16] << 4);

	// d_f16 must be delayed by the SAME DELAY (= 2*MUL_DELAY) cycles as
	// sum_f32 (one multiplier stage, one adder stage) before it reaches
	// the combinational pack logic below -- the same class of bug as
	// q8_quantize_pack.sv's d_pipe / q4_scale.sv's d_f32_pipe.
	logic	[15:0]	d_f16_pipe	[0:DELAY - 1];

	always_ff @(posedge clk) begin
		d_f16_pipe[0] <= d_f16;
		for (int k = DELAY - 1; k > 0; k--)
			d_f16_pipe[k] <= d_f16_pipe[k - 1];
	end

	assign valid_out = add_valid[0];

	// Byte layout matches the C reference exactly: scale FIRST (bytes
	// 0-1), then the 16 packed-nibble bytes (bytes 2-17) -- an earlier
	// version of this put the scale last, a real bug caught by the
	// vector test in rtl/tb/tb_q4_pack.sv (every byte correct, but
	// rotated by 2 positions).
	assign packed_out[7:0] = d_f16_pipe[DELAY - 1][7:0];
	assign packed_out[15:8] = d_f16_pipe[DELAY - 1][15:8];
	assign packed_out[23:16] = packed_bytes_comb[0];
	assign packed_out[31:24] = packed_bytes_comb[1];
	assign packed_out[39:32] = packed_bytes_comb[2];
	assign packed_out[47:40] = packed_bytes_comb[3];
	assign packed_out[55:48] = packed_bytes_comb[4];
	assign packed_out[63:56] = packed_bytes_comb[5];
	assign packed_out[71:64] = packed_bytes_comb[6];
	assign packed_out[79:72] = packed_bytes_comb[7];
	assign packed_out[87:80] = packed_bytes_comb[8];
	assign packed_out[95:88] = packed_bytes_comb[9];
	assign packed_out[103:96] = packed_bytes_comb[10];
	assign packed_out[111:104] = packed_bytes_comb[11];
	assign packed_out[119:112] = packed_bytes_comb[12];
	assign packed_out[127:120] = packed_bytes_comb[13];
	assign packed_out[135:128] = packed_bytes_comb[14];
	assign packed_out[143:136] = packed_bytes_comb[15];
endmodule
