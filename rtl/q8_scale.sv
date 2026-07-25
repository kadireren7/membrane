// Phase 5.2: computes the Q8_0 block scale d = amax/127 (stored, as F16,
// in the packed block header) and the reciprocal id = 127/amax (fed to
// q8_quantize_pack's per-lane multiply), bit-exact with
// q8_0_quant_block_scalar's `d = amax/127.0f; id = amax!=0 ? 127.0f/amax
// : 0.0f;` (src/quant/quant_simd.c) -- note `id` is computed as a direct
// division, NOT as `1.0f/d`, which can round differently in the last bit
// (docs/phase4-ggml-quant-parity.md); this module replicates the direct-
// division form.
//
// The division itself is NOT synthesizable as written: it widens the F32
// operands to F64, uses the simulator's native `real` (double) divide via
// $bitstoreal/$realtobits, and narrows the result back to F32 with
// round-to-nearest-even (rtl/membrane_fp_functions.svh's
// f32_widen_to_f64/f64_narrow_to_f32_rne, verified to reproduce
// correctly-rounded IEEE-754 float32 division in
// rtl/tb/tb_f32_div_vectors.sv, 100,000 vectors). `real`/$bitstoreal/
// $realtobits are simulation-only SystemVerilog constructs with no
// synthesis mapping in any tool used in this phase (Yosys does not
// support them) -- a synthesizable version needs a vendor or custom
// pipelined float32 divider IP, out of scope for this streaming-datapath-
// first prototype (see docs/phase5-fpga-streaming.md for the disclosure
// this implies for synthesis results).
//
// Latency is configurable via the DELAY parameter (default 10, matching
// tools/membrane-hw-sim's default --scale-lat) via a separate
// valid_delay_line instance; the division itself is computed
// combinationally and simply held for DELAY cycles as a placeholder for
// where a real pipelined divider IP's latency would go.
module q8_scale #(
	parameter int DELAY = 10
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[15:0]	amax_f16_in,
	output	logic			valid_out,
	output	logic	[15:0]	d_f16_out,
	output	logic	[31:0]	id_f32_out
);
	import membrane_fp_pkg::*;

	logic	[31:0]	amax_f32;
	logic	[63:0]	amax_f64;
	logic	[63:0]	c127_f64;
	real			amax_r;
	real			d_r;
	real			id_r;
	logic	[31:0]	d_f32;
	logic	[31:0]	id_f32;
	logic	[15:0]	d_f16;

	assign amax_f32 = f16_to_f32_bits(amax_f16_in);
	assign amax_f64 = f32_widen_to_f64(amax_f32);
	assign c127_f64 = f32_widen_to_f64(32'h42FE0000);

	always_comb begin
		amax_r = $bitstoreal(amax_f64);
		d_r = amax_r / $bitstoreal(c127_f64);
		id_r = (amax_r != 0.0) ? ($bitstoreal(c127_f64) / amax_r) : 0.0;
		d_f32 = f64_narrow_to_f32_rne($realtobits(d_r));
		id_f32 = (amax_r != 0.0) ? f64_narrow_to_f32_rne($realtobits(id_r))
			: 32'h00000000;
		d_f16 = f32_to_f16_bits(d_f32);
	end

	valid_delay_line #(.DEPTH(DELAY)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out));

	// Data is held in plain registers delayed by DELAY-1 additional
	// cycles beyond the combinational compute above, so d_f16_out/
	// id_f32_out become valid on the same cycle as valid_out.
	logic	[15:0]	d_pipe	[0:DELAY - 1];
	logic	[31:0]	id_pipe	[0:DELAY - 1];

	always_ff @(posedge clk) begin
		d_pipe[0] <= d_f16;
		id_pipe[0] <= id_f32;
		for (int k = DELAY - 1; k > 0; k--) begin
			d_pipe[k] <= d_pipe[k - 1];
			id_pipe[k] <= id_pipe[k - 1];
		end
	end

	assign d_f16_out = d_pipe[DELAY - 1];
	assign id_f32_out = id_pipe[DELAY - 1];
endmodule
