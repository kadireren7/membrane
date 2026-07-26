// Phase 5.3: computes the Q8_0 block scale d = amax/127 (stored, as F16,
// in the packed block header) and the reciprocal id = 127/amax (fed to
// q8_quantize_pack's per-lane multiply), bit-exact with
// q8_0_quant_block_scalar's `d = amax/127.0f; id = amax!=0 ? 127.0f/amax
// : 0.0f;` (src/quant/quant_simd.c) -- note `id` is computed as a direct
// division, NOT as `1.0f/d`, which can round differently in the last bit
// (docs/phase4-ggml-quant-parity.md); this module replicates the direct-
// division form.
//
// Fully synthesizable: both divisions go through membrane_fp_divider.sv
// (a bit-exact integer-restoring-division-based FP32 divider, no
// `real`/`shortreal`/DPI anywhere in the production datapath from this
// phase on -- see that module's own header for how it stays exact). Two
// divider instances run in parallel (one for d, one for id) rather than
// time-multiplexing a single divider, since both share the same amax
// input and this keeps the module's own latency accounting simple; a
// production design targeting minimum area would likely share one
// divider across both operations at the cost of extra scheduling logic,
// not attempted here (disclosed in docs/phase5-synthesizable-fpga.md).
module q8_scale #(
	parameter int DIV_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[15:0]	amax_f16_in,
	output	logic			valid_out,
	output	logic	[15:0]	d_f16_out,
	output	logic	[31:0]	id_f32_out
);

	logic	[31:0]	amax_f32;
	logic	[31:0]	d_f32_raw;
	logic	[31:0]	id_f32_raw;
	logic			d_valid, id_valid;
	logic			amax_is_zero;
	logic	[31:0]	id_f32_final;

	assign amax_f32 = membrane_fp_pkg::f16_to_f32_bits(amax_f16_in);
	assign amax_is_zero = (amax_f32 == 32'h0);

	membrane_fp_divider #(.DELAY(DIV_DELAY)) u_div_d (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(amax_f32),
		.b_in(32'h42FE0000), .valid_out(d_valid), .result_out(d_f32_raw));

	membrane_fp_divider #(.DELAY(DIV_DELAY)) u_div_id (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(32'h42FE0000),
		.b_in(amax_f32), .valid_out(id_valid), .result_out(id_f32_raw));

	// `amax_is_zero` must be delayed by the SAME DIV_DELAY cycles as the
	// divider results it gates, so the explicit id=0-when-amax=0 mux
	// (matching the C reference's own ternary, not something the
	// divider itself special-cases beyond the general 127/0=+Inf IEEE
	// rule) lines up with the correct transaction.
	logic	[0:0]	zero_pipe	[0:DIV_DELAY - 1];

	always_ff @(posedge clk) begin
		zero_pipe[0] <= amax_is_zero;
		for (int k = DIV_DELAY - 1; k > 0; k--)
			zero_pipe[k] <= zero_pipe[k - 1];
	end

	assign id_f32_final = zero_pipe[DIV_DELAY - 1] ? 32'h0 : id_f32_raw;
	assign valid_out = d_valid && id_valid;
	assign d_f16_out = membrane_fp_pkg::f32_to_f16_bits(d_f32_raw);
	assign id_f32_out = id_f32_final;
endmodule
