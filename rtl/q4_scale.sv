// Phase 5.3: Q4_0 scale computation, factored into its own module for the
// same reason as q4_scan.sv -- see its header comment. d = mx/-8.0,
// id = d!=0 ? 1/d : 0, bit-exact with q4_0_quant_block_scalar
// (src/quant/quant_simd.c).
//
// Fully synthesizable: both divisions go through membrane_fp_divider.sv
// (see that module's header for how it stays exact). `mx/-8.0` is
// mathematically an exact power-of-two division (no rounding is ever
// needed), but is still routed through the same general divider rather
// than a specialized exponent-subtract shortcut, for the same reason
// q8_scale.sv uses one general divider for both of its operations: one
// verified-correct building block reused everywhere is a smaller,
// easier-to-trust design than several different specialized paths, and
// the general divider produces the exactly-correct result for a power-
// of-two divisor by construction anyway (integer division is exact
// regardless of the divisor's value).
module q4_scale #(
	parameter int DIV_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[31:0]	mx_f32,
	output	logic			valid_out,
	output	logic	[15:0]	d_f16_out,
	output	logic	[31:0]	id_f32_out
);

	logic	[31:0]	d_f32_raw;
	logic	[31:0]	id_f32_raw;
	logic			d_valid, id_valid;
	logic	[0:0]	zero_pipe	[0:DIV_DELAY - 1];
	logic	[31:0]	d_f32_pipe	[0:DIV_DELAY - 1];
	logic			d_is_zero;

	membrane_fp_divider #(.DELAY(DIV_DELAY)) u_div_d (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(mx_f32),
		.b_in(32'hC1000000), .valid_out(d_valid), .result_out(d_f32_raw));

	assign d_is_zero = (d_f32_raw == 32'h0);

	membrane_fp_divider #(.DELAY(DIV_DELAY)) u_div_id (
		.clk(clk), .rst_n(rst_n), .valid_in(d_valid), .a_in(32'h3F800000),
		.b_in(d_f32_raw), .valid_out(id_valid), .result_out(id_f32_raw));

	// id's divider is chained one stage AFTER d's (it needs d_f32_raw as
	// an input), so both the zero-gating flag AND d_f32_raw itself must
	// be delay-matched through BOTH dividers' combined latency, not just
	// one DIV_DELAY -- reading d_f32_raw directly here (undelayed) would
	// read a DIFFERENT, newer transaction's result than the one id_valid
	// is reporting completion for in a real back-to-back streaming
	// pipeline, the same class of bug as q8_quantize_pack.sv's qs_pipe
	// (Phase 5.2), caught here during design review rather than by a
	// failing test.
	always_ff @(posedge clk) begin
		zero_pipe[0] <= d_is_zero;
		d_f32_pipe[0] <= d_f32_raw;
		for (int k = DIV_DELAY - 1; k > 0; k--) begin
			zero_pipe[k] <= zero_pipe[k - 1];
			d_f32_pipe[k] <= d_f32_pipe[k - 1];
		end
	end

	assign valid_out = id_valid;
	assign d_f16_out = membrane_fp_pkg::f32_to_f16_bits(d_f32_pipe[DIV_DELAY - 1]);
	assign id_f32_out = zero_pipe[DIV_DELAY - 1] ? 32'h0 : id_f32_raw;
endmodule
