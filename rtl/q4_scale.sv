// Phase 5.3 (revised by EXP-FPGA-DIV-001): Q4_0 scale computation,
// factored into its own module for the same reason as q4_scan.sv -- see
// its header comment. d = mx/-8.0, id = d!=0 ? 1/d : 0, bit-exact with
// q4_0_quant_block_scalar (src/quant/quant_simd.c).
//
// `u_div_d` (`d = mx/-8.0`, a division by a compile-time constant that is
// an exact power of two) uses membrane_fp_scale_neg_pow2.sv, a
// specialized `/`-operator-free shortcut, instead of the general-purpose
// membrane_fp_divider.sv -- differential-tested bit-exact against the
// real divider for this exact operation (2,204,128 cases, 0 mismatches;
// see rtl/tb/tb_fp32_scale_neg_pow2.cpp and
// experiments/EXP-FPGA-DIV-001/phase-b1.md). `u_div_id` (`id = 1/d`, a
// genuinely variable divisor) uses membrane_fp_divider_radix4.sv, an
// exact multi-cycle iterative divider, instead of membrane_fp_divider.sv
// -- also differential-tested bit-exact (4,456,685 cases, 0 mismatches;
// see rtl/tb/tb_membrane_fp_divider_radix4.cpp and
// experiments/EXP-FPGA-DIV-001/phase-b4.md). Both replacements were
// selected because the general-purpose divider's own wide, single-cycle
// combinational significand division is this datapath's largest
// synthesized-cell-count and real-Fmax-risk contributor (see
// experiments/EXP-FPGA-DIV-001/baseline.md section 7); q8_scale.sv's own
// two membrane_fp_divider instances are untouched by this change (see
// this file's own "why q8_scale is unaffected" note below).
//
// ---- q4_scale is no longer a fixed-latency module ----
// membrane_fp_divider_radix4 is a multi-cycle, variable-latency,
// single-in-flight module (real in_valid/in_ready/out_valid/out_ready
// handshake, `busy` while a division is in flight) rather than a fixed-
// DELAY combinational pipe. This module's own `busy` output propagates
// that: a caller MUST NOT reassert `valid_in` while `busy` is high (this
// module has no internal input queue, matching the "single in-flight"
// area-first design point of membrane_fp_divider_radix4 itself) --
// membrane_quant_stream_top.sv's own Q4_0-encode issue gating
// (`q4enc_inflight`) guarantees this by construction for the production
// datapath; a bare standalone instantiation of this module is
// responsible for the same discipline itself, checked by the
// `` `ifndef SYNTHESIS `` assertion below.
//
// A single hold register for `d_f32_raw`/`d_is_zero` replaces the
// fixed-depth delay-matching array a constant-DELAY pipe would need --
// correct precisely because only one transaction is ever in flight
// through this module at a time (the same property
// membrane_fp_divider_radix4.sv itself guarantees for `u_div_id`, and
// which the caller's own single-in-flight discipline enforces for
// `u_div_d`/this whole module).
module q4_scale #(
	parameter int DIV_DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[31:0]	mx_f32,
	output	logic			valid_out,
	output	logic	[15:0]	d_f16_out,
	output	logic	[31:0]	id_f32_out,
	output	logic			busy
);

	logic	[31:0]	d_f32_raw;
	logic	[31:0]	id_f32_raw;
	logic			d_valid, id_valid, id_ready;
	logic			d_is_zero;

	membrane_fp_scale_neg_pow2 #(.SHIFT(3), .DELAY(DIV_DELAY)) u_div_d (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a_in(mx_f32),
		.valid_out(d_valid), .result_out(d_f32_raw));

	assign d_is_zero = (d_f32_raw == 32'h0);

	// `id = 1/d`, the one remaining variable-divisor operation. out_ready
	// tied high: nothing downstream (q4_pack.sv) ever applies its own
	// backpressure, so the divider's result is always consumed the
	// instant it is ready.
	membrane_fp_divider_radix4 u_div_id (
		.clk(clk), .rst_n(rst_n),
		.in_valid(d_valid), .in_ready(id_ready),
		.numerator(32'h3F800000), .denominator(d_f32_raw),
		.out_valid(id_valid), .out_ready(1'b1),
		.quotient(id_f32_raw), .busy(busy));

`ifndef SYNTHESIS
	// d_valid only ever pulses once per Q4_0 encode transaction, and the
	// top-level's issue gating never starts a new one before the
	// previous fully retires -- so u_div_id must always be ready
	// (in_ready) whenever d_valid actually fires. A violation here would
	// mean either that gating discipline broke, or this module was
	// instantiated somewhere without it.
	always_ff @(posedge clk)
		if (rst_n && d_valid)
			assert (id_ready)
				else $error("q4_scale: u_div_id not ready when d_valid pulsed -- single in-flight discipline violated");
`endif

	logic	[31:0]	d_f32_hold;
	logic			zero_hold;

	always_ff @(posedge clk) begin
		if (d_valid) begin
			d_f32_hold <= d_f32_raw;
			zero_hold <= d_is_zero;
		end
	end

	assign valid_out = id_valid;
	assign d_f16_out = membrane_fp_pkg::f32_to_f16_bits(d_f32_hold);
	assign id_f32_out = zero_hold ? 32'h0 : id_f32_raw;
endmodule
