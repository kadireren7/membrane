// Phase 5.3: a synthesizable, bit-exact IEEE-754 binary32 multiplier.
//
// Exact by construction, same philosophy as membrane_fp_divider.sv:
// the 24x24 mantissa multiply is done with the plain Verilog `*`
// operator on unsigned integers (exact 48-bit product, no floating
// point, no `real`/`shortreal`/DPI), which synthesis tools map directly
// to DSP blocks or LUT-based multiplier trees -- unlike division,
// multiplication needs no iterative algorithm at all: the full 48-bit
// product already contains every bit needed for correct rounding, so
// this module is a normalize-and-round step around one multiply, not an
// approximation of any kind.
//
// Math: for normal, nonzero, non-NaN A and B, let N={1,mA}, M={1,mB}
// (24-bit unsigned, in [2^23,2^24)). The exact product P=N*M lies in
// [2^46, 2^48), so its highest set bit is at position 46 or 47. If bit
// 47 is set, P is already normalized (implicit leading 1 at bit 47); if
// only bit 46 is the highest, P needs one left-shift to normalize
// (equivalently, treat bit 46 as the leading 1). Either way, the 23
// explicit mantissa bits sit immediately below the leading 1, and
// everything below THAT is the exact guard/round/sticky information for
// round-to-nearest-even -- no precision is ever discarded before the
// rounding decision, unlike an approximate multiplier would.
//
// Scope: same as membrane_fp_divider.sv -- this datapath's actual
// operands are never subnormal, and NaN/Inf handling is implemented for
// robustness/completeness but not the input domain this datapath's own
// operands (Q8/Q4's amax, id, x) actually reach in practice.
module membrane_fp_multiplier #(
	parameter int DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[31:0]	a_in,
	input	logic	[31:0]	b_in,
	output	logic			valid_out,
	output	logic	[31:0]	result_out
);

	logic			sign_a, sign_b, result_sign;
	logic	[7:0]	exp_a, exp_b;
	logic	[22:0]	mant_a, mant_b;
	logic			a_is_nan, b_is_nan, a_is_inf, b_is_inf, a_is_zero, b_is_zero;
	logic	[47:0]	product;
	logic			norm_shift;
	logic	[47:0]	p_norm;
	logic	[22:0]	mant_result_trunc;
	logic			guard, round_bit, sticky;
	logic			round_up;
	logic	[23:0]	mant_rounded;
	logic			mant_overflow;
	int				exp_a_s, exp_b_s, exp_result;
	logic	[31:0]	general_result;
	logic	[31:0]	result_comb;

	// x86's "real indefinite" QNaN pattern for the Inf*0 invalid
	// operation: sign=1, not derived from the operands' signs -- see
	// the header comment on the special-case mux below.
	localparam logic [31:0] CANON_NAN = 32'hFFC00000;

	assign sign_a = a_in[31];
	assign exp_a = a_in[30:23];
	assign mant_a = a_in[22:0];
	assign sign_b = b_in[31];
	assign exp_b = b_in[30:23];
	assign mant_b = b_in[22:0];

	assign a_is_nan = (exp_a == 8'hFF) && (mant_a != 23'h0);
	assign b_is_nan = (exp_b == 8'hFF) && (mant_b != 23'h0);
	assign a_is_inf = (exp_a == 8'hFF) && (mant_a == 23'h0);
	assign b_is_inf = (exp_b == 8'hFF) && (mant_b == 23'h0);
	assign a_is_zero = (exp_a == 8'h00) && (mant_a == 23'h0);
	assign b_is_zero = (exp_b == 8'h00) && (mant_b == 23'h0);

	assign result_sign = sign_a ^ sign_b;

	// -- exact significand multiply, general (finite, nonzero) case --
	assign product = {1'b0, 1'b1, mant_a} * {1'b0, 1'b1, mant_b};

	assign norm_shift = !product[47];
	assign p_norm = norm_shift ? (product << 1) : product;
	// p_norm now has its implicit leading 1 at bit 47; bits [46:24] are
	// the 23 explicit mantissa bits; bit 23 is guard, bit 22 is round,
	// bits [21:0] OR'd together is sticky.
	assign mant_result_trunc = p_norm[46:24];
	assign guard = p_norm[23];
	assign round_bit = p_norm[22];
	assign sticky = (p_norm[21:0] != 22'h0);
	assign round_up = guard && (round_bit || sticky || mant_result_trunc[0]);
	assign mant_rounded = round_up
		? ({1'b0, mant_result_trunc} + 24'h1) : {1'b0, mant_result_trunc};
	assign mant_overflow = mant_rounded[23];

	always_comb begin
		exp_a_s = exp_a;
		exp_b_s = exp_b;
		exp_result = exp_a_s + exp_b_s - 127 + (norm_shift ? 0 : 1)
			+ (mant_overflow ? 1 : 0);
	end

	always_comb begin
		if (exp_result >= 255)
			general_result = {result_sign, 8'hFF, 23'h0};
		else if (exp_result <= 0)
			// Underflow to zero, same disclosed flush-to-zero (not
			// gradual-underflow-to-subnormal) limitation as
			// membrane_fp_divider.sv -- not reached by this datapath's
			// real operands (see docs/phase5-synthesizable-fpga.md).
			general_result = {result_sign, 31'h0};
		else
			general_result = {result_sign, 8'(exp_result),
				mant_overflow ? 23'h0 : mant_rounded[22:0]};
	end

	// Special-case priority and exact bit patterns verified empirically
	// against this machine's native x86 float32 multiply (the actual C
	// reference every other module in this repo is checked against),
	// not derived from the IEEE-754 standard's text alone -- the
	// standard leaves the sign/payload of most of these cases
	// implementation-defined, and an earlier version of this module
	// used a fixed canonical-NaN sentinel for EVERY NaN/invalid case,
	// which would have silently diverged from the real reference for
	// two real, reachable cases in this datapath:
	//   1. A NaN OPERAND (e.g. an input block element that is itself
	//      NaN) must propagate that operand's own sign and payload,
	//      quiet-ified (its quiet bit forced to 1), not collapse to a
	//      fixed pattern -- `a` wins if `a` is NaN, else `b`, matching
	//      `nan_probe2.c`'s empirical results (verified with a
	//      signaling-vs-quiet NaN pair and a both-quiet-NaN pair).
	//   2. Infinity * zero (the one genuinely undefined case, no NaN
	//      operand involved) produces a FIXED sign=1 pattern
	//      (0xFFC00000, x86's "real indefinite" QNaN) regardless of the
	//      operands' own signs -- NOT sign_a XOR sign_b, confirmed with
	//      all four sign combinations in `nan_probe.c`. This is exactly
	//      the case Q8_0/Q4_0 dequantize's `qs*d` reaches for any lane
	//      whose scale `d` is +Infinity (amax was Infinity) and whose
	//      quantized value happens to be exactly 0 -- and, unlike
	//      quantize's downstream saturation (which collapses any NaN to
	//      -128 regardless of sign, so the sign never mattered there),
	//      dequantize narrows this NaN straight to F16 output with
	//      f32_to_f16_bits, which DOES read the sign bit -- so getting
	//      this wrong would have been a real, user-visible bit-exactness
	//      bug, not a cosmetic one.
	always_comb begin
		if (a_is_nan)
			result_comb = a_in | 32'h00400000;
		else if (b_is_nan)
			result_comb = b_in | 32'h00400000;
		else if ((a_is_inf && b_is_zero) || (a_is_zero && b_is_inf))
			result_comb = CANON_NAN;
		else if (a_is_inf || b_is_inf)
			result_comb = {result_sign, 8'hFF, 23'h0};
		else if (a_is_zero || b_is_zero)
			result_comb = {result_sign, 31'h0};
		else
			result_comb = general_result;
	end

	logic	[31:0]	result_pipe	[0:DELAY - 1];

	always_ff @(posedge clk) begin
		result_pipe[0] <= result_comb;
		for (int k = DELAY - 1; k > 0; k--)
			result_pipe[k] <= result_pipe[k - 1];
	end

	valid_delay_line #(.DEPTH(DELAY)) u_valid (
		.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
		.valid_out(valid_out));

	assign result_out = result_pipe[DELAY - 1];
endmodule
