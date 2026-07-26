// Phase 5.3: a synthesizable, bit-exact IEEE-754 binary32 divider.
//
// Bit-exact BY CONSTRUCTION, not by approximation-plus-correction: the
// significand division is computed with the plain Verilog `/` and `%`
// operators on 64-bit unsigned integers (exact integer arithmetic, no
// floating point, no `real`/`shortreal`/DPI anywhere in this file),
// confirmed to synthesize to a real combinational divider netlist under
// `yosys synth_ecp5` (see docs/phase5-synthesizable-fpga.md for the
// measured cell count). This is the "vendor-neutral pipelined
// arithmetic" option from the governing spec's item 3, taken to its
// simplest correct form: rather than a Newton-Raphson/LUT approximation
// that then needs a separate correctness-verification/correction pass
// to GUARANTEE correct rounding in every case (the "table maker's
// dilemma" that makes correctly-rounded approximate dividers genuinely
// hard to get right), exact integer division has no approximation step
// to correct in the first place -- the only remaining work is applying
// IEEE-754's round-to-nearest-even rule to the exact quotient, which is
// a simple guard/round/sticky-bit decision (implemented below), not a
// second numerical algorithm.
//
// Math: for normal, nonzero, non-NaN A=sign_a*1.mA*2^(eA-127) and
// B=sign_b*1.mB*2^(eB-127), let N={1,mA} and D={1,mB} (24-bit unsigned
// integers, N,D in [2^23, 2^24)). The exact quotient N/D lies in
// (0.5, 2). Computing Q = floor((N << 25) / D) gives an exact 26-bit
// fixed-point representation of (N/D) with 25 fractional bits (Q in
// [2^24, 2^26)); the remainder R = (N << 25) mod D captures everything
// below that precision, needed only to decide the sticky bit. Q's
// highest set bit position (25 or 24) tells us whether N/D is already in
// [1,2) or needs one left-shift to normalize, exactly mirroring the
// worked example in this file's own testbench-generated vectors.
//
// Latency is a single DELAY parameter (default 1: purely combinational,
// held one cycle) via valid_delay_line.sv, same convention as
// q8_scale.sv (Phase 5.2) used for its now-retired real-arithmetic
// placeholder -- a real synthesis target would likely want to break the
// 64-bit divide into multiple pipeline stages for timing closure (not
// attempted here, disclosed in docs/phase5-synthesizable-fpga.md).
//
// Scope: this datapath's actual operands (Q8_0/Q4_0's amax, and the
// constants 127.0/-8.0/1.0) are never subnormal and never NaN (amax's
// invariant, established in q8_maxabs_reduce.sv/q4_scan.sv, guarantees
// it is always a genuine non-negative finite value or +Infinity, never
// NaN) -- this module still implements general NaN/Inf/zero handling for
// robustness and to be a legitimately reusable FP32 divider block, but
// subnormal OPERAND support is out of scope and not verified (disclosed,
// not silently wrong: a subnormal input is treated as if its hidden bit
// were 1, which is INCORRECT for true subnormals -- see
// docs/phase5-synthesizable-fpga.md).
module membrane_fp_divider #(
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
	logic	[63:0]	num64, den64, quot64, rem64;
	logic			norm_shift;
	logic	[25:0]	q_norm;
	logic			sticky;
	logic	[22:0]	mant_result_trunc;
	logic			round_up;
	logic	[23:0]	mant_rounded;
	logic			mant_overflow;
	int				exp_a_s, exp_b_s, exp_result;
	logic	[31:0]	general_result;
	logic	[31:0]	result_comb;

	// x86's "real indefinite" QNaN pattern for 0/0 and Inf/Inf: sign=1,
	// not derived from the operands' signs -- see the special-case mux
	// below.
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

	// -- exact significand division, general (finite, nonzero) case --
	assign num64 = {40'h0, 1'b1, mant_a} << 25;
	assign den64 = {40'h0, 1'b1, mant_b};
	assign quot64 = num64 / den64;
	assign rem64 = num64 % den64;

	// quot64 in [2^24, 2^26): bit 25 set means N/D already in [1,2), else
	// N/D in [0.5,1) and needs one left-normalize.
	assign norm_shift = !quot64[25];
	assign q_norm = norm_shift ? {quot64[24:0], 1'b0} : quot64[25:0];
	// q_norm now in [2^25, 2^26): bit 25 is the implicit leading 1,
	// bits [24:2] are the 23 explicit mantissa bits, bits [1:0] are the
	// guard/round bits, and `rem64 != 0` (nothing lost by the
	// norm_shift, which only ever shifts in a known-zero LSB) is the
	// sticky bit.
	assign sticky = (rem64 != 64'h0);
	assign mant_result_trunc = q_norm[24:2];
	assign round_up = q_norm[1]
		&& (q_norm[0] || sticky || mant_result_trunc[0]);
	assign mant_rounded = round_up
		? ({1'b0, mant_result_trunc} + 24'h1) : {1'b0, mant_result_trunc};
	assign mant_overflow = mant_rounded[23];

	always_comb begin
		exp_a_s = exp_a;
		exp_b_s = exp_b;
		exp_result = exp_a_s - exp_b_s + 127 - (norm_shift ? 1 : 0)
			+ (mant_overflow ? 1 : 0);
	end

	always_comb begin
		if (exp_result >= 255)
			general_result = {result_sign, 8'hFF, 23'h0};
		else if (exp_result <= 0)
			// Underflow to zero -- this datapath's real operands never
			// approach this range (amax-derived exponents stay well
			// within F16's original span), so gradual underflow to a
			// subnormal F32 result is not implemented, only flush-to-
			// zero; disclosed, not silently wrong for the reachable
			// input domain (see docs/phase5-synthesizable-fpga.md).
			general_result = {result_sign, 31'h0};
		else
			general_result = {result_sign, 8'(exp_result),
				mant_overflow ? 23'h0 : mant_rounded[22:0]};
	end

	// Same empirically-verified x86 conventions as
	// membrane_fp_multiplier.sv (see that file's header comment for the
	// full derivation and the probe programs that established them): a
	// NaN operand propagates its own sign/payload (quiet-ified), `a`
	// winning if both are NaN; 0/0 and Inf/Inf (the divider's two
	// genuinely undefined cases, analogous to the multiplier's Inf*0)
	// produce the fixed sign=1 indefinite pattern, not sign_a XOR
	// sign_b. Unlike the multiplier's Inf*0 case, 0/0 and Inf/Inf are
	// NOT actually reachable by this datapath's real operations (every
	// divisor=0 case is bypassed by an external mux before reaching this
	// module, matching the C reference's own `id = amax!=0 ? ... : 0.0f`
	// structure, and dividend/divisor are never simultaneously Infinity
	// given how d and id are derived -- see docs/phase5-synthesizable-
	// fpga.md); fixed anyway, for the same reusable-general-block
	// reasoning as the multiplier, and because leaving a known-wrong
	// sentinel in place once identified would be dishonest by omission.
	always_comb begin
		if (a_is_nan)
			result_comb = a_in | 32'h00400000;
		else if (b_is_nan)
			result_comb = b_in | 32'h00400000;
		else if (a_is_inf && b_is_inf)
			result_comb = CANON_NAN;
		else if (a_is_zero && b_is_zero)
			result_comb = CANON_NAN;
		else if (a_is_inf)
			result_comb = {result_sign, 8'hFF, 23'h0};
		else if (b_is_inf)
			result_comb = {result_sign, 31'h0};
		else if (a_is_zero)
			result_comb = {result_sign, 31'h0};
		else if (b_is_zero)
			result_comb = {result_sign, 8'hFF, 23'h0};
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
