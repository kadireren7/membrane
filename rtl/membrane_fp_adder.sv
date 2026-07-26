// Phase 5.3: a synthesizable, bit-exact IEEE-754 binary32 adder
// (also used for subtraction: b_in's sign is XORed with subtract_in
// before the core add logic runs, the standard "negate and add" trick).
// Needed for Q4_0's pack step (`x*id + 8.5f`), which the retired
// real-arithmetic placeholder computed with two chained real-typed
// operations; this module handles the "+8.5f" half.
//
// Exact by construction: mantissas are aligned via an integer right
// shift (with guard/round/sticky preservation, same technique as
// membrane_fp_pkg.sv's F16 subnormal rounding) and added/subtracted with
// plain integer arithmetic, then normalized and rounded to nearest-even
// -- no approximation, no `real`/`shortreal`/DPI.
module membrane_fp_adder #(
	parameter int DELAY = 1
) (
	input	logic			clk,
	input	logic			rst_n,
	input	logic			valid_in,
	input	logic	[31:0]	a_in,
	input	logic	[31:0]	b_in,
	input	logic			subtract_in,
	output	logic			valid_out,
	output	logic	[31:0]	result_out
);

	logic			sign_a, sign_b;
	logic	[7:0]	exp_a, exp_b;
	logic	[22:0]	mant_a, mant_b;
	logic			a_is_nan, b_is_nan, a_is_inf, b_is_inf, a_is_zero, b_is_zero;

	assign sign_a = a_in[31];
	assign exp_a = a_in[30:23];
	assign mant_a = a_in[22:0];
	assign sign_b = b_in[31] ^ subtract_in;
	assign exp_b = b_in[30:23];
	assign mant_b = b_in[22:0];

	assign a_is_nan = (exp_a == 8'hFF) && (mant_a != 23'h0);
	assign b_is_nan = (exp_b == 8'hFF) && (mant_b != 23'h0);
	assign a_is_inf = (exp_a == 8'hFF) && (mant_a == 23'h0);
	assign b_is_inf = (exp_b == 8'hFF) && (mant_b == 23'h0);
	assign a_is_zero = (exp_a == 8'h00) && (mant_a == 23'h0);
	assign b_is_zero = (exp_b == 8'h00) && (mant_b == 23'h0);

	// -- general (finite, both nonzero) case --
	logic	[23:0]	full_a, full_b;
	int				exp_a_s, exp_b_s, exp_diff;
	logic	[7:0]	exp_diff_mag;
	logic			a_ge_b_exp;
	// Wide aligned significands: 24-bit value plus room to shift right
	// by up to 255 (worst-case exponent difference) without losing the
	// sticky bit -- 24 + 8 guard bits is enough for any shift we
	// actually normalize on (anything shifted further right than the
	// register width is entirely captured by the sticky OR below).
	logic	[31:0]	big_full, small_full;
	logic	[31:0]	small_shifted;
	logic			shift_sticky;
	int				big_exp;
	logic			big_sign, small_sign;
	logic	[32:0]	sum_mag;	// 33 bits: room for a carry out of a 32-bit add
	logic			result_sign_pre;
	logic	[32:0]	mag_a_aligned, mag_b_aligned;
	logic			same_sign;
	logic			a_ge_b_mag;

	always_comb begin
		full_a = {1'b1, mant_a};
		full_b = {1'b1, mant_b};
		exp_a_s = exp_a;
		exp_b_s = exp_b;
		exp_diff = exp_a_s - exp_b_s;
		a_ge_b_exp = (exp_diff >= 0);
		exp_diff_mag = a_ge_b_exp ? 8'(exp_diff) : 8'(-exp_diff);
		// Which operand has the larger MAGNITUDE, not just the larger
		// exponent: when exponents are equal, the exponent alone doesn't
		// decide it -- the mantissas must be compared too. Picking "a" as
		// the big operand whenever exp_diff>=0 (the previous logic) is
		// wrong exactly when exponents tie and b's mantissa is larger
		// (e.g. a=-8.0 vs b=+8.5, both exponent 130): the later
		// same-sign/different-sign subtraction assumed big_full is always
		// >= small_shifted and underflowed the unsigned 33-bit subtract,
		// producing a wildly wrong result -- caught by q4_pack.sv's
		// 20,000-block regression test (the `x*id + 8.5f` step), not by
		// this module's own standalone random-vector test, since it needs
		// the specific combination of equal exponents with unequal
		// mantissas to trigger.
		a_ge_b_mag = (exp_diff > 0) || ((exp_diff == 0) && (full_a >= full_b));

		if (a_ge_b_mag) begin
			big_full = {full_a, 8'h0};
			small_full = {full_b, 8'h0};
			big_exp = exp_a_s;
			big_sign = sign_a;
			small_sign = sign_b;
		end else begin
			big_full = {full_b, 8'h0};
			small_full = {full_a, 8'h0};
			big_exp = exp_b_s;
			big_sign = sign_b;
			small_sign = sign_a;
		end

		if (exp_diff_mag >= 8'd32) begin
			small_shifted = 32'h0;
			shift_sticky = (small_full != 32'h0);
		end else if (exp_diff_mag == 8'd0) begin
			small_shifted = small_full;
			shift_sticky = 1'b0;
		end else begin
			small_shifted = small_full >> exp_diff_mag;
			shift_sticky = ((small_full
				& ((32'h1 << exp_diff_mag) - 32'h1)) != 32'h0);
		end

		same_sign = (big_sign == small_sign);
		mag_a_aligned = {1'b0, big_full};
		mag_b_aligned = {1'b0, small_shifted | {31'h0, shift_sticky}};

		if (same_sign) begin
			sum_mag = mag_a_aligned + mag_b_aligned;
			result_sign_pre = big_sign;
		end else begin
			// big_full's magnitude is always >= small_shifted's (since
			// big_full carries the larger-or-equal exponent and both are
			// normalized 24-bit significands before alignment), so this
			// subtraction never underflows.
			sum_mag = mag_a_aligned - mag_b_aligned;
			result_sign_pre = big_sign;
		end
	end

	// Normalize: sum_mag's integer significand (ignoring the 8
	// fractional/guard bits appended above) is in bits [39:8] of a
	// conceptual 40-bit register; sum_mag here is 33 bits covering the
	// same-sign carry-out case (bit 32) down through the aligned
	// fractional bits. Same-sign addition can carry out by exactly one
	// bit; different-sign subtraction can cancel down to a much smaller
	// magnitude needing a multi-bit renormalizing left shift.
	logic	[3:0]	lead_zero;
	logic	[32:0]	normalized;
	int				exp_result_pre;
	logic	[22:0]	mant_result_trunc;
	logic			guard, round_bit, sticky2;
	logic			round_up;
	logic	[23:0]	mant_rounded;
	logic			mant_overflow;
	logic	[31:0]	general_result;
	logic			result_is_zero;

	// Assigns the function's own name (the implicit return variable)
	// rather than using `return` -- yosys 0.33's Verilog-2005-based
	// frontend was found, during synthesis bring-up, to reject ANY
	// `return` statement inside a MODULE-scoped `function automatic`
	// (casez-based or plain if-based alike; confirmed via a minimal
	// standalone repro), even though the identical `return`-based
	// pattern works fine in membrane_fp_pkg.sv's PACKAGE-scoped
	// functions -- a real, previously undiagnosed root cause of most of
	// this project's modules failing yosys synthesis (only
	// stream_fifo.sv/valid_delay_line.sv, which have no such functions,
	// synthesized cleanly before this fix). This is the same class of
	// fix applied throughout this file's sibling modules (q4_scan.sv,
	// q8_maxabs_reduce.sv, q4_pack.sv, q4_unpack.sv,
	// membrane_quant_stream_top.sv) -- see docs/phase5-synthesizable-
	// fpga.md for the full account.
	function automatic logic [3:0] lzc_9(input logic [8:0] v);
		casez (v)
			9'b1????????: lzc_9 = 4'd0;
			9'b01???????: lzc_9 = 4'd1;
			9'b001??????: lzc_9 = 4'd2;
			9'b0001?????: lzc_9 = 4'd3;
			9'b00001????: lzc_9 = 4'd4;
			9'b000001???: lzc_9 = 4'd5;
			9'b0000001??: lzc_9 = 4'd6;
			9'b00000001?: lzc_9 = 4'd7;
			9'b000000001: lzc_9 = 4'd8;
			default: lzc_9 = 4'd9; // v == 0
		endcase
	endfunction

	always_comb begin
		result_is_zero = (sum_mag == 33'h0);
		if (sum_mag[32]) begin
			// same-sign carry-out: shift right by 1, exponent+1.
			normalized = sum_mag >> 1;
			exp_result_pre = big_exp + 1;
			lead_zero = 4'd0;
		end else if (sum_mag[31]) begin
			normalized = sum_mag;
			exp_result_pre = big_exp;
			lead_zero = 4'd0;
		end else begin
			// Cancellation from different-sign subtraction: find how far
			// left we must shift bits [31:23] (the region containing the
			// normalized-or-shrunk significand) to restore a leading 1
			// at bit 31. lzc_9 inspects the top 9 bits, sufficient for
			// this datapath's actual exponent differences (Q4's `x*id`
			// and the constant 8.5 never differ in exponent by more than
			// a few tens at most for reachable inputs, so cancellation
			// deeper than 8 bits is not exercised -- disclosed in
			// docs/phase5-synthesizable-fpga.md, matching the same
			// scope discipline as the divider/multiplier's subnormal
			// disclosure).
			lead_zero = lzc_9(sum_mag[31:23]);
			normalized = sum_mag << lead_zero;
			exp_result_pre = big_exp - {28'h0, lead_zero};
		end

		mant_result_trunc = normalized[30:8];
		guard = normalized[7];
		round_bit = normalized[6];
		sticky2 = (normalized[5:0] != 6'h0);
		round_up = guard && (round_bit || sticky2 || mant_result_trunc[0]);
		mant_rounded = round_up
			? ({1'b0, mant_result_trunc} + 24'h1)
			: {1'b0, mant_result_trunc};
		mant_overflow = mant_rounded[23];

		if (result_is_zero)
			general_result = 32'h0; // +0, matches RNE convention for exact cancellation
		else if ((exp_result_pre + (mant_overflow ? 1 : 0)) >= 255)
			general_result = {result_sign_pre, 8'hFF, 23'h0};
		else if ((exp_result_pre + (mant_overflow ? 1 : 0)) <= 0)
			general_result = {result_sign_pre, 31'h0};
		else
			general_result = {result_sign_pre,
				8'(exp_result_pre + (mant_overflow ? 1 : 0)),
				mant_overflow ? 23'h0 : mant_rounded[22:0]};
	end

	logic	[31:0]	result_comb;
	localparam logic [31:0] CANON_NAN = 32'hFFC00000;

	always_comb begin
		if (a_is_nan)
			result_comb = a_in | 32'h00400000;
		else if (b_is_nan)
			result_comb = {sign_b, b_in[30:0]} | 32'h00400000;
		else if (a_is_inf && b_is_inf && (sign_a != sign_b))
			result_comb = CANON_NAN;
		else if (a_is_inf)
			result_comb = {sign_a, 8'hFF, 23'h0};
		else if (b_is_inf)
			result_comb = {sign_b, 8'hFF, 23'h0};
		else if (a_is_zero && b_is_zero)
			result_comb = (sign_a && sign_b) ? {1'b1, 31'h0} : 32'h0;
		else if (a_is_zero)
			result_comb = {sign_b, b_in[30:0]};
		else if (b_is_zero)
			result_comb = a_in;
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
