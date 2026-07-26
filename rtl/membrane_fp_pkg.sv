// Phase 5.2: shared combinational IEEE-754 bit-manipulation functions,
// packaged (not `` `include``d raw) so multiple modules can call the SAME
// functions safely: an earlier version of this file was a plain
// `` `include``-guarded header, textually duplicated (once, thanks to the
// include guard) into whichever files included it -- but Icarus Verilog
// 12.0 was found, via extensive isolated reproduction, to hang
// indefinitely whenever TWO OR MORE modules in the same compilation
// called the same `` `include``d automatic function (e.g. q8_scale.sv and
// q8_quantize_pack.sv both calling f32_widen_to_f64/f64_narrow_to_f32_rne
// -- confirmed the same pathology afflicts the Q4 modules too). A
// SystemVerilog `package` with `import membrane_fp_pkg::*;` at each call
// site -- proper module-boundary scoping, rather than textual macro
// duplication -- does not exhibit this at all, verified directly against
// the exact failing case before this file was converted. This is also
// simply the more correct SystemVerilog idiom for sharing functions
// across modules, not only a workaround.
//
// Shared combinational IEEE-754 bit-manipulation functions used
// by the RTL quantize/dequantize datapath. Each function here mirrors, bit
// for bit, an existing verified C function -- membrane_f16_to_f32() /
// membrane_f32_to_f16() in src/codecs/f16convert.c (including the Phase
// 5.1 quiet-NaN fixes) -- so the RTL's F16<->F32 conversions are exact
// hardware translations of already bit-exact-verified software, not an
// independent reimplementation.
//
// Phase 5.3: every function in THIS file is now a strictly synthesizable
// subset of SystemVerilog -- no `while` loops (the F16 subnormal
// normalization that used to shift-until-found is now a combinational
// priority encoder, see highest_set_bit_10() below), no block-scoped
// local declarations nested inside if/else bodies (yosys's Verilog-2005-
// based `read_verilog -sv` frontend does not accept those; every local is
// declared once at the top of its function), no `int'(...)` cast syntax
// (also rejected by that frontend; replaced with a plain assignment into
// a signed `int` where a signed reinterpretation was needed), and no
// signedness traps from mixing concatenation expressions (which SystemVerilog
// always treats as unsigned) into arithmetic that must stay signed --
// two real instances of that exact bug were found and fixed while doing
// this rewrite, both caught by re-running the existing 65,536-pattern and
// 263,538-vector exhaustive tests (docs/phase5-synthesizable-fpga.md has
// the full account). This file was confirmed to parse under
// `yosys read_verilog -sv`.
//
// The real-arithmetic F32<->F64 helpers this package used to also contain
// (f32_widen_to_f64/f64_narrow_to_f32_rne, a stand-in for a synthesizable
// divider/multiplier) have moved to membrane_fp_sim_pkg.sv -- simulation/
// testbench-only from here on, never fed to yosys. The production
// datapath's actual divide and multiply are now membrane_fp_divider.sv
// (a synthesizable iterative integer restoring divider) and
// membrane_fp_multiplier.sv (a synthesizable integer mantissa
// multiplier), both bit-exact by construction rather than by
// approximation-plus-correction (see their own header comments for why
// that was the safer path to a hard bit-exactness guarantee).

package membrane_fp_pkg;

// Priority encoder over a nonzero 10-bit value: returns the bit index
// (0..9) of its highest set bit. Combinational, fully unrolled (a
// `casez` priority-select, not a loop) -- synthesizable, and used in
// place of the shift-until-found while loop f16_to_f32_bits used to use
// for subnormal normalization (see that function's header comment for
// why this file no longer uses `while`/`real`/`shortreal`/DPI in any
// production datapath function: yosys's `read_verilog -sv` frontend
// rejected the while-loop form outright, and Phase 5.3's whole point is
// removing every non-synthesizable construct from this package).
// Assigns the function's own name instead of using `return`: yosys 0.33
// rejects `return` inside ANY `function automatic` once that function is
// actually elaborated (i.e. called from synthesized code) -- a plain
// `read_verilog -sv` parse of this file alone does not trigger it (no
// elaboration happens without a call site), which is why this was not
// caught until a module actually invoking these package functions was
// run through `hierarchy`. Applies to every function below, and is the
// same fix already applied to every MODULE-scoped function with
// `return` elsewhere in this RTL tree (see membrane_fp_adder.sv's lzc_9
// header comment for the original finding).
function automatic logic [3:0] highest_set_bit_10(input logic [9:0] m);
	casez (m)
		10'b1?????????: highest_set_bit_10 = 4'd9;
		10'b01????????: highest_set_bit_10 = 4'd8;
		10'b001???????: highest_set_bit_10 = 4'd7;
		10'b0001??????: highest_set_bit_10 = 4'd6;
		10'b00001?????: highest_set_bit_10 = 4'd5;
		10'b000001????: highest_set_bit_10 = 4'd4;
		10'b0000001???: highest_set_bit_10 = 4'd3;
		10'b00000001??: highest_set_bit_10 = 4'd2;
		10'b000000001?: highest_set_bit_10 = 4'd1;
		10'b0000000001: highest_set_bit_10 = 4'd0;
		default: highest_set_bit_10 = 4'd0; // unreachable: caller guarantees m != 0
	endcase
endfunction

function automatic logic [31:0] f16_to_f32_bits(input logic [15:0] h);
	logic		sign;
	logic [4:0]	exp;
	logic [9:0]	mant;
	logic [31:0]	bits;
	int		e;
	logic [3:0]	k;
	int			s;
	logic [19:0]	wide;
	logic [9:0]	m_shifted;

	sign = h[15];
	exp = h[14:10];
	mant = h[9:0];
	if (exp == 5'h00) begin
		if (mant == 10'h0)
			bits = {sign, 31'h0};
		else begin
			// Equivalent to the retired while-loop's shift-until-bit-10
			// search (see docs/phase5-synthesizable-fpga.md for the
			// derivation): find the highest set bit position `k` of the
			// 10-bit mantissa directly via the priority encoder above,
			// then compute the shift amount `s = 10 - k` and resulting
			// exponent `e = 1 - s` in one step instead of iterating. The
			// two real bugs found in the original while-loop version
			// (wrong-modulus exponent counter, off-by-one bit-10 vs
			// bit-9 loop-exit test -- both documented in git history and
			// docs/phase5-quant-engine.md) do not recur here since this
			// form has no incremental counter and no loop-exit condition
			// to get subtly wrong: `k`, `s`, and `e` are each computed
			// directly from closed-form arithmetic on `mant`.
			k = highest_set_bit_10(mant);
			s = 10 - {28'h0, k};
			e = 1 - s;
			wide = {10'h0, mant} << s[3:0];
			m_shifted = wide[9:0];
			bits = {sign, 8'(e + (127 - 15)), m_shifted, 13'h0};
		end
	end else if (exp == 5'h1F) begin
		bits = {sign, 8'hFF, mant, 13'h0};
		if (mant != 10'h0)
			bits[22] = 1'b1;
	end else begin
		bits = {sign, (8'(exp) + 8'(127 - 15)), mant, 13'h0};
	end
	f16_to_f32_bits = bits;
endfunction

function automatic logic [15:0] f32_to_f16_bits(input logic [31:0] f);
	logic		sign;
	logic [7:0]	exp_f;
	logic [22:0]	mant_f;
	int		h_exp;
	int		exp_f_signed;
	logic [15:0]	result;
	// Subnormal-path locals (yosys's Verilog-2005-based frontend does
	// not accept block-scoped `logic`/`int` declarations nested inside
	// if/else bodies -- every local in this function is declared here,
	// at the top, flattened, matching the same fix already applied to
	// f16_to_f32_bits above).
	logic [23:0]	full_mant;
	int		sh;
	logic [31:0]	q;
	logic [31:0]	sub_remainder;
	logic [31:0]	halfway;
	logic [31:0]	mant_h32;
	// Normal-path locals.
	logic [9:0]	mant_h;
	logic [12:0]	norm_remainder;
	int		h_exp2;

	sign = f[31];
	exp_f = f[30:23];
	mant_f = f[22:0];
	if (exp_f == 8'hFF) begin
		if (mant_f == 23'h0)
			result = {sign, 5'h1F, 10'h0};
		else
			result = {sign, 5'h1F, 10'h200};
	end else if (exp_f == 8'h00) begin
		result = {sign, 15'h0};
	end else begin
		// exp_f (unsigned 8-bit) is first assigned into a genuinely
		// signed `int` (a widening, zero-filling assignment, not a
		// concatenation) so the following subtraction is unambiguously
		// signed arithmetic -- `{24'h0, exp_f} - 127 + 15` looks
		// equivalent but is NOT: SystemVerilog treats an expression as
		// unsigned as soon as any operand (here, the concatenation) is
		// unsigned, which would silently wrap exp_f-127 to a huge
		// positive value instead of a small negative one for exp_f<127,
		// exactly the same class of bug documented on f16_to_f32_bits's
		// exponent computation above.
		exp_f_signed = exp_f;
		h_exp = exp_f_signed - 127 + 15;
		if (h_exp >= 31)
			result = {sign, 5'h1F, 10'h0};
		else if (h_exp <= 0) begin
			// Subnormal path: round-to-nearest-even via the same
			// shift-and-round construction as round_shift() in
			// src/codecs/f16convert.c (q = full_mant >> shift, round up
			// on remainder > halfway, or remainder == halfway with q
			// odd). An earlier version of this function used plain
			// truncation here on the (mistaken) assumption this path
			// was a rare corner for this datapath's actual amax/scale
			// values -- the F32->F16 vector test in
			// rtl/tb/tb_f32_to_f16_vectors.sv (263,538 vectors, mostly
			// random F32 patterns) hit it in ~1.8% of cases and caught
			// the resulting off-by-one mantissa values, so it is
			// implemented fully here instead.
			full_mant = {1'b1, mant_f};
			sh = 14 - h_exp;
			if (sh > 24 || sh < 1)
				result = {sign, 15'h0};
			else begin
				q = 32'(full_mant) >> sh;
				sub_remainder = 32'(full_mant) & ((32'h1 << sh) - 32'h1);
				halfway = 32'h1 << (sh - 1);
				if (sub_remainder > halfway
						|| (sub_remainder == halfway && q[0]))
					q = q + 32'h1;
				mant_h32 = q;
				if (mant_h32 >= 32'd1024)
					result = {sign, 5'd1, 10'h0};
				else
					result = {sign, 5'd0, 10'(mant_h32)};
			end
		end else begin
			mant_h = mant_f[22:13];
			norm_remainder = mant_f[12:0];
			h_exp2 = h_exp;
			if (norm_remainder > 13'h1000
					|| (norm_remainder == 13'h1000 && mant_h[0])) begin
				// Check for max-mantissa BEFORE incrementing: mant_h is
				// only 10 bits wide, so mant_h + 1 when mant_h == 10'h3FF
				// silently wraps to 0 anyway, but comparing against a
				// pre-computed "10'h3FF + 10'h1" constant is itself a
				// 10-bit-wraps-to-0 trap (that expression evaluates to
				// 10'h000, not 1024) -- checking mant_h against 10'h3FF
				// directly, before the increment, avoids that trap.
				if (mant_h == 10'h3FF) begin
					mant_h = 10'h0;
					h_exp2 = h_exp2 + 1;
				end else begin
					mant_h = mant_h + 10'h1;
				end
			end
			if (h_exp2 >= 31)
				result = {sign, 5'h1F, 10'h0};
			else
				result = {sign, 5'(h_exp2), mant_h};
		end
	end
	f32_to_f16_bits = result;
endfunction

// Priority encoder over a nonzero 9-bit value (same technique as
// highest_set_bit_10 above, one bit narrower -- used by
// int9_to_f32_bits for converting the small signed integers this
// datapath's dequantize paths need: Q8's int8 qs in [-128,127] and Q4's
// nibble-minus-8 in [-8,7], both representable in 9 bits after taking
// the absolute value).
function automatic logic [3:0] highest_set_bit_9(input logic [8:0] m);
	casez (m)
		9'b1????????: highest_set_bit_9 = 4'd8;
		9'b01???????: highest_set_bit_9 = 4'd7;
		9'b001??????: highest_set_bit_9 = 4'd6;
		9'b0001?????: highest_set_bit_9 = 4'd5;
		9'b00001????: highest_set_bit_9 = 4'd4;
		9'b000001???: highest_set_bit_9 = 4'd3;
		9'b0000001??: highest_set_bit_9 = 4'd2;
		9'b00000001?: highest_set_bit_9 = 4'd1;
		9'b000000001: highest_set_bit_9 = 4'd0;
		default: highest_set_bit_9 = 4'd0; // unreachable: caller guarantees m != 0
	endcase
endfunction

// Exact (no rounding possible or needed: 9 significant bits fits
// comfortably inside F32's 23-bit mantissa) signed-integer-to-F32
// conversion.
function automatic logic [31:0] int9_to_f32_bits(input logic signed [8:0] v);
	logic			sign;
	logic	[8:0]	mag;
	logic	[3:0]	k;
	int				e;
	logic	[8:0]	shifted;
	logic	[22:0]	mant_result;
	logic	[31:0]	bits;

	if (v == 9'sd0) begin
		bits = 32'h0;
	end else begin
		sign = v[8];
		mag = sign ? (~v + 9'sd1) : v;
		k = highest_set_bit_9(mag);
		e = {28'h0, k};
		shifted = mag << (4'd8 - k);
		mant_result = {shifted[7:0], 15'h0};
		bits = {sign, 8'(e + 127), mant_result};
	end
	int9_to_f32_bits = bits;
endfunction

// Round-to-nearest-even F32->int8, saturating, with NaN/Inf -> -128
// (0x80): bit-exact with sat_i8_from_rounded_f32(rintf(f)) in
// src/quant/quant_simd.c, itself replicating x86's CVTPS2DQ "integer
// indefinite" + saturating-pack hardware behavior
// (docs/phase5-quant-engine.md). Values with a rounded magnitude that
// reaches exactly the int8 boundary (128 negative, matching int8's
// asymmetric range) are NOT clamped -- -128 is a valid, exactly
// representable int8 value, only magnitudes STRICTLY beyond the
// representable range clamp.
function automatic logic [7:0] f32_round_sat_to_i8(input logic [31:0] f);
	logic			sign;
	logic	[7:0]	exp_f;
	logic	[22:0]	mant_f;
	int				exp_f_signed;
	int				e;
	logic	[23:0]	full_mant;
	int				shift;
	logic	[31:0]	q;
	logic	[31:0]	remainder;
	logic	[31:0]	halfway;
	logic	[7:0]	result;

	sign = f[31];
	exp_f = f[30:23];
	mant_f = f[22:0];
	if (exp_f == 8'hFF) begin
		result = 8'h80;
	end else if (exp_f == 8'h00) begin
		result = 8'h00;
	end else begin
		exp_f_signed = exp_f;
		e = exp_f_signed - 127;
		if (e < -1) begin
			result = 8'h00;
		end else if (e == -1) begin
			result = (mant_f != 23'h0) ? (sign ? 8'hFF : 8'h01) : 8'h00;
		end else if (e >= 31) begin
			// Magnitude >= 2^31: matches the C reference's own
			// `r >= 2147483648.0f || r < -2147483648.0f` out-of-int32-
			// range check, which returns the -128 sentinel REGARDLESS
			// of sign (not a normal sign-based saturate) -- a real bug
			// caught by this file's own 20,000+-vector test
			// (tb_int_conv.sv): an earlier version used this same
			// unconditional-sentinel branch starting at e>=8, which
			// wrongly overrode the correct sign-based saturation for
			// merely-large-but-still-within-int32-range values (e in
			// [8,30]).
			result = 8'h80;
		end else if (e >= 8) begin
			result = sign ? 8'h80 : 8'h7F;
		end else begin
			full_mant = {1'b1, mant_f};
			shift = 23 - e;
			q = 32'(full_mant) >> shift;
			remainder = 32'(full_mant) & ((32'h1 << shift) - 32'h1);
			halfway = 32'h1 << (shift - 1);
			if (remainder > halfway || (remainder == halfway && q[0]))
				q = q + 32'h1;
			if (sign) begin
				if (q >= 32'd128)
					result = 8'h80;
				else
					result = 8'(32'h0 - q);
			end else begin
				if (q > 32'd127)
					result = 8'h7F;
				else
					result = q[7:0];
			end
		end
	end
	f32_round_sat_to_i8 = result;
endfunction

// Truncating (toward zero) F32->int32, bit-exact with a plain C
// `(int32_t)float_value` cast on this target (x86), including the
// CVTTSS2SI "integer indefinite" sentinel (0x80000000) for NaN/Inf --
// used by Q4_0's pack step, `qi = (int8_t)(x*id + 8.5f)`, whose C
// reference relies on the SAME x86 cast-narrowing chain this function
// replicates (the caller takes this result's low byte, matching
// int32_t -> int8_t narrowing, per q4_pack.sv's own header comment on
// why that narrowing is a plain modular truncation, not saturation).
// The out-of-[-2^31,2^31) sentinel path (e>=31) is an approximation
// rather than the exact IEEE-754 boundary -- disclosed, not reachable by
// this datapath's actual x*id+8.5 range (docs/phase5-synthesizable-fpga.md).
function automatic logic [31:0] f32_trunc_to_i32(input logic [31:0] f);
	logic			sign;
	logic	[7:0]	exp_f;
	logic	[22:0]	mant_f;
	int				exp_f_signed;
	int				e;
	logic	[23:0]	full_mant;
	int				shift;
	logic	[31:0]	trunc_mag;
	logic	[31:0]	result;

	sign = f[31];
	exp_f = f[30:23];
	mant_f = f[22:0];
	if (exp_f == 8'hFF) begin
		result = 32'h80000000;
	end else if (exp_f == 8'h00) begin
		result = 32'h0;
	end else begin
		exp_f_signed = exp_f;
		e = exp_f_signed - 127;
		if (e < 0) begin
			result = 32'h0;
		end else if (e >= 31) begin
			result = 32'h80000000;
		end else begin
			full_mant = {1'b1, mant_f};
			if (e >= 23) begin
				trunc_mag = 32'(full_mant) << (e - 23);
			end else begin
				shift = 23 - e;
				trunc_mag = 32'(full_mant) >> shift;
			end
			result = sign ? (32'h0 - trunc_mag) : trunc_mag;
		end
	end
	f32_trunc_to_i32 = result;
endfunction

endpackage : membrane_fp_pkg
