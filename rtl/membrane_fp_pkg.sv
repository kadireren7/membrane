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
// independent reimplementation. All four functions here are pure
// combinational bit manipulation (shifts, comparisons, adds) and are
// synthesizable.
//
// f32_widen_to_f64() and f64_narrow_to_f32_rne() are NOT modeled on any
// existing MEMBRANE C function -- they exist only so q8_scale.sv can
// compute a correctly-rounded float32 division using the simulator's
// native double-precision `real` arithmetic (via $bitstoreal/$realtobits)
// as a stand-in for a synthesizable float32 divider, which this prototype
// does not implement (see rtl/q8_scale.sv's own header comment for why,
// and docs/phase5-fpga-streaming.md for the synthesizability disclosure).
// Widening float32->float64 is exact (no rounding: exponent rebias
// 127->1023, mantissa left-shift 23->52 with zero-fill); narrowing
// float64->float32 uses round-to-nearest-even on the dropped 29 mantissa
// bits, which is the single rounding step that makes "widen, divide in
// double, narrow" equivalent to a correctly-rounded float32 divide for a
// single operation (double's 52 mantissa bits are more than double
// float32's 23, so no double-rounding error is introduced).

package membrane_fp_pkg;

function automatic logic [31:0] f16_to_f32_bits(input logic [15:0] h);
	logic		sign;
	logic [4:0]	exp;
	logic [9:0]	mant;
	logic [31:0]	bits;
	logic [10:0]	m;
	int		e;

	sign = h[15];
	exp = h[14:10];
	mant = h[9:0];
	if (exp == 5'h00) begin
		if (mant == 10'h0)
			bits = {sign, 31'h0};
		else begin
			// `e` must be a signed type wide enough to go negative
			// without wrapping (up to 9 left-shifts for a 10-bit
			// subnormal mantissa, so e can reach -8) -- the equivalent
			// C code (membrane_f16_to_f32) uses uint32_t and relies on
			// 32-bit modular wraparound cancelling out in the following
			// `+ (127-15)`, which only works because BOTH the
			// decrement and the addition happen in that SAME 32-bit
			// modulus. A 5-bit unsigned counter here would wrap in a
			// different (mod 32) modulus and silently produce the wrong
			// exponent -- this was a real bug, found by the exhaustive
			// 65536-pattern RTL parity test in
			// rtl/tb/tb_f16_to_f32_exhaustive.sv, fixed by computing the
			// true signed exponent directly instead.
			//
			// `m` must be 11 bits wide and the loop must test bit 10
			// (mirroring the C code's `mant & 0x400`), not bit 9 --
			// the C loop shifts until the found bit lands ONE position
			// past the mantissa's own top bit (bit 9), then keeps ALL
			// 10 low bits (`mant &= 0x3FF`) as the F32 mantissa's top
			// 10 bits (`mant << 13`). Testing bit 9 instead exits the
			// loop one shift too early and then only kept 9 of the 10
			// mantissa bits -- a second real bug caught by the same
			// exhaustive test, alongside the exponent-width bug above.
			e = 1;
			m = {1'b0, mant};
			while (m[10] == 1'b0) begin
				m = m << 1;
				e = e - 1;
			end
			bits = {sign, 8'(e + (127 - 15)), m[9:0], 13'h0};
		end
	end else if (exp == 5'h1F) begin
		bits = {sign, 8'hFF, mant, 13'h0};
		if (mant != 10'h0)
			bits[22] = 1'b1;
	end else begin
		bits = {sign, (8'(exp) + 8'(127 - 15)), mant, 13'h0};
	end
	return bits;
endfunction

function automatic logic [15:0] f32_to_f16_bits(input logic [31:0] f);
	logic		sign;
	logic [7:0]	exp_f;
	logic [22:0]	mant_f;
	int		h_exp;
	logic [15:0]	result;

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
		h_exp = int'(exp_f) - 127 + 15;
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
			logic [23:0]	full_mant;
			int		sh;
			logic [31:0]	q;
			logic [31:0]	remainder;
			logic [31:0]	halfway;
			logic [31:0]	mant_h32;

			full_mant = {1'b1, mant_f};
			sh = 14 - h_exp;
			if (sh > 24 || sh < 1)
				result = {sign, 15'h0};
			else begin
				q = 32'(full_mant) >> sh;
				remainder = 32'(full_mant) & ((32'h1 << sh) - 32'h1);
				halfway = 32'h1 << (sh - 1);
				if (remainder > halfway
						|| (remainder == halfway && q[0]))
					q = q + 32'h1;
				mant_h32 = q;
				if (mant_h32 >= 32'd1024)
					result = {sign, 5'd1, 10'h0};
				else
					result = {sign, 5'd0, 10'(mant_h32)};
			end
		end else begin
			logic [9:0]	mant_h;
			logic [12:0]	remainder;
			int		h_exp2;

			mant_h = mant_f[22:13];
			remainder = mant_f[12:0];
			h_exp2 = h_exp;
			if (remainder > 13'h1000
					|| (remainder == 13'h1000 && mant_h[0])) begin
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
	return result;
endfunction

function automatic logic [63:0] f32_widen_to_f64(input logic [31:0] f);
	logic		sign;
	logic [7:0]	exp_f;
	logic [22:0]	mant_f;
	logic [63:0]	bits;

	sign = f[31];
	exp_f = f[30:23];
	mant_f = f[22:0];
	if (exp_f == 8'h00 && mant_f == 23'h0)
		bits = {sign, 63'h0};
	else if (exp_f == 8'hFF)
		bits = {sign, 11'h7FF, 52'(mant_f) << 29};
	else
		bits = {sign, (11'(exp_f) + 11'(1023 - 127)), 52'(mant_f) << 29};
	return bits;
endfunction

function automatic logic [31:0] f64_narrow_to_f32_rne(input logic [63:0] d);
	logic		sign;
	logic [10:0]	exp_d;
	logic [51:0]	mant_d;
	int		exp32;
	logic [22:0]	mant32;
	logic [28:0]	dropped;
	logic [31:0]	result;

	sign = d[63];
	exp_d = d[62:52];
	mant_d = d[51:0];
	if (exp_d == 11'h000 && mant_d == 52'h0) begin
		result = {sign, 31'h0};
	end else if (exp_d == 11'h7FF) begin
		result = {sign, 8'hFF, (mant_d == 52'h0) ? 23'h0 : 23'h400000};
	end else begin
		exp32 = int'(exp_d) - 1023 + 127;
		mant32 = mant_d[51:29];
		dropped = mant_d[28:0];
		if (dropped[28] == 1'b1
				&& (dropped[27:0] != 28'h0 || mant32[0] == 1'b1)) begin
			if (mant32 == 23'h7FFFFF) begin
				mant32 = 23'h0;
				exp32 = exp32 + 1;
			end else begin
				mant32 = mant32 + 23'h1;
			end
		end
		if (exp32 >= 255)
			result = {sign, 8'hFF, 23'h0};
		else if (exp32 <= 0)
			result = {sign, 31'h0};
		else
			result = {sign, 8'(exp32), mant32};
	end
	return result;
endfunction

endpackage : membrane_fp_pkg
