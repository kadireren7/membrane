// Phase 5.3: simulation-only F32<->F64 helpers, split out of
// membrane_fp_pkg.sv so that file can stay strictly synthesizable.
// f32_widen_to_f64/f64_narrow_to_f32_rne exist only to let a testbench
// or a (now retired from the production datapath) real-arithmetic
// placeholder module compute a correctly-rounded float32 divide/multiply
// via the simulator's native `real` type (via $bitstoreal/$realtobits)
// -- see docs/phase5-fpga-streaming.md (Phase 5.2) for why that
// construction was originally used, and docs/phase5-synthesizable-fpga.md
// (Phase 5.3) for why the production datapath no longer uses it (replaced
// by a genuinely synthesizable integer restoring divider and integer
// multiplier, see membrane_fp_divider.sv / membrane_fp_multiplier.sv).
// This file is `` `include``d or read only by testbenches from here on,
// never by a module yosys is asked to synthesize.
package membrane_fp_sim_pkg;

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
	return (bits);
endfunction

function automatic logic [31:0] f64_narrow_to_f32_rne(input logic [63:0] d);
	logic		sign;
	logic [10:0]	exp_d;
	logic [51:0]	mant_d;
	int		exp32;
	int		exp_d_signed;
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
		exp_d_signed = exp_d;
		exp32 = exp_d_signed - 1023 + 127;
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
	return (result);
endfunction

endpackage : membrane_fp_sim_pkg
