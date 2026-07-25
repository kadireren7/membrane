// A fixed-depth valid/ready shift-register delay line, factored into its
// own module instance rather than a register embedded in a larger
// datapath module. This is a workaround, not a style choice: see the
// header comment in rtl/q8_maxabs_reduce.sv for the Icarus Verilog 12.0
// issue this works around (a shift register or discrete chained
// registers placed in the SAME always_ff/module as this project's
// array-heavy, many-times-automatic-function-calling data path were
// found, via a series of isolated reproductions, to update one cycle
// early -- moving the delay line into a genuinely separate module
// instance, with its own independent process, avoids it entirely and was
// verified correct in every RTL test in this repository.)
module valid_delay_line #(
	parameter int DEPTH = 5
) (
	input	logic	clk,
	input	logic	rst_n,
	input	logic	valid_in,
	output	logic	valid_out
);

	logic	[DEPTH - 1:0]	shift;

	always_ff @(posedge clk or negedge rst_n)
		if (!rst_n)
			shift <= '0;
		else
			shift <= {shift[DEPTH - 2:0], valid_in};

	assign valid_out = shift[DEPTH - 1];
endmodule
