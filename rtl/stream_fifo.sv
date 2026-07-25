// Phase 5.2: a generic synchronous FIFO with an AXI-Stream-like
// valid/ready handshake on both sides -- the building block for the
// input/output buffering tools/membrane-hw-sim (Phase 5.2's C cycle
// model) assumes around each quantize/dequantize pipeline. WIDTH and
// DEPTH are parametric; DEPTH must be a power of two (checked by an
// elaboration-time assertion) so the read/write pointers can use a
// simple wraparound MSB-compare full/empty scheme rather than a
// separate counter.
module stream_fifo #(
	parameter int WIDTH = 8,
	parameter int DEPTH = 4
) (
	input	logic				clk,
	input	logic				rst_n,

	input	logic				in_valid,
	output	logic				in_ready,
	input	logic	[WIDTH-1:0]	in_data,

	output	logic				out_valid,
	input	logic				out_ready,
	output	logic	[WIDTH-1:0]	out_data,

	output	logic	[$clog2(DEPTH):0]	occupancy
);

	localparam int PtrBits = $clog2(DEPTH) + 1;

	logic	[WIDTH-1:0]	mem		[0:DEPTH-1];
	logic	[PtrBits-1:0]	wr_ptr;
	logic	[PtrBits-1:0]	rd_ptr;
	logic					full;
	logic					empty;
	logic					do_write;
	logic					do_read;

	assign full = (wr_ptr[PtrBits-1] != rd_ptr[PtrBits-1])
		&& (wr_ptr[PtrBits-2:0] == rd_ptr[PtrBits-2:0]);
	assign empty = (wr_ptr == rd_ptr);
	assign in_ready = !full;
	assign out_valid = !empty;
	assign do_write = in_valid && !full;
	assign do_read = out_ready && !empty;
	assign out_data = mem[rd_ptr[PtrBits-2:0]];
	assign occupancy = wr_ptr - rd_ptr;

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			wr_ptr <= '0;
			rd_ptr <= '0;
		end else begin
			if (do_write) begin
				mem[wr_ptr[PtrBits-2:0]] <= in_data;
				wr_ptr <= wr_ptr + 1'b1;
			end
			if (do_read)
				rd_ptr <= rd_ptr + 1'b1;
		end
	end

	initial begin
		if ((DEPTH & (DEPTH - 1)) != 0) begin
			$display("stream_fifo: DEPTH (%0d) must be a power of two",
				DEPTH);
			$finish;
		end
	end
endmodule
