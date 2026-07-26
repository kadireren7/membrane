// Phase 5.3: Verilator full-pipeline cosimulation for
// membrane_quant_stream_top against the real C reference
// (src/quant/quant_simd.c, via the same gen_*_vectors.c golden-vector
// generators used throughout this phase's per-module tests -- not a
// reimplementation of the float math in C++). Chosen over Icarus for
// this specific test because Icarus 12.0 was found, during bring-up, to
// hang indefinitely whenever the Q8_0 decode (q8_dequantize) and Q4_0
// decode (q4_unpack) chains are BOTH instantiated in the same
// simulation and driven through this module's FIFO/credit/tag-pipe
// plumbing -- reproduced and isolated via bisection (see
// docs/phase5-synthesizable-fpga.md), not seen with any 3-of-4 chain
// combination or with the two decode modules wired directly to each
// other outside this module's plumbing. Verilator does not exhibit this
// behavior.
//
// Runs, per mode (Q8 encode, Q8 decode, Q4 encode, Q4 decode), at least
// 100,000 transactions (see N_PER_MODE below) plus a final mixed-mode
// interleaved run exercising the shared-latency ordering guarantee
// across mode boundaries, all under randomized valid/ready backpressure
// on both the input and output sides, plus explicit reset-mid-stream
// (pipeline flush) tests. Every mismatch reports transaction id, cycle,
// expected vs actual, and the first differing byte, per the phase
// spec's item 6. A 60-second wall-clock heartbeat prints stage/
// progress/elapsed/ETA per item 12.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include "Vmembrane_quant_stream_top.h"
#include "verilated.h"

static const int N_PER_MODE = 120000;
static const int N_MIX = 40000;
static const int ID_WIDTH = 16;

enum { MODE_Q8_ENC = 0, MODE_Q8_DEC = 1, MODE_Q4_ENC = 2, MODE_Q4_DEC = 3 };

static std::vector<uint16_t>	g_x;		// N*32
static std::vector<uint8_t>	g_q8pack;	// N*34
static std::vector<uint16_t>	g_q8dequant;	// N*32
static std::vector<uint8_t>	g_q4pack;	// N*18
static std::vector<uint16_t>	g_q4unpack;	// N*32

static std::mt19937	g_rng(0xC0FFEEu);

static void	load_hex16(const char *path, std::vector<uint16_t> &out, long n)
{
	FILE	*f = fopen(path, "r");
	unsigned int	v;
	long	i;

	if (!f)
	{
		fprintf(stderr, "cannot open %s\n", path);
		exit(1);
	}
	out.resize(n);
	i = 0;
	while (i < n && fscanf(f, "%x", &v) == 1)
	{
		out[i] = (uint16_t)v;
		i++;
	}
	if (i != n)
	{
		fprintf(stderr, "%s: expected %ld entries, got %ld\n", path, n, i);
		exit(1);
	}
	fclose(f);
}

static void	load_hex8(const char *path, std::vector<uint8_t> &out, long n)
{
	FILE	*f = fopen(path, "r");
	unsigned int	v;
	long	i;

	if (!f)
	{
		fprintf(stderr, "cannot open %s\n", path);
		exit(1);
	}
	out.resize(n);
	i = 0;
	while (i < n && fscanf(f, "%x", &v) == 1)
	{
		out[i] = (uint8_t)v;
		i++;
	}
	if (i != n)
	{
		fprintf(stderr, "%s: expected %ld entries, got %ld\n", path, n, i);
		exit(1);
	}
	fclose(f);
}

static void	build_in_data(int mode, int blk, uint32_t words[16])
{
	int	j;

	memset(words, 0, 16 * sizeof(uint32_t));
	if (mode == MODE_Q8_ENC || mode == MODE_Q4_ENC)
	{
		for (j = 0; j < 32; j++)
		{
			uint16_t	v = g_x[(size_t)blk * 32 + j];
			int		word = j / 2;
			int		half = j % 2;

			words[word] |= ((uint32_t)v) << (half * 16);
		}
	}
	else if (mode == MODE_Q8_DEC)
	{
		for (j = 0; j < 34; j++)
		{
			uint8_t	b = g_q8pack[(size_t)blk * 34 + j];
			int	word = j / 4;
			int	byte_in_word = j % 4;

			words[word] |= ((uint32_t)b) << (byte_in_word * 8);
		}
	}
	else // MODE_Q4_DEC
	{
		for (j = 0; j < 18; j++)
		{
			uint8_t	b = g_q4pack[(size_t)blk * 18 + j];
			int	word = j / 4;
			int	byte_in_word = j % 4;

			words[word] |= ((uint32_t)b) << (byte_in_word * 8);
		}
	}
}

// Returns true on match; on mismatch, prints a detailed report (item 6:
// transaction id, cycle, expected, actual, first differing byte) and
// returns false.
static bool	check_out_data(int mode, int blk, const uint32_t got[16],
		uint64_t txn_id, uint64_t cycle)
{
	int	j;
	uint8_t	got_bytes[64];

	memcpy(got_bytes, got, 64);
	if (mode == MODE_Q8_ENC)
	{
		for (j = 0; j < 34; j++)
		{
			if (got_bytes[j] != g_q8pack[(size_t)blk * 34 + j])
			{
				fprintf(stderr,
					"MISMATCH Q8_ENC txn=%lu cycle=%lu blk=%d first_diff_byte=%d expect=%02x got=%02x\n",
					(unsigned long)txn_id, (unsigned long)cycle, blk, j,
					g_q8pack[(size_t)blk * 34 + j], got_bytes[j]);
				return (false);
			}
		}
	}
	else if (mode == MODE_Q8_DEC)
	{
		for (j = 0; j < 32; j++)
		{
			uint16_t	gv = got_bytes[j * 2] | (got_bytes[j * 2 + 1] << 8);

			if (gv != g_q8dequant[(size_t)blk * 32 + j])
			{
				fprintf(stderr,
					"MISMATCH Q8_DEC txn=%lu cycle=%lu blk=%d first_diff_lane=%d expect=%04x got=%04x\n",
					(unsigned long)txn_id, (unsigned long)cycle, blk, j,
					g_q8dequant[(size_t)blk * 32 + j], gv);
				return (false);
			}
		}
	}
	else if (mode == MODE_Q4_ENC)
	{
		for (j = 0; j < 18; j++)
		{
			if (got_bytes[j] != g_q4pack[(size_t)blk * 18 + j])
			{
				fprintf(stderr,
					"MISMATCH Q4_ENC txn=%lu cycle=%lu blk=%d first_diff_byte=%d expect=%02x got=%02x\n",
					(unsigned long)txn_id, (unsigned long)cycle, blk, j,
					g_q4pack[(size_t)blk * 18 + j], got_bytes[j]);
				return (false);
			}
		}
	}
	else // MODE_Q4_DEC
	{
		for (j = 0; j < 32; j++)
		{
			uint16_t	gv = got_bytes[j * 2] | (got_bytes[j * 2 + 1] << 8);

			if (gv != g_q4unpack[(size_t)blk * 32 + j])
			{
				fprintf(stderr,
					"MISMATCH Q4_DEC txn=%lu cycle=%lu blk=%d first_diff_lane=%d expect=%04x got=%04x\n",
					(unsigned long)txn_id, (unsigned long)cycle, blk, j,
					g_q4unpack[(size_t)blk * 32 + j], gv);
				return (false);
			}
		}
	}
	return (true);
}

struct InFlightTxn
{
	int		mode;
	int		blk;
	uint16_t	id;
};

static Vmembrane_quant_stream_top	*g_dut;
static uint64_t	g_cycle;
static uint64_t	g_fails;
static uint64_t	g_checked;
static std::chrono::steady_clock::time_point	g_start_time;
static std::chrono::steady_clock::time_point	g_last_heartbeat;
static uint64_t	g_total_planned;
static std::string	g_stage;

static void	heartbeat(void)
{
	auto	now = std::chrono::steady_clock::now();
	double	since_hb = std::chrono::duration<double>(now - g_last_heartbeat).count();

	if (since_hb >= 60.0)
	{
		double	elapsed = std::chrono::duration<double>(now - g_start_time).count();
		double	rate = g_checked > 0 ? (double)g_checked / elapsed : 0.0;
		double	eta = rate > 0 ? (double)(g_total_planned - g_checked) / rate : -1.0;

		fprintf(stderr,
			"[heartbeat] stage=%s completed=%lu/%lu elapsed=%.1fs eta=%.1fs cycle=%lu fails=%lu (simulator: Verilator)\n",
			g_stage.c_str(), (unsigned long)g_checked, (unsigned long)g_total_planned,
			elapsed, eta, (unsigned long)g_cycle, (unsigned long)g_fails);
		g_last_heartbeat = now;
	}
}

// A single clock cycle, structured to sample handshake signals at the
// point that actually determines whether THIS edge's handshake fires.
//
// Verilator's eval() fully settles all combinational logic for the
// CURRENT value of clk before returning -- unlike an event-driven
// simulator's delta-cycle scheduling, there is no "read before NBA
// settles" trick available here. So in_ready/out_valid must be sampled
// with clk still LOW, right after inputs are set but BEFORE the rising
// edge that would consume them -- that settled, pre-edge value is
// exactly what the DUT's own synchronous logic (`do_write = in_valid &&
// !full` etc.) uses to decide whether this edge accepts the
// transaction. Sampling AFTER the rising edge instead (as an early,
// buggy version of this file did) reads the POST-edge state -- i.e.
// whether the FIFO has room for the NEXT cycle, not whether THIS one's
// handshake fired -- which silently desynchronizes the testbench's
// notion of "accepted"/"retired" from the DUT's actual behavior at
// FIFO-full/empty boundary transitions, caught by systematic id
// mismatches starting partway through a 120,000-transaction run (not by
// a data-content mismatch -- the underlying pipeline math was already
// correct; only this harness's own sampling point was wrong).
static void	step_cycle(bool &accepted_out, bool &retired_out)
{
	g_dut->clk = 0;
	g_dut->eval();

	accepted_out = g_dut->in_valid && g_dut->in_ready;
	retired_out = g_dut->out_valid && g_dut->out_ready;

	g_dut->clk = 1;
	g_dut->eval();

	g_cycle++;
	heartbeat();
}

// Runs `count` transactions of a fixed mode (or mixed, if fixed_mode<0),
// with randomized backpressure on both in_valid gaps and out_ready
// gaps, checking every retired transaction against the golden vectors.
static void	run_mode(int fixed_mode, int count)
{
	std::vector<InFlightTxn>	inflight;
	int	issued = 0;
	int	checked_local = 0;
	uint64_t	next_id = 0;
	std::uniform_int_distribution<int>	stall_dist(0, 2);
	std::uniform_int_distribution<int>	mode_dist(0, 3);
	size_t	head = 0;

	while (checked_local < count)
	{
		bool	want_issue = (issued < count) && (stall_dist(g_rng) == 0);
		bool	accepted, retired;
		uint32_t	got[16];
		uint16_t	got_id = 0;
		uint8_t		got_mode = 0;

		if (want_issue)
		{
			int	mode = fixed_mode >= 0 ? fixed_mode : mode_dist(g_rng);
			int	blk = issued % N_PER_MODE;
			uint32_t	words[16];

			build_in_data(mode, blk, words);
			g_dut->in_valid = 1;
			g_dut->in_mode = mode;
			g_dut->in_id = (uint16_t)(next_id & 0xFFFFu);
			for (int w = 0; w < 16; w++)
				g_dut->in_data[w] = words[w];
		}
		else
		{
			g_dut->in_valid = 0;
		}
		g_dut->out_ready = (g_rng() & 1) ? 1 : 0;

		// Capture the fields check_out_data/id/mode need BEFORE the
		// rising edge (they must be read at the same settled point as
		// the accepted/retired handshake sample -- see step_cycle's
		// header comment).
		g_dut->eval();
		for (int w = 0; w < 16; w++)
			got[w] = g_dut->out_data[w];
		got_id = (uint16_t)g_dut->out_id;
		got_mode = (uint8_t)g_dut->out_mode;

		step_cycle(accepted, retired);

		if (want_issue && accepted)
		{
			int	mode = g_dut->in_mode;
			InFlightTxn	t;

			t.mode = mode;
			t.blk = issued % N_PER_MODE;
			t.id = (uint16_t)(next_id & 0xFFFFu);
			inflight.push_back(t);
			issued++;
			next_id++;
		}
		if (retired)
		{
			if (head >= inflight.size())
			{
				fprintf(stderr, "PROTOCOL ERROR: retire with no in-flight transaction, cycle=%lu\n",
					(unsigned long)g_cycle);
				g_fails++;
			}
			else
			{
				InFlightTxn	&t = inflight[head];

				if (got_id != t.id)
				{
					fprintf(stderr, "ID MISMATCH at retire: expect=%u got=%u cycle=%lu\n",
						t.id, (unsigned)got_id, (unsigned long)g_cycle);
					g_fails++;
				}
				if (!check_out_data(t.mode, t.blk, got, t.id, g_cycle))
					g_fails++;
				if (got_mode != (uint8_t)t.mode)
				{
					fprintf(stderr, "MODE MISMATCH at retire: expect=%d got=%d cycle=%lu\n",
						t.mode, got_mode, (unsigned long)g_cycle);
					g_fails++;
				}
				head++;
				checked_local++;
				g_checked++;
			}
		}
	}
	g_dut->in_valid = 0;
	g_dut->out_ready = 0;
	g_dut->eval();
}

static void	do_reset(int cycles)
{
	bool	a, r;

	g_dut->rst_n = 0;
	g_dut->in_valid = 0;
	g_dut->out_ready = 0;
	for (int i = 0; i < cycles; i++)
		step_cycle(a, r);
	g_dut->rst_n = 1;
	step_cycle(a, r);
}

int	main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);
	g_dut = new Vmembrane_quant_stream_top;

	fprintf(stderr, "Loading golden vectors (%d blocks per format/direction)...\n", N_PER_MODE);
	load_hex16("/tmp/top_x_120k.txt", g_x, (long)N_PER_MODE * 32);
	load_hex8("/tmp/top_q8pack_120k.txt", g_q8pack, (long)N_PER_MODE * 34);
	load_hex16("/tmp/top_q8dequant_120k.txt", g_q8dequant, (long)N_PER_MODE * 32);
	load_hex8("/tmp/top_q4pack_120k.txt", g_q4pack, (long)N_PER_MODE * 18);
	load_hex16("/tmp/top_q4unpack_120k.txt", g_q4unpack, (long)N_PER_MODE * 32);
	fprintf(stderr, "Vectors loaded.\n");

	g_start_time = std::chrono::steady_clock::now();
	g_last_heartbeat = g_start_time;
	g_total_planned = (uint64_t)N_PER_MODE * 4 + N_MIX;

	do_reset(5);

	// Pipeline-flush / async-reset-mid-stream test: issue a handful of
	// transactions, reset mid-flight (before any retire), and confirm no
	// stale output appears and the pipeline restarts cleanly.
	{
		bool	a, r;
		uint32_t	words[16];

		fprintf(stderr, "[stage] reset-mid-stream flush test\n");
		build_in_data(MODE_Q8_ENC, 0, words);
		g_dut->in_valid = 1;
		g_dut->in_mode = MODE_Q8_ENC;
		g_dut->in_id = 0xABCD & 0xFFFF;
		for (int w = 0; w < 16; w++)
			g_dut->in_data[w] = words[w];
		g_dut->out_ready = 1;
		for (int i = 0; i < 3; i++)
			step_cycle(a, r);
		// reset before this transaction could possibly retire (L_MAX=7)
		g_dut->rst_n = 0;
		g_dut->in_valid = 0;
		for (int i = 0; i < 5; i++)
		{
			g_dut->eval();
			if (g_dut->out_valid)
			{
				fprintf(stderr, "FAIL: out_valid asserted during/after reset with no new input (stale output)\n");
				g_fails++;
			}
			step_cycle(a, r);
		}
		g_dut->rst_n = 1;
		step_cycle(a, r);
		fprintf(stderr, "[stage] reset-mid-stream flush test done (fails so far: %lu)\n",
			(unsigned long)g_fails);
	}

	g_stage = "Q8_ENC";
	fprintf(stderr, "[stage] Q8 encode: %d transactions\n", N_PER_MODE);
	run_mode(MODE_Q8_ENC, N_PER_MODE);
	fprintf(stderr, "[stage] Q8 encode done, checked=%lu fails=%lu\n",
		(unsigned long)g_checked, (unsigned long)g_fails);

	g_stage = "Q8_DEC";
	fprintf(stderr, "[stage] Q8 decode: %d transactions\n", N_PER_MODE);
	run_mode(MODE_Q8_DEC, N_PER_MODE);
	fprintf(stderr, "[stage] Q8 decode done, checked=%lu fails=%lu\n",
		(unsigned long)g_checked, (unsigned long)g_fails);

	g_stage = "Q4_ENC";
	fprintf(stderr, "[stage] Q4 encode: %d transactions\n", N_PER_MODE);
	run_mode(MODE_Q4_ENC, N_PER_MODE);
	fprintf(stderr, "[stage] Q4 encode done, checked=%lu fails=%lu\n",
		(unsigned long)g_checked, (unsigned long)g_fails);

	g_stage = "Q4_DEC";
	fprintf(stderr, "[stage] Q4 decode: %d transactions\n", N_PER_MODE);
	run_mode(MODE_Q4_DEC, N_PER_MODE);
	fprintf(stderr, "[stage] Q4 decode done, checked=%lu fails=%lu\n",
		(unsigned long)g_checked, (unsigned long)g_fails);

	g_stage = "MIXED";
	fprintf(stderr, "[stage] mixed-mode interleave: %d transactions\n", N_MIX);
	run_mode(-1, N_MIX);
	fprintf(stderr, "[stage] mixed-mode interleave done, checked=%lu fails=%lu\n",
		(unsigned long)g_checked, (unsigned long)g_fails);

	double	elapsed = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - g_start_time).count();

	if (g_fails == 0)
		printf("PASS: membrane_quant_stream_top Verilator cosim, %lu transactions, 0 fails, %.1fs\n",
			(unsigned long)g_checked, elapsed);
	else
		printf("FAIL: membrane_quant_stream_top Verilator cosim, %lu / %lu fails, %.1fs\n",
			(unsigned long)g_fails, (unsigned long)g_checked, elapsed);

	delete g_dut;
	return (g_fails == 0 ? 0 : 1);
}
