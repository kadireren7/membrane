// Focused component differential test: the real, general-purpose
// rtl/membrane_fp_divider.sv (the reference) vs.
// rtl/membrane_fp_divider_radix4.sv (radix-4, two quotient bits/cycle,
// 13-cycle main iteration). Both are Verilated from their own real RTL --
// a genuine RTL-vs-RTL bit-exactness check, not RTL-vs-idealized-math.
//
// Originates from EXP-FPGA-DIV-001 Phase B4
// (experiments/EXP-FPGA-DIV-001/phase-b4.md), which ran this same case
// set as a 3-way comparison also including Phase B2's radix-2 divider
// (rtl/experimental/fp_div/fp32_div_iterative_exact.sv, not promoted --
// see experiments/EXP-FPGA-DIV-001/promotion-audit.md). Trimmed to the
// 2-way production form here: B2's divider is not part of the production
// tree, so it is not a reference in this test. Every case category B2's
// own comparison covered (boundary sweep, specials cross product, powers
// of two, random num=1, random general, real Q4_0 runtime d-distribution,
// reset-recovery, throughput) is preserved unchanged.
//
// Every case is driven through both DUTs simultaneously, one case at a
// time (fully drained before the next, since latencies differ: baseline
// is fixed 1 cycle, the radix-4 divider is variable, ~2-16+ cycles).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include "Vmembrane_fp_divider.h"
#include "Vmembrane_fp_divider_radix4.h"
#include "verilated.h"
#include "membrane/f16convert.h"

static const uint32_t	ONE_F32 = 0x3F800000u;
static const uint64_t	MAX_CYCLES_PER_CASE = 10000;

static Vmembrane_fp_divider		*g_baseline;
static Vmembrane_fp_divider_radix4	*g_radix4;

static uint64_t	g_cycle = 0;
static std::chrono::steady_clock::time_point	g_start_time, g_last_heartbeat;
static std::mt19937	g_rng(0xC0FFEEu);

struct LatStats
{
	uint64_t	min_c = UINT64_MAX;
	uint64_t	max_c = 0;
	long double	sum = 0.0L;
	uint64_t	count = 0;
	uint64_t	nobp_min = UINT64_MAX;
	uint64_t	nobp_max = 0;
	long double	nobp_sum = 0.0L;
	uint64_t	nobp_count = 0;

	void	record(uint64_t cyc, bool allow_bp)
	{
		min_c = std::min(min_c, cyc);
		max_c = std::max(max_c, cyc);
		sum += (long double)cyc;
		count++;
		if (!allow_bp)
		{
			nobp_min = std::min(nobp_min, cyc);
			nobp_max = std::max(nobp_max, cyc);
			nobp_sum += (long double)cyc;
			nobp_count++;
		}
	}
};

struct Result
{
	uint64_t	total = 0;
	uint64_t	accepted = 0;
	uint64_t	retired = 0;
	uint64_t	mismatch = 0;
	std::map<std::string, uint64_t>	mismatch_by_category;
	std::vector<std::string>	first_mismatches;
	LatStats	lat;
};

static Result	g_result;
static uint64_t	g_planned = 0;

static std::string	category_of(uint32_t den, const char *tag)
{
	if (tag)
		return std::string(tag);

	uint32_t	exp = (den >> 23) & 0xFFu;
	uint32_t	mant = den & 0x7FFFFFu;
	uint32_t	sign = (den >> 31) & 1u;
	const char	*sgn = sign ? "neg" : "pos";

	if (exp == 0xFFu && mant == 0u)
		return std::string("den_inf_") + sgn;
	if (exp == 0xFFu && mant != 0u)
		return std::string(((mant & 0x400000u) ? "den_nan_quiet_" : "den_nan_signaling_")) + sgn;
	if (exp == 0u && mant == 0u)
		return std::string("den_zero_") + sgn;
	if (exp == 0u && mant != 0u)
		return std::string("den_subnormal_") + sgn;
	return std::string("den_general_") + sgn;
}

static void	heartbeat(void)
{
	auto	now = std::chrono::steady_clock::now();
	double	since = std::chrono::duration<double>(now - g_last_heartbeat).count();

	if (since >= 5.0)
	{
		double	elapsed = std::chrono::duration<double>(now - g_start_time).count();
		double	frac = g_planned ? (double)g_result.total / (double)g_planned : 0.0;
		double	eta = (frac > 0.0) ? (elapsed / frac - elapsed) : 0.0;

		fprintf(stderr,
			"[heartbeat] cases=%lu/%lu mismatches=%lu elapsed=%.0fs eta=%.0fs cycle=%lu\n",
			(unsigned long)g_result.total, (unsigned long)g_planned,
			(unsigned long)g_result.mismatch, elapsed, eta, (unsigned long)g_cycle);
		g_last_heartbeat = now;
	}
}

static void	step(void)
{
	g_baseline->clk = 0;
	g_radix4->clk = 0;
	g_baseline->eval();
	g_radix4->eval();

	g_baseline->clk = 1;
	g_radix4->clk = 1;
	g_baseline->eval();
	g_radix4->eval();
	g_cycle++;
}

static void	reset_all(int cycles)
{
	g_baseline->rst_n = 0;
	g_radix4->rst_n = 0;
	g_baseline->valid_in = 0;
	g_baseline->a_in = 0;
	g_baseline->b_in = 0;
	g_radix4->in_valid = 0;
	g_radix4->numerator = 0;
	g_radix4->denominator = 0;
	g_radix4->out_ready = 1;
	for (int i = 0; i < cycles; i++)
		step();
	g_baseline->rst_n = 1;
	g_radix4->rst_n = 1;
	step();
}

static bool	run_case(uint32_t num, uint32_t den, uint32_t &bo, uint32_t &r4o,
		uint64_t &lat, bool allow_bp)
{
	if (!g_radix4->in_ready)
	{
		fprintf(stderr, "PROTOCOL ERROR: radix4 not in_ready at case start (num=0x%08X den=0x%08X)\n",
			num, den);
		return false;
	}

	g_baseline->valid_in = 1;
	g_baseline->a_in = num;
	g_baseline->b_in = den;
	g_radix4->in_valid = 1;
	g_radix4->numerator = num;
	g_radix4->denominator = den;
	g_radix4->out_ready = 1;

	bool	pre_ready = g_radix4->in_ready;
	bool	pre_valid = g_radix4->in_valid;

	step();
	g_result.accepted += (pre_ready && pre_valid) ? 1 : 0;
	g_baseline->valid_in = 0;
	g_radix4->in_valid = 0;

	if (!g_baseline->valid_out)
	{
		fprintf(stderr, "PROTOCOL ERROR: baseline valid_out not asserted 1 cycle after issue (DELAY=1 violated)\n");
		return false;
	}
	bo = g_baseline->result_out;

	uint64_t	cyc = 1;
	bool	got = false;

	while (cyc < MAX_CYCLES_PER_CASE && !got)
	{
		bool	ready = allow_bp ? (bool)(g_rng() & 1u) : true;

		g_radix4->out_ready = ready;
		if (g_radix4->out_valid && ready)
		{
			r4o = g_radix4->quotient;
			lat = cyc;
			got = true;
		}
		step();
		cyc++;
	}
	if (!got)
	{
		fprintf(stderr, "TIMEOUT/DEADLOCK: radix4 never retired within %lu cycles (num=0x%08X den=0x%08X)\n",
			(unsigned long)MAX_CYCLES_PER_CASE, num, den);
		return false;
	}
	g_result.retired++;
	g_result.lat.record(lat, allow_bp);
	return true;
}

static bool	check_case(uint32_t num, uint32_t den, const char *tag, bool allow_bp)
{
	uint32_t	bo, r4o;
	uint64_t	lat;

	if (!run_case(num, den, bo, r4o, lat, allow_bp))
		exit(1);
	g_result.total++;

	bool	ok = (bo == r4o);

	if (!ok)
	{
		std::string	cat = category_of(den, tag);

		g_result.mismatch++;
		g_result.mismatch_by_category[cat]++;
		if (g_result.first_mismatches.size() < 25)
		{
			char	buf[256];

			snprintf(buf, sizeof(buf),
				"num=0x%08X den=0x%08X category=%s baseline=0x%08X radix4=0x%08X",
				num, den, cat.c_str(), bo, r4o);
			g_result.first_mismatches.push_back(buf);
		}
	}
	heartbeat();
	return ok;
}

static void	reset_recovery_test(uint64_t &fails)
{
	fprintf(stderr, "[stage] reset-mid-computation recovery test (random reset timing on the radix-4 divider)\n");
	if (!g_radix4->in_ready)
	{
		fprintf(stderr, "FAIL: radix4 not idle before reset-recovery test\n");
		fails++;
		return;
	}
	g_radix4->numerator = ONE_F32;
	g_radix4->denominator = 0x40490FDBu;	// pi, arbitrary general-path denominator
	g_radix4->in_valid = 1;
	g_radix4->out_ready = 1;
	step();
	g_radix4->in_valid = 0;
	if (!g_radix4->busy)
	{
		fprintf(stderr, "FAIL: radix4 not busy right after accepting a general-path case\n");
		fails++;
	}
	// Random reset timing within the ~13-cycle main iteration.
	std::uniform_int_distribution<int>	offset_dist(1, 6);
	int	off = offset_dist(g_rng);

	for (int i = 0; i < off; i++)
		step();
	if (g_radix4->out_valid)
	{
		fprintf(stderr, "FAIL: radix4 asserted out_valid too early (%d cycles in) for a general-path case\n", off);
		fails++;
	}

	g_radix4->rst_n = 0;
	for (int i = 0; i < 3; i++)
	{
		step();
		if (g_radix4->out_valid)
		{
			fprintf(stderr, "FAIL: out_valid asserted during reset\n");
			fails++;
		}
	}
	g_radix4->rst_n = 1;
	step();

	if (!g_radix4->in_ready)
	{
		fprintf(stderr, "FAIL: radix4 not in_ready one cycle after reset deassertion\n");
		fails++;
	}
	if (g_radix4->busy)
	{
		fprintf(stderr, "FAIL: radix4 still busy one cycle after reset deassertion\n");
		fails++;
	}

	// baseline was not exercised in this stage -- reset it back to a
	// clean idle state too before resuming the shared 2-way loop.
	g_baseline->rst_n = 0;
	g_baseline->valid_in = 0;
	step();
	g_baseline->rst_n = 1;
	step();

	if (!check_case(ONE_F32, 0x3F000000u, "reset_recovery_followup", false))	// 1/0.5 = 2.0
	{
		fprintf(stderr, "FAIL: post-reset-recovery case mismatched\n");
		fails++;
	}
	fprintf(stderr, "[stage] reset-mid-computation recovery test done, fails so far: %lu\n",
		(unsigned long)fails);
}

static void	throughput_test(uint64_t n, uint64_t &out_total_cycles)
{
	uint64_t	cyc0 = g_cycle;
	std::uniform_int_distribution<uint32_t>	dist(0x00800000u, 0x7F7FFFFFu);

	fprintf(stderr, "[stage] throughput measurement: %lu back-to-back general-path cases\n", (unsigned long)n);
	for (uint64_t i = 0; i < n; i++)
	{
		if (!check_case(ONE_F32, dist(g_rng), "throughput_general", false))
			exit(1);
	}
	out_total_cycles = g_cycle - cyc0;
	fprintf(stderr, "[stage] throughput measurement done: %lu cycles for %lu transactions\n",
		(unsigned long)out_total_cycles, (unsigned long)n);
}

int	main(int argc, char **argv)
{
	// Defaults reproduce the same case counts EXP-FPGA-DIV-001 Phase B4
	// used for its own (3-way) --full run; a CI-bounded subset is passed
	// via argv by scripts/verify-q4-radix4-divider.sh instead of changing
	// these defaults -- see that script and
	// experiments/EXP-FPGA-DIV-001/promotion-plan.md item 7.
	uint64_t	random_count = 4200000;
	uint64_t	general_random_count = 200000;
	uint64_t	q4_dist_count = 50000;

	Verilated::commandArgs(argc, argv);
	if (argc > 1)
		random_count = strtoull(argv[1], nullptr, 10);
	if (argc > 2)
		general_random_count = strtoull(argv[2], nullptr, 10);
	if (argc > 3)
		q4_dist_count = strtoull(argv[3], nullptr, 10);

	g_baseline = new Vmembrane_fp_divider;
	g_radix4 = new Vmembrane_fp_divider_radix4;

	g_start_time = std::chrono::steady_clock::now();
	g_last_heartbeat = g_start_time;

	reset_all(5);

	std::vector<uint32_t>	mantissas = {
		0x000000u, 0x000001u, 0x000002u, 0x3FFFFFu,
		0x400000u, 0x400001u, 0x7FFFFEu, 0x7FFFFFu,
	};
	std::vector<uint32_t>	specials = {
		0x00000000u, 0x80000000u,
		0x00000001u, 0x80000001u,
		0x007FFFFFu, 0x807FFFFFu,
		0x00800000u, 0x80800000u,
		0x7F7FFFFFu, 0xFF7FFFFFu,
		0x7F800000u, 0xFF800000u,
		0x7FC00000u, 0xFFC00000u,
		0x7FA00000u, 0xFFA00000u,
		0x7FFFFFFFu, 0xFFFFFFFFu,
		0x7F800001u, 0xFF800001u,
		0x3F800000u, 0xBF800000u,
		0x40000000u, 0xC0000000u,
	};
	std::vector<uint32_t>	pow2 = {
		0x00800000u, 0x01000000u, 0x3F000000u, 0x3F800000u,
		0x40000000u, 0x40800000u, 0x7E800000u, 0x7F000000u,
		0x80800000u, 0xBF800000u, 0xC0000000u, 0xFE800000u,
	};

	uint64_t	boundary_count = 256ull * 2ull * mantissas.size();
	uint64_t	cross_special_count = (uint64_t)specials.size() * (uint64_t)specials.size();

	g_planned = boundary_count + cross_special_count + pow2.size()
		+ random_count + general_random_count + q4_dist_count;

	fprintf(stderr, "=== membrane_fp_divider_radix4 differential test (vs. membrane_fp_divider) ===\n");
	fprintf(stderr, "planned total cases: %lu\n", (unsigned long)g_planned);

	fprintf(stderr, "[stage] denominator exponent/mantissa boundary sweep (numerator=1.0f): %lu cases\n",
		(unsigned long)boundary_count);
	for (uint32_t exp = 0; exp <= 255; exp++)
		for (uint32_t sign = 0; sign <= 1; sign++)
			for (uint32_t m : mantissas)
			{
				uint32_t	den = (sign << 31) | (exp << 23) | m;

				check_case(ONE_F32, den, nullptr, false);
			}
	fprintf(stderr, "[stage] boundary sweep done, total=%lu\n", (unsigned long)g_result.total);

	fprintf(stderr, "[stage] specials x specials cross product (both operands vary): %lu cases\n",
		(unsigned long)cross_special_count);
	for (uint32_t num : specials)
		for (uint32_t den : specials)
			check_case(num, den, "cross_special", false);
	fprintf(stderr, "[stage] cross-special done, total=%lu\n", (unsigned long)g_result.total);

	fprintf(stderr, "[stage] powers of two (numerator=1.0f): %lu cases\n", (unsigned long)pow2.size());
	for (uint32_t den : pow2)
		check_case(ONE_F32, den, "pow2", false);

	fprintf(stderr, "[stage] uniform random denominator, numerator=1.0f: %lu cases, random out_ready backpressure\n",
		(unsigned long)random_count);
	{
		std::uniform_int_distribution<uint32_t>	dist(0, 0xFFFFFFFFu);

		for (uint64_t i = 0; i < random_count; i++)
			check_case(ONE_F32, dist(g_rng), "random_num1_denrand", (i % 4) == 0);
	}
	fprintf(stderr, "[stage] random (numerator=1.0f) done, total=%lu\n", (unsigned long)g_result.total);

	fprintf(stderr, "[stage] uniform random BOTH operands (general-purpose check): %lu cases\n",
		(unsigned long)general_random_count);
	{
		std::uniform_int_distribution<uint32_t>	dist(0, 0xFFFFFFFFu);

		for (uint64_t i = 0; i < general_random_count; i++)
		{
			uint32_t	num = dist(g_rng);
			uint32_t	den = dist(g_rng);

			check_case(num, den, "random_general", (i % 5) == 0);
		}
	}
	fprintf(stderr, "[stage] random (both operands) done, total=%lu\n", (unsigned long)g_result.total);

	fprintf(stderr, "[stage] q4 runtime d-distribution sample: %lu cases\n", (unsigned long)q4_dist_count);
	{
		std::mt19937	block_rng(0xB10CDu);
		std::uniform_int_distribution<uint32_t>	half_dist(0, 0xFFFFu);

		for (uint64_t i = 0; i < q4_dist_count; i++)
		{
			float	amax = 0.0f;
			float	mx = 0.0f;

			for (int j = 0; j < 32; j++)
			{
				uint16_t	h = (uint16_t)half_dist(block_rng);
				float	v = membrane_f16_to_f32(h);

				if (fabsf(v) > amax)
				{
					amax = fabsf(v);
					mx = v;
				}
			}

			float	d = mx / -8.0f;
			uint32_t	d_bits;

			memcpy(&d_bits, &d, sizeof(d_bits));
			check_case(ONE_F32, d_bits, "q4_runtime_d_dist", (i % 3) == 0);
		}
	}
	fprintf(stderr, "[stage] q4 d-distribution done, total=%lu\n", (unsigned long)g_result.total);

	uint64_t	reset_fails = 0;

	reset_recovery_test(reset_fails);

	uint64_t	throughput_cycles = 0;
	uint64_t	throughput_n = 2000;

	throughput_test(throughput_n, throughput_cycles);

	double	elapsed = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - g_start_time).count();
	double	ii = throughput_n ? (double)throughput_cycles / (double)throughput_n : 0.0;

	printf("=== membrane_fp_divider_radix4 differential test report ===\n");
	printf("total cases:              %lu\n", (unsigned long)g_result.total);
	printf("mismatches:               %lu\n", (unsigned long)g_result.mismatch);
	printf("accepted inputs:          %lu\n", (unsigned long)g_result.accepted);
	printf("retired outputs:          %lu\n", (unsigned long)g_result.retired);
	printf("latency cycles: min=%lu mean=%.3f max=%lu (no-bp: min=%lu mean=%.3f max=%lu, n=%lu)\n",
		(unsigned long)(g_result.lat.min_c == UINT64_MAX ? 0 : g_result.lat.min_c),
		g_result.lat.count ? (double)(g_result.lat.sum / (long double)g_result.lat.count) : 0.0,
		(unsigned long)g_result.lat.max_c,
		(unsigned long)(g_result.lat.nobp_min == UINT64_MAX ? 0 : g_result.lat.nobp_min),
		g_result.lat.nobp_count ? (double)(g_result.lat.nobp_sum / (long double)g_result.lat.nobp_count) : 0.0,
		(unsigned long)g_result.lat.nobp_max, (unsigned long)g_result.lat.nobp_count);
	printf("initiation_interval (measured, %lu back-to-back, no backpressure): %.3f\n", (unsigned long)throughput_n, ii);
	printf("reset_recovery_fails:  %lu\n", (unsigned long)reset_fails);
	printf("wall_time_s:           %.1f\n", elapsed);
	if (!g_result.mismatch_by_category.empty())
	{
		printf("mismatch categories:\n");
		for (auto &kv : g_result.mismatch_by_category)
			printf("  %-28s %lu\n", kv.first.c_str(), (unsigned long)kv.second);
		printf("first %zu mismatch examples:\n", g_result.first_mismatches.size());
		for (auto &s : g_result.first_mismatches)
			printf("  %s\n", s.c_str());
	}

	bool	pass = (g_result.mismatch == 0) && (reset_fails == 0)
		&& (g_result.total == g_planned + 1 + throughput_n);

	if (pass)
	{
		printf("PASS: membrane_fp_divider_radix4 vs membrane_fp_divider, %lu cases, 0 mismatches, 0 reset-recovery fails\n",
			(unsigned long)g_result.total);
		delete g_baseline;
		delete g_radix4;
		return 0;
	}
	printf("FAIL: %lu mismatches, %lu reset-recovery fails -- do not integrate\n",
		(unsigned long)g_result.mismatch, (unsigned long)reset_fails);
	delete g_baseline;
	delete g_radix4;
	return 1;
}
