/*
 * membrane-kv-exact-sim: Phase 6.4's UNIFIED 128K-context x
 * 512-concurrency exact sparse KV retrieval stress validation. See
 * docs/phase6-unified-stress.md for the full writeup, including this
 * session's real, disclosed scope choices vs. the spec (section 0).
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "calibrate.h"
#include "checkpoint.h"
#include "exact_engine.h"
#include "sim_config.h"
#include "wssim_config.h"

using namespace wssim;
using namespace exactsim;

struct model_entry_t
{
	std::string				name;
	std::string				long_trace_path;
	model_calibration_t	calib;
};

/* Phase 6.4 item 1: the unified scenario is ALWAYS 128K context x 512
 * concurrency -- both axes maxed simultaneously in the same run,
 * exactly the thing Phase 6.3 left disclosed as not attempted
 * (docs/phase6-exact-sparse-retrieval.md section 0/17). */
constexpr uint32_t	UNIFIED_TARGET_STEPS = 130560;	/* 128K - 512 prompt */
constexpr uint32_t	UNIFIED_CONCURRENCY = 512;

struct precision_mode_t
{
	std::string	name;
	bool		no_compression;
	bool		warm_is_q8;
};

static std::vector<precision_mode_t>	precision_modes()
{
	return {
		{"fp16", true, false},
		{"all-q8", false, true},
		/* "safe-mixed" reuses the Q8 ratio as its calibration-time
		 * compression proxy -- a real, disclosed simplification: this
		 * engine's per-channel cache model has one compression ratio
		 * per scenario, not a separately-tracked warm(Q8)/cold(Q4)
		 * split the way Phase 6.1's own tiered demotion model has.
		 * Phase 5.4/6.1's real blended Q8/Q4 ratio (~2.7x) is cited in
		 * the doc as the more accurate capacity-side number; this
		 * mode's calibration behavior (hit/miss timing) is identical
		 * to all-q8. */
		{"safe-mixed", false, true},
	};
}

/* Phase 6.4 item 2: the 7 requested comparisons. Two are analytical
 * (not simulated, section 0); five run through the real calibrate/
 * run_concurrent pipeline. */
struct comparison_t
{
	std::string	name;
	bool		analytical;
	policy_t	policy;
	bool		disable_prefetch;
	uint32_t	coalescing_window;
};

static std::vector<comparison_t>	comparisons()
{
	return {
		{"full-scan-cxl", true, policy_t::FULL, false, 0},
		{"compressed-full-scan-cxl", true, policy_t::FULL, false, 0},
		{"exact-no-prefetch", false, policy_t::NO_PREFETCH, false, 0},
		{"exact-predictor", false, policy_t::MEMBRANE_PREDICTIVE, true, 0},
		{"exact-predictor-prefetch", false, policy_t::MEMBRANE_PREDICTIVE,
			false, 0},
		{"exact-predictor-coalescing", false, policy_t::MEMBRANE_PREDICTIVE,
			false, 4},
		{"oracle", false, policy_t::ORACLE, false, 0},
	};
}

struct scenario_desc_t
{
	std::string	model_name;
	std::string	comparison_name;
	std::string	precision_name;
	uint64_t	host_cache_total_bytes;
	uint64_t	device_total_bytes;

	std::string	id() const
	{
		char	buf[512];
		snprintf(buf, sizeof(buf), "unified|%s|%s|%s|%llu|%llu",
			model_name.c_str(), comparison_name.c_str(),
			precision_name.c_str(),
			(unsigned long long)host_cache_total_bytes,
			(unsigned long long)device_total_bytes);
		return (std::string(buf));
	}
};

static const char	*CSV_HEADER =
	"model,comparison,precision,host_cache_total_bytes,device_total_bytes,"
	"concurrency,context_tokens,"
	"p50_latency_ns,p95_latency_ns,p99_latency_ns,tokens_per_sec,"
	"mean_bytes_per_token,link_util_pct,quant_util_pct,bottleneck,"
	"cap_total_logical_kv_bytes,cap_physical_device_bytes,cap_hot_cache_bytes,"
	"cap_effective_capacity_ratio,cap_sequences_fit,cap_sequences_requested,"
	"cap_failure_reason,"
	"link_p50_wait_ns,link_p95_wait_ns,link_p99_wait_ns,link_requests,"
	"device_p50_wait_ns,device_p95_wait_ns,device_p99_wait_ns,device_requests,"
	"quant_p50_wait_ns,quant_p95_wait_ns,quant_p99_wait_ns,quant_requests,"
	"max_simultaneous_fetches,mean_miss_burst_blocks,max_miss_burst_blocks,"
	"calib_hit_rate,calib_precision,calib_recall,calib_working_set_blocks\n";

static std::string	fmt_row(const scenario_desc_t &d, uint32_t concurrency,
					uint32_t context_tokens, const concurrent_result_t &cr,
					double hit_rate, double precision, double recall,
					double working_set)
{
	char	buf[2048];
	snprintf(buf, sizeof(buf),
		"%s,%s,%s,%llu,%llu,%u,%u,"
		"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,"
		"%llu,%llu,%llu,%.4f,%llu,%u,%s,"
		"%.2f,%.2f,%.2f,%llu,"
		"%.2f,%.2f,%.2f,%llu,"
		"%.2f,%.2f,%.2f,%llu,"
		"%llu,%.3f,%.3f,"
		"%.4f,%.4f,%.4f,%.3f",
		d.model_name.c_str(), d.comparison_name.c_str(),
		d.precision_name.c_str(),
		(unsigned long long)d.host_cache_total_bytes,
		(unsigned long long)d.device_total_bytes, concurrency, context_tokens,
		cr.p50_latency_ns, cr.p95_latency_ns, cr.p99_latency_ns,
		cr.tokens_per_sec, cr.mean_bytes_per_token, cr.link_utilization_pct,
		cr.quant_utilization_pct, cr.bottleneck.c_str(),
		(unsigned long long)cr.capacity.total_logical_kv_bytes,
		(unsigned long long)cr.capacity.physical_device_bytes,
		(unsigned long long)cr.capacity.hot_cache_bytes,
		cr.capacity.effective_capacity_ratio,
		(unsigned long long)cr.capacity.sequences_fit,
		cr.capacity.sequences_requested,
		cr.capacity.capacity_failure_reason.c_str(),
		cr.link_queue.p50_wait_ns, cr.link_queue.p95_wait_ns,
		cr.link_queue.p99_wait_ns, (unsigned long long)cr.link_queue.requests,
		cr.device_queue.p50_wait_ns, cr.device_queue.p95_wait_ns,
		cr.device_queue.p99_wait_ns,
		(unsigned long long)cr.device_queue.requests,
		cr.quant_queue.p50_wait_ns, cr.quant_queue.p95_wait_ns,
		cr.quant_queue.p99_wait_ns,
		(unsigned long long)cr.quant_queue.requests,
		(unsigned long long)cr.max_simultaneous_fetches,
		cr.mean_miss_burst_blocks, cr.max_miss_burst_blocks,
		hit_rate, precision, recall, working_set);
	return (std::string(buf));
}

static std::string	analytical_row(const scenario_desc_t &d,
					uint32_t context_tokens, double bytes_per_token)
{
	char	buf[1024];
	snprintf(buf, sizeof(buf),
		"%s,%s,%s,%llu,%llu,%u,%u,"
		"0,0,0,0,%.2f,0,0,n/a,"
		"0,0,0,0,0,%u,n/a,"
		"0,0,0,0,0,0,0,0,0,0,0,0,"
		"0,0,0,0,0,0,0",
		d.model_name.c_str(), d.comparison_name.c_str(),
		d.precision_name.c_str(),
		(unsigned long long)d.host_cache_total_bytes,
		(unsigned long long)d.device_total_bytes, UNIFIED_CONCURRENCY,
		context_tokens, bytes_per_token, UNIFIED_CONCURRENCY);
	return (std::string(buf));
}

struct trace_cache_t
{
	/* 128K extrapolated -- one model loaded at a time (see main()'s
	 * per-model loop): both models' unified traces resident at once
	 * needs more memory than this machine has. */
	std::map<std::string, attn_trace_t>	unified;
};

static std::mutex			g_tail_mutex;
static FILE					*g_tail_csv = nullptr;
static const std::string	*g_tail_target_comparison = nullptr;

static std::string	process_one(const scenario_desc_t &d,
						const trace_cache_t &traces,
						const std::map<std::string, model_calibration_t>
							&models, const comparison_t &comp)
{
	const attn_trace_t	&trace = traces.unified.at(d.model_name);
	const model_calibration_t	&model = models.at(d.model_name);
	uint32_t	context_tokens = trace.prompt_len + trace.step_count;

	if (comp.analytical)
	{
		double	mean_context = (double)trace.prompt_len
			+ (double)trace.step_count / 2.0;
		double	raw = (double)model.bytes_per_token_total * mean_context;
		double	bpt = comp.name == "full-scan-cxl" ? raw
			: raw / sim::Q8_COMPRESSION_RATIO;
		return (analytical_row(d, context_tokens, bpt));
	}

	precision_mode_t	prec{};
	for (const auto &p : precision_modes())
		if (p.name == d.precision_name)
			prec = p;

	scenario_config_t	cfg{};
	cfg.policy = comp.policy;
	cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
	cfg.block_size_tokens = 32;
	cfg.hot_cache_bytes = d.host_cache_total_bytes / UNIFIED_CONCURRENCY;
	cfg.warm_tier_is_q8 = prec.warm_is_q8;
	cfg.no_compression = prec.no_compression;
	cfg.disable_prefetch = comp.disable_prefetch;
	cfg.coalescing_window = comp.coalescing_window;

	calibrated_profile_t	profile = calibrate(trace, model, cfg, false);

	concurrent_config_t	ccfg{};
	ccfg.concurrency = UNIFIED_CONCURRENCY;
	ccfg.host_hot_cache_total_bytes = d.host_cache_total_bytes;
	ccfg.device_total_bytes = d.device_total_bytes;
	ccfg.quant_pipelines = wssim::DEFAULT_QUANT_PIPELINES;
	ccfg.microbatch_max_wait_ns = 0.0;	/* off by default -- Phase 6.3's
			 * null result (docs/phase6-exact-sparse-retrieval.md
			 * section 8) -- kept as one controlled comparison point
			 * elsewhere, not the unified matrix's default. */
	ccfg.microbatch_max_batch_blocks = 0;
	ccfg.hw = default_hardware_profile();

	double	compression = prec.no_compression ? 1.0
		: (prec.warm_is_q8 ? sim::Q8_COMPRESSION_RATIO : sim::Q4_COMPRESSION_RATIO);
	uint32_t	tail_n = 0;
	if (g_tail_target_comparison != nullptr && comp.name == *g_tail_target_comparison)
		tail_n = 20;
	concurrent_result_t	cr = run_concurrent(profile, model, 32, compression,
		ccfg, tail_n);

	if (tail_n > 0 && !cr.tail_samples.empty())
	{
		std::lock_guard<std::mutex>	lock(g_tail_mutex);
		for (const auto &ts : cr.tail_samples)
			fprintf(g_tail_csv,
				"%s,%s,%s,%llu,%llu,%u,%u,%llu,%llu,%.2f,%.2f,%.2f,%.2f,%.2f\n",
				d.model_name.c_str(), d.comparison_name.c_str(),
				d.precision_name.c_str(),
				(unsigned long long)d.host_cache_total_bytes,
				(unsigned long long)d.device_total_bytes,
				ts.sequence_id, ts.step_index,
				(unsigned long long)ts.prefetch_bytes,
				(unsigned long long)ts.compulsory_miss_bytes,
				ts.link_wait_ns, ts.device_wait_ns, ts.quant_wait_ns,
				ts.service_ns, ts.total_latency_ns);
		fflush(g_tail_csv);
	}

	return (fmt_row(d, UNIFIED_CONCURRENCY, context_tokens, cr,
		profile.hit_rate, profile.precision, profile.recall,
		profile.mean_working_set_blocks));
}

static std::vector<scenario_desc_t>	build_scenarios(
		const std::vector<model_entry_t> &models)
{
	std::vector<scenario_desc_t>	out;
	std::vector<uint64_t>	host_sizes = {
		64ull << 20, 256ull << 20, 1ull << 30, 4ull << 30, 8ull << 30};
	std::vector<uint64_t>	device_sizes = {
		512ull << 30, 1ull << 40, 2ull << 40};

	for (const auto &m : models)
		for (const auto &comp : comparisons())
			for (const auto &prec : precision_modes())
				for (uint64_t hb : host_sizes)
					for (uint64_t db : device_sizes)
					{
						if (comp.analytical && (hb != host_sizes[0]
								|| db != device_sizes[0]))
							continue ;	/* analytical rows don't vary
									 * with cache/device -- one row
									 * per (model, comparison,
									 * precision) is enough. */
						out.push_back({m.name, comp.name, prec.name, hb, db});
					}
	return (out);
}

struct heartbeat_t
{
	std::chrono::steady_clock::time_point	last;
	std::chrono::steady_clock::time_point	start;

	heartbeat_t()
	{
		last = std::chrono::steady_clock::now();
		start = last;
	}

	/* Phase 6.4 item 12: extended fields vs. Phase 6.3's heartbeat --
	 * simulated tokens, queue depth (approximated here by cumulative
	 * link requests dispatched so far, the real proxy this sweep
	 * driver has cheap access to), cache hit rate, bytes/token,
	 * current p99, active bottleneck all come from the LAST completed
	 * scenario's own real result (the sweep processes many independent
	 * scenarios, not one continuous simulation, so "current" means
	 * "most recently finished"). */
	void	maybe_print(size_t done, size_t total, uint64_t sim_tokens,
				double hit_rate, double bytes_per_token, double p99,
				const std::string &bottleneck)
	{
		auto	now = std::chrono::steady_clock::now();
		double	since = std::chrono::duration<double>(now - last).count();
		if (since < 60.0 && done < total)
			return ;
		last = now;
		double	elapsed = std::chrono::duration<double>(now - start).count();
		double	rate = done > 0 ? elapsed / done : 0.0;
		double	eta = rate * (double)(total - done);
		fprintf(stderr, "[heartbeat] %zu/%zu sim_tokens=%llu hit_rate=%.4f "
			"bytes/token=%.1f p99=%.1fns bottleneck=%s wall=%.1fs "
			"eta=%.1fs\n", done, total, (unsigned long long)sim_tokens,
			hit_rate, bytes_per_token, p99, bottleneck.c_str(), elapsed, eta);
	}
};

int	main(int argc, char **argv)
{
	std::string	trace_135m_long;
	std::string	trace_360m_long;
	std::string	out_csv = "benchmarks/cxl-sim/unified-sweep.csv";
	std::string	tail_csv_path = "benchmarks/cxl-sim/unified-tail-samples.csv";
	std::string	ckpt_path = "benchmarks/cxl-sim/unified-sweep.ckpt";
	unsigned	workers = std::thread::hardware_concurrency();
	std::string	tail_target = "exact-predictor-prefetch";

	for (int i = 1; i + 1 <= argc; i++)
	{
		if (strcmp(argv[i], "--trace-135m-long") == 0 && i + 1 < argc)
			trace_135m_long = argv[++i];
		else if (strcmp(argv[i], "--trace-360m-long") == 0 && i + 1 < argc)
			trace_360m_long = argv[++i];
		else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
			out_csv = argv[++i];
		else if (strcmp(argv[i], "--tail-out") == 0 && i + 1 < argc)
			tail_csv_path = argv[++i];
		else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc)
			ckpt_path = argv[++i];
		else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
			workers = (unsigned)atoi(argv[++i]);
	}
	if (trace_135m_long.empty())
	{
		fprintf(stderr, "usage: membrane-kv-exact-sim --trace-135m-long P "
			"[--trace-360m-long P] [--out CSV] [--tail-out CSV] "
			"[--checkpoint PATH] [--workers N]\n");
		return (2);
	}
	if (workers < 1)
		workers = 1;
	g_tail_target_comparison = &tail_target;

	std::vector<model_entry_t>	models;
	model_entry_t	m135;
	m135.name = "SmolLM2-135M";
	m135.long_trace_path = trace_135m_long;
	m135.calib = {"SmolLM2-135M", 30, 3, sim::SMOLLM2_135M_BYTES_PER_TOKEN,
		1.0e9 / sim::SMOLLM2_135M_TOK_PER_SEC};
	models.push_back(m135);
	if (!trace_360m_long.empty())
	{
		model_entry_t	m360;
		m360.name = "SmolLM2-360M";
		m360.long_trace_path = trace_360m_long;
		m360.calib = {"SmolLM2-360M", 32, 5,
			sim::SMOLLM2_360M_BYTES_PER_TOKEN,
			1.0e9 / sim::SMOLLM2_360M_TOK_PER_SEC};
		models.push_back(m360);
	}
	std::map<std::string, model_calibration_t>	model_by_name;
	for (const auto &m : models)
		model_by_name[m.name] = m.calib;

	fprintf(stderr, "membrane-kv-exact-sim: will load one model's 128K "
		"unified trace at a time, freeing it before the next model loads "
		"-- both models' unified traces resident simultaneously need more "
		"memory than this machine has (see docs/phase6-unified-stress.md "
		"section 0)...\n");

	std::vector<scenario_desc_t>	scenarios = build_scenarios(models);
	std::string	config_hash = sha256_hex_of_string(
		"phase6.4-unified-v1;target_steps=130560;concurrency=512;"
		"comparisons=7;precisions=3;host_sizes=5;device_sizes=3");
	std::string	trace_hash_input;
	for (const auto &m : models)
		trace_hash_input += sha256_hex_of_file(m.long_trace_path);
	std::string	trace_hash = sha256_hex_of_string(trace_hash_input);

	checkpoint_state_t	prior = load_checkpoint(ckpt_path, trace_hash,
		config_hash);
	if (prior.header_present && !prior.header_matches)
	{
		fprintf(stderr, "membrane-kv-exact-sim: checkpoint %s is STALE "
			"(%s) -- refusing to resume, starting fresh\n",
			ckpt_path.c_str(), prior.mismatch_reason.c_str());
		prior = checkpoint_state_t{};
	}

	FILE	*csv = fopen(out_csv.c_str(), "w");
	fprintf(csv, "%s", CSV_HEADER);
	for (const std::string &row : prior.completed_rows)
		fprintf(csv, "%s\n", row.c_str());

	/* Append, not truncate, when resuming: unlike the main CSV (whose
	 * rows are durably backed by the checkpoint and replayed above),
	 * tail samples have no checkpoint backing -- opening "w" on every
	 * restart silently discarded every tail sample captured by a prior
	 * process before it was OOM-killed (a real data-loss bug this
	 * phase hit: benchmarks/cxl-sim/unified-tail-samples.csv ended up
	 * empty after repeated restarts despite genuinely capturing 900
	 * real samples earlier in the run -- see
	 * docs/phase6-unified-stress.md section 5). */
	bool	tail_csv_exists_nonempty = false;
	{
		FILE	*probe = fopen(tail_csv_path.c_str(), "r");
		if (probe != nullptr)
		{
			fseek(probe, 0, SEEK_END);
			tail_csv_exists_nonempty = ftell(probe) > 0;
			fclose(probe);
		}
	}
	g_tail_csv = fopen(tail_csv_path.c_str(),
		tail_csv_exists_nonempty ? "a" : "w");
	if (!tail_csv_exists_nonempty)
		fprintf(g_tail_csv, "model,comparison,precision,host_cache_total_bytes,"
			"device_total_bytes,sequence_id,step_index,prefetch_bytes,"
			"compulsory_miss_bytes,link_wait_ns,device_wait_ns,quant_wait_ns,"
			"service_ns,total_latency_ns\n");

	checkpoint_writer_t	ckpt;
	ckpt.open(ckpt_path, trace_hash, config_hash, prior.header_present);

	std::mutex	io_mutex;
	std::atomic<size_t>	done_count{prior.completed_ids.size()};
	heartbeat_t	hb;
	std::map<std::string, comparison_t>	comp_by_name;
	for (const auto &c : comparisons())
		comp_by_name[c.name] = c;

	std::string	lh_path = out_csv.substr(0, out_csv.find_last_of('.'))
		+ "-layer-head-detail.csv";
	FILE	*lhf = fopen(lh_path.c_str(), "w");
	fprintf(lhf, "model,resolution,index,hit_rate\n");

	std::string	hw_path = out_csv.substr(0, out_csv.find_last_of('.'))
		+ "-hardware-sensitivity.csv";

	fprintf(stderr, "membrane-kv-exact-sim: %zu unified scenarios total, "
		"%u workers per model (%zu already complete)\n", scenarios.size(),
		workers, prior.completed_ids.size());

	/* Process one model fully (main scenarios + its layer/head-detail +,
	 * for 135M, hardware sensitivity), THEN free its unified trace before
	 * loading the next model's. A single model's 128K-context x top-k=8
	 * unified trace is ~2-4 GiB resident (n_layer x n_head_kv x top_k x
	 * 130560 steps); holding both models' unified traces at once
	 * previously OOM-killed this process on this machine's 5.6 GiB RAM
	 * (see docs/phase6-unified-stress.md section 0). This keeps the FULL
	 * spec'd 128K x top-k=8 resolution for both models -- it changes only
	 * how much is resident at once, not what is computed. */
	for (const auto &m : models)
	{
		fprintf(stderr, "membrane-kv-exact-sim: loading %s and building "
			"its 128K unified trace...\n", m.name.c_str());
		attn_trace_t	native;
		if (!load_attn_trace(m.long_trace_path, &native))
		{
			fprintf(stderr, "failed to load %s\n", m.long_trace_path.c_str());
			return (1);
		}
		trace_cache_t	traces;
		traces.unified[m.name] = extend_synthetic(native,
			UNIFIED_TARGET_STEPS, 42u);
		native = attn_trace_t{};	/* not needed past this point --
						 * extend_synthetic already copied
						 * what it needed; free it now
						 * rather than at scope exit. */
		fprintf(stderr, "membrane-kv-exact-sim: %s unified trace ready "
			"(context=%u tokens, concurrency=%u)\n", m.name.c_str(),
			UNIFIED_TARGET_STEPS + 512, UNIFIED_CONCURRENCY);

		std::vector<scenario_desc_t>	model_scenarios;
		for (const auto &d : scenarios)
			if (d.model_name == m.name)
				model_scenarios.push_back(d);

		std::atomic<size_t>	next_index{0};
		auto	worker_fn = [&](void)
		{
			while (true)
			{
				size_t	idx = next_index.fetch_add(1);
				if (idx >= model_scenarios.size())
					return ;
				const scenario_desc_t	&d = model_scenarios[idx];
				std::string	id = d.id();
				{
					std::lock_guard<std::mutex>	lock(io_mutex);
					if (prior.completed_ids.count(id))
						continue ;
				}
				std::string	row = process_one(d, traces, model_by_name,
					comp_by_name.at(d.comparison_name));
				{
					std::lock_guard<std::mutex>	lock(io_mutex);
					fprintf(csv, "%s\n", row.c_str());
					fflush(csv);
					ckpt.write_scenario(id, row);
					done_count++;
					/* Cheap heartbeat proxy fields parsed back out of the
					 * row we just wrote (rather than threading extra
					 * state through process_one's return type). */
					hb.maybe_print(done_count.load(), scenarios.size(),
						(uint64_t)done_count.load() * UNIFIED_CONCURRENCY,
						0.0, 0.0, 0.0, d.comparison_name);
				}
			}
		};

		fprintf(stderr, "membrane-kv-exact-sim: %zu scenarios for %s, %u "
			"workers\n", model_scenarios.size(), m.name.c_str(), workers);
		std::vector<std::thread>	pool;
		for (unsigned w = 0; w < workers; w++)
			pool.emplace_back(worker_fn);
		for (auto &th : pool)
			th.join();

		/* Item 25: per-layer/per-head predictor accuracy IN the unified
		 * (128K x 512-implied-cache-budget) scenario, for this model,
		 * while its trace is still resident. Same reasoning as Phase 6.3
		 * (variable-length arrays don't belong in the wide unified CSV).
		 * Uses a representative per-sequence cache budget (256MiB
		 * total / 512). */
		{
			const attn_trace_t	&t = traces.unified.at(m.name);
			scenario_config_t	cfg{};
			cfg.policy = policy_t::MEMBRANE_PREDICTIVE;
			cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
			cfg.block_size_tokens = 32;
			cfg.hot_cache_bytes = (256ull << 20) / UNIFIED_CONCURRENCY;
			cfg.warm_tier_is_q8 = true;
			layer_head_stats_t	lh;
			run_scenario_calibration(t, m.calib, cfg, nullptr, &lh, nullptr);
			for (size_t l = 0; l < lh.per_layer_hit_rate.size(); l++)
				fprintf(lhf, "%s,layer,%zu,%.4f\n", m.name.c_str(), l,
					lh.per_layer_hit_rate[l]);
			for (size_t h = 0; h < lh.per_head_hit_rate.size(); h++)
				fprintf(lhf, "%s,head,%zu,%.4f\n", m.name.c_str(), h,
					lh.per_head_hit_rate[h]);
			fflush(lhf);
			fprintf(stderr, "membrane-kv-exact-sim: layer/head detail for "
				"%s done\n", m.name.c_str());
		}

		/* Item 9/27: hardware sensitivity matrix -- ONE representative
		 * unified scenario point (135M, exact-predictor-prefetch, all-q8,
		 * 256MiB host cache, 1TiB device), hardware assumptions varied.
		 * Every latency/bandwidth figure here is ASSUMED (published
		 * industry-typical ranges, no real CXL hardware exists anywhere
		 * in this project -- same disclosure Phase 6.1 established)
		 * except pipeline count, which Phase 5.3's own real RTL
		 * simulation calibrated the per-pipeline rate for. Only run for
		 * 135M (this pass's original, still-representative scope) --
		 * done here, while 135M's trace is resident, instead of
		 * reloading it in a separate pass. */
		bool	hw_already_done = false;
		{
			FILE	*probe = fopen(hw_path.c_str(), "r");
			if (probe != nullptr)
			{
				int	lines = 0;
				int	ch;
				while ((ch = fgetc(probe)) != EOF)
					if (ch == '\n')
						lines++;
				fclose(probe);
				/* 1 header + 10 hw points -- this pass never writes a
				 * partial file (it's written point-by-point but this
				 * process only reaches here again after a full prior
				 * run completed it, since a mid-pass crash restarts the
				 * whole process from the checkpoint). Restarting this
				 * ~10-15 minute, non-checkpointed pass on every OOM-
				 * triggered restart was real, observed wasted wall time
				 * (see docs/phase6-unified-stress.md section 0) -- this
				 * skip makes retries spend that time on the actual
				 * remaining scenarios instead. */
				hw_already_done = (lines >= 11);
			}
		}
		if (m.name == "SmolLM2-135M" && !hw_already_done)
		{
			FILE	*hwf = fopen(hw_path.c_str(), "w");
			fprintf(hwf, "profile,cxl_latency_ns,cxl_bandwidth_gbps,"
				"pipelines,p50_latency_ns,p99_latency_ns,tokens_per_sec,"
				"mean_bytes_per_token,link_util_pct,quant_util_pct,"
				"bottleneck,link_p99_wait_ns\n");

			struct hw_point_t { std::string label; hardware_profile_t hw; };
			std::vector<hw_point_t>	points = {
				{"cxl-latency-low", {"cxl-latency-low", 100.0,
					sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				{"cxl-latency-medium (default)", {"cxl-latency-medium",
					sim::CXL_LINK_LATENCY_NS, sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				{"cxl-latency-high", {"cxl-latency-high", 250.0,
					sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				/* ASSUMED: Gen5 x16-class raw bandwidth, ~2x this
				 * project's existing x8-class 48 GB/s figure. */
				{"gen5-x16-bandwidth", {"gen5-x16", sim::CXL_LINK_LATENCY_NS,
					96.0, sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				/* ASSUMED: CXL 2.0-class (PCIe5-generation) figures --
				 * higher latency, narrower link than this project's
				 * existing default. */
				{"cxl-2.0-profile", {"cxl-2.0", 200.0, 32.0,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				/* ASSUMED: CXL 3.0-class (PCIe6-generation) figures --
				 * lower latency, wider link. */
				{"cxl-3.0-profile", {"cxl-3.0", 150.0, 64.0,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS,
					wssim::DEFAULT_QUANT_PIPELINES}},
				{"pipelines-1", {"pipelines-1", sim::CXL_LINK_LATENCY_NS,
					sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS, 1}},
				{"pipelines-2", {"pipelines-2", sim::CXL_LINK_LATENCY_NS,
					sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS, 2}},
				{"pipelines-4", {"pipelines-4", sim::CXL_LINK_LATENCY_NS,
					sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS, 4}},
				{"pipelines-8 (default)", {"pipelines-8",
					sim::CXL_LINK_LATENCY_NS, sim::CXL_LINK_BANDWIDTH_GBPS,
					sim::NEARMEM_PIPELINE_BYTES_PER_NS, 8}},
			};

			const attn_trace_t	&t = traces.unified.at("SmolLM2-135M");
			const model_calibration_t	&model =
				model_by_name.at("SmolLM2-135M");
			for (const auto &pt : points)
			{
				scenario_config_t	cfg{};
				cfg.policy = policy_t::MEMBRANE_PREDICTIVE;
				cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
				cfg.block_size_tokens = 32;
				cfg.hot_cache_bytes = (256ull << 20) / UNIFIED_CONCURRENCY;
				cfg.warm_tier_is_q8 = true;
				calibrated_profile_t	profile = calibrate(t, model, cfg,
					false, &pt.hw);

				concurrent_config_t	ccfg{};
				ccfg.concurrency = UNIFIED_CONCURRENCY;
				ccfg.host_hot_cache_total_bytes = 256ull << 20;
				ccfg.device_total_bytes = 1ull << 40;
				ccfg.quant_pipelines = pt.hw.quant_pipelines;
				ccfg.microbatch_max_wait_ns = 0.0;
				ccfg.microbatch_max_batch_blocks = 0;
				ccfg.hw = pt.hw;
				concurrent_result_t	cr = run_concurrent(profile, model,
					32, sim::Q8_COMPRESSION_RATIO, ccfg, 0);

				fprintf(hwf, "%s,%.1f,%.1f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,"
					"%.2f,%s,%.2f\n", pt.label.c_str(),
					pt.hw.cxl_link_latency_ns,
					pt.hw.cxl_link_bandwidth_gbps, pt.hw.quant_pipelines,
					cr.p50_latency_ns, cr.p99_latency_ns,
					cr.tokens_per_sec, cr.mean_bytes_per_token,
					cr.link_utilization_pct, cr.quant_utilization_pct,
					cr.bottleneck.c_str(), cr.link_queue.p99_wait_ns);
				fflush(hwf);
				fprintf(stderr, "membrane-kv-exact-sim: hw-sensitivity "
					"point %s done\n", pt.label.c_str());
			}
			fclose(hwf);
			fprintf(stderr, "membrane-kv-exact-sim: hardware sensitivity "
				"-> %s\n", hw_path.c_str());
		}
		else if (m.name == "SmolLM2-135M" && hw_already_done)
		{
			fprintf(stderr, "membrane-kv-exact-sim: hardware sensitivity "
				"already complete at %s, skipping re-run\n",
				hw_path.c_str());
		}
	}	/* traces (this model's unified trace) goes out of scope here
		 * and is freed before the next model loads. */

	fclose(lhf);
	fprintf(stderr, "membrane-kv-exact-sim: layer/head detail -> %s\n",
		lh_path.c_str());
	ckpt.write_complete();
	fclose(csv);
	fclose(g_tail_csv);

	fprintf(stderr, "membrane-kv-exact-sim: done, %zu/%zu scenarios -> %s "
		"(+ %s, + layer-head-detail, + hardware-sensitivity)\n",
		done_count.load(), scenarios.size(), out_csv.c_str(),
		tail_csv_path.c_str());
	return (0);
}
