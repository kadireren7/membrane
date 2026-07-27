/*
 * membrane-kv-exact-sim: Phase 6.3's concurrent, capacity-bound exact
 * sparse KV retrieval simulator. See
 * docs/phase6-exact-sparse-retrieval.md for the full writeup,
 * including this session's real, disclosed scope reductions vs. the
 * original 17-item spec (section 0).
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

/* ---- context tiers: REAL is the actual captured long trace; the
 * rest are extend_synthetic() extrapolations, clearly labeled in
 * every output row via `context_is_real`. ---- */
struct context_tier_t
{
	std::string	label;
	uint32_t	target_steps;	/* 0 == use the real trace as-is */
};

static std::vector<context_tier_t>	context_tiers()
{
	return {
		{"4K-real", 0},
		{"16K", 15872},
		{"32K", 32256},
		{"64K", 65024},
		{"128K", 130560},
	};
}

struct model_entry_t
{
	std::string				name;
	std::string				long_trace_path;
	model_calibration_t	calib;
};

/* One row of work: fields not relevant to a given `group` are left at
 * their default and rendered as 0/n.a. in the CSV -- one unified
 * schema keeps the sharded worker loop and checkpoint id scheme
 * simple (a single process_one() dispatch), at the cost of a wider
 * CSV than any one group strictly needs. */
struct scenario_desc_t
{
	std::string	group;
	std::string	model_name;
	std::string	context_label;
	policy_t	policy;
	uint64_t	host_cache_total_bytes;
	uint64_t	device_total_bytes;
	uint32_t	concurrency;
	uint32_t	coalescing_window;
	double		microbatch_max_wait_ns;
	uint32_t	microbatch_max_batch_blocks;

	std::string	id() const
	{
		char	buf[512];
		snprintf(buf, sizeof(buf), "%s|%s|%s|%s|%llu|%llu|%u|%u|%.1f|%u",
			group.c_str(), model_name.c_str(), context_label.c_str(),
			policy_name(policy), (unsigned long long)host_cache_total_bytes,
			(unsigned long long)device_total_bytes, concurrency,
			coalescing_window, microbatch_max_wait_ns,
			microbatch_max_batch_blocks);
		return (std::string(buf));
	}
};

static const char	*CSV_HEADER =
	"group,model,context_label,context_is_real,policy,"
	"host_cache_total_bytes,device_total_bytes,concurrency,"
	"coalescing_window,microbatch_max_wait_ns,microbatch_max_batch_blocks,"
	"p50_latency_ns,p95_latency_ns,p99_latency_ns,tokens_per_sec,"
	"sequences_fit,mean_bytes_per_token,link_util_pct,quant_util_pct,"
	"bottleneck,host_capacity_bound,device_capacity_bound,"
	"calib_hit_rate,calib_precision,calib_recall,calib_working_set_blocks,"
	"coalesced_naive_requests,coalesced_requests,coalesced_wasted_bytes\n";

/* Shared, read-only once built: one attn_trace_t per (model,
 * context_label). Built single-threaded before workers start so
 * concurrent process_one() calls never race on trace construction. */
struct trace_cache_t
{
	std::map<std::string, attn_trace_t>	by_key;

	static std::string	key(const std::string &model, const std::string &ctx)
	{
		return (model + "|" + ctx);
	}

	const attn_trace_t	&get(const std::string &model,
						const std::string &ctx) const
	{
		return (by_key.at(key(model, ctx)));
	}
};

static std::string	process_one(const scenario_desc_t &d,
						const trace_cache_t &traces,
						const std::map<std::string, model_calibration_t>
							&models)
{
	const attn_trace_t	&trace = traces.get(d.model_name, d.context_label);
	const model_calibration_t	&model = models.at(d.model_name);

	scenario_config_t	cfg{};
	cfg.policy = d.policy;
	cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
	cfg.block_size_tokens = 32;
	cfg.hot_cache_bytes = d.host_cache_total_bytes / d.concurrency;
	cfg.warm_tier_is_q8 = true;
	cfg.coalescing_window = d.coalescing_window;

	coalescing_stats_t	coal{};
	calibrated_profile_t	profile;
	profile.policy_name = policy_name(d.policy);
	profile.context_tokens = trace.prompt_len + trace.step_count;
	profile.source_is_real_capture = trace.is_real_capture;
	scenario_result_t	r = run_scenario_calibration(trace, model, cfg,
		&profile.steps, nullptr, d.coalescing_window > 0 ? &coal : nullptr);
	profile.hit_rate = r.hot_cache_hit_rate;
	profile.precision = r.precision;
	profile.recall = r.recall;
	profile.mean_working_set_blocks = r.mean_working_set_blocks;

	concurrent_config_t	ccfg{};
	ccfg.concurrency = d.concurrency;
	ccfg.host_hot_cache_total_bytes = d.host_cache_total_bytes;
	ccfg.device_total_bytes = d.device_total_bytes;
	ccfg.quant_pipelines = wssim::DEFAULT_QUANT_PIPELINES;
	ccfg.microbatch_max_wait_ns = d.microbatch_max_wait_ns;
	ccfg.microbatch_max_batch_blocks = d.microbatch_max_batch_blocks;

	double	compression = sim::Q8_COMPRESSION_RATIO;
	concurrent_result_t	cr = run_concurrent(profile, model, 32,
		compression, ccfg);

	char	buf[1024];
	snprintf(buf, sizeof(buf),
		"%s,%s,%s,%s,%s,%llu,%llu,%u,%u,%.1f,%u,"
		"%.2f,%.2f,%.2f,%.2f,%llu,%.2f,%.2f,%.2f,%s,%s,%s,"
		"%.4f,%.4f,%.4f,%.3f,%llu,%llu,%llu",
		d.group.c_str(), d.model_name.c_str(), d.context_label.c_str(),
		trace.is_real_capture ? "true" : "false", policy_name(d.policy),
		(unsigned long long)d.host_cache_total_bytes,
		(unsigned long long)d.device_total_bytes, d.concurrency,
		d.coalescing_window, d.microbatch_max_wait_ns,
		d.microbatch_max_batch_blocks, cr.p50_latency_ns, cr.p95_latency_ns,
		cr.p99_latency_ns, cr.tokens_per_sec,
		(unsigned long long)cr.sequences_fit, cr.mean_bytes_per_token,
		cr.link_utilization_pct, cr.quant_utilization_pct,
		cr.bottleneck.c_str(), cr.host_capacity_bound ? "true" : "false",
		cr.device_capacity_bound ? "true" : "false", profile.hit_rate,
		profile.precision, profile.recall, profile.mean_working_set_blocks,
		(unsigned long long)coal.naive_request_count,
		(unsigned long long)coal.coalesced_request_count,
		(unsigned long long)(coal.transferred_bytes_with_padding
			> coal.real_needed_bytes
			? coal.transferred_bytes_with_padding - coal.real_needed_bytes
			: 0));
	return (std::string(buf));
}

static std::vector<scenario_desc_t>	build_scenarios(
		const std::vector<model_entry_t> &models)
{
	std::vector<scenario_desc_t>	out;
	std::vector<policy_t>	seven_policies = {
		policy_t::NO_PREFETCH, policy_t::TOPK_LAG1, policy_t::RECENCY_SINKS,
		policy_t::HEAVY_HITTER, policy_t::RECENCY_FREQUENCY_HYBRID,
		policy_t::MEMBRANE_PREDICTIVE, policy_t::ORACLE};
	std::vector<uint64_t>	host_sizes = {
		64ull << 20, 256ull << 20, 1ull << 30, 4ull << 30, 8ull << 30};
	std::vector<uint64_t>	device_sizes = {
		512ull << 30, 1ull << 40, 2ull << 40};
	std::vector<policy_t>	three_policies = {
		policy_t::ORACLE, policy_t::NO_PREFETCH,
		policy_t::MEMBRANE_PREDICTIVE};
	std::vector<uint32_t>	concurrencies = {1, 8, 32, 128, 512};

	/* Group 1: capacity matrix -- broad, fixed at the real ~4.6K
	 * context tier (cheap to calibrate at full fidelity) and
	 * concurrency=8 (a representative mid-point). */
	for (const auto &m : models)
		for (policy_t p : seven_policies)
			for (uint64_t hb : host_sizes)
				for (uint64_t db : device_sizes)
					out.push_back({"capacity-matrix", m.name, "4K-real", p,
						hb, db, 8, 0, 0.0, 0});

	/* Group 2: context scaling, 135M only (128K-context calibration
	 * for the heavy-hitter-tracking policies costs ~100s each even
	 * after this phase's bounded-tracker fix -- section 0 discloses
	 * why the full model x policy x context product isn't repeated
	 * here). */
	for (const auto &ctx : context_tiers())
		for (policy_t p : three_policies)
			out.push_back({"context-scaling", "SmolLM2-135M", ctx.label, p,
				256ull << 20, 1ull << 40, 8, 0, 0.0, 0});

	/* Group 3: concurrency scaling, 135M only, fixed real 4K context
	 * -- per-sequence host cache shrinks with concurrency
	 * (256MiB/concurrency), so each point is its own real calibration
	 * at that budget, genuinely capturing capacity pressure. */
	for (uint32_t c : concurrencies)
		for (policy_t p : three_policies)
			out.push_back({"concurrency-scaling", "SmolLM2-135M", "4K-real",
				p, 256ull << 20, 1ull << 40, c, 0, 0.0, 0});

	/* Group 4: fetch coalescing granularity. */
	for (uint32_t w : {1u, 2u, 4u, 8u})
		out.push_back({"coalescing", "SmolLM2-135M", "4K-real",
			policy_t::MEMBRANE_PREDICTIVE, 256ull << 20, 1ull << 40, 8,
			w, 0.0, 0});

	/* Group 5: micro-batching Pareto -- same (max_wait, max_batch)
	 * grid Phase 6.1 itself swept (docs/phase6-cxl-near-memory.md
	 * section 6), reused for direct comparability, at concurrency=32
	 * so batching has enough concurrent arrivals to matter. */
	struct mb_point_t { double wait; uint32_t batch; };
	std::vector<mb_point_t>	mb_points = {
		{0.0, 1}, {0.0, 64}, {500.0, 1}, {500.0, 4}, {500.0, 64},
		{8000.0, 1}, {8000.0, 64}};
	for (const auto &mb : mb_points)
		out.push_back({"microbatch-pareto", "SmolLM2-135M", "4K-real",
			policy_t::MEMBRANE_PREDICTIVE, 256ull << 20, 1ull << 40, 32,
			0, mb.wait, mb.batch});

	/* Group 6: the 7 requested headline baseline comparisons.
	 * "full-scan CXL" / "compressed full-scan CXL" are NOT run here
	 * (FULL policy's cost to simulate is O(context^2) at this
	 * fidelity, section 0) -- computed analytically in main() instead
	 * and appended to the same CSV, clearly labeled group=
	 * "baseline-analytical" so they are never confused with a
	 * simulated row. */
	out.push_back({"baseline-oracle", "SmolLM2-135M", "4K-real",
		policy_t::ORACLE, 256ull << 20, 1ull << 40, 8, 0, 0.0, 0});
	out.push_back({"baseline-no-prefetch", "SmolLM2-135M", "4K-real",
		policy_t::NO_PREFETCH, 256ull << 20, 1ull << 40, 8, 0, 0.0, 0});
	out.push_back({"baseline-predictor", "SmolLM2-135M", "4K-real",
		policy_t::MEMBRANE_PREDICTIVE, 256ull << 20, 1ull << 40, 8, 0,
		0.0, 0});
	out.push_back({"baseline-predictor-coalesced", "SmolLM2-135M", "4K-real",
		policy_t::MEMBRANE_PREDICTIVE, 256ull << 20, 1ull << 40, 8, 4,
		0.0, 0});
	out.push_back({"baseline-predictor-coalesced-microbatched",
		"SmolLM2-135M", "4K-real", policy_t::MEMBRANE_PREDICTIVE,
		256ull << 20, 1ull << 40, 8, 4, 500.0, 4});

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

	void	maybe_print(size_t done, size_t total)
	{
		auto	now = std::chrono::steady_clock::now();
		double	since = std::chrono::duration<double>(now - last).count();
		if (since < 60.0 && done < total)
			return ;
		last = now;
		double	elapsed = std::chrono::duration<double>(now - start).count();
		double	rate = done > 0 ? elapsed / done : 0.0;
		double	eta = rate * (double)(total - done);
		fprintf(stderr, "[heartbeat] %zu/%zu wall=%.1fs eta=%.1fs\n",
			done, total, elapsed, eta);
	}
};

int	main(int argc, char **argv)
{
	std::string	trace_135m_long;
	std::string	trace_360m_long;
	std::string	out_csv = "benchmarks/cxl-sim/exact-sweep.csv";
	std::string	ckpt_path = "benchmarks/cxl-sim/exact-sweep.ckpt";
	unsigned	workers = std::thread::hardware_concurrency();

	for (int i = 1; i + 1 <= argc; i++)
	{
		if (strcmp(argv[i], "--trace-135m-long") == 0 && i + 1 < argc)
			trace_135m_long = argv[++i];
		else if (strcmp(argv[i], "--trace-360m-long") == 0 && i + 1 < argc)
			trace_360m_long = argv[++i];
		else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
			out_csv = argv[++i];
		else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc)
			ckpt_path = argv[++i];
		else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc)
			workers = (unsigned)atoi(argv[++i]);
	}
	if (trace_135m_long.empty())
	{
		fprintf(stderr, "usage: membrane-kv-exact-sim --trace-135m-long P "
			"[--trace-360m-long P] [--out CSV] [--checkpoint PATH] "
			"[--workers N]\n");
		return (2);
	}
	if (workers < 1)
		workers = 1;

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

	fprintf(stderr, "membrane-kv-exact-sim: loading real long traces...\n");
	trace_cache_t	traces;
	std::vector<context_tier_t>	tiers = context_tiers();
	for (const auto &m : models)
	{
		attn_trace_t	native;
		if (!load_attn_trace(m.long_trace_path, &native))
		{
			fprintf(stderr, "failed to load %s\n", m.long_trace_path.c_str());
			return (1);
		}
		for (const auto &tier : tiers)
		{
			/* Only 135M gets the full context-tier set built (Group 2
			 * uses 135M only) -- 360M only needs the real 4K-real
			 * tier for Group 1's capacity matrix. */
			if (m.name != "SmolLM2-135M" && tier.target_steps != 0)
				continue ;
			attn_trace_t	t = tier.target_steps == 0
				? native : extend_synthetic(native, tier.target_steps, 7u);
			traces.by_key[trace_cache_t::key(m.name, tier.label)] = t;
		}
	}
	fprintf(stderr, "membrane-kv-exact-sim: traces ready\n");

	std::vector<scenario_desc_t>	scenarios = build_scenarios(models);
	std::string	config_hash = sha256_hex_of_string(
		"phase6.3-v1;groups=capacity-matrix,context-scaling,"
		"concurrency-scaling,coalescing,microbatch-pareto,baselines");
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

	checkpoint_writer_t	ckpt;
	ckpt.open(ckpt_path, trace_hash, config_hash, prior.header_present);

	std::mutex	io_mutex;
	std::atomic<size_t>	next_index{0};
	std::atomic<size_t>	done_count{prior.completed_ids.size()};
	heartbeat_t	hb;

	auto	worker_fn = [&](void)
	{
		while (true)
		{
			size_t	idx = next_index.fetch_add(1);
			if (idx >= scenarios.size())
				return ;
			const scenario_desc_t	&d = scenarios[idx];
			std::string	id = d.id();
			{
				std::lock_guard<std::mutex>	lock(io_mutex);
				if (prior.completed_ids.count(id))
					continue ;
			}
			std::string	row = process_one(d, traces, model_by_name);
			{
				std::lock_guard<std::mutex>	lock(io_mutex);
				fprintf(csv, "%s\n", row.c_str());
				fflush(csv);
				ckpt.write_scenario(id, row);
				done_count++;
				hb.maybe_print(done_count.load(), scenarios.size());
			}
		}
	};

	fprintf(stderr, "membrane-kv-exact-sim: %zu scenarios, %u workers "
		"(%zu already complete)\n", scenarios.size(), workers,
		prior.completed_ids.size());
	std::vector<std::thread>	pool;
	for (unsigned w = 0; w < workers; w++)
		pool.emplace_back(worker_fn);
	for (auto &th : pool)
		th.join();
	ckpt.write_complete();

	/* Group 6's analytical baselines: FULL re-scan, uncompressed and
	 * Q8-compressed. Not simulated (see build_scenarios' comment) --
	 * mean bytes/token = bytes_per_token_total x mean outstanding
	 * context length across the run (grows linearly from prompt_len
	 * to the full context, so its average is the midpoint), divided
	 * by the compression ratio for the compressed variant. This is
	 * exactly Phase 6.1's own "full re-read every step" closed-form
	 * reasoning (docs/phase6-cxl-near-memory.md section 2), reused
	 * here rather than re-simulated. */
	{
		const attn_trace_t	&t = traces.get("SmolLM2-135M", "4K-real");
		double	mean_context = (double)t.prompt_len
			+ (double)t.step_count / 2.0;
		double	bytes_per_token = (double)sim::SMOLLM2_135M_BYTES_PER_TOKEN;
		double	raw_bytes_per_token = bytes_per_token * mean_context;
		double	compressed_bytes_per_token = raw_bytes_per_token
			/ sim::Q8_COMPRESSION_RATIO;
		fprintf(csv, "baseline-analytical,SmolLM2-135M,4K-real,true,"
			"full-scan-cxl-uncompressed,0,0,8,0,0.0,0,0,0,0,0,0,%.2f,0,0,"
			"n/a,false,false,0,0,0,0,0,0,0\n", raw_bytes_per_token);
		fprintf(csv, "baseline-analytical,SmolLM2-135M,4K-real,true,"
			"full-scan-cxl-compressed,0,0,8,0,0.0,0,0,0,0,0,0,%.2f,0,0,"
			"n/a,false,false,0,0,0,0,0,0,0\n", compressed_bytes_per_token);
	}
	fclose(csv);

	/* Section 6: per-layer/per-head hit-rate detail (item 6) -- a
	 * dedicated pass, not folded into the main sweep's unified row
	 * schema (per-layer/head arrays are variable-length and would
	 * force every other row to carry unused columns). One
	 * representative scenario per model: membrane-predictive, real 4K
	 * context, 256MiB/8 per-sequence hot cache -- cheap enough to run
	 * for real here rather than reuse a cached calibration. */
	{
		std::string	lh_path = out_csv.substr(0,
			out_csv.find_last_of('.')) + "-layer-head-detail.csv";
		FILE	*lhf = fopen(lh_path.c_str(), "w");
		fprintf(lhf, "model,resolution,index,hit_rate\n");
		for (const auto &m : models)
		{
			const attn_trace_t	&t = traces.get(m.name, "4K-real");
			scenario_config_t	cfg{};
			cfg.policy = policy_t::MEMBRANE_PREDICTIVE;
			cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
			cfg.block_size_tokens = 32;
			cfg.hot_cache_bytes = (256ull << 20) / 8;
			cfg.warm_tier_is_q8 = true;
			layer_head_stats_t	lh;
			run_scenario_calibration(t, m.calib, cfg, nullptr, &lh, nullptr);
			for (size_t l = 0; l < lh.per_layer_hit_rate.size(); l++)
				fprintf(lhf, "%s,layer,%zu,%.4f\n", m.name.c_str(), l,
					lh.per_layer_hit_rate[l]);
			for (size_t h = 0; h < lh.per_head_hit_rate.size(); h++)
				fprintf(lhf, "%s,head,%zu,%.4f\n", m.name.c_str(), h,
					lh.per_head_hit_rate[h]);
		}
		fclose(lhf);
		fprintf(stderr, "membrane-kv-exact-sim: layer/head detail -> %s\n",
			lh_path.c_str());
	}

	fprintf(stderr, "membrane-kv-exact-sim: done, %zu/%zu scenarios -> %s\n",
		done_count.load(), scenarios.size(), out_csv.c_str());
	return (0);
}
