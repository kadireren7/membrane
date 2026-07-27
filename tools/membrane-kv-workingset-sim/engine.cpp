#include <algorithm>
#include <unordered_set>
#include <vector>

#include "engine.h"
#include "wssim_config.h"

namespace wssim
{

static double	percentile(std::vector<double> v, double p)
{
	if (v.empty())
		return (0.0);
	std::sort(v.begin(), v.end());
	size_t	idx = (size_t)(p * (double)(v.size() - 1));
	return (v[idx]);
}

static double	transfer_ns(uint64_t bytes, int pipelines)
{
	double	effective_bw = std::min(sim::CXL_LINK_BANDWIDTH_GBPS,
		sim::NEARMEM_PIPELINE_BYTES_PER_NS * pipelines);
	return (sim::CXL_LINK_LATENCY_NS + (double)bytes / effective_bw);
}

static double	bytes_that_fit(double ns_budget, int pipelines)
{
	double	effective_bw = std::min(sim::CXL_LINK_BANDWIDTH_GBPS,
		sim::NEARMEM_PIPELINE_BYTES_PER_NS * pipelines);
	double	remaining = ns_budget - sim::CXL_LINK_LATENCY_NS;
	if (remaining <= 0.0)
		return (0.0);
	return (remaining * effective_bw);
}

static bool	vec_contains(const std::vector<uint32_t> &v, uint32_t x)
{
	return (std::find(v.begin(), v.end(), x) != v.end());
}

scenario_result_t	run_scenario_calibration(const attn_trace_t &trace,
						const model_calibration_t &model,
						const scenario_config_t &cfg,
						std::vector<per_step_calib_t> *out_steps,
						layer_head_stats_t *out_layer_head,
						coalescing_stats_t *out_coalescing)
{
	scenario_result_t	res{};
	uint32_t			group_size;
	uint32_t			n_kv_group;
	double				compression;
	double				bytes_per_channel_per_token;

	group_size = model.n_head_kv > 0 ? (trace.n_head / model.n_head_kv) : 1;
	if (group_size == 0)
		group_size = 1;
	n_kv_group = model.n_head_kv;
	compression = cfg.warm_tier_is_q8
		? sim::Q8_COMPRESSION_RATIO : sim::Q4_COMPRESSION_RATIO;
	bytes_per_channel_per_token = (double)model.bytes_per_token_total
		/ (double)(model.n_layer * n_kv_group);

	hot_cache_t	cache(cfg.hot_cache_bytes, cfg.eviction);
	std::vector<channel_predictor_t>	predictors;
	for (uint32_t l = 0; l < trace.n_layer; l++)
		for (uint32_t g = 0; g < n_kv_group; g++)
			predictors.emplace_back(cfg.block_size_tokens, policy_params_t{});

	std::unordered_set<uint64_t>	ever_fetched;
	auto	fetched_key = [](const cache_key_t &k) -> uint64_t
	{
		return (((uint64_t)k.layer << 40) | ((uint64_t)k.kv_group << 24)
			| (uint64_t)k.block_id);
	};

	std::vector<uint64_t>	layer_hits;
	std::vector<uint64_t>	layer_checks;
	std::vector<uint64_t>	head_hits;
	std::vector<uint64_t>	head_checks;
	if (out_layer_head != nullptr)
	{
		layer_hits.assign(trace.n_layer, 0);
		layer_checks.assign(trace.n_layer, 0);
		head_hits.assign(trace.n_head, 0);
		head_checks.assign(trace.n_head, 0);
	}

	if (out_coalescing != nullptr)
		*out_coalescing = coalescing_stats_t{};

	std::vector<double>	step_latencies;
	uint64_t	total_transferred_bytes = 0;
	uint64_t	total_hit_checks = 0;
	uint64_t	total_hits = 0;
	uint64_t	total_redundant_fetches = 0;
	uint64_t	total_wasted_prefetch_bytes = 0;
	uint64_t	total_working_set_blocks = 0;
	uint64_t	total_precision_num = 0;
	uint64_t	total_precision_den = 0;
	uint64_t	total_recall_num = 0;
	uint64_t	total_recall_den = 0;
	uint64_t	total_late_fetches = 0;
	uint64_t	total_prefetch_attempts = 0;
	uint64_t	total_prefetched_ok = 0;

	step_latencies.reserve(trace.step_count);
	if (out_steps != nullptr)
		out_steps->reserve(trace.step_count);
	for (uint32_t step = 0; step < trace.step_count; step++)
	{
		uint32_t	current_num_blocks = (trace.prompt_len + step + 1
				+ cfg.block_size_tokens - 1) / cfg.block_size_tokens;
		double	slack_ns = model.compute_ns_per_step;
		double	slack_bytes = bytes_that_fit(slack_ns,
			DEFAULT_QUANT_PIPELINES);
		uint64_t	prefetch_budget_used = 0;
		uint64_t	metadata_checks = 0;
		uint64_t	exposed_bytes = 0;

		size_t	pred_idx = 0;
		for (uint32_t l = 0; l < trace.n_layer; l++)
		{
			for (uint32_t g = 0; g < n_kv_group; g++, pred_idx++)
			{
				channel_predictor_t	&pred = predictors[pred_idx];
				std::vector<uint32_t>	ground_truth
					= trace.ground_truth_blocks(step, l, g, group_size);
				std::vector<uint32_t>	predicted = pred.predict(cfg.policy,
					step, current_num_blocks, ground_truth);

				total_working_set_blocks += predicted.size();

				/* Prefetch dispatch: ranking order from the policy,
				 * bounded by this step's slack bandwidth budget. */
				for (uint32_t b : predicted)
				{
					cache_key_t	key{l, g, b};
					metadata_checks++;
					if (cache.contains(key))
						continue ;
					uint64_t	blk_bytes = (uint64_t)(bytes_per_channel_per_token
						* cfg.block_size_tokens / compression);
					if (prefetch_budget_used + blk_bytes > (uint64_t)slack_bytes)
						continue ;
					prefetch_budget_used += blk_bytes;
					total_prefetch_attempts++;
					bool	is_useful = vec_contains(ground_truth, b);
					if (!is_useful)
						total_wasted_prefetch_bytes += blk_bytes;
					else
						total_prefetched_ok++;
					uint64_t	key64 = fetched_key(key);
					if (ever_fetched.count(key64))
						total_redundant_fetches++;
					ever_fetched.insert(key64);
					cache.insert(key, blk_bytes, 1.0);
				}

				/* Real need this step: ground truth blocks, checked
				 * against the (now possibly-freshly-prefetched) hot
				 * cache. */
				uint64_t	channel_hits_this_step = 0;
				std::vector<uint32_t>	missed_this_channel;
				for (uint32_t b : ground_truth)
				{
					cache_key_t	key{l, g, b};
					metadata_checks++;
					total_hit_checks++;
					if (out_layer_head != nullptr)
						layer_checks[l]++;
					if (cache.contains(key))
					{
						cache.touch_hit(key, 1.0);
						total_hits++;
						channel_hits_this_step++;
						if (out_layer_head != nullptr)
							layer_hits[l]++;
						continue ;
					}
					uint64_t	blk_bytes = (uint64_t)(bytes_per_channel_per_token
						* cfg.block_size_tokens / compression);
					exposed_bytes += blk_bytes;
					total_late_fetches++;
					if (out_coalescing != nullptr || out_layer_head != nullptr)
						missed_this_channel.push_back(b);
					uint64_t	key64 = fetched_key(key);
					if (ever_fetched.count(key64))
						total_redundant_fetches++;
					ever_fetched.insert(key64);
					cache.insert(key, blk_bytes, 1.0);
				}
				if (out_coalescing != nullptr && !missed_this_channel.empty())
				{
					uint64_t	blk_bytes = (uint64_t)(bytes_per_channel_per_token
						* cfg.block_size_tokens / compression);
					out_coalescing->naive_request_count
						+= missed_this_channel.size();
					out_coalescing->real_needed_bytes
						+= missed_this_channel.size() * blk_bytes;
					/* Sorted (ground_truth already is): greedily group
					 * consecutive missed ids into one request whenever
					 * the gap to the next missed id is within the
					 * coalescing window -- the request then spans
					 * [group_min, group_max], paying for any non-missed
					 * blocks caught inside that span as real padding. */
					size_t	i = 0;
					while (i < missed_this_channel.size())
					{
						uint32_t	group_min = missed_this_channel[i];
						uint32_t	group_max = group_min;
						size_t	j = i + 1;
						while (j < missed_this_channel.size()
								&& missed_this_channel[j] - group_max
									<= cfg.coalescing_window)
						{
							group_max = missed_this_channel[j];
							j++;
						}
						out_coalescing->coalesced_request_count++;
						out_coalescing->transferred_bytes_with_padding
							+= (uint64_t)(group_max - group_min + 1) * blk_bytes;
						i = j;
					}
				}
				/* Per-head resolution: a head's individual block is
				 * "hit" only if it was ALREADY resident before this
				 * step's misses were serviced -- checked against
				 * `missed_this_channel` (captured during the
				 * ground_truth loop above, BEFORE any of this step's
				 * misses were inserted), not by re-querying the cache
				 * now, which would trivially read back 100% (every
				 * ground-truth block, hit or miss, is resident by the
				 * time this point in the loop is reached -- a real
				 * bug caught during this phase's own development: an
				 * earlier version re-queried post-insertion and
				 * reported every head at exactly 100% hit rate,
				 * always, which was the tell). */
				if (out_layer_head != nullptr)
				{
					for (uint32_t h = g * group_size;
							h < (g + 1) * group_size && h < trace.n_head; h++)
					{
						const membrane_attntrace_entry_t	*he
							= trace.at(step, l, h);
						for (uint32_t k = 0; k < trace.top_k; k++)
						{
							if (he[k].block_id == UINT32_MAX)
								continue ;
							head_checks[h]++;
							if (!vec_contains(missed_this_channel,
									he[k].block_id))
								head_hits[h]++;
						}
					}
				}

				size_t	inter = 0;
				for (uint32_t b : predicted)
					if (vec_contains(ground_truth, b))
						inter++;
				total_precision_num += inter;
				total_precision_den += predicted.size();
				total_recall_num += inter;
				total_recall_den += ground_truth.size();

				pred.observe(step, ground_truth);
			}
		}

		total_transferred_bytes += prefetch_budget_used + exposed_bytes;
		if (out_steps != nullptr)
			out_steps->push_back({prefetch_budget_used, exposed_bytes});

		double	metadata_ns = metadata_checks * METADATA_LOOKUP_NS_PER_BLOCK
			+ metadata_checks * HOTCACHE_LOOKUP_NS_PER_BLOCK;
		double	exposed_ns = exposed_bytes > 0
			? transfer_ns(exposed_bytes, DEFAULT_QUANT_PIPELINES)
			+ (double)exposed_bytes / sim::NEARMEM_PIPELINE_BYTES_PER_NS
			: 0.0;
		double	memory_ns = metadata_ns + exposed_ns;
		double	step_latency = std::max(model.compute_ns_per_step, memory_ns);
		step_latencies.push_back(step_latency);
	}

	res.policy_name = policy_name(cfg.policy);
	res.eviction_name = eviction_policy_name(cfg.eviction);
	res.block_size_tokens = cfg.block_size_tokens;
	res.hot_cache_bytes = cfg.hot_cache_bytes;
	res.p50_latency_ns = percentile(step_latencies, 0.50);
	res.p95_latency_ns = percentile(step_latencies, 0.95);
	res.p99_latency_ns = percentile(step_latencies, 0.99);
	res.mean_transferred_bytes_per_token = trace.step_count
		? (double)total_transferred_bytes / trace.step_count : 0.0;
	res.metadata_overhead_ns_per_token = trace.step_count
		? (double)(total_hit_checks) * (METADATA_LOOKUP_NS_PER_BLOCK
			+ HOTCACHE_LOOKUP_NS_PER_BLOCK) / trace.step_count : 0.0;
	res.hot_cache_hit_rate = total_hit_checks
		? (double)total_hits / total_hit_checks : 0.0;
	res.redundant_fetches = total_redundant_fetches;
	res.wasted_prefetch_bytes = total_wasted_prefetch_bytes;
	res.mean_working_set_blocks = trace.step_count
		? (double)total_working_set_blocks
			/ (trace.step_count * trace.n_layer * n_kv_group) : 0.0;
	res.precision = total_precision_den
		? (double)total_precision_num / total_precision_den : 0.0;
	res.recall = total_recall_den
		? (double)total_recall_num / total_recall_den : 0.0;
	res.prefetch_hit_rate = total_hit_checks
		? (double)(total_hit_checks - total_late_fetches) / total_hit_checks
		: 0.0;
	res.late_fetch_rate = total_hit_checks
		? (double)total_late_fetches / total_hit_checks : 0.0;
	res.false_prefetch_rate = total_prefetch_attempts
		? (double)(total_prefetch_attempts - total_prefetched_ok)
			/ total_prefetch_attempts : 0.0;
	res.additional_link_traffic_bytes = total_wasted_prefetch_bytes;

	if (out_layer_head != nullptr)
	{
		out_layer_head->per_layer_hit_rate.resize(trace.n_layer);
		for (uint32_t l = 0; l < trace.n_layer; l++)
			out_layer_head->per_layer_hit_rate[l] = layer_checks[l]
				? (double)layer_hits[l] / layer_checks[l] : 0.0;
		out_layer_head->per_head_hit_rate.resize(trace.n_head);
		for (uint32_t h = 0; h < trace.n_head; h++)
			out_layer_head->per_head_hit_rate[h] = head_checks[h]
				? (double)head_hits[h] / head_checks[h] : 0.0;
	}
	return (res);
}

scenario_result_t	run_scenario(const attn_trace_t &trace,
						const model_calibration_t &model,
						const scenario_config_t &cfg)
{
	return (run_scenario_calibration(trace, model, cfg, nullptr, nullptr));
}

}	/* namespace wssim */
