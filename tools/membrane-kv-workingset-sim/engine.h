#ifndef MEMBRANE_WSSIM_ENGINE_H
#define MEMBRANE_WSSIM_ENGINE_H

#include <cstdint>
#include <string>

#include "attn_workload.h"
#include "hotcache.h"
#include "policy.h"

namespace wssim
{

/*
 * Real model geometry + real per-token byte rate needed to run a
 * scenario -- reused, not re-derived, from Phase 6.1's own real
 * measurements (docs/phase6-cxl-near-memory.md section 3 /
 * tools/membrane-cxl-sim/sim_config.h's SMOLLM2_* constants).
 */
struct model_calibration_t
{
	std::string	name;
	uint32_t	n_layer;
	uint32_t	n_head_kv;
	uint32_t	bytes_per_token_total;	/* REAL, all layers/heads, FP16 */
	double		compute_ns_per_step;	/* REAL end-to-end tok/s floor */
};

struct scenario_config_t
{
	policy_t			policy;
	eviction_policy_t	eviction;
	uint32_t			block_size_tokens;
	uint64_t			hot_cache_bytes;
	bool				warm_tier_is_q8;	/* fetched-tier compression */
};

struct scenario_result_t
{
	std::string	policy_name;
	std::string	eviction_name;
	uint32_t	block_size_tokens;
	uint64_t	hot_cache_bytes;

	double		p50_latency_ns;
	double		p95_latency_ns;
	double		p99_latency_ns;
	double		mean_transferred_bytes_per_token;
	double		metadata_overhead_ns_per_token;
	double		hot_cache_hit_rate;
	uint64_t	redundant_fetches;
	uint64_t	wasted_prefetch_bytes;
	double		mean_working_set_blocks;

	/* Prefetch-predictor quality (section 5). */
	double		precision;
	double		recall;
	double		prefetch_hit_rate;
	double		late_fetch_rate;
	double		false_prefetch_rate;
	uint64_t	additional_link_traffic_bytes;
};

/*
 * Runs one scenario end to end over `trace` (already regrouped to
 * cfg.block_size_tokens by the caller -- see
 * attn_workload.h's regroup_to_block_size) and returns aggregate
 * metrics. Implements the 8-stage per-decode-step pipeline: working-
 * set selection -> metadata lookup -> hot-cache lookup -> prefetch ->
 * CXL miss fetch -> decompression -> attention consumption ->
 * eviction (see docs/phase6-attention-working-set.md section on
 * simulator integration for the exact cost model and what it does
 * NOT model).
 */
scenario_result_t	run_scenario(const attn_trace_t &trace,
						const model_calibration_t &model,
						const scenario_config_t &cfg);

}	/* namespace wssim */

#endif
