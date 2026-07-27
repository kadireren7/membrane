#ifndef MEMBRANE_EXACTSIM_ENGINE_H
#define MEMBRANE_EXACTSIM_ENGINE_H

#include <cstdint>
#include <string>

#include "calibrate.h"
#include "engine.h"

namespace exactsim
{

/*
 * Real, concurrent, capacity-bound exact-retrieval scenario -- reuses
 * tools/membrane-cxl-sim/sim_engine.h's k_server_resource_t for
 * genuine shared CXL-link/quant-engine contention across all
 * `concurrency` sequences, processed in true global event-time order
 * (the same discrete-event design Phase 6.1 established), replaying
 * `profile`'s real calibrated per-step demand for each sequence
 * (small deterministic per-sequence jitter added so concurrent
 * sequences are not bit-identical clones -- same reasoning as Phase
 * 6.1's own synthetic-trace jitter).
 *
 * EXACT semantics, enforced structurally, not just asserted: no data
 * is ever dropped (device capacity is checked, never "solved" by
 * silently discarding a block); a sequence's step cannot complete
 * until its compulsory-miss fetch for that step (if any) actually
 * finishes -- see run_concurrent's implementation for exactly where
 * that dependency is enforced.
 */
struct concurrent_config_t
{
	uint32_t	concurrency;
	uint64_t	host_hot_cache_total_bytes;	/* reporting only -- the
			 * per-sequence split (host_hot_cache_total_bytes /
			 * concurrency) must already match what `profile` was
			 * calibrated with. */
	uint64_t	device_total_bytes;
	int			quant_pipelines;
	/* 0 = dispatch every compulsory-miss request immediately (no
	 * batching). >0 = real time-quantum micro-batching: requests
	 * arriving within the same `microbatch_max_wait_ns`-wide window
	 * are combined into one dispatch (split into
	 * `microbatch_max_batch_blocks`-sized chunks if the combined size
	 * would exceed that) -- section 8's max-wait/max-batch parameters,
	 * a disclosed simplification of true threshold-triggered batching
	 * (fixed time quanta instead of "whichever threshold hits first"),
	 * chosen because it composes cleanly with the discrete-event
	 * queue without a separate deferred-wakeup mechanism. */
	double		microbatch_max_wait_ns;
	uint32_t	microbatch_max_batch_blocks;
};

struct concurrent_result_t
{
	double		p50_latency_ns;
	double		p95_latency_ns;
	double		p99_latency_ns;
	double		tokens_per_sec;
	uint64_t	sequences_fit;
	uint32_t	concurrency;
	double		mean_bytes_per_token;
	double		link_utilization_pct;
	double		quant_utilization_pct;
	std::string	bottleneck;
	/* Capacity-bound flags (section 4's explicit request): host is
	 * flagged from the calibration's own real hit rate (a hit rate
	 * measurably below what the SAME policy achieves with a
	 * functionally unlimited cache means the swept host budget could
	 * not hold the real working set); device is flagged from an
	 * actual, checked per-sequence byte-budget comparison, not
	 * inferred. */
	bool		host_capacity_bound;
	bool		device_capacity_bound;
	uint64_t	total_link_bytes;
	uint64_t	total_quant_bytes;
};

concurrent_result_t	run_concurrent(const calibrated_profile_t &profile,
							const wssim::model_calibration_t &model,
							uint32_t block_size_tokens,
							double compression_ratio,
							const concurrent_config_t &cfg);

}	/* namespace exactsim */

#endif
