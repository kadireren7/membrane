#include <algorithm>
#include <map>
#include <queue>
#include <vector>

#include "exact_engine.h"
#include "sim_config.h"
#include "sim_engine.h"

namespace exactsim
{

enum class event_kind_t { STEP, FLUSH };

struct event_t
{
	double			time;
	event_kind_t	kind;
	uint32_t		seq;	/* STEP only */
	uint32_t		step;	/* STEP only */
	uint64_t		quantum_id;	/* FLUSH only */
};

struct event_cmp_t
{
	bool	operator()(const event_t &a, const event_t &b) const
	{
		return (a.time > b.time);
	}
};

struct pending_req_t
{
	uint32_t	seq;
	uint32_t	step;
	uint64_t	bytes;
	double		compute_ready_ns;	/* this step's compute-bound floor */
};

static uint32_t	xorshift32(uint32_t *state)
{
	uint32_t	x = *state;

	x ^= x << 13; x ^= x >> 17; x ^= x << 5;
	*state = x;
	return (x);
}

concurrent_result_t	run_concurrent(const calibrated_profile_t &profile,
						const wssim::model_calibration_t &model,
						uint32_t block_size_tokens,
						double compression_ratio,
						const concurrent_config_t &cfg)
{
	concurrent_result_t	out{};
	uint32_t				N = cfg.concurrency > 0 ? cfg.concurrency : 1;

	int	pipelines = cfg.quant_pipelines > 0 ? cfg.quant_pipelines : 1;
	sim::k_server_resource_t	link(pipelines, "exact-cxl-link", 0.0);
	sim::k_server_resource_t	quant_engine(pipelines, "exact-quant-engine", 0.0);

	uint64_t	per_seq_device_budget = cfg.device_total_bytes / N;

	std::vector<std::vector<double>>	seq_latencies(N);
	std::vector<bool>					seq_capacity_exceeded(N, false);

	std::priority_queue<event_t, std::vector<event_t>, event_cmp_t>	pq;
	std::map<uint64_t, std::vector<pending_req_t>>						pending;
	uint64_t	total_link_bytes = 0;
	uint64_t	total_transferred_bytes = 0;
	uint64_t	completed_steps = 0;
	uint64_t	total_steps_all_seqs = 0;
	double		sim_end_ns = 0.0;
	uint32_t	jitter_state = 0xabcdef01u;

	double	quantum = cfg.microbatch_max_wait_ns > 0.0
		? cfg.microbatch_max_wait_ns : 1.0;
	double	bytes_per_channel_per_token = (model.n_layer && model.n_head_kv)
		? (double)model.bytes_per_token_total
			/ (double)(model.n_layer * model.n_head_kv) : 0.0;
	uint64_t	bytes_per_block = (uint64_t)(bytes_per_channel_per_token
		* block_size_tokens / compression_ratio);
	uint64_t	max_batch_bytes = cfg.microbatch_max_batch_blocks > 0
		? (uint64_t)cfg.microbatch_max_batch_blocks * bytes_per_block
		: UINT64_MAX;

	for (uint32_t s = 0; s < N; s++)
	{
		pq.push({0.0, event_kind_t::STEP, s, 0, 0});
		total_steps_all_seqs += profile.steps.size();
	}

	/* Flushes every request that landed in quantum `qid`, dispatching
	 * either one combined request (if it fits under max_batch_bytes)
	 * or several chunked ones -- either way through the SAME shared
	 * link/quant_engine resources, so genuine cross-sequence
	 * contention (including contention this microbatching itself
	 * creates) is reflected in every request's real completion time. */
	auto	flush_quantum = [&](uint64_t qid, double now)
	{
		auto	it = pending.find(qid);
		if (it == pending.end())
			return ;
		std::vector<pending_req_t>	&reqs = it->second;
		size_t	i = 0;
		while (i < reqs.size())
		{
			uint64_t	batch_bytes = 0;
			size_t	j = i;
			while (j < reqs.size() && batch_bytes + reqs[j].bytes
					<= max_batch_bytes)
			{
				batch_bytes += reqs[j].bytes;
				j++;
			}
			if (j == i)
			{
				batch_bytes = reqs[i].bytes;
				j = i + 1;
			}
			total_link_bytes += batch_bytes;
			double	completion = link.submit(now,
				sim::transfer_ns({sim::CXL_LINK_LATENCY_NS,
					sim::CXL_LINK_BANDWIDTH_GBPS}, (double)batch_bytes));
			completion = quant_engine.submit(completion,
				(double)batch_bytes / sim::NEARMEM_PIPELINE_BYTES_PER_NS);
			for (size_t k = i; k < j; k++)
			{
				double	step_complete = std::max(reqs[k].compute_ready_ns,
					completion);
				seq_latencies[reqs[k].seq].push_back(
					step_complete - reqs[k].compute_ready_ns
					+ model.compute_ns_per_step);
				sim_end_ns = std::max(sim_end_ns, step_complete);
				completed_steps++;
				pq.push({step_complete, event_kind_t::STEP, reqs[k].seq,
					reqs[k].step + 1, 0});
			}
			i = j;
		}
		pending.erase(it);
	};

	while (!pq.empty())
	{
		event_t	ev = pq.top();
		pq.pop();
		if (ev.kind == event_kind_t::FLUSH)
		{
			flush_quantum(ev.quantum_id, ev.time);
			continue ;
		}
		uint32_t	s = ev.seq;
		if (seq_capacity_exceeded[s] || ev.step >= profile.steps.size())
			continue ;

		uint64_t	device_needed = (uint64_t)((double)model.bytes_per_token_total
			* (double)(profile.context_tokens - profile.steps.size() + ev.step + 1)
			/ compression_ratio);
		if (device_needed > per_seq_device_budget)
		{
			seq_capacity_exceeded[s] = true;
			continue ;
		}

		const wssim::per_step_calib_t	&step = profile.steps[ev.step];
		xorshift32(&jitter_state);
		double	jitter = 1.0 + (((double)(jitter_state % 2001) - 1000.0)
			/ 1000.0) * 0.03;
		uint64_t	prefetch_bytes = (uint64_t)((double)step.prefetch_bytes
			* jitter);
		uint64_t	miss_bytes = (uint64_t)((double)step.compulsory_miss_bytes
			* jitter);
		total_transferred_bytes += prefetch_bytes + miss_bytes;

		/* Prefetch: real link traffic, contends for bandwidth, but
		 * does NOT gate this step's own completion (it was dispatched
		 * during the previous step's slack -- exactly what
		 * "prefetch" means; see wssim::engine.cpp for where that
		 * slack accounting happens during calibration). */
		if (prefetch_bytes > 0)
		{
			total_link_bytes += prefetch_bytes;
			link.submit(ev.time, sim::transfer_ns(
				{sim::CXL_LINK_LATENCY_NS, sim::CXL_LINK_BANDWIDTH_GBPS},
				(double)prefetch_bytes));
		}

		double	compute_ready = ev.time;
		if (miss_bytes == 0)
		{
			/* Nothing to wait on: this step completes at the real
			 * compute-bound floor -- attention was never gated on a
			 * fetch it didn't need. */
			double	step_complete = compute_ready + model.compute_ns_per_step;
			seq_latencies[s].push_back(step_complete - compute_ready);
			sim_end_ns = std::max(sim_end_ns, step_complete);
			completed_steps++;
			pq.push({step_complete, event_kind_t::STEP, s, ev.step + 1, 0});
			continue ;
		}
		if (cfg.microbatch_max_wait_ns <= 0.0)
		{
			/* No micro-batching: this sequence's miss is dispatched
			 * immediately, still through the SHARED resources, so it
			 * still queues behind any other sequence's in-flight
			 * request (real contention, just no deliberate batching
			 * delay added on top). Attention cannot complete until
			 * this fetch's completion -- the exact-mode dependency
			 * item 9 requires. */
			total_link_bytes += miss_bytes;
			double	completion = link.submit(ev.time, sim::transfer_ns(
				{sim::CXL_LINK_LATENCY_NS, sim::CXL_LINK_BANDWIDTH_GBPS},
				(double)miss_bytes));
			completion = quant_engine.submit(completion,
				(double)miss_bytes / sim::NEARMEM_PIPELINE_BYTES_PER_NS);
			double	step_complete = std::max(compute_ready
				+ model.compute_ns_per_step, completion);
			seq_latencies[s].push_back(step_complete - compute_ready);
			sim_end_ns = std::max(sim_end_ns, step_complete);
			completed_steps++;
			pq.push({step_complete, event_kind_t::STEP, s, ev.step + 1, 0});
			continue ;
		}
		/* Micro-batching: enqueue into this time quantum's pending
		 * list; schedule (once per quantum) a FLUSH event at the
		 * quantum's end boundary. */
		uint64_t	qid = (uint64_t)(ev.time / quantum);
		bool	first_in_quantum = pending.find(qid) == pending.end();
		pending[qid].push_back({s, ev.step, miss_bytes,
			compute_ready + model.compute_ns_per_step});
		if (first_in_quantum)
			pq.push({(double)(qid + 1) * quantum, event_kind_t::FLUSH,
				0, 0, qid});
	}

	std::vector<double>	all_latencies;
	uint64_t			fit_count = 0;
	for (uint32_t s = 0; s < N; s++)
	{
		if (!seq_capacity_exceeded[s])
			fit_count++;
		for (double v : seq_latencies[s])
			all_latencies.push_back(v);
	}
	std::sort(all_latencies.begin(), all_latencies.end());
	auto	pct = [&](double p) -> double
	{
		if (all_latencies.empty())
			return (0.0);
		size_t	idx = (size_t)(p * (double)(all_latencies.size() - 1));
		return (all_latencies[idx]);
	};

	out.p50_latency_ns = pct(0.50);
	out.p95_latency_ns = pct(0.95);
	out.p99_latency_ns = pct(0.99);
	out.tokens_per_sec = (sim_end_ns > 0.0 && fit_count > 0)
		? (double)completed_steps / (sim_end_ns / 1.0e9) : 0.0;
	out.sequences_fit = fit_count;
	out.concurrency = N;
	out.mean_bytes_per_token = total_steps_all_seqs
		? (double)total_transferred_bytes / total_steps_all_seqs : 0.0;
	out.link_utilization_pct = link.utilization(sim_end_ns);
	out.quant_utilization_pct = quant_engine.utilization(sim_end_ns);
	out.total_link_bytes = total_link_bytes;
	out.total_quant_bytes = total_link_bytes;
	out.host_capacity_bound = profile.hit_rate < 0.999;
	out.device_capacity_bound = fit_count < N;

	double	max_util = out.link_utilization_pct;
	out.bottleneck = "link";
	if (out.quant_utilization_pct > max_util)
	{
		max_util = out.quant_utilization_pct;
		out.bottleneck = "quant_engine";
	}
	if (fit_count < N)
		out.bottleneck = "device_capacity";
	else if (out.host_capacity_bound && max_util < 5.0)
		out.bottleneck = "host_cache_capacity";
	else if (max_util < 5.0)
		out.bottleneck = "compute";
	return (out);
}

}	/* namespace exactsim */
