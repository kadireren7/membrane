/*
 * Policy, attn-trace-regroup, and engine determinism tests (Phase 6.2
 * item 15). No llama.cpp dependency -- built unconditionally, same
 * pattern as test_checkpoint.cpp / test_hotcache.cpp.
 */

#include <cstdio>
#include <cstdlib>

#include "attn_workload.h"
#include "engine.h"
#include "policy.h"

#define TEST_ASSERT(cond, msg) \
	do { \
		if (!(cond)) \
		{ \
			fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
			abort(); \
		} \
	} while (0)


using namespace wssim;

static void	test_full_returns_all_blocks(void)
{
	channel_predictor_t	p(32, policy_params_t{});
	std::vector<uint32_t>	out = p.predict(policy_t::FULL, 5, 7, {});

	TEST_ASSERT(out.size() == 7, "out.size() == 7");
	for (uint32_t i = 0; i < 7; i++)
		TEST_ASSERT(out[i] == i, "out[i] == i");
	printf("PASS test_full_returns_all_blocks\n");
}

static void	test_oracle_returns_ground_truth_exactly(void)
{
	channel_predictor_t	p(32, policy_params_t{});
	std::vector<uint32_t>	gt = {2, 5, 9};
	std::vector<uint32_t>	out = p.predict(policy_t::ORACLE, 5, 20, gt);

	TEST_ASSERT(out == gt, "out == gt");
	printf("PASS test_oracle_returns_ground_truth_exactly\n");
}

static void	test_sliding_window_respects_block_count(void)
{
	policy_params_t	params;
	params.sliding_window_tokens = 64;	/* 2 blocks at block_size=32 */
	channel_predictor_t	p(32, params);
	std::vector<uint32_t>	out = p.predict(policy_t::SLIDING_WINDOW, 5, 10, {});

	TEST_ASSERT(out.size() == 2, "out.size() == 2");
	TEST_ASSERT(out[0] == 8 && out[1] == 9, "out[0] == 8 && out[1] == 9");
	printf("PASS test_sliding_window_respects_block_count\n");
}

static void	test_topk_lag1_uses_previous_ground_truth(void)
{
	channel_predictor_t	p(32, policy_params_t{});
	std::vector<uint32_t>	gt0 = {1, 2};

	p.observe(0, gt0);
	std::vector<uint32_t>	out = p.predict(policy_t::TOPK_LAG1, 1, 10, {});
	TEST_ASSERT(out == gt0, "out == gt0");
	printf("PASS test_topk_lag1_uses_previous_ground_truth\n");
}

static void	test_heavy_hitter_favors_frequent_block(void)
{
	policy_params_t	params;
	params.heavy_hitter_top_n = 1;
	channel_predictor_t	p(32, params);

	for (uint32_t s = 0; s < 5; s++)
		p.observe(s, {3});	/* block 3 seen every step */
	p.observe(5, {9});	/* block 9 seen once */
	std::vector<uint32_t>	out = p.predict(policy_t::HEAVY_HITTER, 6, 20, {});
	TEST_ASSERT(out.size() == 1 && out[0] == 3, "out.size() == 1 && out[0] == 3");
	printf("PASS test_heavy_hitter_favors_frequent_block\n");
}

static void	test_predictor_is_deterministic(void)
{
	policy_params_t	params;
	auto	run = [&](void) -> std::vector<uint32_t>
	{
		channel_predictor_t	p(32, params);
		for (uint32_t s = 0; s < 8; s++)
			p.observe(s, {s % 3, (s + 1) % 5});
		return (p.predict(policy_t::MEMBRANE_PREDICTIVE, 8, 20, {}));
	};
	std::vector<uint32_t>	a = run();
	std::vector<uint32_t>	b = run();

	TEST_ASSERT(a == b, "a == b");
	printf("PASS test_predictor_is_deterministic\n");
}

static membrane_attntrace_entry_t	e(uint32_t block_id, float score)
{
	membrane_attntrace_entry_t	x;

	x.block_id = block_id;
	x.score = score;
	return (x);
}

static void	test_regroup_coarsen_sums_scores(void)
{
	attn_trace_t	t;

	t.model = "test";
	t.is_real_capture = true;
	t.n_layer = 1;
	t.n_head = 1;
	t.block_size_tokens = 32;
	t.prompt_len = 0;
	t.step_count = 1;
	t.top_k = 4;
	/* Native blocks 0 and 1 (both merge into coarse block 0 at 64) get
	 * scores 0.3 and 0.2 -> merged score 0.5. */
	t.entries = {e(0, 0.3f), e(1, 0.2f), e(5, 0.1f), e(UINT32_MAX, 0.0f)};

	attn_trace_t	coarse = regroup_to_block_size(t, 64);
	const membrane_attntrace_entry_t	*out = coarse.at(0, 0, 0);
	bool	found_merged = false;
	for (uint32_t k = 0; k < coarse.top_k; k++)
		if (out[k].block_id == 0)
		{
			found_merged = true;
			TEST_ASSERT(out[k].score > 0.49f && out[k].score < 0.51f, "out[k].score > 0.49f && out[k].score < 0.51f");
		}
	TEST_ASSERT(found_merged, "found_merged");
	printf("PASS test_regroup_coarsen_sums_scores\n");
}

static void	test_extend_synthetic_keeps_sink_fixed(void)
{
	attn_trace_t	t;

	t.model = "test";
	t.is_real_capture = true;
	t.n_layer = 1;
	t.n_head = 1;
	t.block_size_tokens = 32;
	t.prompt_len = 32;
	t.step_count = 2;
	t.top_k = 2;
	t.entries = {e(0, 0.9f), e(1, 0.1f), e(0, 0.8f), e(1, 0.2f)};

	attn_trace_t	ext = extend_synthetic(t, 6, 42u);
	/* Step 4 falls in the second cycle (cycle 2); block 0 (sink) must
	 * still read as block 0, never shifted. */
	const membrane_attntrace_entry_t	*out = ext.at(4, 0, 0);
	bool	saw_block0 = false;
	for (uint32_t k = 0; k < ext.top_k; k++)
		if (out[k].block_id == 0)
			saw_block0 = true;
	TEST_ASSERT(saw_block0, "saw_block0");
	printf("PASS test_extend_synthetic_keeps_sink_fixed\n");
}

static void	test_extend_synthetic_is_deterministic(void)
{
	attn_trace_t	t;

	t.model = "test";
	t.is_real_capture = true;
	t.n_layer = 1;
	t.n_head = 1;
	t.block_size_tokens = 32;
	t.prompt_len = 32;
	t.step_count = 2;
	t.top_k = 2;
	t.entries = {e(0, 0.9f), e(1, 0.1f), e(0, 0.8f), e(1, 0.2f)};

	attn_trace_t	a = extend_synthetic(t, 10, 7u);
	attn_trace_t	b = extend_synthetic(t, 10, 7u);
	TEST_ASSERT(a.entries.size() == b.entries.size(), "a.entries.size() == b.entries.size()");
	for (size_t i = 0; i < a.entries.size(); i++)
	{
		TEST_ASSERT(a.entries[i].block_id == b.entries[i].block_id, "a.entries[i].block_id == b.entries[i].block_id");
		TEST_ASSERT(a.entries[i].score == b.entries[i].score, "a.entries[i].score == b.entries[i].score");
	}
	printf("PASS test_extend_synthetic_is_deterministic\n");
}

static attn_trace_t	tiny_trace(void)
{
	attn_trace_t	t;

	t.model = "tiny";
	t.is_real_capture = true;
	t.n_layer = 1;
	t.n_head = 1;	/* 1 kv group */
	t.block_size_tokens = 32;
	t.prompt_len = 64;
	t.step_count = 4;
	t.top_k = 2;
	t.entries = {
		e(0, 0.6f), e(1, 0.4f),
		e(0, 0.5f), e(1, 0.5f),
		e(1, 0.7f), e(2, 0.3f),
		e(2, 0.9f), e(1, 0.1f),
	};
	return (t);
}

static void	test_oracle_scenario_has_perfect_precision_recall(void)
{
	attn_trace_t	t = tiny_trace();
	model_calibration_t	calib{"tiny", 1, 1, 512, 1000.0};
	scenario_config_t	cfg{};

	cfg.policy = policy_t::ORACLE;
	cfg.eviction = eviction_policy_t::LRU;
	cfg.block_size_tokens = 32;
	cfg.hot_cache_bytes = 1ull << 30;
	cfg.warm_tier_is_q8 = true;

	scenario_result_t	r = run_scenario(t, calib, cfg);
	TEST_ASSERT(r.precision > 0.999, "r.precision > 0.999");
	TEST_ASSERT(r.recall > 0.999, "r.recall > 0.999");
	printf("PASS test_oracle_scenario_has_perfect_precision_recall\n");
}

static void	test_engine_is_deterministic(void)
{
	attn_trace_t	t = tiny_trace();
	model_calibration_t	calib{"tiny", 1, 1, 512, 1000.0};
	scenario_config_t	cfg{};

	cfg.policy = policy_t::MEMBRANE_PREDICTIVE;
	cfg.eviction = eviction_policy_t::SEGMENTED_LRU;
	cfg.block_size_tokens = 32;
	cfg.hot_cache_bytes = 4096;
	cfg.warm_tier_is_q8 = true;

	scenario_result_t	a = run_scenario(t, calib, cfg);
	scenario_result_t	b = run_scenario(t, calib, cfg);
	TEST_ASSERT(a.mean_transferred_bytes_per_token == b.mean_transferred_bytes_per_token, "a.mean_transferred_bytes_per_token == b.mean_transferred_bytes_per_token");
	TEST_ASSERT(a.hot_cache_hit_rate == b.hot_cache_hit_rate, "a.hot_cache_hit_rate == b.hot_cache_hit_rate");
	TEST_ASSERT(a.precision == b.precision, "a.precision == b.precision");
	TEST_ASSERT(a.recall == b.recall, "a.recall == b.recall");
	printf("PASS test_engine_is_deterministic\n");
}

int	main(void)
{
	test_full_returns_all_blocks();
	test_oracle_returns_ground_truth_exactly();
	test_sliding_window_respects_block_count();
	test_topk_lag1_uses_previous_ground_truth();
	test_heavy_hitter_favors_frequent_block();
	test_predictor_is_deterministic();
	test_regroup_coarsen_sums_scores();
	test_extend_synthetic_keeps_sink_fixed();
	test_extend_synthetic_is_deterministic();
	test_oracle_scenario_has_perfect_precision_recall();
	test_engine_is_deterministic();
	return (0);
}
