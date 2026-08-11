#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime_core.h"
#include "test_helpers.h"

/* A deterministic, non-synthetic-generator pattern -- same convention
 * test_membrane_bench.c uses for its trace fixtures. */
static void	fill_block(uint16_t *out, uint32_t n, uint32_t seed_offset)
{
	uint32_t	i;

	i = 0;
	while (i < n)
	{
		out[i] = (uint16_t)((i + seed_offset) * 13u + 5u);
		i++;
	}
}

static void	test_mode_name_round_trip(void)
{
	membrane_runtime_mode_t	m;

	TEST_ASSERT(strcmp(membrane_runtime_mode_name(
			MEMBRANE_RUNTIME_MODE_BASELINE), "baseline") == 0, "baseline");
	TEST_ASSERT(strcmp(membrane_runtime_mode_name(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8), "shadow-q8") == 0, "shadow-q8");
	TEST_ASSERT(strcmp(membrane_runtime_mode_name(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE), "shadow-adaptive") == 0,
		"shadow-adaptive");
	TEST_ASSERT(membrane_runtime_mode_from_name("shadow-q8", &m)
		&& m == MEMBRANE_RUNTIME_MODE_SHADOW_Q8, "parse shadow-q8");
	TEST_ASSERT(membrane_runtime_mode_from_name("baseline", &m)
		&& m == MEMBRANE_RUNTIME_MODE_BASELINE, "parse baseline");
	TEST_ASSERT(membrane_runtime_mode_from_name("shadow-adaptive", &m)
		&& m == MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE,
		"parse shadow-adaptive");
	TEST_ASSERT(!membrane_runtime_mode_from_name("shadow-q4", &m),
		"q4 runtime mode is not exposed");
	TEST_ASSERT(!membrane_runtime_mode_from_name("bogus", &m),
		"unknown name rejected");
}

static void	test_baseline_mode_never_processes(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[256];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(MEMBRANE_RUNTIME_MODE_BASELINE, 8);
	TEST_ASSERT(c != NULL, "create");
	fill_block(block, 256, 0);
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 0, 0, block, 256) == MEMBRANE_OK,
		"observe succeeds");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.total_blocks == 0 && t.total_values_observed == 0
		&& t.layers_seen == 0 && t.accum.q4_blocks == 0
		&& t.accum.q8_blocks == 0,
		"baseline mode touches no counters at all -- true native path");
	membrane_runtime_collector_destroy(c);
}

static void	test_shadow_q8_forces_q8(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 8);
	fill_block(block, 128, 0);
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 3, 0, block, 128) == MEMBRANE_OK,
		"observe succeeds");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.accum.q4_blocks == 0 && t.accum.q8_blocks == 1,
		"shadow-q8 forces every block to Q8");
	TEST_ASSERT(t.k_blocks == 1 && t.v_blocks == 0, "counted as a K block");
	TEST_ASSERT(t.layers_seen == 1, "one distinct layer observed");
	membrane_runtime_collector_destroy(c);
}

static void	test_shadow_adaptive_assigns_one_precision_per_block(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE, 8);
	fill_block(block, 128, 0);
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 0, 1, block, 128) == MEMBRANE_OK,
		"observe succeeds");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.accum.q4_blocks + t.accum.q8_blocks == 1,
		"adaptive assigns exactly one precision to the one block");
	TEST_ASSERT(t.v_blocks == 1 && t.k_blocks == 0, "counted as a V block");
	membrane_runtime_collector_destroy(c);
}

static void	test_tail_values_excluded_not_processed(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						buf[300];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 8);
	fill_block(buf, 300, 0);
	/* 300 = 2*128 + 44 */
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 0, 0, buf, 300) == MEMBRANE_OK,
		"observe succeeds");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.total_blocks == 2, "only 2 full 128-element blocks");
	TEST_ASSERT(t.tail_values_excluded == 44, "44-element tail excluded");
	TEST_ASSERT(t.total_values_observed == 300,
		"total_values_observed counts everything, including the tail");
	membrane_runtime_collector_destroy(c);
}

static void	test_no_padding_zero_full_blocks(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						buf[64];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE, 8);
	fill_block(buf, 64, 0);
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 0, 0, buf, 64) == MEMBRANE_OK,
		"observe succeeds even with fewer than one block's worth");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.total_blocks == 0, "no full block, none processed");
	TEST_ASSERT(t.tail_values_excluded == 64,
		"all 64 values reported excluded, never silently padded");
	membrane_runtime_collector_destroy(c);
}

static void	test_layer_out_of_range_rejected(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 4);
	fill_block(block, 128, 0);
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), 4, 0, block, 128)
			== MEMBRANE_ERR_INVALID_ARG,
		"layer == max_layers rejected");
	TEST_ASSERT(membrane_runtime_observe_tensor(c,
			membrane_simd_best_backend(), -1, 0, block, 128)
			== MEMBRANE_ERR_INVALID_ARG,
		"negative layer rejected");
	membrane_runtime_collector_destroy(c);
}

static void	test_layers_seen_counts_distinct(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 8);
	fill_block(block, 128, 0);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 0,
		block, 128);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 1,
		block, 128);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 1, 0,
		block, 128);
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.layers_seen == 2,
		"layer 0 observed twice (K and V) still counts once; layer 1 once");
	TEST_ASSERT(t.total_blocks == 3, "3 observations, 3 blocks total");
	membrane_runtime_collector_destroy(c);
}

static void	test_weighted_mean_matches_pooled_formula(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						low[128];
	uint16_t						high[128];
	membrane_runtime_telemetry_t	t;
	double							expect;
	uint32_t						i;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE, 8);
	/* Small-magnitude content: plausibly Q4-eligible. */
	i = 0;
	while (i < 128)
	{
		low[i] = (uint16_t)(i % 4);
		i++;
	}
	/* Larger-magnitude/noisy content: plausibly forced to Q8. */
	fill_block(high, 128, 977);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 0,
		low, 128);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 1, 0,
		high, 128);
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.accum.q4_blocks + t.accum.q8_blocks == 2, "two blocks");
	expect = (t.accum.q4_err.sum + t.accum.q8_err.sum)
		/ (double)(t.accum.q4_blocks + t.accum.q8_blocks);
	TEST_ASSERT(fabs(membrane_runtime_weighted_mean_rel_l2_error(&t)
			- expect) < 1e-12,
		"weighted mean matches sum(err)/count pooled formula exactly");
	membrane_runtime_collector_destroy(c);
}

static void	test_max_rel_l2_is_max_of_both_precisions(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0, sizeof(t));
	t.accum.q4_err.max = 0.01;
	t.accum.q8_err.max = 0.03;
	TEST_ASSERT(membrane_runtime_max_rel_l2_error(&t) == 0.03,
		"max of the two precisions' maxes");
	t.accum.q4_err.max = 0.09;
	TEST_ASSERT(membrane_runtime_max_rel_l2_error(&t) == 0.09,
		"max flips correctly when Q4 is larger");
}

static void	test_storage_accounting_ratios(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0, sizeof(t));
	t.total_values_observed = 300;
	t.tail_values_excluded = 44;
	t.accum.encoded_bytes = 1000;
	TEST_ASSERT(membrane_runtime_fp16_bytes_observed(&t) == 600,
		"fp16 bytes observed includes the tail: 300*2");
	/* encoded values = 256, fp16 bytes of just those = 512; encoded_bytes
	 * (1000) exceeds that -- an implausible input for real Q4/Q8 output,
	 * but the ratio must still be a defined, non-negative value rather
	 * than a nonsensical negative "reduction". */
	TEST_ASSERT(membrane_runtime_theoretical_payload_reduction(&t) == 0.0,
		"clamped to 0 when encoded bytes exceed the full-block fp16 size");
	t.accum.encoded_bytes = 300;
	TEST_ASSERT(fabs(membrane_runtime_theoretical_payload_reduction(&t)
			- (512.0 - 300.0) / 512.0) < 1e-9,
		"typical case: encoded bytes smaller than full-block fp16 bytes");
}

static void	test_zero_observations_zero_ratios(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0, sizeof(t));
	TEST_ASSERT(membrane_runtime_weighted_mean_rel_l2_error(&t) == 0.0,
		"no blocks -> weighted mean is 0, not NaN/div-by-zero");
	TEST_ASSERT(membrane_runtime_theoretical_payload_reduction(&t) == 0.0,
		"no values observed -> reduction ratio is 0, not NaN");
	TEST_ASSERT(membrane_runtime_fp16_bytes_observed(&t) == 0,
		"no values observed -> 0 bytes");
}

static void	test_tokens_equal(void)
{
	int32_t	a[] = {1, 2, 3};
	int32_t	b[] = {1, 2, 3};
	int32_t	c[] = {1, 2, 4};
	int32_t	d[] = {1, 2};

	TEST_ASSERT(membrane_runtime_tokens_equal(a, 3, b, 3),
		"identical sequences equal");
	TEST_ASSERT(!membrane_runtime_tokens_equal(a, 3, c, 3),
		"differing element rejected");
	TEST_ASSERT(!membrane_runtime_tokens_equal(a, 3, d, 2),
		"differing length rejected");
	TEST_ASSERT(membrane_runtime_tokens_equal(NULL, 0, NULL, 0),
		"two empty sequences are equal");
	TEST_ASSERT(!membrane_runtime_tokens_equal(a, 3, NULL, 0),
		"non-empty vs empty rejected");
}

static void	test_step_counters(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8, 8);
	fill_block(block, 128, 0);
	membrane_runtime_begin_step(c);
	TEST_ASSERT(membrane_runtime_step_block_count(c) == 0,
		"fresh step starts at 0 blocks");
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 0,
		block, 128);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 1, 0,
		block, 128);
	TEST_ASSERT(membrane_runtime_step_block_count(c) == 2,
		"two observations in this step");
	membrane_runtime_end_step(c);
	membrane_runtime_begin_step(c);
	TEST_ASSERT(membrane_runtime_step_block_count(c) == 0,
		"counter resets at the next step boundary");
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 0,
		block, 128);
	TEST_ASSERT(membrane_runtime_step_block_count(c) == 1,
		"only this step's observation is counted, not the prior step's");
	membrane_runtime_collector_destroy(c);
}

static void	test_timing_accumulators(void)
{
	membrane_runtime_collector_t	*c;
	membrane_runtime_telemetry_t	t;

	c = membrane_runtime_collector_create(MEMBRANE_RUNTIME_MODE_BASELINE, 4);
	membrane_runtime_add_inference_seconds(c, 0.25);
	membrane_runtime_add_inference_seconds(c, 0.10);
	membrane_runtime_add_membrane_seconds(c, 0.01);
	membrane_runtime_add_membrane_seconds(c, 0.02);
	membrane_runtime_set_generated_tokens(c, 32);
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(fabs(t.inference_seconds - 0.35) < 1e-12,
		"inference seconds accumulate exactly");
	TEST_ASSERT(fabs(t.membrane_seconds - 0.03) < 1e-12,
		"membrane seconds accumulate exactly, tracked independently");
	TEST_ASSERT(t.generated_tokens == 32, "generated token count recorded");
	membrane_runtime_collector_destroy(c);
}

static void	test_finalize_null_collector_is_safe_zeroed(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0xff, sizeof(t));
	membrane_runtime_finalize(NULL, &t);
	TEST_ASSERT(t.total_blocks == 0 && t.generated_tokens == 0,
		"finalize(NULL) zeroes the output rather than crashing/leaking "
		"uninitialized memory");
}

static int	has(const char *s, const char *key)
{
	return (strstr(s, key) != NULL);
}

static void	test_json_shape_and_no_leakage(void)
{
	membrane_runtime_collector_t	*c;
	uint16_t						block[128];
	membrane_runtime_telemetry_t	t;
	int32_t							toks[] = {5, 6, 7};
	char							buf[8192];
	FILE							*f;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE, 8);
	fill_block(block, 128, 0);
	membrane_runtime_observe_tensor(c, membrane_simd_best_backend(), 0, 0,
		block, 128);
	membrane_runtime_set_generated_tokens(c, 3);
	membrane_runtime_finalize(c, &t);
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_runtime_print_json(&t, "model.gguf", "short.txt", toks, 3, f);
	fclose(f);
	TEST_ASSERT(has(buf, "\"mode\":\"shadow-adaptive\""), "mode field");
	TEST_ASSERT(has(buf, "\"model_label\":\"model.gguf\""), "model label");
	TEST_ASSERT(has(buf, "\"prompt_fixture\":\"short.txt\""), "prompt label");
	TEST_ASSERT(has(buf, "\"token_ids\":[5,6,7]"), "token id array");
	TEST_ASSERT(has(buf, "\"kv_observation\""), "kv_observation section");
	TEST_ASSERT(has(buf, "\"storage\""), "storage section");
	TEST_ASSERT(has(buf, "\"overhead\""), "overhead section");
	TEST_ASSERT(!has(buf, "/home/") && !has(buf, "/tmp/"),
		"no absolute path leakage");
	membrane_runtime_collector_destroy(c);
}

static void	test_human_output_states_shadow_disclaimer(void)
{
	membrane_runtime_telemetry_t	t;
	char							buf[8192];
	FILE							*f;

	memset(&t, 0, sizeof(t));
	t.mode = MEMBRANE_RUNTIME_MODE_SHADOW_Q8;
	f = fmemopen(buf, sizeof(buf), "w");
	TEST_ASSERT(f != NULL, "fmemopen");
	membrane_runtime_print_human(&t, f);
	fclose(f);
	TEST_ASSERT(has(buf, "SHADOW") && has(buf, "authoritative"),
		"human output states native llama KV remains authoritative");
	TEST_ASSERT(has(buf, "NOT process memory"),
		"human output disclaims process-memory reduction");
}

int	main(void)
{
	test_mode_name_round_trip();
	test_baseline_mode_never_processes();
	test_shadow_q8_forces_q8();
	test_shadow_adaptive_assigns_one_precision_per_block();
	test_tail_values_excluded_not_processed();
	test_no_padding_zero_full_blocks();
	test_layer_out_of_range_rejected();
	test_layers_seen_counts_distinct();
	test_weighted_mean_matches_pooled_formula();
	test_max_rel_l2_is_max_of_both_precisions();
	test_storage_accounting_ratios();
	test_zero_observations_zero_ratios();
	test_tokens_equal();
	test_step_counters();
	test_timing_accumulators();
	test_finalize_null_collector_is_safe_zeroed();
	test_json_shape_and_no_leakage();
	test_human_output_states_shadow_disclaimer();
	return (0);
}
