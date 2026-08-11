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

/* Covers the actual path-sanitization boundary directly (an absolute
 * path in, only the basename out) -- not just "the printer doesn't
 * corrupt an already-safe basename", which is trivially true and
 * exercises nothing about path stripping at all. */
static void	test_safe_basename_strips_absolute_paths(void)
{
	TEST_ASSERT(strcmp(membrane_runtime_safe_basename(
			"/home/user/models/model.gguf"), "model.gguf") == 0,
		"absolute model path reduced to basename only");
	TEST_ASSERT(strcmp(membrane_runtime_safe_basename(
			"/tmp/fixtures/short.txt"), "short.txt") == 0,
		"absolute prompt path reduced to basename only");
	TEST_ASSERT(strcmp(membrane_runtime_safe_basename("model.gguf"),
			"model.gguf") == 0,
		"a bare filename with no '/' is returned unchanged");
	TEST_ASSERT(strcmp(membrane_runtime_safe_basename("a/b/c/d.gguf"),
			"d.gguf") == 0,
		"only the LAST path component is kept, not any earlier one");
	TEST_ASSERT(strcmp(membrane_runtime_safe_basename("/"), "") == 0,
		"a bare trailing slash yields an empty basename, not a crash");
	TEST_ASSERT(membrane_runtime_safe_basename(NULL) == NULL,
		"NULL in, NULL out -- never dereferenced");
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

/* ------------------------------------------------------------------ */
/* Product Phase 6: injection mode/scope                               */
/* ------------------------------------------------------------------ */

static void	test_inject_mode_names_and_helpers(void)
{
	membrane_runtime_mode_t	m;

	TEST_ASSERT(strcmp(membrane_runtime_mode_name(
			MEMBRANE_RUNTIME_MODE_INJECT_Q8), "inject-q8") == 0,
		"inject-q8 name");
	TEST_ASSERT(strcmp(membrane_runtime_mode_name(
			MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE), "inject-adaptive") == 0,
		"inject-adaptive name");
	TEST_ASSERT(membrane_runtime_mode_from_name("inject-q8", &m)
		&& m == MEMBRANE_RUNTIME_MODE_INJECT_Q8, "parse inject-q8");
	TEST_ASSERT(membrane_runtime_mode_from_name("inject-adaptive", &m)
		&& m == MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE,
		"parse inject-adaptive");
	TEST_ASSERT(membrane_runtime_mode_is_inject(MEMBRANE_RUNTIME_MODE_INJECT_Q8),
		"is_inject true for inject-q8");
	TEST_ASSERT(membrane_runtime_mode_is_inject(
			MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE),
		"is_inject true for inject-adaptive");
	TEST_ASSERT(!membrane_runtime_mode_is_inject(
			MEMBRANE_RUNTIME_MODE_SHADOW_Q8),
		"is_inject false for shadow-q8");
	TEST_ASSERT(!membrane_runtime_mode_is_inject(
			MEMBRANE_RUNTIME_MODE_BASELINE),
		"is_inject false for baseline");
	TEST_ASSERT(membrane_runtime_mode_is_shadow(
			MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE),
		"is_shadow true for shadow-adaptive");
	TEST_ASSERT(!membrane_runtime_mode_is_shadow(
			MEMBRANE_RUNTIME_MODE_INJECT_Q8),
		"is_shadow false for inject-q8");
}

static void	test_scope_default_matches_everything(void)
{
	membrane_runtime_scope_t	s;

	membrane_runtime_scope_init(&s);
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 0, 0),
		"layer 0 K matches by default");
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 29, 1),
		"any layer/tensor matches by default");
	TEST_ASSERT(membrane_runtime_scope_matches_token(&s, 0),
		"token 0 matches by default");
	TEST_ASSERT(membrane_runtime_scope_matches_token(&s, 1000000),
		"any token matches by default");
}

static void	test_scope_layer_filter_is_allowlist(void)
{
	membrane_runtime_scope_t	s;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_add_layer(&s, 5);
	membrane_runtime_scope_add_layer(&s, 10);
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 5, 0),
		"layer 5 explicitly added");
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 10, 1),
		"layer 10 explicitly added");
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 6, 0),
		"layer 6 never added -- excluded once the filter is active");
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 0, 0),
		"layer 0 never added -- excluded");
}

static void	test_scope_tensor_filter(void)
{
	membrane_runtime_scope_t	s;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_set_tensor(&s, MEMBRANE_RUNTIME_TENSOR_K);
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 0, 0),
		"K matches a K-only filter");
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 0, 1),
		"V rejected by a K-only filter");
	membrane_runtime_scope_set_tensor(&s, MEMBRANE_RUNTIME_TENSOR_V);
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 0, 0),
		"K rejected by a V-only filter");
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 0, 1),
		"V matches a V-only filter");
}

static void	test_scope_token_range_is_inclusive(void)
{
	membrane_runtime_scope_t	s;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_set_token_range(&s, 10, 20);
	TEST_ASSERT(!membrane_runtime_scope_matches_token(&s, 9),
		"below range excluded");
	TEST_ASSERT(membrane_runtime_scope_matches_token(&s, 10),
		"range start inclusive");
	TEST_ASSERT(membrane_runtime_scope_matches_token(&s, 20),
		"range end inclusive");
	TEST_ASSERT(!membrane_runtime_scope_matches_token(&s, 21),
		"above range excluded");
}

static void	test_scope_layer_and_tensor_combine(void)
{
	membrane_runtime_scope_t	s;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_add_layer(&s, 3);
	membrane_runtime_scope_set_tensor(&s, MEMBRANE_RUNTIME_TENSOR_K);
	TEST_ASSERT(membrane_runtime_scope_matches_layer_tensor(&s, 3, 0),
		"layer 3 K matches both filters");
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 3, 1),
		"layer 3 V rejected by the tensor filter");
	TEST_ASSERT(!membrane_runtime_scope_matches_layer_tensor(&s, 4, 0),
		"layer 4 K rejected by the layer filter");
}

/* ------------------------------------------------------------------ */
/* Product Phase 6: injection reconstruction (block/tail coverage)     */
/* ------------------------------------------------------------------ */

static void	test_inject_reconstruct_respects_scope(void)
{
	membrane_runtime_scope_t			s;
	uint16_t							buf[256];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_add_layer(&s, 0);
	fill_block(buf, 256, 0);
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			membrane_simd_best_backend(), MEMBRANE_BENCH_POLICY_Q8_ONLY, &s,
			5, 0, 0, 1, 256, buf, &r) == MEMBRANE_OK,
		"call succeeds even when out of scope");
	TEST_ASSERT(r.tokens_in_scope == 0 && r.eligible_blocks == 0
		&& r.injected_blocks == 0,
		"nothing touched when the layer is out of scope");
	TEST_ASSERT(r.ok, "an out-of-scope call is trivially ok");
}

static void	test_inject_reconstruct_per_token_blocks_and_tail(void)
{
	membrane_runtime_scope_t			s;
	/* 2 tokens, 300 elements each = 2 full blocks (256) + 44-element
	 * tail, per token. */
	uint16_t							buf[600];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	fill_block(buf, 600, 0);
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			membrane_simd_best_backend(), MEMBRANE_BENCH_POLICY_Q8_ONLY, &s,
			0, 0, 0, 2, 300, buf, &r) == MEMBRANE_OK, "call succeeds");
	TEST_ASSERT(r.tokens_in_scope == 2, "both tokens in scope (no filter)");
	TEST_ASSERT(r.eligible_blocks == 4, "2 full blocks/token x 2 tokens");
	TEST_ASSERT(r.injected_blocks == 4, "all of them succeed");
	TEST_ASSERT(r.native_tail_values == 88, "44-element tail x 2 tokens");
	TEST_ASSERT(r.ok, "no failure");
}

static void	test_inject_reconstruct_never_straddles_token_boundary(void)
{
	/* elements_per_token = 192 (not a multiple of 128): 1 full block +
	 * 64-element tail per token, even though 192*2=384 would divide
	 * evenly into exactly 3 128-element blocks if treated as one flat
	 * stream -- proving blocks are computed PER TOKEN, never across a
	 * token boundary. */
	membrane_runtime_scope_t			s;
	uint16_t							buf[384];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	fill_block(buf, 384, 0);
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			membrane_simd_best_backend(), MEMBRANE_BENCH_POLICY_Q8_ONLY, &s,
			0, 0, 0, 2, 192, buf, &r) == MEMBRANE_OK, "call succeeds");
	TEST_ASSERT(r.eligible_blocks == 2, "1 block per token x 2, not 3");
	TEST_ASSERT(r.native_tail_values == 128, "64-element tail x 2 tokens");
}

static void	test_inject_reconstruct_token_range_uses_absolute_position(void)
{
	membrane_runtime_scope_t			s;
	uint16_t							buf[3 * 128];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_set_token_range(&s, 5, 5);
	fill_block(buf, 3 * 128, 0);
	/* This call's tokens are absolute positions 4, 5, 6. */
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			membrane_simd_best_backend(), MEMBRANE_BENCH_POLICY_Q8_ONLY, &s,
			0, 0, 4, 3, 128, buf, &r) == MEMBRANE_OK, "call succeeds");
	TEST_ASSERT(r.tokens_in_scope == 1,
		"only absolute token 5 (the middle one here) is in range");
	TEST_ASSERT(r.eligible_blocks == 1,
		"exactly 1 block, from the one in-scope token");
}

static void	test_inject_reconstruct_out_of_scope_values_untouched(void)
{
	membrane_runtime_scope_t			s;
	uint16_t							buf[256];
	uint16_t							original[256];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	membrane_runtime_scope_set_token_range(&s, 0, 0);
	fill_block(buf, 256, 0);
	memcpy(original, buf, sizeof(buf));
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			membrane_simd_best_backend(), MEMBRANE_BENCH_POLICY_Q8_ONLY, &s,
			0, 0, 0, 2, 128, buf, &r) == MEMBRANE_OK, "call succeeds");
	TEST_ASSERT(r.tokens_in_scope == 1, "only token 0");
	TEST_ASSERT(memcmp(buf + 128, original + 128, 128 * sizeof(uint16_t))
			== 0,
		"token 1 (out of scope) completely untouched, byte for byte");
}

static void
	test_inject_reconstruct_stops_at_first_failure_and_preserves_rest(void)
{
	membrane_runtime_scope_t			s;
	uint16_t							buf[256];
	uint16_t							original[256];
	membrane_runtime_inject_result_t	r;

	membrane_runtime_scope_init(&s);
	fill_block(buf, 256, 0);
	memcpy(original, buf, sizeof(buf));
	TEST_ASSERT(membrane_runtime_inject_reconstruct(
			(membrane_simd_backend_t)MEMBRANE_SIMD_COUNT,
			MEMBRANE_BENCH_POLICY_Q8_ONLY, &s, 0, 0, 0, 2, 128, buf, &r)
			== MEMBRANE_OK,
		"the function itself still returns MEMBRANE_OK -- failure is "
		"reported via out->ok, not the return status");
	TEST_ASSERT(!r.ok, "an unsupported backend makes reconstruction fail");
	TEST_ASSERT(r.failed_blocks == 1,
		"exactly the first attempted block is recorded as failed");
	TEST_ASSERT(r.injected_blocks == 0,
		"nothing succeeded before the failure");
	TEST_ASSERT(memcmp(buf, original, sizeof(buf)) == 0,
		"buffer completely untouched on failure -- safe to discard, "
		"never a partial corruption");
}

/* ------------------------------------------------------------------ */
/* Product Phase 6: divergence detection                               */
/* ------------------------------------------------------------------ */

static void	test_divergence_identical_sequences(void)
{
	int32_t							a[] = {1, 2, 3, 4};
	membrane_runtime_divergence_t	d;

	membrane_runtime_detect_divergence(a, 4, a, 4, &d);
	TEST_ASSERT(d.identical, "identical arrays are identical");
	TEST_ASSERT(d.first_divergence_step == -1, "no divergence step");
	TEST_ASSERT(d.divergent_positions == 0, "0 divergent positions");
}

static void	test_divergence_first_mismatch(void)
{
	int32_t							a[] = {1, 2, 3, 4, 5};
	int32_t							b[] = {1, 2, 9, 4, 7};
	membrane_runtime_divergence_t	d;

	membrane_runtime_detect_divergence(a, 5, b, 5, &d);
	TEST_ASSERT(!d.identical, "differ at two positions");
	TEST_ASSERT(d.first_divergence_step == 2, "first mismatch at index 2");
	TEST_ASSERT(d.divergent_positions == 2, "positions 2 and 4 differ");
}

static void	test_divergence_length_mismatch(void)
{
	int32_t							a[] = {1, 2, 3};
	int32_t							b[] = {1, 2, 3, 4};
	membrane_runtime_divergence_t	d;

	membrane_runtime_detect_divergence(a, 3, b, 4, &d);
	TEST_ASSERT(!d.identical, "different lengths are never identical");
	TEST_ASSERT(d.first_divergence_step == 3,
		"first divergence at the point the shorter sequence runs out");
	TEST_ASSERT(d.divergent_positions == 0,
		"every overlapping position still matched");
}

/* ------------------------------------------------------------------ */
/* Product Phase 6: logit comparison / NLL                             */
/* ------------------------------------------------------------------ */

static void	test_compare_logits_identical_vectors(void)
{
	float								a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	membrane_runtime_step_logit_diff_t	diff;

	TEST_ASSERT(membrane_runtime_compare_logits(a, a, 8, 3, &diff)
			== MEMBRANE_OK, "compare succeeds");
	TEST_ASSERT(diff.abs_diff_max == 0.0 && diff.abs_diff_mean == 0.0,
		"no difference between a vector and itself");
	TEST_ASSERT(diff.rel_l2 == 0.0, "zero rel-L2 for identical vectors");
	TEST_ASSERT(diff.top1_preserved, "top1 trivially preserved");
	TEST_ASSERT(diff.topk_overlap == 3, "top-3 fully overlaps with itself");
}

static void	test_compare_logits_top1_flip(void)
{
	float								a[4] = {1.0f, 5.0f, 2.0f, 0.0f};
	float								b[4] = {1.0f, 2.0f, 9.0f, 0.0f};
	membrane_runtime_step_logit_diff_t	diff;

	TEST_ASSERT(membrane_runtime_compare_logits(a, b, 4, 0, &diff)
			== MEMBRANE_OK, "compare succeeds");
	TEST_ASSERT(!diff.top1_preserved, "top1 token changed (index 1 -> 2)");
	TEST_ASSERT(diff.topk_overlap == -1,
		"k <= 0 means top-k overlap is not computed");
	TEST_ASSERT(diff.abs_diff_max > 0.0, "nonzero max diff");
}

static void	test_compare_logits_invalid_args_rejected(void)
{
	float								a[4] = {0, 0, 0, 0};
	membrane_runtime_step_logit_diff_t	diff;

	TEST_ASSERT(membrane_runtime_compare_logits(NULL, a, 4, 1, &diff)
			== MEMBRANE_ERR_INVALID_ARG, "NULL a rejected");
	TEST_ASSERT(membrane_runtime_compare_logits(a, a, 0, 1, &diff)
			== MEMBRANE_ERR_INVALID_ARG, "n_vocab <= 0 rejected");
}

static void	test_nll_higher_for_less_likely_token(void)
{
	float	logits[4] = {5.0f, 0.0f, 0.0f, 0.0f};
	double	nll_likely;
	double	nll_unlikely;

	nll_likely = membrane_runtime_nll(logits, 4, 0);
	nll_unlikely = membrane_runtime_nll(logits, 4, 1);
	TEST_ASSERT(nll_likely < nll_unlikely,
		"NLL is lower for the more probable (target) token");
	TEST_ASSERT(nll_likely >= 0.0, "NLL is non-negative");
}

static void	test_nll_uniform_logits_equals_log_n(void)
{
	float	logits[4] = {0, 0, 0, 0};
	double	nll;

	nll = membrane_runtime_nll(logits, 4, 2);
	TEST_ASSERT(fabs(nll - log(4.0)) < 1e-9,
		"uniform logits -> NLL == log(n_vocab) regardless of target");
}

/* ------------------------------------------------------------------ */
/* Product Phase 6: behavior accumulation                              */
/* ------------------------------------------------------------------ */

static void	test_behavior_accum_aggregates_steps(void)
{
	membrane_runtime_behavior_accum_t		*b;
	membrane_runtime_step_logit_diff_t		d1;
	membrane_runtime_step_logit_diff_t		d2;
	membrane_runtime_behavior_summary_t	sum;

	b = membrane_runtime_behavior_create();
	TEST_ASSERT(b != NULL, "create succeeds");
	memset(&d1, 0, sizeof(d1));
	d1.abs_diff_max = 0.1;
	d1.abs_diff_mean = 0.05;
	d1.rel_l2 = 0.02;
	d1.top1_preserved = 1;
	d1.topk_overlap = 5;
	memset(&d2, 0, sizeof(d2));
	d2.abs_diff_max = 0.3;
	d2.abs_diff_mean = 0.15;
	d2.rel_l2 = 0.04;
	d2.top1_preserved = 0;
	d2.topk_overlap = 3;
	membrane_runtime_behavior_add_step(b, &d1, 1.0, 1.1);
	membrane_runtime_behavior_add_step(b, &d2, 2.0, 2.5);
	membrane_runtime_behavior_finalize(b, &sum);
	TEST_ASSERT(sum.steps == 2, "two steps recorded");
	TEST_ASSERT(fabs(sum.logit_abs_diff_max - 0.3) < 1e-12,
		"max is the max of per-step maxes, not summed");
	TEST_ASSERT(fabs(sum.logit_abs_diff_mean - 0.10) < 1e-12,
		"mean of per-step means: (0.05+0.15)/2");
	TEST_ASSERT(fabs(sum.top1_preservation_rate - 0.5) < 1e-12,
		"1 of 2 steps preserved top1");
	TEST_ASSERT(fabs(sum.topk_overlap_mean - 4.0) < 1e-12, "(5+3)/2");
	TEST_ASSERT(fabs(sum.mean_nll_baseline - 1.5) < 1e-12, "(1.0+2.0)/2");
	TEST_ASSERT(fabs(sum.mean_nll_injected - 1.8) < 1e-12, "(1.1+2.5)/2");
	TEST_ASSERT(fabs(sum.delta_nll - 0.3) < 1e-9,
		"delta = mean injected NLL - mean baseline NLL");
	membrane_runtime_behavior_destroy(b);
}

static void	test_behavior_finalize_empty_is_safe(void)
{
	membrane_runtime_behavior_summary_t	sum;

	membrane_runtime_behavior_finalize(NULL, &sum);
	TEST_ASSERT(sum.steps == 0,
		"no accumulator -> a zeroed summary, not garbage");
}

/* ------------------------------------------------------------------ */
/* Product Phase 6: collector injection recording / failure state      */
/* ------------------------------------------------------------------ */

static void	test_record_injection_folds_into_collector(void)
{
	membrane_runtime_collector_t		*c;
	membrane_runtime_inject_result_t	r1;
	membrane_runtime_inject_result_t	r2;
	membrane_runtime_telemetry_t		t;

	c = membrane_runtime_collector_create(MEMBRANE_RUNTIME_MODE_INJECT_Q8, 8);
	memset(&r1, 0, sizeof(r1));
	r1.tokens_in_scope = 1;
	r1.eligible_blocks = 2;
	r1.injected_blocks = 2;
	r1.ok = 1;
	r1.accum.q8_blocks = 2;
	r1.accum.encoded_bytes = 100;
	membrane_runtime_record_injection(c, 0, 0, &r1);
	memset(&r2, 0, sizeof(r2));
	r2.tokens_in_scope = 1;
	r2.eligible_blocks = 1;
	r2.injected_blocks = 1;
	r2.ok = 1;
	r2.accum.q8_blocks = 1;
	r2.accum.encoded_bytes = 50;
	membrane_runtime_record_injection(c, 0, 1, &r2);
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.injection_eligible_blocks == 3, "2+1 eligible summed");
	TEST_ASSERT(t.injected_blocks == 3, "2+1 injected summed");
	TEST_ASSERT(t.accum.q8_blocks == 3, "accum folded in too");
	TEST_ASSERT(t.accum.encoded_bytes == 150, "encoded bytes summed");
	TEST_ASSERT(t.layers_targeted == 1,
		"same layer touched via K then V -- counted once");
	TEST_ASSERT(t.layers_injected == 1, "same layer, counted once");
	TEST_ASSERT(t.injection_requested == 1, "mode is inject-q8");
	TEST_ASSERT(t.injection_succeeded == 1, "no failures occurred");
	membrane_runtime_collector_destroy(c);
}

static void	test_record_injection_failure_marks_run_failed(void)
{
	membrane_runtime_collector_t		*c;
	membrane_runtime_inject_result_t	r;
	membrane_runtime_telemetry_t		t;

	c = membrane_runtime_collector_create(
			MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE, 8);
	TEST_ASSERT(!membrane_runtime_injection_has_failed(c),
		"a fresh collector has not failed");
	memset(&r, 0, sizeof(r));
	r.tokens_in_scope = 1;
	r.eligible_blocks = 1;
	r.injected_blocks = 0;
	r.failed_blocks = 1;
	r.ok = 0;
	membrane_runtime_record_injection(c, 3, 0, &r);
	TEST_ASSERT(membrane_runtime_injection_has_failed(c),
		"one failed block marks the whole collector failed");
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.injection_succeeded == 0,
		"telemetry reflects the failure -- never silently reports "
		"success");
	TEST_ASSERT(t.failed_blocks == 1, "failed block count surfaced");
	membrane_runtime_collector_destroy(c);
}

static void	test_record_injection_no_scope_match_is_harmless(void)
{
	membrane_runtime_collector_t		*c;
	membrane_runtime_inject_result_t	r;
	membrane_runtime_telemetry_t		t;

	c = membrane_runtime_collector_create(MEMBRANE_RUNTIME_MODE_INJECT_Q8, 8);
	memset(&r, 0, sizeof(r));
	r.ok = 1;
	membrane_runtime_record_injection(c, 0, 0, &r);
	membrane_runtime_finalize(c, &t);
	TEST_ASSERT(t.layers_targeted == 0,
		"an out-of-scope observation doesn't count as targeted");
	TEST_ASSERT(t.injection_succeeded == 1, "no failure occurred");
	membrane_runtime_collector_destroy(c);
}

static void	test_injection_coverage_ratio(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0, sizeof(t));
	t.injected_blocks = 3;
	t.injection_native_tail_values = 96;
	TEST_ASSERT(fabs(membrane_runtime_injection_coverage_ratio(&t)
			- 384.0 / 480.0) < 1e-9,
		"coverage = injected values / (injected + native tail)");
}

static void	test_injection_coverage_ratio_zero_when_nothing_injected(void)
{
	membrane_runtime_telemetry_t	t;

	memset(&t, 0, sizeof(t));
	TEST_ASSERT(membrane_runtime_injection_coverage_ratio(&t) == 0.0,
		"0/0 -> 0.0, not NaN");
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
	test_safe_basename_strips_absolute_paths();
	test_step_counters();
	test_timing_accumulators();
	test_finalize_null_collector_is_safe_zeroed();
	test_json_shape_and_no_leakage();
	test_human_output_states_shadow_disclaimer();
	test_inject_mode_names_and_helpers();
	test_scope_default_matches_everything();
	test_scope_layer_filter_is_allowlist();
	test_scope_tensor_filter();
	test_scope_token_range_is_inclusive();
	test_scope_layer_and_tensor_combine();
	test_inject_reconstruct_respects_scope();
	test_inject_reconstruct_per_token_blocks_and_tail();
	test_inject_reconstruct_never_straddles_token_boundary();
	test_inject_reconstruct_token_range_uses_absolute_position();
	test_inject_reconstruct_out_of_scope_values_untouched();
	test_inject_reconstruct_stops_at_first_failure_and_preserves_rest();
	test_divergence_identical_sequences();
	test_divergence_first_mismatch();
	test_divergence_length_mismatch();
	test_compare_logits_identical_vectors();
	test_compare_logits_top1_flip();
	test_compare_logits_invalid_args_rejected();
	test_nll_higher_for_less_likely_token();
	test_nll_uniform_logits_equals_log_n();
	test_behavior_accum_aggregates_steps();
	test_behavior_finalize_empty_is_safe();
	test_record_injection_folds_into_collector();
	test_record_injection_failure_marks_run_failed();
	test_record_injection_no_scope_match_is_harmless();
	test_injection_coverage_ratio();
	test_injection_coverage_ratio_zero_when_nothing_injected();
	return (0);
}
