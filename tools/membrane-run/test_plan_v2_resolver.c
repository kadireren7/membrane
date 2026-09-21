#include <stdint.h>
#include <string.h>

#include "plan_v2_resolver.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

/* Same synthetic-shape convention as test_membrane_plan.c's own
 * kv_native()/kv_q8()/kv_q5() -- independent per-file fixtures, no
 * sharing (this project's own established one-fixture-per-test-file
 * convention). */
static uint64_t	kv_native(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 256);
}

/* Builds one variant's fully-formed hparams_known request: identical
 * architecture facts across variants (real property of same-family
 * quants, see plan_v2_resolver.h's own top comment), only
 * bytes_per_layer/output_role_bytes/total_weight_bytes scale with the
 * variant's own size. precision_request is pinned to NATIVE so the KV-
 * cache-precision dimension (an orthogonal axis) never interferes with
 * these tests' own weight-quant-driven feasibility. */
static void	fill_variant(membrane_plan_v2_variant_t *v, const char *quant,
				uint64_t bytes_per_layer, uint64_t model_max_context,
				membrane_plan_source_t source, int estimate_only)
{
	membrane_ctxrec_request_t	*req = &v->ctxrec_req;
	uint64_t					ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t						n;
	size_t						i;

	memset(v, 0, sizeof(*v));
	v->identity.model_name = "smoltest-family";
	v->identity.installed = (source == MEMBRANE_PLAN_SOURCE_MODEL_METADATA);
	v->identity.model_path = v->identity.installed
			? "/models/smoltest.gguf" : "";
	v->identity.arch_name = "llama";
	v->identity.arch_known = 1;
	v->identity.display_name = "SmolTest-Family";
	v->identity.parameter_count = "1B";
	v->variant.has_variant = 1;
	v->variant.quant = quant;
	v->variant.estimate_only = estimate_only;
	v->variant.source = source;
	v->hparams_known = 1;

	memset(req, 0, sizeof(*req));
	req->n_layer_all = 10;
	req->bytes_per_layer = bytes_per_layer;
	req->output_role_bytes = bytes_per_layer / 2;
	req->arch_name = "llama";
	req->n_embd = 512;
	req->n_head = 8;
	req->n_head_kv = 8;
	req->precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	req->model_max_context = model_max_context;
	req->model_max_context_known = 1;
	req->total_weight_bytes = bytes_per_layer * 10 + req->output_role_bytes;
	/* device fields are overwritten by membrane_plan_v2_resolve() itself
	 * from the request's own single snapshot -- CPU-only here (0),
	 * exercised by test_snapshot_identical_across_variants below. */

	n = membrane_ctxrec_generate_candidates(req->model_max_context, 0, ctxs,
			MEMBRANE_CTXREC_MAX_CANDIDATES);
	for (i = 0; i < n; ++i)
	{
		req->candidates[i].ctx = ctxs[i];
		req->candidates[i].kv_bytes_native = kv_native(ctxs[i],
				req->n_layer_all);
	}
	req->candidate_count = n;

	memset(&v->ctxrec_meta, 0, sizeof(v->ctxrec_meta));
	v->ctxrec_meta.context_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	v->ctxrec_meta.precision_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	v->ctxrec_meta.gpu_layers_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	v->ctxrec_meta.kv_placement_source = MEMBRANE_PLAN_SOURCE_FALLBACK_DEFAULT;
}

static void	fill_base_hw(membrane_plan_v2_request_t *req,
				uint64_t host_available_bytes)
{
	memset(req, 0, sizeof(*req));
	req->host_total_bytes = 2 * GIB;
	req->host_available_bytes = host_available_bytes;
	req->host_available_known = 1;
	req->device_free_bytes = 0;
	req->device_total_bytes = 0;
}

/* Q8_0-analog "largest" ~1050 MiB total, Q5_K_M-analog ~630 MiB, Q4_K_M-
 * analog ~420 MiB -- real ordering, no scores, matches this file's own
 * comment: caller supplies quality-descending order. */
static void	fill_three_tier_family(membrane_plan_v2_request_t *req)
{
	fill_variant(&req->variants[0], "Q8_0", 100 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_MODEL_METADATA, 0);
	fill_variant(&req->variants[1], "Q5_K_M", 60 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_CATALOG_METADATA, 1);
	fill_variant(&req->variants[2], "Q4_K_M", 40 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_CATALOG_METADATA, 1);
	req->variant_count = 3;
}

/* ------------------------------------------------------------------ */
/* A: multiple variants -- Q8 infeasible, Q5/Q4 feasible, deterministic */
/* ------------------------------------------------------------------ */

static void	test_multi_variant_selection(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 900 * MIB);
	fill_three_tier_family(&req);
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidate_count == 3, "3 candidates evaluated");
	TEST_ASSERT(out.candidates[0].feasibility.feasible == 0,
		"Q8_0 (largest) is infeasible on this constrained host");
	TEST_ASSERT(out.candidates[1].feasibility.feasible == 1,
		"Q5_K_M is feasible");
	TEST_ASSERT(out.candidates[2].feasibility.feasible == 1,
		"Q4_K_M is feasible");
	TEST_ASSERT(out.has_selected == 1, "a candidate was selected");
	TEST_ASSERT(out.selected_index == 1,
		"the higher-quality FEASIBLE variant (Q5_K_M) wins over Q4_K_M");
	TEST_ASSERT(out.feasible == 1, "overall plan is feasible");
}

/* ------------------------------------------------------------------ */
/* B: explicit quant -- forced infeasible variant is never downgraded */
/* ------------------------------------------------------------------ */

static void	test_explicit_quant_never_downgrades(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 900 * MIB);
	/* Only Q8_0 in the candidate array at all -- Part 4's own contract:
	 * the CALLER enforces "only this quant may be considered" by never
	 * including any other variant, never by this module silently
	 * trying siblings on Q8's behalf. */
	fill_variant(&req.variants[0], "Q8_0", 100 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_EXPLICIT_USER, 0);
	req.variant_count = 1;
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidate_count == 1, "exactly one candidate evaluated");
	TEST_ASSERT(out.candidates[0].feasibility.feasible == 0,
		"forced Q8_0 remains infeasible");
	TEST_ASSERT(out.has_selected == 0, "no candidate selected");
	TEST_ASSERT(out.feasible == 0, "overall infeasible -- never downgraded");
}

/* ------------------------------------------------------------------ */
/* C: explicit context -- preserved; variant may still adapt          */
/* ------------------------------------------------------------------ */

static void	test_explicit_context_preserved_variant_adapts(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;
	size_t							i;

	fill_base_hw(&req, 900 * MIB);
	fill_three_tier_family(&req);
	/* Force ctx=4096 as a single hard candidate on every variant --
	 * same single-candidate-request pattern plan_cmd.cpp's own explicit
	 * --ctx path already uses for one model. */
	for (i = 0; i < req.variant_count; ++i)
	{
		membrane_ctxrec_request_t	*r = &req.variants[i].ctxrec_req;

		r->model_max_context = (r->model_max_context_known
				&& 4096 <= r->model_max_context) ? r->model_max_context
				: 4096;
		r->minimum_required_context = 4096;
		r->candidates[0].ctx = 4096;
		r->candidates[0].kv_bytes_native = kv_native(4096, r->n_layer_all);
		r->candidate_count = 1;
		req.variants[i].ctxrec_meta.requested_context = 4096;
		req.variants[i].ctxrec_meta.context_source
				= MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.has_selected == 1, "a candidate was selected");
	TEST_ASSERT(out.candidates[out.selected_index].decisions.context == 4096,
		"explicit context preserved on the selected candidate");
	TEST_ASSERT(out.candidates[out.selected_index].decisions.context_source
			== MEMBRANE_PLAN_SOURCE_EXPLICIT_USER,
		"context_source stays explicit_user");
}

/* ------------------------------------------------------------------ */
/* D: explicit context + explicit quant -- both preserved, honest     */
/* infeasibility                                                      */
/* ------------------------------------------------------------------ */

static void	test_explicit_context_and_quant_honest_infeasible(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;
	membrane_ctxrec_request_t		*r;

	fill_base_hw(&req, 200 * MIB);	/* too small for even Q4_K_M */
	fill_variant(&req.variants[0], "Q4_K_M", 40 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_EXPLICIT_USER, 0);
	req.variant_count = 1;
	r = &req.variants[0].ctxrec_req;
	r->minimum_required_context = 4096;
	r->candidates[0].ctx = 4096;
	r->candidates[0].kv_bytes_native = kv_native(4096, r->n_layer_all);
	r->candidate_count = 1;
	req.variants[0].ctxrec_meta.requested_context = 4096;
	req.variants[0].ctxrec_meta.context_source
			= MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidate_count == 1, "exactly the one forced candidate");
	TEST_ASSERT(out.candidates[0].feasibility.feasible == 0,
		"both explicit constraints preserved -- infeasible, not silently"
		" relaxed");
	TEST_ASSERT(out.feasible == 0, "overall infeasible, honestly reported");
}

/* ------------------------------------------------------------------ */
/* E: variant/context tradeoff -- deterministic documented policy     */
/* ------------------------------------------------------------------ */

static void	test_tier_beats_quality(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 900 * MIB);
	/* variants[0] (highest quality) has NO real hparams -- coarse-only;
	 * variants[1] (lower quality) has real hparams and a real feasible
	 * decision. Part 8's own policy: an evaluated decision always beats
	 * a coarse estimate, even from a higher-quality variant. */
	memset(&req.variants[0], 0, sizeof(req.variants[0]));
	req.variants[0].identity.model_name = "smoltest-family";
	req.variants[0].identity.display_name = "SmolTest-Family";
	req.variants[0].variant.has_variant = 1;
	req.variants[0].variant.quant = "Q8_0";
	req.variants[0].variant.estimate_only = 1;
	req.variants[0].variant.source = MEMBRANE_PLAN_SOURCE_CATALOG_METADATA;
	req.variants[0].hparams_known = 0;
	req.variants[0].coarse_fits = 1;
	fill_variant(&req.variants[1], "Q5_K_M", 60 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_CATALOG_METADATA, 1);
	req.variant_count = 2;
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidates[0].feasibility.feasible == 1,
		"coarse-only Q8_0 counts as feasible (trivially, no decision)");
	TEST_ASSERT(out.candidates[0].decisions.has_decisions == 0,
		"Q8_0 has no real decision");
	TEST_ASSERT(out.candidates[1].decisions.has_decisions == 1,
		"Q5_K_M has a real evaluated decision");
	TEST_ASSERT(out.selected_index == 1,
		"a real evaluated decision beats a higher-quality coarse estimate");
}

/* ------------------------------------------------------------------ */
/* F: alternatives -- bounded, stable order                           */
/* ------------------------------------------------------------------ */

static void	test_alternatives_bounded_and_ordered(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 2000 * MIB);	/* everything fits */
	fill_three_tier_family(&req);
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.selected_index == 0, "highest-quality feasible wins");
	TEST_ASSERT(out.alternative_count == 2, "the other 2 feasible variants");
	TEST_ASSERT(out.alternative_indices[0] == 1,
		"alternatives keep the same quality order (Q5_K_M first)");
	TEST_ASSERT(out.alternative_indices[1] == 2,
		"then Q4_K_M");
}

/* ------------------------------------------------------------------ */
/* G: reason codes -- per-variant failure reasons preserved            */
/* ------------------------------------------------------------------ */

static void	test_per_variant_reasons_preserved(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;
	size_t							i;
	int								found;

	fill_base_hw(&req, 900 * MIB);
	fill_three_tier_family(&req);
	membrane_plan_v2_resolve(&req, &out);
	found = 0;
	for (i = 0; i < out.candidates[0].reason_count; ++i)
		if (out.candidates[0].reasons[i].detail[0] != '\0')
			found = 1;
	TEST_ASSERT(found, "the infeasible Q8_0 candidate keeps its own reason");
	TEST_ASSERT(out.candidates[0].feasibility.limiting_resource[0] != '\0',
		"Q8_0's own limiting_resource is set");
}

/* ------------------------------------------------------------------ */
/* H: provenance -- source marked correctly per variant                */
/* ------------------------------------------------------------------ */

static void	test_provenance_marked_per_variant(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 2000 * MIB);
	fill_three_tier_family(&req);
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidates[0].identity.variant_source
			== MEMBRANE_PLAN_SOURCE_MODEL_METADATA,
		"the real installed variant is sourced model_metadata");
	TEST_ASSERT(out.candidates[0].identity.variant_estimate_only == 0,
		"the real installed variant is not an estimate");
	TEST_ASSERT(out.candidates[1].identity.variant_source
			== MEMBRANE_PLAN_SOURCE_CATALOG_METADATA,
		"a scaled sibling is sourced catalog_metadata");
	TEST_ASSERT(out.candidates[1].identity.variant_estimate_only == 1,
		"a scaled sibling is marked estimate_only");
}

/* ------------------------------------------------------------------ */
/* I: snapshot consistency -- identical hardware snapshot for every    */
/* variant regardless of what each variant's own ctxrec_req carried in */
/* ------------------------------------------------------------------ */

static void	test_snapshot_identical_across_variants(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;

	fill_base_hw(&req, 2000 * MIB);
	fill_three_tier_family(&req);
	/* Deliberately poison each variant's OWN ctxrec_req hardware fields
	 * with different, wrong values -- membrane_plan_v2_resolve() must
	 * override every one of them from the single top-level snapshot. */
	req.variants[0].ctxrec_req.host_total_bytes = 1 * MIB;
	req.variants[0].ctxrec_req.host_available_bytes = 1 * MIB;
	req.variants[1].ctxrec_req.host_available_bytes = 999999 * MIB;
	req.variants[2].ctxrec_req.device_total_bytes = 12345 * MIB;
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidates[0].hardware.host_available_bytes
			== 2000 * MIB, "variant 0's poisoned snapshot was overridden");
	TEST_ASSERT(out.candidates[1].hardware.host_available_bytes
			== 2000 * MIB, "variant 1's poisoned snapshot was overridden");
	TEST_ASSERT(out.candidates[2].hardware.device_total_bytes == 0,
		"variant 2's poisoned device snapshot was overridden");
}

/* ------------------------------------------------------------------ */
/* J: search bound -- worst-case fixture stays within the documented   */
/* MEMBRANE_PLAN_V2_MAX_VARIANTS bound, extra entries are ignored, not  */
/* silently overflowed                                                 */
/* ------------------------------------------------------------------ */

static void	test_search_bound_enforced(void)
{
	membrane_plan_v2_request_t		req;
	membrane_plan_v2_result_t		out;
	size_t							i;

	fill_base_hw(&req, 2000 * MIB);
	for (i = 0; i < MEMBRANE_PLAN_V2_MAX_VARIANTS + 3 && i
			< MEMBRANE_PLAN_V2_MAX_VARIANTS; ++i)
		fill_variant(&req.variants[i], "Q4_K_M", 40 * MIB, 8192,
			MEMBRANE_PLAN_SOURCE_CATALOG_METADATA, 1);
	req.variant_count = MEMBRANE_PLAN_V2_MAX_VARIANTS + 3;	/* over-request */
	membrane_plan_v2_resolve(&req, &out);
	TEST_ASSERT(out.candidate_count == MEMBRANE_PLAN_V2_MAX_VARIANTS,
		"candidate_count is clamped to the documented bound, never"
		" overflowed");
	TEST_ASSERT(out.alternative_count <= MEMBRANE_PLAN_V2_MAX_ALTERNATIVES,
		"alternatives stay within their own documented bound");
}

/* ------------------------------------------------------------------ */
/* L: regression -- a single hparams_known variant behaves exactly     */
/* like G1's own single-model membrane_plan_assemble() path            */
/* ------------------------------------------------------------------ */

static void	test_single_variant_matches_g1_behavior(void)
{
	membrane_plan_v2_request_t		v2req;
	membrane_plan_v2_result_t		v2out;
	membrane_ctxrec_request_t		g1req;
	membrane_ctxrec_result_t		g1rec;
	membrane_plan_identity_input_t	g1id;
	membrane_plan_request_meta_t	g1meta;
	membrane_plan_t					g1plan;

	fill_base_hw(&v2req, 2000 * MIB);
	fill_variant(&v2req.variants[0], "Q4_K_M", 40 * MIB, 8192,
		MEMBRANE_PLAN_SOURCE_MODEL_METADATA, 0);
	v2req.variant_count = 1;
	membrane_plan_v2_resolve(&v2req, &v2out);

	g1req = v2req.variants[0].ctxrec_req;
	g1req.host_total_bytes = v2req.host_total_bytes;
	g1req.host_available_bytes = v2req.host_available_bytes;
	g1req.host_available_known = v2req.host_available_known;
	g1req.device_free_bytes = v2req.device_free_bytes;
	g1req.device_total_bytes = v2req.device_total_bytes;
	membrane_ctxrec_resolve(&g1req, &g1rec);
	g1id = v2req.variants[0].identity;
	g1meta = v2req.variants[0].ctxrec_meta;
	membrane_plan_assemble(&g1rec, &g1req, &g1meta, &g1id,
		&v2req.variants[0].variant, NULL, &g1plan);

	TEST_ASSERT(v2out.candidates[0].feasibility.feasible
			== g1plan.feasibility.feasible, "feasibility matches G1's own");
	TEST_ASSERT(v2out.candidates[0].decisions.context
			== g1plan.decisions.context, "decisions.context matches G1's own");
	TEST_ASSERT(v2out.candidates[0].decisions.gpu_layers
			== g1plan.decisions.gpu_layers,
		"decisions.gpu_layers matches G1's own");
}

int	main(void)
{
	test_multi_variant_selection();
	test_explicit_quant_never_downgrades();
	test_explicit_context_preserved_variant_adapts();
	test_explicit_context_and_quant_honest_infeasible();
	test_tier_beats_quality();
	test_alternatives_bounded_and_ordered();
	test_per_variant_reasons_preserved();
	test_provenance_marked_per_variant();
	test_snapshot_identical_across_variants();
	test_search_bound_enforced();
	test_single_variant_matches_g1_behavior();
	printf("all plan_v2_resolver tests passed\n");
	return (0);
}
