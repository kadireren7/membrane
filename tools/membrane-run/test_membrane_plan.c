#include <stdint.h>
#include <string.h>

#include "membrane_plan.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

/* Same synthetic shape as test_context_recommender.c's own
 * fill_base_request() (Section 27 there: "no model download") -- kept
 * independent per this project's established one-fixture-per-test-file
 * convention, not shared, so this file stays a self-contained read of
 * membrane_plan.h's own contract. */
static uint64_t	kv_native(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 256);
}

static uint64_t	kv_q8(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 128);
}

static uint64_t	kv_q5(uint64_t ctx, int32_t n_layer)
{
	return (ctx * (uint64_t)n_layer * 96);
}

static void	fill_base_request(membrane_ctxrec_request_t *req)
{
	memset(req, 0, sizeof(*req));
	req->n_layer_all = 10;
	req->bytes_per_layer = 10 * MIB;
	req->output_role_bytes = 20 * MIB;
	req->arch_name = "llama";
	req->n_embd = 512;
	req->n_head = 8;
	req->n_head_kv = 8;
	req->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	req->model_max_context = 32768;
	req->model_max_context_known = 1;
	req->total_weight_bytes = 10 * 10 * MIB + 20 * MIB;
	req->host_total_bytes = 8 * GIB;
	req->host_available_bytes = 4 * GIB;
	req->host_available_known = 1;
	req->device_free_bytes = 4 * GIB;
	req->device_total_bytes = 4 * GIB;
}

static size_t	populate_candidates(membrane_ctxrec_request_t *req)
{
	uint64_t	ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n;
	size_t		i;

	n = membrane_ctxrec_generate_candidates(req->model_max_context,
			req->minimum_required_context, ctxs,
			MEMBRANE_CTXREC_MAX_CANDIDATES);
	for (i = 0; i < n; ++i)
	{
		req->candidates[i].ctx = ctxs[i];
		req->candidates[i].kv_bytes_native = kv_native(ctxs[i],
				req->n_layer_all);
		req->candidates[i].kv_bytes_q8 = kv_q8(ctxs[i], req->n_layer_all);
		req->candidates[i].kv_bytes_q5 = kv_q5(ctxs[i], req->n_layer_all);
	}
	req->candidate_count = n;
	return (n);
}

static void	fill_default_identity(membrane_plan_identity_input_t *id)
{
	memset(id, 0, sizeof(*id));
	id->model_name = "smollm2-135m-instruct";
	id->model_path = "/models/smollm2.gguf";
	id->installed = 1;
	id->arch_name = "llama";
	id->arch_known = 1;
	id->display_name = "SmolLM2-135M-Instruct";
	id->parameter_count = "135M";
}

static void	fill_default_meta(membrane_plan_request_meta_t *meta)
{
	memset(meta, 0, sizeof(*meta));
	meta->requested_context = 0;
	meta->context_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	meta->precision_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	meta->gpu_layers_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	meta->kv_placement_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
}

/* ------------------------------------------------------------------ */
/* A/C: required fields + a feasible automatic plan                   */
/* ------------------------------------------------------------------ */

static void	test_feasible_auto_plan_has_required_fields(void)
{
	membrane_ctxrec_request_t		req;
	membrane_ctxrec_result_t		rec;
	membrane_plan_identity_input_t	id;
	membrane_plan_request_meta_t	meta;
	membrane_plan_t					plan;
	int								ok;

	fill_base_request(&req);
	populate_candidates(&req);
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &rec), "fixture ctxrec ok");
	fill_default_identity(&id);
	fill_default_meta(&meta);
	ok = membrane_plan_assemble(&rec, &req, &meta, &id, NULL, NULL, &plan);
	TEST_ASSERT(ok == 1, "feasible plan assembles ok=1");
	TEST_ASSERT(plan.schema_version == MEMBRANE_PLAN_SCHEMA_VERSION,
		"schema_version set");
	TEST_ASSERT(strcmp(plan.identity.model_name, "smollm2-135m-instruct")
		== 0, "identity.model_name echoed");
	TEST_ASSERT(plan.identity.installed == 1, "identity.installed set");
	TEST_ASSERT(plan.has_ctxrec_result == 1, "ctxrec result attached");
	TEST_ASSERT(plan.decisions.has_decisions == 1,
		"a feasible plan has decisions");
	TEST_ASSERT(plan.feasibility.feasible == 1, "feasible == 1");
	TEST_ASSERT(plan.feasibility.max_feasible_context_known == 1,
		"max_feasible_context established for a feasible auto plan");
	TEST_ASSERT(plan.decisions.context == rec.recommended_context,
		"decided context matches ctxrec's own recommendation");
	TEST_ASSERT(plan.decisions.context_source
		== MEMBRANE_PLAN_SOURCE_PLANNER_DECISION,
		"auto context decision attributed to the planner, not the user");
}

/* ------------------------------------------------------------------ */
/* D: explicit override preserved, and reported (not silently changed) */
/* on infeasibility                                                    */
/* ------------------------------------------------------------------ */

static void	test_explicit_context_feasible_is_preserved(void)
{
	membrane_ctxrec_request_t		req;
	membrane_ctxrec_result_t		rec;
	membrane_plan_identity_input_t	id;
	membrane_plan_request_meta_t	meta;
	membrane_plan_t					plan;

	fill_base_request(&req);
	/* Single-candidate request -- the explicit-context adapter pattern
	 * plan_cmd.cpp itself uses: exactly one caller-built candidate,
	 * still evaluated through the real, unchanged ctxrec pipeline. */
	req.model_max_context = 8192;
	req.candidates[0].ctx = 8192;
	req.candidates[0].kv_bytes_native = kv_native(8192, req.n_layer_all);
	req.candidates[0].kv_bytes_q8 = kv_q8(8192, req.n_layer_all);
	req.candidates[0].kv_bytes_q5 = kv_q5(8192, req.n_layer_all);
	req.candidate_count = 1;
	req.minimum_required_context = 8192;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &rec), "explicit ctx fits");
	fill_default_identity(&id);
	fill_default_meta(&meta);
	meta.requested_context = 8192;
	meta.context_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	membrane_plan_assemble(&rec, &req, &meta, &id, NULL, NULL, &plan);
	TEST_ASSERT(plan.decisions.context == 8192,
		"explicit context value is the decided value");
	TEST_ASSERT(plan.decisions.context_source
		== MEMBRANE_PLAN_SOURCE_EXPLICIT_USER,
		"explicit context decision attributed to the user");

	int	found_override_reason = 0;
	size_t	i;

	for (i = 0; i < plan.reason_count; ++i)
		if (strcmp(plan.reasons[i].code,
				MEMBRANE_PLAN_REASON_USER_OVERRIDE_PRESERVED) == 0)
			found_override_reason = 1;
	TEST_ASSERT(found_override_reason,
		"USER_OVERRIDE_PRESERVED reason emitted for an explicit ctx");
}

static void	test_explicit_infeasible_context_reported_not_changed(void)
{
	membrane_ctxrec_request_t		req;
	membrane_ctxrec_result_t		rec;
	membrane_plan_identity_input_t	id;
	membrane_plan_request_meta_t	meta;
	membrane_plan_t					plan;
	int								ok;

	fill_base_request(&req);
	/* An absurd explicit context this synthetic device budget cannot
	 * possibly hold, at any precision -- forces PLANNER_REJECTED_ALL. */
	req.model_max_context = 4U * 1024 * 1024;
	req.candidates[0].ctx = req.model_max_context;
	req.candidates[0].kv_bytes_native
		= kv_native(req.model_max_context, req.n_layer_all);
	req.candidates[0].kv_bytes_q8
		= kv_q8(req.model_max_context, req.n_layer_all);
	req.candidates[0].kv_bytes_q5
		= kv_q5(req.model_max_context, req.n_layer_all);
	req.candidate_count = 1;
	req.minimum_required_context = req.model_max_context;
	req.device_free_bytes = 1 * MIB;
	req.device_total_bytes = 1 * MIB;
	TEST_ASSERT(!membrane_ctxrec_resolve(&req, &rec),
		"fixture is genuinely infeasible");
	fill_default_identity(&id);
	fill_default_meta(&meta);
	meta.requested_context = req.model_max_context;
	meta.context_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	ok = membrane_plan_assemble(&rec, &req, &meta, &id, NULL, NULL, &plan);
	TEST_ASSERT(ok == 0, "assemble reports infeasible, never silently ok");
	TEST_ASSERT(plan.feasibility.feasible == 0, "feasibility.feasible == 0");
	TEST_ASSERT(plan.decisions.has_decisions == 0,
		"no decision exists for an infeasible plan");
	TEST_ASSERT(plan.workload.requested_context == req.model_max_context,
		"the originally-requested explicit context is still reported "
		"(never silently replaced)");
	TEST_ASSERT(plan.feasibility.limiting_resource[0] != '\0',
		"a limiting resource is disclosed");
}

/* ------------------------------------------------------------------ */
/* E: explainability reason codes                                     */
/* ------------------------------------------------------------------ */

static void	test_variant_estimate_only_reason_emitted(void)
{
	membrane_plan_identity_input_t		id;
	membrane_plan_request_meta_t		meta;
	membrane_plan_variant_input_t		variant;
	membrane_plan_t						plan;
	size_t								i;
	int									found = 0;

	fill_default_identity(&id);
	fill_default_meta(&meta);
	memset(&variant, 0, sizeof(variant));
	variant.has_variant = 1;
	variant.quant = "Q4_K_M";
	variant.estimate_only = 1;
	variant.source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	membrane_plan_assemble(NULL, NULL, &meta, &id, &variant, NULL, &plan);
	TEST_ASSERT(plan.identity.variant_known == 1, "variant recorded");
	TEST_ASSERT(plan.identity.variant_estimate_only == 1,
		"variant marked estimate-only");
	for (i = 0; i < plan.reason_count; ++i)
		if (strcmp(plan.reasons[i].code,
				MEMBRANE_PLAN_REASON_MODEL_VARIANT_ESTIMATE_ONLY) == 0)
			found = 1;
	TEST_ASSERT(found, "MODEL_VARIANT_ESTIMATE_ONLY reason emitted");
}

/* ------------------------------------------------------------------ */
/* G/H: no ctxrec pipeline ran at all (catalog-only, not installed)   */
/* ------------------------------------------------------------------ */

static void	test_catalog_only_model_discloses_unavailable_planning(void)
{
	membrane_plan_identity_input_t	id;
	membrane_plan_request_meta_t	meta;
	membrane_plan_t					plan;
	int								ok;
	size_t							i;
	int								found = 0;

	memset(&id, 0, sizeof(id));
	id.model_name = "smollm2-135m-instruct";
	id.installed = 0;
	id.display_name = "SmolLM2-135M-Instruct";
	id.parameter_count = "135M";
	fill_default_meta(&meta);
	ok = membrane_plan_assemble(NULL, NULL, &meta, &id, NULL, NULL, &plan);
	TEST_ASSERT(ok == 1, "a catalog-only plan is not itself 'infeasible'");
	TEST_ASSERT(plan.identity.installed == 0, "installed == 0");
	TEST_ASSERT(plan.identity.model_path[0] == '\0',
		"no model_path fabricated for an uninstalled model");
	TEST_ASSERT(plan.has_ctxrec_result == 0, "no ctxrec pipeline ran");
	TEST_ASSERT(plan.decisions.has_decisions == 0,
		"no context/GPU/KV decision exists without real model metadata");
	for (i = 0; i < plan.reason_count; ++i)
		if (strcmp(plan.reasons[i].code,
				MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE) == 0)
			found = 1;
	TEST_ASSERT(found, "MODEL_METADATA_UNAVAILABLE reason emitted");
}

/* ------------------------------------------------------------------ */
/* B: hardware snapshot -- decisions/hardware echo the SUPPLIED req,  */
/* never a second, independently-read value                          */
/* ------------------------------------------------------------------ */

static void	test_hardware_snapshot_echoed_verbatim(void)
{
	membrane_ctxrec_request_t		req;
	membrane_ctxrec_result_t		rec;
	membrane_plan_identity_input_t	id;
	membrane_plan_request_meta_t	meta;
	membrane_plan_hardware_input_t	hw;
	membrane_plan_t					plan;

	fill_base_request(&req);
	populate_candidates(&req);
	req.host_total_bytes = 12345 * MIB;
	req.host_available_bytes = 6789 * MIB;
	TEST_ASSERT(membrane_ctxrec_resolve(&req, &rec), "fixture ctxrec ok");
	fill_default_identity(&id);
	fill_default_meta(&meta);
	hw.backend = "Vulkan";
	hw.device_name = "Synthetic GPU";
	membrane_plan_assemble(&rec, &req, &meta, &id, NULL, &hw, &plan);
	TEST_ASSERT(plan.hardware.host_total_bytes == 12345 * MIB,
		"hardware.host_total_bytes matches the exact snapshot supplied");
	TEST_ASSERT(plan.hardware.host_available_bytes == 6789 * MIB,
		"hardware.host_available_bytes matches the exact snapshot supplied");
	TEST_ASSERT(strcmp(plan.hardware.backend, "Vulkan") == 0,
		"hardware.backend passthrough");
	TEST_ASSERT(strcmp(plan.hardware.device_name, "Synthetic GPU") == 0,
		"hardware.device_name passthrough");
}

int	main(void)
{
	test_feasible_auto_plan_has_required_fields();
	test_explicit_context_feasible_is_preserved();
	test_explicit_infeasible_context_reported_not_changed();
	test_variant_estimate_only_reason_emitted();
	test_catalog_only_model_discloses_unavailable_planning();
	test_hardware_snapshot_echoed_verbatim();
	printf("all membrane_plan tests passed\n");
	return (0);
}
