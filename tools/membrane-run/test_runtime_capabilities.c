#include <string.h>

#include "runtime_capabilities.h"
#include "test_helpers.h"

/*
 * Milestone H1: runtime_capabilities.h's own tests -- llama-free,
 * synthetic capability/plan fixtures only, same pattern as
 * test_membrane_plan.c/test_plan_v2_resolver.c. Plan fixtures are
 * hand-built directly (zero-init + set the two fields membrane_
 * runtime_negotiate_plan() actually reads) rather than produced via
 * membrane_plan_assemble(), so this test file needs no link against
 * membrane_plan/membrane_context_recommender/membrane_gpu_policy --
 * membrane_plan_t is a plain struct, and this module only ever reads
 * it, never builds one for real (Part 10: plan math is untouched).
 */

/* A: runtime identity -- membrane-native is stable */
static void	test_native_identity_stable(void)
{
	membrane_runtime_descriptor_t	list[4];
	membrane_runtime_descriptor_t	looked_up;
	size_t							n;

	n = membrane_runtime_discover(list, 4);
	TEST_ASSERT(n == 1, "H1 discovers exactly one runtime");
	TEST_ASSERT(strcmp(list[0].id, MEMBRANE_RUNTIME_ID_NATIVE) == 0,
		"the one runtime is membrane-native");
	TEST_ASSERT(list[0].type == MEMBRANE_RUNTIME_TYPE_NATIVE,
		"membrane-native's type is native");
	TEST_ASSERT(list[0].execution_mode == MEMBRANE_RUNTIME_EXEC_EMBEDDED_NATIVE,
		"membrane-native's execution mode is embedded_native");
	TEST_ASSERT(list[0].availability == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE,
		"membrane-native is available");
	TEST_ASSERT(list[0].endpoint_known == 0,
		"an embedded runtime has no separate endpoint");

	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE,
			&looked_up) == 1, "describe() finds membrane-native by id");
	TEST_ASSERT(strcmp(looked_up.id, list[0].id) == 0,
		"describe() and discover() agree on id");
	TEST_ASSERT(looked_up.availability == list[0].availability,
		"describe() and discover() agree on availability");

	TEST_ASSERT(membrane_runtime_describe("does-not-exist", &looked_up) == 0,
		"an unknown runtime id is not found");
	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_OLLAMA,
			&looked_up) == 0,
		"the reserved 'ollama' id is not describable in H1 -- no adapter");
	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_VLLM,
			&looked_up) == 0,
		"the reserved 'vllm' id is not describable in H1 -- no adapter");
	TEST_ASSERT(membrane_runtime_describe(NULL, &looked_up) == 0,
		"a NULL id is handled explicitly, not a crash");
}

/* G (CLI's own "no service dependency" contract, tested here at the
 * pure-logic level): membrane-native reports AVAILABLE unconditionally
 * -- this module never reads service/process state at all, so there is
 * nothing that could make it vary. */
static void	test_availability_independent_of_service_state(void)
{
	membrane_runtime_descriptor_t	d;
	int								i;

	i = 0;
	while (i < 3)
	{
		TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE,
				&d) == 1, "membrane-native is always describable");
		TEST_ASSERT(d.availability == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE,
			"membrane-native's availability never depends on whether "
			"membrane serve/the background service happens to be running");
		i++;
	}
}

/* B: capability model -- expected supported/unsupported values, each
 * one a direct claim this project's H1 audit verified against real
 * code (see runtime_capabilities.c's own field-by-field comments). */
static void	test_native_capability_matrix(void)
{
	membrane_runtime_descriptor_t	d;

	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE,
			&d) == 1, "membrane-native describes successfully");

	TEST_ASSERT(d.capabilities.model_enumeration
		== MEMBRANE_CAPABILITY_SUPPORTED, "model enumeration: supported");
	TEST_ASSERT(d.capabilities.model_load_unload
		== MEMBRANE_CAPABILITY_PARTIAL,
		"model load/unload: partial (load is explicit, unload is only "
		"automatic eviction)");
	TEST_ASSERT(d.capabilities.model_switch == MEMBRANE_CAPABILITY_SUPPORTED,
		"model switch: supported (real /membrane/v1/models/activate)");

	TEST_ASSERT(d.capabilities.context_control
		== MEMBRANE_CAPABILITY_SUPPORTED, "context control: supported");
	TEST_ASSERT(d.capabilities.gpu_layer_control
		== MEMBRANE_CAPABILITY_SUPPORTED, "GPU layer control: supported");
	TEST_ASSERT(d.capabilities.kv_precision_control
		== MEMBRANE_CAPABILITY_SUPPORTED, "KV precision control: supported");
	TEST_ASSERT(d.capabilities.kv_placement_control
		== MEMBRANE_CAPABILITY_SUPPORTED, "KV placement control: supported");
	TEST_ASSERT(d.capabilities.concurrency_control
		== MEMBRANE_CAPABILITY_PARTIAL,
		"concurrency control: partial (startup env var only, no live "
		"per-plan dial, no membrane_plan_t field for it)");
	TEST_ASSERT(d.capabilities.device_selection
		== MEMBRANE_CAPABILITY_SUPPORTED, "device selection: supported");

	TEST_ASSERT(d.capabilities.chat_completions
		== MEMBRANE_CAPABILITY_SUPPORTED, "chat completions: supported");
	TEST_ASSERT(d.capabilities.streaming == MEMBRANE_CAPABILITY_SUPPORTED,
		"streaming: supported");
	TEST_ASSERT(d.capabilities.cancellation == MEMBRANE_CAPABILITY_SUPPORTED,
		"cancellation: supported");

	TEST_ASSERT(d.capabilities.current_model == MEMBRANE_CAPABILITY_SUPPORTED,
		"current model observability: supported");
	TEST_ASSERT(d.capabilities.active_context == MEMBRANE_CAPABILITY_PARTIAL,
		"active context observability: partial");
	TEST_ASSERT(d.capabilities.ram_usage == MEMBRANE_CAPABILITY_PARTIAL,
		"RAM usage observability: partial (estimate only, no live RSS)");
	TEST_ASSERT(d.capabilities.vram_usage == MEMBRANE_CAPABILITY_PARTIAL,
		"VRAM usage observability: partial (estimate only)");
	TEST_ASSERT(d.capabilities.kv_cache_usage == MEMBRANE_CAPABILITY_PARTIAL,
		"KV cache usage observability: partial (estimate only)");
	TEST_ASSERT(d.capabilities.loaded_model_memory
		== MEMBRANE_CAPABILITY_PARTIAL,
		"loaded model memory observability: partial (estimate only)");

	TEST_ASSERT(d.capabilities.live_kv_migration
		== MEMBRANE_CAPABILITY_UNSUPPORTED,
		"live KV migration: unsupported -- does not exist");
	TEST_ASSERT(d.capabilities.dynamic_reconfiguration
		== MEMBRANE_CAPABILITY_UNSUPPORTED,
		"dynamic reconfiguration: unsupported -- sessions are fixed at "
		"construction time");

	TEST_ASSERT(d.capability_provenance
		== MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT,
		"H1's native matrix is a static, source-verified contract, "
		"never a live probe");
}

static membrane_runtime_capabilities_t	full_supported_caps(void)
{
	membrane_runtime_capabilities_t	c;
	membrane_runtime_descriptor_t		d;

	membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE, &d);
	c = d.capabilities;
	return (c);
}

static membrane_plan_t	plan_with_decisions(int variant_known)
{
	membrane_plan_t	p;

	memset(&p, 0, sizeof(p));
	p.decisions.has_decisions = 1;
	p.identity.variant_known = variant_known;
	return (p);
}

/* C: negotiation -- fully supported plan */
static void	test_negotiate_fully_supported(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;

	caps = full_supported_caps();
	plan = plan_with_decisions(1);
	outcome = membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_FULLY_SUPPORTED,
		"a plan every one of membrane-native's real capabilities "
		"satisfies negotiates as fully supported");
	TEST_ASSERT(outcome.unsupported_count == 0,
		"a fully supported plan lists no unsupported dimensions");
}

/* C: negotiation -- partially supported plan (the H1 task's own worked
 * example: context supported, GPU layers/KV precision/KV placement
 * unsupported). */
static void	test_negotiate_partial(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;

	plan = plan_with_decisions(0);
	memset(&caps, 0, sizeof(caps));
	caps.context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	caps.gpu_layer_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	caps.kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	caps.kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;

	outcome = membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_PARTIAL,
		"context-only support against a 4-dimension plan negotiates "
		"as partially supported");
	TEST_ASSERT(outcome.unsupported_count == 3,
		"exactly 3 unsupported dimensions are reported");
	{
		int		found_gpu;
		int		found_kv_prec;
		int		found_kv_place;
		size_t	i;

		found_gpu = 0;
		found_kv_prec = 0;
		found_kv_place = 0;
		i = 0;
		while (i < outcome.unsupported_count)
		{
			if (strcmp(outcome.unsupported[i],
					MEMBRANE_NEGOTIATION_DIM_GPU_LAYERS_CONTROL) == 0)
				found_gpu = 1;
			if (strcmp(outcome.unsupported[i],
					MEMBRANE_NEGOTIATION_DIM_KV_PRECISION_CONTROL) == 0)
				found_kv_prec = 1;
			if (strcmp(outcome.unsupported[i],
					MEMBRANE_NEGOTIATION_DIM_KV_PLACEMENT_CONTROL) == 0)
				found_kv_place = 1;
			i++;
		}
		TEST_ASSERT(found_gpu && found_kv_prec && found_kv_place,
			"the unsupported list names exactly GPU_LAYERS_CONTROL, "
			"KV_PRECISION_CONTROL, KV_PLACEMENT_CONTROL");
	}
}

/* C: negotiation -- unsupported plan */
static void	test_negotiate_unsupported(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;

	plan = plan_with_decisions(0);
	memset(&caps, 0, sizeof(caps));
	caps.context_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	caps.gpu_layer_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	caps.kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	caps.kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;

	outcome = membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_UNSUPPORTED,
		"a plan none of whose needed dimensions the runtime supports "
		"negotiates as unsupported");
}

/* A plan with no decisions and no known variant asks nothing of any
 * runtime -- trivially fully supported, never a fabricated deficiency. */
static void	test_negotiate_no_dimensions_needed(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;

	memset(&caps, 0, sizeof(caps));
	memset(&plan, 0, sizeof(plan));
	outcome = membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_FULLY_SUPPORTED,
		"a plan with no decisions and no known variant needs nothing "
		"from the runtime");
	TEST_ASSERT(outcome.unsupported_count == 0,
		"no dimensions are reported unsupported when none were needed");
}

/* D: unknown capability -- handled explicitly, never treated as
 * supported. */
static void	test_negotiate_unknown_not_treated_as_supported(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;
	int									i;
	int									found;

	caps = full_supported_caps();
	plan = plan_with_decisions(0);
	caps.context_control = MEMBRANE_CAPABILITY_UNKNOWN;
	outcome = membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(outcome.result != MEMBRANE_NEGOTIATION_FULLY_SUPPORTED,
		"an UNKNOWN capability is never silently treated as satisfying "
		"the plan's own requirement for it");
	found = 0;
	i = 0;
	while (i < (int)outcome.unsupported_count)
	{
		if (strcmp(outcome.unsupported[i],
				MEMBRANE_NEGOTIATION_DIM_CONTEXT_CONTROL) == 0)
			found = 1;
		i++;
	}
	TEST_ASSERT(found, "CONTEXT_CONTROL is explicitly listed as "
		"unsupported when its capability state is UNKNOWN");
}

/* Negotiation never mutates its inputs (Part 5: "read-only negotiation
 * only"). */
static void	test_negotiate_is_read_only(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_runtime_capabilities_t	caps_before;
	membrane_plan_t						plan_before;

	caps = full_supported_caps();
	plan = plan_with_decisions(1);
	caps_before = caps;
	plan_before = plan;
	(void)membrane_runtime_negotiate_plan(&caps, &plan);
	TEST_ASSERT(memcmp(&caps, &caps_before, sizeof(caps)) == 0,
		"negotiation never mutates the caller's capabilities struct");
	TEST_ASSERT(memcmp(&plan, &plan_before, sizeof(plan)) == 0,
		"negotiation never mutates the caller's plan (Planner v2 "
		"compatibility: plan representation stays exactly as produced)");
}

static void	test_negotiate_null_safety(void)
{
	membrane_runtime_capabilities_t	caps;
	membrane_plan_t						plan;
	membrane_negotiation_outcome_t		outcome;

	caps = full_supported_caps();
	plan = plan_with_decisions(1);
	outcome = membrane_runtime_negotiate_plan(NULL, &plan);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_UNSUPPORTED,
		"a NULL capabilities pointer fails closed, never a crash");
	outcome = membrane_runtime_negotiate_plan(&caps, NULL);
	TEST_ASSERT(outcome.result == MEMBRANE_NEGOTIATION_UNSUPPORTED,
		"a NULL plan pointer fails closed, never a crash");
}

static void	test_name_helpers_never_return_null(void)
{
	TEST_ASSERT(membrane_capability_state_name(
			(membrane_capability_state_t)99) != NULL,
		"an out-of-range capability state still returns a string");
	TEST_ASSERT(strcmp(membrane_capability_state_name(
			MEMBRANE_CAPABILITY_SUPPORTED), "supported") == 0,
		"SUPPORTED names as 'supported'");
	TEST_ASSERT(strcmp(membrane_runtime_type_name(
			MEMBRANE_RUNTIME_TYPE_NATIVE), "native") == 0,
		"NATIVE names as 'native'");
	TEST_ASSERT(strcmp(membrane_runtime_availability_name(
			MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE), "available") == 0,
		"AVAILABLE names as 'available'");
	TEST_ASSERT(strcmp(membrane_negotiation_result_name(
			MEMBRANE_NEGOTIATION_PARTIAL), "partially_supported") == 0,
		"PARTIAL negotiation result names as 'partially_supported'");
}

int	main(void)
{
	test_native_identity_stable();
	test_availability_independent_of_service_state();
	test_native_capability_matrix();
	test_negotiate_fully_supported();
	test_negotiate_partial();
	test_negotiate_unsupported();
	test_negotiate_no_dimensions_needed();
	test_negotiate_unknown_not_treated_as_supported();
	test_negotiate_is_read_only();
	test_negotiate_null_safety();
	test_name_helpers_never_return_null();
	printf("test_runtime_capabilities: all tests passed\n");
	return (0);
}
