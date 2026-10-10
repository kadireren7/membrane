#include <string.h>

#include "orchestrator.h"
#include "test_helpers.h"

/*
 * Milestone L: synthetic matrices for the pure orchestration decision.
 * No host, no network, no clock. Capability matrices for the external
 * runtimes are hand-built "like Ollama" / "like vLLM" fixtures (the real
 * adapters' matrices are tested by their own suites); the native one is the
 * real static matrix.
 */

#define GIB	(1024ull * 1024 * 1024)

static void	caps_native(membrane_runtime_capabilities_t *c)
{
	membrane_runtime_descriptor_t	d;

	TEST_ASSERT(membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE, &d),
		"native describable");
	*c = d.capabilities;
}

static void	caps_ollama_like(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));
	c->context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->current_model = MEMBRANE_CAPABILITY_SUPPORTED;
	c->active_context = MEMBRANE_CAPABILITY_SUPPORTED;
	c->ram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->vram_usage = MEMBRANE_CAPABILITY_PARTIAL;
	c->loaded_model_memory = MEMBRANE_CAPABILITY_PARTIAL;
}

static void	caps_vllm_like(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));
	c->current_model = MEMBRANE_CAPABILITY_SUPPORTED;
	c->active_context = MEMBRANE_CAPABILITY_PARTIAL;
	c->context_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
}

static void	input(membrane_orch_runtime_input_t *in, const char *id)
{
	memset(in, 0, sizeof(*in));
	snprintf(in->runtime_id, sizeof(in->runtime_id), "%s", id);
	snprintf(in->display_name, sizeof(in->display_name), "%s", id);
	in->availability = MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	in->health = MEMBRANE_RUNTIME_HEALTH_HEALTHY;
	in->model_present = 1;
	snprintf(in->model_id, sizeof(in->model_id), "m");
	in->plan_gpu_layers = -1;
}

static void	native_in(membrane_orch_runtime_input_t *in, int feasible)
{
	input(in, MEMBRANE_RUNTIME_ID_NATIVE);
	caps_native(&in->caps);
	in->plan_valid = 1;
	in->plan_feasible = feasible;
	in->plan_context = 4096;
	in->plan_gpu_layers = 0;
	snprintf(in->plan_kv_precision, sizeof(in->plan_kv_precision), "q8");
	snprintf(in->plan_kv_placement, sizeof(in->plan_kv_placement), "host");
	in->plan_mem_known = 1;
	in->plan_host_required_bytes = 2 * GIB;
	in->context_max_known = 1;
	in->context_max = 8192;
	in->size_known = 1;
	in->size_bytes = 1 * GIB;
}

static void	ollama_in(membrane_orch_runtime_input_t *in, int resident,
				uint64_t size)
{
	input(in, MEMBRANE_RUNTIME_ID_OLLAMA);
	caps_ollama_like(&in->caps);
	in->resident_known = 1;
	in->resident = resident;
	in->size_known = size > 0;
	in->size_bytes = size;
	in->context_max_known = 1;
	in->context_max = 32768;
}

static void	vllm_in(membrane_orch_runtime_input_t *in)
{
	input(in, MEMBRANE_RUNTIME_ID_VLLM);
	caps_vllm_like(&in->caps);
	in->resident_known = 1;
	in->resident = 1;			/* vLLM serves what it was started with */
	in->context_max_known = 1;
	in->context_max = 131072;
}

static void	topo(membrane_topology_t *t, uint64_t ram_total, uint64_t ram_avail,
				int n_gpu, uint64_t vram_total, uint64_t vram_free)
{
	int	i;

	memset(t, 0, sizeof(*t));
	t->ram_known = 1;
	t->ram_total_bytes = ram_total;
	t->ram_available_bytes = ram_avail;
	for (i = 0; i < n_gpu; ++i)
	{
		membrane_orch_device_t	*d = &t->devices[t->device_count++];

		snprintf(d->id, sizeof(d->id), "gpu%d", i);
		snprintf(d->backend, sizeof(d->backend), "Vulkan");
		snprintf(d->name, sizeof(d->name), "GPU %d", i);
		d->kind = MEMBRANE_DEV_KIND_DISCRETE;
		d->total_bytes = vram_total;
		d->free_known = 1;
		d->free_bytes = vram_free;
	}
}

static int	has_reason(const membrane_orch_decision_t *d, const char *code)
{
	size_t	i;

	for (i = 0; i < d->reason_count; ++i)
		if (!strcmp(d->reasons[i], code))
			return (1);
	return (0);
}

static const char	*sel_id(const membrane_orch_decision_t *d)
{
	return (d->has_selection ? d->candidates[d->selected].runtime_id : "");
}

/* --- single-runtime worlds ---------------------------------------------- */

static void	test_native_only(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	topo(&t, 8 * GIB, 6 * GIB, 0, 0, 0);
	native_in(&in[0], 1);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.has_selection && !strcmp(sel_id(&d), "membrane-native"),
		"native selected");
	TEST_ASSERT(has_reason(&d, MEMBRANE_ORCH_REASON_ONLY_CANDIDATE)
		&& has_reason(&d, MEMBRANE_ORCH_REASON_PLANNED), "reasons");
	TEST_ASSERT(!strcmp(d.confidence, "high"), "planned => high confidence");
	TEST_ASSERT(d.candidates[0].context == 4096 && d.candidates[0].mem_bytes
		== 2 * GIB && !strcmp(d.candidates[0].mem_provenance, "planner_v2"),
		"configuration + provenance");
	TEST_ASSERT(d.candidates[0].limitation_count == 0, "full control => no "
		"limitations");
}

static void	test_ollama_only_never_gets_planner_accuracy(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	topo(&t, 16 * GIB, 12 * GIB, 0, 0, 0);
	ollama_in(&in[0], 0, 2 * GIB);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.has_selection, "selected");
	TEST_ASSERT(d.candidates[0].evidence == MEMBRANE_EVIDENCE_ESTIMATE
		&& !strcmp(d.confidence, "medium"), "an estimate from a reported "
			"size, never 'planned'");
	TEST_ASSERT(d.candidates[0].gpu_layers == -1
		&& d.candidates[0].kv_precision[0] == '\0' && !d.candidates[0]
			.context_known, "no fabricated dimensions");
	TEST_ASSERT(d.candidates[0].limitation_count == 3, "KV precision, KV "
		"placement and GPU-layer limits are disclosed");
	TEST_ASSERT(!strcmp(d.candidates[0].mem_provenance, "runtime_reported"),
		"provenance");
}

static void	test_vllm_only(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	topo(&t, 16 * GIB, 12 * GIB, 1, 24 * GIB, 20 * GIB);
	vllm_in(&in[0]);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.has_selection && !strcmp(sel_id(&d), "vllm")
		&& d.candidates[0].evidence == MEMBRANE_EVIDENCE_RESIDENT
		&& d.candidates[0].limitation_count == 4 && d.candidates[0]
			.control_count == 0, "vLLM: resident, read-only, every control "
				"limitation disclosed");
	TEST_ASSERT(!d.candidates[0].mem_known, "no memory figure invented");
}

/* --- multi-runtime ------------------------------------------------------- */

static void	test_native_plus_ollama(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[2];
	membrane_orch_decision_t	d;

	topo(&t, 16 * GIB, 12 * GIB, 1, 4 * GIB, 3 * GIB);
	native_in(&in[0], 1);
	ollama_in(&in[1], 1, 2 * GIB);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 2, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "membrane-native"), "planned + more "
		"observable");
	TEST_ASSERT(d.order_count == 2 && d.order[1] == 1, "alternatives kept");
	/* interactive: Ollama already has it loaded; native would have to load
	 * it => resident wins, even though native has a plan */
	membrane_orchestrate(MEMBRANE_INTENT_INTERACTIVE, &t, in, 2, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "ollama")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_RESIDENT),
		"interactive prefers what is already loaded; MEMBRANE does not "
		"favor itself");
	/* now native not planned-feasible: excluded, Ollama carries it */
	native_in(&in[0], 0);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 2, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "ollama") && !d.candidates[0].viable
		&& !strcmp(d.candidates[0].exclusion_code,
			MEMBRANE_ORCH_REASON_INFEASIBLE), "infeasible native excluded");
}

static void	test_intent_policies_differ(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[3];
	membrane_orch_decision_t	d;

	topo(&t, 32 * GIB, 28 * GIB, 1, 8 * GIB, 7 * GIB);
	native_in(&in[0], 1);					/* 2 GiB planned, ctx max 8192 */
	ollama_in(&in[1], 0, 1 * GIB);			/* 1 GiB reported, ctx max 32768 */
	vllm_in(&in[2]);						/* resident, ctx max 131072 */
	membrane_orchestrate(MEMBRANE_INTENT_MEMORY_SAVING, &t, in, 3, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "ollama")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_SMALLEST_MEMORY),
		"memory_saving: smallest KNOWN footprint (vLLM's is unknown, last)");
	membrane_orchestrate(MEMBRANE_INTENT_LONG_CONTEXT, &t, in, 3, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "vllm")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_LARGEST_CONTEXT),
		"long_context: largest KNOWN max context");
	membrane_orchestrate(MEMBRANE_INTENT_INTERACTIVE, &t, in, 3, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "vllm"), "interactive: resident first; "
		"between two residents-or-none the next keys decide");
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 3, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "membrane-native"), "general: planned "
		"evidence outranks resident");
	TEST_ASSERT(membrane_orch_intent_name(MEMBRANE_INTENT_MEMORY_SAVING)
		!= NULL && !strcmp(membrane_orch_intent_name(MEMBRANE_INTENT_GENERAL),
			"general"), "names");
	{
		membrane_orch_intent_t	i;

		TEST_ASSERT(membrane_orch_intent_parse("long_context", &i)
			&& i == MEMBRANE_INTENT_LONG_CONTEXT
			&& !membrane_orch_intent_parse("throughput", &i),
			"throughput is deliberately not an intent");
	}
}

static void	test_memory_saving_prefers_more_control_on_equal_memory(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[2];
	membrane_orch_decision_t	d;

	topo(&t, 32 * GIB, 28 * GIB, 0, 0, 0);
	native_in(&in[0], 1);
	native_in(&in[1], 1);
	snprintf(in[1].runtime_id, sizeof(in[1].runtime_id), "ollama");
	snprintf(in[1].display_name, sizeof(in[1].display_name), "ollama");
	caps_ollama_like(&in[1].caps);
	in[1].plan_mem_known = 1;
	in[1].plan_host_required_bytes = in[0].plan_host_required_bytes;
	in[1].plan_valid = 0;
	in[1].size_known = 1;
	in[1].size_bytes = in[0].plan_host_required_bytes;
	in[1].resident_known = 1;
	in[1].resident = 1;
	membrane_orchestrate(MEMBRANE_INTENT_MEMORY_SAVING, &t, in, 2, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "membrane-native")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_MORE_CONTROL),
		"same footprint: the runtime with more memory-reducing controls");
}

/* --- availability / feasibility / honesty ------------------------------- */

static void	test_unavailable_and_missing_model(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[3];
	membrane_orch_decision_t	d;

	topo(&t, 16 * GIB, 12 * GIB, 0, 0, 0);
	native_in(&in[0], 1);
	in[0].model_present = 0;
	ollama_in(&in[1], 1, 1 * GIB);
	in[1].availability = MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE;
	snprintf(in[1].unavailable_reason, sizeof(in[1].unavailable_reason),
		"Ollama API is not reachable.");
	vllm_in(&in[2]);
	in[2].availability = MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN;
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 3, &d);
	TEST_ASSERT(!d.has_selection && d.order_count == 0
		&& has_reason(&d, MEMBRANE_ORCH_REASON_NO_CANDIDATE)
		&& has_reason(&d, MEMBRANE_ORCH_REASON_NO_MODEL)
		&& has_reason(&d, MEMBRANE_ORCH_REASON_UNAVAILABLE),
		"nothing viable: every exclusion explained");
	TEST_ASSERT(strstr(d.candidates[1].exclusion_text, "not reachable") != NULL,
		"the runtime's own reason is preserved");
}

static void	test_low_ram_and_low_vram_no_false_feasibility(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	/* 6 GiB model on a 4 GiB RAM / 2 GiB VRAM box: exceeds everything */
	topo(&t, 4 * GIB, 3 * GIB, 1, 2 * GIB, 1 * GIB);
	ollama_in(&in[0], 0, 6 * GIB + 1);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(!d.has_selection && has_reason(&d,
			MEMBRANE_ORCH_REASON_INFEASIBLE), "cannot fit anywhere");
	/* 3 GiB model: total memory could hold it, free memory cannot prove it */
	topo(&t, 4 * GIB, 1 * GIB, 1, 2 * GIB, 512 * 1024 * 1024ull);
	ollama_in(&in[0], 0, 3 * GIB);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.has_selection && d.candidates[0].evidence
			== MEMBRANE_EVIDENCE_UNKNOWN && !strcmp(d.confidence, "low")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_UNKNOWN_FIT),
		"possible but unproven stays 'unknown', low confidence");
	/* unknown host memory: never 'estimate' */
	topo(&t, 0, 0, 0, 0, 0);
	t.ram_known = 0;
	ollama_in(&in[0], 0, 1 * GIB);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.candidates[0].evidence == MEMBRANE_EVIDENCE_UNKNOWN,
		"unknown hardware facts => unknown feasibility");
	/* no size, not resident, not native: unknown */
	topo(&t, 64 * GIB, 60 * GIB, 0, 0, 0);
	ollama_in(&in[0], 0, 0);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.candidates[0].evidence == MEMBRANE_EVIDENCE_UNKNOWN,
		"no size => unknown");
}

static void	test_unknown_capabilities_are_limitations_not_support(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	topo(&t, 16 * GIB, 12 * GIB, 0, 0, 0);
	ollama_in(&in[0], 1, 1 * GIB);
	memset(&in[0].caps, 0, sizeof(in[0].caps));		/* all UNKNOWN */
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(d.candidates[0].control_count == 0
		&& d.candidates[0].limitation_count == 4
		&& strstr(d.candidates[0].limitations[0], "unknown") != NULL
		&& d.candidates[0].observation_quality == 0,
		"unknown capability is never counted as control or observation");
}

/* --- determinism ---------------------------------------------------------- */

static void	test_deterministic_and_order_independent(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	a[3];
	membrane_orch_runtime_input_t	b[3];
	membrane_orch_decision_t	d1;
	membrane_orch_decision_t	d2;

	topo(&t, 32 * GIB, 28 * GIB, 1, 8 * GIB, 7 * GIB);
	native_in(&a[0], 1);
	ollama_in(&a[1], 1, 2 * GIB);
	vllm_in(&a[2]);
	b[0] = a[2];
	b[1] = a[0];
	b[2] = a[1];
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, a, 3, &d1);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, a, 3, &d2);
	TEST_ASSERT(memcmp(&d1, &d2, sizeof(d1)) == 0, "byte-identical re-run");
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, b, 3, &d2);
	TEST_ASSERT(!strcmp(sel_id(&d1), sel_id(&d2))
		&& !strcmp(d1.candidates[d1.order[1]].runtime_id,
			d2.candidates[d2.order[1]].runtime_id)
		&& !strcmp(d1.candidates[d1.order[2]].runtime_id,
			d2.candidates[d2.order[2]].runtime_id),
		"the ranking does not depend on input order");
}

static void	test_tie_is_broken_by_id_and_flagged_low_confidence(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[2];
	membrane_orch_decision_t	d;

	topo(&t, 32 * GIB, 28 * GIB, 0, 0, 0);
	ollama_in(&in[0], 1, 1 * GIB);
	ollama_in(&in[1], 1, 1 * GIB);
	snprintf(in[0].runtime_id, sizeof(in[0].runtime_id), "zeta");
	snprintf(in[0].display_name, sizeof(in[0].display_name), "zeta");
	snprintf(in[1].runtime_id, sizeof(in[1].runtime_id), "alpha");
	snprintf(in[1].display_name, sizeof(in[1].display_name), "alpha");
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 2, &d);
	TEST_ASSERT(!strcmp(sel_id(&d), "alpha")
		&& has_reason(&d, MEMBRANE_ORCH_REASON_TIEBREAK)
		&& !strcmp(d.confidence, "low"), "stable id tie-break, honestly "
			"labeled low confidence");
}

/* --- topology ------------------------------------------------------------- */

static void	test_topology_multi_gpu_is_preserved(void)
{
	membrane_topology_t	t;

	topo(&t, 32 * GIB, 28 * GIB, 3, 8 * GIB, 4 * GIB);
	t.devices[1].free_bytes = 6 * GIB;
	t.devices[2].free_bytes = 6 * GIB;
	TEST_ASSERT(t.device_count == 3 && membrane_topology_gpu_count(&t) == 3,
		"all three devices are kept (not 'the first GPU')");
	TEST_ASSERT(membrane_topology_pick_device(&t, 5 * GIB) == 1,
		"most free wins; tie => lowest index");
	TEST_ASSERT(membrane_topology_pick_device(&t, 7 * GIB) == -1,
		"no device covers it => none, never a guess");
	t.devices[1].free_known = 0;
	t.devices[2].free_known = 0;
	t.devices[0].free_known = 0;
	TEST_ASSERT(membrane_topology_pick_device(&t, 1) == -1,
		"unknown free memory => never picked");
	/* UMA devices share host RAM: not a separate tier to pick */
	topo(&t, 16 * GIB, 8 * GIB, 1, 16 * GIB, 8 * GIB);
	t.devices[0].kind = MEMBRANE_DEV_KIND_UMA;
	TEST_ASSERT(membrane_topology_pick_device(&t, 1) == -1
		&& membrane_topology_gpu_count(&t) == 1, "UMA is counted as a GPU "
			"but is not a distinct VRAM tier");
	/* no GPU at all */
	topo(&t, 16 * GIB, 8 * GIB, 0, 0, 0);
	TEST_ASSERT(membrane_topology_gpu_count(&t) == 0
		&& membrane_topology_pick_device(&t, 1) == -1, "CPU-only host");
}

static void	test_uma_does_not_double_count_memory(void)
{
	membrane_topology_t			t;
	membrane_orch_runtime_input_t	in[1];
	membrane_orch_decision_t	d;

	/* An APU reports 16 GiB "VRAM" that is host RAM. A 20 GiB model must
	 * not be called feasible on 16 GiB RAM + a 16 GiB UMA "device". */
	topo(&t, 16 * GIB, 12 * GIB, 1, 16 * GIB, 12 * GIB);
	t.devices[0].kind = MEMBRANE_DEV_KIND_UMA;
	ollama_in(&in[0], 0, 20 * GIB);
	membrane_orchestrate(MEMBRANE_INTENT_GENERAL, &t, in, 1, &d);
	TEST_ASSERT(!d.has_selection && has_reason(&d,
			MEMBRANE_ORCH_REASON_INFEASIBLE), "UMA memory is not added to "
				"host RAM a second time");
}

int	main(void)
{
	test_native_only();
	test_ollama_only_never_gets_planner_accuracy();
	test_vllm_only();
	test_native_plus_ollama();
	test_intent_policies_differ();
	test_memory_saving_prefers_more_control_on_equal_memory();
	test_unavailable_and_missing_model();
	test_low_ram_and_low_vram_no_false_feasibility();
	test_unknown_capabilities_are_limitations_not_support();
	test_deterministic_and_order_independent();
	test_tie_is_broken_by_id_and_flagged_low_confidence();
	test_topology_multi_gpu_is_preserved();
	test_uma_does_not_double_count_memory();
	printf("test_orchestrator: all tests passed\n");
	return (0);
}
