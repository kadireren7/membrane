#include <string.h>

#include "memory_intelligence.h"
#include "test_helpers.h"

/*
 * Milestone I3: memory_intelligence.h/.c's own tests -- llama-free,
 * synthetic snapshot/capability fixtures only (no host probe, no
 * planner run), same pattern as test_observation.c/
 * test_runtime_capabilities.c. The CLI layer (`membrane advise`) is
 * covered end to end by tools/membrane/test_advise_cmd.cpp.
 */

#define GIB	(1024ull * 1024 * 1024)
#define MIB	(1024ull * 1024)

/* A healthy, fully-known baseline: 8 GiB RAM with 4 GiB headroom (50%),
 * an 8 GiB GPU with 4 GiB headroom (50%), context 4096/8192 (50%), a
 * small KV estimate, one resident model. Every test starts here and
 * overrides exactly the field(s) it needs to isolate one condition at a
 * time (Part 25's own "explicit boundary" discipline). */
static void	build_baseline(membrane_observation_snapshot_t *s)
{
	membrane_obs_snapshot_init(s, MEMBRANE_RUNTIME_ID_NATIVE);
	s->runtime_observable = 1;
	membrane_obs_str_set(&s->runtime_availability, "available",
		MEMBRANE_OBS_PROV_STATIC_METADATA, "runtime_capabilities");
	membrane_obs_u64_set(&s->ram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s->ram_available_bytes, 4 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s->gpu_device_count, 1, MEMBRANE_OBS_PROV_MEASURED,
		"ggml_backend_dev");
	membrane_obs_u64_set(&s->vram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_u64_set(&s->vram_free_bytes, 4 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_u64_set(&s->context_planned, 4096, MEMBRANE_OBS_PROV_ESTIMATED,
		"planner_v2");
	membrane_obs_u64_set(&s->context_model_max, 8192,
		MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	membrane_obs_u64_set(&s->kv_estimated_bytes, 64 * MIB,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s->kv_planned_precision, "native",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s->kv_planned_placement, "host",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_u64_set(&s->resident_model_count, 1,
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "server_status");
	membrane_obs_str_set(&s->configured_model, "test-model",
		MEMBRANE_OBS_PROV_CONFIGURED, "server_config.default_model");
	membrane_obs_snapshot_finalize(s);
}

static void	fully_supported_caps(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));
	c->context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->quant_variant_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->kv_precision_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->model_load_unload = MEMBRANE_CAPABILITY_PARTIAL;
}

/* A synthetic "limited external runtime" matrix -- hand-built, not
 * linked from the real Ollama adapter (same precedent as test_runtime_
 * capabilities.c's own hand-built fixtures: this module only ever READS
 * a membrane_runtime_capabilities_t, so a real adapter is unnecessary). */
static void	limited_caps(membrane_runtime_capabilities_t *c)
{
	memset(c, 0, sizeof(*c));
	c->context_control = MEMBRANE_CAPABILITY_SUPPORTED;
	c->gpu_layer_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->quant_variant_control = MEMBRANE_CAPABILITY_PARTIAL;
	c->kv_precision_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->kv_placement_control = MEMBRANE_CAPABILITY_UNSUPPORTED;
	c->model_load_unload = MEMBRANE_CAPABILITY_PARTIAL;
}

static int	has_finding(const membrane_memory_intelligence_t *r,
				const char *code)
{
	size_t	i;

	i = 0;
	while (i < r->finding_count)
	{
		if (strcmp(r->findings[i].code, code) == 0)
			return (1);
		i++;
	}
	return (0);
}

static int	has_recommendation(const membrane_memory_intelligence_t *r,
				const char *code)
{
	size_t	i;

	i = 0;
	while (i < r->recommendation_count)
	{
		if (strcmp(r->recommendations[i].code, code) == 0)
			return (1);
		i++;
	}
	return (0);
}

static const membrane_intel_finding_t	*find_finding(
				const membrane_memory_intelligence_t *r, const char *code)
{
	size_t	i;

	i = 0;
	while (i < r->finding_count)
	{
		if (strcmp(r->findings[i].code, code) == 0)
			return (&r->findings[i]);
		i++;
	}
	return (NULL);
}

/* A: healthy, low-pressure native case. */
static void	test_healthy_native(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	fully_supported_caps(&c);
	TEST_ASSERT(membrane_memory_intelligence_assemble(&s, &c, &r)
		== MEMBRANE_INTEL_STATUS_OK, "healthy baseline is OK");
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_OK, "status field matches");
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"no RAM finding when healthy");
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_VRAM_HEADROOM_LOW),
		"no VRAM finding when healthy");
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_NO_ACTION_NEEDED),
		"NO_ACTION_NEEDED is recommended when nothing is wrong");
	TEST_ASSERT(r.mutates_state == 0, "top-level mutates_state is always 0");
	TEST_ASSERT(r.model_known == 1 && strcmp(r.model_id, "") != 0,
		"model identity is set from the resident model");
}

/* B: measured RAM warning. */
static void	test_ram_warning(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_available_bytes, 800 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_WARNING,
		"low but non-critical RAM headroom is a warning");
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"HOST_MEMORY_HEADROOM_LOW is present");
	TEST_ASSERT(find_finding(&r,
			MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW)->provenance
		== MEMBRANE_OBS_PROV_MEASURED, "RAM finding provenance is measured");
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_REDUCE_CONTEXT),
		"REDUCE_CONTEXT is recommended");
	TEST_ASSERT(!has_recommendation(&r,
			MEMBRANE_INTEL_REC_USE_LOWER_MEMORY_VARIANT),
		"USE_LOWER_MEMORY_VARIANT is critical-only, not warning");
}

/* C: measured RAM critical. */
static void	test_ram_critical(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_available_bytes, 100 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_CRITICAL,
		"very low RAM headroom is critical");
	TEST_ASSERT(has_finding(&r,
			MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL),
		"HOST_MEMORY_HEADROOM_CRITICAL is present");
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_REDUCE_CONTEXT),
		"REDUCE_CONTEXT is still recommended");
	TEST_ASSERT(has_recommendation(&r,
			MEMBRANE_INTEL_REC_USE_LOWER_MEMORY_VARIANT),
		"USE_LOWER_MEMORY_VARIANT is recommended at critical RAM pressure");
}

/* D: measured VRAM warning. */
static void	test_vram_warning(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.vram_free_bytes, 1 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_VRAM_HEADROOM_LOW),
		"VRAM_HEADROOM_LOW is present");
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_REDUCE_GPU_LAYERS),
		"REDUCE_GPU_LAYERS is recommended");
}

/* E/F: estimate-only KV warning, and it is labeled estimated. */
static void	test_kv_estimate_high_labeled(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;
	const membrane_intel_finding_t		*f;

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_available_bytes, 2 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.kv_estimated_bytes, (uint64_t)(1.2 * (double)GIB),
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"RAM itself stays healthy (25% headroom) -- isolates the KV check");
	f = find_finding(&r, MEMBRANE_INTEL_CODE_KV_FOOTPRINT_ESTIMATE_HIGH);
	TEST_ASSERT(f != NULL, "KV_FOOTPRINT_ESTIMATE_HIGH fires");
	TEST_ASSERT(f->severity == MEMBRANE_INTEL_SEVERITY_NOTICE,
		"an estimate-only finding never exceeds NOTICE");
	TEST_ASSERT(f->provenance == MEMBRANE_OBS_PROV_ESTIMATED,
		"the KV finding is explicitly labeled estimated, never measured");
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_NOTICE,
		"an estimate-only finding never outranks a healthy measured state");
	TEST_ASSERT(has_recommendation(&r,
			MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION),
		"USE_LOWER_KV_PRECISION is recommended");
}

/* An estimate must never outrank a MEASURED critical finding (Part 16). */
static void	test_estimate_never_outranks_measured_critical(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_available_bytes, 50 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	/* > 0.5 * 50 MiB headroom -- the KV estimate finding fires too. */
	membrane_obs_u64_set(&s.kv_estimated_bytes, 40 * MIB,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_KV_FOOTPRINT_ESTIMATE_HIGH),
		"the estimate-based finding still fires alongside the measured one");
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_CRITICAL,
		"measured RAM critical still dominates overall status");
}

/* G: Ollama-shaped multiple models without pressure -> info only. */
static void	test_multiple_models_no_pressure(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;
	const membrane_intel_finding_t		*f;

	build_baseline(&s);
	membrane_obs_u64_set(&s.resident_model_count, 2,
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "api_ps");
	membrane_obs_snapshot_finalize(&s);
	limited_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	f = find_finding(&r, MEMBRANE_INTEL_CODE_MULTIPLE_MODELS_RESIDENT);
	TEST_ASSERT(f != NULL, "MULTIPLE_MODELS_RESIDENT fires");
	TEST_ASSERT(f->severity == MEMBRANE_INTEL_SEVERITY_INFO,
		"no pressure -> info-only severity");
	TEST_ASSERT(!has_recommendation(&r, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL),
		"no unload recommendation without memory pressure");
}

/* H: Ollama-shaped multiple models + pressure -> unload recommendation. */
static void	test_multiple_models_with_pressure(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;
	const membrane_intel_recommendation_t	*rec;
	size_t									i;

	build_baseline(&s);
	membrane_obs_u64_set(&s.resident_model_count, 2,
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "api_ps");
	membrane_obs_u64_set(&s.vram_free_bytes, 200 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_snapshot_finalize(&s);
	limited_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL),
		"unload is recommended once real pressure is present");
	rec = NULL;
	i = 0;
	while (i < r.recommendation_count && rec == NULL)
	{
		if (strcmp(r.recommendations[i].code,
				MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL) == 0)
			rec = &r.recommendations[i];
		i++;
	}
	TEST_ASSERT(rec != NULL && rec->mutates_state == 0,
		"the recommendation never claims to mutate state");
	TEST_ASSERT(rec != NULL
		&& rec->applicability == MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY,
		"unloading is always advisory (external action), never claimed"
		" controllable");
}

/* I: unsupported KV precision control. */
static void	test_kv_precision_unsupported(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;
	const membrane_intel_recommendation_t	*rec;
	size_t									i;

	build_baseline(&s);
	membrane_obs_u64_set(&s.vram_free_bytes, 200 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_snapshot_finalize(&s);
	limited_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r,
			MEMBRANE_INTEL_CODE_KV_PRECISION_CONTROL_UNAVAILABLE),
		"KV_PRECISION_CONTROL_UNAVAILABLE fires when a KV plan exists");
	rec = NULL;
	i = 0;
	while (i < r.recommendation_count && rec == NULL)
	{
		if (strcmp(r.recommendations[i].code,
				MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION) == 0)
			rec = &r.recommendations[i];
		i++;
	}
	TEST_ASSERT(rec != NULL, "the recommendation still surfaces");
	TEST_ASSERT(rec != NULL && rec->applicability
		== MEMBRANE_INTEL_APPLICABILITY_UNAVAILABLE_ON_RUNTIME,
		"but is marked unavailable on this runtime, never controllable");
}

/* J: partial GPU-layer control. */
static void	test_gpu_layer_control_partial(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	limited_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_GPU_LAYER_CONTROL_PARTIAL),
		"GPU_LAYER_CONTROL_PARTIAL fires when a GPU is present");
}

/* K: context near model maximum. */
static void	test_context_near_maximum(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;
	const membrane_intel_finding_t		*f;

	build_baseline(&s);
	membrane_obs_u64_set(&s.context_planned, 7500, MEMBRANE_OBS_PROV_ESTIMATED,
		"planner_v2");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	f = find_finding(&r, MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM);
	TEST_ASSERT(f != NULL, "CONTEXT_NEAR_MODEL_MAXIMUM fires at 7500/8192");
	TEST_ASSERT(f->severity == MEMBRANE_INTEL_SEVERITY_NOTICE,
		"near-maximum is a notice, not a warning");
}

static void	test_context_exceeds_maximum(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.context_planned, 9000, MEMBRANE_OBS_PROV_ESTIMATED,
		"planner_v2");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r,
			MEMBRANE_INTEL_CODE_CONTEXT_EXCEEDS_MODEL_MAXIMUM),
		"CONTEXT_EXCEEDS_MODEL_MAXIMUM fires when planned > model max");
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM),
		"exceeds and near-max are mutually exclusive");
}

/* L: unknown context -> no fabricated finding. */
static void	test_unknown_context_no_fabrication(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_unknown(&s.context_model_max, "not_in_gguf");
	membrane_obs_snapshot_finalize(&s);
	fully_supported_caps(&c);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM)
		&& !has_finding(&r,
			MEMBRANE_INTEL_CODE_CONTEXT_EXCEEDS_MODEL_MAXIMUM),
		"an unknown model max never produces a context finding");
	TEST_ASSERT(r.status != MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA,
		"other known dimensions keep this a real assessment, not "
		"insufficient_data");
}

/* M: insufficient data. */
static void	test_insufficient_data(void)
{
	membrane_observation_snapshot_t	s;
	membrane_memory_intelligence_t		r;

	membrane_obs_snapshot_init(&s, MEMBRANE_RUNTIME_ID_NATIVE);
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(membrane_memory_intelligence_assemble(&s, NULL, &r)
		== MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA,
		"an all-unknown snapshot is insufficient_data, never OK");
	TEST_ASSERT(r.finding_count == 1
		&& strcmp(r.findings[0].code, MEMBRANE_INTEL_CODE_TELEMETRY_INCOMPLETE)
			== 0, "exactly one TELEMETRY_INCOMPLETE finding is produced");
	TEST_ASSERT(r.recommendation_count == 0,
		"no recommendation is fabricated when there is no data to base it on");

	membrane_memory_intelligence_assemble(NULL, NULL, &r);
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA,
		"a NULL snapshot is handled explicitly, not a crash");
}

/* N: deterministic recommendation ordering. */
static void	test_deterministic_ordering(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r1;
	membrane_memory_intelligence_t		r2;
	size_t								i;

	membrane_obs_snapshot_init(&s, "ollama");
	s.runtime_observable = 1;
	membrane_obs_str_set(&s.runtime_availability, "available",
		MEMBRANE_OBS_PROV_STATIC_METADATA, "runtime_ollama");
	membrane_obs_u64_set(&s.ram_total_bytes, 8 * GIB, MEMBRANE_OBS_PROV_MEASURED,
		"proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 100 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.gpu_device_count, 1, MEMBRANE_OBS_PROV_MEASURED,
		"ggml_backend_dev");
	membrane_obs_u64_set(&s.vram_total_bytes, 8 * GIB, MEMBRANE_OBS_PROV_MEASURED,
		"ggml_backend_dev_memory");
	membrane_obs_u64_set(&s.vram_free_bytes, 200 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_u64_set(&s.resident_model_count, 2,
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "api_ps");
	membrane_obs_str_set(&s.kv_planned_precision, "native",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s.kv_planned_placement, "host",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(&s);
	limited_caps(&c);

	membrane_memory_intelligence_assemble(&s, &c, &r1);
	membrane_memory_intelligence_assemble(&s, &c, &r2);
	TEST_ASSERT(r1.recommendation_count == r2.recommendation_count
		&& r1.recommendation_count > 1,
		"the fixture actually produces more than one recommendation");
	i = 0;
	while (i < r1.recommendation_count)
	{
		TEST_ASSERT(strcmp(r1.recommendations[i].code,
				r2.recommendations[i].code) == 0,
			"repeated assembly of the same inputs is byte-identical");
		i++;
	}
	/* The fixed Part 17 tiers: measured critical/warning pressure before
	 * runtime-reported residency before capability-driven KV items. */
	TEST_ASSERT(strcmp(r1.recommendations[0].code,
			MEMBRANE_INTEL_REC_REDUCE_CONTEXT) == 0,
		"critical measured RAM pressure ranks first");
	TEST_ASSERT(strcmp(r1.recommendations[1].code,
			MEMBRANE_INTEL_REC_USE_LOWER_MEMORY_VARIANT) == 0,
		"still tier 1 (critical RAM)");
	TEST_ASSERT(strcmp(r1.recommendations[2].code,
			MEMBRANE_INTEL_REC_REDUCE_GPU_LAYERS) == 0,
		"VRAM pressure ranks next");
	TEST_ASSERT(strcmp(r1.recommendations[3].code,
			MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL) == 0,
		"runtime-reported residency ranks before capability-only items");
}

/* Explicit boundary checks (Part 25): just below / at / above each
 * threshold, strict "<" semantics documented in memory_intelligence.c.
 * Every byte value below is chosen so the boundary ratio (15%, 5%) is an
 * EXACT rational with a power-of-two denominator -- e.g. 3 GiB / 20 GiB
 * == 0.15 exactly, in both the mathematical sense and after IEEE-754
 * rounding (source literal 0.15 and the runtime division 3*GIB/(20*GIB)
 * round the same real number to the same nearest double) -- so "exactly
 * at the threshold" is unambiguous and never a truncated approximation
 * of it. */
static void	test_ram_ratio_warning_boundary(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	fully_supported_caps(&c);

	/* 3 GiB / 20 GiB == 0.15 exactly: must NOT warn (ratio < 0.15 is
	 * false when ratio == 0.15). */
	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 20 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 3 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"exactly at the 15% ratio does not warn (exclusive '<')");

	/* One byte less: ratio strictly < 0.15 -- must warn. */
	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 20 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 3 * GIB - 1,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"one byte below the 15% ratio warns");
}

static void	test_ram_ratio_critical_boundary(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	fully_supported_caps(&c);

	/* 1 GiB / 20 GiB == 0.05 exactly: must warn, not critical. */
	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 20 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 1 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW)
		&& !has_finding(&r,
			MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL),
		"exactly at the 5% ratio is a warning, not yet critical");

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 20 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 1 * GIB - 1,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r,
			MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_CRITICAL),
		"one byte below the 5% ratio is critical");
}

/* Isolates the absolute floor from the ratio check: at a small enough
 * total, 768 MiB is well above the 15% ratio line (2 GiB * 0.15 ==
 * 307.2 MiB), so a headroom exactly AT 768 MiB is ratio-healthy (37.5%)
 * and only the absolute floor is at stake. */
static void	test_ram_absolute_floor_boundary(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	fully_supported_caps(&c);

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 2 * GIB, MEMBRANE_OBS_PROV_MEASURED,
		"proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 768 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(!has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"exactly at the absolute floor, ratio-healthy -> no warning");

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_total_bytes, 2 * GIB, MEMBRANE_OBS_PROV_MEASURED,
		"proc_meminfo");
	membrane_obs_u64_set(&s.ram_available_bytes, 768 * MIB - 1,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_HOST_MEMORY_HEADROOM_LOW),
		"one byte below the absolute floor warns, purely via the "
		"absolute check (ratio alone would still be healthy here)");
}

static void	test_context_ratio_boundary(void)
{
	membrane_observation_snapshot_t	s;
	membrane_runtime_capabilities_t	c;
	membrane_memory_intelligence_t		r;

	fully_supported_caps(&c);

	/* Exactly at 90% of 8192 == 7372.8 -- use an exact-integer ratio
	 * instead so the boundary is unambiguous: 900/1000. */
	build_baseline(&s);
	membrane_obs_u64_set(&s.context_model_max, 1000,
		MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	membrane_obs_u64_set(&s.context_planned, 900, MEMBRANE_OBS_PROV_ESTIMATED,
		"planner_v2");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(has_finding(&r, MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM),
		"exactly at the 90% ratio counts as near (inclusive '>=')");

	build_baseline(&s);
	membrane_obs_u64_set(&s.context_model_max, 1000,
		MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	membrane_obs_u64_set(&s.context_planned, 899, MEMBRANE_OBS_PROV_ESTIMATED,
		"planner_v2");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, &c, &r);
	TEST_ASSERT(!has_finding(&r,
			MEMBRANE_INTEL_CODE_CONTEXT_NEAR_MODEL_MAXIMUM),
		"one unit below the 90% ratio does not count as near");
}

static void	test_caps_null_is_safe(void)
{
	membrane_observation_snapshot_t	s;
	membrane_memory_intelligence_t		r;

	build_baseline(&s);
	membrane_obs_u64_set(&s.ram_available_bytes, 50 * MIB,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_snapshot_finalize(&s);
	membrane_memory_intelligence_assemble(&s, NULL, &r);
	TEST_ASSERT(r.status == MEMBRANE_INTEL_STATUS_CRITICAL,
		"a NULL capability matrix never blocks a measured finding");
	TEST_ASSERT(has_recommendation(&r, MEMBRANE_INTEL_REC_REDUCE_CONTEXT),
		"the recommendation still surfaces with NULL capabilities");
}

static void	test_names_never_crash(void)
{
	TEST_ASSERT(strcmp(membrane_intel_status_name(
		(membrane_intel_status_t)99), "ok") == 0,
		"out-of-range status name degrades to ok, never garbage");
	TEST_ASSERT(strcmp(membrane_intel_severity_name(
		(membrane_intel_severity_t)99), "info") == 0,
		"out-of-range severity name degrades to info");
	TEST_ASSERT(strcmp(membrane_intel_dimension_name(
		(membrane_intel_dimension_t)99), "telemetry") == 0,
		"out-of-range dimension name degrades to telemetry");
	TEST_ASSERT(strcmp(membrane_intel_applicability_name(
		(membrane_intel_applicability_t)99), "advisory_only") == 0,
		"out-of-range applicability name degrades to advisory_only");
}

int	main(void)
{
	test_healthy_native();
	test_ram_warning();
	test_ram_critical();
	test_vram_warning();
	test_kv_estimate_high_labeled();
	test_estimate_never_outranks_measured_critical();
	test_multiple_models_no_pressure();
	test_multiple_models_with_pressure();
	test_kv_precision_unsupported();
	test_gpu_layer_control_partial();
	test_context_near_maximum();
	test_context_exceeds_maximum();
	test_unknown_context_no_fabrication();
	test_insufficient_data();
	test_deterministic_ordering();
	test_ram_ratio_warning_boundary();
	test_ram_ratio_critical_boundary();
	test_ram_absolute_floor_boundary();
	test_context_ratio_boundary();
	test_caps_null_is_safe();
	test_names_never_crash();
	printf("test_memory_intelligence: all tests passed\n");
	return (0);
}
