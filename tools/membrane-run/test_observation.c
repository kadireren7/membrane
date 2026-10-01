#include <string.h>

#include "observation.h"
#include "test_helpers.h"

/*
 * Milestone I1: the pure observation model's own tests -- llama-free,
 * synthetic values only (no host probe), same pattern as
 * test_membrane_plan.c. The native provider and `membrane observe` are
 * covered end to end by tools/membrane/test_observe_cmd.cpp.
 */

static void	test_provenance_names(void)
{
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(MEMBRANE_OBS_PROV_UNKNOWN),
		"unknown") == 0, "unknown name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(MEMBRANE_OBS_PROV_MEASURED),
		"measured") == 0, "measured name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED), "runtime_reported") == 0,
		"runtime_reported name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(
		MEMBRANE_OBS_PROV_ESTIMATED), "estimated") == 0, "estimated name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(
		MEMBRANE_OBS_PROV_STATIC_METADATA), "static_metadata") == 0,
		"static_metadata name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(
		MEMBRANE_OBS_PROV_CONFIGURED), "configured") == 0, "configured name");
	TEST_ASSERT(strcmp(membrane_obs_provenance_name(
		(membrane_obs_provenance_t)99), "unknown") == 0,
		"out-of-range provenance is unknown, never a real category");
	TEST_ASSERT(strcmp(membrane_obs_status_name(MEMBRANE_OBS_STATUS_COMPLETE),
		"complete") == 0, "complete name");
	TEST_ASSERT(strcmp(membrane_obs_status_name(MEMBRANE_OBS_STATUS_PARTIAL),
		"partial") == 0, "partial name");
	TEST_ASSERT(strcmp(membrane_obs_status_name(
		MEMBRANE_OBS_STATUS_UNAVAILABLE), "unavailable") == 0,
		"unavailable name");
}

static void	test_setter_invariants(void)
{
	membrane_obs_u64_t	u;
	membrane_obs_bool_t	b;
	membrane_obs_str_t	s;

	membrane_obs_u64_set(&u, 42, MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	TEST_ASSERT(u.known && u.value == 42, "known u64 keeps its value");
	TEST_ASSERT(u.provenance == MEMBRANE_OBS_PROV_MEASURED, "provenance kept");
	TEST_ASSERT(strcmp(u.source, "proc_meminfo") == 0, "source kept");

	/* A known value with UNKNOWN provenance must be impossible. */
	membrane_obs_u64_set(&u, 42, MEMBRANE_OBS_PROV_UNKNOWN, "why");
	TEST_ASSERT(!u.known && u.value == 0
		&& u.provenance == MEMBRANE_OBS_PROV_UNKNOWN,
		"set() with UNKNOWN provenance degrades to unknown");

	membrane_obs_u64_set(&u, 7, MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_u64_unknown(&u, "no_gpu_device");
	TEST_ASSERT(!u.known && u.value == 0, "unknown clears the old value");
	TEST_ASSERT(u.provenance == MEMBRANE_OBS_PROV_UNKNOWN,
		"unknown resets provenance");
	TEST_ASSERT(strcmp(u.source, "no_gpu_device") == 0,
		"unknown keeps its reason");

	membrane_obs_bool_set(&b, 5, MEMBRANE_OBS_PROV_MEASURED, "systemctl");
	TEST_ASSERT(b.known && b.value == 1, "bool normalized to 0/1");
	membrane_obs_bool_set(&b, 1, MEMBRANE_OBS_PROV_UNKNOWN, NULL);
	TEST_ASSERT(!b.known && b.value == 0 && b.source[0] == '\0',
		"bool unknown via UNKNOWN provenance, NULL source -> \"\"");

	membrane_obs_str_set(&s, "Vulkan", MEMBRANE_OBS_PROV_MEASURED, "ggml");
	TEST_ASSERT(s.known && strcmp(s.value, "Vulkan") == 0, "str known");
	membrane_obs_str_set(&s, NULL, MEMBRANE_OBS_PROV_MEASURED, "ggml");
	TEST_ASSERT(!s.known && s.value[0] == '\0'
		&& s.provenance == MEMBRANE_OBS_PROV_UNKNOWN,
		"NULL string value is unknown, not an empty known string");

	/* Oversized inputs are truncated, never overflow. */
	{
		char	big[512];

		memset(big, 'x', sizeof(big) - 1);
		big[sizeof(big) - 1] = '\0';
		membrane_obs_str_set(&s, big, MEMBRANE_OBS_PROV_CONFIGURED, big);
		TEST_ASSERT(strlen(s.value) == MEMBRANE_OBS_STR_MAX - 1,
			"value truncated to its buffer");
		TEST_ASSERT(strlen(s.source) == MEMBRANE_OBS_SOURCE_MAX - 1,
			"source truncated to its buffer");
	}
}

static void	test_init_all_unknown(void)
{
	membrane_observation_snapshot_t	s;
	membrane_obs_field_ref_t		refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t							n;
	size_t							i;
	size_t							known;
	size_t							total;

	memset(&s, 0xAB, sizeof(s));
	membrane_obs_snapshot_init(&s, "membrane-native");
	TEST_ASSERT(s.schema_version == MEMBRANE_OBSERVATION_SCHEMA_VERSION,
		"schema version set");
	TEST_ASSERT(strcmp(s.runtime_id, "membrane-native") == 0, "runtime id");
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"a fresh snapshot is unavailable until finalized");
	TEST_ASSERT(s.resident_model_len == 0, "no resident models");
	n = membrane_obs_snapshot_fields(&s, refs, MEMBRANE_OBS_MAX_FIELDS);
	TEST_ASSERT(n > 0 && n <= MEMBRANE_OBS_MAX_FIELDS, "field table fits");
	for (i = 0; i < n; ++i)
	{
		TEST_ASSERT(!membrane_obs_field_known(&refs[i]),
			"every field starts unknown");
		TEST_ASSERT(membrane_obs_field_provenance(&refs[i])
			== MEMBRANE_OBS_PROV_UNKNOWN, "every field starts UNKNOWN");
		TEST_ASSERT(strcmp(membrane_obs_field_source(&refs[i]),
			"not_collected") == 0, "every field says not_collected");
	}
	membrane_obs_snapshot_count(&s, &known, &total);
	TEST_ASSERT(known == 0 && total == n, "count agrees with the table");
}

static void	test_field_table_paths_unique(void)
{
	membrane_observation_snapshot_t	s;
	membrane_obs_field_ref_t		refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t							n;
	size_t							i;
	size_t							j;

	membrane_obs_snapshot_init(&s, "membrane-native");
	n = membrane_obs_snapshot_fields(&s, refs, MEMBRANE_OBS_MAX_FIELDS);
	for (i = 0; i < n; ++i)
	{
		TEST_ASSERT(strchr(refs[i].path, '.') != NULL,
			"every path is section.field");
		for (j = i + 1; j < n; ++j)
		{
			TEST_ASSERT(strcmp(refs[i].path, refs[j].path) != 0,
				"field paths are unique");
			TEST_ASSERT(refs[i].field != refs[j].field,
				"no field listed twice");
		}
	}
	/* A short output buffer still reports the true count. */
	TEST_ASSERT(membrane_obs_snapshot_fields(&s, refs, 2) == n,
		"table size is independent of max_out");
}

static void	set_measured_ram(membrane_observation_snapshot_t *s,
				uint64_t total, uint64_t avail)
{
	membrane_obs_u64_set(&s->ram_total_bytes, total,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
	membrane_obs_u64_set(&s->ram_available_bytes, avail,
		MEMBRANE_OBS_PROV_MEASURED, "proc_meminfo");
}

static void	test_derived_fields(void)
{
	membrane_observation_snapshot_t	s;

	membrane_obs_snapshot_init(&s, "membrane-native");
	s.runtime_observable = 1;
	set_measured_ram(&s, 8000, 3000);
	membrane_obs_u64_set(&s.vram_total_bytes, 4000,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_u64_set(&s.vram_free_bytes, 1500,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.ram_used_bytes.known && s.ram_used_bytes.value == 5000,
		"used RAM = total - available");
	TEST_ASSERT(s.ram_used_bytes.provenance == MEMBRANE_OBS_PROV_MEASURED,
		"derived from measured stays measured");
	TEST_ASSERT(s.vram_used_bytes.known && s.vram_used_bytes.value == 2500,
		"used VRAM = total - free");
	TEST_ASSERT(s.ram_headroom_bytes.known
		&& s.ram_headroom_bytes.value == 3000,
		"RAM headroom is the raw available figure");
	TEST_ASSERT(s.vram_headroom_bytes.known
		&& s.vram_headroom_bytes.value == 1500,
		"VRAM headroom is the raw free figure");

	/* Inconsistent inputs never produce a wrapped-around number. */
	set_measured_ram(&s, 1000, 2000);
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(!s.ram_used_bytes.known
		&& strcmp(s.ram_used_bytes.source, "inputs_inconsistent") == 0,
		"available > total -> used unknown");

	/* Missing input -> derived unknown, not 0. */
	membrane_obs_u64_unknown(&s.ram_total_bytes, "probe_failed");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(!s.ram_used_bytes.known
		&& strcmp(s.ram_used_bytes.source, "inputs_unknown") == 0,
		"total unknown -> used unknown");

	/* Mixed provenance is never silently blended into one category. */
	set_measured_ram(&s, 8000, 3000);
	membrane_obs_u64_set(&s.ram_available_bytes, 3000,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(!s.ram_used_bytes.known, "measured - estimated is refused");

	membrane_obs_u64_set(&s.vram_free_bytes, 5000,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(!s.vram_used_bytes.known && !s.vram_headroom_bytes.known,
		"VRAM free > total -> used and headroom unknown");
}

static void	test_estimate_is_never_promoted(void)
{
	membrane_observation_snapshot_t	s;

	membrane_obs_snapshot_init(&s, "membrane-native");
	s.runtime_observable = 1;
	membrane_obs_u64_set(&s.kv_estimated_bytes, 620u << 20,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.kv_estimated_bytes.provenance == MEMBRANE_OBS_PROV_ESTIMATED,
		"planner KV stays estimated through finalize");
	TEST_ASSERT(!s.kv_measured_bytes.known
		&& s.kv_measured_bytes.provenance == MEMBRANE_OBS_PROV_UNKNOWN,
		"an estimate never fills the measured KV field");
}

static void	test_status(void)
{
	membrane_observation_snapshot_t	s;
	membrane_obs_field_ref_t		refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t							n;
	size_t							i;

	membrane_obs_snapshot_init(&s, "membrane-native");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_UNAVAILABLE,
		"runtime not observable -> unavailable");

	s.runtime_observable = 1;
	set_measured_ram(&s, 8000, 3000);
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_PARTIAL,
		"RAM known, the rest unknown -> partial");

	/* Every field known -> complete. */
	n = membrane_obs_snapshot_fields(&s, refs, MEMBRANE_OBS_MAX_FIELDS);
	for (i = 0; i < n; ++i)
	{
		if (refs[i].kind == MEMBRANE_OBS_KIND_U64)
			membrane_obs_u64_set((membrane_obs_u64_t *)refs[i].field, 1,
				MEMBRANE_OBS_PROV_MEASURED, "t");
		else if (refs[i].kind == MEMBRANE_OBS_KIND_BOOL)
			membrane_obs_bool_set((membrane_obs_bool_t *)refs[i].field, 1,
				MEMBRANE_OBS_PROV_MEASURED, "t");
		else
			membrane_obs_str_set((membrane_obs_str_t *)refs[i].field, "v",
				MEMBRANE_OBS_PROV_MEASURED, "t");
	}
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_COMPLETE,
		"every field known -> complete");

	/* One unknown field (e.g. active context) -> partial again. */
	membrane_obs_u64_unknown(&s.context_active, "not_reported_by_runtime");
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_PARTIAL,
		"a single unknown field -> partial, deterministically");

	/* finalize is idempotent. */
	membrane_obs_snapshot_finalize(&s);
	TEST_ASSERT(s.status == MEMBRANE_OBS_STATUS_PARTIAL, "idempotent");
}

static void	test_format_utc(void)
{
	char	buf[32];

	TEST_ASSERT(membrane_obs_format_utc(0, buf, sizeof(buf)), "epoch ok");
	TEST_ASSERT(strcmp(buf, "1970-01-01T00:00:00.000Z") == 0, "epoch text");
	TEST_ASSERT(membrane_obs_format_utc(951782400123LL, buf, sizeof(buf)),
		"leap day ok");
	TEST_ASSERT(strcmp(buf, "2000-02-29T00:00:00.123Z") == 0,
		"2000-02-29 (leap century)");
	TEST_ASSERT(membrane_obs_format_utc(1790246096789LL, buf, sizeof(buf)),
		"2026 ok");
	TEST_ASSERT(strcmp(buf, "2026-09-24T10:34:56.789Z") == 0, "2026 text");
	TEST_ASSERT(!membrane_obs_format_utc(-1, buf, sizeof(buf)),
		"negative rejected");
	TEST_ASSERT(!membrane_obs_format_utc(0, buf, 10), "short buffer rejected");
}

int	main(void)
{
	test_provenance_names();
	test_setter_invariants();
	test_init_all_unknown();
	test_field_table_paths_unique();
	test_derived_fields();
	test_estimate_is_never_promoted();
	test_status();
	test_format_utc();
	printf("test_observation: all tests passed\n");
	return (0);
}
