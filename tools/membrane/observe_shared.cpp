#include "observe_shared.h"

#include <cstdio>

using json = nlohmann::json;

/* See observe_shared.h's own top comment. Moved out of observe_cmd.cpp
 * (Milestone I1) verbatim except for taking raw parameters instead of the
 * membrane-native-only membrane_observe_inputs_t, so Milestone I2's Ollama
 * provider (observe_ollama.cpp) can call the SAME mapping/rendering
 * instead of duplicating it. */

/* ------------------------------------------------------------------ */
/* Host / GPU mapping -- MEASURED, runtime-agnostic                    */
/* ------------------------------------------------------------------ */

void	membrane_observe_build_host_fields(bool meminfo_probed,
			const membrane_host_meminfo_t &meminfo, bool swap_supported,
			const std::string &meminfo_source,
			membrane_observation_snapshot_t *s)
{
	const char	*src = meminfo_source.c_str();

	if (!meminfo_probed || !meminfo.ok)
	{
		const char	*why = meminfo_probed ? "probe_failed" : "not_probed";

		membrane_obs_u64_unknown(&s->ram_total_bytes, why);
		membrane_obs_u64_unknown(&s->ram_available_bytes, why);
		membrane_obs_u64_unknown(&s->swap_total_bytes, why);
		membrane_obs_u64_unknown(&s->swap_free_bytes, why);
	}
	else
	{
		membrane_obs_u64_set(&s->ram_total_bytes, meminfo.total_bytes,
			MEMBRANE_OBS_PROV_MEASURED, src);
		membrane_obs_u64_set(&s->ram_available_bytes,
			meminfo.available_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
		if (swap_supported)
		{
			membrane_obs_u64_set(&s->swap_total_bytes,
				meminfo.swap_total_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
			membrane_obs_u64_set(&s->swap_free_bytes,
				meminfo.swap_free_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
		}
		else
		{
			membrane_obs_u64_unknown(&s->swap_total_bytes,
				"not_probed_on_platform");
			membrane_obs_u64_unknown(&s->swap_free_bytes,
				"not_probed_on_platform");
		}
	}
	/* There is no RSS probe in this codebase, and `membrane observe`'s
	 * own RSS would say nothing about a runtime it is merely observing. */
	membrane_obs_u64_unknown(&s->process_rss_bytes, "not_instrumented");
}

void	membrane_observe_build_gpu_fields(bool gpu_enumerated,
			const std::vector<membrane_gpu_device_info_t> &devices,
			membrane_observation_snapshot_t *s)
{
	const char	*src = "ggml_backend_dev";
	int			gpu_index = -1;
	uint64_t	n_gpu = 0;

	if (!gpu_enumerated)
	{
		membrane_obs_u64_unknown(&s->gpu_device_count, "not_probed");
		membrane_obs_str_unknown(&s->gpu_backend, "not_probed");
		membrane_obs_str_unknown(&s->gpu_device_name, "not_probed");
		membrane_obs_str_unknown(&s->gpu_device_description, "not_probed");
		membrane_obs_u64_unknown(&s->vram_total_bytes, "not_probed");
		membrane_obs_u64_unknown(&s->vram_free_bytes, "not_probed");
		return ;
	}
	for (size_t i = 0; i < devices.size(); ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
		{
			if (gpu_index < 0)
				gpu_index = (int)i;
			n_gpu++;
		}
	membrane_obs_u64_set(&s->gpu_device_count, n_gpu,
		MEMBRANE_OBS_PROV_MEASURED, src);
	if (gpu_index < 0)
	{
		membrane_obs_str_unknown(&s->gpu_backend, "no_gpu_device");
		membrane_obs_str_unknown(&s->gpu_device_name, "no_gpu_device");
		membrane_obs_str_unknown(&s->gpu_device_description,
			"no_gpu_device");
		membrane_obs_u64_unknown(&s->vram_total_bytes, "no_gpu_device");
		membrane_obs_u64_unknown(&s->vram_free_bytes, "no_gpu_device");
		return ;
	}

	const membrane_gpu_device_info_t	&d = devices[(size_t)gpu_index];

	membrane_obs_str_set(&s->gpu_backend, d.backend,
		MEMBRANE_OBS_PROV_MEASURED, src);
	membrane_obs_str_set(&s->gpu_device_name, d.name,
		MEMBRANE_OBS_PROV_MEASURED, src);
	membrane_obs_str_set(&s->gpu_device_description, d.description,
		MEMBRANE_OBS_PROV_MEASURED, src);
	/* A backend that cannot query memory reports 0/0 -- that is "not
	 * reported", never "a 0-byte GPU". */
	if (d.memory_total == 0)
	{
		membrane_obs_u64_unknown(&s->vram_total_bytes, "not_reported_by_backend");
		membrane_obs_u64_unknown(&s->vram_free_bytes, "not_reported_by_backend");
		return ;
	}
	membrane_obs_u64_set(&s->vram_total_bytes, d.memory_total,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
	membrane_obs_u64_set(&s->vram_free_bytes, d.memory_free,
		MEMBRANE_OBS_PROV_MEASURED, "ggml_backend_dev_memory");
}

/* ------------------------------------------------------------------ */
/* Rendering -- runtime-agnostic                                       */
/* ------------------------------------------------------------------ */

static json	field_json(const membrane_obs_field_ref_t &r)
{
	json	j;

	if (!membrane_obs_field_known(&r))
		j["value"] = nullptr;
	else if (r.kind == MEMBRANE_OBS_KIND_U64)
		j["value"] = ((const membrane_obs_u64_t *)r.field)->value;
	else if (r.kind == MEMBRANE_OBS_KIND_BOOL)
		j["value"] = ((const membrane_obs_bool_t *)r.field)->value != 0;
	else
		j["value"] = std::string(((const membrane_obs_str_t *)r.field)->value);
	j["known"] = membrane_obs_field_known(&r) != 0;
	j["provenance"] = membrane_obs_provenance_name(
			membrane_obs_field_provenance(&r));
	j["source"] = membrane_obs_field_source(&r);
	return (j);
}

static json	u64_json(const membrane_obs_u64_t &f)
{
	membrane_obs_field_ref_t	r = {"", MEMBRANE_OBS_KIND_U64, &f};

	return (field_json(r));
}

static json	str_json(const membrane_obs_str_t &f)
{
	membrane_obs_field_ref_t	r = {"", MEMBRANE_OBS_KIND_STR, &f};

	return (field_json(r));
}

json	membrane_observe_snapshot_json(const membrane_observation_snapshot_t &s)
{
	membrane_obs_field_ref_t	refs[MEMBRANE_OBS_MAX_FIELDS];
	size_t						n = membrane_obs_snapshot_fields(&s, refs,
			MEMBRANE_OBS_MAX_FIELDS);
	json						j;
	json						unknown = json::array();
	json						sources = json::object();
	size_t						known = 0;

	j["schema_version"] = s.schema_version;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "observe";
	j["ok"] = s.status != MEMBRANE_OBS_STATUS_UNAVAILABLE;
	j["status"] = membrane_obs_status_name(s.status);
	j["timestamp"] = {
		{"utc", s.timestamp_utc},
		{"unix_ms", s.timestamp_unix_ms},
		{"clock", "wall_clock_at_collection_start"},
		{"collection_duration_ms", s.collection_duration_ms},
	};
	j["runtime"] = {{"id", s.runtime_id},
		{"observable", s.runtime_observable != 0}};
	/* Fixed section order (deterministic structure even when a section
	 * has no known field). */
	for (const char *sec : {"host_memory", "gpu", "model", "context", "kv",
			"service", "headroom"})
		j[sec] = json::object();
	for (size_t i = 0; i < n && i < MEMBRANE_OBS_MAX_FIELDS; ++i)
	{
		std::string	path = refs[i].path;
		size_t		dot = path.find('.');
		std::string	sec = path.substr(0, dot);
		std::string	key = path.substr(dot + 1);

		j[sec][key] = field_json(refs[i]);
		if (membrane_obs_field_known(&refs[i]))
		{
			known++;
			sources[membrane_obs_field_source(&refs[i])].push_back(path);
		}
		else
			unknown.push_back(path);
	}

	json	resident = json::array();

	for (size_t i = 0; i < s.resident_model_len; ++i)
	{
		const membrane_obs_resident_model_t	&r = s.resident_models[i];
		membrane_obs_field_ref_t	b = {"", MEMBRANE_OBS_KIND_U64,
			&r.gpu_layers};

		resident.push_back({
			{"name", str_json(r.name)},
			{"state", str_json(r.state)},
			{"backend", str_json(r.backend)},
			{"gpu_layers", field_json(b)},
			{"kv_precision", str_json(r.kv_precision)},
			{"estimated_model_bytes", u64_json(r.estimated_model_bytes)},
			{"estimated_kv_bytes", u64_json(r.estimated_kv_bytes)},
			/* Milestone I2 (Ollama GET /api/ps -- observation.h's own top
			 * comment on this struct). */
			{"digest", str_json(r.digest)},
			{"family", str_json(r.family)},
			{"quant", str_json(r.quant)},
			{"reported_size_bytes", u64_json(r.reported_size_bytes)},
			{"reported_gpu_bytes", u64_json(r.reported_gpu_bytes)},
			{"reported_context", u64_json(r.reported_context)},
			{"expires_at", str_json(r.expires_at)},
		});
	}
	j["model"]["resident_models"] = resident;
	j["fields"] = {{"known", known}, {"total", n},
		{"unknown", unknown}};
	j["sources"] = sources;
	return (j);
}

static std::string	fmt_bytes(uint64_t b)
{
	char	buf[64];

	if (b >= (1ull << 30))
		snprintf(buf, sizeof(buf), "%.2f GiB", (double)b / (double)(1ull << 30));
	else
		snprintf(buf, sizeof(buf), "%.1f MiB", (double)b / (double)(1ull << 20));
	return (buf);
}

static void	line_u64(const char *label, const membrane_obs_u64_t &f,
				bool bytes)
{
	if (!f.known)
	{
		printf("  %-22s unknown (%s)\n", label, f.source);
		return ;
	}
	std::string	v = bytes ? fmt_bytes(f.value)
		: std::to_string((unsigned long long)f.value);

	printf("  %-22s %-14s [%s]\n", label, v.c_str(),
		membrane_obs_provenance_name(f.provenance));
}

static void	line_str(const char *label, const membrane_obs_str_t &f)
{
	if (!f.known)
	{
		printf("  %-22s unknown (%s)\n", label, f.source);
		return ;
	}
	printf("  %-22s %-14s [%s]\n", label, f.value,
		membrane_obs_provenance_name(f.provenance));
}

static void	line_bool(const char *label, const membrane_obs_bool_t &f)
{
	if (!f.known)
	{
		printf("  %-22s unknown (%s)\n", label, f.source);
		return ;
	}
	printf("  %-22s %-14s [%s]\n", label, f.value ? "yes" : "no",
		membrane_obs_provenance_name(f.provenance));
}

void	membrane_observe_print_human(const membrane_observation_snapshot_t &s)
{
	size_t	known;
	size_t	total;

	membrane_obs_snapshot_count(&s, &known, &total);
	printf("Observation\n");
	printf("  %-22s %s\n", "Runtime:", s.runtime_id);
	printf("  %-22s %s (%zu/%zu fields known)\n", "Status:",
		membrane_obs_status_name(s.status), known, total);
	printf("  %-22s %s\n", "Timestamp:", s.timestamp_utc);
	if (!s.runtime_observable)
		return ;

	printf("\nHost memory\n");
	line_u64("Total:", s.ram_total_bytes, true);
	line_u64("Available:", s.ram_available_bytes, true);
	line_u64("Used:", s.ram_used_bytes, true);
	line_u64("Swap total:", s.swap_total_bytes, true);
	line_u64("Swap free:", s.swap_free_bytes, true);
	line_u64("Server RSS:", s.process_rss_bytes, true);

	printf("\nGPU (first GPU/iGPU enumerated -- the device `membrane plan` "
		"uses; device-wide, independent of the observed runtime)\n");
	line_u64("GPU devices:", s.gpu_device_count, false);
	line_str("Device:", s.gpu_device_description);
	line_str("Device id:", s.gpu_device_name);
	line_str("Backend:", s.gpu_backend);
	line_u64("VRAM total:", s.vram_total_bytes, true);
	line_u64("VRAM free:", s.vram_free_bytes, true);
	line_u64("VRAM used (device):", s.vram_used_bytes, true);

	printf("\nModel\n");
	line_str("Configured:", s.configured_model);
	line_bool("Registered:", s.configured_model_registered);
	line_str("Arch:", s.model_arch);
	line_str("Quant:", s.model_quant);
	line_u64("File size:", s.model_file_size_bytes, true);
	line_u64("Resident models:", s.resident_model_count, false);
	for (size_t i = 0; i < s.resident_model_len; ++i)
	{
		const membrane_obs_resident_model_t	&r = s.resident_models[i];

		printf("    - %s\n", r.name.known ? r.name.value : "?");
		if (r.backend.known || r.kv_precision.known
			|| r.estimated_kv_bytes.known)
			printf("        backend %s, kv %s%s%s "
				"[runtime_reported; bytes estimated]\n",
				r.backend.known ? r.backend.value : "?",
				r.kv_precision.known ? r.kv_precision.value : "?",
				r.estimated_kv_bytes.known ? ", est. KV " : "",
				r.estimated_kv_bytes.known
					? fmt_bytes(r.estimated_kv_bytes.value).c_str() : "");
		if (r.reported_size_bytes.known || r.reported_gpu_bytes.known
			|| r.reported_context.known || r.expires_at.known)
			printf("        size %s, gpu %s, context %s, expires %s "
				"[runtime_reported]\n",
				r.reported_size_bytes.known
					? fmt_bytes(r.reported_size_bytes.value).c_str()
					: "unknown",
				r.reported_gpu_bytes.known
					? fmt_bytes(r.reported_gpu_bytes.value).c_str()
					: "unknown",
				r.reported_context.known
					? std::to_string((unsigned long long)
						r.reported_context.value).c_str() : "unknown",
				r.expires_at.known ? r.expires_at.value : "unknown");
	}

	printf("\nContext\n");
	line_u64("Active:", s.context_active, false);
	line_u64("Planned:", s.context_planned, false);
	line_u64("Model max:", s.context_model_max, false);

	printf("\nKV\n");
	line_str("Planned precision:", s.kv_planned_precision);
	line_str("Planned placement:", s.kv_planned_placement);
	line_u64("Estimated footprint:", s.kv_estimated_bytes, true);
	line_u64("Measured usage:", s.kv_measured_bytes, true);

	printf("\nService\n");
	line_str("Manager:", s.service_manager);
	line_bool("Installed:", s.service_installed);
	line_bool("Active:", s.service_active);
	line_str("Endpoint:", s.server_endpoint);
	line_bool("Reachable:", s.server_reachable);
	line_str("Server version:", s.server_version);

	printf("\nHeadroom (raw, no reserve subtracted)\n");
	line_u64("RAM:", s.ram_headroom_bytes, true);
	line_u64("VRAM:", s.vram_headroom_bytes, true);
}
