#include "observe_ollama.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>

#include "observe_shared.h"

/* See observe_ollama.h's own top comment for the full read-only contract. */

void	membrane_observe_ollama_inputs_init(
			membrane_observe_ollama_inputs_t *in)
{
	memset(&in->runtime, 0, sizeof(in->runtime));
	in->timestamp_unix_ms = 0;
	in->collection_duration_ms = 0;
	in->meminfo_probed = false;
	memset(&in->meminfo, 0, sizeof(in->meminfo));
	in->swap_supported = false;
	in->meminfo_source.clear();
	in->gpu_enumerated = false;
	in->devices.clear();
	in->ps_called = false;
	in->ps_ok = false;
	in->models.clear();
	in->ps_error = membrane_runtime_error_t();
}

/* ------------------------------------------------------------------ */
/* Stage 1: collection (the only code here that touches the network)   */
/* ------------------------------------------------------------------ */

static const char	*host_meminfo_source(void)
{
#if defined(_WIN32)
	return ("GlobalMemoryStatusEx");
#elif defined(__APPLE__)
	return ("mach_host_statistics64");
#else
	return ("proc_meminfo");
#endif
}

static bool	host_meminfo_reads_swap(void)
{
#if defined(_WIN32) || defined(__APPLE__)
	return (false);
#else
	return (true);
#endif
}

void	membrane_observe_ollama_collect_inputs(
			membrane_observe_ollama_inputs_t *in)
{
	auto	wall_start = std::chrono::system_clock::now();
	auto	mono_start = std::chrono::steady_clock::now();

	membrane_observe_ollama_inputs_init(in);
	in->timestamp_unix_ms = std::chrono::duration_cast<
		std::chrono::milliseconds>(wall_start.time_since_epoch()).count();

	/* membrane_ollama_describe() always fills *out (H2 contract) -- one
	 * bounded GET /api/version, the SAME discovery probe `membrane
	 * runtime inspect ollama` already performs. No new route here. */
	membrane_ollama_describe(&in->runtime);
	if (in->runtime.availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
	{
		/* Part 15: discovery did not conclusively succeed -- /api/ps is
		 * never requested, and neither is anything else. */
		in->collection_duration_ms = (uint64_t)std::chrono::duration_cast<
			std::chrono::milliseconds>(std::chrono::steady_clock::now()
				- mono_start).count();
		return ;
	}

	/* Host RAM/GPU are MEMBRANE's own device-wide facts (Part 8) --
	 * independent of Ollama, probed the same way regardless of which
	 * runtime is being observed. */
	membrane_read_host_meminfo(&in->meminfo);
	in->meminfo_probed = true;
	in->swap_supported = host_meminfo_reads_swap();
	in->meminfo_source = host_meminfo_source();

	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n = membrane_gpu_list_devices(devices,
			MEMBRANE_GPU_MAX_DEVICES);

	in->gpu_enumerated = true;
	in->devices.assign(devices, devices + n);

	/* The one new route I2 adds to the allowlist (runtime_ollama.h). */
	in->ps_called = true;
	in->ps_ok = membrane_ollama_list_running(&in->models, &in->ps_error);

	in->collection_duration_ms = (uint64_t)std::chrono::duration_cast<
		std::chrono::milliseconds>(std::chrono::steady_clock::now()
			- mono_start).count();
}

/* ------------------------------------------------------------------ */
/* Stage 2: pure snapshot assembly                                     */
/* ------------------------------------------------------------------ */

/* MEMBRANE-native concepts (default-model config, registry, GGUF header,
 * Planner v2) have no meaning for an externally-owned Ollama model --
 * honestly unknown, never guessed from a sibling field or from an
 * /api/tags//api/show call this command does not make (Part 5). */
static void	build_not_applicable(membrane_observation_snapshot_t *s)
{
	const char	*na = "not_applicable_external_runtime";

	membrane_obs_str_unknown(&s->configured_model, na);
	membrane_obs_bool_unknown(&s->configured_model_registered, na);
	membrane_obs_u64_unknown(&s->model_file_size_bytes, na);
	membrane_obs_str_unknown(&s->model_quant, na);
	membrane_obs_str_unknown(&s->model_arch, na);
	membrane_obs_u64_unknown(&s->context_planned, na);
	membrane_obs_u64_unknown(&s->context_model_max, na);
	membrane_obs_str_unknown(&s->kv_planned_precision, na);
	membrane_obs_str_unknown(&s->kv_planned_placement, na);
	membrane_obs_u64_unknown(&s->kv_estimated_bytes, na);
	/* No live KV instrumentation exists anywhere in this codebase yet
	 * (I1's own limitation, unchanged by I2). */
	membrane_obs_u64_unknown(&s->kv_measured_bytes, "not_instrumented");
	/* MEMBRANE never probes an OS service manager for an external runtime
	 * -- "binary installed" and "daemon reachable" are different facts,
	 * and only the second is ever checked (docs/runtime-ollama.md sec 2). */
	membrane_obs_str_unknown(&s->service_manager, na);
	membrane_obs_bool_unknown(&s->service_installed, na);
	membrane_obs_bool_unknown(&s->service_active, na);
}

static void	build_service(const membrane_observe_ollama_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	const char	*src = "http_get_api_version";

	if (in.runtime.endpoint_known)
		membrane_obs_str_set(&s->server_endpoint, in.runtime.endpoint,
			MEMBRANE_OBS_PROV_CONFIGURED, "ollama_endpoint");
	else
		membrane_obs_str_unknown(&s->server_endpoint,
			"invalid_endpoint_configuration");
	/* build_service() is only called once runtime_observable is true
	 * (membrane_observe_ollama_build_snapshot() returns early otherwise),
	 * so within this function Ollama was, by construction, just reached. */
	membrane_obs_bool_set(&s->server_reachable, 1, MEMBRANE_OBS_PROV_MEASURED,
		src);
	if (in.runtime.version_known)
		membrane_obs_str_set(&s->server_version, in.runtime.version,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, src);
	else
		membrane_obs_str_unknown(&s->server_version, "not_reported_by_runtime");
}

static void	build_resident_ollama(const membrane_ollama_process_model_t &m,
				membrane_obs_resident_model_t *r)
{
	const char	*rt = "api_ps";
	const char	*na = "not_reported_by_api_ps";

	membrane_obs_str_set(&r->name, m.name.c_str(),
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	/* Presence in GET /api/ps IS the runtime's own definition of
	 * loaded/resident (Part 10) -- nothing beyond that is inferred about
	 * process lifetime. */
	membrane_obs_str_set(&r->state, "loaded",
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	/* Native-only fields: /api/ps has no per-layer GPU-offload count and
	 * no KV-cache precision (OLLAMA_KV_CACHE_TYPE is server-start env,
	 * never returned by any API call -- docs/runtime-ollama.md sec 7). */
	membrane_obs_str_unknown(&r->backend, na);
	membrane_obs_u64_unknown(&r->gpu_layers, na);
	membrane_obs_str_unknown(&r->kv_precision, na);
	/* Ollama's own size figures are RUNTIME_REPORTED, never framed as an
	 * "estimate" the way the native server's load-time guess is (Part 7)
	 * -- see reported_size_bytes/reported_gpu_bytes below instead. */
	membrane_obs_u64_unknown(&r->estimated_model_bytes,
		"reported_as_reported_size_bytes");
	membrane_obs_u64_unknown(&r->estimated_kv_bytes, na);

	if (!m.digest.empty())
		membrane_obs_str_set(&r->digest, m.digest.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->digest, na);
	if (!m.family.empty())
		membrane_obs_str_set(&r->family, m.family.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->family, na);
	if (!m.quant.empty())
		membrane_obs_str_set(&r->quant, m.quant.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->quant, na);
	if (m.size_known)
		membrane_obs_u64_set(&r->reported_size_bytes, m.size_bytes,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_u64_unknown(&r->reported_size_bytes, na);
	/* THIS model's own GPU allocation -- model-specific, never the
	 * device-wide gpu.vram_* fields above (Part 8). */
	if (m.size_vram_known)
		membrane_obs_u64_set(&r->reported_gpu_bytes, m.size_vram_bytes,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_u64_unknown(&r->reported_gpu_bytes, na);
	if (m.context_length_known)
		membrane_obs_u64_set(&r->reported_context, m.context_length,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_u64_unknown(&r->reported_context, na);
	if (!m.expires_at.empty())
		membrane_obs_str_set(&r->expires_at, m.expires_at.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->expires_at, na);
}

static std::string	lower_copy(const std::string &s)
{
	std::string	out(s);

	std::transform(out.begin(), out.end(), out.begin(),
		[](unsigned char c) { return ((char)std::tolower(c)); });
	return (out);
}

static void	build_models(const membrane_observe_ollama_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	s->resident_model_len = 0;
	if (!in.ps_called)
	{
		/* Unreachable at all -- membrane_observe_ollama_build_snapshot()
		 * never gets here in that case, but this stays honest if that
		 * invariant ever changes. */
		membrane_obs_u64_unknown(&s->resident_model_count, "not_probed");
		membrane_obs_u64_unknown(&s->context_active, "not_probed");
		return ;
	}
	if (!in.ps_ok)
	{
		/* Part 16: a broken /api/ps degrades this section to unknown --
		 * it does NOT make the whole snapshot unavailable (Ollama itself
		 * was reached; only this one call failed). */
		std::string	reason = "ps_" + lower_copy(in.ps_error.code);

		membrane_obs_u64_unknown(&s->resident_model_count, reason.c_str());
		membrane_obs_u64_unknown(&s->context_active, reason.c_str());
		return ;
	}
	/* Part 14: zero loaded models is success, not an error -- a real,
	 * known 0, never "unknown". Part 6: the true count is reported even
	 * when more than MEMBRANE_OBS_MAX_RESIDENT models are loaded; only
	 * the per-model detail array is capped (documented limitation). */
	membrane_obs_u64_set(&s->resident_model_count, in.models.size(),
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "api_ps");
	for (size_t i = 0; i < in.models.size() && i < MEMBRANE_OBS_MAX_RESIDENT;
			++i)
	{
		build_resident_ollama(in.models[i], &s->resident_models[i]);
		s->resident_model_len++;
	}
	/* Part 9: a single active context is only meaningful for exactly one
	 * loaded model -- with zero or several, "the" active context is not
	 * one number, so it stays honestly unknown rather than picking one
	 * model arbitrarily (the same "don't silently choose one" rule as
	 * Part 6's resident-model handling). */
	if (in.models.empty())
		membrane_obs_u64_unknown(&s->context_active, "no_loaded_model");
	else if (in.models.size() > 1)
		membrane_obs_u64_unknown(&s->context_active,
			"ambiguous_multiple_loaded_models");
	else if (in.models[0].context_length_known)
		membrane_obs_u64_set(&s->context_active, in.models[0].context_length,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "api_ps");
	else
		membrane_obs_u64_unknown(&s->context_active, "not_reported_by_api_ps");
}

void	membrane_observe_ollama_build_snapshot(
			const membrane_observe_ollama_inputs_t &in,
			membrane_observation_snapshot_t *s)
{
	membrane_obs_snapshot_init(s, MEMBRANE_RUNTIME_ID_OLLAMA);
	s->timestamp_unix_ms = in.timestamp_unix_ms;
	if (!membrane_obs_format_utc(in.timestamp_unix_ms, s->timestamp_utc,
			sizeof(s->timestamp_utc)))
		s->timestamp_utc[0] = '\0';
	s->collection_duration_ms = in.collection_duration_ms;

	/* Unlike the native runtime's compiled-in STATIC_METADATA
	 * availability, Ollama's availability IS a live measurement this
	 * snapshot just took (one bounded GET /api/version). */
	membrane_obs_str_set(&s->runtime_availability,
		membrane_runtime_availability_name(in.runtime.availability),
		MEMBRANE_OBS_PROV_MEASURED, "http_get_api_version");
	s->runtime_observable = in.runtime.availability
		== MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	if (!s->runtime_observable)
	{
		membrane_obs_snapshot_finalize(s);
		return ;
	}
	membrane_observe_build_host_fields(in.meminfo_probed, in.meminfo,
		in.swap_supported, in.meminfo_source, s);
	membrane_observe_build_gpu_fields(in.gpu_enumerated, in.devices, s);
	build_not_applicable(s);
	build_service(in, s);
	build_models(in, s);
	membrane_obs_snapshot_finalize(s);
}
