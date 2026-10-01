#include "observe_cmd.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

#include "plan_cmd.h"
#include "plan_v2_resolver.h"
#include "product_cli.h"
#include "status_client.h"
#include "kv_residency_policy.h"

using json = nlohmann::json;

/* See observe_cmd.h's own top comment for the full read-only contract. */

void	membrane_observe_inputs_init(membrane_observe_inputs_t *in)
{
	*in = membrane_observe_inputs_t();
	in->timestamp_unix_ms = 0;
	in->collection_duration_ms = 0;
	in->runtime_described = false;
	memset(&in->runtime, 0, sizeof(in->runtime));
	in->meminfo_probed = false;
	memset(&in->meminfo, 0, sizeof(in->meminfo));
	in->swap_supported = false;
	in->gpu_enumerated = false;
	in->service_probed = false;
	in->service.manager_available = false;
	in->service.installed = false;
	in->service.active = false;
	in->config_loaded = false;
	in->config = membrane_server_config_defaults();
	in->server_probed = false;
	in->server_reachable = false;
	in->registry_loaded = false;
	in->entry_found = false;
	in->gguf_read = false;
	memset(&in->gguf, 0, sizeof(in->gguf));
	in->plan_resolved = false;
	in->plan_has_decisions = false;
	in->plan_context = 0;
	in->plan_kv_precision = 0;
	in->plan_kv_placement = 0;
	in->plan_kv_bytes_known = false;
	in->plan_kv_bytes = 0;
	in->plan_quant_known = false;
}

/* ------------------------------------------------------------------ */
/* Stage 1: collection (the only code here that touches the machine)   */
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

/* Planner v2's plan for the INSTALLED file itself -- never a sibling
 * variant it merely evaluated (those are hypothetical downloads). */
static void	collect_plan(const std::string &model_name,
				membrane_observe_inputs_t *in)
{
	std::unique_ptr<membrane_plan_v2_result_t>	res(
			new membrane_plan_v2_result_t());
	std::string	err_message;

	if (!membrane_plan_resolve_installed_v2(model_name, res.get(),
			&in->plan_error_code, &err_message))
		return ;
	in->plan_resolved = true;
	for (size_t i = 0; i < res->candidate_count; ++i)
	{
		const membrane_plan_t	&c = res->candidates[i];

		if (!c.identity.installed)
			continue ;
		if (c.identity.variant_known)
		{
			in->plan_quant_known = true;
			in->plan_quant = c.identity.variant;
		}
		if (!c.decisions.has_decisions)
			return ;
		in->plan_has_decisions = true;
		in->plan_context = c.decisions.context;
		in->plan_kv_precision = c.decisions.kv_precision;
		in->plan_kv_placement = c.decisions.kv_placement;

		/* The joint planner's own selected candidate is the one sized
		 * at the recommended context (context_recommender.c: recommended
		 * == hardware_fit, selected_plan = that candidate). Only trust
		 * its KV bytes while that stays true. */
		const membrane_ctxrec_result_t		&rec = c.ctxrec_result;
		const membrane_joint_plan_result_t	&jp = rec.selected_plan;

		if (c.has_ctxrec_result && rec.ok && jp.ok && jp.selected_index >= 0
			&& jp.selected_index < jp.candidate_count
			&& rec.recommended_context == rec.hardware_fit_context
			&& rec.recommended_context == c.decisions.context)
		{
			const membrane_joint_candidate_t	&jc
					= jp.candidates[jp.selected_index];

			in->plan_kv_bytes_known = true;
			in->plan_kv_bytes = jc.estimated_kv_gpu_bytes
				+ jc.estimated_host_kv_bytes;
		}
		return ;
	}
}

void	membrane_observe_collect_inputs(const std::string &runtime_id,
			membrane_observe_inputs_t *in)
{
	auto	wall_start = std::chrono::system_clock::now();
	auto	mono_start = std::chrono::steady_clock::now();

	membrane_observe_inputs_init(in);
	in->timestamp_unix_ms = std::chrono::duration_cast<
		std::chrono::milliseconds>(wall_start.time_since_epoch()).count();

	in->runtime_described = membrane_runtime_describe(runtime_id.c_str(),
			&in->runtime) != 0;
	if (!in->runtime_described)
		return ;

	membrane_read_host_meminfo(&in->meminfo);
	in->meminfo_probed = true;
	in->swap_supported = host_meminfo_reads_swap();
	in->meminfo_source = host_meminfo_source();

	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n = membrane_gpu_list_devices(devices,
			MEMBRANE_GPU_MAX_DEVICES);

	in->gpu_enumerated = true;
	in->devices.assign(devices, devices + n);

	in->service = membrane_probe_service();
	in->service_probed = true;

	membrane_server_config_error_t	cfg_err;
	std::string						cfg_path
			= membrane_server_config_resolve_path();

	in->config = membrane_server_config_defaults();
	in->config_loaded = !cfg_path.empty()
		&& membrane_server_config_load(cfg_path, &in->config, &cfg_err);
	if (!in->config_loaded)
		in->config = membrane_server_config_defaults();

	/* Same bounded, read-only GET /v1/status `membrane status` uses --
	 * probed even when no service unit is installed, because a
	 * foreground `membrane serve` is just as real. */
	if (in->config_loaded)
	{
		in->server_probed = true;
		in->server_reachable = membrane_fetch_server_status(
				in->config.listen_address, in->config.port,
				&in->server_status);
	}

	std::string					reg_path = membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;

	in->registry_loaded = !reg_path.empty()
		&& membrane_registry_load(reg_path, &reg, &reg_err);
	if (in->registry_loaded && in->config_loaded
		&& !in->config.default_model.empty())
	{
		const membrane_registry_entry_t	*e = membrane_registry_find(reg,
				in->config.default_model);

		if (e != NULL)
		{
			in->entry_found = true;
			in->entry = *e;
			in->gguf_read = membrane_gpu_estimate_model(e->path.c_str(),
					&in->gguf) != 0;
			if (in->gguf_read)
				collect_plan(in->config.default_model, in);
		}
	}

	in->collection_duration_ms = (uint64_t)std::chrono::duration_cast<
		std::chrono::milliseconds>(std::chrono::steady_clock::now()
			- mono_start).count();
}

/* ------------------------------------------------------------------ */
/* Stage 2: pure snapshot assembly                                     */
/* ------------------------------------------------------------------ */

static const char	*kv_precision_label(int p)
{
	if (p == MEMBRANE_JOINT_KV_NATIVE)
		return ("native");
	if (p == MEMBRANE_JOINT_KV_Q8)
		return ("q8");
	if (p == MEMBRANE_JOINT_KV_Q5)
		return ("q5");
	return (NULL);
}

static void	build_host(const membrane_observe_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	const char	*src = in.meminfo_source.c_str();

	if (!in.meminfo_probed || !in.meminfo.ok)
	{
		const char	*why = in.meminfo_probed ? "probe_failed" : "not_probed";

		membrane_obs_u64_unknown(&s->ram_total_bytes, why);
		membrane_obs_u64_unknown(&s->ram_available_bytes, why);
		membrane_obs_u64_unknown(&s->swap_total_bytes, why);
		membrane_obs_u64_unknown(&s->swap_free_bytes, why);
	}
	else
	{
		membrane_obs_u64_set(&s->ram_total_bytes, in.meminfo.total_bytes,
			MEMBRANE_OBS_PROV_MEASURED, src);
		membrane_obs_u64_set(&s->ram_available_bytes,
			in.meminfo.available_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
		if (in.swap_supported)
		{
			membrane_obs_u64_set(&s->swap_total_bytes,
				in.meminfo.swap_total_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
			membrane_obs_u64_set(&s->swap_free_bytes,
				in.meminfo.swap_free_bytes, MEMBRANE_OBS_PROV_MEASURED, src);
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
	 * own RSS would say nothing about the serving process. */
	membrane_obs_u64_unknown(&s->process_rss_bytes, "not_instrumented");
}

static void	build_gpu(const membrane_observe_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	const char	*src = "ggml_backend_dev";
	int			gpu_index = -1;
	uint64_t	n_gpu = 0;

	if (!in.gpu_enumerated)
	{
		membrane_obs_u64_unknown(&s->gpu_device_count, "not_probed");
		membrane_obs_str_unknown(&s->gpu_backend, "not_probed");
		membrane_obs_str_unknown(&s->gpu_device_name, "not_probed");
		membrane_obs_str_unknown(&s->gpu_device_description, "not_probed");
		membrane_obs_u64_unknown(&s->vram_total_bytes, "not_probed");
		membrane_obs_u64_unknown(&s->vram_free_bytes, "not_probed");
		return ;
	}
	for (size_t i = 0; i < in.devices.size(); ++i)
		if (in.devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| in.devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
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

	const membrane_gpu_device_info_t	&d = in.devices[(size_t)gpu_index];

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

static std::string	json_str(const json &j, const char *key)
{
	if (j.is_object() && j.contains(key) && j[key].is_string())
		return (j[key].get<std::string>());
	return (std::string());
}

static bool	json_u64(const json &j, const char *key, uint64_t *out)
{
	if (!j.is_object() || !j.contains(key)
		|| !j[key].is_number_integer() || j[key].get<int64_t>() < 0)
		return (false);
	*out = j[key].get<uint64_t>();
	return (true);
}

static void	build_resident(const json &m, membrane_obs_resident_model_t *r)
{
	const char	*rt = "server_status";
	const char	*est = "server_status:load_time_estimate";
	std::string	v;
	uint64_t	u;

	v = json_str(m, "model");
	if (!v.empty())
		membrane_obs_str_set(&r->name, v.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->name, "not_reported_by_runtime");
	v = json_str(m, "state");
	if (!v.empty())
		membrane_obs_str_set(&r->state, v.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->state, "not_reported_by_runtime");
	v = json_str(m, "backend");
	if (!v.empty())
		membrane_obs_str_set(&r->backend, v.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->backend, "not_reported_by_runtime");
	if (json_u64(m, "gpu_layers", &u))
		membrane_obs_u64_set(&r->gpu_layers, u,
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_u64_unknown(&r->gpu_layers, "not_reported_by_runtime");
	v = json_str(m, "kv_precision");
	if (!v.empty())
		membrane_obs_str_set(&r->kv_precision, v.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, rt);
	else
		membrane_obs_str_unknown(&r->kv_precision, "not_reported_by_runtime");
	/* The server's own figures are load-time ESTIMATES
	 * (runtime_session.cpp gs.estimated_*) -- reported by the runtime,
	 * but still estimates, so they are labeled as such. */
	if (json_u64(m, "estimated_model_bytes", &u) && u > 0)
		membrane_obs_u64_set(&r->estimated_model_bytes, u,
			MEMBRANE_OBS_PROV_ESTIMATED, est);
	else
		membrane_obs_u64_unknown(&r->estimated_model_bytes,
			"not_reported_by_runtime");
	if (json_u64(m, "estimated_kv_bytes", &u) && u > 0)
		membrane_obs_u64_set(&r->estimated_kv_bytes, u,
			MEMBRANE_OBS_PROV_ESTIMATED, est);
	else
		membrane_obs_u64_unknown(&r->estimated_kv_bytes,
			"not_reported_by_runtime");
}

static void	build_service(const membrane_observe_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	if (!in.service_probed)
	{
		membrane_obs_str_unknown(&s->service_manager, "not_probed");
		membrane_obs_bool_unknown(&s->service_installed, "not_probed");
		membrane_obs_bool_unknown(&s->service_active, "not_probed");
	}
	else
	{
		const char	*mgr = in.service.manager_name.empty() ? "service_manager"
			: in.service.manager_name.c_str();

		membrane_obs_str_set(&s->service_manager, mgr,
			MEMBRANE_OBS_PROV_MEASURED, "service_state");
		if (!in.service.manager_available)
		{
			membrane_obs_bool_unknown(&s->service_installed,
				"service_manager_unavailable");
			membrane_obs_bool_unknown(&s->service_active,
				"service_manager_unavailable");
		}
		else
		{
			membrane_obs_bool_set(&s->service_installed, in.service.installed,
				MEMBRANE_OBS_PROV_MEASURED, mgr);
			membrane_obs_bool_set(&s->service_active,
				in.service.installed && in.service.active,
				MEMBRANE_OBS_PROV_MEASURED, mgr);
		}
	}

	if (!in.config_loaded)
	{
		membrane_obs_str_unknown(&s->server_endpoint, "config_unavailable");
		membrane_obs_bool_unknown(&s->server_reachable, "config_unavailable");
		membrane_obs_str_unknown(&s->server_version, "config_unavailable");
		return ;
	}
	std::string	endpoint = "http://" + in.config.listen_address + ":"
		+ std::to_string(in.config.port);

	membrane_obs_str_set(&s->server_endpoint, endpoint.c_str(),
		MEMBRANE_OBS_PROV_CONFIGURED, "server_config");
	if (!in.server_probed)
	{
		membrane_obs_bool_unknown(&s->server_reachable, "not_probed");
		membrane_obs_str_unknown(&s->server_version, "not_probed");
		return ;
	}
	membrane_obs_bool_set(&s->server_reachable, in.server_reachable,
		MEMBRANE_OBS_PROV_MEASURED, "http_get_v1_status");
	std::string	ver = in.server_reachable
		? json_str(in.server_status, "version") : std::string();

	if (!ver.empty())
		membrane_obs_str_set(&s->server_version, ver.c_str(),
			MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "server_status");
	else
		membrane_obs_str_unknown(&s->server_version, in.server_reachable
			? "not_reported_by_runtime" : "server_unreachable");
}

static void	build_model(const membrane_observe_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	/* Configured intent. */
	bool	have_configured = in.config_loaded
		&& !in.config.default_model.empty();

	if (!in.config_loaded)
		membrane_obs_str_unknown(&s->configured_model, "config_unavailable");
	else if (!have_configured)
		membrane_obs_str_unknown(&s->configured_model, "none_configured");
	else
		membrane_obs_str_set(&s->configured_model,
			in.config.default_model.c_str(), MEMBRANE_OBS_PROV_CONFIGURED,
			"server_config.default_model");

	if (!have_configured)
		membrane_obs_bool_unknown(&s->configured_model_registered,
			"no_configured_model");
	else if (!in.registry_loaded)
		membrane_obs_bool_unknown(&s->configured_model_registered,
			"registry_unreadable");
	else
		membrane_obs_bool_set(&s->configured_model_registered,
			in.entry_found, MEMBRANE_OBS_PROV_STATIC_METADATA, "registry");

	if (in.entry_found)
		membrane_obs_u64_set(&s->model_file_size_bytes,
			in.entry.file_size_bytes, MEMBRANE_OBS_PROV_STATIC_METADATA,
			"registry");
	else
		membrane_obs_u64_unknown(&s->model_file_size_bytes,
			have_configured ? "not_registered" : "no_configured_model");

	if (in.gguf_read && in.gguf.hparams_available && in.gguf.arch_name[0])
		membrane_obs_str_set(&s->model_arch, in.gguf.arch_name,
			MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	else
		membrane_obs_str_unknown(&s->model_arch, in.entry_found
			? "gguf_unreadable" : "no_configured_model_file");

	/* The registry does not record quant; Planner v2 identifies it only
	 * by an exact catalog filename match (plan_cmd.cpp). */
	if (in.plan_quant_known)
		membrane_obs_str_set(&s->model_quant, in.plan_quant.c_str(),
			MEMBRANE_OBS_PROV_STATIC_METADATA, "catalog_filename_match");
	else
		membrane_obs_str_unknown(&s->model_quant, in.plan_resolved
			? "not_in_catalog" : "no_configured_model_file");

	/* What the running runtime says is resident. Unreachable server =
	 * unknown, never "nothing loaded". */
	s->resident_model_len = 0;
	if (!in.server_reachable)
	{
		membrane_obs_u64_unknown(&s->resident_model_count,
			"server_unreachable");
		return ;
	}
	if (!in.server_status.is_object()
		|| !in.server_status.contains("resident_models")
		|| !in.server_status["resident_models"].is_array())
	{
		membrane_obs_u64_unknown(&s->resident_model_count,
			"not_reported_by_runtime");
		return ;
	}
	const json	&arr = in.server_status["resident_models"];

	membrane_obs_u64_set(&s->resident_model_count, arr.size(),
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "server_status");
	for (size_t i = 0; i < arr.size() && i < MEMBRANE_OBS_MAX_RESIDENT; ++i)
	{
		build_resident(arr[i], &s->resident_models[i]);
		s->resident_model_len++;
	}
}

static void	build_context_kv(const membrane_observe_inputs_t &in,
				membrane_observation_snapshot_t *s)
{
	const char	*no_plan = !in.entry_found ? "no_configured_model_file"
		: !in.plan_resolved ? "planner_unavailable"
		: "planner_no_feasible_plan";

	/* The native server reports context_policy only, never a live n_ctx
	 * (server.cpp handle_status) -- so the active context is unknown. */
	membrane_obs_u64_unknown(&s->context_active, "not_reported_by_runtime");
	membrane_obs_u64_unknown(&s->kv_measured_bytes, "not_instrumented");

	if (in.gguf_read && in.gguf.model_max_context_available)
		membrane_obs_u64_set(&s->context_model_max, in.gguf.model_max_context,
			MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	else
		membrane_obs_u64_unknown(&s->context_model_max, in.gguf_read
			? "not_in_gguf" : "no_configured_model_file");

	if (!in.plan_has_decisions)
	{
		membrane_obs_u64_unknown(&s->context_planned, no_plan);
		membrane_obs_str_unknown(&s->kv_planned_precision, no_plan);
		membrane_obs_str_unknown(&s->kv_planned_placement, no_plan);
		membrane_obs_u64_unknown(&s->kv_estimated_bytes, no_plan);
		return ;
	}
	membrane_obs_u64_set(&s->context_planned, in.plan_context,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s->kv_planned_precision,
		kv_precision_label(in.plan_kv_precision),
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s->kv_planned_placement,
		membrane_kv_placement_mode_name(in.plan_kv_placement),
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	if (in.plan_kv_bytes_known)
		membrane_obs_u64_set(&s->kv_estimated_bytes, in.plan_kv_bytes,
			MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	else
		membrane_obs_u64_unknown(&s->kv_estimated_bytes,
			"planner_kv_bytes_unavailable");
}

void	membrane_observe_build_snapshot(const std::string &runtime_id,
			const membrane_observe_inputs_t &in,
			membrane_observation_snapshot_t *s)
{
	membrane_obs_snapshot_init(s, runtime_id.c_str());
	s->timestamp_unix_ms = in.timestamp_unix_ms;
	if (!membrane_obs_format_utc(in.timestamp_unix_ms, s->timestamp_utc,
			sizeof(s->timestamp_utc)))
		s->timestamp_utc[0] = '\0';
	s->collection_duration_ms = in.collection_duration_ms;

	if (!in.runtime_described)
	{
		membrane_obs_str_unknown(&s->runtime_availability, "unknown_runtime");
		membrane_obs_snapshot_finalize(s);
		return ;
	}
	membrane_obs_str_set(&s->runtime_availability,
		membrane_runtime_availability_name(in.runtime.availability),
		MEMBRANE_OBS_PROV_STATIC_METADATA, "runtime_capabilities");
	s->runtime_observable = in.runtime.availability
		== MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE;
	if (!s->runtime_observable)
	{
		membrane_obs_snapshot_finalize(s);
		return ;
	}
	build_host(in, s);
	build_gpu(in, s);
	build_service(in, s);
	build_model(in, s);
	build_context_kv(in, s);
	membrane_obs_snapshot_finalize(s);
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
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
		"uses)\n");
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

		printf("    - %s: backend %s, kv %s", r.name.known ? r.name.value
			: "?", r.backend.known ? r.backend.value : "?",
			r.kv_precision.known ? r.kv_precision.value : "?");
		if (r.estimated_kv_bytes.known)
			printf(", est. KV %s", fmt_bytes(r.estimated_kv_bytes.value)
				.c_str());
		printf(" [runtime_reported; bytes estimated]\n");
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

/* ------------------------------------------------------------------ */
/* Dispatch                                                             */
/* ------------------------------------------------------------------ */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", code}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane observe: %s\n", message.c_str());
}

int	membrane_observe_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	std::string	runtime_id = MEMBRANE_RUNTIME_ID_NATIVE;

	for (size_t i = 0; i < args.size(); ++i)
	{
		if (args[i] == "--runtime" && i + 1 < args.size())
		{
			runtime_id = args[++i];
			continue ;
		}
		print_err(want_json, "CLI_ERROR", "unknown option '" + args[i]
			+ "' -- usage: membrane observe [--runtime membrane-native] "
			"[--json]");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (runtime_id != MEMBRANE_RUNTIME_ID_NATIVE)
	{
		membrane_runtime_descriptor_t	d;
		bool	known = membrane_runtime_describe(runtime_id.c_str(), &d)
			|| runtime_id == MEMBRANE_RUNTIME_ID_OLLAMA
			|| runtime_id == MEMBRANE_RUNTIME_ID_VLLM;

		print_err(want_json, "CLI_ERROR", known
			? "observation of runtime '" + runtime_id + "' is not "
				"implemented yet -- only membrane-native can be observed"
			: "unknown runtime '" + runtime_id + "' -- see `membrane "
				"runtime list`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	membrane_observe_inputs_t		in;
	std::unique_ptr<membrane_observation_snapshot_t>	s(
			new membrane_observation_snapshot_t());

	membrane_observe_collect_inputs(runtime_id, &in);
	membrane_observe_build_snapshot(runtime_id, in, s.get());
	if (want_json)
		printf("%s\n", membrane_observe_snapshot_json(*s).dump().c_str());
	else
		membrane_observe_print_human(*s);
	return (s->status == MEMBRANE_OBS_STATUS_UNAVAILABLE
		? MEMBRANE_EXIT_RUNTIME_ERROR : MEMBRANE_EXIT_SUCCESS);
}
