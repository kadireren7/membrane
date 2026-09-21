#include "plan_cmd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <nlohmann/json.hpp>

#include <ggml.h>

#include "registry_core.h"
#include "model_catalog.h"
#include "variant_selector.h"
#include "gpu_device.h"
#include "runtime_session.h"
#include "kv_store_telemetry.h"
#include "membrane_plan.h"
#include "product_cli.h"

using json = nlohmann::json;

/* See plan_cmd.h's own top comment for the full read-only contract. */

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
		fprintf(stderr, "membrane plan: %s\n", message.c_str());
}

struct s_plan_opts
{
	std::string	name;
	bool		want_ctx = false;
	bool		ctx_auto = true;
	uint64_t	ctx_value = 0;
	bool		want_kv = false;
	int			precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	bool		want_gpu_layers = false;
	int32_t		gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	bool		want_quant = false;
	std::string	quant;
};

static bool	parse_plan_opts(const std::vector<std::string> &args,
				s_plan_opts *o, std::string *err)
{
	for (size_t i = 0; i < args.size(); ++i)
	{
		const std::string	&a = args[i];

		if (a == "--ctx" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_ctx = true;
			if (v == "auto")
				o->ctx_auto = true;
			else
			{
				char	*end = NULL;
				long long	n = strtoll(v.c_str(), &end, 10);

				if (end == NULL || *end != '\0' || n <= 0)
				{
					*err = "--ctx must be 'auto' or a positive integer";
					return (false);
				}
				o->ctx_auto = false;
				o->ctx_value = (uint64_t)n;
			}
			continue ;
		}
		if (a == "--kv" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_kv = true;
			if (v == "native")
				o->precision_request = MEMBRANE_JOINT_KV_NATIVE;
			else if (v == "q8")
				o->precision_request = MEMBRANE_JOINT_KV_Q8;
			else if (v == "q5")
				o->precision_request = MEMBRANE_JOINT_KV_Q5;
			else if (v == "adaptive")
				o->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
			else
			{
				*err = "--kv must be one of native|q8|q5|adaptive";
				return (false);
			}
			continue ;
		}
		if (a == "--gpu-layers" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_gpu_layers = true;
			if (v == "all")
				o->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL;
			else if (v == "auto")
				o->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
			else
			{
				char	*end = NULL;
				long long	n = strtoll(v.c_str(), &end, 10);

				if (end == NULL || *end != '\0' || n < 0)
				{
					*err = "--gpu-layers must be all|auto|N (N >= 0)";
					return (false);
				}
				o->gpu_layers_request = (int32_t)n;
			}
			continue ;
		}
		if ((a == "--quant" || a == "--variant") && i + 1 < args.size())
		{
			o->want_quant = true;
			o->quant = args[++i];
			continue ;
		}
		if (o->name.empty() && a.size() > 0 && a[0] != '-')
		{
			o->name = a;
			continue ;
		}
		*err = "unknown option '" + a + "'";
		return (false);
	}
	if (o->name.empty())
	{
		*err = "usage: membrane plan MODEL [--ctx auto|N] [--kv "
			"native|q8|q5|adaptive] [--gpu-layers all|auto|N] "
			"[--quant QUANT]";
		return (false);
	}
	return (true);
}

/* Same real ggml_row_size()-based per-mode KV byte formula every other
 * real caller in this project uses (context_recommender_dryrun.cpp's
 * own real_kv_bytes(), main.cpp's native_kv_bytes()/q8_kv_bytes()/
 * q5_kv_bytes()) -- never a second, independently-derived formula. */
static uint64_t	real_kv_bytes(const membrane_gpu_model_estimate_t &m,
					uint64_t ctx, enum ggml_type type)
{
	int64_t	n_embd_gqa = (m.n_head > 0)
			? (int64_t)(m.n_embd / m.n_head) * m.n_head_kv : 0;
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)m.n_layer;
	b.kv_size = ctx;
	b.bytes_per_token_k = ggml_row_size(type, n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(type, n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static int	find_gpu_device_index(const membrane_gpu_device_info_t *devices,
				size_t n_devices)
{
	for (size_t i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
			return ((int)i);
	return (-1);
}

/* Best-effort, exact-match-only quant identification for an already-
 * installed model -- the registry (registry_core.h) does NOT store
 * which catalog quant a file is (Section 13 of registry_core.h's own
 * top comment: only a cheap size/mtime identity signature), so this
 * never guesses from file size alone. A real match requires the
 * registry's own basename to equal exactly one catalog variant's real
 * filename for a family this model's name/registry resolves to (Part 7:
 * "do not invent claims existing code cannot calculate" -- no match
 * simply leaves the variant unknown). */
static bool	best_effort_installed_variant(const std::string &model_name,
				const std::string &basename, std::string *out_quant)
{
	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*fam
			= membrane_catalog_resolve(cat, model_name);

	if (fam == NULL)
		return (false);
	for (const auto &v : fam->variants)
		if (v.filename == basename)
		{
			*out_quant = v.quant;
			return (true);
		}
	return (false);
}

static void	print_json_reasons(json *j, const membrane_plan_t &plan)
{
	json	arr = json::array();

	for (size_t i = 0; i < plan.reason_count; ++i)
		arr.push_back({{"code", plan.reasons[i].code},
			{"detail", plan.reasons[i].detail}});
	(*j)["reasons"] = arr;
}

static const char	*kv_precision_label(int p)
{
	if (p == MEMBRANE_JOINT_KV_NATIVE)
		return ("native");
	if (p == MEMBRANE_JOINT_KV_Q8)
		return ("q8");
	if (p == MEMBRANE_JOINT_KV_Q5)
		return ("q5");
	return ("unknown");
}

static const char	*kv_placement_label(int m)
{
	return (membrane_kv_placement_mode_name(m));
}

static const char	*precision_request_label(int p)
{
	if (p == MEMBRANE_JOINT_PRECISION_REQUEST_AUTO)
		return ("adaptive");
	return (kv_precision_label(p));
}

static const char	*gpu_layers_request_label(int32_t v)
{
	if (v == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL)
		return ("all");
	if (v == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO)
		return ("auto");
	static char	buf[32];

	snprintf(buf, sizeof(buf), "%d", v);
	return (buf);
}

static void	render_json(const membrane_plan_t &plan)
{
	json	j;

	j["schema_version"] = MEMBRANE_PLAN_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "plan";
	j["ok"] = true;

	j["identity"] = {
		{"model_name", plan.identity.model_name},
		{"installed", plan.identity.installed != 0},
		{"model_path", plan.identity.installed ? plan.identity.model_path
			: ""},
		{"display_name", plan.identity.display_name},
		{"parameter_count", plan.identity.parameter_count},
		{"arch_known", plan.identity.arch_known != 0},
		{"arch_name", plan.identity.arch_known ? plan.identity.arch_name
			: ""},
		{"variant_known", plan.identity.variant_known != 0},
		{"variant", plan.identity.variant_known ? plan.identity.variant : ""},
		{"variant_source",
			membrane_plan_source_name(plan.identity.variant_source)},
		{"variant_estimate_only", plan.identity.variant_estimate_only != 0},
	};

	j["workload"] = {
		{"requested_context", plan.workload.requested_context},
		{"context_source",
			membrane_plan_source_name(plan.workload.context_source)},
		{"precision_request",
			precision_request_label(plan.workload.precision_request)},
		{"precision_source",
			membrane_plan_source_name(plan.workload.precision_source)},
		{"gpu_layers_request",
			gpu_layers_request_label(plan.workload.gpu_layers_request)},
		{"gpu_layers_source",
			membrane_plan_source_name(plan.workload.gpu_layers_source)},
		{"kv_placement_request",
			kv_placement_label(plan.workload.kv_placement_mode)},
		{"kv_placement_source",
			membrane_plan_source_name(plan.workload.kv_placement_source)},
	};

	j["hardware"] = {
		{"host_total_bytes", plan.hardware.host_total_bytes},
		{"host_available_bytes", plan.hardware.host_available_bytes},
		{"host_available_known", plan.hardware.host_available_known != 0},
		{"host_reserve_bytes", plan.hardware.host_reserve_bytes},
		{"device_known", plan.hardware.device_known != 0},
		{"backend", plan.hardware.device_known ? plan.hardware.backend : ""},
		{"device_name",
			plan.hardware.device_known ? plan.hardware.device_name : ""},
		{"device_total_bytes", plan.hardware.device_total_bytes},
		{"device_free_bytes", plan.hardware.device_free_bytes},
	};

	if (plan.decisions.has_decisions)
		j["decisions"] = {
			{"context", plan.decisions.context},
			{"context_source",
				membrane_plan_source_name(plan.decisions.context_source)},
			{"gpu_layers", plan.decisions.gpu_layers},
			{"gpu_layers_source",
				membrane_plan_source_name(plan.decisions.gpu_layers_source)},
			{"kv_precision", kv_precision_label(plan.decisions.kv_precision)},
			{"kv_precision_source",
				membrane_plan_source_name(plan.decisions.kv_precision_source)},
			{"kv_placement",
				kv_placement_label(plan.decisions.kv_placement)},
			{"kv_placement_source",
				membrane_plan_source_name(plan.decisions.kv_placement_source)},
		};
	else
		j["decisions"] = nullptr;

	j["feasibility"] = {
		{"feasible", plan.feasibility.feasible != 0},
		{"limiting_resource", plan.feasibility.limiting_resource},
		{"max_feasible_context_known",
			plan.feasibility.max_feasible_context_known != 0},
		{"max_feasible_context",
			plan.feasibility.max_feasible_context_known
				? plan.feasibility.max_feasible_context : 0},
		{"host_required_bytes", plan.feasibility.host_required_bytes},
		{"host_headroom_known", plan.feasibility.host_headroom_known != 0},
		{"host_headroom_bytes", plan.feasibility.host_headroom_known
			? plan.feasibility.host_headroom_bytes : 0},
	};

	print_json_reasons(&j, plan);
	j["explanation"] = plan.explanation;
	printf("%s\n", j.dump().c_str());
}

static std::string	fmt_mib(uint64_t bytes)
{
	char	buf[64];

	snprintf(buf, sizeof(buf), "%.1f MiB", (double)bytes / (1024.0 * 1024.0));
	return (std::string(buf));
}

static void	render_human(const membrane_plan_t &plan)
{
	printf("Model\n");
	printf("  %s\n", plan.identity.display_name);
	if (!plan.identity.installed)
		printf("  (not installed -- catalog metadata only)\n");
	if (plan.identity.variant_known)
		printf("  Variant: %s%s\n", plan.identity.variant,
			plan.identity.variant_estimate_only ? " (estimate only)" : "");
	else
		printf("  Variant: unknown\n");
	if (plan.identity.arch_known)
		printf("  Architecture: %s\n", plan.identity.arch_name);

	printf("\nHardware\n");
	if (plan.hardware.device_known)
		printf("  Backend: %s\n  Device: %s\n  VRAM total: %s\n"
			"  VRAM free: %s\n", plan.hardware.backend,
			plan.hardware.device_name,
			fmt_mib(plan.hardware.device_total_bytes).c_str(),
			fmt_mib(plan.hardware.device_free_bytes).c_str());
	else
		printf("  Backend: CPU-only (no GPU device detected)\n");
	if (plan.hardware.host_available_known)
		printf("  Host RAM total: %s\n  Host RAM available: %s\n",
			fmt_mib(plan.hardware.host_total_bytes).c_str(),
			fmt_mib(plan.hardware.host_available_bytes).c_str());
	else
		printf("  Host RAM: unavailable (could not read /proc/meminfo)\n");

	printf("\nWorkload\n");
	printf("  Context target: %s\n",
		plan.workload.requested_context > 0
			? std::to_string(plan.workload.requested_context).c_str()
			: "auto");

	printf("\nPlan\n");
	if (plan.decisions.has_decisions)
	{
		printf("  Context: %llu (%s)\n",
			(unsigned long long)plan.decisions.context,
			membrane_plan_source_name(plan.decisions.context_source));
		printf("  GPU layers: %d (%s)\n", plan.decisions.gpu_layers,
			membrane_plan_source_name(plan.decisions.gpu_layers_source));
		printf("  KV precision: %s (%s)\n",
			kv_precision_label(plan.decisions.kv_precision),
			membrane_plan_source_name(plan.decisions.kv_precision_source));
		printf("  KV placement: %s (%s)\n",
			kv_placement_label(plan.decisions.kv_placement),
			membrane_plan_source_name(plan.decisions.kv_placement_source));
	}
	else
		printf("  No decision -- see below.\n");
	printf("  Feasible: %s\n", plan.feasibility.feasible ? "yes" : "no");
	if (!plan.feasibility.feasible
		&& plan.feasibility.limiting_resource[0] != '\0')
		printf("  Limiting constraint: %s\n",
			plan.feasibility.limiting_resource);

	if (plan.feasibility.host_headroom_known)
	{
		printf("\nHeadroom\n");
		printf("  Host RAM: %s\n",
			fmt_mib(plan.feasibility.host_headroom_bytes).c_str());
	}

	if (plan.reason_count > 0)
	{
		printf("\nWhy\n");
		for (size_t i = 0; i < plan.reason_count; ++i)
			printf("  - [%s] %s\n", plan.reasons[i].code,
				plan.reasons[i].detail);
	}
	if (plan.explanation[0] != '\0')
		printf("\n%s\n", plan.explanation);
}

static int	plan_installed_model(const std::string &name,
				const membrane_registry_entry_t &entry,
				const s_plan_opts &opts, bool want_json)
{
	membrane_gpu_model_estimate_t	m;

	if (!membrane_gpu_estimate_model(entry.path.c_str(), &m))
	{
		print_err(want_json, "MODEL_FILE_UNREADABLE", "'" + entry.path
			+ "' (registered as '" + name + "') could not be read as a "
			"GGUF file -- it may have moved or been corrupted since it "
			"was registered");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	membrane_host_meminfo_t		meminfo;

	membrane_read_host_meminfo(&meminfo);

	membrane_gpu_device_info_t		devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t							n_devices
			= membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	int								gpu_index
			= find_gpu_device_index(devices, n_devices);

	membrane_ctxrec_request_t		req;

	memset(&req, 0, sizeof(req));
	req.n_layer_all = m.n_layer;
	req.bytes_per_layer = m.bytes_per_layer;
	req.output_role_bytes = m.output_role_bytes;
	req.arch_name = m.hparams_available ? m.arch_name : "";
	req.n_embd = m.n_embd;
	req.n_head = m.n_head;
	req.n_head_kv = m.n_head_kv;
	req.total_weight_bytes = m.total_bytes;
	req.host_total_bytes = meminfo.total_bytes;
	req.host_available_bytes = meminfo.available_bytes;
	req.host_available_known = meminfo.ok;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;

	membrane_plan_request_meta_t	meta;

	memset(&meta, 0, sizeof(meta));
	meta.kv_placement_source = MEMBRANE_PLAN_SOURCE_FALLBACK_DEFAULT;

	if (opts.want_kv)
	{
		req.precision_request = opts.precision_request;
		meta.precision_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else if (gpu_index < 0)
	{
		/* No real GPU device -- matches context_recommender_dryrun.cpp's
		 * own documented default (adaptive has no working CPU-only
		 * fallback in the existing, unchanged joint planner today):
		 * explicit native precision is the real, working CPU-only
		 * default when the user did not ask for anything specific. */
		req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
		meta.precision_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}
	else
	{
		req.precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
		meta.precision_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	if (opts.want_gpu_layers)
	{
		req.gpu_layers_request = opts.gpu_layers_request;
		meta.gpu_layers_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else
	{
		req.gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
		meta.gpu_layers_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	if (gpu_index >= 0)
	{
		req.device_free_bytes = devices[gpu_index].memory_free;
		req.device_total_bytes = devices[gpu_index].memory_total;
	}

	if (opts.want_ctx && !opts.ctx_auto)
	{
		/* Explicit single-candidate path (Part 6/8): exactly ONE
		 * candidate, still evaluated through the real, unchanged
		 * ctxrec pipeline (joint planner + host-memory guard), so an
		 * explicit --ctx gets the exact same feasibility rigor an
		 * auto-recommended one does -- never a second, parallel
		 * feasibility check. */
		req.model_max_context
			= (m.model_max_context_available
				&& opts.ctx_value <= m.model_max_context)
				? m.model_max_context : opts.ctx_value;
		req.model_max_context_known = 1;
		req.minimum_required_context = opts.ctx_value;
		req.candidates[0].ctx = opts.ctx_value;
		req.candidates[0].kv_bytes_native
			= real_kv_bytes(m, opts.ctx_value, GGML_TYPE_F16);
		req.candidates[0].kv_bytes_q8
			= real_kv_bytes(m, opts.ctx_value, GGML_TYPE_Q8_0);
		req.candidates[0].kv_bytes_q5
			= real_kv_bytes(m, opts.ctx_value, GGML_TYPE_Q5_1);
		req.candidate_count = 1;
		meta.requested_context = opts.ctx_value;
		meta.context_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else
	{
		req.model_max_context = m.model_max_context;
		req.model_max_context_known = m.model_max_context_available;
		uint64_t	ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
		size_t		n_ctxs = membrane_ctxrec_generate_candidates(
				req.model_max_context, 0, ctxs,
				MEMBRANE_CTXREC_MAX_CANDIDATES);

		for (size_t i = 0; i < n_ctxs; ++i)
		{
			req.candidates[i].ctx = ctxs[i];
			req.candidates[i].kv_bytes_native
				= real_kv_bytes(m, ctxs[i], GGML_TYPE_F16);
			req.candidates[i].kv_bytes_q8
				= real_kv_bytes(m, ctxs[i], GGML_TYPE_Q8_0);
			req.candidates[i].kv_bytes_q5
				= real_kv_bytes(m, ctxs[i], GGML_TYPE_Q5_1);
		}
		req.candidate_count = n_ctxs;
		meta.requested_context = 0;
		meta.context_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	membrane_ctxrec_result_t	rec;

	membrane_ctxrec_resolve(&req, &rec);

	membrane_plan_identity_input_t	identity;

	memset(&identity, 0, sizeof(identity));
	identity.model_name = name.c_str();
	identity.model_path = entry.path.c_str();
	identity.installed = 1;
	identity.arch_known = m.hparams_available;
	identity.arch_name = m.arch_name;
	identity.display_name = name.c_str();

	std::string	found_quant;
	membrane_plan_variant_input_t	variant;
	bool			have_variant
			= best_effort_installed_variant(name, entry.basename,
				&found_quant);

	memset(&variant, 0, sizeof(variant));
	if (have_variant)
	{
		variant.has_variant = 1;
		variant.quant = found_quant.c_str();
		variant.estimate_only = 0;
		variant.source = MEMBRANE_PLAN_SOURCE_MODEL_METADATA;
	}

	membrane_plan_hardware_input_t	hw;

	memset(&hw, 0, sizeof(hw));
	if (gpu_index >= 0)
	{
		hw.backend = devices[gpu_index].backend;
		hw.device_name = devices[gpu_index].name;
	}

	membrane_plan_t	plan;

	membrane_plan_assemble(&rec, &req, &meta, &identity,
		have_variant ? &variant : NULL, &hw, &plan);
	if (want_json)
		render_json(plan);
	else
		render_human(plan);
	return (MEMBRANE_EXIT_SUCCESS);
}

static void	push_no_variant_fits_note(membrane_plan_t *plan,
				const std::vector<membrane_variant_fit_t> &considered)
{
	if (plan->reason_count >= MEMBRANE_PLAN_MAX_REASONS)
		return ;

	membrane_plan_reason_t	*r = &plan->reasons[plan->reason_count];
	std::string				detail = "no catalog variant of this model is "
			"currently estimated to fit this host's available memory (" +
			std::to_string(considered.size()) + " considered)";

	snprintf(r->code, sizeof(r->code), "%s",
		MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT);
	snprintf(r->detail, sizeof(r->detail), "%s", detail.c_str());
	plan->reason_count++;
}

static int	plan_catalog_only_model(const membrane_catalog_family_t &fam,
				const s_plan_opts &opts, bool want_json)
{
	membrane_host_meminfo_t				meminfo;
	membrane_variant_selector_input_t	hw;

	membrane_read_host_meminfo(&meminfo);
	hw.host_total_bytes = meminfo.total_bytes;
	hw.host_available_bytes = meminfo.available_bytes;
	hw.host_available_known = meminfo.ok;

	std::vector<membrane_variant_fit_t>	considered;
	const membrane_catalog_variant_t		*selected = NULL;
	bool									explicit_variant
			= opts.want_quant;

	if (opts.want_quant)
	{
		selected = membrane_catalog_find_variant(fam, opts.quant);
		if (selected == NULL)
		{
			print_err(want_json, "NOT_FOUND", "'" + opts.quant + "' is not "
				"an available variant of '" + fam.name + "' -- see "
				"`membrane model info " + fam.name + "`");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		membrane_select_variant(fam, hw, &considered);
	}
	else
		selected = membrane_select_variant(fam, hw, &considered);

	membrane_plan_identity_input_t	identity;

	memset(&identity, 0, sizeof(identity));
	identity.model_name = fam.name.c_str();
	identity.installed = 0;
	identity.display_name = fam.display_name.c_str();
	identity.parameter_count = fam.parameter_count.c_str();
	identity.arch_known = !fam.arch.empty();
	identity.arch_name = fam.arch.c_str();

	membrane_plan_variant_input_t	variant;

	memset(&variant, 0, sizeof(variant));
	if (selected != NULL)
	{
		variant.has_variant = 1;
		variant.quant = selected->quant.c_str();
		variant.estimate_only = 1;
		variant.source = explicit_variant ? MEMBRANE_PLAN_SOURCE_EXPLICIT_USER
				: MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	membrane_plan_request_meta_t	meta;

	memset(&meta, 0, sizeof(meta));

	membrane_plan_t	plan;

	membrane_plan_assemble(NULL, NULL, &meta, &identity,
		selected != NULL ? &variant : NULL, NULL, &plan);
	/* No ctxrec pipeline ran (has_ctxrec_result == 0), so membrane_plan_
	 * assemble() never saw a membrane_ctxrec_request_t to read host
	 * facts from -- but this exact invocation DID already take one real
	 * host-memory snapshot above (meminfo, used for variant fit); reuse
	 * it here rather than reading /proc/meminfo a second time (Part 8:
	 * one snapshot per invocation). */
	plan.hardware.host_total_bytes = meminfo.total_bytes;
	plan.hardware.host_available_bytes = meminfo.available_bytes;
	plan.hardware.host_available_known = meminfo.ok;
	if (selected == NULL)
	{
		push_no_variant_fits_note(&plan, considered);
		plan.feasibility.feasible = 0;
		snprintf(plan.feasibility.limiting_resource,
			sizeof(plan.feasibility.limiting_resource), "%s",
			MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT);
	}
	if (want_json)
		render_json(plan);
	else
		render_human(plan);
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_plan_cmd_dispatch(const std::vector<std::string> &args,
				bool want_json)
{
	s_plan_opts	opts;
	std::string	err;

	if (!parse_plan_opts(args, &opts, &err))
	{
		print_err(want_json, "CLI_ERROR", err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	std::string					registry_path = membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;

	if (registry_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_DATA_HOME nor HOME "
			"is set");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		print_err(want_json, reg_err.code, reg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	/* Same resolution precedence as `membrane use` (Section 23 there):
	 * an already-registered name always wins outright; otherwise an
	 * exact catalog id/alias. Never mutates either (Part 4: `membrane
	 * plan` is READ-ONLY -- see plan_cmd.h's own top comment). */
	const membrane_registry_entry_t	*entry
			= membrane_registry_find(reg, opts.name);

	if (entry != NULL)
		return (plan_installed_model(opts.name, *entry, opts, want_json));

	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*fam
			= membrane_catalog_resolve(cat, opts.name);

	if (fam == NULL)
	{
		print_err(want_json, "NOT_FOUND", "'" + opts.name + "' is not an "
			"installed model or a known catalog model -- try `membrane "
			"model search " + opts.name + "`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	return (plan_catalog_only_model(*fam, opts, want_json));
}
