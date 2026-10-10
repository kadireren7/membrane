#include "orchestrate_cmd.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>

#include <nlohmann/json.hpp>

#include "cli_shared.h"
#include "gpu_device.h"
#include "plan_cmd.h"
#include "plan_v2_resolver.h"
#include "product_cli.h"
#include "registry_core.h"
#include "runtime_adapter.h"
#include "runtime_capabilities.h"
#include "runtime_session.h"
#include "server_config.h"
#include "status_client.h"
#include "ui.h"
#include "use_cmd.h"

using json = nlohmann::json;

/* See orchestrate_cmd.h's top comment. */

static void	copy(char *dst, size_t n, const std::string &s)
{
	if (n == 0)
		return ;
	snprintf(dst, n, "%s", s.c_str());
}

static std::string	lower(const std::string &s)
{
	std::string	r;

	for (char c : s)
		r.push_back((char)std::tolower((unsigned char)c));
	return (r);
}

std::string	membrane_orch_normalize_model(const std::string &name)
{
	std::string	s = lower(name);
	std::string	r;

	if (s.size() > 7 && s.compare(s.size() - 7, 7, ":latest") == 0)
		s.erase(s.size() - 7);
	for (char c : s)
		if (c != '-' && c != '_' && c != ':' && c != '/' && c != ' ')
			r.push_back(c);
	return (r);
}

static bool	same_model(const std::string &query, const std::string &candidate)
{
	return (lower(query) == lower(candidate)
		|| membrane_orch_normalize_model(query)
			== membrane_orch_normalize_model(candidate));
}

/* ------------------------------------------------------------------ */
/* topology                                                             */
/* ------------------------------------------------------------------ */

void	membrane_orch_build_topology(membrane_topology_t *t)
{
	membrane_host_meminfo_t			mem;
	membrane_gpu_device_info_t		devs[MEMBRANE_GPU_MAX_DEVICES];
	size_t							n;
	size_t							i;
	int								gpu_idx = 0;

	memset(t, 0, sizeof(*t));
	membrane_read_host_meminfo(&mem);
	t->ram_known = mem.ok ? 1 : 0;
	t->ram_total_bytes = mem.total_bytes;
	t->ram_available_bytes = mem.available_bytes;
	n = membrane_gpu_list_devices(devs, MEMBRANE_GPU_MAX_DEVICES);
	for (i = 0; i < n && t->device_count < MEMBRANE_ORCH_MAX_DEVICES; ++i)
	{
		membrane_orch_device_t	*d;

		if (devs[i].type != MEMBRANE_DEV_TYPE_GPU
			&& devs[i].type != MEMBRANE_DEV_TYPE_IGPU)
			continue ;			/* the CPU is the host RAM tier, not a GPU */
		d = &t->devices[t->device_count++];
		snprintf(d->id, sizeof(d->id), "gpu%d", gpu_idx++);
		copy(d->backend, sizeof(d->backend), devs[i].backend);
		copy(d->name, sizeof(d->name), devs[i].description[0]
			? std::string(devs[i].description) : std::string(devs[i].name));
		d->kind = devs[i].type == MEMBRANE_DEV_TYPE_IGPU
			? MEMBRANE_DEV_KIND_UMA : MEMBRANE_DEV_KIND_DISCRETE;
		d->total_bytes = devs[i].memory_total;
		d->free_known = devs[i].memory_total > 0;
		d->free_bytes = devs[i].memory_free;
	}
}

/* ------------------------------------------------------------------ */
/* gathering                                                            */
/* ------------------------------------------------------------------ */

static void	base_input(membrane_orch_runtime_input_t *in,
				const membrane_runtime_descriptor_t &d)
{
	memset(in, 0, sizeof(*in));
	copy(in->runtime_id, sizeof(in->runtime_id), d.id);
	copy(in->display_name, sizeof(in->display_name), d.display_name);
	in->availability = d.availability;
	in->health = d.health;
	copy(in->unavailable_reason, sizeof(in->unavailable_reason),
		d.unavailable_reason);
	in->caps = d.capabilities;
	in->plan_gpu_layers = -1;
}

static void	gather_native(const std::string &model,
				membrane_orch_gather_t *g)
{
	membrane_runtime_descriptor_t	d;
	membrane_orch_runtime_input_t	in;
	std::string						path = membrane_registry_resolve_path();
	membrane_registry_t				reg;
	membrane_registry_error_t		rerr;

	membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE, &d);
	base_input(&in, d);
	if (!path.empty() && membrane_registry_load(path, &reg, &rerr))
	{
		const membrane_registry_entry_t	*entry = NULL;

		for (const auto &e : reg.entries)
			if (same_model(model, e.name))
			{
				entry = &e;
				break ;
			}
		if (entry != NULL)
		{
			std::unique_ptr<membrane_plan_v2_result_t>	res(
					new membrane_plan_v2_result_t());
			std::string									ec;
			std::string									em;
			membrane_server_config_t					cfg
				= membrane_server_config_defaults();
			membrane_server_config_error_t				cerr;
			std::string									cpath
				= membrane_server_config_resolve_path();

			in.model_present = 1;
			copy(in.model_id, sizeof(in.model_id), entry->name);
			in.size_known = entry->file_size_bytes > 0;
			in.size_bytes = entry->file_size_bytes;
			in.context_max_known = entry->model_max_context > 0;
			in.context_max = entry->model_max_context;
			if (membrane_plan_resolve_installed_v2(entry->name, res.get(),
					&ec, &em) && res->has_selected)
			{
				const membrane_plan_t	&p = res->candidates[res->selected_index];

				in.plan_valid = 1;
				in.plan_feasible = res->feasible;
				if (p.decisions.has_decisions)
				{
					static const char	*kvp[3] = {"native", "q8", "q5"};

					in.plan_context = p.decisions.context;
					in.plan_gpu_layers = p.decisions.gpu_layers;
					copy(in.plan_kv_precision, sizeof(in.plan_kv_precision),
						p.decisions.kv_precision >= 0
						&& p.decisions.kv_precision <= 2
						? kvp[p.decisions.kv_precision] : "");
					copy(in.plan_kv_placement, sizeof(in.plan_kv_placement),
						membrane_kv_placement_mode_name(
							p.decisions.kv_placement));
				}
				if (p.feasibility.host_required_bytes > 0)
				{
					in.plan_mem_known = 1;
					in.plan_host_required_bytes
						= p.feasibility.host_required_bytes;
				}
			}
			if (!cpath.empty())
				membrane_server_config_load(cpath, &cfg, &cerr);
			json	status;

			if (membrane_fetch_server_status(cfg.listen_address, cfg.port,
					&status))
			{
				in.resident_known = 1;
				if (status.contains("resident_models")
					&& status["resident_models"].is_array())
					for (const auto &m : status["resident_models"])
						if (m.value("model", std::string()) == entry->name)
							in.resident = 1;
			}
		}
	}
	g->inputs.push_back(in);
	g->endpoints.push_back("");
}

static void	gather_external(const membrane_runtime_adapter_t *a,
				const std::string &model, membrane_orch_gather_t *g)
{
	membrane_runtime_descriptor_t	d;
	membrane_orch_runtime_input_t	in;

	a->describe(&d);
	base_input(&in, d);
	g->endpoints.push_back(d.endpoint_known ? d.endpoint : "");
	if (d.availability == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
		&& a->list_models != NULL)
	{
		std::vector<membrane_external_model_t>	models;
		membrane_runtime_error_t				err;

		if (a->list_models(&models, &err))
			for (const auto &m : models)
				if (same_model(model, m.runtime_model_id)
					|| same_model(model, m.display_name))
				{
					in.model_present = 1;
					copy(in.model_id, sizeof(in.model_id), m.runtime_model_id);
					in.size_known = m.size_known;
					in.size_bytes = m.size_bytes;
					in.context_max_known = m.context_length_known;
					in.context_max = m.context_length;
					break ;
				}
		if (in.model_present && a->resident_models != NULL)
		{
			std::vector<std::string>	res;

			if (a->resident_models(&res, &err))
			{
				in.resident_known = 1;
				for (const auto &r : res)
					if (r == in.model_id)
						in.resident = 1;
			}
		}
	}
	g->inputs.push_back(in);
}

bool	membrane_orch_gather(const std::string &model,
			const std::string &runtime_filter, membrane_orch_gather_t *out,
			std::string *err)
{
	bool	all = runtime_filter.empty() || runtime_filter == "auto";
	bool	matched = false;

	*out = membrane_orch_gather_t();
	membrane_orch_build_topology(&out->topology);
	for (size_t i = 0; i < membrane_runtime_registry_count(); ++i)
	{
		const membrane_runtime_adapter_t	*a = membrane_runtime_registry_at(i);

		if (!all && runtime_filter != a->id)
			continue ;
		matched = true;
		if (std::string(a->id) == MEMBRANE_RUNTIME_ID_NATIVE)
			gather_native(model, out);
		else
			gather_external(a, model, out);
	}
	if (!matched)
	{
		*err = "unknown runtime '" + runtime_filter + "' -- see `membrane "
			"runtime list`";
		return (false);
	}
	return (true);
}

/* ------------------------------------------------------------------ */
/* rendering                                                            */
/* ------------------------------------------------------------------ */

static std::string	gib(uint64_t b)
{
	char	buf[32];

	snprintf(buf, sizeof(buf), "%.1f GiB", (double)b / (1024.0 * 1024 * 1024));
	return (buf);
}

static std::string	reason_text(const std::string &code)
{
	static const struct { const char *code; const char *text; }	tbl[] = {
		{MEMBRANE_ORCH_REASON_ONLY_CANDIDATE, "it is the only runtime that "
			"can serve this model"},
		{MEMBRANE_ORCH_REASON_PLANNED, "Planner v2 found a plan that fits "
			"this machine"},
		{MEMBRANE_ORCH_REASON_RESIDENT, "the model is already loaded there"},
		{MEMBRANE_ORCH_REASON_STRONGER_EVIDENCE, "stronger evidence that it "
			"fits"},
		{MEMBRANE_ORCH_REASON_OBSERVATION, "it exposes more of its state for "
			"observation"},
		{MEMBRANE_ORCH_REASON_SMALLEST_MEMORY, "smallest known memory "
			"footprint"},
		{MEMBRANE_ORCH_REASON_MORE_CONTROL, "more memory-reducing controls"},
		{MEMBRANE_ORCH_REASON_LARGEST_CONTEXT, "largest known maximum "
			"context"},
		{MEMBRANE_ORCH_REASON_TIEBREAK, "the candidates are tied; a stable "
			"name order decided"},
		{MEMBRANE_ORCH_REASON_UNKNOWN_FIT, "for at least one candidate it is "
			"not known whether the model fits"},
		{MEMBRANE_ORCH_REASON_NO_CANDIDATE, "no runtime can serve it right "
			"now"}};

	for (const auto &t : tbl)
		if (code == t.code)
			return (t.text);
	return ("");
}

static json	decision_json(const std::string &model,
				const membrane_orch_gather_t &g,
				const membrane_orch_decision_t &d)
{
	json	j;
	json	cands = json::array();
	json	reasons = json::array();
	json	devs = json::array();

	for (size_t i = 0; i < d.candidate_count; ++i)
	{
		const membrane_orch_candidate_t	&c = d.candidates[i];
		json							lim = json::array();
		json							k;

		for (size_t l = 0; l < c.limitation_count; ++l)
			lim.push_back(c.limitations[l]);
		k["runtime"] = c.runtime_id;
		k["model"] = c.model_id[0] ? json(c.model_id) : json(nullptr);
		k["viable"] = c.viable != 0;
		k["exclusion"] = c.viable ? json(nullptr)
			: json({{"code", c.exclusion_code}, {"text", c.exclusion_text}});
		k["evidence"] = c.viable ? json(membrane_orch_evidence_name(
				c.evidence)) : json(nullptr);
		k["evidence_text"] = c.evidence_text;
		k["resident"] = c.resident != 0;
		k["configuration"] = {{"context", c.context_known ? json(c.context)
				: json(nullptr)}, {"gpu_layers", c.gpu_layers >= 0
				? json(c.gpu_layers) : json(nullptr)},
			{"kv_precision", c.kv_precision[0] ? json(c.kv_precision)
				: json(nullptr)}, {"kv_placement", c.kv_placement[0]
				? json(c.kv_placement) : json(nullptr)}};
		k["memory"] = c.mem_known ? json({{"bytes", c.mem_bytes},
			{"provenance", c.mem_provenance}}) : json(nullptr);
		k["context_max"] = c.context_max_known ? json(c.context_max)
			: json(nullptr);
		k["control_count"] = c.control_count;
		k["observation_quality"] = c.observation_quality;
		k["limitations"] = lim;
		k["endpoint"] = i < g.endpoints.size() && !g.endpoints[i].empty()
			? json(g.endpoints[i]) : json(nullptr);
		cands.push_back(k);
	}
	for (size_t i = 0; i < d.reason_count; ++i)
		reasons.push_back(d.reasons[i]);
	for (size_t i = 0; i < g.topology.device_count; ++i)
	{
		const membrane_orch_device_t	&x = g.topology.devices[i];

		devs.push_back({{"id", x.id}, {"backend", x.backend}, {"name", x.name},
			{"kind", membrane_orch_device_kind_name(x.kind)},
			{"total_bytes", x.total_bytes}, {"free_bytes", x.free_known
				? json(x.free_bytes) : json(nullptr)}});
	}
	j["schema_version"] = MEMBRANE_ORCH_SCHEMA_VERSION;
	j["mode"] = "run";
	j["model"] = model;
	j["intent"] = membrane_orch_intent_name(d.intent);
	j["selected"] = d.has_selection ? json(d.candidates[d.selected].runtime_id)
		: json(nullptr);
	j["confidence"] = d.has_selection ? json(d.confidence) : json(nullptr);
	j["confidence_text"] = d.confidence_text;
	j["reasons"] = reasons;
	j["summary"] = d.summary;
	j["candidates"] = cands;
	j["ranking"] = json::array();
	for (size_t i = 0; i < d.order_count; ++i)
		j["ranking"].push_back(d.candidates[d.order[i]].runtime_id);
	j["hardware"] = {{"ram_total_bytes", g.topology.ram_known
			? json(g.topology.ram_total_bytes) : json(nullptr)},
		{"ram_available_bytes", g.topology.ram_known
			? json(g.topology.ram_available_bytes) : json(nullptr)},
		{"gpus", devs}};
	return (j);
}

static void	print_human(const std::string &model,
				const membrane_orch_gather_t &g,
				const membrane_orch_decision_t &d)
{
	printf("Runtime choice for \"%s\" (intent: %s)\n\n", model.c_str(),
		membrane_orch_intent_name(d.intent));
	for (size_t i = 0; i < d.candidate_count; ++i)
	{
		const membrane_orch_candidate_t	&c = d.candidates[i];
		bool							sel = d.has_selection
			&& (int)i == d.selected;

		if (c.viable)
			printf("  %s %-16s %s\n", sel ? membrane_ui_sym_ok().c_str()
				: membrane_ui_sym_dot().c_str(), c.runtime_id,
				sel ? membrane_ui_paint(MEMBRANE_UI_OK, "selected").c_str()
					: "also possible");
		else
			printf("  %s %-16s %s\n", membrane_ui_sym_dot().c_str(),
				c.runtime_id, membrane_ui_paint(MEMBRANE_UI_DIM,
					std::string("not used -- ") + c.exclusion_text).c_str());
		if (c.viable)
		{
			printf("      %s [%s]\n", c.evidence_text,
				membrane_orch_evidence_name(c.evidence));
			if (c.mem_known)
				printf("      memory: %s (%s)\n", gib(c.mem_bytes).c_str(),
					c.mem_provenance);
		}
	}
	printf("\n");
	if (!d.has_selection)
	{
		printf("%s\n", d.summary);
		return ;
	}
	printf("Why:\n");
	for (size_t i = 0; i < d.reason_count; ++i)
	{
		std::string	t = reason_text(d.reasons[i]);

		if (!t.empty())
			printf("  - %s\n", t.c_str());
	}
	printf("Confidence: %s -- %s\n", d.confidence, d.confidence_text);
	const membrane_orch_candidate_t	&s = d.candidates[d.selected];

	if (s.limitation_count > 0)
	{
		printf("\nWhat MEMBRANE cannot control on %s:\n", s.runtime_id);
		for (size_t i = 0; i < s.limitation_count; ++i)
			printf("  - %s\n", s.limitations[i]);
	}
	printf("\nHardware: RAM %s (%s available)", g.topology.ram_known
		? gib(g.topology.ram_total_bytes).c_str() : "unknown",
		g.topology.ram_known ? gib(g.topology.ram_available_bytes).c_str()
			: "unknown");
	if (g.topology.device_count == 0)
		printf(", no GPU\n");
	else
	{
		printf("\n");
		for (size_t i = 0; i < g.topology.device_count; ++i)
		{
			const membrane_orch_device_t	&x = g.topology.devices[i];

			printf("  %s %s (%s, %s): %s%s\n", x.id, x.name, x.backend,
				membrane_orch_device_kind_name(x.kind),
				gib(x.total_bytes).c_str(), x.free_known
				? (" total, " + gib(x.free_bytes) + " free").c_str() : "");
		}
	}
}

/* ------------------------------------------------------------------ */
/* dispatch                                                             */
/* ------------------------------------------------------------------ */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
		printf("%s\n", json{{"ok", false}, {"error", {{"code", code},
			{"message", message}}}}.dump().c_str());
	else
		fprintf(stderr, "membrane run: %s\n", message.c_str());
}

int	membrane_run_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json)
{
	std::string				model;
	std::string				runtime = "auto";
	bool					explain = false;
	membrane_orch_intent_t	intent = MEMBRANE_INTENT_GENERAL;

	for (size_t i = 0; i < args.size(); ++i)
	{
		const std::string	&a = args[i];

		if (a == "--runtime" && i + 1 < args.size())
			runtime = args[++i];
		else if (a == "--intent" && i + 1 < args.size())
		{
			if (!membrane_orch_intent_parse(args[++i].c_str(), &intent))
			{
				print_err(want_json, "CLI_ERROR", "--intent must be general|"
					"interactive|memory_saving|long_context");
				return (MEMBRANE_EXIT_CLI_ERROR);
			}
		}
		else if (a == "--explain" || a == "--dry-run")
			explain = true;
		else if (model.empty() && !a.empty() && a[0] != '-')
			model = a;
		else
		{
			print_err(want_json, "CLI_ERROR", "unknown option '" + a + "' -- "
				"usage: membrane run MODEL [--runtime auto|membrane-native|"
				"ollama|vllm] [--intent general|interactive|memory_saving|"
				"long_context] [--explain] [--json]");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
	}
	if (model.empty())
	{
		print_err(want_json, "CLI_ERROR", "usage: membrane run MODEL "
			"[--runtime auto|...] [--intent ...] [--explain]");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::unique_ptr<membrane_orch_gather_t>	g(new membrane_orch_gather_t());
	std::unique_ptr<membrane_orch_decision_t>	d(
			new membrane_orch_decision_t());
	std::string								err;

	if (!membrane_orch_gather(model, runtime, g.get(), &err))
	{
		print_err(want_json, "CLI_ERROR", err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	membrane_orchestrate(intent, &g->topology, g->inputs.data(),
		g->inputs.size(), d.get());
	if (want_json)
		printf("%s\n", decision_json(model, *g, *d).dump().c_str());
	else
		print_human(model, *g, *d);
	if (!d->has_selection)
	{
		bool	absent_everywhere = true;

		for (size_t i = 0; i < d->candidate_count; ++i)
			if (d->candidates[i].exclusion_code
				!= std::string(MEMBRANE_ORCH_REASON_NO_MODEL))
				absent_everywhere = false;
		if (!want_json)
		{
			printf("\nNext:\n");
			if (absent_everywhere)
				printf("  membrane model list           (see what is "
					"installed)\n  membrane use %s   (install it, with "
					"your consent)\n", model.c_str());
			else
				printf("  membrane runtime list         (see which "
					"runtimes are available)\n");
		}
		return (absent_everywhere ? MEMBRANE_EXIT_MODEL_ERROR
			: MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	const membrane_orch_candidate_t	&s = d->candidates[d->selected];

	if (explain)
	{
		if (!want_json)
			printf("\n(--explain: nothing was changed.)\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (std::string(s.runtime_id) == MEMBRANE_RUNTIME_ID_NATIVE)
	{
		/* exactly `membrane use MODEL`: activate in the running service */
		std::vector<std::string>	use_args = {s.model_id};
		int							rc;
		{
			stdout_silencer_t	quiet(want_json);

			rc = membrane_use_cmd_dispatch(use_args, want_json);
		}
		if (!want_json)
			printf("\nNext: membrane chat\n");
		return (rc);
	}
	if (!want_json)
		printf("\nMEMBRANE does not launch or control %s. Point your client "
			"at %s and use the model id \"%s\".\n", s.runtime_id,
			g->endpoints[d->selected].empty() ? "its endpoint"
				: g->endpoints[d->selected].c_str(), s.model_id);
	return (MEMBRANE_EXIT_SUCCESS);
}
