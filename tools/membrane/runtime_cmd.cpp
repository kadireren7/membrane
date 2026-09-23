#include "runtime_cmd.h"

#include <cstdio>

#include "runtime_adapter.h"
#include "runtime_capabilities.h"
#include "product_cli.h"

using json = nlohmann::json;

/* See runtime_cmd.h's own top comment for the full contract. */

static void	print_err_code(bool want_json, const std::string &code,
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
		fprintf(stderr, "membrane runtime: %s\n", message.c_str());
}

static void	print_err(bool want_json, const std::string &message)
{
	print_err_code(want_json, "CLI_ERROR", message);
}

static bool	is_reserved_unimplemented(const std::string &runtime_id)
{
	return (runtime_id == MEMBRANE_RUNTIME_ID_VLLM);
}

/* The one place "unknown id" vs. "reserved but no adapter implemented"
 * is decided -- reused by inspect/models/model-inspect, --json and human
 * alike, so their error text can never drift apart. */
static const membrane_runtime_adapter_t	*find_or_reason(
				const std::string &runtime_id, std::string *err_message)
{
	const membrane_runtime_adapter_t	*a;

	a = membrane_runtime_registry_find(runtime_id);
	if (a != NULL)
		return (a);
	if (is_reserved_unimplemented(runtime_id))
		*err_message = "runtime '" + runtime_id + "' is a reserved "
			"identifier for a future external-runtime adapter; no "
			"adapter is implemented in this build";
	else
		*err_message = "unknown runtime '" + runtime_id + "'";
	return (NULL);
}

static bool	describe_or_reason(const std::string &runtime_id,
				membrane_runtime_descriptor_t *out, std::string *err_message)
{
	const membrane_runtime_adapter_t	*a;

	a = find_or_reason(runtime_id, err_message);
	if (a == NULL)
		return (false);
	a->describe(out);
	return (true);
}

static json	capabilities_json(const membrane_runtime_capabilities_t &c)
{
	json	j;

	j["model_enumeration"]
		= membrane_capability_state_name(c.model_enumeration);
	j["model_load_unload"]
		= membrane_capability_state_name(c.model_load_unload);
	j["model_switch"] = membrane_capability_state_name(c.model_switch);
	j["model_metadata"] = membrane_capability_state_name(c.model_metadata);
	j["context_control"] = membrane_capability_state_name(c.context_control);
	j["gpu_layer_control"]
		= membrane_capability_state_name(c.gpu_layer_control);
	j["quant_variant_control"]
		= membrane_capability_state_name(c.quant_variant_control);
	j["kv_precision_control"]
		= membrane_capability_state_name(c.kv_precision_control);
	j["kv_placement_control"]
		= membrane_capability_state_name(c.kv_placement_control);
	j["concurrency_control"]
		= membrane_capability_state_name(c.concurrency_control);
	j["device_selection"] = membrane_capability_state_name(c.device_selection);
	j["memory_headroom_telemetry"]
		= membrane_capability_state_name(c.memory_headroom_telemetry);
	j["chat_completions"] = membrane_capability_state_name(c.chat_completions);
	j["streaming"] = membrane_capability_state_name(c.streaming);
	j["cancellation"] = membrane_capability_state_name(c.cancellation);
	j["current_model"] = membrane_capability_state_name(c.current_model);
	j["active_context"] = membrane_capability_state_name(c.active_context);
	j["ram_usage"] = membrane_capability_state_name(c.ram_usage);
	j["vram_usage"] = membrane_capability_state_name(c.vram_usage);
	j["kv_cache_usage"] = membrane_capability_state_name(c.kv_cache_usage);
	j["loaded_model_memory"]
		= membrane_capability_state_name(c.loaded_model_memory);
	j["live_kv_migration"]
		= membrane_capability_state_name(c.live_kv_migration);
	j["dynamic_reconfiguration"]
		= membrane_capability_state_name(c.dynamic_reconfiguration);
	return (j);
}

static json	descriptor_json(const membrane_runtime_descriptor_t &d)
{
	json	j;

	j["id"] = std::string(d.id);
	j["display_name"] = std::string(d.display_name);
	j["type"] = membrane_runtime_type_name(d.type);
	j["execution_mode"]
		= membrane_runtime_execution_mode_name(d.execution_mode);
	j["status"] = membrane_runtime_availability_name(d.availability);
	/* H2: also carried for UNKNOWN availability (a malformed probe
	 * response), so a non-available runtime always says why. */
	if (d.availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
		&& d.unavailable_reason[0] != '\0')
		j["unavailable_reason"] = std::string(d.unavailable_reason);
	else
		j["unavailable_reason"] = nullptr;
	j["health"] = membrane_runtime_health_name(d.health);
	if (d.version_known)
		j["version"] = std::string(d.version);
	else
		j["version"] = nullptr;
	j["version_provenance"]
		= membrane_capability_provenance_name(d.version_provenance);
	if (d.endpoint_known)
		j["endpoint"] = std::string(d.endpoint);
	else
		j["endpoint"] = nullptr;
	j["capability_provenance"]
		= membrane_capability_provenance_name(d.capability_provenance);
	j["capabilities"] = capabilities_json(d.capabilities);
	return (j);
}

json	membrane_runtime_list_json(void)
{
	membrane_runtime_descriptor_t	d;
	size_t							i;
	json							j;

	j["schema_version"] = MEMBRANE_RUNTIME_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["runtimes"] = json::array();
	i = 0;
	while (i < membrane_runtime_registry_count())
	{
		membrane_runtime_registry_at(i)->describe(&d);
		j["runtimes"].push_back(descriptor_json(d));
		i++;
	}
	return (j);
}

bool	membrane_runtime_inspect_json(const std::string &runtime_id,
			json *out, std::string *err_message)
{
	membrane_runtime_descriptor_t	d;
	json							j;

	if (!describe_or_reason(runtime_id, &d, err_message))
		return (false);
	j["schema_version"] = MEMBRANE_RUNTIME_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["runtime"] = descriptor_json(d);
	*out = j;
	return (true);
}

static void	print_list_human(void)
{
	membrane_runtime_descriptor_t	d;
	size_t							i;

	printf("%-18s %-10s %-12s %s\n", "ID", "TYPE", "STATUS", "VERSION");
	i = 0;
	while (i < membrane_runtime_registry_count())
	{
		membrane_runtime_registry_at(i)->describe(&d);
		printf("%-18s %-10s %-12s %s\n", d.id,
			membrane_runtime_type_name(d.type),
			membrane_runtime_availability_name(d.availability),
			d.version_known ? d.version : "-");
		i++;
	}
}

static void	print_line(const char *label, membrane_capability_state_t s)
{
	printf("    %s: %s\n", label, membrane_capability_state_name(s));
}

/* Human-view-only rollup of the 4 raw telemetry fields into one line
 * (Part 8: "do not make the output enormous") -- --json always
 * itemizes all 4 separately (ram_usage/vram_usage/kv_cache_usage/
 * loaded_model_memory), this never replaces that, only summarizes it.
 * "Worst" means least confidently supported: SUPPORTED only if all 4
 * are, UNSUPPORTED only if all 4 are, UNKNOWN only if none of the 4
 * carry any signal at all, PARTIAL otherwise -- membrane-native's real
 * all-PARTIAL telemetry rolls up to "partial", matching the H1 task's
 * own worked example output. */
static membrane_capability_state_t	rollup_telemetry(
				const membrane_runtime_capabilities_t &c)
{
	membrane_capability_state_t	vals[4];
	int								all_supported;
	int								all_unsupported;
	int								any_known;
	int								i;

	vals[0] = c.ram_usage;
	vals[1] = c.vram_usage;
	vals[2] = c.kv_cache_usage;
	vals[3] = c.loaded_model_memory;
	all_supported = 1;
	all_unsupported = 1;
	any_known = 0;
	i = 0;
	while (i < 4)
	{
		if (vals[i] != MEMBRANE_CAPABILITY_SUPPORTED)
			all_supported = 0;
		if (vals[i] != MEMBRANE_CAPABILITY_UNSUPPORTED)
			all_unsupported = 0;
		if (vals[i] != MEMBRANE_CAPABILITY_UNKNOWN)
			any_known = 1;
		i++;
	}
	if (all_supported)
		return (MEMBRANE_CAPABILITY_SUPPORTED);
	if (all_unsupported)
		return (MEMBRANE_CAPABILITY_UNSUPPORTED);
	if (!any_known)
		return (MEMBRANE_CAPABILITY_UNKNOWN);
	return (MEMBRANE_CAPABILITY_PARTIAL);
}

static void	print_inspect_human(const membrane_runtime_descriptor_t &d)
{
	const membrane_runtime_capabilities_t	&c = d.capabilities;

	printf("Runtime\n");
	printf("  ID: %s\n", d.id);
	printf("  Type: %s\n", membrane_runtime_type_name(d.type));
	printf("  Execution mode: %s\n",
		membrane_runtime_execution_mode_name(d.execution_mode));
	printf("  Status: %s\n",
		membrane_runtime_availability_name(d.availability));
	/* H2: each line below only appears when it carries real information,
	 * so membrane-native's own output is exactly what H1 printed. */
	if (d.health != MEMBRANE_RUNTIME_HEALTH_NOT_PROBED)
		printf("  Health: %s\n", membrane_runtime_health_name(d.health));
	if (d.endpoint_known)
		printf("  Endpoint: %s\n", d.endpoint);
	if (d.version_known)
		printf("  Version: %s (%s)\n", d.version,
			membrane_capability_provenance_name(d.version_provenance));
	printf("\n");
	if (d.availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
		&& d.unavailable_reason[0] != '\0')
	{
		printf("Reason\n");
		printf("  %s\n", d.unavailable_reason);
		printf("\n");
	}
	printf("Capabilities\n");
	printf("  Model lifecycle:\n");
	print_line("Enumeration", c.model_enumeration);
	print_line("Switch", c.model_switch);
	print_line("Load/unload", c.model_load_unload);
	print_line("Metadata", c.model_metadata);
	printf("  Planning control:\n");
	print_line("Context", c.context_control);
	print_line("GPU layers", c.gpu_layer_control);
	print_line("Quant/variant", c.quant_variant_control);
	print_line("KV precision", c.kv_precision_control);
	print_line("KV placement", c.kv_placement_control);
	print_line("Concurrency", c.concurrency_control);
	print_line("Device selection", c.device_selection);
	print_line("Memory/headroom telemetry (planning)",
		c.memory_headroom_telemetry);
	printf("  Inference:\n");
	print_line("Chat completions", c.chat_completions);
	print_line("Streaming", c.streaming);
	print_line("Cancellation", c.cancellation);
	printf("  Observability:\n");
	print_line("Current model", c.current_model);
	print_line("Active context", c.active_context);
	print_line("Memory telemetry (RAM/VRAM/KV/model)", rollup_telemetry(c));
	printf("  Advanced:\n");
	print_line("Live KV migration", c.live_kv_migration);
	print_line("Dynamic reconfiguration", c.dynamic_reconfiguration);
	if (d.type == MEMBRANE_RUNTIME_TYPE_EXTERNAL)
	{
		printf("\nNote\n");
		printf("  Capabilities describe what this runtime's documented API "
			"exposes (%s).\n",
			membrane_capability_provenance_name(d.capability_provenance));
		printf("  This build only READS its version, model list and model "
			"metadata:\n  no inference, no model or configuration "
			"changes.\n");
	}
}

/* ---------------------------------------------------------------- */
/* Milestone H2: external-runtime model inventory / metadata         */
/* ---------------------------------------------------------------- */

static json	str_or_null(const std::string &s)
{
	if (s.empty())
		return (nullptr);
	return (s);
}

static json	external_model_json(const membrane_external_model_t &m)
{
	json	j;

	j["runtime_id"] = m.runtime_id;
	j["runtime_model_id"] = m.runtime_model_id;
	j["display_name"] = m.display_name;
	j["family"] = str_or_null(m.family);
	j["families"] = m.families;
	j["format"] = str_or_null(m.format);
	j["parameter_size"] = str_or_null(m.parameter_size);
	j["quantization"] = str_or_null(m.quantization);
	j["size_bytes"] = m.size_known ? json(m.size_bytes) : json(nullptr);
	j["digest"] = str_or_null(m.digest);
	j["modified_at"] = str_or_null(m.modified_at);
	j["context_length"] = m.context_length_known ? json(m.context_length)
		: json(nullptr);
	j["remote"] = m.remote;
	j["remote_host"] = str_or_null(m.remote_host);
	j["runtime_capabilities"] = m.runtime_capabilities;
	j["provenance"] = membrane_capability_provenance_name(m.provenance);
	return (j);
}

static bool	lookup_inventory_adapter(const std::string &runtime_id,
				const membrane_runtime_adapter_t **out,
				membrane_runtime_error_t *err)
{
	std::string	msg;

	*out = find_or_reason(runtime_id, &msg);
	if (*out == NULL)
	{
		err->code = "CLI_ERROR";
		err->message = msg;
		return (false);
	}
	if ((*out)->list_models == NULL || (*out)->inspect_model == NULL)
	{
		err->code = "CLI_ERROR";
		err->message = "runtime '" + runtime_id + "' has no external model "
			"inventory; its models are MEMBRANE's own registry -- use "
			"`membrane model list`";
		return (false);
	}
	return (true);
}

bool	membrane_runtime_models_json(const std::string &runtime_id,
			json *out, membrane_runtime_error_t *err)
{
	const membrane_runtime_adapter_t		*a;
	std::vector<membrane_external_model_t>	models;
	json									j;

	if (!lookup_inventory_adapter(runtime_id, &a, err))
		return (false);
	if (!a->list_models(&models, err))
		return (false);
	j["schema_version"] = MEMBRANE_RUNTIME_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["runtime_id"] = runtime_id;
	j["models"] = json::array();
	for (const auto &m : models)
		j["models"].push_back(external_model_json(m));
	*out = j;
	return (true);
}

bool	membrane_runtime_model_inspect_json(const std::string &runtime_id,
			const std::string &model, json *out,
			membrane_runtime_error_t *err)
{
	const membrane_runtime_adapter_t	*a;
	membrane_external_model_detail_t	d;
	json								m;
	json								j;

	if (!lookup_inventory_adapter(runtime_id, &a, err))
		return (false);
	membrane_external_model_detail_init(&d);
	if (!a->inspect_model(model, &d, err))
		return (false);
	m = external_model_json(d.model);
	m["architecture"] = str_or_null(d.architecture);
	m["parameter_count"] = d.parameter_count_known
		? json(d.parameter_count) : json(nullptr);
	m["parameters"] = str_or_null(d.parameters);
	m["parameters_truncated"] = d.parameters_truncated;
	m["has_template"] = d.has_template;
	m["has_system"] = d.has_system;
	m["has_license"] = d.has_license;
	j["schema_version"] = MEMBRANE_RUNTIME_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["runtime_id"] = runtime_id;
	j["model"] = m;
	*out = j;
	return (true);
}

/* Decimal units, matching how Ollama itself prints model sizes. */
static std::string	human_size(const json &v)
{
	char	buf[32];
	double	b;

	if (!v.is_number())
		return ("-");
	b = v.get<double>();
	if (b >= 1e9)
		snprintf(buf, sizeof(buf), "%.1f GB", b / 1e9);
	else if (b >= 1e6)
		snprintf(buf, sizeof(buf), "%.0f MB", b / 1e6);
	else if (b >= 1e3)
		snprintf(buf, sizeof(buf), "%.0f KB", b / 1e3);
	else
		snprintf(buf, sizeof(buf), "%.0f B", b);
	return (buf);
}

static std::string	str_or_dash(const json &v)
{
	if (v.is_string() && !v.get<std::string>().empty())
		return (v.get<std::string>());
	return ("-");
}

static void	print_models_human(const json &j)
{
	size_t	width = 24;
	size_t	remote = 0;

	if (j["models"].empty())
	{
		printf("Runtime '%s' reports no local models.\n",
			j["runtime_id"].get<std::string>().c_str());
		return ;
	}
	for (const auto &m : j["models"])
		if (m["display_name"].get<std::string>().size() + 2 > width)
			width = m["display_name"].get<std::string>().size() + 2;
	printf("%-*s %-10s %-12s %-8s %s\n", (int)width, "NAME", "SIZE",
		"FAMILY", "PARAMS", "QUANT");
	for (const auto &m : j["models"])
	{
		printf("%-*s %-10s %-12s %-8s %s\n", (int)width,
			m["display_name"].get<std::string>().c_str(),
			human_size(m["size_bytes"]).c_str(),
			str_or_dash(m["family"]).c_str(),
			str_or_dash(m["parameter_size"]).c_str(),
			str_or_dash(m["quantization"]).c_str());
		if (m["remote"].get<bool>())
			remote++;
	}
	if (remote > 0)
		printf("\n%zu entr%s the runtime reports as remote stubs "
			"(remote_host set); MEMBRANE never contacts remote hosts.\n",
			remote, remote == 1 ? "y is" : "ies are");
}

static void	print_model_line(const char *label, const std::string &v)
{
	if (!v.empty() && v != "-")
		printf("  %s: %s\n", label, v.c_str());
}

static void	print_model_inspect_human(const json &j)
{
	const json	&m = j["model"];
	std::string	caps;

	printf("Model\n");
	printf("  Runtime: %s\n", j["runtime_id"].get<std::string>().c_str());
	printf("  Name: %s\n", m["runtime_model_id"].get<std::string>().c_str());
	print_model_line("Architecture", str_or_dash(m["architecture"]));
	print_model_line("Family", str_or_dash(m["family"]));
	print_model_line("Format", str_or_dash(m["format"]));
	print_model_line("Parameter size", str_or_dash(m["parameter_size"]));
	if (m["parameter_count"].is_number())
		printf("  Parameter count: %llu\n",
			(unsigned long long)m["parameter_count"].get<uint64_t>());
	print_model_line("Quantization", str_or_dash(m["quantization"]));
	if (m["context_length"].is_number())
		printf("  Context length (model maximum): %llu\n",
			(unsigned long long)m["context_length"].get<uint64_t>());
	print_model_line("Modified", str_or_dash(m["modified_at"]));
	for (const auto &c : m["runtime_capabilities"])
		caps += (caps.empty() ? "" : ", ") + c.get<std::string>();
	print_model_line("Model capabilities", caps);
	if (m["remote"].get<bool>())
		print_model_line("Remote host (stub)", str_or_dash(m["remote_host"]));
	printf("  Template: %s   System prompt: %s   License: %s\n",
		m["has_template"].get<bool>() ? "present" : "none",
		m["has_system"].get<bool>() ? "present" : "none",
		m["has_license"].get<bool>() ? "present" : "none");
	if (m["parameters"].is_string())
	{
		printf("\nRuntime default parameters\n");
		std::string	p = m["parameters"].get<std::string>();
		size_t		start = 0;
		size_t		nl;

		while (start < p.size())
		{
			nl = p.find('\n', start);
			if (nl == std::string::npos)
				nl = p.size();
			if (nl > start)
				printf("  %s\n", p.substr(start, nl - start).c_str());
			start = nl + 1;
		}
		if (m["parameters_truncated"].get<bool>())
			printf("  ... (truncated)\n");
	}
}

static int	exit_code_for(const membrane_runtime_error_t &err)
{
	if (err.code == MEMBRANE_RUNTIME_ERR_MODEL_NOT_FOUND
		|| err.code == MEMBRANE_RUNTIME_ERR_CLOUD_REFUSED)
		return (MEMBRANE_EXIT_MODEL_ERROR);
	if (err.code == MEMBRANE_RUNTIME_ERR_UNAVAILABLE
		|| err.code == MEMBRANE_RUNTIME_ERR_HTTP
		|| err.code == MEMBRANE_RUNTIME_ERR_MALFORMED
		|| err.code == MEMBRANE_RUNTIME_ERR_TOO_LARGE)
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	return (MEMBRANE_EXIT_CLI_ERROR);
}

static int	dispatch_models(const std::vector<std::string> &args,
				bool want_json)
{
	membrane_runtime_error_t	err;
	json						j;

	if (args.size() != 2)
	{
		print_err(want_json, "usage: membrane runtime models ID");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!membrane_runtime_models_json(args[1], &j, &err))
	{
		print_err_code(want_json, err.code, err.message);
		return (exit_code_for(err));
	}
	if (want_json)
		printf("%s\n", j.dump().c_str());
	else
		print_models_human(j);
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	dispatch_model(const std::vector<std::string> &args,
				bool want_json)
{
	membrane_runtime_error_t	err;
	json						j;

	if (args.size() != 4 || args[1] != "inspect")
	{
		print_err(want_json, "usage: membrane runtime model inspect ID MODEL");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!membrane_runtime_model_inspect_json(args[2], args[3], &j, &err))
	{
		print_err_code(want_json, err.code, err.message);
		return (exit_code_for(err));
	}
	if (want_json)
		printf("%s\n", j.dump().c_str());
	else
		print_model_inspect_human(j);
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	dispatch_list(const std::vector<std::string> &args, bool want_json)
{
	if (args.size() != 1)
	{
		print_err(want_json, "'membrane runtime list' takes no arguments");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (want_json)
		printf("%s\n", membrane_runtime_list_json().dump().c_str());
	else
		print_list_human();
	return (MEMBRANE_EXIT_SUCCESS);
}

static int	dispatch_inspect(const std::vector<std::string> &args,
				bool want_json)
{
	membrane_runtime_descriptor_t	d;
	std::string						err;

	if (args.size() != 2)
	{
		print_err(want_json, "usage: membrane runtime inspect ID");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (want_json)
	{
		json	j;

		if (!membrane_runtime_inspect_json(args[1], &j, &err))
		{
			print_err(true, err);
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		printf("%s\n", j.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (!describe_or_reason(args[1], &d, &err))
	{
		print_err(false, err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	print_inspect_human(d);
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_runtime_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	if (args.empty())
	{
		print_err(want_json, "usage: membrane runtime list | inspect ID | "
			"models ID | model inspect ID MODEL");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (args[0] == "list")
		return (dispatch_list(args, want_json));
	if (args[0] == "inspect")
		return (dispatch_inspect(args, want_json));
	if (args[0] == "models")
		return (dispatch_models(args, want_json));
	if (args[0] == "model")
		return (dispatch_model(args, want_json));
	print_err(want_json, "unknown runtime subcommand '" + args[0] + "'");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
