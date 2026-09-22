#include "runtime_cmd.h"

#include <cstdio>

#include "runtime_capabilities.h"
#include "product_cli.h"

using json = nlohmann::json;

/* See runtime_cmd.h's own top comment for the full contract. */

static void	print_err(bool want_json, const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", "CLI_ERROR"}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane runtime: %s\n", message.c_str());
}

static bool	is_reserved_unimplemented(const std::string &runtime_id)
{
	return (runtime_id == MEMBRANE_RUNTIME_ID_OLLAMA
		|| runtime_id == MEMBRANE_RUNTIME_ID_VLLM);
}

/* The one place "unknown id" vs. "reserved but no adapter implemented"
 * is decided -- reused by both the --json and human inspect paths so
 * their error text can never drift apart. */
static bool	describe_or_reason(const std::string &runtime_id,
				membrane_runtime_descriptor_t *out, std::string *err_message)
{
	if (membrane_runtime_describe(runtime_id.c_str(), out))
		return (true);
	if (is_reserved_unimplemented(runtime_id))
		*err_message = "runtime '" + runtime_id + "' is a reserved "
			"identifier for a future external-runtime adapter; no "
			"adapter is implemented in this build";
	else
		*err_message = "unknown runtime '" + runtime_id + "'";
	return (false);
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
	if (d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE)
		j["unavailable_reason"] = std::string(d.unavailable_reason);
	else
		j["unavailable_reason"] = nullptr;
	if (d.version_known)
		j["version"] = std::string(d.version);
	else
		j["version"] = nullptr;
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
	membrane_runtime_descriptor_t	list[8];
	size_t							n;
	size_t							i;
	json							j;

	n = membrane_runtime_discover(list, 8);
	j["schema_version"] = MEMBRANE_RUNTIME_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["runtimes"] = json::array();
	i = 0;
	while (i < n)
	{
		j["runtimes"].push_back(descriptor_json(list[i]));
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
	membrane_runtime_descriptor_t	list[8];
	size_t							n;
	size_t							i;

	n = membrane_runtime_discover(list, 8);
	printf("%-18s %-10s %-12s %s\n", "ID", "TYPE", "STATUS", "VERSION");
	i = 0;
	while (i < n)
	{
		printf("%-18s %-10s %-12s %s\n", list[i].id,
			membrane_runtime_type_name(list[i].type),
			membrane_runtime_availability_name(list[i].availability),
			list[i].version_known ? list[i].version : "-");
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
	if (d.availability == MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE)
		printf("  Reason: %s\n", d.unavailable_reason);
	printf("\n");
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
		print_err(want_json, "usage: membrane runtime list|inspect ID");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (args[0] == "list")
		return (dispatch_list(args, want_json));
	if (args[0] == "inspect")
		return (dispatch_inspect(args, want_json));
	print_err(want_json, "unknown runtime subcommand '" + args[0] + "'");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
