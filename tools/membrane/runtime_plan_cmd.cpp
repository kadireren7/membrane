#include "runtime_plan_cmd.h"

#include <cstdio>

#include "joint_planner.h"
#include "product_cli.h"
#include "runtime_cmd.h"

using json = nlohmann::json;

/* See runtime_plan_cmd.h's own top comment for the full contract. */

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

static json	str_or_null(const std::string &s)
{
	if (s.empty())
		return (nullptr);
	return (s);
}

static json	runtime_summary_json(const membrane_runtime_descriptor_t &d)
{
	json	j;

	j["id"] = std::string(d.id);
	j["type"] = membrane_runtime_type_name(d.type);
	j["execution_mode"]
		= membrane_runtime_execution_mode_name(d.execution_mode);
	j["status"] = membrane_runtime_availability_name(d.availability);
	j["health"] = membrane_runtime_health_name(d.health);
	j["version"] = d.version_known ? json(std::string(d.version))
		: json(nullptr);
	j["endpoint"] = d.endpoint_known ? json(std::string(d.endpoint))
		: json(nullptr);
	j["unavailable_reason"]
		= (d.availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
			&& d.unavailable_reason[0] != '\0')
		? json(std::string(d.unavailable_reason)) : json(nullptr);
	j["capability_provenance"]
		= membrane_capability_provenance_name(d.capability_provenance);
	return (j);
}

json	membrane_runtime_assessment_json(
			const membrane_runtime_descriptor_t &runtime,
			const membrane_runtime_plan_assessment_t &a,
			const json &model, const json &planner_plan)
{
	json	j;
	json	dims = json::array();
	json	reasons = json::array();
	size_t	i;

	j["schema_version"] = MEMBRANE_RUNTIME_ASSESSMENT_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "runtime_plan_assessment";
	j["ok"] = true;
	j["mutates_state"] = false;
	j["runtime"] = runtime_summary_json(runtime);
	j["model"] = model;
	j["planning_level"] = membrane_planning_level_name(a.planning_level);
	j["planner_plan"] = planner_plan;
	j["assessment"] = {
		{"actionability", membrane_runtime_actionability_name(a.actionability)},
		{"required_dimensions", a.required_count},
		{"controllable_dimensions", a.controllable_count}};
	for (i = 0; i < MEMBRANE_ASSESS_DIM_COUNT; ++i)
	{
		const membrane_assessment_dimension_result_t	&r = a.dimensions[i];
		json											d;

		d["name"] = membrane_assessment_dimension_name(r.dimension);
		d["planner_dimension"] = r.planner_dimension != 0;
		d["required"] = r.required != 0;
		d["value"] = r.value_known ? json(std::string(r.value)) : json(nullptr);
		d["value_source"] = membrane_assessment_value_source_name(
			r.value_source);
		d["plan_source"] = r.value_source == MEMBRANE_ASSESS_VALUE_PLANNER
			? json(membrane_plan_source_name(r.plan_source)) : json(nullptr);
		d["capability"] = membrane_capability_state_name(r.capability);
		d["applicability"]
			= membrane_dimension_applicability_name(r.applicability);
		d["reason"] = std::string(r.reason_code);
		dims.push_back(d);
	}
	j["dimensions"] = dims;
	for (i = 0; i < a.reason_count; ++i)
		reasons.push_back({{"code", a.reasons[i].code},
			{"detail", a.reasons[i].detail}});
	j["reasons"] = reasons;
	return (j);
}

/* ---------------------------------------------------------------- */
/* Human output                                                     */
/* ---------------------------------------------------------------- */

static const char	*dimension_label(const std::string &name)
{
	if (name == "quant_variant")
		return ("Quant variant");
	if (name == "context")
		return ("Context");
	if (name == "gpu_layers")
		return ("GPU layers");
	if (name == "kv_precision")
		return ("KV precision");
	if (name == "kv_placement")
		return ("KV placement");
	if (name == "device_selection")
		return ("Device selection");
	return ("Concurrency");
}

static const char	*actionability_label(const std::string &a)
{
	if (a == "fully_actionable")
		return ("Fully actionable");
	if (a == "partially_actionable")
		return ("Partially actionable");
	if (a == "advisory_only")
		return ("Advisory only");
	return ("Unsupported (cannot be assessed)");
}

static void	print_group(const json &doc, const char *applicability,
				const char *title)
{
	bool	printed = false;

	for (const auto &d : doc["dimensions"])
	{
		if (d["applicability"] != applicability)
			continue ;
		if (!printed)
			printf("\n%s\n", title);
		printed = true;
		printf("  %s%s\n", dimension_label(d["name"].get<std::string>()),
			d["required"].get<bool>() ? "" : "  (not in this plan)");
	}
}

void	membrane_runtime_assessment_print_human(const json &doc)
{
	const json	&rt = doc["runtime"];
	const json	&m = doc["model"];
	bool		any_value = false;

	printf("Runtime\n");
	printf("  %s (%s, %s, %s", rt["id"].get<std::string>().c_str(),
		rt["type"].get<std::string>().c_str(),
		rt["execution_mode"].get<std::string>().c_str(),
		rt["status"].get<std::string>().c_str());
	if (rt["version"].is_string())
		printf(", version %s", rt["version"].get<std::string>().c_str());
	printf(")\n");
	if (rt["unavailable_reason"].is_string())
		printf("  Reason: %s\n",
			rt["unavailable_reason"].get<std::string>().c_str());

	printf("\nModel\n");
	printf("  %s", m["runtime_model_id"].get<std::string>().c_str());
	if (m["identity_namespace"] == "runtime_inventory")
		printf("  (%s runtime model -- not a MEMBRANE registry model)",
			rt["id"].get<std::string>().c_str());
	printf("\n");
	if (m.contains("architecture") && m["architecture"].is_string())
		printf("  Architecture: %s\n",
			m["architecture"].get<std::string>().c_str());
	if (m.contains("parameter_size") && m["parameter_size"].is_string())
		printf("  Parameter size: %s\n",
			m["parameter_size"].get<std::string>().c_str());
	if (m.contains("quantization") && m["quantization"].is_string())
		printf("  Quantization: %s\n",
			m["quantization"].get<std::string>().c_str());
	if (m.contains("max_context_length") && m["max_context_length"].is_number())
		printf("  Context length (model maximum): %llu\n",
			(unsigned long long)m["max_context_length"].get<uint64_t>());

	printf("\nPlanning level\n  %s\n",
		doc["planning_level"].get<std::string>().c_str());

	/* Planner v2 values are MEMBRANE's recommendation; explicit flags on
	 * the capability-only path are the user's own request -- labelled as
	 * such, never passed off as a recommendation. */
	printf("\n%s\n", doc["planner_plan"].is_null()
		? "Requested settings" : "Recommendation (Planner v2)");
	for (const auto &d : doc["dimensions"])
	{
		if (!d["required"].get<bool>() || !d["value"].is_string())
			continue ;
		any_value = true;
		printf("  %s: %s (%s)\n", dimension_label(d["name"].get<std::string>()),
			d["value"].get<std::string>().c_str(),
			d["plan_source"].is_string()
				? d["plan_source"].get<std::string>().c_str()
				: "requested");
	}
	if (!any_value && doc["planner_plan"].is_null())
		printf("  None. No Planner v2 plan exists for this runtime model, "
			"and no --ctx/--kv/--gpu-layers/--quant was given.\n");
	else if (!any_value)
		printf("  None. Planner v2 made no variant/context/GPU/KV "
			"decision for this model.\n");

	printf("\nRuntime applicability\n  %s",
		actionability_label(doc["assessment"]["actionability"]));
	if (doc["assessment"]["required_dimensions"].get<size_t>() > 0)
		printf(" (%zu of %zu required settings controllable)",
			doc["assessment"]["controllable_dimensions"].get<size_t>(),
			doc["assessment"]["required_dimensions"].get<size_t>());
	printf("\n");
	print_group(doc, "controllable", "Runtime can control");
	print_group(doc, "partially_controllable",
		"Runtime can partially control");
	print_group(doc, "observable_only", "Runtime can only report");
	print_group(doc, "unsupported", "Runtime cannot control");
	print_group(doc, "unknown", "Unknown");

	if (!doc["reasons"].empty())
	{
		printf("\nWhy\n");
		for (const auto &r : doc["reasons"])
			printf("  - [%s] %s\n", r["code"].get<std::string>().c_str(),
				r["detail"].get<std::string>().c_str());
	}
	printf("\nNote\n  Recommendations only. No settings were changed.\n");
}

static int	emit(const json &doc, bool want_json)
{
	if (want_json)
		printf("%s\n", doc.dump().c_str());
	else
		membrane_runtime_assessment_print_human(doc);
	return (MEMBRANE_EXIT_SUCCESS);
}

/* ---------------------------------------------------------------- */
/* Native                                                           */
/* ---------------------------------------------------------------- */

int	membrane_runtime_plan_native(const membrane_plan_t &plan,
		const json &planner_plan_json, bool want_json)
{
	const membrane_runtime_adapter_t	*a
			= membrane_runtime_registry_find(MEMBRANE_RUNTIME_ID_NATIVE);
	membrane_runtime_descriptor_t		d;
	membrane_runtime_recommend_input_t	in;
	membrane_runtime_plan_assessment_t	assessment;
	json								model;

	a->describe(&d);
	in = membrane_runtime_recommend_input_t();
	in.runtime = &d;
	in.plan = &plan;
	membrane_runtime_recommend_plan(&in, &assessment);
	model["runtime_id"] = MEMBRANE_RUNTIME_ID_NATIVE;
	model["runtime_model_id"] = std::string(plan.identity.model_name);
	model["identity_namespace"] = plan.identity.installed
		? "membrane_registry" : "membrane_catalog";
	model["display_name"] = std::string(plan.identity.display_name);
	model["architecture"] = plan.identity.arch_known
		? json(std::string(plan.identity.arch_name)) : json(nullptr);
	model["parameter_size"]
		= str_or_null(std::string(plan.identity.parameter_count));
	model["quantization"] = plan.identity.variant_known
		? json(std::string(plan.identity.variant)) : json(nullptr);
	return (emit(membrane_runtime_assessment_json(d, assessment, model,
		planner_plan_json), want_json));
}

/* ---------------------------------------------------------------- */
/* External runtime (capability-only)                               */
/* ---------------------------------------------------------------- */

static json	external_model_json(const std::string &runtime_id,
				const std::string &runtime_model_id,
				const membrane_external_model_detail_t *d)
{
	json	m;

	m["runtime_id"] = runtime_id;
	m["runtime_model_id"] = runtime_model_id;
	m["identity_namespace"] = "runtime_inventory";
	if (d == NULL)
		return (m);
	m["architecture"] = str_or_null(d->architecture);
	m["family"] = str_or_null(d->model.family);
	m["parameter_size"] = str_or_null(d->model.parameter_size);
	m["parameter_count"] = d->parameter_count_known
		? json(d->parameter_count) : json(nullptr);
	m["quantization"] = str_or_null(d->model.quantization);
	m["max_context_length"] = d->model.context_length_known
		? json(d->model.context_length) : json(nullptr);
	m["remote"] = d->model.remote;
	m["metadata_provenance"]
		= membrane_capability_provenance_name(d->model.provenance);
	return (m);
}

int	membrane_runtime_plan_external(const membrane_runtime_adapter_t &a,
		const std::string &runtime_model_id,
		const membrane_runtime_plan_request_t &req, bool want_json)
{
	membrane_runtime_descriptor_t		d;
	membrane_external_model_detail_t	detail;
	membrane_runtime_error_t			err;
	membrane_runtime_recommend_input_t	in;
	membrane_runtime_plan_assessment_t	assessment;
	bool								have_detail = false;

	a.describe(&d);
	in = membrane_runtime_recommend_input_t();
	in.runtime = &d;
	in.runtime_model_id = runtime_model_id.c_str();
	if (d.availability == MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
	{
		/* The adapter's existing read-only metadata call (H2) -- the only
		 * model-level runtime I/O H3 performs. */
		membrane_external_model_detail_init(&detail);
		if (a.inspect_model == NULL
			|| !a.inspect_model(runtime_model_id, &detail, &err))
		{
			if (a.inspect_model == NULL)
			{
				err.code = "CLI_ERROR";
				err.message = "runtime '" + std::string(a.id) + "' exposes "
					"no model metadata";
			}
			print_err(want_json, err.code, err.message);
			return (membrane_runtime_error_exit_code(err));
		}
		have_detail = true;
		in.model_known = 1;
		in.model_quantization = detail.model.quantization.c_str();
		in.model_max_context_known = detail.model.context_length_known;
		in.model_max_context = detail.model.context_length;
	}
	in.requested_context_known = req.context_known;
	in.requested_context = req.context;
	in.requested_gpu_layers_known = req.gpu_layers_known;
	in.requested_gpu_layers = req.gpu_layers;
	in.requested_kv_precision_known = req.kv_precision_known;
	in.requested_kv_precision = req.kv_precision;
	in.requested_quant = req.quant.c_str();
	membrane_runtime_recommend_plan(&in, &assessment);
	emit(membrane_runtime_assessment_json(d, assessment,
		external_model_json(a.id, runtime_model_id,
			have_detail ? &detail : NULL), nullptr), want_json);
	/* The assessment is still printed for an unreachable runtime (it says
	 * why nothing could be assessed), but the exit code reports it. */
	if (assessment.planning_level == MEMBRANE_PLANNING_LEVEL_UNAVAILABLE)
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	return (MEMBRANE_EXIT_SUCCESS);
}
