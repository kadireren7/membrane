#include "advise_cmd.h"

#include <cstdio>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "memory_intelligence.h"
#include "observation.h"
#include "observe_cmd.h"
#include "observe_ollama.h"
#include "product_cli.h"
#include "runtime_capabilities.h"

using json = nlohmann::json;

/* See advise_cmd.h's own top comment for the full read-only contract. */

/* ------------------------------------------------------------------ */
/* Rendering -- runtime-agnostic, reads only membrane_memory_           */
/* intelligence_t and the snapshot it was assembled from                */
/* ------------------------------------------------------------------ */

static json	obs_summary_json(const membrane_observation_snapshot_t &s)
{
	json	j;

	j["status"] = membrane_obs_status_name(s.status);
	j["ram_total_bytes"] = s.ram_total_bytes.known
		? json(s.ram_total_bytes.value) : json(nullptr);
	j["ram_headroom_bytes"] = s.ram_headroom_bytes.known
		? json(s.ram_headroom_bytes.value) : json(nullptr);
	j["vram_total_bytes"] = s.vram_total_bytes.known
		? json(s.vram_total_bytes.value) : json(nullptr);
	j["vram_headroom_bytes"] = s.vram_headroom_bytes.known
		? json(s.vram_headroom_bytes.value) : json(nullptr);
	j["context_planned"] = s.context_planned.known
		? json(s.context_planned.value) : json(nullptr);
	j["context_model_max"] = s.context_model_max.known
		? json(s.context_model_max.value) : json(nullptr);
	j["kv_estimated_bytes"] = s.kv_estimated_bytes.known
		? json(s.kv_estimated_bytes.value) : json(nullptr);
	j["resident_model_count"] = s.resident_model_count.known
		? json(s.resident_model_count.value) : json(nullptr);
	return (j);
}

static json	finding_json(const membrane_intel_finding_t &f)
{
	json	j;
	json	ev = json::array();

	j["code"] = f.code;
	j["severity"] = membrane_intel_severity_name(f.severity);
	j["summary"] = f.summary;
	j["provenance"] = membrane_obs_provenance_name(f.provenance);
	j["dimension"] = membrane_intel_dimension_name(f.dimension);
	for (size_t i = 0; i < f.evidence_count; ++i)
		ev.push_back(f.evidence[i]);
	j["evidence"] = ev;
	return (j);
}

static json	recommendation_json(const membrane_intel_recommendation_t &r)
{
	json	j;
	json	reasons = json::array();

	j["code"] = r.code;
	j["action"] = r.action;
	for (size_t i = 0; i < r.reason_code_count; ++i)
		reasons.push_back(r.reason_codes[i]);
	j["reason_codes"] = reasons;
	j["affected_runtime"] = r.affected_runtime;
	j["runtime_capability"] = membrane_capability_state_name(
			r.runtime_capability);
	j["applicability"] = membrane_intel_applicability_name(r.applicability);
	j["mutates_state"] = r.mutates_state != 0;
	return (j);
}

static json	membrane_advise_result_json(
				const membrane_observation_snapshot_t &snap,
				const membrane_memory_intelligence_t &r)
{
	json	j;
	json	findings = json::array();
	json	recs = json::array();
	json	reasons = json::array();

	j["schema_version"] = r.schema_version;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "advise";
	j["timestamp"] = {{"utc", snap.timestamp_utc},
		{"unix_ms", snap.timestamp_unix_ms}};
	j["runtime"] = r.runtime_id;
	j["model"] = {{"known", r.model_known != 0},
		{"id", r.model_known ? json(r.model_id) : json(nullptr)}};
	j["status"] = membrane_intel_status_name(r.status);
	j["observation_summary"] = obs_summary_json(snap);
	for (size_t i = 0; i < r.finding_count; ++i)
		findings.push_back(finding_json(r.findings[i]));
	j["findings"] = findings;
	for (size_t i = 0; i < r.recommendation_count; ++i)
		recs.push_back(recommendation_json(r.recommendations[i]));
	j["recommendations"] = recs;
	for (size_t i = 0; i < r.reason_count; ++i)
		reasons.push_back(r.reasons[i]);
	j["reasons"] = reasons;
	j["mutates_state"] = r.mutates_state != 0;
	return (j);
}

static void	print_human(const membrane_memory_intelligence_t &r)
{
	printf("Memory intelligence\n");
	printf("  Runtime: %s\n", r.runtime_id);
	if (r.model_known)
		printf("  Model:   %s\n", r.model_id);
	printf("  Status:  %s\n", membrane_intel_status_name(r.status));
	if (r.status == MEMBRANE_INTEL_STATUS_INSUFFICIENT_DATA)
	{
		printf("\n(not enough observation data was available to assess "
			"memory pressure for this runtime)\n");
		return ;
	}
	printf("\nFindings\n");
	if (r.finding_count == 0)
		printf("  (none)\n");
	for (size_t i = 0; i < r.finding_count; ++i)
	{
		const membrane_intel_finding_t	&f = r.findings[i];

		printf("  [%s] %s\n", membrane_intel_severity_name(f.severity),
			f.summary);
		for (size_t j = 0; j < f.evidence_count; ++j)
			printf("      %s\n", f.evidence[j]);
		printf("      Source: %s\n", membrane_obs_provenance_name(
				f.provenance));
	}
	printf("\nRecommendations\n");
	if (r.recommendation_count == 0)
		printf("  (none)\n");
	for (size_t i = 0; i < r.recommendation_count; ++i)
	{
		const membrane_intel_recommendation_t	&rec = r.recommendations[i];

		printf("  %s\n", rec.action);
		printf("      Runtime support: %s\n",
			membrane_intel_applicability_name(rec.applicability));
		if (rec.reason_code_count > 0)
			printf("      Reason: %s\n", rec.reason_codes[0]);
	}
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
		fprintf(stderr, "membrane advise: %s\n", message.c_str());
}

static int	dispatch_for_snapshot(const membrane_observation_snapshot_t &snap,
				const membrane_runtime_capabilities_t &caps, bool want_json)
{
	membrane_memory_intelligence_t	result;

	membrane_memory_intelligence_assemble(&snap, &caps, &result);
	if (want_json)
		printf("%s\n", membrane_advise_result_json(snap, result).dump()
			.c_str());
	else
		print_human(result);
	return (snap.status == MEMBRANE_OBS_STATUS_UNAVAILABLE
		? MEMBRANE_EXIT_RUNTIME_ERROR : MEMBRANE_EXIT_SUCCESS);
}

int	membrane_advise_cmd_dispatch(const std::vector<std::string> &args,
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
			+ "' -- usage: membrane advise [--runtime membrane-native] "
			"[--json]");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (runtime_id == MEMBRANE_RUNTIME_ID_NATIVE)
	{
		membrane_observe_inputs_t				in;
		std::unique_ptr<membrane_observation_snapshot_t>	s(
				new membrane_observation_snapshot_t());

		membrane_observe_collect_inputs(runtime_id, &in);
		membrane_observe_build_snapshot(runtime_id, in, s.get());
		return (dispatch_for_snapshot(*s, in.runtime.capabilities,
				want_json));
	}
	if (runtime_id == MEMBRANE_RUNTIME_ID_OLLAMA)
	{
		membrane_observe_ollama_inputs_t		in;
		std::unique_ptr<membrane_observation_snapshot_t>	s(
				new membrane_observation_snapshot_t());

		membrane_observe_ollama_collect_inputs(&in);
		membrane_observe_ollama_build_snapshot(in, s.get());
		return (dispatch_for_snapshot(*s, in.runtime.capabilities,
				want_json));
	}
	membrane_runtime_descriptor_t	d;
	bool	known = membrane_runtime_describe(runtime_id.c_str(), &d)
		|| runtime_id == MEMBRANE_RUNTIME_ID_VLLM;

	print_err(want_json, "CLI_ERROR", known
		? "advice for runtime '" + runtime_id + "' is not implemented yet "
			"-- only membrane-native and ollama are supported"
		: "unknown runtime '" + runtime_id + "' -- see `membrane runtime "
			"list`");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
