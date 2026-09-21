#include <stdio.h>
#include <string.h>

#include "membrane_plan.h"
#include "gpu_policy.h"

const char	*membrane_plan_source_name(membrane_plan_source_t src)
{
	if (src == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		return ("explicit_user");
	if (src == MEMBRANE_PLAN_SOURCE_CATALOG_METADATA)
		return ("catalog_metadata");
	if (src == MEMBRANE_PLAN_SOURCE_MODEL_METADATA)
		return ("model_metadata");
	if (src == MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO)
		return ("hardware_auto");
	if (src == MEMBRANE_PLAN_SOURCE_PLANNER_DECISION)
		return ("planner_decision");
	if (src == MEMBRANE_PLAN_SOURCE_FALLBACK_DEFAULT)
		return ("fallback_default");
	return ("unknown");
}

static void	copy_str(char *dst, size_t dst_size, const char *src)
{
	size_t	len;

	if (dst_size == 0)
		return ;
	if (src == NULL)
		src = "";
	len = strlen(src);
	if (len >= dst_size)
		len = dst_size - 1;
	memcpy(dst, src, len);
	dst[len] = '\0';
}

static void	push_reason(membrane_plan_t *out, const char *code,
				const char *detail)
{
	membrane_plan_reason_t	*r;

	if (out->reason_count >= MEMBRANE_PLAN_MAX_REASONS)
		return ;
	r = &out->reasons[out->reason_count];
	copy_str(r->code, sizeof(r->code), code);
	copy_str(r->detail, sizeof(r->detail), detail);
	out->reason_count++;
}

static void	fill_identity(membrane_plan_t *out,
				const membrane_plan_identity_input_t *identity,
				const membrane_plan_variant_input_t *variant)
{
	memset(&out->identity, 0, sizeof(out->identity));
	copy_str(out->identity.model_name, sizeof(out->identity.model_name),
		identity->model_name);
	out->identity.installed = identity->installed;
	if (identity->installed)
		copy_str(out->identity.model_path, sizeof(out->identity.model_path),
			identity->model_path);
	out->identity.arch_known = identity->arch_known;
	if (identity->arch_known)
		copy_str(out->identity.arch_name, sizeof(out->identity.arch_name),
			identity->arch_name);
	if (identity->display_name != NULL && identity->display_name[0] != '\0')
		copy_str(out->identity.display_name,
			sizeof(out->identity.display_name), identity->display_name);
	else
		copy_str(out->identity.display_name,
			sizeof(out->identity.display_name), identity->model_name);
	copy_str(out->identity.parameter_count,
		sizeof(out->identity.parameter_count), identity->parameter_count);
	if (variant != NULL && variant->has_variant)
	{
		out->identity.variant_known = 1;
		copy_str(out->identity.variant, sizeof(out->identity.variant),
			variant->quant);
		out->identity.variant_source = variant->source;
		out->identity.variant_estimate_only = variant->estimate_only;
	}
}

static void	fill_workload(membrane_plan_t *out,
				const membrane_ctxrec_request_t *req,
				const membrane_plan_request_meta_t *meta)
{
	memset(&out->workload, 0, sizeof(out->workload));
	out->workload.requested_context = meta->requested_context;
	out->workload.context_source = meta->context_source;
	out->workload.precision_source = meta->precision_source;
	out->workload.gpu_layers_source = meta->gpu_layers_source;
	out->workload.kv_placement_source = meta->kv_placement_source;
	if (req != NULL)
	{
		out->workload.precision_request = req->precision_request;
		out->workload.gpu_layers_request = req->gpu_layers_request;
		out->workload.kv_placement_mode = req->kv_placement_mode;
	}
	else
	{
		out->workload.precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
		out->workload.gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
		out->workload.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	}
}

static void	fill_hardware(membrane_plan_t *out,
				const membrane_ctxrec_request_t *req,
				const membrane_ctxrec_result_t *rec,
				const membrane_plan_hardware_input_t *hardware)
{
	memset(&out->hardware, 0, sizeof(out->hardware));
	if (req == NULL)
		return ;
	out->hardware.host_total_bytes = req->host_total_bytes;
	out->hardware.host_available_bytes = req->host_available_bytes;
	out->hardware.host_available_known = req->host_available_known;
	if (rec != NULL)
		out->hardware.host_reserve_bytes = rec->host_reserve_bytes;
	out->hardware.device_total_bytes = req->device_total_bytes;
	out->hardware.device_free_bytes = req->device_free_bytes;
	out->hardware.device_known = (req->device_total_bytes != 0);
	if (out->hardware.device_known && hardware != NULL)
	{
		copy_str(out->hardware.backend, sizeof(out->hardware.backend),
			hardware->backend);
		copy_str(out->hardware.device_name,
			sizeof(out->hardware.device_name), hardware->device_name);
	}
}

/* Part 7: a dimension's DECIDED-value provenance is EXPLICIT_USER only
 * when this exact invocation supplied a hard constraint for it (the
 * joint planner/context recommender then never searched that dimension
 * at all -- see joint_planner.h's own "explicit dimension never produces
 * more than one candidate" contract); otherwise the concrete value came
 * from the planner's own search/ranking policy. */
static membrane_plan_source_t	decided_source(membrane_plan_source_t
				requested_source)
{
	if (requested_source == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		return (MEMBRANE_PLAN_SOURCE_EXPLICIT_USER);
	return (MEMBRANE_PLAN_SOURCE_PLANNER_DECISION);
}

static void	fill_decisions_feasible(membrane_plan_t *out,
				const membrane_ctxrec_result_t *rec,
				const membrane_plan_request_meta_t *meta)
{
	const membrane_joint_candidate_t	*c;

	c = &rec->selected_plan.candidates[rec->selected_plan.selected_index];
	out->decisions.has_decisions = 1;
	out->decisions.context = rec->recommended_context;
	out->decisions.context_source = decided_source(meta->context_source);
	out->decisions.gpu_layers = c->gpu_layers;
	out->decisions.gpu_layers_source = decided_source(meta->gpu_layers_source);
	out->decisions.kv_precision = c->kv_precision;
	out->decisions.kv_precision_source = decided_source(meta->precision_source);
	out->decisions.kv_placement = c->kv_placement;
	out->decisions.kv_placement_source
		= decided_source(meta->kv_placement_source);
}

static void	fill_feasibility_infeasible(membrane_plan_t *out,
				const membrane_ctxrec_result_t *rec)
{
	size_t	i;
	int		host_limited;
	int		vram_limited;

	out->feasibility.feasible = 0;
	out->feasibility.max_feasible_context_known = 0;
	host_limited = 0;
	vram_limited = 0;
	for (i = 0; i < rec->evaluated_count; ++i)
	{
		const membrane_ctxrec_evaluated_t	*ev = &rec->evaluated[i];

		if (ev->host_memory_checked && !ev->host_memory_fit)
			host_limited = 1;
		if (strcmp(ev->reason_code,
				MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT) == 0
			|| strcmp(ev->reason_code,
				MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT) == 0)
			vram_limited = 1;
	}
	if (host_limited)
		copy_str(out->feasibility.limiting_resource,
			sizeof(out->feasibility.limiting_resource),
			MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT);
	else if (vram_limited)
		copy_str(out->feasibility.limiting_resource,
			sizeof(out->feasibility.limiting_resource),
			MEMBRANE_PLAN_REASON_VRAM_LIMIT);
	else
		copy_str(out->feasibility.limiting_resource,
			sizeof(out->feasibility.limiting_resource),
			MEMBRANE_PLAN_REASON_NO_FEASIBLE_PLAN);
	push_reason(out, out->feasibility.limiting_resource, rec->reason);
}

static void	fill_feasibility_feasible(membrane_plan_t *out,
				const membrane_ctxrec_result_t *rec)
{
	out->feasibility.feasible = 1;
	out->feasibility.limiting_resource[0] = '\0';
	out->feasibility.max_feasible_context = rec->hardware_fit_context;
	out->feasibility.max_feasible_context_known = 1;
	out->feasibility.host_required_bytes = rec->host_required_bytes;
	out->feasibility.host_headroom_known = rec->host_memory_checked;
	if (rec->host_memory_checked)
	{
		uint64_t	spent;

		spent = rec->host_required_bytes + rec->host_reserve_bytes;
		if (rec->host_available_bytes > spent)
			out->feasibility.host_headroom_bytes
				= rec->host_available_bytes - spent;
		else
			out->feasibility.host_headroom_bytes = 0;
	}
}

static void	push_decision_reasons(membrane_plan_t *out,
				const membrane_ctxrec_result_t *rec,
				const membrane_ctxrec_request_t *req,
				const membrane_plan_variant_input_t *variant,
				const membrane_plan_request_meta_t *meta)
{
	const membrane_joint_candidate_t	*c;

	c = &rec->selected_plan.candidates[rec->selected_plan.selected_index];
	if (c->gpu_layers == 0 || !out->hardware.device_known)
		push_reason(out, MEMBRANE_PLAN_REASON_CPU_ONLY_PLAN, c->reason_code);
	else if (req != NULL && c->gpu_layers == req->n_layer_all)
		push_reason(out, MEMBRANE_PLAN_REASON_FULL_GPU_OFFLOAD_FEASIBLE,
			c->reason_code);
	if (c->kv_precision == MEMBRANE_JOINT_KV_Q8)
		push_reason(out, MEMBRANE_PLAN_REASON_KV_Q8_SELECTED_FOR_HEADROOM,
			c->reason_code);
	else if (c->kv_precision == MEMBRANE_JOINT_KV_Q5)
		push_reason(out, MEMBRANE_PLAN_REASON_KV_Q5_SELECTED_FOR_HEADROOM,
			c->reason_code);
	if (meta->context_source == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		push_reason(out, MEMBRANE_PLAN_REASON_USER_OVERRIDE_PRESERVED,
			"explicit --ctx honored unchanged");
	if (meta->precision_source == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		push_reason(out, MEMBRANE_PLAN_REASON_USER_OVERRIDE_PRESERVED,
			"explicit --kv honored unchanged");
	if (meta->gpu_layers_source == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		push_reason(out, MEMBRANE_PLAN_REASON_USER_OVERRIDE_PRESERVED,
			"explicit --gpu-layers honored unchanged");
	if (variant != NULL && variant->has_variant
		&& variant->source == MEMBRANE_PLAN_SOURCE_EXPLICIT_USER)
		push_reason(out, MEMBRANE_PLAN_REASON_USER_FORCED_VARIANT,
			"explicit --quant honored unchanged");
}

int	membrane_plan_assemble(const membrane_ctxrec_result_t *rec,
		const membrane_ctxrec_request_t *req,
		const membrane_plan_request_meta_t *meta,
		const membrane_plan_identity_input_t *identity,
		const membrane_plan_variant_input_t *variant,
		const membrane_plan_hardware_input_t *hardware,
		membrane_plan_t *out)
{
	if (out == NULL || identity == NULL || meta == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_PLAN_SCHEMA_VERSION;
	fill_identity(out, identity, variant);
	fill_workload(out, req, meta);
	fill_hardware(out, req, rec, hardware);
	if (variant != NULL && variant->has_variant && variant->estimate_only)
		push_reason(out, MEMBRANE_PLAN_REASON_MODEL_VARIANT_ESTIMATE_ONLY,
			"variant fit estimated from catalog size_bytes, not a real "
			"GGUF measurement");
	if (rec == NULL)
	{
		out->has_ctxrec_result = 0;
		out->decisions.has_decisions = 0;
		out->feasibility.feasible = 1;
		out->feasibility.limiting_resource[0] = '\0';
		push_reason(out, MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE,
			"full context/GPU/KV planning requires the installed model's "
			"real GGUF metadata -- install this model to get a complete "
			"plan");
		copy_str(out->explanation, sizeof(out->explanation),
			"Estimate only: full planning requires the model to be "
			"installed.");
		return (1);
	}
	out->has_ctxrec_result = 1;
	out->ctxrec_result = *rec;
	copy_str(out->explanation, sizeof(out->explanation), rec->explanation);
	if (rec->ok)
	{
		fill_decisions_feasible(out, rec, meta);
		fill_feasibility_feasible(out, rec);
		push_decision_reasons(out, rec, req, variant, meta);
	}
	else
	{
		out->decisions.has_decisions = 0;
		fill_feasibility_infeasible(out, rec);
	}
	return (out->feasibility.feasible);
}
